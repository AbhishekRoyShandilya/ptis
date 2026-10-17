# avt/stock_ranker.py
"""
Stock Selection Ranker — The Hidden Edge
==========================================

When 5 stocks all show the same strategy signal, which one should you trade?

The human picks based on familiarity or whatever they see first. Random.

The AVT ranks by multi-dimensional quality:
  1. Conviction score (0.30) — how well does this match the ideal pattern?
  2. Stock quality (0.25)    — RVOL, spread, range remaining, trend alignment
  3. Variant performance (0.20) — historical win rate for this variant+regime
  4. Setup freshness (0.10)  — just triggered vs stale signal
  5. Portfolio fit (0.15)    — diversification vs concentration

This is an edge the human can never develop alone because they can't
watch 200 stocks simultaneously and compute composite scores in real time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from avt.models import (
    RankedCandidate,
    SetupCandidate,
    StrategyVersion,
    VirtualTrade,
)

logger = logging.getLogger(__name__)

# Default weights
DEFAULT_WEIGHTS = {
    "conviction": 0.30,
    "stock_quality": 0.25,
    "variant_performance": 0.20,
    "freshness": 0.10,
    "portfolio_fit": 0.15,
}

# Stock quality sub-weights
QUALITY_WEIGHTS = {
    "rvol": 0.35,
    "spread": 0.15,
    "range_remaining": 0.20,
    "trend_alignment": 0.30,
}


class StockSelectionRanker:
    """
    Multi-dimensional stock ranking for setup selection.

    Integrates with:
      - ske/feature_computer.py for real-time feature computation
      - ske/regime_fingerprint.py for regime context
      - ske/variant_engine.py for variant historical performance
    """

    def __init__(
        self,
        feature_computer: Any = None,
        regime_engine: Any = None,
        market_data: Any = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self._features = feature_computer
        self._regime = regime_engine
        self._market = market_data

        self._weights = weights or dict(DEFAULT_WEIGHTS)

        # Cache variant performance lookups
        self._variant_perf_cache: Dict[str, Dict[str, float]] = {}

        # Per-(strategy_code, symbol) historical edge from the benchmark replay
        # (StrategySymbolPerformance). Loaded via load_symbol_performance();
        # empty until a baseline is built → ranking falls back to intrinsic quality.
        self._symbol_perf_cache: Dict[tuple, Dict[str, float]] = {}

    def set_market_data(self, market_data: Any) -> None:
        """Inject market data interface (set by engine at runtime)."""
        self._market = market_data

    # ── Historical edge from the benchmark replay (StrategySymbolPerformance) ──

    def load_symbol_performance(self, db: Any, trader_id: Any = None) -> int:
        """
        Load the per-(strategy, stock) ranking baseline (the symbol-overall
        regime='*' rows) into a cache. Source of truth = the BENCHMARK replay
        (clean AVT executions), never user trades. Graceful: empty table → empty
        cache → ranking is unaffected (intrinsic quality only). Returns row count.
        """
        try:
            from core.models import StrategySymbolPerformance as SSP
            q = db.query(SSP).filter(SSP.regime == "*")
            if trader_id is not None:
                q = q.filter(SSP.trader_id == str(trader_id))
            cache: Dict[tuple, Dict[str, float]] = {}
            for r in q.all():
                cache[(str(r.strategy_code), str(r.symbol))] = {
                    "expectancy": float(r.expectancy or 0.0),
                    "confidence": float(r.confidence_score or 0.0),
                    "trades": int(r.trade_count or 0),
                }
            self._symbol_perf_cache = cache
            logger.info("StockRanker: loaded %d (strategy,symbol) edge rows", len(cache))
            return len(cache)
        except Exception as e:
            logger.debug("load_symbol_performance failed (ranking falls back): %s", e)
            return 0

    def _symbol_hist_score(self, strategy_id: Any, symbol: str) -> Optional[float]:
        """
        0-1 historical-edge score for (strategy, symbol), or None if no data.
        Neutral 0.5; shifted by CONFIDENCE × clipped expectancy so thin samples
        stay near neutral and only deep, profitable cells push the rank up.
        """
        rec = self._symbol_perf_cache.get((str(strategy_id), str(symbol)))
        if not rec:
            return None
        exp = max(-1.0, min(1.0, rec["expectancy"] / 2.0))   # +2R→+1, −2R→−1
        return round(min(1.0, max(0.0, 0.5 + rec["confidence"] * 0.5 * exp)), 4)

    # ── Main Ranking ───────────────────────────────────────────

    async def rank_candidates(
        self,
        candidates: List[SetupCandidate],
        current_regime: Any = None,
        current_positions: Optional[List[VirtualTrade]] = None,
        strategy_versions: Optional[Dict[str, StrategyVersion]] = None,
    ) -> List[RankedCandidate]:
        """
        Rank all setup candidates by composite desirability.

        Returns sorted list with rank positions assigned.
        """
        if not candidates:
            return []

        positions = current_positions or []
        ranked = []

        for candidate in candidates:
            scores: Dict[str, float] = {}

            # 1. CONVICTION SCORE (already computed by SKE)
            scores["conviction"] = self._extract_conviction(candidate)

            # 2. STOCK QUALITY SCORE
            scores["stock_quality"] = await self._compute_stock_quality(
                candidate.symbol
            )

            # 3. VARIANT HISTORICAL PERFORMANCE in current regime
            scores["variant_performance"] = self._get_variant_regime_performance(
                candidate, current_regime, strategy_versions
            )

            # 4. SETUP FRESHNESS
            scores["freshness"] = self._compute_freshness(candidate)

            # 5. PORTFOLIO FIT
            scores["portfolio_fit"] = self._compute_portfolio_fit(
                candidate, positions
            )

            # Weighted composite
            composite = sum(
                scores[dim] * self._weights.get(dim, 0.0)
                for dim in scores
            )

            # Generate explanation
            explanation = self._generate_explanation(scores, candidate)

            ranked.append(RankedCandidate(
                candidate=candidate,
                composite_score=round(composite, 4),
                score_breakdown=scores,
                ranking_explanation=explanation,
            ))

        # Sort by composite score descending
        ranked.sort(key=lambda r: r.composite_score, reverse=True)

        # Assign rank positions
        for i, r in enumerate(ranked):
            r.rank = i + 1

        return ranked

    # ── Dimension 1: Conviction ────────────────────────────────

    @staticmethod
    def _extract_conviction(candidate: SetupCandidate) -> float:
        """Extract conviction score from signal."""
        signal = candidate.signal
        if signal is None:
            return 0.0
        return getattr(signal, "conviction_score", 0.0)

    # ── Dimension 2: Stock Quality ─────────────────────────────

    async def _compute_stock_quality(self, symbol: str) -> float:
        """
        Real-time stock quality assessment.

        Components:
          a) RVOL — relative volume today (higher = more institutional interest)
          b) Spread — tighter = better execution
          c) Range remaining — less consumed = more room for trade
          d) Trend alignment — stock trending in trade direction on HTF
        """
        if not self._market:
            return 0.5  # neutral default

        scores: Dict[str, float] = {}

        try:
            # a) Relative Volume
            rvol = await self._compute_rvol(symbol)
            scores["rvol"] = min(1.0, rvol / 2.5)  # 2.5x RVOL = perfect 1.0

            # b) Spread
            spread_pct = await self._market.get_spread_pct(symbol)
            scores["spread"] = max(0.0, 1.0 - (spread_pct / 0.15))  # <0.15% = good

            # c) Range remaining
            range_used = await self._compute_range_used(symbol)
            scores["range_remaining"] = max(0.0, 1.0 - range_used)

            # d) Trend alignment
            scores["trend_alignment"] = await self._compute_trend_alignment(symbol)

        except Exception as e:
            logger.debug("Stock quality error for %s: %s", symbol, e)
            return 0.5

        # Weighted composite
        quality = sum(
            scores.get(k, 0.5) * QUALITY_WEIGHTS.get(k, 0.25)
            for k in QUALITY_WEIGHTS
        )
        return round(min(1.0, max(0.0, quality)), 4)

    async def _compute_rvol(self, symbol: str) -> float:
        """Compute relative volume: today's volume / 20-day average volume."""
        try:
            candles = await self._market.get_candles(symbol, count=25)
            if not candles or len(candles) < 21:
                return 1.0

            # Last candle = today's volume (so far)
            today_vol = candles[-1].get("volume", 0)

            # Average of prior 20 bars
            prior_vols = [c.get("volume", 0) for c in candles[-21:-1]]
            avg_vol = sum(prior_vols) / len(prior_vols) if prior_vols else 1
            if avg_vol <= 0:
                return 1.0

            return today_vol / avg_vol
        except Exception:
            return 1.0

    async def _compute_range_used(self, symbol: str) -> float:
        """
        What % of avg daily range has already been consumed?

        If stock already moved 80% of its average daily range,
        less room left for your trade.
        """
        try:
            candles = await self._market.get_candles(symbol, count=25)
            if not candles or len(candles) < 21:
                return 0.5

            # Average daily range from prior 20 bars
            daily_ranges = [c.get("high", 0) - c.get("low", 0) for c in candles[-21:-1]]
            avg_range = sum(daily_ranges) / len(daily_ranges) if daily_ranges else 1.0
            if avg_range <= 0:
                return 0.5

            # Today's range so far
            today = candles[-1]
            today_range = today.get("high", 0) - today.get("low", 0)

            return min(1.0, today_range / avg_range)
        except Exception:
            return 0.5

    async def _compute_trend_alignment(self, symbol: str) -> float:
        """
        Is the stock trending in the trade direction on higher timeframe?

        Uses 20-bar EMA slope as a simple alignment measure.
        Returns 0-1 (0 = strong downtrend, 1 = strong uptrend).
        """
        try:
            candles = await self._market.get_candles(symbol, count=30)
            if not candles or len(candles) < 20:
                return 0.5

            closes = [c.get("close", 0) for c in candles]

            # Compute 20-period EMA
            alpha = 2 / (20 + 1)
            ema = closes[0]
            for c in closes[1:]:
                ema = alpha * c + (1 - alpha) * ema

            # Current price relative to EMA
            current = closes[-1]
            if ema <= 0:
                return 0.5

            ratio = current / ema
            # Normalize: 0.97 → 0.0, 1.0 → 0.5, 1.03 → 1.0
            alignment = min(1.0, max(0.0, (ratio - 0.97) / 0.06))

            return round(alignment, 4)
        except Exception:
            return 0.5

    # ── Dimension 3: Variant Performance ───────────────────────

    def _get_variant_regime_performance(
        self,
        candidate: SetupCandidate,
        current_regime: Any,
        strategy_versions: Optional[Dict[str, StrategyVersion]] = None,
    ) -> float:
        """
        Historical performance of the matched variant in the current regime.

        A stock matching Variant A (72% WR, 0.8R expectancy) in a trending
        regime ranks higher than one matching Variant B (55% WR, 0.2R)
        in a ranging regime.
        """
        signal = candidate.signal
        if signal is None:
            return 0.5

        variant_id = getattr(signal, "matched_variant", None)
        variant_similarity = getattr(signal, "variant_similarity", 0.0)
        regime_match = getattr(signal, "regime_match_score", 0.5)
        variant_regime_match = getattr(signal, "variant_regime_match", 0.5)

        # Variant similarity × regime match (or neutral-ish if no variant matched)
        if not variant_id:
            variant_quality = 0.4
        else:
            variant_quality = variant_similarity * 0.5 + variant_regime_match * 0.5
            # Anti-variants (from SKE) should score very low
            cached = self._variant_perf_cache.get(f"{candidate.strategy_id}:{variant_id}")
            if cached:
                variant_quality = (variant_quality + cached.get("score", 0.5)) / 2

        # Blend in the BENCHMARK-replay historical edge for THIS (strategy, stock)
        # when available (else unchanged). Equal weight: intrinsic variant signal
        # vs proven per-stock edge; confidence already shrinks thin cells to neutral.
        hist = self._symbol_hist_score(candidate.strategy_id, candidate.symbol)
        if hist is not None:
            variant_quality = 0.5 * variant_quality + 0.5 * hist

        return round(min(1.0, max(0.0, variant_quality)), 4)

    def update_variant_performance(
        self,
        strategy_id: str,
        variant_id: str,
        win_rate: float,
        expectancy_r: float,
    ) -> None:
        """
        Update cached variant performance data.
        Called by self_improver after analyzing AVT's own trades.
        """
        key = f"{strategy_id}:{variant_id}"

        # Normalize to 0-1 score
        wr_score = min(1.0, win_rate)
        exp_score = min(1.0, max(0.0, (expectancy_r + 1.0) / 3.0))  # -1R→0, 0R→0.33, 2R→1.0

        self._variant_perf_cache[key] = {
            "win_rate": win_rate,
            "expectancy_r": expectancy_r,
            "score": wr_score * 0.4 + exp_score * 0.6,
        }

    # ── Dimension 4: Setup Freshness ───────────────────────────

    @staticmethod
    def _compute_freshness(candidate: SetupCandidate) -> float:
        """
        How recently did the signal trigger?

        Just triggered (this candle): 1.0
        Triggered 2-3 candles ago:   0.7
        Triggered 5+ candles ago:    0.3
        """
        signal = candidate.signal
        if signal is None:
            return 0.5

        # Use signal timestamp if available
        signal_time = getattr(signal, "timestamp", None)
        if signal_time and candidate.timestamp:
            age_seconds = (candidate.timestamp - signal_time).total_seconds()
            age_minutes = age_seconds / 60

            if age_minutes < 5:
                return 1.0
            elif age_minutes < 15:
                return 0.8
            elif age_minutes < 30:
                return 0.6
            elif age_minutes < 60:
                return 0.4
            else:
                return 0.2

        # If no timestamp, assume moderately fresh
        return 0.7

    # ── Dimension 5: Portfolio Fit ─────────────────────────────

    @staticmethod
    def _compute_portfolio_fit(
        candidate: SetupCandidate,
        current_positions: List[VirtualTrade],
    ) -> float:
        """
        Does this trade diversify or concentrate the portfolio?

        Penalizes:
          a) Sector concentration — already holding same sector
          b) Strategy concentration — all positions from same strategy
          c) Direction concentration — all longs in uptrend

        Returns 0-1 (1 = great diversification, 0 = highly concentrated).
        """
        if not current_positions:
            return 1.0  # empty portfolio → everything fits

        n = len(current_positions)
        penalties = 0.0

        # a) Sector concentration
        sector = candidate.sector
        same_sector = sum(1 for p in current_positions if getattr(p, "sector", "unknown") == sector)
        if same_sector >= 2:
            penalties += 0.3
        elif same_sector >= 1:
            penalties += 0.1

        # b) Strategy concentration
        strategy_id = candidate.strategy_id
        same_strategy = sum(1 for p in current_positions if p.strategy_id == strategy_id)
        strategy_ratio = same_strategy / n if n > 0 else 0
        if strategy_ratio > 0.6:
            penalties += 0.3
        elif strategy_ratio > 0.4:
            penalties += 0.15

        # c) Direction concentration
        # Simple: count longs vs shorts
        longs = sum(1 for p in current_positions if p.direction == "long")
        direction_ratio = longs / n if n > 0 else 0.5
        if direction_ratio > 0.8 or direction_ratio < 0.2:
            penalties += 0.15

        # d) Symbol already in portfolio? (shouldn't happen — engine filters this)
        if any(p.symbol == candidate.symbol for p in current_positions):
            penalties += 0.5

        return round(max(0.0, 1.0 - penalties), 4)

    # ── Explanation ─────────────────────────────────────────────

    @staticmethod
    def _generate_explanation(
        scores: Dict[str, float],
        candidate: SetupCandidate,
    ) -> str:
        """Generate a human-readable explanation for the ranking."""
        parts = []

        conv = scores.get("conviction", 0)
        if conv >= 0.7:
            parts.append(f"Strong conviction ({conv:.2f})")
        elif conv >= 0.5:
            parts.append(f"Moderate conviction ({conv:.2f})")
        else:
            parts.append(f"Low conviction ({conv:.2f})")

        sq = scores.get("stock_quality", 0)
        if sq >= 0.7:
            parts.append("high stock quality")
        elif sq < 0.4:
            parts.append("weak stock quality")

        vp = scores.get("variant_performance", 0)
        if vp >= 0.7:
            parts.append("strong variant track record")
        elif vp < 0.4:
            parts.append("weak variant history")

        fr = scores.get("freshness", 0)
        if fr >= 0.8:
            parts.append("fresh signal")
        elif fr < 0.4:
            parts.append("stale signal")

        pf = scores.get("portfolio_fit", 0)
        if pf >= 0.8:
            parts.append("good portfolio fit")
        elif pf < 0.4:
            parts.append("concentrated exposure")

        return f"{candidate.symbol}: " + ", ".join(parts)
