# coplanner/opportunity_ranker.py
"""
Opportunity Ranker — 7-Factor Multi-Dimensional Ranking.

Takes cross-stock comparison output and applies portfolio-aware
ranking with timing, rarity, and correlation adjustments.

COMPOSITE = (
    conviction_score    × 0.35 +
    regime_boost        × 0.20 +
    rr_quality          × 0.20 +
    liquidity_edge      × 0.08 +
    opportunity_rarity  × 0.07 +
    timing_score        × 0.05 +
    correlation_adj     × 0.05
)

Rank Tiers:
  PRIME   (>= 0.75) → FULL_SIZE
  STRONG  (>= 0.60) → REDUCED_SIZE
  VIABLE  (>= 0.45) → MINIMAL_SIZE
  MARGINAL(>= 0.30) → SKIP
  AVOID   (<  0.30) → SKIP
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from coplanner.models import (
    ConvictionResult,
    RankedOpportunity,
    SetupData,
)
from coplanner.strategy_regime_matrix import StrategyRegimeMatrix

logger = logging.getLogger(__name__)

# Factor weights
RANKING_WEIGHTS = {
    "conviction": 0.35,
    "regime_boost": 0.20,
    "rr_quality": 0.20,
    "liquidity_edge": 0.08,
    "opportunity_rarity": 0.07,
    "timing": 0.05,
    "correlation_adj": 0.05,
}

# Portfolio constraints
MAX_SIMULTANEOUS = 3
MAX_PER_SECTOR = 2

# Indian FNO stock sector mapping
DEFAULT_SECTOR_MAP = {
    # Banking
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "SBIN": "BANKING",
    "KOTAKBANK": "BANKING", "AXISBANK": "BANKING", "INDUSINDBK": "BANKING",
    "BANKBARODA": "BANKING", "PNB": "BANKING", "FEDERALBNK": "BANKING",
    "IDFCFIRSTB": "BANKING", "BANDHANBNK": "BANKING", "AUBANK": "BANKING",
    # IT
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "LTIMINDTREE": "IT", "MPHASIS": "IT", "COFORGE": "IT",
    # Auto
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "M&M": "AUTO",
    "BAJAJ-AUTO": "AUTO", "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO",
    "ASHOKLEY": "AUTO", "TVSMOTOR": "AUTO",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    "COLPAL": "FMCG", "TATACONSUM": "FMCG",
    # Pharma
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "DIVISLAB": "PHARMA", "APOLLOHOSP": "PHARMA", "BIOCON": "PHARMA",
    # Metals
    "TATASTEEL": "METALS", "HINDALCO": "METALS", "JSWSTEEL": "METALS",
    "VEDL": "METALS", "COALINDIA": "METALS", "NMDC": "METALS",
    # Energy
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY",
    "IOC": "ENERGY", "NTPC": "ENERGY", "POWERGRID": "ENERGY",
    "ADANIGREEN": "ENERGY", "TATAPOWER": "ENERGY",
    # Finance (non-bank)
    "BAJFINANCE": "FINANCE", "BAJAJFINSV": "FINANCE",
    "SBILIFE": "FINANCE", "HDFCLIFE": "FINANCE",
    "ICICIPRULI": "FINANCE", "CHOLAFIN": "FINANCE",
    # Infra/Cement
    "ULTRACEMCO": "INFRA", "GRASIM": "INFRA", "ADANIENT": "INFRA",
    "LT": "INFRA", "ACC": "INFRA", "AMBUJACEM": "INFRA",
    # Telecom
    "BHARTIARTL": "TELECOM", "IDEA": "TELECOM",
}


class OpportunityRanker:
    """
    Multi-factor ranking engine with portfolio constraints.

    Usage:
        ranker = OpportunityRanker(regime_matrix)
        ranked = ranker.rank(evaluated_setups, open_positions)
    """

    def __init__(
        self,
        regime_matrix: StrategyRegimeMatrix,
        open_positions: Optional[List[Dict[str, Any]]] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ):
        self._matrix = regime_matrix
        self._open_positions = open_positions or []
        self._sector_map = sector_map or DEFAULT_SECTOR_MAP

    def rank(
        self,
        evaluated_setups: List[Dict[str, Any]],
        open_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RankedOpportunity]:
        """
        Rank evaluated setups (from CrossStockComparator) with full
        7-factor scoring and portfolio constraints.
        """
        if open_positions is not None:
            self._open_positions = open_positions

        ranked: List[RankedOpportunity] = []

        for setup_data in evaluated_setups:
            conv: ConvictionResult = setup_data["conviction"]
            symbol = setup_data["symbol"]
            strategy = setup_data["strategy"]
            regime_wr = setup_data.get("regime_wr", 0.5)
            expected_rr = setup_data.get("expected_rr", 0.0)
            rr_ok = setup_data.get("rr_acceptable", True)

            scores: Dict[str, float] = {}

            # 1. Conviction (0.35)
            scores["conviction"] = conv.conviction_score

            # 2. Regime boost (0.20) — relative position among strategies
            scores["regime_boost"] = self._regime_boost(strategy, setup_data)

            # 3. RR quality (0.20) — graduated, not pass/fail
            scores["rr_quality"] = self._rr_quality(expected_rr, rr_ok)

            # 4. Liquidity edge (0.08)
            scores["liquidity_edge"] = conv.components.get("liquidity_score", 0.5)

            # 5. Opportunity rarity (0.07)
            scores["opportunity_rarity"] = self._opportunity_rarity(strategy)

            # 6. Timing (0.05)
            scores["timing"] = self._timing_score()

            # 7. Correlation adjustment (0.05)
            scores["correlation_adj"] = self._correlation_adjustment(
                symbol, strategy
            )

            composite = sum(
                scores[k] * RANKING_WEIGHTS.get(k, 0.0) for k in scores
            )
            composite = round(min(1.0, max(0.0, composite)), 4)

            # Determine action and sizing
            action, sizing = self._action_and_sizing(composite, rr_ok)

            ranked.append(RankedOpportunity(
                rank=0,  # assigned after sort
                symbol=symbol,
                strategy_name=strategy,
                composite_score=composite,
                conviction=conv,
                rr_assessment=conv.rr_assessment,
                regime_stats=None,
                avoidance=setup_data.get("avoidance"),
                score_breakdown=scores,
                action=action,
                sizing=sizing,
                explanation=self._build_explanation(symbol, scores, action),
            ))

        # Sort and assign ranks
        ranked.sort(key=lambda r: r.composite_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1

        # Apply portfolio constraints
        self._apply_portfolio_constraints(ranked)

        return ranked

    # ─────────────────────────────────────────────
    # FACTOR 2: Regime Boost
    # ─────────────────────────────────────────────

    def _regime_boost(self, strategy: str, setup: Dict[str, Any]) -> float:
        """
        Where does this strategy rank among all strategies
        for the current regime? Position-based scoring.
        """
        # Extract current regime from the conviction result's strategy
        conv = setup.get("conviction")
        if not conv:
            return 0.35

        # Get regime from the setup's regime_wr context or matrix
        regime_wr = setup.get("regime_wr", 0.5)

        try:
            current_regime = self._matrix.get_current_regime()
            best = self._matrix.get_best_strategies_for_regime(current_regime)
        except Exception:
            best = []

        if not best:
            return 0.35  # unproven

        # Find position of this strategy
        normalized_strategy = strategy.strip().lower().replace(" ", "_")
        for i, entry in enumerate(best):
            if entry["strategy"] == normalized_strategy:
                # Position-based: #1 → 1.0, last → 0.2
                position_score = 1.0 - (i / max(len(best), 1)) * 0.8
                absolute_wr = entry["win_rate"]
                return round(position_score * 0.6 + absolute_wr * 0.4, 4)

        return 0.35  # not found = unproven

    # ─────────────────────────────────────────────
    # FACTOR 3: RR Quality
    # ─────────────────────────────────────────────

    @staticmethod
    def _rr_quality(expected_rr: float, rr_ok: bool) -> float:
        """Graduated quality score, not just pass/fail."""
        if not rr_ok and expected_rr < 1.0:
            return 0.10
        if expected_rr >= 3.0:
            return 1.0
        if expected_rr >= 2.5:
            return 0.90
        if expected_rr >= 2.0:
            return 0.80
        if expected_rr >= 1.5:
            return 0.65
        if expected_rr >= 1.2:
            return 0.45
        if expected_rr >= 1.0:
            return 0.30
        return 0.10

    # ─────────────────────────────────────────────
    # FACTOR 5: Opportunity Rarity
    # ─────────────────────────────────────────────

    @staticmethod
    def _opportunity_rarity(strategy: str) -> float:
        """
        How rare is this setup type? Rarer = higher value.
        Uses rough frequency estimates (can be calibrated from trade history).
        """
        # Default frequency estimates — can be overridden from trade history
        frequency_map = {
            "breakout": 0.35,
            "pullback": 0.40,
            "reversal": 0.15,
            "momentum": 0.45,
            "range": 0.50,
        }
        normalized = strategy.strip().lower().replace(" ", "_")
        freq = frequency_map.get(normalized, 0.30)
        return round(max(0.1, 1.0 - freq * 1.5), 4)

    # ─────────────────────────────────────────────
    # FACTOR 6: Timing
    # ─────────────────────────────────────────────

    @staticmethod
    def _timing_score() -> float:
        """
        Market hour quality (IST time slots).
        """
        now = datetime.now(timezone.utc)
        # Convert to IST (UTC+5:30)
        ist_hour = (now.hour + 5) % 24
        ist_minute = now.minute + 30
        if ist_minute >= 60:
            ist_hour = (ist_hour + 1) % 24
            ist_minute -= 60

        h = ist_hour + ist_minute / 60.0

        # IST trading hours scoring
        if 9.25 <= h < 10.0:
            score = 0.90   # opening
        elif 10.0 <= h < 11.5:
            score = 1.0    # peak
        elif 11.5 <= h < 13.0:
            score = 0.85   # midday
        elif 13.0 <= h < 14.5:
            score = 0.80   # post-lunch
        elif 14.5 <= h < 15.5:
            score = 0.85   # close approach
        else:
            score = 0.50   # outside market hours

        # Day of week adjustment
        weekday = now.weekday()
        if weekday in (0, 4):  # Monday, Friday
            score -= 0.10
        elif weekday == 3:  # Thursday
            score += 0.05

        return round(max(0.0, min(1.0, score)), 4)

    # ─────────────────────────────────────────────
    # FACTOR 7: Correlation Adjustment
    # ─────────────────────────────────────────────

    def _correlation_adjustment(self, symbol: str, strategy: str) -> float:
        """Penalize doubling up on same stock/sector/strategy."""
        if not self._open_positions:
            return 1.0  # empty portfolio → no penalty

        penalty = 0.0
        symbol_upper = symbol.upper()
        symbol_sector = self._sector_map.get(symbol_upper)

        for pos in self._open_positions:
            pos_symbol = pos.get("symbol", "").upper()

            # Same stock — heavy penalty
            if pos_symbol == symbol_upper:
                penalty += 0.30

            # Same sector — moderate penalty
            elif symbol_sector:
                pos_sector = pos.get("sector") or self._sector_map.get(pos_symbol)
                if pos_sector and pos_sector == symbol_sector:
                    penalty += 0.12

        # Same strategy concentration
        same_strategy = sum(
            1 for p in self._open_positions
            if p.get("strategy", "").lower() == strategy.lower()
        )
        if same_strategy >= 2:
            penalty += 0.15

        return round(max(0.0, 1.0 - penalty), 4)

    # ─────────────────────────────────────────────
    # ACTION & SIZING
    # ─────────────────────────────────────────────

    @staticmethod
    def _action_and_sizing(composite: float, rr_ok: bool) -> tuple[str, float]:
        if not rr_ok:
            return "SKIP", 0.0
        if composite >= 0.75:
            return "TAKE", 1.0
        if composite >= 0.60:
            return "TAKE", round(0.6 + (composite - 0.60) / 0.15 * 0.3, 2)
        if composite >= 0.45:
            return "CAUTIOUS", round(0.3 + (composite - 0.45) / 0.15 * 0.2, 2)
        return "SKIP", 0.0

    # ─────────────────────────────────────────────
    # PORTFOLIO CONSTRAINTS
    # ─────────────────────────────────────────────

    def _apply_portfolio_constraints(
        self, ranked: List[RankedOpportunity],
    ) -> None:
        """Downgrade opportunities that violate constraints."""
        active_new = 0
        sector_counts: Dict[str, int] = {}

        # Count existing positions by sector
        for pos in self._open_positions:
            pos_symbol = pos.get("symbol", "").upper()
            sector = pos.get("sector") or self._sector_map.get(pos_symbol, "OTHER")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        for r in ranked:
            if r.action in ("TAKE", "CAUTIOUS"):
                active_new += 1
                symbol_sector = self._sector_map.get(r.symbol.upper(), "OTHER")

                # Max simultaneous new trades
                if active_new > MAX_SIMULTANEOUS:
                    r.action = "SKIP"
                    r.sizing = 0.0
                    r.explanation += " [Exceeds max simultaneous new trades]"
                    continue

                # Max per sector
                current_sector_count = sector_counts.get(symbol_sector, 0)
                if current_sector_count >= MAX_PER_SECTOR:
                    r.action = "SKIP"
                    r.sizing = 0.0
                    r.explanation += f" [Sector {symbol_sector} at max capacity]"
                    continue

                sector_counts[symbol_sector] = current_sector_count + 1

    # ─────────────────────────────────────────────
    # EXPLANATION
    # ─────────────────────────────────────────────

    @staticmethod
    def _build_explanation(
        symbol: str,
        scores: Dict[str, float],
        action: str,
    ) -> str:
        parts = [f"{symbol}:"]
        conv = scores.get("conviction", 0)
        if conv >= 0.70:
            parts.append(f"strong conviction ({conv:.2f})")
        elif conv >= 0.55:
            parts.append(f"moderate conviction ({conv:.2f})")
        else:
            parts.append(f"low conviction ({conv:.2f})")

        rr = scores.get("rr_quality", 0)
        if rr >= 0.80:
            parts.append("excellent RR")
        elif rr < 0.45:
            parts.append("weak RR")

        timing = scores.get("timing", 0)
        if timing >= 0.90:
            parts.append("good timing")
        elif timing < 0.60:
            parts.append("suboptimal timing")

        return ", ".join(parts) + f" → {action}"
