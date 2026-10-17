# automation/smart_watchlist_seeder.py
"""
Smart Watchlist Seeder
======================
Automatically populates a "Smart Picks" strategy watchlist per user with the
top technically-qualified FNO stocks — no LLM, no manual curation required.

Scoring (100% rule-based, 4 dimensions):
  1. Volume surge     (30%) — recent 3-bar avg vs 20-bar avg
  2. Momentum         (25%) — price distance from SMA-20 (above = bullish)
  3. Trend clarity    (20%) — consecutive directional bars (last 5)
  4. Range compression(25%) — tight consolidation relative to 20-bar range

The "Smart Picks" watchlist uses watchlist_type="strategy" so it appears
immediately in the co-planner dashboard without needing strategy profiles.

Usage:
    seeder = SmartWatchlistSeeder(db)
    seeder.set_fetcher(market_data_svc.fetch_candles_for_scanner)
    result = seeder.seed(trader_id)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Type alias — matches setup_scanner / watchlist_analyzer CandleFetchFn
CandleFetchFn = Callable[[str, int], list[dict]]

# How many top stocks to keep in the smart watchlist
TOP_N = 25

# Minimum technical score to be included (0-1 scale)
MIN_SCORE = 0.32

# Min bars required to score a symbol
MIN_BARS = 20


class SmartWatchlistSeeder:
    """
    Scores the FNO universe with pure technical rules and upserts the top
    picks into each user's "Smart Picks" strategy watchlist.

    The seeder is idempotent — safe to run multiple times.
    """

    def __init__(self, db: Session) -> None:
        self._db = db
        self._fetcher: Optional[CandleFetchFn] = None

    def set_fetcher(self, fn: CandleFetchFn) -> None:
        self._fetcher = fn

    # ─────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────

    def seed(self, trader_id: str) -> dict:
        """
        Create/refresh the "Smart Picks" watchlist for a single trader.
        Returns summary dict: {watchlist_id, added, updated, total, scored_from}.
        """
        from core.models import Watchlist, WatchlistItem

        trader_uuid = UUID(trader_id)

        # Get or create "Smart Picks" watchlist
        smart_wl = (
            self._db.query(Watchlist)
            .filter(
                Watchlist.trader_id == trader_uuid,
                Watchlist.name == "Smart Picks",
                Watchlist.is_active == True,
            )
            .first()
        )
        if not smart_wl:
            smart_wl = Watchlist(
                trader_id=trader_uuid,
                name="Smart Picks",
                watchlist_type="strategy",   # shows in co-planner without strategy profile
                description="Auto-curated FNO picks scored by volume, momentum, and compression",
                auto_scan=True,
            )
            self._db.add(smart_wl)
            self._db.flush()
            logger.info("Created 'Smart Picks' watchlist for trader %s", trader_id)

        # Score FNO universe
        scored = self._score_universe()
        top = scored[:TOP_N]

        if not top:
            logger.info(
                "SmartWatchlistSeeder: no candle data available — watchlist unchanged"
            )
            return {
                "watchlist_id": str(smart_wl.id),
                "added": [],
                "updated": [],
                "total": 0,
                "scored_from": len(scored),
                "note": "No candle data — wire a candle fetcher",
            }

        top_symbols = {e["symbol"] for e in top}

        # Upsert top stocks
        added, updated = [], []
        now = datetime.now(timezone.utc)

        for entry in top:
            existing = (
                self._db.query(WatchlistItem)
                .filter(
                    WatchlistItem.watchlist_id == smart_wl.id,
                    WatchlistItem.symbol == entry["symbol"],
                )
                .first()
            )
            if existing:
                existing.match_score = entry["score"]
                existing.match_reason = entry["reason"]
                existing.status = "matched"
                existing.updated_at = now
                existing.last_analysis = entry.get("analysis", {})
                updated.append(entry["symbol"])
            else:
                item = WatchlistItem(
                    watchlist_id=smart_wl.id,
                    symbol=entry["symbol"],
                    added_by="system",
                    status="matched",
                    match_score=entry["score"],
                    match_reason=entry["reason"],
                    last_analysis=entry.get("analysis", {}),
                )
                self._db.add(item)
                added.append(entry["symbol"])

        # Archive stocks that dropped out of top N
        stale = (
            self._db.query(WatchlistItem)
            .filter(
                WatchlistItem.watchlist_id == smart_wl.id,
                WatchlistItem.added_by == "system",
                ~WatchlistItem.symbol.in_(top_symbols),
                WatchlistItem.status == "matched",
            )
            .all()
        )
        for stale_item in stale:
            stale_item.status = "filtered_out"

        smart_wl.last_scanned_at = now
        self._db.commit()

        logger.info(
            "SmartWatchlistSeeder: added=%d updated=%d archived=%d (scored %d/%d FNO stocks)",
            len(added), len(updated), len(stale),
            len(scored), len(top) + len(stale),
        )
        return {
            "watchlist_id": str(smart_wl.id),
            "added": added,
            "updated": updated,
            "archived": len(stale),
            "total": len(top),
            "scored_from": len(scored),
        }

    # ─────────────────────────────────────────────
    #  Scoring engine
    # ─────────────────────────────────────────────

    def _score_universe(self) -> list[dict]:
        """Score entire FNO universe. Returns list sorted by score desc."""
        from discovery.orchestrator import FNO_UNIVERSE

        results = []
        for symbol in FNO_UNIVERSE:
            try:
                candles = self._fetch(symbol)
                if not candles or len(candles) < MIN_BARS:
                    continue
                score, reason, analysis = self._technical_score(symbol, candles)
                if score >= MIN_SCORE:
                    results.append({
                        "symbol": symbol,
                        "score": round(score, 3),
                        "reason": reason,
                        "analysis": analysis,
                    })
            except Exception as e:
                logger.debug("Scoring failed for %s: %s", symbol, e)

        results.sort(key=lambda x: x["score"], reverse=True)
        logger.info(
            "SmartWatchlistSeeder: scored %d/%d FNO symbols above threshold",
            len(results), len(FNO_UNIVERSE),
        )
        return results

    def rank_by_factor(self, factor: str, top_n: int = 30) -> list[str]:
        """
        Rank the FNO universe by a SINGLE technical factor and return the
        top-N symbols. Reuses the same candle scoring as Smart Picks.

        factor: 'volume' (relative volume) | 'momentum' (price vs SMA-20).

        Used by the dynamic universe bundles (High Volume, Momentum Breakout):
        the result is a real, data-driven snapshot of the strongest names on
        that one dimension, which then feed the user's strategy matching.
        Returns [] if no candle fetcher is wired (caller falls back to a static list).
        """
        from discovery.orchestrator import FNO_UNIVERSE

        key = {"volume": "vol_score", "momentum": "mom_score"}.get(factor, "vol_score")
        scored: list[tuple[str, float]] = []
        for symbol in FNO_UNIVERSE:
            try:
                candles = self._fetch(symbol)
                if not candles or len(candles) < MIN_BARS:
                    continue
                _s, _r, analysis = self._technical_score(symbol, candles)
                scored.append((symbol, float(analysis.get(key, 0.0))))
            except Exception as e:
                logger.debug("rank_by_factor(%s) failed for %s: %s", factor, symbol, e)

        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [s for s, sc in scored[:top_n] if sc > 0]
        logger.info(
            "rank_by_factor(%s): ranked %d symbols, returning top %d",
            factor, len(scored), len(ranked),
        )
        return ranked

    def _fetch(self, symbol: str, n_bars: int = 50) -> list[dict]:
        if not self._fetcher:
            return []
        try:
            return self._fetcher(symbol, n_bars)
        except Exception as e:
            logger.debug("Candle fetch failed for %s: %s", symbol, e)
            return []

    def _technical_score(
        self, symbol: str, candles: list[dict]
    ) -> tuple[float, str, dict]:
        """
        Score a symbol 0–1 on 4 technical dimensions.
        Returns (score, reason_string, raw_analysis_dict).
        """
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        vols = [c.get("volume", 0) for c in candles]

        current = closes[-1]
        sma20 = sum(closes[-20:]) / 20

        # ── 1. Volume surge (30%) ─────────────────────────────────
        vol_score = 0.5
        vol_ratio = 0.0
        if any(vols):
            avg_vol20 = sum(vols[-20:]) / 20
            avg_vol3 = sum(vols[-3:]) / 3
            if avg_vol20 > 0:
                vol_ratio = avg_vol3 / avg_vol20
                # 2× volume → 1.0; below average → 0.0
                vol_score = min(1.0, max(0.0, (vol_ratio - 0.5) / 1.5))

        # ── 2. Momentum (25%) — price vs SMA-20 ──────────────────
        mom_score = 0.5
        pct_from_sma = 0.0
        if sma20 > 0:
            pct_from_sma = (current - sma20) / sma20
            if pct_from_sma > 0.04:
                mom_score = 1.0
            elif pct_from_sma > 0.02:
                mom_score = 0.80
            elif pct_from_sma > 0.005:
                mom_score = 0.65
            elif pct_from_sma > -0.005:
                mom_score = 0.50
            elif pct_from_sma > -0.02:
                mom_score = 0.30
            else:
                mom_score = 0.10

        # ── 3. Trend clarity (20%) — directional consistency ─────
        trend_score = 0.5
        if len(closes) >= 6:
            up_bars = sum(
                1 for i in range(-5, 0) if closes[i] > closes[i - 1]
            )
            trend_score = up_bars / 5.0  # 5/5 = 1.0, 0/5 = 0.0

        # ── 4. Range compression (25%) — coiled-spring setup ─────
        # Small recent bars vs large 20-bar range = setup ready to break
        compression_score = 0.5
        if len(candles) >= 20:
            bar_ranges = [h - l for h, l in zip(highs[-8:], lows[-8:])]
            avg_recent_range = sum(bar_ranges) / len(bar_ranges)
            total_range_20 = max(highs[-20:]) - min(lows[-20:])
            if total_range_20 > 0:
                ratio = avg_recent_range / total_range_20
                # ratio < 0.15 = tightly compressed = 1.0
                # ratio > 0.5  = wide candles = 0.0
                compression_score = max(0.0, min(1.0, 1.0 - (ratio / 0.5)))

        # ── Weighted total ────────────────────────────────────────
        score = (
            vol_score * 0.30
            + mom_score * 0.25
            + trend_score * 0.20
            + compression_score * 0.25
        )

        # ── Human-readable reason ─────────────────────────────────
        parts = []
        if vol_ratio >= 1.5:
            parts.append(f"{vol_ratio:.1f}x volume")
        if pct_from_sma >= 0.02:
            parts.append(f"+{pct_from_sma*100:.1f}% above SMA20")
        elif pct_from_sma <= -0.02:
            parts.append(f"{pct_from_sma*100:.1f}% below SMA20")
        if compression_score >= 0.65:
            parts.append("range compressed")
        if trend_score >= 0.80:
            parts.append("strong trend")
        elif trend_score <= 0.20:
            parts.append("downtrend")
        if not parts:
            parts.append("neutral setup")

        reason = ", ".join(parts)

        analysis = {
            "vol_score": round(vol_score, 2),
            "vol_ratio": round(vol_ratio, 2),
            "mom_score": round(mom_score, 2),
            "pct_from_sma20": round(pct_from_sma * 100, 2),
            "trend_score": round(trend_score, 2),
            "compression_score": round(compression_score, 2),
            "composite_score": round(score, 3),
            "current_price": current,
            "sma20": round(sma20, 2),
            "analysis_method": "smart_seed_technicals",
        }

        return score, reason, analysis
