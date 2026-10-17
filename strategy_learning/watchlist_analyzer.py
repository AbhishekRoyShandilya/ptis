# strategy_learning/watchlist_analyzer.py
"""
Watchlist Analyzer — Scans master watchlist and distributes stocks to strategy watchlists.

Flow:
1. User adds symbols to their master watchlist
2. System fetches candle data for each symbol
3. For each active StrategyProfile, analyzes whether the stock's current
   chart setup matches the learned strategy
4. Auto-adds matching stocks to the strategy's dedicated watchlist
5. Updates match_score and match_reason on each item

Uses the same CandleFetchFn interface as SetupScanner.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from agents.llm_client import LLMClient, LLMMessage
from config.settings import settings
from core.events import Event, EventType, get_event_bus
from core.models import StrategyProfile, Watchlist, WatchlistItem

logger = logging.getLogger(__name__)

# Type alias matching setup_scanner's CandleFetchFn
CandleFetchFn = Callable[[str, int], list[dict]]

MATCH_PROMPT = """You are analyzing whether a stock's current chart setup matches a trader's learned strategy.

## Strategy: {strategy_name}
{narrative}

Entry Criteria: {entry_criteria}
Regime Preferences: {regime_preferences}
Time Preferences: {time_preferences}

## Current Chart Data for {symbol}
Recent price action (last 20 bars):
{price_summary}

Current Price: {current_price}
Trend: {trend}
Volatility: {volatility}
Volume: {volume_profile}

## Instructions
Rate how well this stock's CURRENT setup matches the strategy on 0-100.
Consider: Is it forming the pattern this strategy looks for? Is it in the right regime?

Return JSON:
{{
  "match_score": 0-100,
  "match_reason": "1-2 sentence explanation",
  "setup_forming": true/false,
  "regime_suitable": true/false,
  "key_levels": {{"support": null, "resistance": null}},
  "recommended_action": "watch|prepare|skip"
}}

Return ONLY JSON:"""


class WatchlistAnalyzer:
    """Analyzes watchlist stocks and distributes to strategy-specific watchlists."""

    def __init__(self, db: Session, llm: Optional[LLMClient] = None):
        self.db = db
        self._llm = llm
        self._fetcher: Optional[CandleFetchFn] = None

    def set_fetcher(self, fn: CandleFetchFn) -> None:
        """Inject candle data fetcher at runtime."""
        self._fetcher = fn

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(
                provider=settings.LLM_PROVIDER,
                model=settings.LLM_MODEL,
            )
        return self._llm

    # ──────────────────────────────────────────────
    #  Core: scan master watchlist → distribute
    # ──────────────────────────────────────────────

    def scan_and_distribute(self, master_watchlist_id: UUID) -> dict:
        """
        Scan all symbols in a master watchlist against all active strategy profiles.
        Auto-add matches to strategy-specific watchlists.

        Returns summary: {strategy_name: {added: [...], updated: [...], removed: [...]}}
        """
        master = self.db.query(Watchlist).get(master_watchlist_id)
        if not master or master.watchlist_type != "master":
            return {"error": "Master watchlist not found"}

        # Get active items
        items = self.db.query(WatchlistItem).filter(
            WatchlistItem.watchlist_id == master_watchlist_id,
            WatchlistItem.status == "active",
        ).all()

        if not items:
            return {"message": "No active symbols in watchlist"}

        symbols = [item.symbol for item in items]

        # Fetch candle data for all symbols
        candles_by_symbol = self._fetch_candles(symbols)

        # Volume pre-filter: skip symbols with insufficient liquidity
        # NSE FNO-grade liquidity: avg volume per bar >= 200k (saves LLM calls + noise)
        MIN_AVG_VOLUME = 200_000
        liquid_symbols: set[str] = set()
        for sym, candles in candles_by_symbol.items():
            if not candles:
                continue
            if len(candles) >= 20:
                avg_vol = sum(c.get("volume", 0) for c in candles[-20:]) / 20
                if avg_vol >= MIN_AVG_VOLUME:
                    liquid_symbols.add(sym)
                else:
                    logger.debug(
                        "Skipping illiquid symbol %s: avg vol %.0f < %.0f",
                        sym, avg_vol, MIN_AVG_VOLUME,
                    )
            else:
                # Thin data — include anyway; _basic_analysis handles it gracefully
                liquid_symbols.add(sym)

        # Get all active strategy profiles
        profiles = self.db.query(StrategyProfile).filter(
            StrategyProfile.status == "active"
        ).all()

        if not profiles:
            return {"message": "No active strategy profiles"}

        results = {}
        for profile in profiles:
            # Ensure strategy watchlist exists
            strategy_wl = self._ensure_strategy_watchlist(profile)

            # Analyze each symbol against this profile
            added, updated, removed = [], [], []
            for item in items:
                # Skip illiquid symbols — no point analyzing thin stocks
                if item.symbol not in liquid_symbols:
                    continue
                candles = candles_by_symbol.get(item.symbol, [])
                analysis = self._analyze_match(item.symbol, candles, profile)

                # Update master item's last_analysis
                item.last_analysis = analysis
                item.updated_at = datetime.now(timezone.utc)

                match_score = analysis.get("match_score", 0)

                # Auto-add to strategy watchlist if score >= 60
                if match_score >= 60:
                    result = self._add_or_update_strategy_item(
                        strategy_wl, item, analysis, master_watchlist_id
                    )
                    if result == "added":
                        added.append(item.symbol)
                    elif result == "updated":
                        updated.append(item.symbol)
                else:
                    # Remove from strategy watchlist if score dropped below threshold
                    removed_sym = self._remove_if_exists(strategy_wl, item.symbol)
                    if removed_sym:
                        removed.append(item.symbol)

            results[profile.name] = {
                "added": added,
                "updated": updated,
                "removed": removed,
                "total_matched": len(added) + len(updated),
            }

        # Update master watchlist scan timestamp
        master.last_scanned_at = datetime.now(timezone.utc)
        self.db.commit()

        # Publish event
        bus = get_event_bus()
        bus.publish(Event(
            event_type=EventType.WATCHLIST_SCANNED,
            source="watchlist_analyzer",
            data={
                "master_watchlist_id": str(master_watchlist_id),
                "symbols_scanned": len(symbols),
                "results": {k: v["total_matched"] for k, v in results.items()},
            },
        ))

        return results

    def analyze_single(self, symbol: str, profile_id: UUID) -> dict:
        """Analyze a single symbol against a specific profile."""
        profile = self.db.query(StrategyProfile).get(profile_id)
        if not profile:
            return {"error": "Profile not found"}

        candles = self._fetch_candles([symbol]).get(symbol, [])
        return self._analyze_match(symbol, candles, profile)

    # ──────────────────────────────────────────────
    #  Analysis logic
    # ──────────────────────────────────────────────

    def _analyze_match(
        self, symbol: str, candles: list[dict], profile: StrategyProfile
    ) -> dict:
        """Analyze whether a symbol's chart matches a strategy profile."""
        if not candles or len(candles) < 20:
            return self._basic_analysis(symbol, candles, profile)

        # Try LLM analysis first
        try:
            return self._llm_analysis(symbol, candles, profile)
        except Exception:
            logger.debug("LLM analysis failed for %s, using basic analysis", symbol)
            return self._basic_analysis(symbol, candles, profile)

    def _llm_analysis(
        self, symbol: str, candles: list[dict], profile: StrategyProfile
    ) -> dict:
        """LLM-based match analysis."""
        # Summarize price action
        recent = candles[-20:]
        price_summary = self._format_price_summary(recent)
        current_price = candles[-1]["close"]
        trend = self._detect_trend(candles)
        volatility = self._calc_volatility(candles)
        volume_profile = self._volume_profile(candles)

        prompt = MATCH_PROMPT.format(
            strategy_name=profile.name,
            narrative=profile.narrative or "Not yet learned",
            entry_criteria=json.dumps(profile.entry_criteria or [], indent=2),
            regime_preferences=json.dumps(profile.regime_preferences or {}, indent=2),
            time_preferences=json.dumps(profile.time_preferences or {}, indent=2),
            symbol=symbol,
            price_summary=price_summary,
            current_price=current_price,
            trend=trend,
            volatility=volatility,
            volume_profile=volume_profile,
        )

        from agents.prompts import WATCHLIST_SCANNER

        llm = self._get_llm()
        _msgs = [
            LLMMessage(role="system", content=WATCHLIST_SCANNER),
            LLMMessage(role="user", content=prompt),
        ]
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        llm.chat(
                            messages=_msgs,
                            temperature=0.2,
                            max_tokens=400,
                            caller="watchlist_analyzer",
                        )
                    ).result()
            else:
                result = asyncio.run(llm.chat(
                    messages=_msgs,
                    temperature=0.2,
                    max_tokens=400,
                    caller="watchlist_analyzer",
                ))
        except Exception:
            raise

        content = result.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
            data["analysis_method"] = "llm"
            return data
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response")

    def _basic_analysis(
        self, symbol: str, candles: list[dict], profile: StrategyProfile
    ) -> dict:
        """Rule-based match analysis when LLM is unavailable."""
        score = 50  # Start neutral
        reasons = []

        if not candles or len(candles) < 5:
            return {
                "match_score": 0,
                "match_reason": "Insufficient candle data",
                "setup_forming": False,
                "regime_suitable": False,
                "recommended_action": "skip",
                "analysis_method": "basic",
            }

        # Trend detection
        trend = self._detect_trend(candles)
        regime_prefs = profile.regime_preferences or {}
        preferred = regime_prefs.get("preferred", [])

        # Trend alignment with regime preferences
        if preferred:
            trending_prefs = [r for r in preferred if "trend" in r.lower()]
            ranging_prefs = [r for r in preferred if "rang" in r.lower()]
            if trend in ("strong_up", "strong_down") and trending_prefs:
                score += 20
                reasons.append(f"Strong trend aligns with strategy preferences")
            elif trend == "ranging" and ranging_prefs:
                score += 15
                reasons.append(f"Ranging market suits strategy")
            elif trend in ("strong_up", "strong_down") and ranging_prefs:
                score -= 15
                reasons.append(f"Trending market doesn't suit range strategy")

        # Volume surge check
        vol = self._volume_profile(candles)
        if "above_average" in vol:
            score += 10
            reasons.append("Volume above average")

        # Volatility check
        volatility = self._calc_volatility(candles)
        risk_params = profile.risk_parameters or {}
        if "high" in volatility and risk_params.get("max_risk_pct", 2) < 1:
            score -= 10
            reasons.append("High volatility may exceed risk tolerance")

        score = max(0, min(100, score))
        return {
            "match_score": score,
            "match_reason": "; ".join(reasons) if reasons else "Basic analysis — no strong signals",
            "setup_forming": score >= 60,
            "regime_suitable": score >= 50,
            "recommended_action": "watch" if score >= 60 else "skip",
            "analysis_method": "basic",
        }

    # ──────────────────────────────────────────────
    #  Helper methods
    # ──────────────────────────────────────────────

    def _fetch_candles(self, symbols: list[str], n_bars: int = 100) -> dict[str, list[dict]]:
        """Fetch candles for multiple symbols."""
        result = {}
        for symbol in symbols:
            if self._fetcher:
                try:
                    result[symbol] = self._fetcher(symbol, n_bars)
                except Exception:
                    logger.warning("Failed to fetch candles for %s", symbol)
                    result[symbol] = []
            else:
                result[symbol] = []
        return result

    def _ensure_strategy_watchlist(self, profile: StrategyProfile) -> Watchlist:
        """Get or create the strategy-specific watchlist."""
        wl = self.db.query(Watchlist).filter(
            Watchlist.strategy_profile_id == profile.id,
            Watchlist.watchlist_type == "strategy",
        ).first()

        if not wl:
            wl = Watchlist(
                name=f"{profile.name} Watchlist",
                watchlist_type="strategy",
                strategy_profile_id=profile.id,
                description=f"Auto-managed watchlist for {profile.name} strategy",
            )
            self.db.add(wl)
            self.db.flush()

        return wl

    def _add_or_update_strategy_item(
        self,
        watchlist: Watchlist,
        master_item: WatchlistItem,
        analysis: dict,
        source_watchlist_id: UUID,
    ) -> str:
        """Add or update item in strategy watchlist. Returns 'added' or 'updated'."""
        existing = self.db.query(WatchlistItem).filter(
            WatchlistItem.watchlist_id == watchlist.id,
            WatchlistItem.symbol == master_item.symbol,
        ).first()

        if existing:
            existing.match_score = analysis.get("match_score", 0) / 100
            existing.match_reason = analysis.get("match_reason", "")
            existing.last_analysis = analysis
            existing.status = "matched"
            existing.updated_at = datetime.now(timezone.utc)
            return "updated"
        else:
            item = WatchlistItem(
                watchlist_id=watchlist.id,
                symbol=master_item.symbol,
                added_by="system",
                status="matched",
                match_score=analysis.get("match_score", 0) / 100,
                match_reason=analysis.get("match_reason", ""),
                last_analysis=analysis,
                source_watchlist_id=source_watchlist_id,
                source_item_id=master_item.id,
            )
            self.db.add(item)
            return "added"

    def _remove_if_exists(self, watchlist: Watchlist, symbol: str) -> bool:
        """Remove symbol from strategy watchlist if it exists. Returns True if removed."""
        existing = self.db.query(WatchlistItem).filter(
            WatchlistItem.watchlist_id == watchlist.id,
            WatchlistItem.symbol == symbol,
            WatchlistItem.added_by == "system",  # only auto-remove system-added items
        ).first()

        if existing:
            existing.status = "filtered_out"
            return True
        return False

    def _format_price_summary(self, candles: list[dict]) -> str:
        """Format recent candles for LLM context."""
        lines = []
        for i, c in enumerate(candles[-10:]):
            o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
            v = c.get("volume", 0)
            body = "green" if cl >= o else "red"
            lines.append(f"Bar {i+1}: O={o:.2f} H={h:.2f} L={l:.2f} C={cl:.2f} ({body}) Vol={v:,.0f}")
        return "\n".join(lines)

    def _detect_trend(self, candles: list[dict]) -> str:
        """Simple trend detection from price data."""
        if len(candles) < 20:
            return "unknown"
        closes = [c["close"] for c in candles]
        sma20 = sum(closes[-20:]) / 20
        sma5 = sum(closes[-5:]) / 5
        current = closes[-1]

        if current > sma20 * 1.02 and sma5 > sma20:
            return "strong_up"
        elif current < sma20 * 0.98 and sma5 < sma20:
            return "strong_down"
        elif abs(current - sma20) / sma20 < 0.01:
            return "ranging"
        elif current > sma20:
            return "mild_up"
        else:
            return "mild_down"

    def _calc_volatility(self, candles: list[dict]) -> str:
        """Calculate volatility classification."""
        if len(candles) < 14:
            return "unknown"
        ranges = [(c["high"] - c["low"]) / c["close"] * 100 for c in candles[-14:]]
        avg_range = sum(ranges) / len(ranges)
        if avg_range > 3:
            return "high"
        elif avg_range > 1.5:
            return "moderate"
        else:
            return "low"

    def _volume_profile(self, candles: list[dict]) -> str:
        """Volume profile relative to average."""
        if len(candles) < 20:
            return "unknown"
        vols = [c.get("volume", 0) for c in candles]
        if not any(vols):
            return "no_volume_data"
        avg_vol = sum(vols[-20:]) / 20
        recent_vol = sum(vols[-3:]) / 3
        if avg_vol == 0:
            return "no_volume_data"
        ratio = recent_vol / avg_vol
        if ratio > 1.5:
            return "above_average (surge)"
        elif ratio > 1.0:
            return "above_average"
        elif ratio > 0.7:
            return "average"
        else:
            return "below_average"


def get_watchlist_analyzer(db: Session) -> WatchlistAnalyzer:
    """Factory function for API dependency injection."""
    return WatchlistAnalyzer(db)
