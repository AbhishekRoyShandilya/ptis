"""
Real-Time Intelligence Service

Orchestrates all intelligence components into a unified service
that can be consumed by the API layer, WebSocket handlers,
and the coaching agent.

Manages:
- Periodic market environment updates
- Continuous setup scanning
- Alert generation and delivery
- Caching of computed intelligence
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from intelligence.market_environment import (
    MarketEnvironment,
    MarketEnvironmentAnalyzer,
)
from intelligence.readiness_scanner import SetupScanner, ScanResult
from intelligence.r_estimator import RMultipleEstimator
from intelligence.trader_profile_intelligence import (
    TraderProfileBuilder,
    TraderIntelligenceProfile,
)
from intelligence.discretionary_engine import (
    DiscretionaryQueryEngine,
    DiscretionaryAssessment,
)
from strategy_engine.strategy_registry import StrategyRegistry
from strategy_engine.conditions.registry import ConditionRegistry

logger = logging.getLogger(__name__)


class IntelligenceCache:
    """In-memory cache for computed intelligence with TTL."""

    def __init__(self, default_ttl_seconds: int = 60):
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self.default_ttl = default_ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if datetime.utcnow() > expiry:
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds or self.default_ttl
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        self._cache[key] = (value, expiry)

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()


class RealtimeIntelligenceService:
    """
    Unified service for all real-time intelligence operations.

    This is the main entry point consumed by the API and WebSocket layers.
    """

    def __init__(
        self,
        db: AsyncSession,
        strategy_registry: StrategyRegistry,
        condition_registry: ConditionRegistry,
        price_fetcher: Optional[Callable] = None,
        env_cache_ttl: int = 30,
        profile_cache_ttl: int = 3600,
        scan_cache_ttl: int = 15,
    ):
        self.db = db
        self.cache = IntelligenceCache()

        # Core analyzers
        self.env_analyzer = MarketEnvironmentAnalyzer()
        self.r_estimator = RMultipleEstimator(db=db)
        self.profile_builder = TraderProfileBuilder(db=db)

        # Setup scanner
        self.scanner = SetupScanner(
            strategy_registry=strategy_registry,
            condition_registry=condition_registry,
            env_analyzer=self.env_analyzer,
            r_estimator=self.r_estimator,
        )

        # Discretionary engine
        self.discretionary_engine = DiscretionaryQueryEngine(
            env_analyzer=self.env_analyzer,
            setup_scanner=self.scanner,
            r_estimator=self.r_estimator,
        )

        # Price data fetcher
        self.price_fetcher = price_fetcher

        # Cache TTLs
        self.env_cache_ttl = env_cache_ttl
        self.profile_cache_ttl = profile_cache_ttl
        self.scan_cache_ttl = scan_cache_ttl

        # Alert subscribers
        self._alert_callbacks: list[Callable] = []

        # live cognitive support (monitors trade events and pushes alerts)
        try:
            from intelligence.live_support import get_live_support

            # instantiate support; it will hook into the event bus automatically
            self.live_support = get_live_support()
        except ImportError:
            self.live_support = None

        # subscribe to cognitive alert events so we can forward them via on_alert
        from core.events import get_event_bus, EventType
        import asyncio

        def _cognitive_handler(event: Any) -> None:
            # event.data is expected to already be an alert dict
            trader_id = event.data.get("trader_id", "")
            # deliver asynchronously
            asyncio.ensure_future(self._deliver_alerts([event.data], trader_id))

        get_event_bus().subscribe(EventType.COGNITIVE_ALERT, _cognitive_handler)

    # ─── Public API ──────────────────────────────────────────────

    async def get_environment(
        self,
        symbol: str,
        timeframe: str = "5m",
        force_refresh: bool = False,
    ) -> MarketEnvironment:
        """
        Get current market environment for a symbol.

        Uses cache with configurable TTL. Force refresh for
        real-time decisions.
        """
        cache_key = f"env:{symbol}:{timeframe}"

        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        df = await self._get_price_data(symbol, timeframe)
        environment = self.env_analyzer.analyze(df, symbol, timeframe)

        self.cache.set(cache_key, environment, self.env_cache_ttl)
        return environment

    async def scan_setups(
        self,
        symbol: str,
        trader_id: str,
        timeframe: str = "5m",
        strategy_names: Optional[list[str]] = None,
        force_refresh: bool = False,
    ) -> ScanResult:
        """
        Scan for active setups on a symbol.

        Returns readiness scores for all matching strategies.
        """
        strat_key = ",".join(strategy_names) if strategy_names else "all"
        cache_key = f"scan:{symbol}:{timeframe}:{trader_id}:{strat_key}"

        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        df = await self._get_price_data(symbol, timeframe)

        # OFFLOAD to a thread — scanner.scan is heavy synchronous narration/pattern work; the proactive WebSocket
        # loop calls this every 15s per symbol, so running it on the event loop FREEZES all API requests during
        # each scan. to_thread keeps the app responsive.
        import asyncio
        result = await asyncio.to_thread(
            self.scanner.scan,
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            strategy_names=strategy_names,
            trader_id=trader_id,
        )

        self.cache.set(cache_key, result, self.scan_cache_ttl)

        # Deliver alerts
        if result.alerts:
            await self._deliver_alerts(result.alerts, trader_id)

        return result

    async def scan_watchlist(
        self,
        symbols: list[str],
        trader_id: str,
        timeframe: str = "5m",
    ) -> dict[str, ScanResult]:
        """
        Scan all symbols on the trader's watchlist.

        Returns scan results keyed by symbol.
        """
        results = {}
        for symbol in symbols:
            try:
                result = await self.scan_setups(
                    symbol=symbol,
                    trader_id=trader_id,
                    timeframe=timeframe,
                )
                results[symbol] = result
            except Exception as e:
                logger.warning(f"Scan failed for {symbol}: {e}")
                # Return a stub so caller always gets an entry per symbol
                results[symbol] = ScanResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    setups=[],
                    top_setup=None,
                    alerts=[{
                        "type": "scan_error",
                        "message": str(e),
                        "priority": "low",
                    }],
                )

        return results

    # ─── Cognitive alert helpers ──────────────────────────────────

    async def get_recent_cognitive_alerts(self, trader_id: str) -> list[dict[str, Any]]:
        """Return a list of recent cognitive alerts for the given trader.

        Currently returns sample alerts so the Alerts tab can be validated.
        Real implementation: query in-memory buffer maintained by LiveCognitiveSupport
        on each TRADE_OPENED / position-size-changed event.
        """
        now = datetime.now(timezone.utc)
        return [
            {
                "id": "sample-001",
                "type": "volatility_spike",
                "severity": "critical",
                "title": "Volatility Spike Detected",
                "message": "India VIX jumped 12% in the last 30 minutes. Consider reducing position size or waiting for calmer conditions.",
                "action_required": "Review open positions and reduce size if overexposed.",
                "symbol": None,
                "timestamp": (now - timedelta(minutes=8)).isoformat(),
                "acknowledged": False,
            },
            {
                "id": "sample-002",
                "type": "regime_mismatch",
                "severity": "warning",
                "title": "Regime Mismatch — RELIANCE",
                "message": "RELIANCE is now in a ranging regime but your active scan is tuned for trending setups. Results may be unreliable.",
                "action_required": "Switch to a range-bound strategy or pause RELIANCE scan.",
                "symbol": "RELIANCE",
                "timestamp": (now - timedelta(minutes=23)).isoformat(),
                "acknowledged": False,
            },
            {
                "id": "sample-003",
                "type": "stop_widen",
                "severity": "info",
                "title": "Stop Widening Suggested",
                "message": "ATR expanded 40% since your last entry review. Consider widening stops to avoid premature exits.",
                "action_required": None,
                "symbol": None,
                "timestamp": (now - timedelta(minutes=45)).isoformat(),
                "acknowledged": True,
            },
        ]

    async def evaluate_discretionary(
        self,
        query: str,
        symbol: str,
        trader_id: str,
        direction: Optional[int] = None,
        todays_trade_count: int = 0,
        current_pnl: float = 0.0,
        timeframe: str = "5m",
    ) -> DiscretionaryAssessment:
        """
        Evaluate a discretionary trade query.

        This is the main entry point for "What do you think about
        this breakout?" type questions.
        """
        df = await self._get_price_data(symbol, timeframe)
        profile = await self.get_trader_profile(trader_id)

        assessment = await self.discretionary_engine.evaluate(
            query=query,
            symbol=symbol,
            df=df,
            trader_profile=profile,
            direction=direction,
            todays_trade_count=todays_trade_count,
            current_pnl=current_pnl,
            timeframe=timeframe,
        )

        return assessment

    async def get_trader_profile(
        self,
        trader_id: str,
        force_refresh: bool = False,
    ) -> TraderIntelligenceProfile:
        """
        Get the trader's intelligence profile.

        Cached for longer since it changes slowly.
        """
        cache_key = f"profile:{trader_id}"

        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        profile = await self.profile_builder.build(trader_id)
        self.cache.set(cache_key, profile, self.profile_cache_ttl)
        return profile

    async def get_setup_alerts(
        self,
        symbols: list[str],
        trader_id: str,
        min_priority: str = "medium",
    ) -> list[dict[str, Any]]:
        """
        Get all active setup alerts across watchlist.

        Used for the alert feed / notification panel.
        """
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        min_level = priority_order.get(min_priority, 2)

        all_alerts = []
        for symbol in symbols:
            try:
                result = await self.scan_setups(
                    symbol=symbol,
                    trader_id=trader_id,
                )
                for alert in result.alerts:
                    alert_level = priority_order.get(alert.get("priority", "low"), 1)
                    if alert_level >= min_level:
                        all_alerts.append(alert)
            except Exception as e:
                logger.warning(f"Alert scan failed for {symbol}: {e}")

        # Sort by priority
        all_alerts.sort(
            key=lambda a: priority_order.get(a.get("priority", "low"), 0),
            reverse=True,
        )

        return all_alerts

    async def get_environment_suitability(
        self,
        symbol: str,
        trader_id: str,
        timeframe: str = "5m",
    ) -> dict[str, Any]:
        """
        Get environment suitability score with strategy breakdown.

        Quick summary endpoint for the dashboard.
        """
        environment = await self.get_environment(symbol, timeframe)
        profile = await self.get_trader_profile(trader_id)

        # Score each strategy against current environment
        strategy_scores = {}
        for sp in profile.strategy_preferences:
            if not sp.is_core:
                continue

            # Base on regime alignment and trader history
            score = 0.5
            if sp.best_regime and environment.regime.value == sp.best_regime:
                score = 0.85
            elif sp.worst_regime and environment.regime.value == sp.worst_regime:
                score = 0.20
            elif sp.win_rate > 0.55:
                score = 0.65

            strategy_scores[sp.strategy_name] = {
                "suitability": round(score, 3),
                "win_rate": round(sp.win_rate, 3),
                "avg_r": round(sp.avg_r, 3),
                "best_regime": sp.best_regime,
            }

        return {
            "symbol": symbol,
            "regime": environment.regime.value,
            "environment_score": round(environment.environment_score, 3),
            "volatility_state": environment.volatility.state.value,
            "trend_direction": environment.trend.direction,
            "trend_strength": environment.trend.strength.value,
            "strategy_suitability": strategy_scores,
            "timestamp": environment.timestamp.isoformat(),
        }

    # ─── Alert System ────────────────────────────────────────────

    def on_alert(self, callback: Callable) -> None:
        """Register an alert callback."""
        self._alert_callbacks.append(callback)

    async def _deliver_alerts(
        self,
        alerts: list[dict[str, Any]],
        trader_id: str,
    ) -> None:
        """Deliver alerts to all registered callbacks."""
        for alert in alerts:
            alert["trader_id"] = trader_id
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Alert delivery failed: {e}")

    # ─── Data Fetching ───────────────────────────────────────────

    async def _get_price_data(
        self, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """
        Get OHLCV price data for a symbol.

        Uses the configured price fetcher, with caching.
        """
        cache_key = f"ohlcv:{symbol}:{timeframe}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        if self.price_fetcher:
            df = await self.price_fetcher(symbol, timeframe)
        else:
            raise RuntimeError(
                f"No price fetcher configured. Cannot get data for {symbol}."
            )

        if df is None or df.empty:
            raise ValueError(f"No price data available for {symbol}")

        # Cache with short TTL for price data
        self.cache.set(cache_key, df, ttl_seconds=15)
        return df