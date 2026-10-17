"""
Strategy Hot-Reload Handler
============================
Listens for STRATEGY_DISCOVERED (pattern discovery) and BOOTSTRAP_COMPLETED
(user-taught strategy) events, then immediately propagates the new strategy
to every subsystem that depends on it:

  1. StrategyRegistry  — reload() picks up the new YAML
  2. SetupScanner      — refresh_strategy_templates() so new strategy
                         is matched in ORB/VWAP/pullback detections
  3. All traders       — COGNITIVE_ALERT: "New strategy added — now scanning"

Result: the moment a user teaches a new strategy or accepts a discovered
pattern, the whole system (watchlist scanner, intelligence evaluation,
co-planner) instantly knows about it without restarting.
"""
from __future__ import annotations

import logging
from typing import Any

from core.events import Event, EventType, get_event_bus

logger = logging.getLogger(__name__)


class StrategyHotReloadHandler:
    """
    Subscribes to strategy lifecycle events and propagates changes
    to all dependent subsystems immediately.
    """

    def __init__(self) -> None:
        self._bus = get_event_bus()
        self._registered = False

    def register(self) -> None:
        """Subscribe to strategy events. Call once at startup."""
        if self._registered:
            return
        self._bus.subscribe(EventType.STRATEGY_DISCOVERED, self._on_strategy_event)
        self._bus.subscribe(EventType.BOOTSTRAP_COMPLETED, self._on_strategy_event)
        self._registered = True
        logger.info("StrategyHotReloadHandler registered for STRATEGY_DISCOVERED + BOOTSTRAP_COMPLETED")

    def _on_strategy_event(self, event: Event) -> None:
        """Handle any strategy-added event."""
        data = event.data or {}
        strategy_name = (
            data.get("strategy_name")
            or data.get("name")
            or "New Strategy"
        )
        yaml_path = data.get("yaml_path") or data.get("file_path")

        logger.info(
            "Strategy hot-reload triggered: '%s' (event=%s)",
            strategy_name, event.event_type.value,
        )

        # 1. Reload the strategy registry (picks up new YAML)
        self._reload_registry(strategy_name)

        # 2. Refresh scanner strategy templates
        self._refresh_scanner(strategy_name)

        # 3. Emit STRATEGY_RELOADED so frontend + other subscribers know
        self._bus.publish(Event(
            event_type=EventType.STRATEGY_RELOADED,
            source="strategy_hotreload",
            data={
                "strategy_name": strategy_name,
                "yaml_path": yaml_path,
                "trigger": event.event_type.value,
            },
        ))

        # 4. Push cognitive alert to the trader's dashboard
        self._bus.publish(Event(
            event_type=EventType.COGNITIVE_ALERT,
            source="strategy_hotreload",
            data={
                "alert_type": "STRATEGY_ADDED",
                "severity": "info",
                "message": (
                    f"Strategy '{strategy_name}' added — watchlist scanner, "
                    f"intelligence evaluation, and co-planner are now aware of it."
                ),
                "recommendation": (
                    "The scanner will start surfacing setups for this strategy "
                    "on your next watchlist scan cycle."
                ),
                "ttl": 300,
            },
        ))

    def _reload_registry(self, strategy_name: str) -> None:
        """Reload strategy templates from disk into the singleton registry."""
        try:
            from strategy_engine.strategy_registry import get_registry
            reg = get_registry()
            count = reg.reload()
            logger.info(
                "StrategyRegistry reloaded after '%s': %d templates active",
                strategy_name, count,
            )
        except Exception as e:
            logger.error("StrategyRegistry reload failed: %s", e)

    def _refresh_scanner(self, strategy_name: str) -> None:
        """Tell the running ScannerLoop about the new strategy template."""
        try:
            from automation.setup_scanner import get_scanner_loop
            scanner = get_scanner_loop()
            scanner.refresh_strategy_templates()
            logger.info(
                "ScannerLoop strategy templates refreshed after '%s'",
                strategy_name,
            )
        except Exception as e:
            logger.warning("Scanner template refresh failed: %s", e)


# Module-level singleton
_handler: StrategyHotReloadHandler | None = None


def get_hotreload_handler() -> StrategyHotReloadHandler:
    global _handler
    if _handler is None:
        _handler = StrategyHotReloadHandler()
    return _handler


def register_hotreload_handler() -> None:
    """Call once at app startup to activate strategy hot-reload."""
    get_hotreload_handler().register()
