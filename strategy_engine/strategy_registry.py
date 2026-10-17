# strategy_engine/strategy_registry.py
"""
Central registry that combines YAML templates with runtime strategy state.
Single source of truth for all strategy lookups across the system.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from core.enums import MarketRegime, TradeDirection
from strategy_engine.templates.loader import StrategyTemplate, TemplateLoader

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Singleton-style registry wrapping TemplateLoader with
    runtime state, metrics caching, and lookup optimizations.
    """

    def __init__(self, templates_dir: str | None = None):
        self._loader = TemplateLoader(templates_dir)
        self._runtime_state: dict[str, dict[str, Any]] = {}
        self._initialized = False

    def initialize(self) -> int:
        """Load all templates. Call once at app startup."""
        count = self._loader.load_all()
        self._initialized = True

        for strategy_id in self._loader.templates:
            self._runtime_state[strategy_id] = {
                "enabled": True,
                "trades_today": 0,
                "daily_pnl": 0.0,
                "paused_until": None,
            }

        logger.info("Strategy registry initialized with %d strategies", count)
        return count

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def get(self, strategy_id: str) -> StrategyTemplate | None:
        return self._loader.get(strategy_id)

    def get_all_active(self) -> list[StrategyTemplate]:
        return [
            t for t in self._loader.active_templates.values()
            if self._runtime_state.get(t.id, {}).get("enabled", True)
        ]

    def get_candidates(
        self,
        direction: TradeDirection,
        regime: Optional[MarketRegime] = None,
    ) -> list[StrategyTemplate]:
        candidates = self._loader.get_candidates(direction, regime)
        return [
            c for c in candidates
            if self._runtime_state.get(c.id, {}).get("enabled", True)
        ]

    def list_ids(self) -> list[str]:
        return list(self._loader.templates.keys())

    def summary(self) -> list[dict[str, Any]]:
        return self._loader.summary()

    # ── Runtime State ──

    def increment_trade_count(self, strategy_id: str) -> None:
        if strategy_id in self._runtime_state:
            self._runtime_state[strategy_id]["trades_today"] += 1

    def update_daily_pnl(self, strategy_id: str, pnl_delta: float) -> None:
        if strategy_id in self._runtime_state:
            self._runtime_state[strategy_id]["daily_pnl"] += pnl_delta

    def is_daily_limit_reached(self, strategy_id: str) -> bool:
        template = self.get(strategy_id)
        state = self._runtime_state.get(strategy_id)
        if not template or not state:
            return False

        max_trades = template.risk_rules.max_daily_trades
        if max_trades and state["trades_today"] >= max_trades:
            return True
        return False

    def reset_daily_state(self) -> None:
        for state in self._runtime_state.values():
            state["trades_today"] = 0
            state["daily_pnl"] = 0.0
            state["paused_until"] = None

    def reload(self) -> int:
        count = self._loader.load_all()
        # preserve runtime state for existing strategies, init new ones
        for sid in self._loader.templates:
            if sid not in self._runtime_state:
                self._runtime_state[sid] = {
                    "enabled": True,
                    "trades_today": 0,
                    "daily_pnl": 0.0,
                    "paused_until": None,
                }
        logger.info("Strategy registry reloaded: %d strategies", count)
        return count


# ── Module-level singleton ──
_registry: StrategyRegistry | None = None


def get_registry(templates_dir: str | None = None) -> StrategyRegistry:
    global _registry
    if _registry is None:
        _registry = StrategyRegistry(templates_dir)
    return _registry