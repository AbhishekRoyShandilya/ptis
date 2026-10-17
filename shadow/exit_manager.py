# shadow/exit_manager.py
"""
Shadow Exit Manager — centralised exit logic for shadow trades.

Handles stop-loss, trailing stops, partial exits, time-based exits,
and target hits. Called by ShadowEngine.update_price() but can also
be used standalone for backtesting shadow scenarios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from shadow.models import ShadowTrade

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    """Result of an exit check."""
    triggered: bool
    exit_price: float = 0.0
    exit_reason: str = ""
    partial: bool = False
    partial_size: float = 0.0
    partial_r: float = 0.0
    target_level: float = 0.0


class ShadowExitManager:
    """
    Evaluates all exit conditions for a shadow trade given a price tick.

    Exit priority (first match wins for full exits):
    1. Stop loss hit
    2. Trailing stop hit
    3. Full target hit (when no partial plan)
    4. Time-based exit
    """

    def check_exits(
        self,
        shadow: ShadowTrade,
        price: float,
        timestamp: datetime,
    ) -> ExitSignal:
        """Check all exit conditions. Returns ExitSignal."""

        # 1. Stop loss
        signal = self._check_stop_loss(shadow, price, timestamp)
        if signal.triggered:
            return signal

        # 2. Trailing stop
        signal = self._check_trailing_stop(shadow, price, timestamp)
        if signal.triggered:
            return signal

        # 3. Full target (no partials)
        signal = self._check_full_target(shadow, price, timestamp)
        if signal.triggered:
            return signal

        # 4. Time exit
        signal = self._check_time_exit(shadow, price, timestamp)
        if signal.triggered:
            return signal

        return ExitSignal(triggered=False)

    def check_partial_exits(
        self,
        shadow: ShadowTrade,
        price: float,
        timestamp: datetime,
    ) -> ExitSignal | None:
        """Check if a partial exit target is hit. Returns signal or None."""
        config = shadow.config or {}
        partial_plan = config.get("partial_exit_plan", [])
        if not partial_plan:
            return None

        if not shadow.risk_amount or shadow.risk_amount <= 0:
            return None

        remaining = shadow.remaining_size or shadow.position_size
        if remaining <= 0:
            return None

        already_exited = shadow.partial_exits or []
        exited_levels = {pe.get("target_level") for pe in already_exited}

        current_r = self._current_r(shadow, price)

        for level in partial_plan:
            r_target = level["r_multiple"]
            size_pct = level["size_pct"]

            if r_target in exited_levels:
                continue

            if current_r >= r_target:
                exit_size = min(
                    shadow.position_size * (size_pct / 100.0),
                    remaining,
                )
                return ExitSignal(
                    triggered=True,
                    exit_price=price,
                    exit_reason="partial_target",
                    partial=True,
                    partial_size=exit_size,
                    partial_r=round(current_r, 2),
                    target_level=r_target,
                )

        return None

    def update_trailing_stop(
        self,
        shadow: ShadowTrade,
        price: float,
    ) -> float | None:
        """
        Compute and return the new trailing stop level.
        Returns None if trailing is not active.
        """
        trailing = shadow.trailing_config or {}
        if not trailing.get("enabled"):
            return None

        # Check activation threshold
        activate_r = trailing.get("activate_at_r", 1.0)
        if shadow.risk_amount and shadow.risk_amount > 0:
            current_r = self._current_r(shadow, price)
            if current_r < activate_r:
                return shadow.trailing_stop

        trail_type = trailing.get("type", "percentage")
        params = trailing.get("params", {})
        is_long = shadow.direction == "long"

        if trail_type == "percentage":
            pct = params.get("percentage", 1.0) / 100
            new_stop = price * (1 - pct) if is_long else price * (1 + pct)

        elif trail_type == "atr":
            atr_mult = params.get("atr_multiplier", 1.5)
            atr = params.get("atr_value", 1.0)
            new_stop = (price - atr * atr_mult) if is_long else (price + atr * atr_mult)

        elif trail_type == "candle":
            # Use swing low/high with buffer
            buffer = params.get("buffer_ticks", 2) * 0.01
            if is_long:
                new_stop = (shadow.low_since_entry or shadow.entry_price) - buffer
            else:
                new_stop = (shadow.high_since_entry or shadow.entry_price) + buffer
        else:
            return shadow.trailing_stop

        # Only ratchet in favorable direction
        current = shadow.trailing_stop
        if current is not None:
            if is_long:
                new_stop = max(new_stop, current)
            else:
                new_stop = min(new_stop, current)

        return round(new_stop, 4)

    def compute_pnl(
        self,
        shadow: ShadowTrade,
        exit_price: float,
    ) -> tuple[float, float | None]:
        """
        Compute realized PnL including partials.
        Returns (total_pnl, r_multiple).
        """
        total_pnl = 0.0
        partials = shadow.partial_exits or []
        partial_size_exited = sum(p.get("size", 0) for p in partials)

        for partial in partials:
            p_price = partial["price"]
            p_size = partial["size"]
            if shadow.direction == "long":
                total_pnl += (p_price - shadow.entry_price) * p_size
            else:
                total_pnl += (shadow.entry_price - p_price) * p_size

        # Remaining position
        remaining = shadow.position_size - partial_size_exited
        if remaining > 0:
            if shadow.direction == "long":
                total_pnl += (exit_price - shadow.entry_price) * remaining
            else:
                total_pnl += (shadow.entry_price - exit_price) * remaining

        r_multiple = None
        if shadow.risk_amount and shadow.risk_amount > 0:
            r_multiple = round(total_pnl / shadow.risk_amount, 2)

        return round(total_pnl, 2), r_multiple

    # ── internal checks ──

    def _check_stop_loss(
        self, shadow: ShadowTrade, price: float, ts: datetime
    ) -> ExitSignal:
        if shadow.stop_loss is None:
            return ExitSignal(triggered=False)

        if shadow.direction == "long" and price <= shadow.stop_loss:
            return ExitSignal(
                triggered=True,
                exit_price=shadow.stop_loss,
                exit_reason="stop_hit",
            )
        if shadow.direction == "short" and price >= shadow.stop_loss:
            return ExitSignal(
                triggered=True,
                exit_price=shadow.stop_loss,
                exit_reason="stop_hit",
            )
        return ExitSignal(triggered=False)

    def _check_trailing_stop(
        self, shadow: ShadowTrade, price: float, ts: datetime
    ) -> ExitSignal:
        trailing = shadow.trailing_config or {}
        if not trailing.get("enabled"):
            return ExitSignal(triggered=False)

        trail_stop = shadow.trailing_stop
        if trail_stop is None:
            return ExitSignal(triggered=False)

        if shadow.direction == "long" and price <= trail_stop:
            return ExitSignal(
                triggered=True,
                exit_price=trail_stop,
                exit_reason="trailing_stop",
            )
        if shadow.direction == "short" and price >= trail_stop:
            return ExitSignal(
                triggered=True,
                exit_price=trail_stop,
                exit_reason="trailing_stop",
            )
        return ExitSignal(triggered=False)

    def _check_full_target(
        self, shadow: ShadowTrade, price: float, ts: datetime
    ) -> ExitSignal:
        config = shadow.config or {}
        if config.get("partial_exit_plan"):
            return ExitSignal(triggered=False)

        if shadow.take_profit is None:
            return ExitSignal(triggered=False)

        if shadow.direction == "long" and price >= shadow.take_profit:
            return ExitSignal(
                triggered=True,
                exit_price=shadow.take_profit,
                exit_reason="target_hit",
            )
        if shadow.direction == "short" and price <= shadow.take_profit:
            return ExitSignal(
                triggered=True,
                exit_price=shadow.take_profit,
                exit_reason="target_hit",
            )
        return ExitSignal(triggered=False)

    def _check_time_exit(
        self, shadow: ShadowTrade, price: float, ts: datetime
    ) -> ExitSignal:
        config = shadow.config or {}
        time_exit = config.get("time_exit", {})
        if not time_exit.get("enabled"):
            return ExitSignal(triggered=False)

        max_dur = time_exit.get("max_duration_minutes")
        if max_dur and shadow.entry_time:
            elapsed = (ts - shadow.entry_time).total_seconds() / 60
            if elapsed >= max_dur:
                return ExitSignal(
                    triggered=True,
                    exit_price=price,
                    exit_reason="time_exit",
                )

        close_before = time_exit.get("close_before")
        if close_before:
            try:
                from datetime import time as dt_time
                parts = close_before.split(":")
                close_t = dt_time(int(parts[0]), int(parts[1]))
                if ts.time() >= close_t:
                    return ExitSignal(
                        triggered=True,
                        exit_price=price,
                        exit_reason="time_exit",
                    )
            except (ValueError, IndexError):
                pass

        return ExitSignal(triggered=False)

    @staticmethod
    def _current_r(shadow: ShadowTrade, price: float) -> float:
        if not shadow.risk_amount or shadow.risk_amount <= 0:
            return 0.0
        if shadow.direction == "long":
            return ((price - shadow.entry_price) * shadow.position_size) / shadow.risk_amount
        return ((shadow.entry_price - price) * shadow.position_size) / shadow.risk_amount
