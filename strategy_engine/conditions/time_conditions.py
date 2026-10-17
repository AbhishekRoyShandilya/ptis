# time_conditions.py
# strategy_engine/conditions/time_conditions.py
"""Time-based condition evaluators."""

from __future__ import annotations

from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from core.enums import ConditionStatus
from strategy_engine.conditions.base_condition import (
    BaseCondition,
    ConditionResult,
    MarketSnapshot,
    TradeContext,
)
from strategy_engine.conditions.registry import register_condition


@register_condition
class TimeInsideWindow(BaseCondition):
    """Check if trade entry time falls within a time window."""

    @property
    def condition_type(self) -> str:
        return "time_inside_window"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        tz_name = params.get("timezone", "Asia/Kolkata")
        start_str = params.get("start", "09:15")
        end_str = params.get("end", "15:30")

        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("Asia/Kolkata")

        entry_local = trade.entry_time.astimezone(tz)
        entry_t = entry_local.time()
        start_t = time(*map(int, start_str.split(":")))
        end_t = time(*map(int, end_str.split(":")))

        is_inside = start_t <= entry_t <= end_t

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if is_inside else ConditionStatus.NOT_MET,
            actual_value=entry_t.strftime("%H:%M:%S"),
            expected_value=f"{start_str}-{end_str}",
            details=f"Entry at {entry_t.strftime('%H:%M')} ({'inside' if is_inside else 'outside'} {start_str}-{end_str})",
            confidence=1.0,
        )


@register_condition
class TimeOutsideWindow(BaseCondition):
    """Avoid condition: true if trade time is AFTER given time."""

    @property
    def condition_type(self) -> str:
        return "time_outside_window"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        tz_name = params.get("timezone", "Asia/Kolkata")

        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("Asia/Kolkata")

        entry_local = trade.entry_time.astimezone(tz)
        entry_t = entry_local.time()

        after_str = params.get("after")
        before_str = params.get("before")

        violated = False
        detail_parts = []

        if after_str:
            after_t = time(*map(int, after_str.split(":")))
            if entry_t > after_t:
                violated = True
                detail_parts.append(f"entry {entry_t.strftime('%H:%M')} after {after_str}")

        if before_str:
            before_t = time(*map(int, before_str.split(":")))
            if entry_t < before_t:
                violated = True
                detail_parts.append(f"entry {entry_t.strftime('%H:%M')} before {before_str}")

        # For AVOID conditions: MET means the avoid condition IS triggered (bad)
        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if violated else ConditionStatus.NOT_MET,
            actual_value=entry_t.strftime("%H:%M:%S"),
            details="; ".join(detail_parts) if detail_parts else "Time window OK",
            confidence=1.0,
        )