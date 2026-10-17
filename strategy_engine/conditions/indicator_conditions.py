# indicator_conditions.py
# strategy_engine/conditions/indicator_conditions.py
"""Indicator-based condition evaluators."""

from __future__ import annotations

from typing import Any

from core.enums import ConditionStatus, TradeDirection
from strategy_engine.conditions.base_condition import (
    BaseCondition,
    ConditionResult,
    MarketSnapshot,
    TradeContext,
)
from strategy_engine.conditions.registry import register_condition


@register_condition
class EMAAlignment(BaseCondition):
    """Check if fast/slow EMAs are aligned with trade direction."""

    @property
    def condition_type(self) -> str:
        return "ema_alignment"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        fast_period = params.get("fast_period", 9)
        slow_period = params.get("slow_period", 20)
        require_direction = params.get("require_trend_direction", True)

        fast_val = market.get_indicator(f"ema_{fast_period}")
        slow_val = market.get_indicator(f"ema_{slow_period}")

        if fast_val is None or slow_val is None:
            return self._insufficient_data(f"ema_{fast_period} or ema_{slow_period}")

        fast_above_slow = fast_val > slow_val

        if require_direction:
            aligned = (
                (fast_above_slow and trade.direction == TradeDirection.LONG) or
                (not fast_above_slow and trade.direction == TradeDirection.SHORT)
            )
        else:
            aligned = fast_above_slow if trade.direction == TradeDirection.LONG else not fast_above_slow

        gap_pct = abs(fast_val - slow_val) / slow_val * 100 if slow_val else 0
        confidence = min(1.0, gap_pct / 0.5)  # higher gap = more confident alignment

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if aligned else ConditionStatus.NOT_MET,
            actual_value=f"EMA{fast_period}={fast_val:.2f}, EMA{slow_period}={slow_val:.2f}",
            expected_value=f"{'fast > slow' if trade.direction == TradeDirection.LONG else 'fast < slow'}",
            details=f"EMA{fast_period} {'>' if fast_above_slow else '<'} EMA{slow_period}, gap {gap_pct:.2f}%",
            confidence=confidence,
        )


@register_condition
class EMATrend(BaseCondition):
    """Check price position relative to EMA for trend confirmation."""

    @property
    def condition_type(self) -> str:
        return "ema_trend"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        period = params.get("period", 20)
        price_above_for_long = params.get("price_above_ema_for_long", True)

        ema_val = market.get_indicator(f"ema_{period}")
        if ema_val is None:
            return self._insufficient_data(f"ema_{period}")

        price_above = trade.entry_price > ema_val

        if price_above_for_long:
            aligned = (
                (price_above and trade.direction == TradeDirection.LONG) or
                (not price_above and trade.direction == TradeDirection.SHORT)
            )
        else:
            aligned = True  # no direction requirement

        dist_pct = abs(trade.entry_price - ema_val) / ema_val * 100

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if aligned else ConditionStatus.NOT_MET,
            actual_value=f"price={trade.entry_price:.2f}, ema_{period}={ema_val:.2f}",
            details=f"Price {'above' if price_above else 'below'} EMA{period} by {dist_pct:.2f}%",
            confidence=1.0,
        )