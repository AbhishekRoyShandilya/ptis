# structure_conditions.py
# strategy_engine/conditions/structure_conditions.py
"""Market structure condition evaluators."""

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
class HigherLowStructure(BaseCondition):
    """Detect higher-low (or lower-high) structure in recent candles."""

    @property
    def condition_type(self) -> str:
        return "higher_low_structure"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        lookback = params.get("lookback_bars", 20)

        candles = market.recent_candles
        if not candles or len(candles) < 6:
            return self._insufficient_data("recent_candles (need >= 6)")

        candles_to_check = candles[:min(lookback, len(candles))]

        # Find swing lows (simple: bar lower than neighbors)
        lows = [c["low"] for c in candles_to_check]
        swing_lows = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swing_lows.append(lows[i])

        if len(swing_lows) < 2:
            return ConditionResult(
                condition_type=self.condition_type,
                status=ConditionStatus.INCONCLUSIVE,
                details=f"Only {len(swing_lows)} swing lows found, need at least 2",
            )

        # Note: swing_lows[0] is more recent (candles are most-recent-first)
        if trade.direction == TradeDirection.LONG:
            structure_intact = swing_lows[0] > swing_lows[1]  # higher low
            detail = f"Recent swing low {swing_lows[0]:.2f} vs prior {swing_lows[1]:.2f}"
        else:
            highs = [c["high"] for c in candles_to_check]
            swing_highs = []
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                    swing_highs.append(highs[i])
            if len(swing_highs) < 2:
                return ConditionResult(
                    condition_type=self.condition_type,
                    status=ConditionStatus.INCONCLUSIVE,
                    details="Not enough swing highs for lower-high check",
                )
            structure_intact = swing_highs[0] < swing_highs[1]  # lower high
            detail = f"Recent swing high {swing_highs[0]:.2f} vs prior {swing_highs[1]:.2f}"

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if structure_intact else ConditionStatus.NOT_MET,
            details=detail,
            confidence=0.8,
        )


@register_condition
class TrendBreak(BaseCondition):
    """Invalidation: price breaks and holds below/above trend EMA."""

    @property
    def condition_type(self) -> str:
        return "trend_break"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        period = params.get("ema_period", 20)
        ema_val = market.get_indicator(f"ema_{period}")

        if ema_val is None:
            return self._insufficient_data(f"ema_{period}")

        current_price = market.close or trade.entry_price

        if trade.direction == TradeDirection.LONG:
            broken = current_price < ema_val
        else:
            broken = current_price > ema_val

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if broken else ConditionStatus.NOT_MET,
            actual_value=f"{current_price:.2f}",
            expected_value=f"EMA{period}={ema_val:.2f}",
            details=f"Price {current_price:.2f} {'broke through' if broken else 'holding'} EMA{period} {ema_val:.2f}",
            confidence=1.0,
        )


@register_condition
class CounterTrendMomentum(BaseCondition):
    """Avoid: strong counter-trend momentum (e.g., RSI extreme)."""

    @property
    def condition_type(self) -> str:
        return "counter_trend_momentum"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        rsi_val = market.get_indicator("rsi_14") or market.get_indicator("rsi")
        if rsi_val is None:
            return self._insufficient_data("rsi")

        rsi_below = params.get("rsi_below")
        rsi_above = params.get("rsi_above")
        for_long = params.get("for_long", True)

        is_counter = False
        if for_long and trade.direction == TradeDirection.LONG and rsi_below:
            is_counter = rsi_val < rsi_below
        elif not for_long and trade.direction == TradeDirection.SHORT and rsi_above:
            is_counter = rsi_val > rsi_above

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if is_counter else ConditionStatus.NOT_MET,
            actual_value=f"RSI={rsi_val:.1f}",
            details=f"RSI {rsi_val:.1f} — {'counter-trend signal' if is_counter else 'no counter-trend'}",
            confidence=1.0,
        )