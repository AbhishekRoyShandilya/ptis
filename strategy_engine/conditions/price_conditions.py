# price_conditions.py
# strategy_engine/conditions/price_conditions.py
"""Price action condition evaluators."""

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
class PriceBreakout(BaseCondition):
    """Detect if entry price breaks out of a reference range."""

    @property
    def condition_type(self) -> str:
        return "price_breakout"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        reference = params.get("reference", "opening_range")
        min_extension_pct = params.get("min_extension_pct", 0.0)

        if reference == "opening_range":
            high = market.opening_range_high
            low = market.opening_range_low
        elif reference == "prev_day":
            high = market.prev_day_high
            low = market.prev_day_low
        else:
            return self._insufficient_data(f"reference '{reference}'")

        if high is None or low is None:
            return self._insufficient_data("opening_range/prev_day levels")

        range_width = high - low
        if range_width <= 0:
            return ConditionResult(
                condition_type=self.condition_type,
                status=ConditionStatus.INCONCLUSIVE,
                details="Range width is zero",
            )

        min_ext = range_width * (min_extension_pct / 100.0)
        is_breakout = False

        if trade.direction == TradeDirection.LONG:
            is_breakout = trade.entry_price >= (high + min_ext)
            actual = f"entry={trade.entry_price:.2f} vs range_high={high:.2f}+{min_ext:.2f}"
        else:
            is_breakout = trade.entry_price <= (low - min_ext)
            actual = f"entry={trade.entry_price:.2f} vs range_low={low:.2f}-{min_ext:.2f}"

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if is_breakout else ConditionStatus.NOT_MET,
            actual_value=f"{trade.entry_price:.4f}",
            expected_value=f"{'>' if trade.direction == TradeDirection.LONG else '<'} {high + min_ext if trade.direction == TradeDirection.LONG else low - min_ext:.4f}",
            details=actual,
            confidence=1.0,
        )


@register_condition
class PullbackToLevel(BaseCondition):
    """Check if price pulled back near key levels (EMA, VWAP, etc.)."""

    @property
    def condition_type(self) -> str:
        return "pullback_to_level"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        levels_config = params.get("levels", [])
        tolerance_atr_pct = params.get("tolerance_atr_pct", 0.3)

        atr = market.atr
        if atr is None:
            return self._insufficient_data("atr")

        tolerance = atr * tolerance_atr_pct
        levels_checked = []
        closest_distance = float("inf")
        closest_level_name = None

        for level_name in levels_config:
            level_value = None
            if level_name == "ema_20":
                level_value = market.get_indicator("ema_20")
            elif level_name == "ema_9":
                level_value = market.get_indicator("ema_9")
            elif level_name == "vwap":
                level_value = market.get_indicator("vwap")
            elif level_name == "prior_breakout":
                # Check support/resistance levels
                for sr in market.support_levels + market.resistance_levels:
                    dist = abs(trade.entry_price - sr)
                    if dist < tolerance and dist < closest_distance:
                        closest_distance = dist
                        closest_level_name = f"sr_level_{sr:.2f}"
                continue

            if level_value is not None:
                dist = abs(trade.entry_price - level_value)
                levels_checked.append((level_name, level_value, dist))
                if dist < closest_distance:
                    closest_distance = dist
                    closest_level_name = level_name

        if not levels_checked and closest_level_name is None:
            return self._insufficient_data("pullback reference levels")

        is_near = closest_distance <= tolerance

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if is_near else ConditionStatus.NOT_MET,
            actual_value=f"dist={closest_distance:.4f}",
            expected_value=f"<= {tolerance:.4f} (ATR*{tolerance_atr_pct})",
            details=f"Closest level: {closest_level_name}, distance: {closest_distance:.4f}, tolerance: {tolerance:.4f}",
            confidence=min(1.0, tolerance / max(closest_distance, 0.001)),
        )


@register_condition
class PriceReclaim(BaseCondition):
    """Invalidation: price reclaims inside a reference range after breakout."""

    @property
    def condition_type(self) -> str:
        return "price_reclaim"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        reference = params.get("reference", "opening_range")

        if reference == "opening_range":
            high = market.opening_range_high
            low = market.opening_range_low
        else:
            return self._insufficient_data(f"reference '{reference}'")

        if high is None or low is None:
            return self._insufficient_data("range levels")

        current_price = market.close or trade.entry_price

        if trade.direction == TradeDirection.LONG:
            reclaimed = current_price < high
        else:
            reclaimed = current_price > low

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if reclaimed else ConditionStatus.NOT_MET,
            actual_value=f"{current_price:.4f}",
            details=f"Price {'reclaimed inside range' if reclaimed else 'holding breakout'}",
            confidence=1.0,
        )