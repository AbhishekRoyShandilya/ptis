# candle_conditions.py
# strategy_engine/conditions/candle_conditions.py
"""Candle pattern condition evaluators."""

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


def _classify_candle(candle: dict[str, float], atr: float | None = None) -> list[str]:
    """Classify a single candle into pattern types."""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body = abs(c - o)
    full_range = h - l
    if full_range == 0:
        return ["doji"]

    body_ratio = body / full_range
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    patterns = []

    # Doji
    if body_ratio < 0.1:
        patterns.append("doji")

    # Hammer (long lower wick, small body near top)
    if lower_wick > body * 2 and upper_wick < body * 0.5 and c >= o:
        patterns.append("hammer")

    # Inverted hammer
    if upper_wick > body * 2 and lower_wick < body * 0.5 and c <= o:
        patterns.append("inverted_hammer")

    # Pin bar (either direction)
    if lower_wick > full_range * 0.6 or upper_wick > full_range * 0.6:
        patterns.append("pin_bar")

    # Engulfing requires previous candle (handled in multi-candle)
    # Marubozu (strong body, no wicks)
    if body_ratio > 0.85:
        patterns.append("marubozu")
        if c > o:
            patterns.append("bullish_marubozu")
        else:
            patterns.append("bearish_marubozu")

    # Bullish/bearish
    if c > o:
        patterns.append("bullish")
    elif c < o:
        patterns.append("bearish")

    return patterns


def _check_engulfing(current: dict, previous: dict) -> list[str]:
    """Check for engulfing patterns."""
    patterns = []
    curr_body = abs(current["close"] - current["open"])
    prev_body = abs(previous["close"] - previous["open"])

    if curr_body > prev_body:
        if current["close"] > current["open"] and previous["close"] < previous["open"]:
            patterns.append("bullish_engulfing")
        elif current["close"] < current["open"] and previous["close"] > previous["open"]:
            patterns.append("bearish_engulfing")

    return patterns


@register_condition
class RejectionCandle(BaseCondition):
    """Check for rejection candle patterns at key levels."""

    @property
    def condition_type(self) -> str:
        return "rejection_candle"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        required_types = set(params.get("types", ["hammer", "bullish_engulfing", "pin_bar"]))

        candles = market.recent_candles
        if not candles:
            return self._insufficient_data("recent_candles")

        current = candles[0]
        detected = set(_classify_candle(current, market.atr))

        # Check engulfing if we have previous candle
        if len(candles) >= 2:
            detected.update(_check_engulfing(current, candles[1]))

        matches = detected & required_types
        is_met = len(matches) > 0

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if is_met else ConditionStatus.NOT_MET,
            actual_value=", ".join(sorted(detected)) if detected else "none",
            expected_value=", ".join(sorted(required_types)),
            details=f"Detected: {sorted(detected)}, Required any of: {sorted(required_types)}, Matches: {sorted(matches)}",
            confidence=1.0 if is_met else 0.0,
        )


@register_condition
class CandleCloseBeyond(BaseCondition):
    """Check if candle closed beyond a reference level."""

    @property
    def condition_type(self) -> str:
        return "candle_close_beyond"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        reference = params.get("reference", "breakout_level")

        ref_value = None
        if reference == "breakout_level":
            if trade.direction == TradeDirection.LONG:
                ref_value = market.opening_range_high
            else:
                ref_value = market.opening_range_low
        elif reference == "vwap":
            ref_value = market.get_indicator("vwap")

        if ref_value is None:
            return self._insufficient_data(f"reference '{reference}'")

        candle_close = market.close
        if candle_close is None and market.recent_candles:
            candle_close = market.recent_candles[0].get("close")

        if candle_close is None:
            return self._insufficient_data("candle close price")

        if trade.direction == TradeDirection.LONG:
            beyond = candle_close > ref_value
        else:
            beyond = candle_close < ref_value

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if beyond else ConditionStatus.NOT_MET,
            actual_value=f"{candle_close:.4f}",
            expected_value=f"{'>' if trade.direction == TradeDirection.LONG else '<'} {ref_value:.4f}",
            details=f"Close {candle_close:.4f} {'>' if beyond else '<='} reference {ref_value:.4f}",
            confidence=1.0,
        )