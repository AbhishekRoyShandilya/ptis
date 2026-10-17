# volume_conditions.py
# strategy_engine/conditions/volume_conditions.py
"""Volume-based condition evaluators."""

from __future__ import annotations

from typing import Any

from core.enums import ConditionStatus
from strategy_engine.conditions.base_condition import (
    BaseCondition,
    ConditionResult,
    MarketSnapshot,
    TradeContext,
)
from strategy_engine.conditions.registry import register_condition


@register_condition
class VolumeSurge(BaseCondition):
    """Detect if current volume is above a relative threshold."""

    @property
    def condition_type(self) -> str:
        return "volume_surge"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        min_rvol = params.get("min_relative_volume", 1.5)
        lookback = params.get("lookback_bars", 5)

        if market.relative_volume is not None:
            rvol = market.relative_volume
        elif market.volume and market.avg_volume and market.avg_volume > 0:
            rvol = market.volume / market.avg_volume
        elif market.recent_candles and len(market.recent_candles) >= lookback:
            avg_vol = sum(c.get("volume", 0) for c in market.recent_candles[:lookback]) / lookback
            rvol = (market.volume / avg_vol) if avg_vol > 0 and market.volume else None
        else:
            return self._insufficient_data("volume data")

        if rvol is None:
            return self._insufficient_data("relative_volume computation")

        is_surge = rvol >= min_rvol

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if is_surge else ConditionStatus.NOT_MET,
            actual_value=f"{rvol:.2f}x",
            expected_value=f">= {min_rvol}x",
            details=f"Relative volume {rvol:.2f}x vs required {min_rvol}x",
            confidence=min(1.0, rvol / min_rvol) if rvol > 0 else 0.0,
        )


@register_condition
class VolumeFade(BaseCondition):
    """Invalidation: volume drops after entry (breakout losing steam)."""

    @property
    def condition_type(self) -> str:
        return "volume_fade"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        min_sustain = params.get("min_sustain_ratio", 0.8)
        bars_after = params.get("bars_after_entry", 3)

        candles = market.recent_candles
        if not candles or len(candles) < bars_after + 1:
            return self._insufficient_data("recent_candles for volume fade check")

        entry_vol = candles[bars_after].get("volume", 0)  # candle at entry
        if entry_vol <= 0:
            return self._insufficient_data("entry candle volume")

        post_vols = [c.get("volume", 0) for c in candles[:bars_after]]
        avg_post = sum(post_vols) / len(post_vols) if post_vols else 0
        ratio = avg_post / entry_vol

        is_fading = ratio < min_sustain

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if is_fading else ConditionStatus.NOT_MET,
            actual_value=f"{ratio:.2f}",
            expected_value=f">= {min_sustain}",
            details=f"Post-entry volume ratio {ratio:.2f} vs sustain threshold {min_sustain}",
            confidence=1.0,
        )


@register_condition
class RangeWidthExcessive(BaseCondition):
    """Avoid: opening range too wide relative to ATR."""

    @property
    def condition_type(self) -> str:
        return "range_width_excessive"

    def evaluate(
        self, params: dict[str, Any], trade: TradeContext, market: MarketSnapshot
    ) -> ConditionResult:
        max_atr_mult = params.get("max_atr_multiple", 2.0)

        atr = market.atr
        or_high = market.opening_range_high
        or_low = market.opening_range_low

        if atr is None or or_high is None or or_low is None:
            return self._insufficient_data("ATR or opening range")

        range_width = or_high - or_low
        ratio = range_width / atr if atr > 0 else float("inf")
        excessive = ratio > max_atr_mult

        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.MET if excessive else ConditionStatus.NOT_MET,
            actual_value=f"{ratio:.2f}x ATR",
            expected_value=f"<= {max_atr_mult}x ATR",
            details=f"Range width {range_width:.2f} = {ratio:.2f}x ATR (max {max_atr_mult}x)",
            confidence=1.0,
        )