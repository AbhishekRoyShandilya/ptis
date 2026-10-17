# tests/unit/test_enums.py
"""Tests for core/enums.py — verify all enum values exist and are str-compatible."""

import pytest
from core.enums import (
    TradeDirection, MarketRegime, VolatilityRegime, EmotionalState,
    TradeStatus, SessionType, AttributionTier, ConditionStatus,
    OverrideType, OverrideSeverity, BehavioralEventType, ShadowTradeType,
    SKOStatus, ConvictionBand, ScoringFunctionType,
)


class TestTradeDirection:
    def test_values(self):
        assert TradeDirection.LONG.value == "long"
        assert TradeDirection.SHORT.value == "short"

    def test_from_string(self):
        assert TradeDirection("long") == TradeDirection.LONG
        assert TradeDirection("short") == TradeDirection.SHORT

    def test_is_str(self):
        assert isinstance(TradeDirection.LONG, str)
        assert TradeDirection.LONG == "long"


class TestMarketRegime:
    EXPECTED = [
        "trending_up", "trending_down", "range", "ranging",
        "mean_reverting", "volatile", "high_volatility",
        "low_vol", "low_volatility", "breakout", "unknown",
    ]

    def test_all_values_present(self):
        values = [m.value for m in MarketRegime]
        for v in self.EXPECTED:
            assert v in values, f"Missing MarketRegime value: {v}"

    def test_values_unique(self):
        # aliases (e.g. "range"/"ranging") are distinct values, never duplicates of one another
        values = [m.value for m in MarketRegime]
        assert len(values) == len(set(values))

    def test_string_lookup(self):
        assert MarketRegime("trending_up") == MarketRegime.TRENDING_UP
        assert MarketRegime("breakout") == MarketRegime.BREAKOUT


class TestVolatilityRegime:
    def test_values(self):
        assert VolatilityRegime.LOW.value == "low"
        assert VolatilityRegime.AVERAGE.value == "average"
        assert VolatilityRegime.HIGH.value == "high"
        assert VolatilityRegime.EXTREME.value == "extreme"

    def test_count(self):
        assert len(VolatilityRegime) == 4


class TestEmotionalState:
    def test_count(self):
        assert len(EmotionalState) == 11

    def test_key_values(self):
        assert EmotionalState.CALM.value == "calm"
        assert EmotionalState.REVENGE.value == "revenge"
        assert EmotionalState.FOMO.value == "fomo"
        assert EmotionalState.NEUTRAL.value == "neutral"


class TestTradeStatus:
    def test_values(self):
        assert TradeStatus.OPEN.value == "open"
        assert TradeStatus.CLOSED.value == "closed"
        assert TradeStatus.CANCELLED.value == "cancelled"


class TestOverrideType:
    def test_count(self):
        assert len(OverrideType) == 18

    def test_key_values(self):
        assert OverrideType.EARLY_ENTRY.value == "early_entry"
        assert OverrideType.STOP_MOVED.value == "stop_moved"
        assert OverrideType.REGIME_MISMATCH.value == "regime_mismatch"


class TestOverrideSeverity:
    def test_ordering(self):
        vals = [s.value for s in OverrideSeverity]
        assert vals == ["low", "medium", "high", "critical"]


class TestSKOStatus:
    def test_lifecycle(self):
        assert SKOStatus.HYPOTHESIS.value == "hypothesis"
        assert SKOStatus.DEVELOPING.value == "developing"
        assert SKOStatus.VALIDATED.value == "validated"
        assert SKOStatus.PROVEN.value == "proven"
        assert SKOStatus.ARCHIVED.value == "archived"


class TestConvictionBand:
    def test_values(self):
        assert ConvictionBand.SKIP.value == "skip"
        assert ConvictionBand.MAX_CONVICTION.value == "max"

    def test_count(self):
        assert len(ConvictionBand) == 5


class TestInvalidLookup:
    def test_invalid_trade_direction_raises(self):
        with pytest.raises(ValueError):
            TradeDirection("sideways")

    def test_invalid_regime_raises(self):
        with pytest.raises(ValueError):
            MarketRegime("nonexistent")
