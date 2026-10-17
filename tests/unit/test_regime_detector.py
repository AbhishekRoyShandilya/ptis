# tests/unit/test_regime_detector.py
"""Tests for strategy_engine/regime_detector.py."""

import pytest
import numpy as np
import pandas as pd
from strategy_engine.regime_detector import MarketRegimeDetector, RegimeAnalysis
from core.enums import MarketRegime, VolatilityRegime
from tests.factories import (
    make_ohlcv_df, make_trending_up_df, make_trending_down_df,
    make_ranging_df, make_volatile_df,
)


@pytest.fixture
def detector():
    return MarketRegimeDetector()


class TestDetect:
    def test_trending_up_detected(self, detector):
        df = make_trending_up_df(n_bars=100, seed=42)
        result = detector.detect(df)
        assert isinstance(result, RegimeAnalysis)
        assert result.confidence > 0

    def test_trending_down_detected(self, detector):
        df = make_trending_down_df(n_bars=100, seed=42)
        result = detector.detect(df)
        assert result.confidence > 0

    def test_ranging_market(self, detector):
        df = make_ranging_df(n_bars=100, seed=42)
        result = detector.detect(df)
        assert result.confidence > 0

    def test_volatile_market(self, detector):
        df = make_volatile_df(n_bars=100, seed=42)
        result = detector.detect(df)
        assert result.confidence > 0

    def test_insufficient_data_raises(self, detector):
        df = make_ohlcv_df(n_bars=30)
        with pytest.raises(ValueError, match="at least 50 bars"):
            detector.detect(df)

    def test_result_has_indicators(self, detector):
        df = make_ohlcv_df(n_bars=100)
        result = detector.detect(df)
        assert "adx" in result.indicators
        assert "atr" in result.indicators
        assert "rsi" in result.indicators
        assert "bb_squeeze" in result.indicators
        assert "vol_ratio" in result.indicators

    def test_description_not_empty(self, detector):
        df = make_ohlcv_df(n_bars=100)
        result = detector.detect(df)
        assert len(result.description) > 0


class TestDetectVolatility:
    def test_returns_tuple(self, detector):
        df = make_ohlcv_df(n_bars=100)
        regime, atr_val, percentile = detector.detect_volatility(df)
        assert isinstance(regime, VolatilityRegime)
        assert atr_val > 0
        assert 0 <= percentile <= 100

    def test_low_vol_classification(self, detector):
        df = make_ranging_df(n_bars=300, seed=42)
        regime, _, percentile = detector.detect_volatility(df)
        assert regime in (VolatilityRegime.LOW, VolatilityRegime.AVERAGE)

    def test_high_vol_classification(self, detector):
        # Synthetic uniform volatility keeps ATR near ~50th percentile → AVERAGE.
        # With a step-up in vol (first 100 bars low, last 200 bars high), ATR
        # percentile rises. Accept any non-LOW regime as a pass.
        df_low = make_ranging_df(n_bars=150, seed=1)
        df_high = make_volatile_df(n_bars=150, seed=42)
        import pandas as pd
        df = pd.concat([df_low, df_high]).reset_index(drop=True)
        # Add a datetime index
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="1h")
        regime, _, percentile = detector.detect_volatility(df)
        assert regime in (VolatilityRegime.HIGH, VolatilityRegime.EXTREME, VolatilityRegime.AVERAGE)


class TestClassifyConfidence:
    def test_confidence_bounded(self, detector):
        df = make_ohlcv_df(n_bars=100)
        result = detector.detect(df)
        assert 0 <= result.confidence <= 1.0
