# tests/unit/test_risk_rules.py
"""Tests for strategy_engine/risk_rules.py."""

import pytest
import pandas as pd
import numpy as np
from strategy_engine.risk_rules import AdaptiveRiskManager
from core.enums import TradeDirection, VolatilityRegime, MarketRegime
from tests.factories import make_ohlcv_df


@pytest.fixture
def rm():
    return AdaptiveRiskManager(trailing_trigger_pts=70.0, trail_step_pts=20.0)


@pytest.fixture
def df():
    return make_ohlcv_df(n_bars=50, start_price=100.0, volatility=0.015, seed=42)


class TestCalculateSL:
    def test_long_sl_below_entry(self, rm, df):
        sl = rm.calculate_sl(df, 2.0, VolatilityRegime.AVERAGE, TradeDirection.LONG, entry=100.0)
        assert sl < 100.0

    def test_short_sl_above_entry(self, rm, df):
        sl = rm.calculate_sl(df, 2.0, VolatilityRegime.AVERAGE, TradeDirection.SHORT, entry=100.0)
        assert sl > 100.0

    def test_high_vol_wider_stop(self, rm, df):
        sl_avg = rm.calculate_sl(df, 2.0, VolatilityRegime.AVERAGE, TradeDirection.LONG, entry=100.0)
        sl_high = rm.calculate_sl(df, 2.0, VolatilityRegime.HIGH, TradeDirection.LONG, entry=100.0)
        assert sl_high <= sl_avg

    def test_sl_bounded_by_min_atr(self, rm, df):
        atr = 2.0
        entry = 100.0
        sl = rm.calculate_sl(df, atr, VolatilityRegime.LOW, TradeDirection.LONG, entry=entry)
        assert entry - sl >= 0.5 * atr - 0.01

    def test_sl_bounded_by_max_atr(self, rm, df):
        atr = 2.0
        entry = 100.0
        sl = rm.calculate_sl(df, atr, VolatilityRegime.EXTREME, TradeDirection.LONG, entry=entry)
        assert entry - sl <= 3.5 * atr + 0.01

    def test_insufficient_data_raises(self, rm):
        tiny_df = pd.DataFrame({"open": [100], "high": [101], "low": [99], "close": [100.5]})
        with pytest.raises(ValueError):
            rm.calculate_sl(tiny_df, 2.0, VolatilityRegime.AVERAGE, TradeDirection.LONG)


class TestCalculateRR:
    def test_long_target_above_entry(self, rm):
        target, trail, rr = rm.calculate_rr(
            entry=100.0, sl=98.0, direction=TradeDirection.LONG,
            volatility=VolatilityRegime.AVERAGE, regime=MarketRegime.TRENDING_UP,
        )
        assert target > 100.0
        assert rr >= 1.5

    def test_short_target_below_entry(self, rm):
        target, trail, rr = rm.calculate_rr(
            entry=100.0, sl=102.0, direction=TradeDirection.SHORT,
            volatility=VolatilityRegime.AVERAGE, regime=MarketRegime.TRENDING_DOWN,
        )
        assert target < 100.0

    def test_trending_rr_higher_than_range(self, rm):
        _, _, rr_trend = rm.calculate_rr(100.0, 98.0, TradeDirection.LONG, VolatilityRegime.AVERAGE, MarketRegime.TRENDING_UP)
        _, _, rr_range = rm.calculate_rr(100.0, 98.0, TradeDirection.LONG, VolatilityRegime.AVERAGE, MarketRegime.RANGE)
        assert rr_trend >= rr_range

    def test_zero_risk_raises(self, rm):
        with pytest.raises(ValueError):
            rm.calculate_rr(100.0, 100.0, TradeDirection.LONG, VolatilityRegime.AVERAGE, MarketRegime.UNKNOWN)


class TestTrailingStop:
    def test_no_trigger_keeps_sl(self, rm):
        new_sl = rm.update_trailing_stop(130.0, 130.0, 170.0, 90.0, TradeDirection.LONG)
        assert new_sl == 90.0

    def test_at_trigger_moves_to_breakeven(self, rm):
        new_sl = rm.update_trailing_stop(170.0, 170.0, 170.0, 90.0, TradeDirection.LONG)
        assert new_sl == 170.0

    def test_one_step_beyond_trigger(self, rm):
        new_sl = rm.update_trailing_stop(190.0, 190.0, 170.0, 90.0, TradeDirection.LONG)
        assert new_sl == 170.0

    def test_two_steps_beyond_trigger(self, rm):
        new_sl = rm.update_trailing_stop(210.0, 210.0, 170.0, 90.0, TradeDirection.LONG)
        assert new_sl == 190.0

    def test_sl_never_moves_backwards_long(self, rm):
        new_sl = rm.update_trailing_stop(210.0, 210.0, 170.0, 195.0, TradeDirection.LONG)
        assert new_sl >= 195.0

    def test_short_trailing(self, rm):
        new_sl = rm.update_trailing_stop(30.0, 30.0, 30.0, 110.0, TradeDirection.SHORT)
        assert new_sl == 30.0


class TestPortfolioHeat:
    def test_allowed_within_cap(self):
        allowed, current, projected = AdaptiveRiskManager.check_portfolio_heat(1.5, [1.0, 1.5], 6.0)
        assert allowed
        assert projected == 4.0

    def test_blocked_over_cap(self):
        allowed, current, projected = AdaptiveRiskManager.check_portfolio_heat(2.0, [2.0, 2.0, 1.0], 6.0)
        assert not allowed
        assert projected > 6.0

    def test_empty_open_trades(self):
        allowed, current, projected = AdaptiveRiskManager.check_portfolio_heat(2.0, [], 6.0)
        assert allowed
        assert current == 0.0
