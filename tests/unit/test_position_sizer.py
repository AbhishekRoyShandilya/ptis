# tests/unit/test_position_sizer.py
"""Tests for strategy_engine/position_sizer.py."""

import pytest
from strategy_engine.position_sizer import (
    PositionSizer, SizingConfig, SizingResult, default_sizer, aggressive_sizer,
)
from core.enums import VolatilityRegime


@pytest.fixture
def sizer():
    return PositionSizer()


class TestBasicSizing:
    def test_basic_compute(self, sizer):
        result = sizer.compute(
            account_balance=100_000,
            entry=100.0,
            stop_loss=98.0,
            atr=2.0,
        )
        assert result.shares > 0
        assert result.risk_pct == pytest.approx(1.0, abs=0.1)
        assert not result.blocked

    def test_risk_amount_matches(self, sizer):
        result = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0, atr=2.0,
        )
        expected_risk = result.shares * 2.0  # stop distance = 2
        assert result.risk_amount == pytest.approx(expected_risk, rel=0.01)

    def test_zero_balance_blocked(self, sizer):
        result = sizer.compute(
            account_balance=0, entry=100.0, stop_loss=98.0, atr=2.0,
        )
        assert result.blocked
        assert result.shares == 0

    def test_equal_entry_stop_blocked(self, sizer):
        result = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=100.0, atr=2.0,
        )
        assert result.blocked

    def test_short_trade_sizing(self, sizer):
        result = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=102.0, atr=2.0,
        )
        assert result.shares > 0
        assert not result.blocked


class TestDrawdownScaling:
    def test_no_drawdown_full_size(self, sizer):
        result = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0,
            atr=2.0, drawdown_pct=0.0,
        )
        assert result.scaling_factor == pytest.approx(1.0)

    def test_10pct_drawdown_reduces_size(self, sizer):
        full = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0,
            atr=2.0, drawdown_pct=0.0,
        )
        reduced = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0,
            atr=2.0, drawdown_pct=0.10,
        )
        assert reduced.shares < full.shares

    def test_20pct_drawdown_halts_trading(self, sizer):
        result = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0,
            atr=2.0, drawdown_pct=0.25,
        )
        assert result.blocked
        assert result.shares == 0


class TestVolatilityScaling:
    def test_high_vol_reduces_size(self, sizer):
        normal = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0,
            atr=2.0, vol_regime=VolatilityRegime.AVERAGE,
        )
        high = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0,
            atr=2.0, vol_regime=VolatilityRegime.HIGH,
        )
        assert high.shares < normal.shares

    def test_extreme_vol_cuts_in_half(self, sizer):
        normal = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0,
            atr=2.0, vol_regime=VolatilityRegime.AVERAGE,
        )
        extreme = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0,
            atr=2.0, vol_regime=VolatilityRegime.EXTREME,
        )
        ratio = extreme.shares / normal.shares
        assert ratio == pytest.approx(0.5, abs=0.05)


class TestKelly:
    def test_kelly_with_sufficient_history(self, sizer):
        result = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0, atr=2.0,
            win_rate=0.55, avg_win_r=1.8, avg_loss_r=1.0, n_history=50,
        )
        assert result.kelly_edge > 0
        assert not result.blocked

    def test_kelly_insufficient_history_ignored(self, sizer):
        result = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0, atr=2.0,
            win_rate=0.55, avg_win_r=1.8, avg_loss_r=1.0, n_history=10,
        )
        assert result.kelly_edge == 0.0


class TestPortfolioHeat:
    def test_heat_cap_reduces_size(self, sizer):
        result = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0,
            atr=2.0, open_risk_pct=5.5,
        )
        assert result.risk_pct <= 0.6

    def test_heat_cap_blocks_entry(self, sizer):
        result = sizer.compute(
            account_balance=100_000, entry=100.0, stop_loss=98.0,
            atr=2.0, open_risk_pct=5.9,
        )
        assert result.blocked

    def test_check_portfolio_heat_allowed(self, sizer):
        allowed, msg = sizer.check_portfolio_heat(2.0, 1.5)
        assert allowed

    def test_check_portfolio_heat_blocked(self, sizer):
        allowed, msg = sizer.check_portfolio_heat(5.0, 2.0)
        assert not allowed


class TestConvictionAdjusted:
    def test_skip_returns_zero(self, sizer):
        assert sizer.conviction_adjusted_size(100, "skip") == 0.0

    def test_standard_returns_same(self, sizer):
        assert sizer.conviction_adjusted_size(100, "standard") == 100.0

    def test_aggressive_multiplier(self, sizer):
        assert sizer.conviction_adjusted_size(100, "aggressive") == 150.0

    def test_max_multiplier(self, sizer):
        assert sizer.conviction_adjusted_size(100, "max") == 200.0


class TestFactoryFunctions:
    def test_default_sizer(self):
        s = default_sizer()
        assert s.cfg.base_risk_pct == 1.0

    def test_aggressive_sizer(self):
        s = aggressive_sizer()
        assert s.cfg.base_risk_pct == 2.0
        assert s.cfg.kelly_fraction == 0.33
