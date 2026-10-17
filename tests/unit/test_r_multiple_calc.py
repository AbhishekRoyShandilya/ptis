# tests/unit/test_r_multiple_calc.py
"""Tests for R-multiple calculation logic.

R-multiple = PnL / risk_amount, or equivalently:
  Long:  R = (exit - entry) / (entry - stop)
  Short: R = (entry - exit) / (stop - entry)
"""

import pytest


def calc_r_multiple(entry: float, exit_: float, stop: float, direction: str) -> float:
    """Pure R-multiple calculation matching PTIS convention."""
    if direction == "long":
        risk = entry - stop
    else:
        risk = stop - entry
    if risk <= 0:
        return 0.0
    if direction == "long":
        return (exit_ - entry) / risk
    else:
        return (entry - exit_) / risk


class TestRMultipleLong:
    def test_1r_winner(self):
        assert calc_r_multiple(100, 102, 98, "long") == pytest.approx(1.0)

    def test_2r_winner(self):
        assert calc_r_multiple(100, 104, 98, "long") == pytest.approx(2.0)

    def test_half_r_winner(self):
        assert calc_r_multiple(100, 101, 98, "long") == pytest.approx(0.5)

    def test_1r_loser(self):
        assert calc_r_multiple(100, 98, 98, "long") == pytest.approx(-1.0)

    def test_breakeven(self):
        assert calc_r_multiple(100, 100, 98, "long") == pytest.approx(0.0)

    def test_beyond_stop_loss(self):
        r = calc_r_multiple(100, 97, 98, "long")
        assert r < -1.0


class TestRMultipleShort:
    def test_1r_winner(self):
        assert calc_r_multiple(100, 98, 102, "short") == pytest.approx(1.0)

    def test_2r_winner(self):
        assert calc_r_multiple(100, 96, 102, "short") == pytest.approx(2.0)

    def test_1r_loser(self):
        assert calc_r_multiple(100, 102, 102, "short") == pytest.approx(-1.0)

    def test_breakeven(self):
        assert calc_r_multiple(100, 100, 102, "short") == pytest.approx(0.0)


class TestEdgeCases:
    def test_zero_risk_returns_zero(self):
        assert calc_r_multiple(100, 105, 100, "long") == 0.0

    def test_negative_risk_returns_zero(self):
        assert calc_r_multiple(100, 105, 102, "long") == 0.0

    def test_large_r_multiple(self):
        r = calc_r_multiple(100, 110, 99.5, "long")
        assert r == pytest.approx(20.0)

    def test_fractional_prices(self):
        r = calc_r_multiple(2456.50, 2487.30, 2441.10, "long")
        expected = (2487.30 - 2456.50) / (2456.50 - 2441.10)
        assert r == pytest.approx(expected)
