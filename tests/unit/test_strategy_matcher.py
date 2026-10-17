# tests/unit/test_strategy_matcher.py
"""Tests for Tier 2 strategy auto-matching."""

import pytest
from datetime import datetime, timezone

from core.enums import ConditionStatus, MarketRegime, TradeDirection
from strategy_engine.conditions.base_condition import MarketSnapshot, TradeContext
from strategy_engine.strategy_matcher import StrategyMatcher

# Import condition evaluators to register them
import strategy_engine.conditions.time_conditions
import strategy_engine.conditions.price_conditions
import strategy_engine.conditions.volume_conditions
import strategy_engine.conditions.indicator_conditions
import strategy_engine.conditions.candle_conditions
import strategy_engine.conditions.structure_conditions


class TestStrategyMatcher:

    def _make_trade(self, **overrides) -> TradeContext:
        defaults = {
            "symbol": "AAPL",
            "direction": TradeDirection.LONG,
            "entry_price": 180.50,
            "entry_time": datetime(2024, 1, 15, 14, 45, 0, tzinfo=timezone.utc),  # 9:45 ET
            "position_size": 100,
            "stop_loss": 179.00,
            "take_profit": 183.00,
            "risk_pct": 0.8,
            "account_balance": 50000,
        }
        defaults.update(overrides)
        return TradeContext(**defaults)

    def _make_market(self, **overrides) -> MarketSnapshot:
        defaults = {
            "symbol": "AAPL",
            "timestamp": datetime(2024, 1, 15, 14, 45, 0, tzinfo=timezone.utc),
            "open": 179.00,
            "high": 181.00,
            "low": 178.50,
            "close": 180.60,
            "volume": 500000,
            "avg_volume": 200000,
            "relative_volume": 2.5,
            "opening_range_high": 180.00,
            "opening_range_low": 178.50,
            "indicators": {
                "ema_9": 180.30,
                "ema_20": 179.80,
                "atr_14": 1.50,
                "vwap": 179.90,
                "rsi_14": 58.0,
            },
            "regime": MarketRegime.TRENDING_UP,
        }
        defaults.update(overrides)
        return MarketSnapshot(**defaults)

    def test_match_returns_results(self, registry):
        matcher = StrategyMatcher(registry)
        trade = self._make_trade()
        market = self._make_market()

        output = matcher.match(trade, market)

        assert output.strategies_evaluated > 0
        assert len(output.matches) > 0

    def test_breakout_conditions_evaluated(self, registry):
        matcher = StrategyMatcher(registry)
        trade = self._make_trade(entry_price=180.50)  # above OR high of 180.00
        market = self._make_market()

        output = matcher.match(trade, market)

        # Find test_breakout match
        test_match = None
        for m in output.matches:
            if m.strategy_id == "test_breakout":
                test_match = m
                break

        assert test_match is not None
        assert test_match.required_total == 2
        assert len(test_match.condition_results) > 0

    # tests/unit/test_strategy_matcher.py (continued)

    def test_high_score_on_full_match(self, registry):
        matcher = StrategyMatcher(registry)
        # Entry at 09:45 IST (UTC+5:30) = 04:15 UTC — within the 09:35–11:00 IST window
        from datetime import timezone
        trade = self._make_trade(
            entry_price=180.50,
            entry_time=datetime(2024, 1, 15, 4, 15, 0, tzinfo=timezone.utc),
        )
        market = self._make_market(relative_volume=2.5)

        output = matcher.match(trade, market)
        test_match = next(
            (m for m in output.matches if m.strategy_id == "test_breakout"), None
        )

        assert test_match is not None
        assert test_match.all_required_met is True
        assert test_match.match_score > 0.5
        assert test_match.is_valid_match is True

    def test_direction_filter(self, registry):
        """Short trade should not match a long-only strategy."""
        matcher = StrategyMatcher(registry)
        trade = self._make_trade(direction=TradeDirection.SHORT)
        market = self._make_market()

        output = matcher.match(trade, market)
        test_match = next(
            (m for m in output.matches if m.strategy_id == "test_breakout"), None
        )

        # Strategy is long-only, so short trade should be disqualified
        if test_match:
            assert test_match.disqualified is True
            assert test_match.direction_compatible is False

    def test_regime_mismatch_disqualifies(self, registry):
        matcher = StrategyMatcher(registry)
        trade = self._make_trade(market_regime=MarketRegime.LOW_VOL)
        market = self._make_market(regime=MarketRegime.LOW_VOL)

        output = matcher.match(trade, market)
        test_match = next(
            (m for m in output.matches if m.strategy_id == "test_breakout"), None
        )

        if test_match:
            assert test_match.disqualified is True

    def test_missing_required_lowers_score(self, registry):
        """No volume surge should fail required condition."""
        matcher = StrategyMatcher(registry)
        trade = self._make_trade(entry_price=180.50)
        market = self._make_market(
            relative_volume=0.5,  # below 1.5x threshold
            volume=100000,
            avg_volume=200000,
        )

        output = matcher.match(trade, market)
        test_match = next(
            (m for m in output.matches if m.strategy_id == "test_breakout"), None
        )

        assert test_match is not None
        assert test_match.all_required_met is False
        assert test_match.is_valid_match is False

    def test_avoid_condition_disqualifies(self, registry):
        """Entry after 11:30 ET should trigger avoid condition."""
        matcher = StrategyMatcher(registry)
        # 17:00 UTC = 12:00 ET (after 11:30 avoid)
        trade = self._make_trade(
            entry_time=datetime(2024, 1, 15, 17, 0, 0, tzinfo=timezone.utc)
        )
        market = self._make_market()

        output = matcher.match(trade, market)
        test_match = next(
            (m for m in output.matches if m.strategy_id == "test_breakout"), None
        )

        if test_match:
            assert test_match.avoid_triggered > 0

    def test_match_single_strategy(self, registry):
        matcher = StrategyMatcher(registry)
        trade = self._make_trade(entry_price=180.50)
        market = self._make_market()

        result = matcher.match_single("test_breakout", trade, market)
        assert result is not None
        assert result.strategy_id == "test_breakout"

    def test_match_nonexistent_strategy(self, registry):
        matcher = StrategyMatcher(registry)
        trade = self._make_trade()
        market = self._make_market()

        result = matcher.match_single("nonexistent", trade, market)
        assert result is None