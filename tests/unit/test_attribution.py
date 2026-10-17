# test_attribution.py
# tests/unit/test_attribution.py
"""Tests for the unified 3-tier attribution engine."""

import pytest
from datetime import datetime, timezone

from core.enums import AttributionTier, MarketRegime, TradeDirection
from strategy_engine.attribution import AttributionEngine
from strategy_engine.conditions.base_condition import MarketSnapshot, TradeContext

import strategy_engine.conditions.time_conditions
import strategy_engine.conditions.price_conditions
import strategy_engine.conditions.volume_conditions
import strategy_engine.conditions.indicator_conditions


class TestAttributionEngine:

    def _make_context(self, **kw):
        defaults = dict(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=180.50,
            entry_time=datetime(2024, 1, 15, 14, 45, 0, tzinfo=timezone.utc),
            position_size=100,
            stop_loss=179.00,
            take_profit=183.00,
            risk_pct=0.8,
            account_balance=50000,
        )
        defaults.update(kw)
        return TradeContext(**defaults)

    def _make_market(self, **kw):
        defaults = dict(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 14, 45, 0, tzinfo=timezone.utc),
            open=179.00, high=181.00, low=178.50, close=180.60,
            volume=500000, avg_volume=200000, relative_volume=2.5,
            opening_range_high=180.00, opening_range_low=178.50,
            indicators={"ema_9": 180.30, "ema_20": 179.80, "atr_14": 1.50, "vwap": 179.90},
            regime=MarketRegime.TRENDING_UP,
        )
        defaults.update(kw)
        return MarketSnapshot(**defaults)

    def test_tier1_declared_match(self, registry):
        engine = AttributionEngine(registry)
        trade = self._make_context(declared_strategy_id="test_breakout")
        market = self._make_market()

        result = engine.evaluate(trade, market)

        assert result.tier1_declared is True
        assert result.tier1_strategy_id == "test_breakout"
        assert result.resolved_strategy_id == "test_breakout"
        assert result.resolved_tier == AttributionTier.TIER1_DECLARED

    def test_tier2_auto_match(self, registry):
        engine = AttributionEngine(registry)
        trade = self._make_context(declared_strategy_id=None)
        market = self._make_market()

        result = engine.evaluate(trade, market)

        assert result.tier1_declared is False
        assert result.tier2_output is not None
        assert result.tier2_output.strategies_evaluated > 0

    def test_result_has_insights(self, registry):
        engine = AttributionEngine(registry)
        trade = self._make_context()
        market = self._make_market()

        result = engine.evaluate(trade, market)

        assert isinstance(result.insights, list)

    def test_result_serializable(self, registry):
        engine = AttributionEngine(registry)
        trade = self._make_context(declared_strategy_id="test_breakout")
        market = self._make_market()

        result = engine.evaluate(trade, market)
        d = result.to_dict()

        assert "tier1" in d
        assert "tier2" in d
        assert "tier3" in d
        assert "resolved" in d
        assert "insights" in d

    def test_no_stop_lowers_discipline(self, registry):
        engine = AttributionEngine(registry)
        trade = self._make_context(
            declared_strategy_id="test_breakout",
            stop_loss=None,
        )
        market = self._make_market()

        result = engine.evaluate(trade, market)

        assert result.discipline_score < 100
        assert result.tier3_override_report is not None
        assert result.tier3_override_report.total_count > 0