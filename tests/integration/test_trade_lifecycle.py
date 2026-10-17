# tests/integration/test_trade_lifecycle.py
"""
Full integration test: trade entry → attribution → close → behavioral events.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from core.enums import (
    EmotionalState,
    MarketRegime,
    TradeDirection,
    TradeStatus,
)
from core.models import (
    BehavioralEvent,
    ConditionEvaluation,
    OverrideEvent,
    StrategyAttribution,
    Trade,
)
from core.trade_pipeline import MarketDataProvider, TradePipeline, TradeInput
from strategy_engine.conditions.base_condition import MarketSnapshot
from strategy_engine.strategy_registry import get_registry

import strategy_engine.conditions.time_conditions
import strategy_engine.conditions.price_conditions
import strategy_engine.conditions.volume_conditions
import strategy_engine.conditions.indicator_conditions


TRADER = "00000000-0000-0000-0000-0000000000a1"   # every real trade belongs to a trader (a users.id UUID)


class MockMarketDataProvider(MarketDataProvider):
    def get_snapshot(self, symbol, timestamp):
        return MarketSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            open=179.00, high=181.00, low=178.50, close=180.60,
            volume=500000, avg_volume=200000, relative_volume=2.5,
            opening_range_high=180.00, opening_range_low=178.50,
            indicators={
                "ema_9": 180.30, "ema_20": 179.80,
                "atr_14": 1.50, "vwap": 179.90, "rsi_14": 58.0,
            },
            regime=MarketRegime.TRENDING_UP,
        )


class TestTradeLifecycle:

    @pytest.fixture(autouse=True)
    def setup_registry(self, registry, monkeypatch):
        """The pipeline reads the global strategy registry; point it at the conftest test strategies."""
        monkeypatch.setattr("core.trade_pipeline.get_registry", lambda: registry)

    def test_full_trade_entry_flow(self, db, registry):
        pipeline = TradePipeline(market_data_provider=MockMarketDataProvider())

        input_data = TradeInput(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=180.50,
            position_size=100,
            stop_loss=179.00,
            take_profit=183.00,
            declared_strategy_id="test_breakout",
            emotion_pre=EmotionalState.CONFIDENT,
            conviction_score=7.5,
            account_balance=50000.0,
            trader_id=TRADER,
        )

        result = pipeline.enter_trade(db, input_data)

        # Verify trade created
        assert result.success is True
        assert result.trade_id is not None

        trade = db.query(Trade).filter(Trade.id == result.trade_id).first()
        assert trade is not None
        assert trade.symbol == "AAPL"
        assert trade.status == TradeStatus.OPEN

        # Verify attribution persisted
        attr = db.query(StrategyAttribution).filter(
            StrategyAttribution.trade_id == result.trade_id
        ).first()
        assert attr is not None
        assert attr.tier1_declared is True
        assert attr.resolved_strategy_id is not None
        assert attr.discipline_score is not None

        # Verify condition evaluations persisted
        cond_evals = db.query(ConditionEvaluation).filter(
            ConditionEvaluation.trade_id == result.trade_id
        ).all()
        assert len(cond_evals) > 0

        # Verify insights returned
        assert isinstance(result.insights, list)

    def test_full_trade_close_flow(self, db, registry):
        pipeline = TradePipeline(market_data_provider=MockMarketDataProvider())

        # Enter trade
        input_data = TradeInput(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=180.50,
            position_size=100,
            stop_loss=179.00,
            account_balance=50000.0,
            trader_id=TRADER,
        )
        entry_result = pipeline.enter_trade(db, input_data)
        assert entry_result.success

        # Close trade
        close_result = pipeline.close_trade(
            db=db,
            trade_id=entry_result.trade_id,
            exit_price=182.50,
            emotion_post=EmotionalState.CALM,
        )

        assert close_result["success"] is True
        assert close_result["pnl"] > 0

        # Verify trade updated
        trade = db.query(Trade).filter(Trade.id == entry_result.trade_id).first()
        assert trade.status == TradeStatus.CLOSED
        assert trade.pnl > 0
        assert trade.exit_price == 182.50

    def test_no_stop_creates_override_and_behavioral(self, db, registry):
        pipeline = TradePipeline(market_data_provider=MockMarketDataProvider())

        input_data = TradeInput(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=180.50,
            position_size=100,
            stop_loss=None,  # NO STOP
            declared_strategy_id="test_breakout",
            account_balance=50000.0,
            trader_id=TRADER,
        )

        result = pipeline.enter_trade(db, input_data)
        assert result.success

        # Override events should exist
        overrides = db.query(OverrideEvent).filter(
            OverrideEvent.trade_id == result.trade_id
        ).all()
        assert len(overrides) > 0

        no_stop_overrides = [o for o in overrides if o.override_type.value == "no_stop_set"]
        assert len(no_stop_overrides) > 0

        # Behavioral events should exist
        behavioral = db.query(BehavioralEvent).filter(
            BehavioralEvent.trade_id == result.trade_id
        ).all()
        assert len(behavioral) > 0

    def test_behavioral_outcome_backfill(self, db, registry):
        pipeline = TradePipeline(market_data_provider=MockMarketDataProvider())

        input_data = TradeInput(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=180.50,
            position_size=100,
            stop_loss=None,
            declared_strategy_id="test_breakout",
            account_balance=50000.0,
            trader_id=TRADER,
        )

        entry_result = pipeline.enter_trade(db, input_data)

        # Before close: outcomes should be null
        events_before = db.query(BehavioralEvent).filter(
            BehavioralEvent.trade_id == entry_result.trade_id,
            BehavioralEvent.outcome_validated == False,
        ).all()

        # Close trade
        pipeline.close_trade(db, entry_result.trade_id, exit_price=178.00)

        # After close: outcomes should be backfilled
        events_after = db.query(BehavioralEvent).filter(
            BehavioralEvent.trade_id == entry_result.trade_id,
            BehavioralEvent.outcome_validated == True,
        ).all()
        assert len(events_after) > 0
        assert events_after[0].outcome_pnl is not None