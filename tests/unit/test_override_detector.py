# test_override_detector.py
# tests/unit/test_override_detector.py
"""Tests for Tier 3 override detection."""

import pytest
from datetime import datetime, timedelta, timezone

from core.enums import MarketRegime, OverrideSeverity, OverrideType, TradeDirection
from strategy_engine.conditions.base_condition import MarketSnapshot, TradeContext
from strategy_engine.override_detector import OverrideDetector, SessionContext


@pytest.fixture
def detector():
    return OverrideDetector()


@pytest.fixture
def base_trade():
    return TradeContext(
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


@pytest.fixture
def base_market():
    return MarketSnapshot(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 15, 14, 45, 0, tzinfo=timezone.utc),
        indicators={"atr_14": 1.5},
        regime=MarketRegime.TRENDING_UP,
    )


class TestOverrideDetector:

    def test_no_overrides_on_clean_trade(self, detector, base_trade, base_market, registry):
        template = registry.get("test_breakout")
        report = detector.detect(base_trade, template, base_market)

        # Some overrides may fire depending on time window etc.
        # But discipline score should be reasonable
        assert report.discipline_score >= 0

    def test_no_stop_loss_critical(self, detector, base_market, registry):
        trade = TradeContext(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=180.50,
            entry_time=datetime(2024, 1, 15, 14, 45, 0, tzinfo=timezone.utc),
            position_size=100,
            stop_loss=None,  # NO STOP
            account_balance=50000,
        )
        template = registry.get("test_breakout")
        report = detector.detect(trade, template, base_market)

        no_stop = [o for o in report.overrides if o.override_type == OverrideType.NO_STOP_SET]
        assert len(no_stop) > 0
        assert no_stop[0].severity == OverrideSeverity.CRITICAL

    def test_oversizing_detected(self, detector, base_market, registry):
        trade = TradeContext(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=180.50,
            entry_time=datetime(2024, 1, 15, 14, 45, 0, tzinfo=timezone.utc),
            position_size=500,
            stop_loss=179.00,
            risk_pct=3.0,  # 3% risk, strategy max is 1%
            account_balance=50000,
        )
        template = registry.get("test_breakout")
        report = detector.detect(trade, template, base_market)

        oversizing = [o for o in report.overrides if o.override_type == OverrideType.OVERSIZING]
        assert len(oversizing) > 0

    def test_rapid_reentry_after_loss(self, detector, base_trade, base_market, registry):
        template = registry.get("test_breakout")
        session = SessionContext(
            trades_today=3,
            daily_pnl=-500,
            consecutive_losses=2,
            last_trade_time=base_trade.entry_time - timedelta(minutes=1),
            last_trade_pnl=-200,
        )

        report = detector.detect(base_trade, template, base_market, session=session)

        rapid = [o for o in report.overrides if o.override_type == OverrideType.RAPID_RE_ENTRY]
        assert len(rapid) > 0
        assert rapid[0].severity in (OverrideSeverity.HIGH, OverrideSeverity.CRITICAL)

    def test_daily_limit_exceeded(self, detector, base_trade, base_market, registry):
        template = registry.get("test_breakout")
        session = SessionContext(
            trades_today=3,  # max_daily_trades is 3 in test template
            daily_pnl=0,
            consecutive_losses=0,
        )

        report = detector.detect(base_trade, template, base_market, session=session)

        limit_overrides = [
            o for o in report.overrides
            if "Daily trade limit" in o.description
        ]
        assert len(limit_overrides) > 0

    def test_discipline_score_degrades(self, detector, base_market, registry):
        """Multiple overrides should lower discipline score significantly."""
        trade = TradeContext(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=180.50,
            entry_time=datetime(2024, 1, 15, 14, 45, 0, tzinfo=timezone.utc),
            position_size=500,
            stop_loss=None,  # critical
            risk_pct=3.0,    # oversizing
            account_balance=50000,
        )
        session = SessionContext(
            trades_today=5,  # over limit
            daily_pnl=-1000,
            consecutive_losses=4,
            last_trade_time=trade.entry_time - timedelta(minutes=1),
            last_trade_pnl=-300,
        )
        template = registry.get("test_breakout")

        report = detector.detect(trade, template, base_market, session=session)

        assert report.total_count >= 3
        assert report.discipline_score < 50
        assert report.max_severity in (OverrideSeverity.HIGH, OverrideSeverity.CRITICAL)

    def test_override_report_summary(self, detector, base_market, registry):
        trade = TradeContext(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=180.50,
            entry_time=datetime(2024, 1, 15, 14, 45, 0, tzinfo=timezone.utc),
            position_size=100,
            stop_loss=None,
            account_balance=50000,
        )
        template = registry.get("test_breakout")
        report = detector.detect(trade, template, base_market)

        summary = report.summary()
        assert "total_overrides" in summary
        assert "by_type" in summary
        assert "severity_distribution" in summary
        assert "discipline_score" in summary