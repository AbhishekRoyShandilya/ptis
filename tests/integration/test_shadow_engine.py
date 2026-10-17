# test_shadow_engine.py
# tests/integration/test_shadow_engine.py
"""Integration tests for the shadow trading shadow_engine."""

import pytest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from core.enums import ShadowTradeType, TradeDirection, TradeStatus
from core.models import Trade
from shadow.engine import ShadowEngine
from shadow.models import ShadowSession, ShadowTrade


class TestShadowEngine:

    @pytest.fixture
    def shadow_engine(self):
        # not named `engine`: that would shadow the SQLAlchemy engine fixture the `db` fixture uses
        return ShadowEngine()

    @pytest.fixture
    def mock_trade(self, db):
        trade = Trade(
            id=uuid.uuid4(),
            symbol="AAPL",
            direction=TradeDirection.LONG,
            status=TradeStatus.OPEN,
            entry_price=180.50,
            entry_time=datetime(2024, 1, 15, 14, 45, tzinfo=timezone.utc),
            position_size=100,
            stop_loss=179.00,
            take_profit=183.00,
            risk_amount=150.0,
            risk_pct=0.3,
            declared_strategy_id="test_breakout",
        )
        db.add(trade)
        db.flush()
        return trade

    def test_create_shadow_session(self, shadow_engine, db, mock_trade, registry):
        session = shadow_engine.create_shadow_session(db, mock_trade)

        assert session is not None
        assert session.real_trade_id == mock_trade.id
        assert session.status == "active"

        shadows = (
            db.query(ShadowTrade)
            .filter(ShadowTrade.session_id == session.id)
            .all()
        )

        # Should have: mirror + variants (at least 2)
        assert len(shadows) >= 2

        types = [s.shadow_type for s in shadows]
        assert ShadowTradeType.MIRROR.value in types

    def test_mirror_shadow_matches_trade(self, shadow_engine, db, mock_trade, registry):
        session = shadow_engine.create_shadow_session(db, mock_trade)

        mirror = (
            db.query(ShadowTrade)
            .filter(
                ShadowTrade.session_id == session.id,
                ShadowTrade.shadow_type == ShadowTradeType.MIRROR.value,
            )
            .first()
        )

        assert mirror is not None
        assert mirror.entry_price == mock_trade.entry_price
        assert mirror.position_size == mock_trade.position_size
        assert mirror.stop_loss == mock_trade.stop_loss
        assert mirror.direction == mock_trade.direction.value

    def test_price_update(self, shadow_engine, db, mock_trade, registry):
        session = shadow_engine.create_shadow_session(db, mock_trade)

        mirror = (
            db.query(ShadowTrade)
            .filter(
                ShadowTrade.session_id == session.id,
                ShadowTrade.shadow_type == ShadowTradeType.MIRROR.value,
            )
            .first()
        )

        result = shadow_engine.update_price(db, mirror.id, 181.50)

        assert result["status"] == "open"
        assert result["unrealized_pnl"] > 0

        db.refresh(mirror)
        assert mirror.current_price == 181.50
        assert mirror.price_updates == 1

    def test_stop_loss_hit_closes_shadow(self, shadow_engine, db, mock_trade, registry):
        session = shadow_engine.create_shadow_session(db, mock_trade)

        mirror = (
            db.query(ShadowTrade)
            .filter(
                ShadowTrade.session_id == session.id,
                ShadowTrade.shadow_type == ShadowTradeType.MIRROR.value,
            )
            .first()
        )

        # Price drops to stop
        result = shadow_engine.update_price(db, mirror.id, 178.50)

        assert result["status"] == "closed"
        assert result["exit_reason"] == "stop_hit"

        db.refresh(mirror)
        assert mirror.status == "stop_hit"
        assert mirror.realized_pnl < 0

    def test_update_session_prices(self, shadow_engine, db, mock_trade, registry):
        session = shadow_engine.create_shadow_session(db, mock_trade)

        results = shadow_engine.update_session_prices(
            db, mock_trade.id, 181.00
        )

        # Should update all open shadows
        assert len(results) >= 2
        for r in results:
            assert "shadow_type" in r

    def test_close_session(self, shadow_engine, db, mock_trade, registry):
        session = shadow_engine.create_shadow_session(db, mock_trade)

        result = shadow_engine.close_session(db, mock_trade.id, final_price=182.00)

        assert result["status"] == "closed"
        assert len(result["shadows"]) >= 2

        db.refresh(session)
        assert session.status == "closed"


# tests/integration/test_shadow_comparison.py
"""Integration tests for shadow comparison."""

class TestShadowComparison:

    @pytest.fixture
    def setup_closed_trade(self, db, registry):
        """Create and close a trade with shadow session."""
        shadow_engine = ShadowEngine()

        trade = Trade(
            id=uuid.uuid4(),
            symbol="AAPL",
            direction=TradeDirection.LONG,
            status=TradeStatus.OPEN,
            entry_price=180.50,
            entry_time=datetime(2024, 1, 15, 14, 45, tzinfo=timezone.utc),
            position_size=100,
            stop_loss=179.00,
            take_profit=183.00,
            risk_amount=150.0,
            declared_strategy_id="test_breakout",
        )
        db.add(trade)
        db.flush()

        # Create shadow session
        shadow_engine.create_shadow_session(db, trade)

        # Simulate price movement
        for price in [180.80, 181.20, 181.80, 182.00, 181.50]:
            shadow_engine.update_session_prices(db, trade.id, price)

        # Close real trade
        trade.exit_price = 181.50
        trade.exit_time = trade.entry_time + timedelta(hours=1)
        trade.pnl = (181.50 - 180.50) * 100
        trade.r_multiple = trade.pnl / trade.risk_amount
        trade.status = TradeStatus.CLOSED
        db.flush()

        # Close shadows
        shadow_engine.close_session(db, trade.id, final_price=181.50)

        return trade

    def test_comparison_report(self, db, setup_closed_trade, registry):
        from shadow.comparison import ShadowComparisonEngine

        comparison = ShadowComparisonEngine()
        report = comparison.compare(db, setup_closed_trade.id)

        assert report is not None
        assert report.real_pnl == setup_closed_trade.pnl
        assert len(report.comparisons) >= 2
        assert report.execution_quality >= 0
        assert isinstance(report.insights, list)

    def test_comparison_creates_divergences(self, db, setup_closed_trade, registry):
        from shadow.comparison import ShadowComparisonEngine
        from shadow.models import TradeDivergence

        comparison = ShadowComparisonEngine()
        comparison.compare(db, setup_closed_trade.id)

        divergences = (
            db.query(TradeDivergence)
            .filter(TradeDivergence.real_trade_id == setup_closed_trade.id)
            .all()
        )

        # Should have at least PnL divergences
        assert len(divergences) >= 1

    def test_comparison_serializable(self, db, setup_closed_trade, registry):
        from shadow.comparison import ShadowComparisonEngine

        comparison = ShadowComparisonEngine()
        report = comparison.compare(db, setup_closed_trade.id)

        d = report.to_dict()
        assert "real_pnl" in d
        assert "comparisons" in d
        assert "insights" in d
        assert "execution_quality" in d

        # Should be JSON serializable
        import json
        json_str = json.dumps(d)
        assert len(json_str) > 0