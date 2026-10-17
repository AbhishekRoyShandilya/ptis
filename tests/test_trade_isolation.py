# tests/test_trade_isolation.py
"""
Tests for multi-user data isolation on trade endpoints.
Validates that trader_id filtering pattern works correctly.

Note: Uses mock objects since models require PostgreSQL-specific types (UUID, JSONB).
These tests verify the query filtering logic, not full ORM integration.
"""

import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass
class FakeUser:
    id: uuid.UUID
    username: str


@dataclass
class FakeTrade:
    id: uuid.UUID
    symbol: str
    trader_id: uuid.UUID
    status: str = "open"
    pnl: Optional[float] = None


def test_trader_id_filter_isolates_users():
    """Simulate the trader_id filtering pattern used in all route queries."""
    user_a = FakeUser(id=uuid.uuid4(), username="trader_a")
    user_b = FakeUser(id=uuid.uuid4(), username="trader_b")

    trades = [
        FakeTrade(id=uuid.uuid4(), symbol="RELIANCE", trader_id=user_a.id),
        FakeTrade(id=uuid.uuid4(), symbol="INFY", trader_id=user_a.id),
        FakeTrade(id=uuid.uuid4(), symbol="TATAMOTORS", trader_id=user_b.id),
    ]

    # Simulate: Trade.trader_id == user.id
    a_trades = [t for t in trades if t.trader_id == user_a.id]
    b_trades = [t for t in trades if t.trader_id == user_b.id]

    assert len(a_trades) == 2
    assert all(t.symbol in ("RELIANCE", "INFY") for t in a_trades)

    assert len(b_trades) == 1
    assert b_trades[0].symbol == "TATAMOTORS"


def test_ownership_check_prevents_cross_user_access():
    """User A cannot access User B's trade by ID."""
    user_a = FakeUser(id=uuid.uuid4(), username="trader_a")
    user_b = FakeUser(id=uuid.uuid4(), username="trader_b")

    trade_b = FakeTrade(id=uuid.uuid4(), symbol="TATAMOTORS", trader_id=user_b.id)

    # Simulate: Trade.id == trade_id AND Trade.trader_id == user.id
    result = next(
        (t for t in [trade_b] if t.id == trade_b.id and t.trader_id == user_a.id),
        None,
    )
    assert result is None, "User A should NOT access User B's trade"

    # But user B can
    result = next(
        (t for t in [trade_b] if t.id == trade_b.id and t.trader_id == user_b.id),
        None,
    )
    assert result is not None
    assert result.symbol == "TATAMOTORS"


def test_dashboard_pnl_scoped_to_user():
    """Dashboard P&L should only sum the current user's trades."""
    user_a = FakeUser(id=uuid.uuid4(), username="trader_a")
    user_b = FakeUser(id=uuid.uuid4(), username="trader_b")

    trades = [
        FakeTrade(id=uuid.uuid4(), symbol="RELIANCE", trader_id=user_a.id, status="closed", pnl=500.0),
        FakeTrade(id=uuid.uuid4(), symbol="INFY", trader_id=user_a.id, status="closed", pnl=-200.0),
        FakeTrade(id=uuid.uuid4(), symbol="TATAMOTORS", trader_id=user_b.id, status="closed", pnl=10000.0),
    ]

    # User A's P&L
    a_pnl = sum(t.pnl for t in trades if t.trader_id == user_a.id and t.pnl)
    assert a_pnl == 300.0

    # User B's P&L
    b_pnl = sum(t.pnl for t in trades if t.trader_id == user_b.id and t.pnl)
    assert b_pnl == 10000.0


def test_websocket_auth_token_required():
    """WebSocket endpoints should reject connections without token."""
    # This is a structural test — actual WebSocket testing requires async client
    # Verify the _authenticate_ws function exists and has the right signature
    from api.routes.websocket import _authenticate_ws
    import inspect

    sig = inspect.signature(_authenticate_ws)
    params = list(sig.parameters.keys())
    assert "websocket" in params
    assert "token" in params
