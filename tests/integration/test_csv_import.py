# tests/integration/test_csv_import.py
"""Tests for CSV trade import — verifies Trade model creation from parsed CSV data."""

import pytest
import uuid
from datetime import datetime, timezone

from core.models import Trade, Base
from core.enums import TradeDirection, TradeStatus


class TestCSVImport:
    """Test creating trades from CSV-like parsed data using the DB session."""

    def _make_trade_from_row(self, row: dict) -> Trade:
        """Simulate CSV row → Trade model conversion."""
        return Trade(
            id=uuid.uuid4(),
            symbol=row["symbol"],
            direction=TradeDirection(row["direction"]),
            status=TradeStatus(row.get("status", "closed")),
            entry_price=float(row["entry_price"]),
            exit_price=float(row.get("exit_price", 0)) or None,
            entry_time=datetime.fromisoformat(row["entry_time"]),
            exit_time=datetime.fromisoformat(row["exit_time"]) if row.get("exit_time") else None,
            position_size=float(row["position_size"]),
            stop_loss=float(row.get("stop_loss", 0)) or None,
            pnl=float(row.get("pnl", 0)) or None,
        )

    def test_single_trade_import(self, db):
        row = {
            "symbol": "RELIANCE",
            "direction": "long",
            "entry_price": "2500.0",
            "exit_price": "2580.0",
            "entry_time": "2026-01-15T09:30:00+05:30",
            "exit_time": "2026-01-15T14:00:00+05:30",
            "position_size": "10",
            "stop_loss": "2450.0",
            "status": "closed",
            "pnl": "800.0",
        }
        trade = self._make_trade_from_row(row)
        db.add(trade)
        db.flush()

        result = db.query(Trade).filter_by(symbol="RELIANCE").first()
        assert result is not None
        assert result.entry_price == 2500.0
        assert result.pnl == 800.0

    def test_bulk_import(self, db):
        rows = [
            {"symbol": "INFY", "direction": "long", "entry_price": "1500", "position_size": "20",
             "entry_time": "2026-01-15T09:30:00+05:30", "status": "open"},
            {"symbol": "TCS", "direction": "short", "entry_price": "3500", "position_size": "5",
             "entry_time": "2026-01-15T10:00:00+05:30", "status": "open"},
            {"symbol": "HDFC", "direction": "long", "entry_price": "2700", "position_size": "15",
             "entry_time": "2026-01-15T10:30:00+05:30", "status": "open"},
        ]
        for row in rows:
            db.add(self._make_trade_from_row(row))
        db.flush()

        count = db.query(Trade).count()
        assert count >= 3

    def test_missing_optional_fields(self, db):
        row = {
            "symbol": "SBIN",
            "direction": "long",
            "entry_price": "500",
            "position_size": "50",
            "entry_time": "2026-02-01T09:30:00+05:30",
        }
        trade = self._make_trade_from_row(row)
        db.add(trade)
        db.flush()

        result = db.query(Trade).filter_by(symbol="SBIN").first()
        assert result.exit_price is None
        assert result.pnl is None

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            TradeDirection("sideways")
