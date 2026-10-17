# csv_importer.py
# ingestion/csv_importer.py
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from core.models import Trade, TradeDirection, TradeStatus, SetupType
from core.enums import SessionType
import logging

logger = logging.getLogger(__name__)


class CSVTradeImporter:
    """
    Flexible CSV trade importer with column mapping support.
    Handles exports from common brokers (ThinkorSwim, IBKR, Tradovate, etc.)
    """

    # Default column mappings for common brokers
    BROKER_MAPPINGS = {
        "thinkorswim": {
            "symbol": "Symbol",
            "direction": "Side",
            "entry_price": "Price",
            "quantity": "Qty",
            "entry_time": "Date/Time",
        },
        "ibkr": {
            "symbol": "Symbol",
            "direction": "Buy/Sell",
            "entry_price": "T. Price",
            "quantity": "Quantity",
            "entry_time": "Date/Time",
        },
        "generic": {
            "symbol": "symbol",
            "direction": "direction",
            "entry_price": "entry_price",
            "exit_price": "exit_price",
            "quantity": "quantity",
            "entry_time": "entry_time",
            "exit_time": "exit_time",
            "stop_loss": "stop_loss",
            "take_profit": "take_profit",
            "strategy_name": "strategy",
            "setup_type": "setup_type",
            "notes": "notes",
        }
    }

    def __init__(self, db: Session, broker: str = "generic"):
        self.db = db
        self.column_mapping = self.BROKER_MAPPINGS.get(broker, self.BROKER_MAPPINGS["generic"])

    def import_file(
        self,
        file_path: str,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> List[Trade]:
        """Import trades from a CSV file."""
        if column_mapping:
            self.column_mapping = column_mapping

        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")

        trades = []
        errors = []

        for idx, row in df.iterrows():
            try:
                trade = self._row_to_trade(row)
                if trade:
                    trades.append(trade)
            except Exception as e:
                errors.append({"row": idx, "error": str(e)})
                logger.warning(f"Row {idx} failed: {e}")

        # Deduplicate against existing trades
        new_trades = self._deduplicate(trades)

        # Bulk insert
        self.db.add_all(new_trades)
        self.db.flush()

        logger.info(
            f"Imported {len(new_trades)} new trades "
            f"({len(trades) - len(new_trades)} duplicates skipped, "
            f"{len(errors)} errors)"
        )

        return new_trades

    def _row_to_trade(self, row: pd.Series) -> Optional[Trade]:
        """Convert a DataFrame row to a Trade model."""
        mapping = self.column_mapping

        # Parse direction
        direction_raw = str(self._get_value(row, mapping.get("direction", ""))).lower()
        if direction_raw in ("buy", "long", "b"):
            direction = TradeDirection.LONG
        elif direction_raw in ("sell", "short", "s"):
            direction = TradeDirection.SHORT
        else:
            raise ValueError(f"Unknown direction: {direction_raw}")

        # Parse times
        entry_time = pd.to_datetime(self._get_value(row, mapping.get("entry_time", "")))
        exit_time_raw = self._get_value(row, mapping.get("exit_time", ""))
        exit_time = pd.to_datetime(exit_time_raw) if pd.notna(exit_time_raw) else None

        # Parse prices
        entry_price = float(self._get_value(row, mapping.get("entry_price", "")))
        exit_price_raw = self._get_value(row, mapping.get("exit_price", ""))
        exit_price = float(exit_price_raw) if pd.notna(exit_price_raw) else None

        # Parse stop loss
        stop_raw = self._get_value(row, mapping.get("stop_loss", ""))
        stop_loss = float(stop_raw) if pd.notna(stop_raw) else None

        # Calculate R multiple
        actual_r = None
        if exit_price and stop_loss and stop_loss != entry_price:
            risk_per_unit = abs(entry_price - stop_loss)
            if direction == TradeDirection.LONG:
                actual_r = (exit_price - entry_price) / risk_per_unit
            else:
                actual_r = (entry_price - exit_price) / risk_per_unit

        quantity = float(self._get_value(row, mapping.get("quantity", "")))

        # Determine session type from entry time
        session_type = self._classify_session(entry_time)

        # Determine status
        status = TradeStatus.CLOSED if exit_price else TradeStatus.OPEN

        trade = Trade(
            symbol=str(self._get_value(row, mapping.get("symbol", ""))).upper().strip(),
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            quantity=quantity,
            position_value=entry_price * quantity,
            stop_loss=stop_loss,
            take_profit=float(self._get_value(row, mapping.get("take_profit", "")))
            if pd.notna(self._get_value(row, mapping.get("take_profit", ""))) else None,
            actual_r_multiple=actual_r,
            setup_type=self._parse_setup_type(
                self._get_value(row, mapping.get("setup_type", ""))
            ),
            strategy_name=self._get_value(row, mapping.get("strategy_name", "")),
            status=status,
            session_type=session_type,
            notes=self._get_value(row, mapping.get("notes", "")),
            source="csv",
        )

        return trade

    def _get_value(self, row: pd.Series, column_name: str):
        """Safely get a value from a row."""
        if not column_name or column_name not in row.index:
            return None
        return row[column_name]

    def _classify_session(self, dt: datetime) -> Optional[SessionType]:
        """Classify time into session type."""
        if dt is None:
            return None
        hour = dt.hour
        minute = dt.minute
        time_decimal = hour + minute / 60

        if time_decimal < 9.5:
            return SessionType.PREMARKET
        elif time_decimal < 10.5:
            return SessionType.OPEN
        elif time_decimal < 14.0:
            return SessionType.MIDDAY
        elif time_decimal < 16.0:
            return SessionType.POWER_HOUR
        else:
            return SessionType.AFTER_HOURS

    def _parse_setup_type(self, value) -> Optional[SetupType]:
        """Parse setup type string to enum."""
        if not value or pd.isna(value):
            return None
        value = str(value).lower().strip()
        mapping = {
            "breakout": SetupType.BREAKOUT,
            "pullback": SetupType.PULLBACK,
            "reversal": SetupType.REVERSAL,
            "momentum": SetupType.MOMENTUM,
            "range": SetupType.RANGE,
        }
        return mapping.get(value, SetupType.CUSTOM)

    def _deduplicate(self, trades: List[Trade]) -> List[Trade]:
        """Remove trades that already exist in database."""
        new_trades = []
        for trade in trades:
            existing = self.db.query(Trade).filter(
                Trade.symbol == trade.symbol,
                Trade.entry_time == trade.entry_time,
                Trade.entry_price == trade.entry_price,
                Trade.direction == trade.direction,
            ).first()
            if not existing:
                new_trades.append(trade)
        return new_trades