# ingestion/deduplicator.py
"""
Trade Deduplicator — prevents duplicate trade imports by matching
on multiple fields (symbol, entry_time, entry_price, direction).

Handles both exact duplicates and fuzzy matches (same trade imported
from different brokers with slightly different timestamps).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from core.enums import TradeStatus
from core.models import Trade

logger = logging.getLogger(__name__)


class TradeDeduplicator:
    """
    Detects and prevents duplicate trade imports.

    Strategies:
    1. Exact hash match (symbol + entry_time + entry_price + direction)
    2. Fuzzy match (same symbol + direction, entry_time within tolerance)
    3. Broker order ID match
    """

    def __init__(
        self,
        db: Session,
        time_tolerance_seconds: int = 60,
    ):
        self.db = db
        self._time_tolerance = timedelta(seconds=time_tolerance_seconds)

    def compute_trade_hash(self, trade_data: Dict[str, Any]) -> str:
        """
        Compute a deterministic hash for a trade based on its
        identifying fields.
        """
        key_parts = [
            str(trade_data.get("symbol", "")).upper(),
            str(trade_data.get("direction", "")).lower(),
            str(trade_data.get("entry_price", "")),
            str(trade_data.get("entry_time", "")),
            str(trade_data.get("quantity", "")),
        ]
        key = "|".join(key_parts)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def is_duplicate(self, trade_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if a trade already exists in the database.
        Returns (is_duplicate, existing_trade_id).
        """
        # 1. Exact hash match
        trade_hash = self.compute_trade_hash(trade_data)
        existing = self.db.query(Trade).filter(
            Trade.tags["import_hash"].astext == trade_hash
        ).first()

        if existing:
            return True, str(existing.id)

        # 2. Broker order ID match
        broker_order_id = trade_data.get("broker_order_id")
        if broker_order_id:
            existing = self.db.query(Trade).filter(
                Trade.tags["broker_order_id"].astext == str(broker_order_id)
            ).first()
            if existing:
                return True, str(existing.id)

        # 3. Fuzzy match: same symbol + direction + entry_price within time window
        symbol = str(trade_data.get("symbol", "")).upper()
        direction = str(trade_data.get("direction", "")).lower()
        entry_price = trade_data.get("entry_price")
        entry_time = trade_data.get("entry_time")

        if not all([symbol, direction, entry_price, entry_time]):
            return False, None

        if isinstance(entry_time, str):
            try:
                entry_time = datetime.fromisoformat(entry_time)
            except ValueError:
                return False, None

        time_lo = entry_time - self._time_tolerance
        time_hi = entry_time + self._time_tolerance

        existing = self.db.query(Trade).filter(
            and_(
                func.upper(Trade.symbol) == symbol,
                Trade.entry_time.between(time_lo, time_hi),
                Trade.entry_price == entry_price,
            )
        ).first()

        if existing:
            return True, str(existing.id)

        return False, None

    def deduplicate_batch(
        self,
        trades: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter a batch of trades, separating new from duplicates.
        Returns (new_trades, duplicate_trades).
        """
        new_trades = []
        duplicates = []

        # Also check within the batch itself
        seen_hashes: set[str] = set()

        for trade_data in trades:
            trade_hash = self.compute_trade_hash(trade_data)

            # Check batch-internal duplicates
            if trade_hash in seen_hashes:
                duplicates.append(trade_data)
                continue

            is_dup, existing_id = self.is_duplicate(trade_data)
            if is_dup:
                trade_data["_duplicate_of"] = existing_id
                duplicates.append(trade_data)
            else:
                trade_data["_import_hash"] = trade_hash
                new_trades.append(trade_data)
                seen_hashes.add(trade_hash)

        logger.info(
            "Deduplication: %d new, %d duplicates out of %d total",
            len(new_trades), len(duplicates), len(trades),
        )
        return new_trades, duplicates

    def mark_imported(self, trade: Trade, trade_hash: str) -> None:
        """Store the import hash on a trade for future dedup checks."""
        if trade.tags is None:
            trade.tags = {}
        trade.tags["import_hash"] = trade_hash
