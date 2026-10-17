# backfill_shadows.py
# scripts/backfill_shadows.py
"""
Create shadow trades for historical trades that don't have them.
Useful when you import CSV trades and want shadow comparison.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from sqlalchemy import and_
from core.database import get_db
from core.models import Trade
from core.enums import TradeStatus
from shadow.models import ShadowTrade
from shadow.engine import ShadowTradeEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backfill(days: int = 90):
    logger.info(f"Backfilling shadow trades for last {days} days...")
    cutoff = datetime.now() - timedelta(days=days)

    with get_db() as db:
        # Find trades without shadow trades
        trades = db.query(Trade).filter(
            and_(
                Trade.entry_time >= cutoff,
                Trade.status == TradeStatus.CLOSED,
            )
        ).all()

        created = 0
        skipped = 0

        engine = ShadowTradeEngine(db)

        for trade in trades:
            # Check if shadows already exist
            existing = db.query(ShadowTrade).filter(
                ShadowTrade.real_trade_id == trade.id
            ).count()

            if existing > 0:
                skipped += 1
                continue

            try:
                shadows = engine.on_real_trade_entered(trade)

                # For closed trades, simulate the shadow trade lifecycle
                # using the trade's exit price as if it happened in real-time
                if trade.exit_price and trade.exit_time:
                    engine.update_price(
                        trade.symbol,
                        trade.exit_price,
                        trade.exit_time
                    )

                    # Also sync the real trade close
                    engine.on_real_trade_closed(trade)

                created += len(shadows)
            except Exception as e:
                logger.error(f"Failed for trade {trade.id}: {e}")

        logger.info(f"Backfill complete: {created} shadows created, {skipped} trades already had shadows")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    backfill(args.days)