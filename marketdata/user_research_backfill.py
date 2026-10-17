"""
marketdata.user_research_backfill — populate the USER behavioural record from history.

Replays the user's already-synced CLOSED trades through the user-research capture
so the behavioural substrate (discipline, FOMO/revenge/overtrading detection,
execution quality) populates from real history immediately, instead of waiting for
new trades to flow. This is the batch form of the user-initiated Replay.

IMPORTANT — this does NOT touch the edge matrix. Strategy edge is measured from
AVT (perfect-discipline executions), never from raw user trades, because user
trades are contaminated by human noise. To populate the edge matrix, run
`python -m marketdata.strategy_research` instead.

PRECONDITION: this is only meaningful once the user has TAUGHT their strategies —
otherwise strategy-adherence can't be judged and the behavioural read is hollow.

POINT-IN-TIME for free: user_research.capture_entry freezes regime/features as-of
each historical entry_time from the lake (causal), and _behavior_state filters to
trades BEFORE that entry — so replaying a 2024 trade reconstructs exactly what was
knowable then. fill_outcome then attaches the realized result.

Usage:
    python -m marketdata.user_research_backfill [--trader <uuid>] [--since-days N]

Requires the lake (candles) + the DB (trades) + migrations applied.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def backfill_trades(db, trader_id: Optional[str] = None, since_days: int = 1825) -> dict:
    """
    Replay closed user trades → capture_entry + fill_outcome (behavioural record).
    Idempotent: user_research writes dedupe on sample_id, so re-running is safe.
    """
    from core.models import Trade
    from core.enums import TradeStatus
    from marketdata import user_research as U

    cutoff = datetime.utcnow() - timedelta(days=since_days)
    q = (
        db.query(Trade)
        .filter(Trade.status == TradeStatus.CLOSED)
        .filter(Trade.entry_time >= cutoff)
        .order_by(Trade.entry_time.asc())
    )
    if trader_id:
        q = q.filter(Trade.trader_id == trader_id)
    trades = q.all()

    captured = filled = errors = 0
    for t in trades:
        try:
            U.capture_entry(t, db)
            captured += 1
            if t.exit_time is not None:
                U.fill_outcome(t, db)
                filled += 1
        except Exception as e:
            errors += 1
            logger.warning("user_research backfill failed for trade %s: %s", t.id, e)

    logger.info("user_research backfill: %d captured, %d outcomes, %d errors", captured, filled, errors)
    return {"trades": len(trades), "captured": captured, "filled": filled, "errors": errors}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Backfill the user-research record from historical trades")
    p.add_argument("--trader", default=None, help="restrict to one trader_id (default: all)")
    p.add_argument("--since-days", type=int, default=1825, help="lookback window (default 5y)")
    args = p.parse_args(argv)

    from core.database import SessionLocal
    db = SessionLocal()
    try:
        stats = backfill_trades(db, trader_id=args.trader, since_days=args.since_days)
    finally:
        db.close()
    print(f"user_research: {stats['captured']} samples captured, {stats['filled']} outcomes filled "
          f"({stats['errors']} errors) from {stats['trades']} trades")
    print("note: strategy edge is NOT computed here — run `python -m marketdata.strategy_research` for that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
