# recompute_analytics.py
# scripts/recompute_analytics.py
"""
Recompute all analytics from raw trade data.
Useful after fixing data or changing scoring logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, date
from sqlalchemy import and_
from core.database import get_db
from core.models import Trade, DailyPerformanceSummary
from core.enums import TradeStatus
from strategy_engine.strategy_dna.loader import StrategyDNALoader
from strategy_engine.strategy_dna.learning_updater import StrategyDNALearningUpdater
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recompute(days: int = 180):
    logger.info(f"Recomputing analytics for last {days} days...")

    with get_db() as db:
        # 1. Recompute daily summaries
        cutoff = datetime.now() - timedelta(days=days)
        trades = db.query(Trade).filter(
            and_(
                Trade.entry_time >= cutoff,
                Trade.status == TradeStatus.CLOSED,
            )
        ).order_by(Trade.entry_time).all()

        # Group by date
        by_date = {}
        for trade in trades:
            d = trade.entry_time.date()
            if d not in by_date:
                by_date[d] = []
            by_date[d].append(trade)

        for d, day_trades in by_date.items():
            pnls = [t.pnl for t in day_trades if t.pnl is not None]
            rs = [t.actual_r_multiple for t in day_trades if t.actual_r_multiple is not None]
            winners = sum(1 for p in pnls if p > 0)

            discipline_scores = []
            deviations = 0
            for t in day_trades:
                if t.tags:
                    ds = t.tags.get("discipline_score")
                    if ds is not None:
                        discipline_scores.append(ds)
                    if t.tags.get("overrides"):
                        deviations += 1

            existing = db.query(DailyPerformanceSummary).filter(
                DailyPerformanceSummary.date == d
            ).first()

            summary = existing or DailyPerformanceSummary(date=d)
            summary.total_trades = len(day_trades)
            summary.winners = winners
            summary.losers = len(pnls) - winners
            summary.total_pnl = sum(pnls) if pnls else 0
            summary.total_r = sum(rs) if rs else 0
            summary.win_rate = (winners / len(pnls) * 100) if pnls else 0
            summary.avg_r = np.mean(rs) if rs else 0
            summary.rule_adherence_rate = (
                (1 - deviations / len(day_trades)) * 100 if day_trades else 100
            )
            summary.avg_decision_quality = np.mean(discipline_scores) if discipline_scores else None
            summary.deviations_count = deviations

            if not existing:
                db.add(summary)

        logger.info(f"Recomputed {len(by_date)} daily summaries")

        # 2. Update strategy DNA statistics
        loader = StrategyDNALoader()
        loader.load_all()
        updater = StrategyDNALearningUpdater(db, loader)
        results = updater.update_all_strategies(lookback_days=days)
        logger.info(f"Updated DNA for {len(results)} strategies")

    logger.info("Analytics recomputation complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()
    recompute(args.days)