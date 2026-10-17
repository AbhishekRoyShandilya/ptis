# generate_daily_summary.py
# scripts/generate_daily_summary.py
"""
Generate end-of-day summary.
Run via cron at market close or manually.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, date
from sqlalchemy import and_
from core.database import get_db
from core.models import Trade, DailyPerformanceSummary, JournalEntry
from core.enums import TradeStatus
from shadow.comparison import ComparisonEngine
from strategy_engine.strategy_dna.loader import StrategyDNALoader
from strategy_engine.strategy_dna.learning_updater import StrategyDNALearningUpdater
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_summary(target_date: date = None):
    target_date = target_date or date.today()
    logger.info(f"Generating daily summary for {target_date}")

    with get_db() as db:
        start = datetime.combine(target_date, datetime.min.time())
        end = datetime.combine(target_date, datetime.max.time())

        # Get today's closed trades
        trades = db.query(Trade).filter(
            and_(
                Trade.entry_time >= start,
                Trade.entry_time <= end,
                Trade.status == TradeStatus.CLOSED,
            )
        ).all()

        if not trades:
            logger.info("No trades today.")
            return

        # Calculate metrics
        pnls = [t.pnl for t in trades if t.pnl is not None]
        rs = [t.actual_r_multiple for t in trades if t.actual_r_multiple is not None]
        winners = sum(1 for p in pnls if p > 0)

        # Discipline
        discipline_scores = []
        deviations = 0
        for t in trades:
            if t.tags:
                ds = t.tags.get("discipline_score")
                if ds is not None:
                    discipline_scores.append(ds)
                if t.tags.get("overrides"):
                    deviations += 1

        # Create/update summary
        existing = db.query(DailyPerformanceSummary).filter(
            DailyPerformanceSummary.date == target_date
        ).first()

        summary = existing or DailyPerformanceSummary(date=target_date)

        summary.total_trades = len(trades)
        summary.winners = winners
        summary.losers = len(pnls) - winners
        summary.total_pnl = sum(pnls) if pnls else 0
        summary.total_r = sum(rs) if rs else 0
        summary.win_rate = (winners / len(pnls) * 100) if pnls else 0
        summary.avg_r = np.mean(rs) if rs else 0
        summary.rule_adherence_rate = (
            (1 - deviations / len(trades)) * 100 if trades else 100
        )
        summary.avg_decision_quality = np.mean(discipline_scores) if discipline_scores else None
        summary.deviations_count = deviations

        if not existing:
            db.add(summary)

        # Update strategy DNA learned statistics
        loader = StrategyDNALoader()
        loader.load_all()
        updater = StrategyDNALearningUpdater(db, loader)
        updater.update_all_strategies(lookback_days=90)

        # Generate shadow comparison
        try:
            comparison_engine = ComparisonEngine(db)
            comparison_engine.compare_session(target_date)
        except Exception as e:
            logger.error(f"Shadow comparison failed: {e}")

        # scripts/generate_daily_summary.py (CONTINUED)

        logger.info(
            f"Summary: {summary.total_trades} trades | "
            f"PnL: ${summary.total_pnl:.2f} | "
            f"WR: {summary.win_rate:.0f}% | "
            f"Discipline: {summary.avg_decision_quality:.0f}/100 | "
            f"Deviations: {summary.deviations_count}"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="YYYY-MM-DD", default=None)
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else None
    generate_summary(target)