# behavioral/improvement_tracker.py
"""
Tracks behavioral improvement over time.
Computes trend metrics for discipline, override frequency,
clean execution rate, and emotional regulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func, and_
from sqlalchemy.orm import Session as DBSession

from core.enums import BehavioralEventType, TradeStatus
from core.models import BehavioralEvent, OverrideEvent, StrategyAttribution, Trade

logger = logging.getLogger(__name__)


@dataclass
class PeriodMetrics:
    """Metrics for a single time period (week)."""
    period_start: datetime
    period_end: datetime
    trade_count: int = 0

    avg_discipline: float = 0.0
    override_rate: float = 0.0
    clean_execution_rate: float = 0.0
    negative_event_rate: float = 0.0
    positive_event_rate: float = 0.0

    avg_r_multiple: Optional[float] = None
    win_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "trade_count": self.trade_count,
            "avg_discipline": round(self.avg_discipline, 1),
            "override_rate": round(self.override_rate, 3),
            "clean_execution_rate": round(self.clean_execution_rate, 3),
            "negative_event_rate": round(self.negative_event_rate, 3),
            "positive_event_rate": round(self.positive_event_rate, 3),
            "avg_r_multiple": round(self.avg_r_multiple, 2) if self.avg_r_multiple else None,
            "win_rate": round(self.win_rate, 3),
        }


@dataclass
class ImprovementReport:
    """Overall improvement analysis."""
    periods: list[PeriodMetrics] = field(default_factory=list)

    # Trends (slope of metric over time)
    discipline_trend: float = 0.0      # positive = improving
    override_trend: float = 0.0        # negative = improving
    clean_exec_trend: float = 0.0      # positive = improving
    performance_trend: float = 0.0     # positive = improving

    # Milestones
    milestones: list[str] = field(default_factory=list)

    # Overall score
    improvement_score: float = 0.0     # -100 to +100

    def to_dict(self) -> dict[str, Any]:
        return {
            "periods": [p.to_dict() for p in self.periods],
            "trends": {
                "discipline": round(self.discipline_trend, 3),
                "override_rate": round(self.override_trend, 3),
                "clean_execution": round(self.clean_exec_trend, 3),
                "performance": round(self.performance_trend, 3),
            },
            "improvement_score": round(self.improvement_score, 1),
            "milestones": self.milestones,
        }


class ImprovementTracker:
    """
    Tracks trader improvement across weekly periods.
    Computes trend lines and identifies milestones.
    """

    def __init__(self, weeks_lookback: int = 12):
        self._weeks = weeks_lookback

    def compute_report(self, db: DBSession) -> ImprovementReport:
        """Build improvement report from weekly metrics."""
        report = ImprovementReport()
        now = datetime.utcnow()

        # Build weekly periods
        for week in range(self._weeks - 1, -1, -1):
            period_end = now - timedelta(weeks=week)
            period_start = period_end - timedelta(weeks=1)
            metrics = self._compute_period(db, period_start, period_end)
            if metrics.trade_count > 0:
                report.periods.append(metrics)

        if len(report.periods) < 2:
            return report

        # Compute trends
        report.discipline_trend = self._compute_trend(
            [p.avg_discipline for p in report.periods]
        )
        report.override_trend = self._compute_trend(
            [p.override_rate for p in report.periods]
        )
        report.clean_exec_trend = self._compute_trend(
            [p.clean_execution_rate for p in report.periods]
        )
        r_multiples = [
            p.avg_r_multiple for p in report.periods
            if p.avg_r_multiple is not None
        ]
        if r_multiples:
            report.performance_trend = self._compute_trend(r_multiples)

        # Overall improvement score
        report.improvement_score = self._compute_improvement_score(report)

        # Milestones
        report.milestones = self._detect_milestones(report)

        return report

    def _compute_period(
        self, db: DBSession, start: datetime, end: datetime
    ) -> PeriodMetrics:
        """Compute metrics for a single time period."""
        metrics = PeriodMetrics(period_start=start, period_end=end)

        trades = (
            db.query(Trade)
            .filter(
                Trade.entry_time >= start,
                Trade.entry_time < end,
                Trade.status == TradeStatus.CLOSED,
            )
            .all()
        )

        metrics.trade_count = len(trades)
        if not trades:
            return metrics

        trade_ids = [t.id for t in trades]

        # Win rate
        winners = sum(1 for t in trades if t.pnl and t.pnl > 0)
        metrics.win_rate = winners / len(trades)

        # Avg R-multiple
        r_vals = [t.r_multiple for t in trades if t.r_multiple is not None]
        if r_vals:
            metrics.avg_r_multiple = sum(r_vals) / len(r_vals)

        # Discipline scores
        attributions = (
            db.query(StrategyAttribution.discipline_score)
            .filter(StrategyAttribution.trade_id.in_(trade_ids))
            .all()
        )
        disc_scores = [a[0] for a in attributions if a[0] is not None]
        if disc_scores:
            metrics.avg_discipline = sum(disc_scores) / len(disc_scores)

        # Override rate
        override_count = (
            db.query(func.count(OverrideEvent.id))
            .filter(OverrideEvent.trade_id.in_(trade_ids))
            .scalar()
        )
        metrics.override_rate = override_count / len(trades)

        # Behavioral events
        behavioral = (
            db.query(BehavioralEvent)
            .filter(BehavioralEvent.trade_id.in_(trade_ids))
            .all()
        )
        neg = sum(1 for b in behavioral if not b.is_positive)
        pos = sum(1 for b in behavioral if b.is_positive)
        metrics.negative_event_rate = neg / len(trades)
        metrics.positive_event_rate = pos / len(trades)

        clean = sum(
            1 for b in behavioral
            if b.event_type == BehavioralEventType.CLEAN_EXECUTION
        )
        metrics.clean_execution_rate = clean / len(trades)

        return metrics

    def _compute_trend(self, values: list[float]) -> float:
        """Simple linear regression slope."""
        n = len(values)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum(
            (i - x_mean) * (v - y_mean) for i, v in enumerate(values)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _compute_improvement_score(self, report: ImprovementReport) -> float:
        """
        Composite improvement score (-100 to +100).
        Positive = improving, negative = declining.
        """
        weights = {
            "discipline": 0.35,
            "override": -0.25,   # negative trend is good
            "clean_exec": 0.25,
            "performance": 0.15,
        }

        score = (
            report.discipline_trend * weights["discipline"] * 100
            + report.override_trend * weights["override"] * 100
            + report.clean_exec_trend * weights["clean_exec"] * 100
            + report.performance_trend * weights["performance"] * 50
        )

        return max(-100, min(100, score))

    # behavioral/improvement_tracker.py (continued from _detect_milestones)

    def _detect_milestones(self, report: ImprovementReport) -> list[str]:
        """Detect notable milestones in improvement journey."""
        milestones = []
        periods = report.periods

        if len(periods) < 2:
            return milestones

        latest = periods[-1]
        previous = periods[-2]

        # ── Discipline milestones ──
        if latest.avg_discipline >= 90 and previous.avg_discipline < 90:
            milestones.append(
                "🏆 Discipline score crossed 90 for the first time this period"
            )

        if latest.avg_discipline >= 80:
            consecutive_80 = 0
            for p in reversed(periods):
                if p.avg_discipline >= 80:
                    consecutive_80 += 1
                else:
                    break
            if consecutive_80 >= 4:
                milestones.append(
                    f"🔥 {consecutive_80} consecutive weeks with discipline above 80"
                )

        # ── Clean execution milestones ──
        if latest.clean_execution_rate >= 0.7 and previous.clean_execution_rate < 0.7:
            milestones.append(
                "✅ Clean execution rate crossed 70% — strong rule adherence"
            )

        if latest.clean_execution_rate >= 0.9:
            milestones.append(
                "⭐ 90%+ clean execution rate this week — exceptional discipline"
            )

        # ── Override reduction ──
        if (
            latest.override_rate < previous.override_rate * 0.5
            and previous.override_rate > 0.1
        ):
            milestones.append(
                "📉 Override rate dropped by 50%+ — significant improvement"
            )

        if latest.override_rate == 0 and latest.trade_count >= 5:
            milestones.append(
                "🎯 Zero overrides this week across all trades"
            )

        # ── Behavioral improvement ──
        if latest.negative_event_rate == 0 and latest.trade_count >= 5:
            milestones.append(
                "🧘 Zero negative behavioral events this week"
            )

        # ── Performance + discipline combo ──
        if (
            latest.avg_discipline >= 80
            and latest.avg_r_multiple is not None
            and latest.avg_r_multiple > 0.5
            and latest.win_rate >= 0.5
        ):
            milestones.append(
                "💎 Strong discipline + positive expectancy — edge is working"
            )

        # ── Winning streak ──
        if latest.win_rate >= 0.7 and latest.trade_count >= 10:
            milestones.append(
                f"📈 {latest.win_rate*100:.0f}% win rate on {latest.trade_count} trades"
            )

        # ── Overall trend ──
        if report.improvement_score > 20:
            milestones.append(
                f"📊 Overall improvement score: +{report.improvement_score:.0f} — "
                f"strong upward trajectory"
            )
        elif report.improvement_score < -20:
            milestones.append(
                f"⚠️ Overall improvement score: {report.improvement_score:.0f} — "
                f"review recent behavioral changes"
            )

        return milestones