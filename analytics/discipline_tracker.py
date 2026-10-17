"""
Discipline Tracker
===================
Tracks discipline score trends over time.
Detects deterioration before it becomes a performance problem.
Identifies the specific rules being broken most frequently.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional


@dataclass
class DisciplineDataPoint:
    date: str
    avg_score: float
    trade_count: int
    override_count: int
    override_rate: float    # overrides / trades


@dataclass
class DisciplineReport:
    current_score: float = 0.0
    trend_7d: str = "stable"     # "improving" | "stable" | "deteriorating"
    trend_30d: str = "stable"
    avg_score_7d: float = 0.0
    avg_score_30d: float = 0.0
    avg_score_90d: float = 0.0
    worst_score: float = 0.0
    best_score: float = 100.0
    daily_series: list = field(default_factory=list)   # list[DisciplineDataPoint]
    top_rule_violations: list = field(default_factory=list)  # [(rule, count)]
    deterioration_days: list = field(default_factory=list)   # dates below 50
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "current_score": round(self.current_score, 1),
            "trends": {
                "7d": self.trend_7d, "30d": self.trend_30d,
                "avg_7d": round(self.avg_score_7d, 1),
                "avg_30d": round(self.avg_score_30d, 1),
                "avg_90d": round(self.avg_score_90d, 1),
            },
            "extremes": {"worst": round(self.worst_score, 1),
                         "best": round(self.best_score, 1)},
            "daily_series": [
                {"date": d.date, "avg_score": round(d.avg_score, 1),
                 "trade_count": d.trade_count, "override_count": d.override_count,
                 "override_rate": round(d.override_rate, 4)}
                for d in self.daily_series
            ],
            "top_rule_violations": self.top_rule_violations,
            "deterioration_days": self.deterioration_days,
            "computed_at": self.computed_at,
        }


class DisciplineTracker:

    def compute(self, trades: list) -> DisciplineReport:
        trades_with_disc = [t for t in trades
                            if getattr(t, "discipline_score", None) is not None]
        if not trades_with_disc:
            return DisciplineReport()

        # Group by date
        by_date: dict[str, list] = {}
        for t in trades_with_disc:
            et = getattr(t, "entry_time", None) or getattr(t, "exit_time", None)
            if et:
                d = et.date().isoformat() if isinstance(et, datetime) else str(et)
                by_date.setdefault(d, []).append(t)

        daily_series: list[DisciplineDataPoint] = []
        for d, ts in sorted(by_date.items()):
            scores = [t.discipline_score for t in ts]
            ovrs = sum(len(getattr(t, "overrides", [])) for t in ts)
            daily_series.append(DisciplineDataPoint(
                date=d, avg_score=statistics.mean(scores),
                trade_count=len(ts), override_count=ovrs,
                override_rate=ovrs / len(ts) if ts else 0.0,
            ))

        all_scores = [d.avg_score for d in daily_series]
        now = datetime.utcnow().date()

        def _avg_window(days: int) -> float:
            cutoff = (now - timedelta(days=days)).isoformat()
            pts = [d.avg_score for d in daily_series if d.date >= cutoff]
            return statistics.mean(pts) if pts else 0.0

        def _trend(days: int) -> str:
            cutoff = (now - timedelta(days=days)).isoformat()
            pts = [d for d in daily_series if d.date >= cutoff]
            if len(pts) < 4:
                return "stable"
            mid = len(pts) // 2
            first = statistics.mean(d.avg_score for d in pts[:mid])
            last  = statistics.mean(d.avg_score for d in pts[mid:])
            if last > first + 5: return "improving"
            if last < first - 5: return "deteriorating"
            return "stable"

        # Rule violations: look at behavioral events override types
        violation_counts: dict[str, int] = {}
        for t in trades:
            for ov in getattr(t, "overrides", []):
                ot = getattr(ov, "override_type", None)
                if ot:
                    key = str(ot.value if hasattr(ot, "value") else ot)
                    violation_counts[key] = violation_counts.get(key, 0) + 1
        top_violations = sorted(violation_counts.items(), key=lambda x: -x[1])[:5]

        detr_days = [d.date for d in daily_series if d.avg_score < 50]
        current = daily_series[-1].avg_score if daily_series else 0.0

        return DisciplineReport(
            current_score=current,
            trend_7d=_trend(7), trend_30d=_trend(30),
            avg_score_7d=_avg_window(7), avg_score_30d=_avg_window(30),
            avg_score_90d=_avg_window(90),
            worst_score=min(all_scores, default=0.0),
            best_score=max(all_scores, default=100.0),
            daily_series=daily_series,
            top_rule_violations=top_violations,
            deterioration_days=detr_days,
        )

    def from_db(self, db, lookback_days: int = 90) -> DisciplineReport:
        from core.models import Trade
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        trades = db.query(Trade).filter(Trade.entry_time >= cutoff).all()
        return self.compute(trades)
