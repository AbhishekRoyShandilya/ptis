"""
Weekly Report
==============
Rolls up daily data into ISO week buckets.
Surfaces week-over-week P&L trends, best/worst weeks, and consistency metrics.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Optional


@dataclass
class WeekRecord:
    iso_year: int
    iso_week: int
    week_label: str          # e.g. "2024-W03"
    start_date: str
    end_date: str
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    gross_pnl: float
    avg_daily_pnl: float
    avg_r: float
    avg_discipline: float
    override_count: int
    trading_days: int


@dataclass
class WeeklyReport:
    weeks: list = field(default_factory=list)        # list[WeekRecord]
    total_weeks: int = 0
    profitable_weeks: int = 0
    losing_weeks: int = 0
    avg_weekly_pnl: float = 0.0
    best_week: Optional[str] = None
    worst_week: Optional[str] = None
    best_week_pnl: float = 0.0
    worst_week_pnl: float = 0.0
    weekly_pnl_std: float = 0.0       # consistency measure
    week_over_week_trend: str = "stable"   # "improving" | "stable" | "deteriorating"
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "weeks": [
                {"week": w.week_label, "start": w.start_date, "end": w.end_date,
                 "trade_count": w.trade_count, "win_rate": round(w.win_rate, 4),
                 "gross_pnl": round(w.gross_pnl, 2),
                 "avg_daily_pnl": round(w.avg_daily_pnl, 2),
                 "avg_r": round(w.avg_r, 3), "avg_discipline": round(w.avg_discipline, 2),
                 "override_count": w.override_count, "trading_days": w.trading_days}
                for w in self.weeks
            ],
            "summary": {
                "total_weeks": self.total_weeks,
                "profitable_weeks": self.profitable_weeks,
                "losing_weeks": self.losing_weeks,
                "avg_weekly_pnl": round(self.avg_weekly_pnl, 2),
                "best_week": self.best_week,
                "worst_week": self.worst_week,
                "best_week_pnl": round(self.best_week_pnl, 2),
                "worst_week_pnl": round(self.worst_week_pnl, 2),
                "weekly_pnl_std": round(self.weekly_pnl_std, 2),
                "wow_trend": self.week_over_week_trend,
            },
            "computed_at": self.computed_at,
        }


def _week_label(dt: date) -> tuple[int, int, str, str, str]:
    iso = dt.isocalendar()
    y, w = iso[0], iso[1]
    # Week start (Monday) and end (Friday)
    monday = dt - timedelta(days=dt.weekday())
    friday = monday + timedelta(days=4)
    label = f"{y}-W{w:02d}"
    return y, w, label, monday.isoformat(), friday.isoformat()


class WeeklyReportBuilder:

    def compute(self, trades: list) -> WeeklyReport:
        closed = [t for t in trades if getattr(t, "pnl", None) is not None]
        if not closed:
            return WeeklyReport()

        by_week: dict[tuple, list] = {}
        for t in closed:
            et = getattr(t, "entry_time", None) or getattr(t, "exit_time", None)
            if et:
                d = et.date() if isinstance(et, datetime) else et
                y, w, label, start, end = _week_label(d)
                key = (y, w, label, start, end)
                by_week.setdefault(key, []).append(t)

        week_records: list[WeekRecord] = []
        for (y, w, label, start, end), ts in sorted(by_week.items()):
            pnls = [t.pnl for t in ts]
            wins = [p for p in pnls if p > 0]
            rs = [t.r_multiple for t in ts if getattr(t, "r_multiple", None) is not None]
            discs = [t.discipline_score for t in ts
                     if getattr(t, "discipline_score", None) is not None]
            ovrs = sum(len(getattr(t, "overrides", [])) for t in ts)
            # Count distinct trading days in this week
            days_set = set()
            for t in ts:
                et = getattr(t, "entry_time", None) or getattr(t, "exit_time", None)
                if et:
                    d = et.date() if isinstance(et, datetime) else et
                    days_set.add(d)
            trading_days = len(days_set)
            gross = sum(pnls)
            week_records.append(WeekRecord(
                iso_year=y, iso_week=w, week_label=label,
                start_date=start, end_date=end,
                trade_count=len(ts), win_count=len(wins),
                loss_count=len([p for p in pnls if p < 0]),
                win_rate=len(wins) / len(ts),
                gross_pnl=gross,
                avg_daily_pnl=gross / trading_days if trading_days else 0.0,
                avg_r=statistics.mean(rs) if rs else 0.0,
                avg_discipline=statistics.mean(discs) if discs else 0.0,
                override_count=ovrs,
                trading_days=trading_days,
            ))

        weekly_pnls = [w.gross_pnl for w in week_records]
        best_w = max(week_records, key=lambda w: w.gross_pnl)
        worst_w = min(week_records, key=lambda w: w.gross_pnl)

        # Week-over-week trend: compare first half vs second half
        wow_trend = "stable"
        if len(week_records) >= 4:
            mid = len(week_records) // 2
            first_avg = statistics.mean(w.gross_pnl for w in week_records[:mid])
            last_avg = statistics.mean(w.gross_pnl for w in week_records[mid:])
            if last_avg > first_avg * 1.1:
                wow_trend = "improving"
            elif last_avg < first_avg * 0.9:
                wow_trend = "deteriorating"

        return WeeklyReport(
            weeks=week_records,
            total_weeks=len(week_records),
            profitable_weeks=sum(1 for w in week_records if w.gross_pnl > 0),
            losing_weeks=sum(1 for w in week_records if w.gross_pnl < 0),
            avg_weekly_pnl=statistics.mean(weekly_pnls) if weekly_pnls else 0.0,
            best_week=best_w.week_label,
            worst_week=worst_w.week_label,
            best_week_pnl=best_w.gross_pnl,
            worst_week_pnl=worst_w.gross_pnl,
            weekly_pnl_std=statistics.stdev(weekly_pnls) if len(weekly_pnls) >= 2 else 0.0,
            week_over_week_trend=wow_trend,
        )

    def from_db(self, db, lookback_days: int = 90) -> WeeklyReport:
        from core.models import Trade
        from core.enums import TradeStatus
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        trades = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED,
                                        Trade.entry_time >= cutoff).all()
        return self.compute(trades)
