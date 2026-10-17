"""
Daily Summary
==============
Aggregates per-day performance: P&L, trade count, win rate, discipline, emotions.
Used to power the daily dashboard card and day-over-day trend charts.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional


@dataclass
class DayRecord:
    date: str
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    gross_pnl: float
    net_pnl: float          # gross minus commission (if available)
    avg_pnl: float
    best_trade: float
    worst_trade: float
    avg_r: float
    avg_discipline: float
    override_count: int
    dominant_emotion: Optional[str]


@dataclass
class DailySummaryReport:
    days: list = field(default_factory=list)         # list[DayRecord]
    total_trading_days: int = 0
    profitable_days: int = 0
    losing_days: int = 0
    breakeven_days: int = 0
    avg_daily_pnl: float = 0.0
    best_day: Optional[str] = None
    worst_day: Optional[str] = None
    best_day_pnl: float = 0.0
    worst_day_pnl: float = 0.0
    current_day_streak: int = 0          # positive = profitable days, negative = losing days
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "days": [
                {"date": d.date, "trade_count": d.trade_count,
                 "win_count": d.win_count, "loss_count": d.loss_count,
                 "win_rate": round(d.win_rate, 4),
                 "gross_pnl": round(d.gross_pnl, 2), "net_pnl": round(d.net_pnl, 2),
                 "avg_pnl": round(d.avg_pnl, 2),
                 "best_trade": round(d.best_trade, 2), "worst_trade": round(d.worst_trade, 2),
                 "avg_r": round(d.avg_r, 3), "avg_discipline": round(d.avg_discipline, 2),
                 "override_count": d.override_count, "dominant_emotion": d.dominant_emotion}
                for d in self.days
            ],
            "summary": {
                "total_trading_days": self.total_trading_days,
                "profitable_days": self.profitable_days,
                "losing_days": self.losing_days,
                "breakeven_days": self.breakeven_days,
                "avg_daily_pnl": round(self.avg_daily_pnl, 2),
                "best_day": self.best_day,
                "worst_day": self.worst_day,
                "best_day_pnl": round(self.best_day_pnl, 2),
                "worst_day_pnl": round(self.worst_day_pnl, 2),
                "current_day_streak": self.current_day_streak,
            },
            "computed_at": self.computed_at,
        }


class DailySummaryBuilder:

    def compute(self, trades: list) -> DailySummaryReport:
        closed = [t for t in trades if getattr(t, "pnl", None) is not None]
        if not closed:
            return DailySummaryReport()

        by_date: dict[str, list] = {}
        for t in closed:
            et = getattr(t, "entry_time", None) or getattr(t, "exit_time", None)
            if et:
                dk = et.date().isoformat() if isinstance(et, datetime) else str(et)
                by_date.setdefault(dk, []).append(t)

        day_records: list[DayRecord] = []
        for dk, ts in sorted(by_date.items()):
            pnls = [t.pnl for t in ts]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            rs = [t.r_multiple for t in ts if getattr(t, "r_multiple", None) is not None]
            discs = [t.discipline_score for t in ts
                     if getattr(t, "discipline_score", None) is not None]
            ovrs = sum(len(getattr(t, "overrides", [])) for t in ts)
            # Commission if present
            comms = [getattr(t, "commission", 0.0) or 0.0 for t in ts]
            gross = sum(pnls)
            net = gross - sum(comms)

            # Dominant pre-emotion
            em_counts: dict[str, int] = {}
            for t in ts:
                em = getattr(t, "emotion_pre", None)
                if em:
                    key = str(em.value if hasattr(em, "value") else em)
                    em_counts[key] = em_counts.get(key, 0) + 1
            dom_em = max(em_counts, key=em_counts.get) if em_counts else None

            day_records.append(DayRecord(
                date=dk, trade_count=len(ts),
                win_count=len(wins), loss_count=len(losses),
                win_rate=len(wins) / len(ts),
                gross_pnl=gross, net_pnl=net, avg_pnl=statistics.mean(pnls),
                best_trade=max(pnls), worst_trade=min(pnls),
                avg_r=statistics.mean(rs) if rs else 0.0,
                avg_discipline=statistics.mean(discs) if discs else 0.0,
                override_count=ovrs, dominant_emotion=dom_em,
            ))

        daily_pnls = [d.gross_pnl for d in day_records]
        best_rec = max(day_records, key=lambda d: d.gross_pnl)
        worst_rec = min(day_records, key=lambda d: d.gross_pnl)

        # Current day streak
        streak = 0
        for d in reversed(day_records):
            if d.gross_pnl > 0:
                if streak >= 0:
                    streak += 1
                else:
                    break
            elif d.gross_pnl < 0:
                if streak <= 0:
                    streak -= 1
                else:
                    break
            else:
                break

        return DailySummaryReport(
            days=day_records,
            total_trading_days=len(day_records),
            profitable_days=sum(1 for d in day_records if d.gross_pnl > 0),
            losing_days=sum(1 for d in day_records if d.gross_pnl < 0),
            breakeven_days=sum(1 for d in day_records if d.gross_pnl == 0),
            avg_daily_pnl=statistics.mean(daily_pnls) if daily_pnls else 0.0,
            best_day=best_rec.date,
            worst_day=worst_rec.date,
            best_day_pnl=best_rec.gross_pnl,
            worst_day_pnl=worst_rec.gross_pnl,
            current_day_streak=streak,
        )

    def from_db(self, db, lookback_days: int = 90) -> DailySummaryReport:
        from core.models import Trade
        from core.enums import TradeStatus
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        trades = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED,
                                        Trade.entry_time >= cutoff).all()
        return self.compute(trades)
