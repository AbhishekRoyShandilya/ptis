"""
Session Analytics
==================
Breaks performance down by trading session (morning, afternoon, late, pre-market).
Reveals whether a trader performs better early or late in the day.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class SessionPerformance:
    session: str            # "pre_market" | "morning" | "afternoon" | "late"
    trade_count: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    avg_r: float
    avg_discipline: float
    avg_duration_min: float


@dataclass
class SessionReport:
    total_trades: int = 0
    by_session: list = field(default_factory=list)      # list[SessionPerformance]
    best_session: Optional[str] = None
    worst_session: Optional[str] = None
    hourly_win_rate: dict = field(default_factory=dict)  # {"09": 0.6, "10": 0.55, …}
    hourly_avg_pnl: dict = field(default_factory=dict)
    peak_hour: Optional[str] = None
    worst_hour: Optional[str] = None
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "by_session": [
                {"session": s.session, "trade_count": s.trade_count,
                 "win_rate": round(s.win_rate, 4), "avg_pnl": round(s.avg_pnl, 2),
                 "total_pnl": round(s.total_pnl, 2), "avg_r": round(s.avg_r, 3),
                 "avg_discipline": round(s.avg_discipline, 2),
                 "avg_duration_min": round(s.avg_duration_min, 1)}
                for s in self.by_session
            ],
            "best_session": self.best_session,
            "worst_session": self.worst_session,
            "hourly_win_rate": {h: round(v, 4) for h, v in self.hourly_win_rate.items()},
            "hourly_avg_pnl": {h: round(v, 2) for h, v in self.hourly_avg_pnl.items()},
            "peak_hour": self.peak_hour,
            "worst_hour": self.worst_hour,
            "computed_at": self.computed_at,
        }


def _bucket(hour: int) -> str:
    if hour < 9:
        return "pre_market"
    if hour < 11:
        return "morning"
    if hour < 14:
        return "afternoon"
    return "late"


class SessionAnalytics:

    def compute(self, trades: list) -> SessionReport:
        closed = [t for t in trades if getattr(t, "pnl", None) is not None
                  and getattr(t, "entry_time", None) is not None]
        if not closed:
            return SessionReport()

        by_sess: dict[str, list] = {}
        hourly: dict[str, list] = {}

        for t in closed:
            h = t.entry_time.hour
            sess = _bucket(h)
            by_sess.setdefault(sess, []).append(t)
            hk = f"{h:02d}"
            hourly.setdefault(hk, []).append(t)

        sp_list: list[SessionPerformance] = []
        for sess, ts in by_sess.items():
            pnls = [t.pnl for t in ts]
            wins = [p for p in pnls if p > 0]
            rs = [t.r_multiple for t in ts if getattr(t, "r_multiple", None) is not None]
            discs = [t.discipline_score for t in ts
                     if getattr(t, "discipline_score", None) is not None]
            durations = []
            for t in ts:
                if getattr(t, "exit_time", None):
                    dur = (t.exit_time - t.entry_time).total_seconds() / 60
                    durations.append(dur)
            sp_list.append(SessionPerformance(
                session=sess,
                trade_count=len(ts),
                win_rate=len(wins) / len(ts),
                avg_pnl=statistics.mean(pnls),
                total_pnl=sum(pnls),
                avg_r=statistics.mean(rs) if rs else 0.0,
                avg_discipline=statistics.mean(discs) if discs else 0.0,
                avg_duration_min=statistics.mean(durations) if durations else 0.0,
            ))

        valid = [s for s in sp_list if s.trade_count >= 3]
        best = max(valid, key=lambda s: s.avg_pnl).session if valid else None
        worst = min(valid, key=lambda s: s.avg_pnl).session if valid else None

        hourly_wr = {}
        hourly_pnl = {}
        for hk, ts in hourly.items():
            pnls = [t.pnl for t in ts]
            hourly_wr[hk] = len([p for p in pnls if p > 0]) / len(pnls)
            hourly_pnl[hk] = statistics.mean(pnls)

        peak_h = max(hourly_pnl, key=hourly_pnl.get) if hourly_pnl else None
        worst_h = min(hourly_pnl, key=hourly_pnl.get) if hourly_pnl else None

        return SessionReport(
            total_trades=len(closed),
            by_session=sorted(sp_list, key=lambda s: -s.trade_count),
            best_session=best,
            worst_session=worst,
            hourly_win_rate=dict(sorted(hourly_wr.items())),
            hourly_avg_pnl=dict(sorted(hourly_pnl.items())),
            peak_hour=peak_h,
            worst_hour=worst_h,
        )

    def from_db(self, db, lookback_days: int = 90) -> SessionReport:
        from core.models import Trade
        from core.enums import TradeStatus
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        trades = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED,
                                        Trade.entry_time >= cutoff).all()
        return self.compute(trades)
