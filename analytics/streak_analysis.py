"""
Streak Analysis
================
Win/loss streak monitoring — a leading indicator of emotional deterioration and edge degradation.
Consecutive losses → revenge trading risk.  Consecutive wins → overconfidence risk.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class StreakReport:
    current_streak_type: str = "none"; current_streak_length: int = 0
    max_win_streak: int = 0; max_loss_streak: int = 0
    win_streaks: list = field(default_factory=list)
    loss_streaks: list = field(default_factory=list)
    avg_pnl_after_win_streak: float = 0.0; avg_pnl_after_loss_streak: float = 0.0
    avg_risk_pct_after_loss: float = 0.0; risk_escalation_detected: bool = False
    streak_distribution: dict = field(default_factory=dict)
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "current": {"type": self.current_streak_type, "length": self.current_streak_length},
            "historical": {"max_win_streak": self.max_win_streak,
                           "max_loss_streak": self.max_loss_streak},
            "streaks": {"wins": self.win_streaks, "losses": self.loss_streaks},
            "performance_after": {
                "avg_pnl_after_win_streak": round(self.avg_pnl_after_win_streak, 2),
                "avg_pnl_after_loss_streak": round(self.avg_pnl_after_loss_streak, 2)},
            "risk_escalation": {"avg_risk_pct_after_loss": round(self.avg_risk_pct_after_loss, 2),
                                "detected": self.risk_escalation_detected},
            "streak_distribution": self.streak_distribution,
            "computed_at": self.computed_at,
        }


class StreakAnalyzer:

    def compute(self, trades: list) -> StreakReport:
        closed = sorted(
            [t for t in trades if getattr(t, "pnl", None) is not None],
            key=lambda t: getattr(t, "exit_time", getattr(t, "entry_time", datetime.min)),
        )
        if not closed:
            return StreakReport()
        wins_s: list[int] = []; loss_s: list[int] = []
        cur_t = None; cur_l = 0; ends: list[tuple] = []
        for i, t in enumerate(closed):
            o = "win" if t.pnl > 0 else "loss" if t.pnl < 0 else "be"
            if o == "be":
                if cur_l:
                    ends.append((cur_t, cur_l, i-1))
                    (wins_s if cur_t == "win" else loss_s).append(cur_l)
                cur_t = None; cur_l = 0; continue
            if o == cur_t: cur_l += 1
            else:
                if cur_l:
                    ends.append((cur_t, cur_l, i-1))
                    (wins_s if cur_t == "win" else loss_s).append(cur_l)
                cur_t = o; cur_l = 1
        if cur_l:
            ends.append((cur_t, cur_l, len(closed)-1))
            (wins_s if cur_t == "win" else loss_s).append(cur_l)
        aw: list[float] = []; al: list[float] = []; alr: list[float] = []
        for st, _, ei in ends:
            if ei+1 >= len(closed): continue
            nt = closed[ei+1]
            if st == "win": aw.append(nt.pnl)
            elif st == "loss":
                al.append(nt.pnl)
                rp = getattr(nt, "risk_pct", None)
                if rp: alr.append(rp)
        all_r = [getattr(t, "risk_pct", None) for t in closed if getattr(t, "risk_pct", None)]
        avg_r = statistics.mean(all_r) if all_r else 1.0
        avg_al = statistics.mean(alr) if alr else 0.0
        dist: dict[str, int] = {}
        for l in wins_s + loss_s: dist[str(l)] = dist.get(str(l), 0) + 1
        return StreakReport(
            current_streak_type=cur_t or "none", current_streak_length=cur_l,
            max_win_streak=max(wins_s, default=0), max_loss_streak=max(loss_s, default=0),
            win_streaks=wins_s, loss_streaks=loss_s,
            avg_pnl_after_win_streak=statistics.mean(aw) if aw else 0.0,
            avg_pnl_after_loss_streak=statistics.mean(al) if al else 0.0,
            avg_risk_pct_after_loss=avg_al,
            risk_escalation_detected=bool(alr and avg_al > avg_r * 1.25),
            streak_distribution=dist,
        )

    def from_db(self, db, lookback_days: int = 90) -> StreakReport:
        from core.models import Trade
        from core.enums import TradeStatus
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        return self.compute(db.query(Trade).filter(Trade.status == TradeStatus.CLOSED,
                                                   Trade.entry_time >= cutoff)
                            .order_by(Trade.exit_time).all())
