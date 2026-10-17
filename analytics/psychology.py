"""
Psychology Analytics
=====================
Emotional state analysis for traders.
Correlates pre/post emotions with trade outcomes and discipline.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class EmotionPerformance:
    emotion: str
    trade_count: int
    win_rate: float
    avg_pnl: float
    avg_r: float
    avg_discipline: float
    avg_risk_pct: float


@dataclass
class PsychologyReport:
    total_trades: int = 0
    # Emotion → performance mapping
    by_pre_emotion: list = field(default_factory=list)   # list[EmotionPerformance]
    by_post_emotion: dict = field(default_factory=dict)  # emotion → {win,loss count}

    # Conviction correlations
    conviction_win_correlation: float = 0.0   # +1 = high conviction always wins
    avg_conviction_winners: float = 0.0
    avg_conviction_losers: float = 0.0

    # Best / worst emotional states
    best_entry_emotion: Optional[str] = None
    worst_entry_emotion: Optional[str] = None

    # Emotional regularity
    emotional_consistency_score: float = 0.0   # 0-100; 100 = same emotion across sessions
    most_common_pre_emotion: Optional[str] = None
    most_common_post_emotion: Optional[str] = None

    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "by_pre_emotion": [
                {"emotion": e.emotion, "trade_count": e.trade_count,
                 "win_rate": round(e.win_rate, 4), "avg_pnl": round(e.avg_pnl, 2),
                 "avg_r": round(e.avg_r, 3), "avg_discipline": round(e.avg_discipline, 2),
                 "avg_risk_pct": round(e.avg_risk_pct, 2)}
                for e in self.by_pre_emotion
            ],
            "by_post_emotion": self.by_post_emotion,
            "conviction": {
                "win_correlation": round(self.conviction_win_correlation, 3),
                "avg_winners": round(self.avg_conviction_winners, 2),
                "avg_losers": round(self.avg_conviction_losers, 2),
            },
            "best_entry_emotion": self.best_entry_emotion,
            "worst_entry_emotion": self.worst_entry_emotion,
            "emotional_consistency_score": round(self.emotional_consistency_score, 1),
            "most_common_pre_emotion": self.most_common_pre_emotion,
            "most_common_post_emotion": self.most_common_post_emotion,
            "computed_at": self.computed_at,
        }


class PsychologyAnalytics:

    def compute(self, trades: list) -> PsychologyReport:
        if not trades:
            return PsychologyReport()

        closed = [t for t in trades if getattr(t, "pnl", None) is not None]

        # Group by pre-emotion
        by_pre: dict[str, list] = {}
        for t in closed:
            em = getattr(t, "emotion_pre", None)
            if em:
                key = str(em.value if hasattr(em, "value") else em)
                by_pre.setdefault(key, []).append(t)

        ep_list: list[EmotionPerformance] = []
        for em, ts in by_pre.items():
            pnls = [t.pnl for t in ts]
            rs = [t.r_multiple for t in ts if getattr(t, "r_multiple", None) is not None]
            discs = [getattr(t, "discipline_score", None) for t in ts
                     if getattr(t, "discipline_score", None) is not None]
            rps = [t.risk_pct for t in ts if getattr(t, "risk_pct", None)]
            wins = [p for p in pnls if p > 0]
            ep_list.append(EmotionPerformance(
                emotion=em, trade_count=len(ts),
                win_rate=len(wins)/len(ts) if ts else 0.0,
                avg_pnl=statistics.mean(pnls) if pnls else 0.0,
                avg_r=statistics.mean(rs) if rs else 0.0,
                avg_discipline=statistics.mean(discs) if discs else 0.0,
                avg_risk_pct=statistics.mean(rps) if rps else 0.0,
            ))

        # Best/worst entry emotion (by avg pnl, min 3 trades)
        valid = [e for e in ep_list if e.trade_count >= 3]
        best_em = max(valid, key=lambda e: e.avg_pnl).emotion if valid else None
        worst_em = min(valid, key=lambda e: e.avg_pnl).emotion if valid else None

        # Post-emotion distribution
        by_post: dict[str, dict] = {}
        for t in closed:
            em = getattr(t, "emotion_post", None)
            if em:
                key = str(em.value if hasattr(em, "value") else em)
                rec = by_post.setdefault(key, {"wins": 0, "losses": 0, "total": 0})
                rec["total"] += 1
                if t.pnl > 0: rec["wins"] += 1
                else: rec["losses"] += 1

        # Conviction correlation
        conv_wins = [getattr(t, "conviction_score", None) for t in closed
                     if t.pnl > 0 and getattr(t, "conviction_score", None) is not None]
        conv_loss = [getattr(t, "conviction_score", None) for t in closed
                     if t.pnl <= 0 and getattr(t, "conviction_score", None) is not None]
        avg_cw = statistics.mean(conv_wins) if conv_wins else 0.0
        avg_cl = statistics.mean(conv_loss) if conv_loss else 0.0
        corr = (avg_cw - avg_cl) / 10.0  # normalize 0-10 scale → -1 to 1

        # Most common emotions
        pre_counts = {e: len(ts) for e, ts in by_pre.items()}
        post_counts = {e: d["total"] for e, d in by_post.items()}
        most_pre = max(pre_counts, key=pre_counts.get) if pre_counts else None
        most_post = max(post_counts, key=post_counts.get) if post_counts else None

        # Consistency score: Shannon entropy (inverted)
        total = sum(pre_counts.values())
        if total and pre_counts:
            import math
            ent = sum(-(c/total)*math.log2(c/total) for c in pre_counts.values() if c > 0)
            max_ent = math.log2(len(pre_counts)) if len(pre_counts) > 1 else 1.0
            consistency = (1.0 - ent / max_ent) * 100 if max_ent > 0 else 100.0
        else:
            consistency = 100.0

        return PsychologyReport(
            total_trades=len(closed),
            by_pre_emotion=sorted(ep_list, key=lambda e: -e.trade_count),
            by_post_emotion=by_post,
            conviction_win_correlation=min(1.0, max(-1.0, corr)),
            avg_conviction_winners=avg_cw,
            avg_conviction_losers=avg_cl,
            best_entry_emotion=best_em,
            worst_entry_emotion=worst_em,
            emotional_consistency_score=consistency,
            most_common_pre_emotion=most_pre,
            most_common_post_emotion=most_post,
        )

    def from_db(self, db, lookback_days: int = 90) -> PsychologyReport:
        from core.models import Trade
        from core.enums import TradeStatus
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        trades = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED,
                                        Trade.entry_time >= cutoff).all()
        return self.compute(trades)
