"""
Behavior Analytics
===================
Aggregates behavioral patterns from trades and behavioral events.
Surfaces bias frequencies, override patterns, and discipline trends.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class BiasRecord:
    bias_type: str
    frequency: int
    avg_pnl_impact: float
    pct_of_trades: float
    trending: str   # "increasing" | "stable" | "decreasing"


@dataclass
class BehaviorReport:
    total_trades_analysed: int = 0
    override_rate: float = 0.0          # % trades with at least 1 override
    avg_overrides_per_trade: float = 0.0
    top_biases: list = field(default_factory=list)
    discipline_trend: str = "stable"    # "improving" | "stable" | "deteriorating"
    avg_discipline_score: float = 0.0
    bias_pnl_impact: dict = field(default_factory=dict)  # {bias: total_pnl_lost}
    override_by_emotion: dict = field(default_factory=dict)
    worst_time_of_day: Optional[str] = None   # "morning" | "afternoon" | "late"
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "total_trades_analysed": self.total_trades_analysed,
            "override_rate": round(self.override_rate, 4),
            "avg_overrides_per_trade": round(self.avg_overrides_per_trade, 2),
            "top_biases": [
                {"bias_type": b.bias_type, "frequency": b.frequency,
                 "avg_pnl_impact": round(b.avg_pnl_impact, 2),
                 "pct_of_trades": round(b.pct_of_trades, 4),
                 "trending": b.trending}
                for b in self.top_biases
            ],
            "discipline_trend": self.discipline_trend,
            "avg_discipline_score": round(self.avg_discipline_score, 2),
            "bias_pnl_impact": {k: round(v, 2) for k, v in self.bias_pnl_impact.items()},
            "override_by_emotion": self.override_by_emotion,
            "worst_time_of_day": self.worst_time_of_day,
            "computed_at": self.computed_at,
        }


class BehaviorAnalytics:

    def compute(self, trades: list, behavioral_events: list = None,
                lookback_days: int = 90) -> BehaviorReport:
        if not trades:
            return BehaviorReport()

        closed = [t for t in trades if getattr(t, "pnl", None) is not None]
        bevents = behavioral_events or []

        # Override analysis
        override_counts = [len(getattr(t, "overrides", [])) for t in trades]
        has_override = [c > 0 for c in override_counts]
        override_rate = sum(has_override) / len(trades) if trades else 0.0
        avg_overrides = statistics.mean(override_counts) if override_counts else 0.0

        # Bias frequency from behavioral events
        bias_trades: dict[str, list] = {}
        for ev in bevents:
            bt = getattr(ev, "event_type", None)
            if bt:
                key = str(bt.value if hasattr(bt, "value") else bt)
                trade = getattr(ev, "trade", None)
                if trade and getattr(trade, "pnl", None) is not None:
                    bias_trades.setdefault(key, []).append(trade.pnl)

        top_biases = []
        for bias, pnls in sorted(bias_trades.items(), key=lambda x: -len(x[1]))[:8]:
            top_biases.append(BiasRecord(
                bias_type=bias,
                frequency=len(pnls),
                avg_pnl_impact=statistics.mean(pnls) if pnls else 0.0,
                pct_of_trades=len(pnls) / len(trades) if trades else 0.0,
                trending="stable",
            ))

        bias_pnl_impact = {b: sum(pnls) for b, pnls in bias_trades.items()}

        # Discipline trend
        disc_scores = [getattr(t, "discipline_score", None) for t in trades
                       if getattr(t, "discipline_score", None) is not None]
        avg_disc = statistics.mean(disc_scores) if disc_scores else 0.0
        disc_trend = "stable"
        if len(disc_scores) >= 10:
            first_half = statistics.mean(disc_scores[:len(disc_scores)//2])
            second_half = statistics.mean(disc_scores[len(disc_scores)//2:])
            if second_half > first_half + 5:
                disc_trend = "improving"
            elif second_half < first_half - 5:
                disc_trend = "deteriorating"

        # Override by emotion
        override_by_emotion: dict[str, int] = {}
        for t in trades:
            if getattr(t, "overrides", None) and getattr(t, "emotion_pre", None):
                em = str(t.emotion_pre.value if hasattr(t.emotion_pre, "value") else t.emotion_pre)
                override_by_emotion[em] = override_by_emotion.get(em, 0) + len(t.overrides)

        # Worst time of day (by avg pnl)
        time_pnl: dict[str, list] = {"morning": [], "afternoon": [], "late": []}
        for t in closed:
            et = getattr(t, "entry_time", None)
            if et:
                h = et.hour
                bucket = "morning" if h < 11 else "afternoon" if h < 14 else "late"
                time_pnl[bucket].append(t.pnl)
        worst_time = None
        worst_avg = float("inf")
        for bucket, pnls in time_pnl.items():
            if pnls:
                avg = statistics.mean(pnls)
                if avg < worst_avg:
                    worst_avg, worst_time = avg, bucket

        return BehaviorReport(
            total_trades_analysed=len(trades),
            override_rate=override_rate,
            avg_overrides_per_trade=avg_overrides,
            top_biases=top_biases,
            discipline_trend=disc_trend,
            avg_discipline_score=avg_disc,
            bias_pnl_impact=bias_pnl_impact,
            override_by_emotion=override_by_emotion,
            worst_time_of_day=worst_time,
        )

    def from_db(self, db, lookback_days: int = 90) -> BehaviorReport:
        from core.models import Trade, BehavioralEvent
        from core.enums import TradeStatus
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        trades = db.query(Trade).filter(Trade.entry_time >= cutoff).all()
        bevents = db.query(BehavioralEvent).filter(BehavioralEvent.created_at >= cutoff).all()
        return self.compute(trades, bevents, lookback_days)
