# baseline_modeler.py
# behavioral/baseline_modeler.py
"""
Learns trader's baseline behavioral patterns from historical data.
Builds profiles for optimal behavior, deviation patterns, and emotional triggers.
Used by the Behavior Agent to detect anomalies in real-time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func, and_, case
from sqlalchemy.orm import Session as DBSession

from core.enums import (
    BehavioralEventType,
    EmotionalState,
    MarketRegime,
    OverrideSeverity,
    SessionType,
    TradeStatus,
)
from core.models import BehavioralEvent, OverrideEvent, StrategyAttribution, Trade

logger = logging.getLogger(__name__)


@dataclass
class BehavioralBaseline:
    """Statistical baseline of trader behavior."""
    # Trade frequency
    avg_trades_per_day: float = 0.0
    std_trades_per_day: float = 0.0
    max_trades_per_day: int = 0

    # Timing
    avg_time_between_trades_min: float = 0.0
    min_time_between_trades_min: float = 0.0
    preferred_session: Optional[SessionType] = None
    session_distribution: dict[str, float] = field(default_factory=dict)

    # Discipline
    avg_discipline_score: float = 0.0
    discipline_trend: str = "stable"  # improving, declining, stable
    override_frequency: float = 0.0
    most_common_overrides: list[str] = field(default_factory=list)

    # Risk behavior
    avg_risk_pct: float = 0.0
    max_risk_pct: float = 0.0
    stop_usage_rate: float = 0.0

    # Emotional patterns
    emotional_distribution: dict[str, float] = field(default_factory=dict)
    worst_emotion_for_pnl: Optional[str] = None
    best_emotion_for_pnl: Optional[str] = None

    # Performance context
    avg_consecutive_losses_before_tilt: float = 0.0
    loss_recovery_trades: float = 0.0

    # Positive behavior
    clean_execution_rate: float = 0.0
    patience_events_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_frequency": {
                "avg_per_day": round(self.avg_trades_per_day, 1),
                "std_per_day": round(self.std_trades_per_day, 1),
                "max_per_day": self.max_trades_per_day,
            },
            "timing": {
                "avg_between_trades_min": round(self.avg_time_between_trades_min, 1),
                "min_between_trades_min": round(self.min_time_between_trades_min, 1),
                "preferred_session": self.preferred_session.value if self.preferred_session else None,
                "session_distribution": self.session_distribution,
            },
            "discipline": {
                "avg_score": round(self.avg_discipline_score, 1),
                "trend": self.discipline_trend,
                "override_frequency": round(self.override_frequency, 3),
                "most_common_overrides": self.most_common_overrides,
                "clean_execution_rate": round(self.clean_execution_rate, 3),
            },
            "risk_behavior": {
                "avg_risk_pct": round(self.avg_risk_pct, 2),
                "max_risk_pct": round(self.max_risk_pct, 2),
                "stop_usage_rate": round(self.stop_usage_rate, 3),
            },
            "emotional": {
                "distribution": self.emotional_distribution,
                "worst_for_pnl": self.worst_emotion_for_pnl,
                "best_for_pnl": self.best_emotion_for_pnl,
            },
            "resilience": {
                "avg_losses_before_tilt": round(self.avg_consecutive_losses_before_tilt, 1),
                "avg_recovery_trades": round(self.loss_recovery_trades, 1),
            },
        }


class BaselineModeler:
    """
    Computes and maintains the trader's behavioral baseline.
    Queries historical data to establish normal patterns.
    """

    def __init__(self, lookback_days: int = 90):
        self._lookback_days = lookback_days

    def compute_baseline(self, db: DBSession) -> BehavioralBaseline:
        """Build baseline from historical trades within lookback period."""
        baseline = BehavioralBaseline()
        cutoff = datetime.utcnow() - timedelta(days=self._lookback_days)

        trades = (
            db.query(Trade)
            .filter(
                Trade.entry_time >= cutoff,
                Trade.status == TradeStatus.CLOSED,
            )
            .order_by(Trade.entry_time.asc())
            .all()
        )

        if not trades:
            logger.info("No trades in lookback period — cannot compute baseline")
            return baseline

        # ── Trade frequency ──
        self._compute_frequency(baseline, trades)

        # ── Timing patterns ──
        self._compute_timing(baseline, trades)

        # ── Discipline metrics ──
        self._compute_discipline(baseline, trades, db)

        # ── Risk behavior ──
        self._compute_risk(baseline, trades)

        # ── Emotional patterns ──
        self._compute_emotional(baseline, trades)

        # ── Tilt detection baseline ──
        self._compute_tilt_baseline(baseline, trades, db)

        logger.info("Behavioral baseline computed from %d trades", len(trades))
        return baseline

    def _compute_frequency(
        self, baseline: BehavioralBaseline, trades: list[Trade]
    ) -> None:
        from collections import Counter

        daily_counts: Counter = Counter()
        for trade in trades:
            day = trade.entry_time.date()
            daily_counts[day] += 1

        if daily_counts:
            counts = list(daily_counts.values())
            baseline.avg_trades_per_day = sum(counts) / len(counts)
            baseline.max_trades_per_day = max(counts)

            import statistics
            if len(counts) > 1:
                baseline.std_trades_per_day = statistics.stdev(counts)

    def _compute_timing(
        self, baseline: BehavioralBaseline, trades: list[Trade]
    ) -> None:
        from collections import Counter

        # Time between trades
        gaps = []
        for i in range(1, len(trades)):
            if trades[i].entry_time.date() == trades[i - 1].entry_time.date():
                gap = (trades[i].entry_time - trades[i - 1].entry_time).total_seconds() / 60
                gaps.append(gap)

        if gaps:
            baseline.avg_time_between_trades_min = sum(gaps) / len(gaps)
            baseline.min_time_between_trades_min = min(gaps)

        # Session distribution
        session_counts: Counter = Counter()
        for trade in trades:
            if trade.session_type:
                session_counts[trade.session_type.value] += 1

        total = sum(session_counts.values())
        if total > 0:
            baseline.session_distribution = {
                k: round(v / total, 3) for k, v in session_counts.items()
            }
            baseline.preferred_session = SessionType(session_counts.most_common(1)[0][0])

    def _compute_discipline(
        self,
        baseline: BehavioralBaseline,
        trades: list[Trade],
        db: DBSession,
    ) -> None:
        trade_ids = [t.id for t in trades]

        # Discipline scores from attributions
        attributions = (
            db.query(StrategyAttribution)
            .filter(StrategyAttribution.trade_id.in_(trade_ids))
            .all()
        )

        scores = [
            a.discipline_score for a in attributions
            if a.discipline_score is not None
        ]

        if scores:
            baseline.avg_discipline_score = sum(scores) / len(scores)

            # Trend: compare first half vs second half
            mid = len(scores) // 2
            if mid > 5:
                first_half = sum(scores[:mid]) / mid
                second_half = sum(scores[mid:]) / (len(scores) - mid)
                diff = second_half - first_half
                if diff > 5:
                    baseline.discipline_trend = "improving"
                elif diff < -5:
                    baseline.discipline_trend = "declining"

        # Override frequency
        override_count = (
            db.query(func.count(OverrideEvent.id))
            .filter(OverrideEvent.trade_id.in_(trade_ids))
            .scalar()
        )
        if trades:
            baseline.override_frequency = override_count / len(trades)

        # Most common override types
        common_overrides = (
            db.query(
                OverrideEvent.override_type,
                func.count(OverrideEvent.id).label("cnt"),
            )
            .filter(OverrideEvent.trade_id.in_(trade_ids))
            .group_by(OverrideEvent.override_type)
            .order_by(func.count(OverrideEvent.id).desc())
            .limit(5)
            .all()
        )
        baseline.most_common_overrides = [
            str(row[0].value if hasattr(row[0], 'value') else row[0])
            for row in common_overrides
        ]

        # Clean execution rate
        clean_events = (
            db.query(func.count(BehavioralEvent.id))
            .filter(
                BehavioralEvent.trade_id.in_(trade_ids),
                BehavioralEvent.event_type == BehavioralEventType.CLEAN_EXECUTION,
            )
            .scalar()
        )
        if trades:
            baseline.clean_execution_rate = clean_events / len(trades)

    def _compute_risk(
        self, baseline: BehavioralBaseline, trades: list[Trade]
    ) -> None:
        risk_pcts = [t.risk_pct for t in trades if t.risk_pct is not None]
        if risk_pcts:
            baseline.avg_risk_pct = sum(risk_pcts) / len(risk_pcts)
            baseline.max_risk_pct = max(risk_pcts)

        stops_set = sum(1 for t in trades if t.stop_loss is not None)
        if trades:
            baseline.stop_usage_rate = stops_set / len(trades)

    def _compute_emotional(
        self, baseline: BehavioralBaseline, trades: list[Trade]
    ) -> None:
        from collections import defaultdict

        emotion_pnls: dict[str, list[float]] = defaultdict(list)

        for trade in trades:
            if trade.emotion_pre and trade.pnl is not None:
                emotion_pnls[trade.emotion_pre.value].append(trade.pnl)

        total_emotional = sum(len(v) for v in emotion_pnls.values())
        if total_emotional > 0:
            baseline.emotional_distribution = {
                k: round(len(v) / total_emotional, 3)
                for k, v in emotion_pnls.items()
            }

            avg_by_emotion = {
                k: sum(v) / len(v) for k, v in emotion_pnls.items() if len(v) >= 3
            }

            if avg_by_emotion:
                baseline.best_emotion_for_pnl = max(avg_by_emotion, key=avg_by_emotion.get)
                baseline.worst_emotion_for_pnl = min(avg_by_emotion, key=avg_by_emotion.get)

    def _compute_tilt_baseline(
        self,
        baseline: BehavioralBaseline,
        trades: list[Trade],
        db: DBSession,
    ) -> None:
        """Analyze how many consecutive losses before behavioral breakdown."""
        trade_ids = [t.id for t in trades]

        tilt_events = (
            db.query(BehavioralEvent)
            .filter(
                BehavioralEvent.trade_id.in_(trade_ids),
                BehavioralEvent.event_type.in_([
                    BehavioralEventType.REVENGE_TRADE,
                    BehavioralEventType.RAPID_RE_ENTRY,
                    BehavioralEventType.FOMO_ENTRY,
                ]),
            )
            .all()
        )

        if tilt_events:
            losses_before_tilt = []
            for event in tilt_events:
                consec = (event.trigger_conditions or {}).get("consecutive_losses", 0)
                if consec:
                    losses_before_tilt.append(consec)

            if losses_before_tilt:
                baseline.avg_consecutive_losses_before_tilt = (
                    sum(losses_before_tilt) / len(losses_before_tilt)
                )