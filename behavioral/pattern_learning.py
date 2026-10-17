"""
PTIS Behavioral Pattern Learning Layer
========================================

Detects recurring behavioral patterns from trade history
and behavioral events. Identifies cycles, triggers, and
predictable behavior sequences that the coaching agent
can use for proactive intervention.

Detected Patterns
-----------------
• Discipline breakdown cycles
• Post-loss behavior changes
• Oversizing patterns
• Emotional trigger sequences
• Time-of-day behavior shifts
• Streak-driven behavior changes
• Recovery patterns (positive)

Design
------
• Sliding window pattern detection
• Sequence analysis on behavioral events
• Threshold-based classification
• Fully deterministic
• Incremental-capable
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════

class PatternType(str, Enum):
    DISCIPLINE_BREAKDOWN = "discipline_breakdown"
    POST_LOSS_DETERIORATION = "post_loss_deterioration"
    OVERSIZING_PATTERN = "oversizing_pattern"
    REVENGE_CYCLE = "revenge_cycle"
    TILT_CASCADE = "tilt_cascade"
    OVERTRADING_BURST = "overtrading_burst"
    END_OF_DAY_DETERIORATION = "end_of_day_deterioration"
    WIN_STREAK_OVERCONFIDENCE = "win_streak_overconfidence"
    LOSS_STREAK_PARALYSIS = "loss_streak_paralysis"
    FOMO_CHASING = "fomo_chasing"
    RECOVERY_PATTERN = "recovery_pattern"
    IMPROVEMENT_TREND = "improvement_trend"


class PatternSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    POSITIVE = "positive"


class PatternStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    RECURRING = "recurring"
    NEW = "new"


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PatternLearningConfig:
    """Configuration for pattern detection."""
    # Minimum occurrences to establish a pattern
    min_occurrences: int = 3

    # Discipline breakdown
    discipline_score_threshold: float = 40.0
    discipline_window_trades: int = 10

    # Post-loss
    post_loss_window_trades: int = 5
    post_loss_degradation_threshold: float = 0.3  # 30% worse

    # Oversizing
    oversize_ratio_threshold: float = 1.5
    oversize_frequency_threshold: float = 0.15  # 15% of trades

    # Revenge trading
    revenge_time_window_seconds: int = 300
    revenge_size_increase_threshold: float = 1.3

    # Tilt cascade
    tilt_consecutive_low_quality: int = 3
    tilt_quality_threshold: float = 40.0

    # Overtrading
    overtrading_daily_threshold: int = 8
    overtrading_quality_decay_threshold: float = 0.2

    # Time analysis
    eod_minutes_threshold: int = 60  # last hour
    eod_quality_drop_threshold: float = 15.0  # points

    # Win streak
    win_streak_overconfidence_length: int = 4
    overconfidence_size_increase: float = 1.3

    # Recovery
    recovery_improvement_threshold: float = 10.0  # score points
    recovery_window_trades: int = 20

    # Staleness
    pattern_stale_days: int = 30


# ═══════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TradeSnapshot:
    """Lightweight trade record for pattern detection."""
    trade_id: Optional[UUID] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    r_multiple: Optional[float] = None
    outcome: str = "breakeven"  # win/loss/breakeven
    position_pct: Optional[float] = None
    emotional_state: Optional[str] = None
    override_count: int = 0
    decision_quality_score: Optional[float] = None
    strategy_name: Optional[str] = None
    session_type: Optional[str] = None
    minutes_from_open: Optional[float] = None
    trade_number_today: int = 0
    time_since_last_trade_seconds: Optional[float] = None
    planned_position_size: Optional[float] = None
    actual_position_size: Optional[float] = None


@dataclass
class DetectedPattern:
    """A behavioral pattern detected from trade history."""
    pattern_id: UUID = field(default_factory=uuid4)
    pattern_type: PatternType = PatternType.DISCIPLINE_BREAKDOWN
    severity: PatternSeverity = PatternSeverity.MODERATE
    status: PatternStatus = PatternStatus.NEW
    detected_at: datetime = field(default_factory=datetime.utcnow)

    # Description
    title: str = ""
    description: str = ""
    coaching_note: str = ""

    # Evidence
    occurrence_count: int = 0
    total_pnl_impact: float = 0.0
    avg_quality_impact: float = 0.0
    affected_trade_ids: List[UUID] = field(default_factory=list)

    # Temporal
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    avg_interval_days: Optional[float] = None

    # Triggering conditions
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)

    # Confidence
    confidence: float = 0.0  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": str(self.pattern_id),
            "pattern_type": self.pattern_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "detected_at": self.detected_at.isoformat(),
            "title": self.title,
            "description": self.description,
            "coaching_note": self.coaching_note,
            "occurrence_count": self.occurrence_count,
            "total_pnl_impact": round(self.total_pnl_impact, 2),
            "avg_quality_impact": round(self.avg_quality_impact, 1),
            "confidence": round(self.confidence, 2),
            "trigger_conditions": self.trigger_conditions,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }


@dataclass
class PatternReport:
    """Complete pattern analysis output."""
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    trade_count: int = 0
    patterns: List[DetectedPattern] = field(default_factory=list)
    active_patterns: List[DetectedPattern] = field(default_factory=list)
    resolved_patterns: List[DetectedPattern] = field(default_factory=list)
    positive_patterns: List[DetectedPattern] = field(default_factory=list)

    @property
    def critical_patterns(self) -> List[DetectedPattern]:
        return [
            p for p in self.patterns
            if p.severity in (PatternSeverity.CRITICAL, PatternSeverity.HIGH)
            and p.status in (PatternStatus.ACTIVE, PatternStatus.RECURRING)
        ]

    def coaching_summary(self) -> str:
        """Natural language summary for coaching agent."""
        lines = [
            f"Behavioral Pattern Analysis ({self.trade_count} trades):",
        ]

        if self.critical_patterns:
            lines.append(f"\n⚠️ CRITICAL PATTERNS ({len(self.critical_patterns)}):")
            for p in self.critical_patterns:
                lines.append(f"  - {p.title}: {p.description}")
                lines.append(f"    Impact: ${p.total_pnl_impact:.0f} total PnL")
                lines.append(f"    Coaching: {p.coaching_note}")

        active_non_critical = [
            p for p in self.active_patterns
            if p.severity not in (PatternSeverity.CRITICAL, PatternSeverity.HIGH)
        ]
        if active_non_critical:
            lines.append(f"\n📊 ACTIVE PATTERNS ({len(active_non_critical)}):")
            for p in active_non_critical:
                lines.append(f"  - {p.title}: {p.description}")

        if self.positive_patterns:
            lines.append(f"\n✅ POSITIVE PATTERNS ({len(self.positive_patterns)}):")
            for p in self.positive_patterns:
                lines.append(f"  - {p.title}: {p.description}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyzed_at": self.analyzed_at.isoformat(),
            "trade_count": self.trade_count,
            "total_patterns": len(self.patterns),
            "critical_count": len(self.critical_patterns),
            "patterns": [p.to_dict() for p in self.patterns],
        }


# ═══════════════════════════════════════════════════════════════════
# PATTERN LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════

class PatternLearningEngine:
    """
    Detects recurring behavioral patterns from trade history.

    Usage
    -----
    >>> engine = PatternLearningEngine()
    >>> report = engine.analyze(trades)
    >>> for pattern in report.critical_patterns:
    ...     print(pattern.title, pattern.coaching_note)
    """

    def __init__(
        self, config: Optional[PatternLearningConfig] = None
    ) -> None:
        self.cfg = config or PatternLearningConfig()
        logger.info("PatternLearningEngine initialized.")

    def analyze(self, trades: Sequence[TradeSnapshot]) -> PatternReport:
        """Run all pattern detectors and produce report."""
        report = PatternReport(trade_count=len(trades))

        if len(trades) < self.cfg.min_occurrences:
            return report

        # Run all detectors
        detectors = [
            self._detect_post_loss_deterioration,
            self._detect_revenge_cycles,
            self._detect_tilt_cascades,
            self._detect_oversizing_patterns,
            self._detect_overtrading_bursts,
            self._detect_eod_deterioration,
            self._detect_win_streak_overconfidence,
            self._detect_loss_streak_paralysis,
            self._detect_discipline_breakdown_cycles,
            self._detect_recovery_patterns,
            self._detect_improvement_trends,
        ]

        for detector in detectors:
            try:
                patterns = detector(trades)
                report.patterns.extend(patterns)
            except Exception:
                logger.exception("Pattern detector %s failed.", detector.__name__)

        # Classify patterns
        report.active_patterns = [
            p for p in report.patterns
            if p.status in (PatternStatus.ACTIVE, PatternStatus.RECURRING, PatternStatus.NEW)
            and p.severity != PatternSeverity.POSITIVE
        ]
        report.resolved_patterns = [
            p for p in report.patterns if p.status == PatternStatus.RESOLVED
        ]
        report.positive_patterns = [
            p for p in report.patterns if p.severity == PatternSeverity.POSITIVE
        ]

        # Sort by severity then confidence
        severity_order = {
            PatternSeverity.CRITICAL: 0,
            PatternSeverity.HIGH: 1,
            PatternSeverity.MODERATE: 2,
            PatternSeverity.LOW: 3,
            PatternSeverity.POSITIVE: 4,
        }
        report.patterns.sort(
            key=lambda p: (severity_order.get(p.severity, 9), -p.confidence)
        )

        logger.info(
            "Pattern analysis complete: %d patterns (%d critical, %d positive)",
            len(report.patterns),
            len(report.critical_patterns),
            len(report.positive_patterns),
        )

        return report

    # ── Pattern Detectors ──────────────────────────────────────

    def _detect_post_loss_deterioration(
        self, trades: Sequence[TradeSnapshot]
    ) -> List[DetectedPattern]:
        """Detect degraded performance after losses."""
        patterns: List[DetectedPattern] = []
        occurrences = 0
        pnl_impact = 0.0
        quality_impacts: List[float] = []
        affected_ids: List[UUID] = []
        timestamps: List[datetime] = []

        window = self.cfg.post_loss_window_trades

        for i in range(len(trades)):
            if trades[i].outcome != "loss":
                continue

            # Examine next `window` trades after this loss
            post_loss_trades = trades[i + 1: i + 1 + window]
            if len(post_loss_trades) < 2:
                continue

            # Compare decision quality of post-loss vs baseline
            post_dq = [
                t.decision_quality_score for t in post_loss_trades
                if t.decision_quality_score is not None
            ]

            # Baseline: trades NOT immediately after a loss
            baseline_dq = [
                t.decision_quality_score for j, t in enumerate(trades)
                if t.decision_quality_score is not None
                and (j == 0 or trades[j - 1].outcome != "loss")
            ]

            if post_dq and baseline_dq:
                avg_post = statistics.mean(post_dq)
                avg_baseline = statistics.mean(baseline_dq)

                if avg_baseline > 0:
                    degradation = (avg_baseline - avg_post) / avg_baseline
                    if degradation > self.cfg.post_loss_degradation_threshold:
                        occurrences += 1
                        for t in post_loss_trades:
                            pnl_impact += t.pnl
                            if t.trade_id:
                                affected_ids.append(t.trade_id)
                        quality_impacts.append(avg_baseline - avg_post)
                        if trades[i].entry_time:
                            timestamps.append(trades[i].entry_time)

        if occurrences >= self.cfg.min_occurrences:
            patterns.append(DetectedPattern(
                pattern_type=PatternType.POST_LOSS_DETERIORATION,
                severity=PatternSeverity.HIGH if occurrences >= 5 else PatternSeverity.MODERATE,
                status=PatternStatus.ACTIVE if timestamps and (
                    datetime.utcnow() - timestamps[-1]
                ).days < self.cfg.pattern_stale_days else PatternStatus.RESOLVED,
                title="Post-Loss Performance Deterioration",
                description=(
                    f"Decision quality drops by an average of "
                    f"{statistics.mean(quality_impacts):.0f} points after losses. "
                    f"Detected {occurrences} times."
                ),
                coaching_note=(
                    "After a loss, your next trades tend to be worse quality. "
                    "Implement a mandatory 5-minute reset after every losing trade. "
                    "Re-run your setup checklist before the next entry."
                ),
                occurrence_count=occurrences,
                total_pnl_impact=pnl_impact,
                avg_quality_impact=(
                    statistics.mean(quality_impacts) if quality_impacts else 0.0
                ),
                affected_trade_ids=affected_ids[:20],
                confidence=min(1.0, occurrences / 8),
                first_seen=timestamps[0] if timestamps else None,
                last_seen=timestamps[-1] if timestamps else None,
                trigger_conditions={"trigger": "loss", "window": window},
            ))

        return patterns

    def _detect_revenge_cycles(
        self, trades: Sequence[TradeSnapshot]
    ) -> List[DetectedPattern]:
        """Detect revenge trading after losses."""
        patterns: List[DetectedPattern] = []
        revenge_events: List[Tuple[int, TradeSnapshot]] = []

        for i in range(1, len(trades)):
            prev = trades[i - 1]
            curr = trades[i]

            if prev.outcome != "loss":
                continue

            is_revenge = False
            reasons: List[str] = []

            # Quick re-entry
            if (
                curr.time_since_last_trade_seconds is not None
                and curr.time_since_last_trade_seconds < self.cfg.revenge_time_window_seconds
            ):
                is_revenge = True
                reasons.append(f"re-entry in {curr.time_since_last_trade_seconds:.0f}s")

            # Size increase after loss
            if (
                curr.position_pct is not None
                and prev.position_pct is not None
                and prev.position_pct > 0
            ):
                ratio = curr.position_pct / prev.position_pct
                if ratio > 1.3:
                    is_revenge = True
                    reasons.append(f"size increased {ratio:.1f}x after loss")

            if is_revenge:
                revenge_events.append((i, curr))

        if len(revenge_events) >= self.cfg.min_occurrences:
            affected_ids = [str(t.trade_id) for _, t in revenge_events]
            patterns.append(DetectedPattern(
                pattern_type=PatternType.REVENGE_TRADING,
                severity=PatternSeverity.CRITICAL,
                status=PatternStatus.ACTIVE,
                title="Revenge Trading After Losses",
                description=(
                    f"Detected {len(revenge_events)} revenge-trade events "
                    f"(quick re-entry or size increase after a loss)."
                ),
                occurrences=len(revenge_events),
                impact_score=-0.8,
                coaching_note="Implement a mandatory 5-minute cooldown after any loss.",
                intervention_strategies=["cooldown_timer", "size_lock_after_loss"],
                affected_trade_ids=affected_ids[:20],
                confidence=min(1.0, len(revenge_events) / 6),
            ))

        return patterns


# ═══════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════

# Alias for callers that import by the old name
BehavioralPatternLearner = PatternLearningEngine

_learner: Optional[PatternLearningEngine] = None


def get_pattern_learner() -> PatternLearningEngine:
    """Get or create a singleton PatternLearningEngine."""
    global _learner
    if _learner is None:
        _learner = PatternLearningEngine()
    return _learner