# avt/calibration_tracker.py
"""
Calibration Tracker — Signal vs User Trade Alignment
======================================================

Before AVT trades autonomously, this module compares AVT's
mechanical signals against the user's actual trades.

Measures:
  - Signal Match Rate (true positives)
  - False Positive Rate (AVT sees setups user wouldn't take)
  - False Negative Rate (user trades AVT missed)
  - Alignment Score = TP / (TP + FP + FN)

When alignment >= threshold, strategy graduates to autonomous.
User can annotate mismatches to refine the strategy translation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_MIN_COMPARISONS = 20
DEFAULT_GRADUATION_THRESHOLD = 0.70
DEFAULT_MATCH_WINDOW_MINUTES = 15  # ~2-3 candles on 5min
DEFAULT_MAX_CALIBRATION_DAYS = 30


@dataclass
class CalibrationConfig:
    min_comparisons: int = DEFAULT_MIN_COMPARISONS
    graduation_threshold: float = DEFAULT_GRADUATION_THRESHOLD
    match_window_minutes: int = DEFAULT_MATCH_WINDOW_MINUTES
    max_calibration_days: int = DEFAULT_MAX_CALIBRATION_DAYS


@dataclass
class CalibrationSignal:
    """An AVT signal recorded during calibration."""
    id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    direction: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    conviction: float = 0.0
    signal_details: Dict[str, Any] = field(default_factory=dict)
    # Matching
    matched_trade_id: Optional[str] = None
    match_type: Optional[str] = None  # "true_positive" | "false_positive"
    user_annotation: Optional[str] = None


@dataclass
class CalibrationUserTrade:
    """A user trade recorded during calibration."""
    id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    direction: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    trade_details: Dict[str, Any] = field(default_factory=dict)
    # Matching
    matched_signal_id: Optional[str] = None
    match_type: Optional[str] = None  # "true_positive" | "false_negative"


@dataclass
class CalibrationMismatch:
    """A mismatch between AVT signal and user behavior."""
    id: str = ""
    type: str = ""  # "false_positive" | "false_negative"
    symbol: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    avt_signal: Optional[Dict[str, Any]] = None
    user_trade: Optional[Dict[str, Any]] = None
    user_annotation: Optional[str] = None


@dataclass
class CalibrationReport:
    """Calibration period summary."""
    strategy_id: str = ""
    signals_generated: int = 0
    user_trades: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    alignment_score: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    calibration_days: int = 0
    ready_to_graduate: bool = False
    graduation_blockers: List[str] = field(default_factory=list)
    total_comparisons: int = 0
    top_mismatch_reasons: List[str] = field(default_factory=list)


class SignalCalibrationTracker:
    """
    Compares AVT signals vs user trades during calibration.

    Signal-trade matching: same symbol + same direction +
    within time window (default ±15 min).
    """

    def __init__(
        self,
        strategy_id: str,
        config: Optional[CalibrationConfig] = None,
    ):
        self._strategy_id = strategy_id
        self._config = config or CalibrationConfig()
        self._started_at = datetime.now(timezone.utc)

        self._signals: List[CalibrationSignal] = []
        self._user_trades: List[CalibrationUserTrade] = []
        self._mismatches: List[CalibrationMismatch] = []

        # Tracking
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0

        # Persistence callbacks — set by integration layer (optional)
        # Signature: callback(signal: CalibrationSignal) or callback(trade: CalibrationUserTrade)
        self._persist_signal_callback: Optional[Any] = None
        self._persist_trade_callback: Optional[Any] = None

    def record_avt_signal(
        self,
        symbol: str,
        direction: str,
        conviction: float,
        timestamp: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record an AVT signal. Returns signal ID."""
        signal = CalibrationSignal(
            symbol=symbol,
            direction=direction,
            timestamp=timestamp or datetime.now(timezone.utc),
            conviction=conviction,
            signal_details=details or {},
        )
        self._signals.append(signal)

        # Try to match against unmatched user trades
        matched = self._try_match_signal(signal)
        if not matched:
            signal.match_type = "false_positive"
            self._false_positives += 1
            self._mismatches.append(CalibrationMismatch(
                id=signal.id,
                type="false_positive",
                symbol=symbol,
                timestamp=signal.timestamp,
                avt_signal={
                    "conviction": conviction,
                    "direction": direction,
                    **signal.signal_details,
                },
            ))

        # Persist to DB if callback wired
        if self._persist_signal_callback:
            try:
                self._persist_signal_callback(signal)
            except Exception as _e:
                logger.warning("Calibration signal persist failed: %s", _e)

        return signal.id

    def record_user_trade(
        self,
        symbol: str,
        direction: str,
        timestamp: Optional[datetime] = None,
        trade_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a user trade. Returns trade ID."""
        trade = CalibrationUserTrade(
            id=trade_id or str(uuid4()),
            symbol=symbol,
            direction=direction,
            timestamp=timestamp or datetime.now(timezone.utc),
            trade_details=details or {},
        )
        self._user_trades.append(trade)

        # Try to match against unmatched signals
        matched = self._try_match_trade(trade)
        if not matched:
            trade.match_type = "false_negative"
            self._false_negatives += 1
            self._mismatches.append(CalibrationMismatch(
                id=trade.id,
                type="false_negative",
                symbol=symbol,
                timestamp=trade.timestamp,
                user_trade={
                    "direction": direction,
                    **trade.trade_details,
                },
            ))

        # Persist to DB if callback wired
        if self._persist_trade_callback:
            try:
                self._persist_trade_callback(trade)
            except Exception as _e:
                logger.warning("Calibration user trade persist failed: %s", _e)

        return trade.id

    def annotate_mismatch(
        self,
        mismatch_id: str,
        reason: str,
    ) -> bool:
        """User annotates why a mismatch occurred."""
        for m in self._mismatches:
            if m.id == mismatch_id:
                m.user_annotation = reason
                # Also update the signal/trade
                for s in self._signals:
                    if s.id == mismatch_id:
                        s.user_annotation = reason
                return True
        return False

    def should_graduate(self) -> Tuple[bool, str]:
        """Check if calibration can graduate to autonomous."""
        report = self.compute_alignment()

        if report.total_comparisons < self._config.min_comparisons:
            return False, (
                f"Need {self._config.min_comparisons} comparisons, "
                f"have {report.total_comparisons}"
            )

        elapsed = (datetime.now(timezone.utc) - self._started_at).days
        if elapsed > self._config.max_calibration_days:
            return False, (
                f"Calibration exceeded {self._config.max_calibration_days} days "
                f"without reaching alignment threshold"
            )

        if report.alignment_score >= self._config.graduation_threshold:
            return True, (
                f"Alignment {report.alignment_score:.0%} >= "
                f"{self._config.graduation_threshold:.0%}"
            )

        return False, (
            f"Alignment {report.alignment_score:.0%} < "
            f"{self._config.graduation_threshold:.0%}"
        )

    def compute_alignment(self) -> CalibrationReport:
        """Compute current alignment metrics."""
        tp = self._true_positives
        fp = self._false_positives
        fn = self._false_negatives
        total = tp + fp + fn

        alignment = tp / total if total > 0 else 0.0
        fp_rate = fp / total if total > 0 else 0.0
        fn_rate = fn / total if total > 0 else 0.0

        elapsed = (datetime.now(timezone.utc) - self._started_at).days

        blockers = []
        if total < self._config.min_comparisons:
            blockers.append(
                f"Need {self._config.min_comparisons - total} more comparisons"
            )
        if alignment < self._config.graduation_threshold and total >= self._config.min_comparisons:
            blockers.append(
                f"Alignment {alignment:.0%} below "
                f"{self._config.graduation_threshold:.0%} threshold"
            )
        if fn_rate > 0.30:
            blockers.append(
                "AVT is missing >30% of user's trades — "
                "timeframe or context issue"
            )
        if fp_rate > 0.30:
            blockers.append(
                "AVT generates >30% false signals — "
                "conviction thresholds or skip conditions need tightening"
            )

        # Top mismatch reasons
        annotated = [
            m.user_annotation for m in self._mismatches
            if m.user_annotation
        ]
        reason_counts = defaultdict(int)
        for r in annotated:
            reason_counts[r] += 1
        top_reasons = sorted(
            reason_counts.keys(),
            key=lambda k: reason_counts[k],
            reverse=True,
        )[:5]

        can_grad, _ = self.should_graduate()

        return CalibrationReport(
            strategy_id=self._strategy_id,
            signals_generated=len(self._signals),
            user_trades=len(self._user_trades),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            alignment_score=round(alignment, 3),
            false_positive_rate=round(fp_rate, 3),
            false_negative_rate=round(fn_rate, 3),
            calibration_days=elapsed,
            ready_to_graduate=can_grad,
            graduation_blockers=blockers,
            total_comparisons=total,
            top_mismatch_reasons=top_reasons,
        )

    def get_mismatches(
        self, limit: int = 50
    ) -> List[CalibrationMismatch]:
        """Get recent mismatches for UI display."""
        return list(reversed(self._mismatches[-limit:]))

    def get_report(self) -> Dict[str, Any]:
        """Dict version for lifecycle manager."""
        report = self.compute_alignment()
        return {
            "alignment_score": report.alignment_score,
            "signals_generated": report.signals_generated,
            "total_comparisons": report.total_comparisons,
            "false_positive_rate": report.false_positive_rate,
            "false_negative_rate": report.false_negative_rate,
            "ready_to_graduate": report.ready_to_graduate,
            "blockers": report.graduation_blockers,
        }

    # ── Matching Logic ─────────────────────────────────────────────

    def _try_match_signal(self, signal: CalibrationSignal) -> bool:
        """Try to match a new signal against unmatched user trades."""
        window = timedelta(minutes=self._config.match_window_minutes)

        for trade in self._user_trades:
            if trade.matched_signal_id is not None:
                continue
            if trade.symbol != signal.symbol:
                continue
            if trade.direction != signal.direction:
                continue
            if abs(trade.timestamp - signal.timestamp) > window:
                continue

            # Match found
            signal.matched_trade_id = trade.id
            signal.match_type = "true_positive"
            trade.matched_signal_id = signal.id
            trade.match_type = "true_positive"
            self._true_positives += 1

            # Remove from false negatives if previously counted
            if trade.id in [m.id for m in self._mismatches]:
                self._mismatches = [
                    m for m in self._mismatches if m.id != trade.id
                ]
                self._false_negatives -= 1

            return True

        return False

    def _try_match_trade(self, trade: CalibrationUserTrade) -> bool:
        """Try to match a new user trade against unmatched signals."""
        window = timedelta(minutes=self._config.match_window_minutes)

        for signal in self._signals:
            if signal.matched_trade_id is not None:
                continue
            if signal.symbol != trade.symbol:
                continue
            if signal.direction != trade.direction:
                continue
            if abs(signal.timestamp - trade.timestamp) > window:
                continue

            # Match found
            trade.matched_signal_id = signal.id
            trade.match_type = "true_positive"
            signal.matched_trade_id = trade.id
            signal.match_type = "true_positive"
            self._true_positives += 1

            # Remove from false positives if previously counted
            if signal.id in [m.id for m in self._mismatches]:
                self._mismatches = [
                    m for m in self._mismatches if m.id != signal.id
                ]
                self._false_positives -= 1

            return True

        return False
