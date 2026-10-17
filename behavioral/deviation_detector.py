# deviation_detector.py
# behavioral/deviation_detector.py
"""
Real-time behavioral deviation detector.
Compares current session state against baseline to flag anomalies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from core.enums import BehavioralEventType, OverrideSeverity
from behavioral.baseline_modeler import BehavioralBaseline
from strategy_engine.override_detector import SessionContext

logger = logging.getLogger(__name__)


@dataclass
class DeviationAlert:
    """A detected behavioral deviation from baseline."""
    alert_type: BehavioralEventType
    severity: OverrideSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "recommendation": self.recommendation,
        }


class DeviationDetector:
    """
    Compares current session state against learned baseline.
    Generates real-time behavioral alerts.
    """

    def __init__(self, baseline: BehavioralBaseline):
        self._baseline = baseline

    def check_session(
        self,
        session: SessionContext,
        current_time: Optional[datetime] = None,
    ) -> list[DeviationAlert]:
        """
        Run all deviation checks against current session state.
        Returns list of active alerts.
        """
        current_time = current_time or datetime.now(timezone.utc)
        alerts = []

        alerts.extend(self._check_trade_frequency(session))
        alerts.extend(self._check_rapid_trading(session))
        alerts.extend(self._check_loss_spiral(session))
        alerts.extend(self._check_overtrading(session))
        alerts.extend(self._check_tilt_threshold(session))

        return alerts

    def _check_trade_frequency(self, session: SessionContext) -> list[DeviationAlert]:
        """Check if trading frequency is above baseline."""
        alerts = []
        baseline_max = self._baseline.avg_trades_per_day + (2 * self._baseline.std_trades_per_day)

        if baseline_max > 0 and session.trades_today > baseline_max:
            excess = session.trades_today - baseline_max
            severity = (
                OverrideSeverity.CRITICAL if excess > 3
                else OverrideSeverity.HIGH if excess > 1
                else OverrideSeverity.MEDIUM
            )
            alerts.append(DeviationAlert(
                alert_type=BehavioralEventType.OVERTRADE,
                severity=severity,
                message=(
                    f"Trading frequency above baseline: {session.trades_today} trades "
                    f"(baseline avg {self._baseline.avg_trades_per_day:.1f}, "
                    f"2σ threshold {baseline_max:.0f})"
                ),
                details={
                    "trades_today": session.trades_today,
                    "baseline_avg": self._baseline.avg_trades_per_day,
                    "threshold": baseline_max,
                },
                recommendation=(
                    "Consider pausing. You're trading more frequently than your "
                    "historical pattern, which correlates with lower quality decisions."
                ),
            ))

        return alerts

    def _check_rapid_trading(self, session: SessionContext) -> list[DeviationAlert]:
        """Check if time between trades is below baseline minimum."""
        if not session.last_trade_time:
            return []

        alerts = []
        safe_min_gap = max(
            self._baseline.min_time_between_trades_min * 0.5,
            2.0,
        )

        if session.last_trade_time:
            now = datetime.now(timezone.utc)
            minutes_since = (now - session.last_trade_time).total_seconds() / 60

            if minutes_since < safe_min_gap and session.last_trade_pnl and session.last_trade_pnl < 0:
                alerts.append(DeviationAlert(
                    alert_type=BehavioralEventType.RAPID_RE_ENTRY,
                    severity=OverrideSeverity.HIGH,
                    message=(
                        f"Rapid trading after a loss: {minutes_since:.1f}min since last trade "
                        f"(baseline min gap: {self._baseline.min_time_between_trades_min:.1f}min)"
                    ),
                    details={
                        "minutes_since_last": minutes_since,
                        "last_pnl": session.last_trade_pnl,
                    },
                    recommendation=(
                        "Step away for at least 5 minutes. "
                        "Your best trades come with patience, not speed."
                    ),
                ))

        return alerts

    def _check_loss_spiral(self, session: SessionContext) -> list[DeviationAlert]:
        """Detect potential loss spiral / revenge trading conditions."""
        alerts = []

        if session.consecutive_losses >= 3:
            severity = (
                OverrideSeverity.CRITICAL if session.consecutive_losses >= 5
                else OverrideSeverity.HIGH
            )
            alerts.append(DeviationAlert(
                alert_type=BehavioralEventType.REVENGE_TRADE,
                severity=severity,
                message=(
                    f"{session.consecutive_losses} consecutive losses. "
                    f"High risk of emotional decision-making."
                ),
                details={
                    "consecutive_losses": session.consecutive_losses,
                    "daily_pnl": session.daily_pnl,
                },
                recommendation=(
                    f"Your baseline shows tilt typically occurs after "
                    f"{self._baseline.avg_consecutive_losses_before_tilt:.0f} losses. "
                    f"Consider stopping for today."
                ),
            ))

        return alerts

    def _check_overtrading(self, session: SessionContext) -> list[DeviationAlert]:
        """Detect overtrading based on daily loss and continued trading."""
        alerts = []

        if session.account_balance and session.daily_pnl < 0:
            loss_pct = abs(session.daily_pnl / session.account_balance) * 100

            if loss_pct > 3.0:
                alerts.append(DeviationAlert(
                    alert_type=BehavioralEventType.LOSS_CHASING,
                    severity=OverrideSeverity.CRITICAL,
                    message=(
                        f"Daily loss of {loss_pct:.1f}% — continuing to trade "
                        f"increases risk of larger drawdown"
                    ),
                    details={
                        "daily_pnl": session.daily_pnl,
                        "loss_pct": loss_pct,
                    },
                    recommendation=(
                        "Stop trading for today. A 3%+ daily loss requires "
                        "a cooling off period. Review trades tomorrow."
                    ),
                ))
            elif loss_pct > 1.5 and session.trades_today > self._baseline.avg_trades_per_day:
                alerts.append(DeviationAlert(
                    alert_type=BehavioralEventType.OVERTRADE,
                    severity=OverrideSeverity.HIGH,
                    message=(
                        f"Down {loss_pct:.1f}% with above-average trade count. "
                        f"Pattern suggests attempting to recover losses."
                    ),
                    details={
                        "daily_pnl": session.daily_pnl,
                        "loss_pct": loss_pct,
                        "trades_today": session.trades_today,
                    },
                    recommendation=(
                        "Reduce position size by 50% or switch to paper trading "
                        "for the rest of the session."
                    ),
                ))

        return alerts

    def _check_tilt_threshold(self, session: SessionContext) -> list[DeviationAlert]:
        """Check if trader is approaching their historical tilt threshold."""
        alerts = []
        tilt_threshold = self._baseline.avg_consecutive_losses_before_tilt

        if tilt_threshold > 0 and session.consecutive_losses > 0:
            proximity = session.consecutive_losses / tilt_threshold

            if proximity >= 0.8 and proximity < 1.0:
                alerts.append(DeviationAlert(
                    alert_type=BehavioralEventType.IMPULSE_ENTRY,
                    severity=OverrideSeverity.MEDIUM,
                    message=(
                        f"Approaching tilt threshold: "
                        f"{session.consecutive_losses}/{tilt_threshold:.0f} consecutive losses. "
                        f"Historical data shows discipline breakdown near this point."
                    ),
                    details={
                        "consecutive_losses": session.consecutive_losses,
                        "tilt_threshold": tilt_threshold,
                        "proximity": proximity,
                    },
                    recommendation=(
                        "Next trade requires extra scrutiny. "
                        "Only take A+ setups or step away."
                    ),
                ))

        return alerts