"""
Confidence Decay Curve Tracker
================================

Tracks strategy edge confidence over time as a continuous curve, projecting
forward to give the trader TIME to react instead of a sudden alarm.

Complements the existing DecayDetector (binary alarm based on stat tests):
  - DecayDetector says:  "Edge is dead — pause now."
  - ConfidenceTracker says: "Edge at 78%, declining 8%/week, hits critical in 4 weeks.
                              Reduce sizing now and start testing alternatives."

Confidence is a 0-1 score blended from 5 components:
  1. Recent win rate vs historical baseline (weight 0.25)
  2. Recent avg R vs historical baseline    (weight 0.25)
  3. Recent profit factor vs baseline       (weight 0.20)
  4. Current regime alignment with strategy (weight 0.15)
  5. Recent drawdown severity               (weight 0.15)

A ConfidenceCurve includes:
  - Trajectory: stable | declining | recovering | collapsed
  - Slope: confidence change per week
  - weeks_to_threshold: at current slope, when does it hit critical
  - is_accelerating: is decay getting faster (structural, not noise)

Real-money safety:
  - Outputs ALERTS only (info|warning|critical) — never auto-pauses
  - Trader reviews and decides
  - Use alongside DecayDetector for confirmation before acting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Defaults ────────────────────────────────────────────────────────

DEFAULT_WARNING_THRESHOLD: float = 0.70      # below this = warning territory
DEFAULT_CRITICAL_THRESHOLD: float = 0.30     # below this = critical
DEFAULT_MIN_TRADES_PER_POINT: int = 10
DEFAULT_LOOKBACK_WEEKS: int = 12             # max history kept per strategy
DEFAULT_TRAJECTORY_DECLINING_SLOPE: float = -0.025  # -2.5pp/week → "declining" (sensitive enough to catch real decay)
DEFAULT_TRAJECTORY_RECOVERING_SLOPE: float = 0.03   # +3pp/week → "recovering"
DEFAULT_APPROACHING_WEEKS: float = 8.0       # alert when projected critical < this many weeks (gives lead time)


# ── Output dataclasses ──────────────────────────────────────────────


@dataclass
class ConfidencePoint:
    """A single weekly snapshot of edge confidence."""
    timestamp: datetime
    confidence: float           # 0-1, blended from 5 components
    win_rate: float
    avg_r: float
    profit_factor: float
    sample_size: int            # trades in this measurement window
    regime_alignment: float     # 0-1 how well current regime matches preferred


@dataclass
class ConfidenceCurve:
    """Trajectory snapshot for a single strategy."""
    strategy_id: str
    points: List[ConfidencePoint]

    # Derived metrics
    current_confidence: float
    trend_slope: float                      # confidence change per week
    weeks_to_threshold: Optional[float]     # at current slope, when it hits critical
    trajectory: str                         # stable | declining | recovering | collapsed

    # Acceleration & decay rate
    decay_rate_per_week: float
    is_accelerating: bool                   # decay getting faster?

    generated_at: str

    def short_summary(self) -> str:
        """One-line summary for logs / API."""
        weeks_str = (
            f", hits critical in {self.weeks_to_threshold:.1f}wk"
            if self.weeks_to_threshold is not None
            else ""
        )
        accel_str = " ACCELERATING" if self.is_accelerating else ""
        return (
            f"{self.strategy_id}: {self.current_confidence:.0%} "
            f"({self.trajectory}, slope {self.trend_slope:+.1%}/wk{weeks_str}){accel_str}"
        )


@dataclass
class ConfidenceAlert:
    """A trader-facing alert about a strategy's confidence trajectory."""
    strategy_id: str
    alert_type: str
    # Values:
    #   "confidence_declining"    — slope < -0.05/wk
    #   "approaching_threshold"   — projected to hit critical within 4 weeks
    #   "decay_accelerating"      — decay rate is increasing
    #   "confidence_recovering"   — slope > +0.03/wk after a decline
    #   "collapsed"               — current < critical threshold

    current_confidence: float
    previous_confidence: float
    decay_rate: float                  # absolute confidence loss per week
    weeks_to_threshold: Optional[float]
    severity: str                      # "info" | "warning" | "critical"
    suggested_action: str
    detected_at: str


# ── Tracker ─────────────────────────────────────────────────────────


class ConfidenceTracker:
    """
    Per-strategy confidence trajectory tracker.

    Usage:
        tracker = ConfidenceTracker()

        # On each measurement window (e.g. weekly):
        point = tracker.update(
            strategy_id="ema_pullback",
            recent_trades=[{"r_multiple": 1.5}, ...],   # last N trades
            historical_baseline={
                "avg_win_rate": 0.55,
                "avg_r": 0.75,
                "avg_pf": 1.6,
            },
            regime_probs={"trending_up": 0.6, "ranging": 0.3, ...},
            strategy_preferred_regimes=["trending_up", "trending_down"],
        )

        # Get the full curve
        curve = tracker.get_curve("ema_pullback")
        print(curve.short_summary())
        # → "ema_pullback: 78% (declining, slope -3.2%/wk, hits critical in 15.0wk)"

        # Get alerts (run after each update)
        alerts = tracker.check_alerts("ema_pullback")
    """

    def __init__(
        self,
        warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
        min_trades_per_point: int = DEFAULT_MIN_TRADES_PER_POINT,
        lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
        declining_slope: float = DEFAULT_TRAJECTORY_DECLINING_SLOPE,
        recovering_slope: float = DEFAULT_TRAJECTORY_RECOVERING_SLOPE,
        approaching_weeks: float = DEFAULT_APPROACHING_WEEKS,
    ):
        self.warning = warning_threshold
        self.critical = critical_threshold
        self.min_trades = min_trades_per_point
        self.lookback_weeks = lookback_weeks
        self.declining_slope = declining_slope
        self.recovering_slope = recovering_slope
        self.approaching_weeks = approaching_weeks

        # Per-strategy point history (in-memory; persistence is caller's job)
        self._curves: Dict[str, List[ConfidencePoint]] = {}

    # ── Update ──

    def update(
        self,
        strategy_id: str,
        recent_trades: List[dict],
        historical_baseline: dict,
        regime_probs: Optional[dict] = None,
        strategy_preferred_regimes: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[ConfidencePoint]:
        """
        Add a new confidence measurement for a strategy.

        Args:
            strategy_id:        unique id of the strategy
            recent_trades:      list of dicts with at least "r_multiple"
            historical_baseline: {avg_win_rate, avg_r, avg_pf} from full history
            regime_probs:       current regime probabilities (optional)
            strategy_preferred_regimes: list of regime labels strategy prefers
            timestamp:          when this measurement applies (defaults to now)

        Returns:
            ConfidencePoint if computed, None if insufficient data
        """
        if len(recent_trades) < self.min_trades:
            logger.debug(
                "%s: %d trades < %d required for confidence point",
                strategy_id, len(recent_trades), self.min_trades,
            )
            return None

        ts = timestamp or datetime.utcnow()
        r_mults = [float(t.get("r_multiple", 0) or 0) for t in recent_trades]

        # ── Five component scores ──
        wr_score = self._win_rate_score(r_mults, historical_baseline.get("avg_win_rate", 0.5))
        r_score = self._avg_r_score(r_mults, historical_baseline.get("avg_r", 0.5))
        pf_score = self._pf_score(r_mults, historical_baseline.get("avg_pf", 1.5))
        regime_score = self._regime_alignment_score(regime_probs, strategy_preferred_regimes)
        dd_score = self._drawdown_score(r_mults)

        confidence = (
            wr_score * 0.25
            + r_score * 0.25
            + pf_score * 0.20
            + regime_score * 0.15
            + dd_score * 0.15
        )
        confidence = max(0.0, min(1.0, confidence))

        # Aggregate stats for the point
        wins = sum(1 for r in r_mults if r > 0)
        gross_p = sum(r for r in r_mults if r > 0)
        gross_l = sum(abs(r) for r in r_mults if r <= 0)
        pf = (gross_p / gross_l) if gross_l > 0 else 10.0  # cap inf

        point = ConfidencePoint(
            timestamp=ts,
            confidence=round(confidence, 4),
            win_rate=round(wins / len(r_mults), 4),
            avg_r=round(float(np.mean(r_mults)), 4),
            profit_factor=round(pf, 4),
            sample_size=len(r_mults),
            regime_alignment=round(regime_score, 4),
        )

        # Append + trim
        self._curves.setdefault(strategy_id, []).append(point)
        cutoff = ts - timedelta(weeks=self.lookback_weeks * 2)
        self._curves[strategy_id] = [
            p for p in self._curves[strategy_id] if p.timestamp > cutoff
        ]

        return point

    # ── Get curve ──

    def get_curve(self, strategy_id: str) -> Optional[ConfidenceCurve]:
        """
        Build a full ConfidenceCurve for a strategy from its history.
        Returns None if fewer than 3 points (need at least 3 for slope).
        """
        points = self._curves.get(strategy_id, [])
        if len(points) < 3:
            return None

        confidences = [p.confidence for p in points]
        timestamps = [p.timestamp for p in points]
        current = confidences[-1]

        # Slope (confidence change per week)
        weeks_span = (timestamps[-1] - timestamps[0]).days / 7.0
        if weeks_span > 0:
            slope = (confidences[-1] - confidences[0]) / weeks_span
        else:
            slope = 0.0

        # Weeks-to-threshold projection
        if slope < -0.005 and current > self.critical:
            weeks_to = (current - self.critical) / abs(slope)
        else:
            weeks_to = None

        # Trajectory classification
        if current < self.critical:
            trajectory = "collapsed"
        elif slope < self.declining_slope:
            trajectory = "declining"
        elif slope > self.recovering_slope:
            trajectory = "recovering"
        else:
            trajectory = "stable"

        # Acceleration: is the second half of history declining faster than the first half?
        is_accelerating = False
        if len(confidences) >= 6:
            mid = len(confidences) // 2
            first_half_change = confidences[mid] - confidences[0]
            second_half_change = confidences[-1] - confidences[mid]
            # Accelerating decay: second-half change is more negative than first-half
            is_accelerating = (
                second_half_change < first_half_change - 0.02
                and second_half_change < 0
            )

        return ConfidenceCurve(
            strategy_id=strategy_id,
            points=points,
            current_confidence=round(current, 4),
            trend_slope=round(slope, 4),
            weeks_to_threshold=round(weeks_to, 1) if weeks_to is not None else None,
            trajectory=trajectory,
            decay_rate_per_week=round(abs(min(slope, 0.0)), 4),
            is_accelerating=is_accelerating,
            generated_at=datetime.utcnow().isoformat(),
        )

    # ── Alerts ──

    def check_alerts(self, strategy_id: str) -> List[ConfidenceAlert]:
        """
        Generate alerts for a strategy based on its current curve.
        Returns 0 to N alerts depending on severity / state.
        """
        curve = self.get_curve(strategy_id)
        if not curve:
            return []

        prev_conf = curve.points[-2].confidence if len(curve.points) >= 2 else 1.0
        now_str = datetime.utcnow().isoformat()
        alerts: List[ConfidenceAlert] = []

        # 1. Collapsed
        if curve.trajectory == "collapsed":
            alerts.append(ConfidenceAlert(
                strategy_id=strategy_id,
                alert_type="collapsed",
                current_confidence=curve.current_confidence,
                previous_confidence=prev_conf,
                decay_rate=curve.decay_rate_per_week,
                weeks_to_threshold=0.0,
                severity="critical",
                suggested_action=(
                    f"PAUSE {strategy_id}. Confidence at {curve.current_confidence:.0%} "
                    f"is below critical ({self.critical:.0%}). Edge likely structurally dead."
                ),
                detected_at=now_str,
            ))

        # 2. Approaching threshold (within configured weeks at current slope)
        elif (
            curve.weeks_to_threshold is not None
            and curve.weeks_to_threshold < self.approaching_weeks
        ):
            severity = "critical" if curve.weeks_to_threshold < 4 else "warning"
            alerts.append(ConfidenceAlert(
                strategy_id=strategy_id,
                alert_type="approaching_threshold",
                current_confidence=curve.current_confidence,
                previous_confidence=prev_conf,
                decay_rate=curve.decay_rate_per_week,
                weeks_to_threshold=curve.weeks_to_threshold,
                severity=severity,
                suggested_action=(
                    f"{strategy_id} declining: {curve.current_confidence:.0%} confidence, "
                    f"projected to hit critical in {curve.weeks_to_threshold:.0f} weeks. "
                    f"Reduce sizing 50% and begin testing alternatives."
                ),
                detected_at=now_str,
            ))

        # 2b. Generic declining (catches "below warning AND clearly trending down"
        #     even when projected weeks > approaching_weeks)
        elif (
            curve.trajectory == "declining"
            and curve.current_confidence < self.warning
        ):
            alerts.append(ConfidenceAlert(
                strategy_id=strategy_id,
                alert_type="confidence_declining",
                current_confidence=curve.current_confidence,
                previous_confidence=prev_conf,
                decay_rate=curve.decay_rate_per_week,
                weeks_to_threshold=curve.weeks_to_threshold,
                severity="warning",
                suggested_action=(
                    f"{strategy_id} confidence at {curve.current_confidence:.0%} and falling "
                    f"({curve.decay_rate_per_week:.1%}/wk). Watch closely — reduce sizing on "
                    f"new trades."
                ),
                detected_at=now_str,
            ))

        # 3. Decay accelerating
        if curve.is_accelerating and curve.trajectory == "declining":
            alerts.append(ConfidenceAlert(
                strategy_id=strategy_id,
                alert_type="decay_accelerating",
                current_confidence=curve.current_confidence,
                previous_confidence=prev_conf,
                decay_rate=curve.decay_rate_per_week,
                weeks_to_threshold=curve.weeks_to_threshold,
                severity="critical" if curve.current_confidence < self.warning else "warning",
                suggested_action=(
                    f"{strategy_id} decay is ACCELERATING. Loss rate "
                    f"{curve.decay_rate_per_week:.1%}/week. This is structural, not drawdown."
                ),
                detected_at=now_str,
            ))

        # 4. Recovery
        if (
            curve.trajectory == "recovering"
            and prev_conf < curve.current_confidence
            and prev_conf < self.warning
        ):
            alerts.append(ConfidenceAlert(
                strategy_id=strategy_id,
                alert_type="confidence_recovering",
                current_confidence=curve.current_confidence,
                previous_confidence=prev_conf,
                decay_rate=0.0,
                weeks_to_threshold=None,
                severity="info",
                suggested_action=(
                    f"{strategy_id} recovering: {prev_conf:.0%} → {curve.current_confidence:.0%}. "
                    f"Consider restoring normal sizing if trend continues for 2+ more weeks."
                ),
                detected_at=now_str,
            ))

        return alerts

    # ── Component scores (each returns 0-1) ──

    @staticmethod
    def _win_rate_score(r_mults: List[float], baseline_wr: float) -> float:
        recent_wr = float(np.mean([1 if r > 0 else 0 for r in r_mults]))
        if baseline_wr <= 0:
            return 0.5
        ratio = recent_wr / baseline_wr
        return float(max(0.0, min(1.0, ratio)))

    @staticmethod
    def _avg_r_score(r_mults: List[float], baseline_r: float) -> float:
        recent_r = float(np.mean(r_mults))
        if baseline_r <= 0:
            return 0.5 if recent_r >= 0 else 0.2
        ratio = recent_r / baseline_r
        return float(max(0.0, min(1.0, ratio)))

    @staticmethod
    def _pf_score(r_mults: List[float], baseline_pf: float) -> float:
        gross_p = sum(r for r in r_mults if r > 0)
        gross_l = sum(abs(r) for r in r_mults if r <= 0)
        pf = (gross_p / gross_l) if gross_l > 0 else 10.0
        if baseline_pf <= 0:
            return 0.5
        ratio = pf / baseline_pf
        return float(max(0.0, min(1.0, ratio)))

    @staticmethod
    def _regime_alignment_score(
        regime_probs: Optional[dict],
        preferred: Optional[List[str]],
    ) -> float:
        if not regime_probs or not preferred:
            return 0.5
        alignment = sum(float(regime_probs.get(r, 0.0)) for r in preferred)
        # Boost: 0.6 alignment → 0.9 confidence (regime probs sum to 1, full preferred = 1.0)
        return float(min(alignment * 1.5, 1.0))

    @staticmethod
    def _drawdown_score(r_mults: List[float]) -> float:
        """Confidence based on max drawdown depth (in R-multiples) over recent trades."""
        if not r_mults:
            return 0.5
        equity = np.cumsum(r_mults)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        if max_dd < 2:
            return 1.0
        if max_dd < 4:
            return 0.75
        if max_dd < 6:
            return 0.50
        if max_dd < 10:
            return 0.25
        return 0.10

    # ── Convenience formatters ──

    def format_curve(self, curve: ConfidenceCurve, sparkline_n: int = 10) -> str:
        """Markdown-friendly summary with a small sparkline of recent confidences."""
        lines = [
            f"## Confidence Curve — {curve.strategy_id}",
            f"Current: **{curve.current_confidence:.0%}** | "
            f"Trajectory: **{curve.trajectory}** | "
            f"Slope: {curve.trend_slope:+.1%}/wk",
        ]
        if curve.weeks_to_threshold is not None:
            lines.append(
                f"Projected to hit critical in **{curve.weeks_to_threshold:.0f} weeks**"
            )
        if curve.is_accelerating:
            lines.append("Decay is **accelerating** — structural, not noise")

        if curve.points:
            recent = curve.points[-sparkline_n:]
            line = " -> ".join(f"{p.confidence:.0%}" for p in recent)
            lines.append(f"\nRecent: {line}")

        return "\n".join(lines)

    # ── Bulk operations ──

    def update_all(
        self,
        snapshots: Dict[str, dict],
    ) -> Dict[str, Optional[ConfidencePoint]]:
        """
        Bulk update across multiple strategies.

        Args:
            snapshots: {strategy_id: {recent_trades, historical_baseline, regime_probs, preferred_regimes}}
        Returns: {strategy_id: ConfidencePoint or None}
        """
        results: Dict[str, Optional[ConfidencePoint]] = {}
        for sid, snap in snapshots.items():
            try:
                point = self.update(
                    strategy_id=sid,
                    recent_trades=snap.get("recent_trades", []),
                    historical_baseline=snap.get("historical_baseline", {}),
                    regime_probs=snap.get("regime_probs"),
                    strategy_preferred_regimes=snap.get("strategy_preferred_regimes"),
                )
                results[sid] = point
            except Exception as e:
                logger.warning("Confidence update failed for %s: %s", sid, e)
                results[sid] = None
        return results

    def all_alerts(self, strategy_ids: List[str]) -> List[ConfidenceAlert]:
        out: List[ConfidenceAlert] = []
        for sid in strategy_ids:
            out.extend(self.check_alerts(sid))
        return out


# ── Module singleton ───────────────────────────────────────────────


_singleton: Optional[ConfidenceTracker] = None


def get_confidence_tracker() -> ConfidenceTracker:
    """Module-level singleton."""
    global _singleton
    if _singleton is None:
        _singleton = ConfidenceTracker()
    return _singleton
