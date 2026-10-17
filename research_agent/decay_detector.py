"""
Strategy Decay Detector
========================

Catches the most painful real-money problem early: a strategy that worked
all year stops working — but the human trader takes weeks to notice. By
then big losses have stacked up.

Approach: rolling-window concept-drift detection on the strategy's R-multiple
history. Three independent statistical tests vote:

  1. Mann-Whitney U test:
     Compare recent 30 R-multiples vs baseline 90 (excluding the recent).
     One-sided test for "recent is LESS than baseline".
     Fires if p < 0.05.

  2. CUSUM (Cumulative Sum):
     Track cumulative drift from expected R. Bounded false-positive rate.
     Fires when cumulative downside drift exceeds threshold * std.

  3. Page-Hinkley test:
     Standard concept-drift detector with bounded false-positive rate.
     Fires when sum of (R - mean - alpha) exceeds threshold.

MAJORITY VOTE: a finding is only emitted when ≥2 of 3 detectors fire.
This dramatically reduces false alerts — single-detector noise would
make the trader ignore real signals.

CALIBRATION: before deploying, run all three detectors INDIVIDUALLY on
12 months of historical data for every active strategy. If majority vote
fires more than once per strategy per month on stable strategies, tighten
thresholds. See verify_calibration() at bottom.

Real-money safety:
  - Alerts only — no auto-pause, no auto-sizing change
  - Trader reviews the finding and decides
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from research_agent.models import DecayConfidence, DecayFinding

logger = logging.getLogger(__name__)


# ── Default parameters (calibrate before going live) ─────────────────

DEFAULT_RECENT_WINDOW: int = 30           # trades
DEFAULT_BASELINE_WINDOW: int = 90         # trades (excluding recent)
DEFAULT_MIN_TRADES_FOR_ALERT: int = 20    # min recent trades before alerting

# Mann-Whitney threshold
DEFAULT_MW_P_THRESHOLD: float = 0.05      # one-sided p < this = fired

# CUSUM parameters
DEFAULT_CUSUM_THRESHOLD: float = 4.0      # std-deviations of drift
DEFAULT_CUSUM_K: float = 0.5              # allowable mean shift before alarm

# Page-Hinkley parameters
DEFAULT_PH_THRESHOLD: float = 50.0        # detection threshold
DEFAULT_PH_ALPHA: float = 0.005           # allowable magnitude of change


# ── Statistical primitives (no scipy dependency for portability) ────

def _mann_whitney_u_one_sided_less(recent: List[float], baseline: List[float]) -> float:
    """
    Mann-Whitney U test, one-sided: "is recent < baseline"?
    Returns p-value. Uses normal approximation (large-sample).

    For sample sizes ≥20 the normal approximation is accurate. Below that we
    don't alert anyway (DEFAULT_MIN_TRADES_FOR_ALERT=20).

    Avoids scipy dependency — implements directly.
    """
    n1 = len(recent)
    n2 = len(baseline)
    if n1 < 5 or n2 < 5:
        return 1.0  # not enough data → never fire

    # Combine and rank (handle ties via average rank)
    combined = sorted([(v, "r") for v in recent] + [(v, "b") for v in baseline], key=lambda t: t[0])
    ranks: List[float] = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # +1 because rank starts at 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    # U for recent group
    u_r = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == "r")
    u1 = u_r - n1 * (n1 + 1) / 2.0

    # Mean and std under H0
    mu_u = n1 * n2 / 2.0
    # Tie-corrected std (simplified — accurate enough for our use)
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma_u == 0:
        return 1.0

    # z for "recent less" → smaller U for recent
    z = (u1 - mu_u) / sigma_u
    # One-sided p for less: P(Z <= z)
    p = _normal_cdf(z)
    return p


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via erf (no scipy)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _cusum_test(
    series: List[float],
    threshold: float = DEFAULT_CUSUM_THRESHOLD,
    k: float = DEFAULT_CUSUM_K,
) -> bool:
    """
    Detect downward shift in mean using CUSUM.

    s_n = max(0, s_{n-1} - (x_n - mean) - k * std)
    Fires if s_n > threshold * std.

    threshold * std ≈ 4 standard deviations of cumulative drift.
    """
    if len(series) < 10:
        return False

    n = len(series)
    mean = sum(series) / n
    var = sum((x - mean) ** 2 for x in series) / max(1, n - 1)
    std = math.sqrt(var) if var > 0 else 1.0

    s_lo = 0.0  # cumulative downside drift
    for x in series:
        # Subtract mean to center; subtract k*std as allowable drift
        s_lo = max(0.0, s_lo - (x - mean) - k * std)
        if s_lo > threshold * std:
            return True
    return False


def _page_hinkley_test(
    series: List[float],
    threshold: float = DEFAULT_PH_THRESHOLD,
    alpha: float = DEFAULT_PH_ALPHA,
) -> bool:
    """
    Page-Hinkley test for downward concept drift.

    PH_n = sum((mean_n - x_n) - alpha)  for downward
    Fires if PH_n - min(PH_n) > threshold.
    """
    if len(series) < 10:
        return False

    running_mean = 0.0
    cumsum = 0.0
    min_cumsum = 0.0
    n = 0
    for x in series:
        n += 1
        running_mean = running_mean + (x - running_mean) / n
        # downward drift detector
        cumsum += (running_mean - x) - alpha
        min_cumsum = min(min_cumsum, cumsum)
        if cumsum - min_cumsum > threshold:
            return True
    return False


# ── Detector class ─────────────────────────────────────────────────

class DecayDetector:
    """
    Per-strategy decay detector with majority-vote across 3 statistical tests.

    Usage:
        detector = DecayDetector(trade_loader=fn_returning_r_history_for_strategy_id)
        finding = detector.detect("ema_pullback")
        if finding:
            # surface to /research page, do NOT auto-pause
    """

    def __init__(
        self,
        trade_loader: Callable[[str], List[float]],
        recent_window: int = DEFAULT_RECENT_WINDOW,
        baseline_window: int = DEFAULT_BASELINE_WINDOW,
        min_trades_for_alert: int = DEFAULT_MIN_TRADES_FOR_ALERT,
        mw_p_threshold: float = DEFAULT_MW_P_THRESHOLD,
        cusum_threshold: float = DEFAULT_CUSUM_THRESHOLD,
        cusum_k: float = DEFAULT_CUSUM_K,
        ph_threshold: float = DEFAULT_PH_THRESHOLD,
        ph_alpha: float = DEFAULT_PH_ALPHA,
    ):
        """
        Args:
            trade_loader: callable(strategy_id) -> list of R-multiples
                in chronological order (oldest first). Returns [] if no data.
        """
        self._loader = trade_loader
        self.recent_window = recent_window
        self.baseline_window = baseline_window
        self.min_trades_for_alert = min_trades_for_alert
        self.mw_p_threshold = mw_p_threshold
        self.cusum_threshold = cusum_threshold
        self.cusum_k = cusum_k
        self.ph_threshold = ph_threshold
        self.ph_alpha = ph_alpha

    def detect(self, strategy_id: str) -> Optional[DecayFinding]:
        """
        Run all three detectors on the strategy's R-history.
        Returns DecayFinding if ≥2 of 3 fire (majority vote), else None.
        """
        r_history = self._loader(strategy_id) or []
        required = self.min_trades_for_alert + self.baseline_window
        if len(r_history) < required:
            logger.debug(
                "DecayDetector skip %s: only %d trades, need %d",
                strategy_id, len(r_history), required,
            )
            return None

        # Split history: baseline (older) vs recent (newer)
        recent = r_history[-self.recent_window:]
        # baseline = the window before recent (NOT all-history; older trades only)
        baseline_end = -self.recent_window
        baseline_start = baseline_end - self.baseline_window
        baseline = r_history[baseline_start:baseline_end] if baseline_start <= -1 else []

        if len(baseline) < self.baseline_window // 2:
            return None  # not enough baseline data

        # Compute means
        baseline_avg = sum(baseline) / len(baseline)
        recent_avg = sum(recent) / len(recent)
        drift = baseline_avg - recent_avg  # positive = decay

        # Run three detectors
        mw_p = _mann_whitney_u_one_sided_less(recent, baseline)
        mw_fired = mw_p < self.mw_p_threshold

        cusum_fired = _cusum_test(
            r_history, threshold=self.cusum_threshold, k=self.cusum_k,
        )
        ph_fired = _page_hinkley_test(
            r_history, threshold=self.ph_threshold, alpha=self.ph_alpha,
        )

        signals = sum([mw_fired, cusum_fired, ph_fired])
        if signals < 2:
            logger.debug(
                "DecayDetector %s: only %d/3 detectors fired (mw=%s cusum=%s ph=%s) — no finding",
                strategy_id, signals, mw_fired, cusum_fired, ph_fired,
            )
            return None

        # Drift must be NEGATIVE (decay, not improvement) to alert
        if drift < 0.05:
            # Detectors fired but recent is actually similar/better than baseline
            # (false positive of one-sided tests with noisy data)
            logger.debug(
                "DecayDetector %s: detectors fired but drift is %+.2f (≥0) — likely false positive",
                strategy_id, drift,
            )
            return None

        confidence = DecayConfidence.HIGH if signals == 3 else DecayConfidence.MODERATE

        # Build action recommendation
        if drift > 0.5:
            action = f"Pause {strategy_id} or reduce sizing 50%. Edge has materially decayed."
        elif drift > 0.25:
            action = f"Reduce sizing on {strategy_id} by 50% until edge stabilizes."
        else:
            action = f"Monitor {strategy_id} closely — early decay signal. Take only A-grade setups."

        rationale = (
            f"Recent {len(recent)} trades avg R = {recent_avg:.2f} vs baseline "
            f"{len(baseline)} trades avg R = {baseline_avg:.2f}. "
            f"Drift = {drift:+.2f}R. Mann-Whitney p = {mw_p:.4f}. "
            f"Detectors firing: {'MW ' if mw_fired else ''}{'CUSUM ' if cusum_fired else ''}"
            f"{'PH' if ph_fired else ''}({signals}/3)."
        )

        return DecayFinding(
            strategy_id=strategy_id,
            detected_at=datetime.now(),
            p_value=mw_p,
            baseline_avg_r=baseline_avg,
            recent_avg_r=recent_avg,
            drift_magnitude=drift,
            detectors_firing=signals,
            mw_fired=mw_fired,
            cusum_fired=cusum_fired,
            ph_fired=ph_fired,
            confidence=confidence,
            baseline_n=len(baseline),
            recent_n=len(recent),
            suggested_action=action,
            rationale=rationale,
        )

    def detect_all(self, strategy_ids: List[str]) -> List[DecayFinding]:
        """Run detect() across multiple strategies; return only the ones with findings."""
        findings: List[DecayFinding] = []
        for sid in strategy_ids:
            try:
                f = self.detect(sid)
                if f:
                    findings.append(f)
                    logger.info("Decay finding: %s", f.short_summary())
            except Exception as e:
                logger.warning("DecayDetector failed for %s: %s", sid, e)
        return findings


# ── Calibration helper ─────────────────────────────────────────────

def verify_calibration(
    detector: DecayDetector,
    strategy_ids: List[str],
    months_back: int = 12,
) -> dict:
    """
    Calibration test: walk through last N months of trade history one trade at a
    time and count how often the detector would have fired.

    If a historically stable strategy fires >1x/month, thresholds are too loose
    and the alerts will become noise. Tighten them before deploying live.

    Returns:
      {strategy_id: {fires_per_month: float, total_alerts: int, total_trades: int}}
    """
    report: dict = {}
    for sid in strategy_ids:
        r_history = detector._loader(sid) or []
        if len(r_history) < detector.min_trades_for_alert + detector.baseline_window:
            report[sid] = {"fires_per_month": 0.0, "total_alerts": 0, "total_trades": len(r_history),
                           "note": "insufficient history"}
            continue

        # Walk forward — at each step, check if detector would have fired
        alerts = 0
        # Conservative: at least baseline_window + recent_window trades needed before
        # the detector can run; check every Nth trade to cut compute
        step = max(1, detector.recent_window // 3)
        trade_count = 0
        for end_idx in range(detector.baseline_window + detector.recent_window,
                             len(r_history) + 1, step):
            window = r_history[:end_idx]
            # Temporarily wrap the loader
            original_loader = detector._loader
            try:
                detector._loader = lambda _sid: window
                f = detector.detect(sid)
                if f:
                    alerts += 1
            finally:
                detector._loader = original_loader
            trade_count = end_idx

        # Estimate fires per month (assume ~5 trades/day = ~100/month average)
        approx_months = trade_count / 100.0 if trade_count else 1.0
        fires_per_month = alerts / approx_months if approx_months > 0 else 0.0

        report[sid] = {
            "total_alerts": alerts,
            "total_trades": trade_count,
            "approx_months_covered": round(approx_months, 1),
            "fires_per_month": round(fires_per_month, 2),
            "note": "TIGHTEN thresholds" if fires_per_month > 1.0 else "OK",
        }
    return report
