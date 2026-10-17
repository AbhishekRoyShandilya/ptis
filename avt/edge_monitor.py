# avt/edge_monitor.py
"""
Continuous Edge Monitor
========================

Monitors strategy performance during autonomous trading.
Compares rolling metrics against baseline to detect:
  - Edge decay (performance degradation over time)
  - Translation drift (AVT's R distribution diverges from expected)
  - Regime-conditional degradation vs genuine edge loss

Uses scipy KS test for distribution comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

ROLLING_WINDOW = 30
WARNING_EXPECTANCY_DROP_PCT = 0.30
DEGRADED_EXPECTANCY_DROP_PCT = 0.50
WARNING_WINRATE_DROP = 0.10
DEGRADED_WINRATE_DROP = 0.15
CRITICAL_DRAWDOWN_PCT = 0.15
DRIFT_THRESHOLD = 0.30


class DegradationLevel(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class MonitorAction(str, Enum):
    CONTINUE = "continue"
    ALERT = "alert"
    REDUCE_SIZE = "reduce_size"
    PAUSE = "pause"


@dataclass
class PerformanceBaseline:
    """Baseline metrics established at strategy start or from history."""
    strategy_id: str = ""
    source: str = ""  # "user_history" | "probation" | "calibration"
    expectancy_r: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_winner_r: float = 0.0
    avg_loser_r: float = 0.0
    r_distribution: List[float] = field(default_factory=list)
    trade_count: int = 0
    established_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def r_std(self) -> float:
        if len(self.r_distribution) < 2:
            return 1.0
        return float(np.std(self.r_distribution))


@dataclass
class DegradationCheck:
    """Result of a degradation check."""
    level: str = DegradationLevel.HEALTHY.value
    action: str = MonitorAction.CONTINUE.value
    expectancy_delta_pct: float = 0.0
    win_rate_delta: float = 0.0
    profit_factor_current: float = 0.0
    drift_score: float = 0.0
    rolling_expectancy: float = 0.0
    rolling_win_rate: float = 0.0
    rolling_trade_count: int = 0
    message: str = ""


class ContinuousEdgeMonitor:
    """
    Monitors strategy edge during autonomous AVT trading.

    Compares rolling 30-trade window against established baseline.
    Triggers position reduction or pause on degradation.
    """

    def __init__(
        self,
        strategy_id: str,
        baseline: Optional[PerformanceBaseline] = None,
    ):
        self._strategy_id = strategy_id
        self._baseline = baseline
        self._trades: List[float] = []  # r_multiples
        self._equity_curve: List[float] = [0.0]  # cumulative R
        self._peak_equity: float = 0.0

    def set_baseline(self, baseline: PerformanceBaseline) -> None:
        self._baseline = baseline

    def establish_baseline_from_trades(
        self,
        r_multiples: List[float],
        source: str = "historical",
    ) -> PerformanceBaseline:
        """Build baseline from a list of R-multiples."""
        arr = np.array(r_multiples)
        winners = arr[arr > 0]
        losers = arr[arr < 0]

        baseline = PerformanceBaseline(
            strategy_id=self._strategy_id,
            source=source,
            expectancy_r=float(np.mean(arr)),
            win_rate=float(np.mean(arr > 0)),
            profit_factor=(
                float(np.sum(winners) / abs(np.sum(losers)))
                if len(losers) > 0 and np.sum(losers) != 0
                else 999.0
            ),
            avg_winner_r=float(np.mean(winners)) if len(winners) > 0 else 0.0,
            avg_loser_r=float(np.mean(losers)) if len(losers) > 0 else 0.0,
            r_distribution=r_multiples.copy(),
            trade_count=len(r_multiples),
        )
        self._baseline = baseline
        return baseline

    def record_trade(self, r_multiple: float) -> None:
        """Record a closed trade result."""
        self._trades.append(r_multiple)
        cum = self._equity_curve[-1] + r_multiple
        self._equity_curve.append(cum)
        self._peak_equity = max(self._peak_equity, cum)

    def check_degradation(self) -> DegradationCheck:
        """
        Compare rolling window against baseline.
        Returns degradation level and recommended action.
        """
        check = DegradationCheck()

        if not self._baseline or len(self._trades) < 10:
            check.message = "insufficient data"
            return check

        # Rolling window
        window = self._trades[-ROLLING_WINDOW:]
        arr = np.array(window)

        rolling_exp = float(np.mean(arr))
        rolling_wr = float(np.mean(arr > 0))
        winners = arr[arr > 0]
        losers = arr[arr < 0]
        rolling_pf = (
            float(np.sum(winners) / abs(np.sum(losers)))
            if len(losers) > 0 and np.sum(losers) != 0
            else 999.0
        )

        check.rolling_expectancy = round(rolling_exp, 4)
        check.rolling_win_rate = round(rolling_wr, 4)
        check.profit_factor_current = round(rolling_pf, 3)
        check.rolling_trade_count = len(window)

        # Expectancy delta
        baseline_exp = self._baseline.expectancy_r
        if baseline_exp > 0:
            check.expectancy_delta_pct = round(
                (rolling_exp - baseline_exp) / baseline_exp, 3
            )
        elif baseline_exp == 0:
            check.expectancy_delta_pct = -1.0 if rolling_exp < 0 else 0.0
        else:
            check.expectancy_delta_pct = 0.0

        # Win rate delta
        check.win_rate_delta = round(
            rolling_wr - self._baseline.win_rate, 3
        )

        # Translation drift (KS test)
        check.drift_score = self._compute_drift(window)

        # Drawdown check
        drawdown = self._peak_equity - self._equity_curve[-1]
        drawdown_pct = (
            drawdown / max(abs(self._peak_equity), 1.0)
            if self._peak_equity > 0
            else 0.0
        )

        # Determine level
        if (
            drawdown_pct > CRITICAL_DRAWDOWN_PCT
            or (rolling_exp < 0 and len(window) >= ROLLING_WINDOW)
        ):
            check.level = DegradationLevel.CRITICAL.value
            check.action = MonitorAction.PAUSE.value
            check.message = (
                f"CRITICAL: drawdown={drawdown_pct:.1%}, "
                f"expectancy={rolling_exp:+.3f}R"
            )
        elif (
            check.expectancy_delta_pct < -DEGRADED_EXPECTANCY_DROP_PCT
            or (
                check.win_rate_delta < -DEGRADED_WINRATE_DROP
                and rolling_pf < 0.9
            )
        ):
            check.level = DegradationLevel.DEGRADED.value
            check.action = MonitorAction.REDUCE_SIZE.value
            check.message = (
                f"DEGRADED: expectancy dropped {abs(check.expectancy_delta_pct):.0%}, "
                f"win_rate delta={check.win_rate_delta:+.1%}"
            )
        elif (
            check.expectancy_delta_pct < -WARNING_EXPECTANCY_DROP_PCT
            or check.win_rate_delta < -WARNING_WINRATE_DROP
        ):
            check.level = DegradationLevel.WARNING.value
            check.action = MonitorAction.ALERT.value
            check.message = (
                f"WARNING: expectancy dropped {abs(check.expectancy_delta_pct):.0%}"
            )
        else:
            check.level = DegradationLevel.HEALTHY.value
            check.action = MonitorAction.CONTINUE.value
            check.message = "healthy"

        return check

    def _compute_drift(self, rolling_r: List[float]) -> float:
        """
        KS test comparing rolling R distribution vs baseline.
        Returns 0-1 where 0 = identical, 1 = completely different.
        """
        if (
            not self._baseline
            or len(self._baseline.r_distribution) < 10
            or len(rolling_r) < 10
        ):
            return 0.0

        try:
            from scipy.stats import ks_2samp
            stat, _ = ks_2samp(
                self._baseline.r_distribution, rolling_r
            )
            return round(float(stat), 3)
        except ImportError:
            # Fallback: simple mean/std comparison
            baseline_mean = np.mean(self._baseline.r_distribution)
            rolling_mean = np.mean(rolling_r)
            baseline_std = max(np.std(self._baseline.r_distribution), 0.1)
            return round(
                min(1.0, abs(rolling_mean - baseline_mean) / baseline_std),
                3,
            )

    def get_status(self) -> Dict[str, Any]:
        """Current monitor status for API/frontend."""
        check = self.check_degradation()
        return {
            "level": check.level,
            "action": check.action,
            "rolling_expectancy": check.rolling_expectancy,
            "rolling_win_rate": check.rolling_win_rate,
            "rolling_trade_count": check.rolling_trade_count,
            "drift_score": check.drift_score,
            "baseline_expectancy": (
                self._baseline.expectancy_r if self._baseline else None
            ),
            "baseline_win_rate": (
                self._baseline.win_rate if self._baseline else None
            ),
            "total_trades": len(self._trades),
            "message": check.message,
        }
