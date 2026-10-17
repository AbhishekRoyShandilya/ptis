"""
Stage 5b — Regime Target Optimizer
===================================

For each market regime, finds the R-multiple target that maximizes profit factor
using MFE (Max Favorable Excursion) data already captured in BacktestResult.

Also computes:
  - Reversal cliff:   the lowest R-level where >40% of trades that reach it
                      end up reversing to <0.5R (catastrophic give-back)
  - Reversal curve:   continuous picture of "reach R → end up where" across MFE levels

Output guides:
  - AVT exit logic: use regime-specific target instead of a single 1.5R/3R rule
  - Co-planner trade plans: show regime-specific T1/T2 levels
  - Trader awareness: visualise where holding past X R becomes statistically painful

This is PURE MATH on existing data. No ML. No drift. Argmax over a small R-grid.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from discovery.models import BacktestResult

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────

# R-multiple grid to search for the optimal target.
# Wider grid = more granular but costs nothing — still pure argmax.
_TARGET_R_GRID: List[float] = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Reversal cliff thresholds — see find_reversal_cliff() docstring for rationale
_CLIFF_R_GRID: List[float] = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
_CLIFF_REVERSAL_THRESHOLD: float = 0.40    # >40% reversal rate = cliff
_CLIFF_GIVEBACK_R: float = 0.5             # final R below this = catastrophic give-back
_CLIFF_MIN_SAMPLE: int = 10                # min trades that reached this MFE to call it

# Reversal curve granularity
_CURVE_R_START: float = 0.5
_CURVE_R_END: float = 5.0
_CURVE_R_STEP: float = 0.25
_CURVE_MIN_SAMPLE: int = 5                 # min trades per MFE level to include in curve


# ── Output dataclasses ─────────────────────────────────────────────

@dataclass
class TargetSimResult:
    """Result of simulating one R-target on a set of trades."""
    target_r: float
    n_trades: int
    win_rate: float
    avg_r: float
    sum_r: float
    profit_factor: float
    n_hit_target: int


@dataclass
class ReversalCurvePoint:
    """One point on the 'reach R → end up where' curve."""
    mfe_level: float
    n_reached: int
    reversal_pct: float        # fraction that ended < _CLIFF_GIVEBACK_R
    avg_final_r: float


@dataclass
class RegimeTargetReport:
    """Complete target-optimization output for a single regime."""
    regime: str
    sample_size: int
    optimal_target_r: float
    optimal_profit_factor: float
    optimal_avg_r: float
    optimal_win_rate: float
    reversal_cliff_r: Optional[float]      # None if no cliff detected
    target_simulations: List[TargetSimResult]
    reversal_curve: List[ReversalCurvePoint]
    recommendation: str                    # human-readable summary

    def short_summary(self) -> str:
        """One-line summary for logs / co-planner."""
        cliff = (
            f", cliff at {self.reversal_cliff_r:.1f}R"
            if self.reversal_cliff_r is not None
            else ", no cliff"
        )
        return (
            f"{self.regime}: optimal target = {self.optimal_target_r:.1f}R "
            f"(PF {self.optimal_profit_factor:.2f}{cliff}, n={self.sample_size})"
        )


# ── Core algorithms ────────────────────────────────────────────────

def simulate_target(trades: List[BacktestResult], target_r: float) -> TargetSimResult:
    """
    Simulate exiting every trade at `target_r` if its MFE reached that level,
    otherwise at the actual exit. Returns aggregate stats.

    Cost-aware: subtracts each trade's per-trade `cost_r` (set by CostModel) when
    simulating a target exit, since the trader pays brokerage+slippage+STT to book.
    Trades that didn't reach target use t.r_multiple which is already NET of costs.

    This is the foundation: for each candidate target, what would we have NETTED?
    """
    if not trades:
        return TargetSimResult(
            target_r=target_r, n_trades=0, win_rate=0.0, avg_r=0.0,
            sum_r=0.0, profit_factor=0.0, n_hit_target=0,
        )

    realized_r: List[float] = []
    hits = 0
    for t in trades:
        if t.max_favorable_excursion >= target_r:
            # Trade had the opportunity → exit at target_r minus this trade's cost
            cost = getattr(t, "cost_r", 0.0) or 0.0
            realized_r.append(target_r - cost)
            hits += 1
        else:
            # Trade never reached target → use actual exit (already net of costs)
            realized_r.append(t.r_multiple)

    arr = np.asarray(realized_r, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0

    return TargetSimResult(
        target_r=target_r,
        n_trades=len(arr),
        win_rate=float((arr > 0).mean()),
        avg_r=float(arr.mean()),
        sum_r=float(arr.sum()),
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        n_hit_target=hits,
    )


def find_optimal_target(trades: List[BacktestResult]) -> TargetSimResult:
    """
    Return the R-target that maximizes profit factor across the search grid.
    If profit_factor is infinite (no losing trades) for multiple, prefer higher avg_r.
    """
    if not trades:
        return TargetSimResult(
            target_r=0.0, n_trades=0, win_rate=0.0, avg_r=0.0,
            sum_r=0.0, profit_factor=0.0, n_hit_target=0,
        )

    sims = [simulate_target(trades, r) for r in _TARGET_R_GRID]
    # Sort by (profit_factor, avg_r) descending — handles inf PF tie via avg_r
    sims_sorted = sorted(sims, key=lambda s: (s.profit_factor, s.avg_r), reverse=True)
    return sims_sorted[0]


def find_reversal_cliff(trades: List[BacktestResult]) -> Optional[float]:
    """
    Reversal cliff = lowest R-level where:
      1. MFE reached that R (trade HAD the opportunity to take it)
      2. Final exit R < _CLIFF_GIVEBACK_R (trade gave it ALL back, catastrophic)
      3. >_CLIFF_REVERSAL_THRESHOLD of trades that reached this MFE reversed badly
      4. At least _CLIFF_MIN_SAMPLE trades reached that MFE level

    Why these specific numbers (vs initial draft of <1R / 50%):
      - <0.5R catches catastrophic give-back, not just trims
      - 40% flags earlier than 50% — by the time half reverse, you've already lost a lot
      - min 10 samples avoids noise on rare MFE levels
    """
    if not trades:
        return None

    for target_r in _CLIFF_R_GRID:
        reached = [t for t in trades if t.max_favorable_excursion >= target_r]
        if len(reached) < _CLIFF_MIN_SAMPLE:
            continue
        reversed_badly = [t for t in reached if t.r_multiple < _CLIFF_GIVEBACK_R]
        if len(reversed_badly) / len(reached) > _CLIFF_REVERSAL_THRESHOLD:
            return target_r

    return None  # Strategy doesn't have this problem at any tested R-level


def reversal_curve(trades: List[BacktestResult]) -> List[ReversalCurvePoint]:
    """
    Continuous picture of 'trade reaches MFE level X → ends up where'.

    Gives the trader the full picture instead of one cliff number:
      "at 1.5R MFE, 15% reverse to <0.5R. At 2.0R, 38%. At 2.5R, 55%."

    The cliff jumps out visually when you plot this.
    """
    points: List[ReversalCurvePoint] = []
    if not trades:
        return points

    for r in np.arange(_CURVE_R_START, _CURVE_R_END + 1e-9, _CURVE_R_STEP):
        r = float(r)
        reached = [t for t in trades if t.max_favorable_excursion >= r]
        if len(reached) < _CURVE_MIN_SAMPLE:
            # Stop curve here — beyond this MFE level, sample too thin
            break
        reversed_badly = [t for t in reached if t.r_multiple < _CLIFF_GIVEBACK_R]
        avg_final = float(np.mean([t.r_multiple for t in reached]))
        points.append(ReversalCurvePoint(
            mfe_level=round(r, 2),
            n_reached=len(reached),
            reversal_pct=len(reversed_badly) / len(reached),
            avg_final_r=avg_final,
        ))

    return points


# ── Per-regime aggregation ──────────────────────────────────────────

def _build_recommendation(report: RegimeTargetReport) -> str:
    """One-sentence trader-facing recommendation for this regime."""
    target = report.optimal_target_r
    pf = report.optimal_profit_factor
    cliff = report.reversal_cliff_r

    # Verdict words by target zone
    if target >= 2.5:
        action = "let it run"
    elif target >= 1.5:
        action = "standard hold"
    elif target >= 1.0:
        action = "book early"
    else:
        action = "scalp only"

    cliff_note = ""
    if cliff is not None:
        # Find the matching curve point for context
        curve_pt = next(
            (p for p in report.reversal_curve if abs(p.mfe_level - cliff) < 0.01),
            None,
        )
        rev_pct = (curve_pt.reversal_pct * 100) if curve_pt else 40
        cliff_note = (
            f" — cliff at {cliff:.1f}R ({rev_pct:.0f}% reverse below 0.5R)"
        )

    return (
        f"Optimal target = {target:.1f}R (PF {pf:.2f}) — {action}{cliff_note}"
    )


def optimize_for_regime(
    regime: str,
    trades: List[BacktestResult],
) -> RegimeTargetReport:
    """
    Run the full optimizer for a single regime's trades.
    Returns a complete report with optimal target + cliff + curve + recommendation.
    """
    sims = [simulate_target(trades, r) for r in _TARGET_R_GRID]
    optimal = (
        sorted(sims, key=lambda s: (s.profit_factor, s.avg_r), reverse=True)[0]
        if sims else simulate_target([], 0.0)
    )
    cliff = find_reversal_cliff(trades)
    curve = reversal_curve(trades)

    report = RegimeTargetReport(
        regime=regime,
        sample_size=len(trades),
        optimal_target_r=optimal.target_r,
        optimal_profit_factor=optimal.profit_factor,
        optimal_avg_r=optimal.avg_r,
        optimal_win_rate=optimal.win_rate,
        reversal_cliff_r=cliff,
        target_simulations=sims,
        reversal_curve=curve,
        recommendation="",
    )
    report.recommendation = _build_recommendation(report)
    return report


def optimize_all_regimes(
    trades: List[BacktestResult],
    min_trades_per_regime: int = 15,
) -> Dict[str, RegimeTargetReport]:
    """
    Group trades by regime_at_entry, run optimize_for_regime() for each that has
    enough sample. Skips regimes with < min_trades_per_regime trades (too noisy).
    """
    by_regime: Dict[str, List[BacktestResult]] = defaultdict(list)
    for t in trades:
        regime = (t.regime_at_entry or "unknown").lower()
        by_regime[regime].append(t)

    reports: Dict[str, RegimeTargetReport] = {}
    for regime, regime_trades in by_regime.items():
        if len(regime_trades) < min_trades_per_regime:
            logger.debug(
                "Skipping regime %s — only %d trades (need %d)",
                regime, len(regime_trades), min_trades_per_regime,
            )
            continue
        reports[regime] = optimize_for_regime(regime, regime_trades)
        logger.info("Target optimizer: %s", reports[regime].short_summary())

    return reports


# ── Public façade ───────────────────────────────────────────────────

class RegimeTargetOptimizer:
    """
    Façade for the optimizer — provides a stable API for callers
    (AVT engine, co-planner, research agent).
    """

    def __init__(self, min_trades_per_regime: int = 15):
        self.min_trades_per_regime = min_trades_per_regime
        self._cache: Dict[str, Dict[str, RegimeTargetReport]] = {}

    def optimize(
        self,
        trades: List[BacktestResult],
        cache_key: Optional[str] = None,
    ) -> Dict[str, RegimeTargetReport]:
        """
        Run optimizer across all regimes. Optionally cache by a key
        (e.g., strategy_id + date) for repeated lookups.
        """
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        reports = optimize_all_regimes(trades, self.min_trades_per_regime)
        if cache_key:
            self._cache[cache_key] = reports
        return reports

    def get_target_for_regime(
        self,
        trades: List[BacktestResult],
        regime: str,
        fallback_r: float = 1.5,
    ) -> float:
        """
        Quick lookup: given current regime, what target R should AVT use?
        Falls back to fallback_r if regime isn't represented in the data.
        """
        reports = self.optimize(trades)
        regime_lower = (regime or "").lower()
        report = reports.get(regime_lower)
        return report.optimal_target_r if report else fallback_r

    def get_cliff_for_regime(
        self,
        trades: List[BacktestResult],
        regime: str,
    ) -> Optional[float]:
        """Quick lookup: where does the reversal cliff sit in this regime?"""
        reports = self.optimize(trades)
        regime_lower = (regime or "").lower()
        report = reports.get(regime_lower)
        return report.reversal_cliff_r if report else None

    def clear_cache(self) -> None:
        self._cache.clear()


# ── Module-level singleton (lazy init) ──────────────────────────────

_singleton: Optional[RegimeTargetOptimizer] = None


def get_target_optimizer() -> RegimeTargetOptimizer:
    """Get the module-level optimizer singleton."""
    global _singleton
    if _singleton is None:
        _singleton = RegimeTargetOptimizer()
    return _singleton
