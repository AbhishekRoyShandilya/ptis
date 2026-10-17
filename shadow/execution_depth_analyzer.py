"""Shadow Trade Depth Metrics — divergence tracking and pattern analysis.

Implements the five depth metrics requested in the product review:
  1. Missed Profit        — shadow hit the target, real trade exited early
  2. Exit Efficiency      — how much of the available move was captured
  3. Premature Exit Cost  — $ and R cost of exiting before the target
  4. RR Deviation         — actual realised RR vs planned RR
  5. Pattern Detection    — recurring execution problems across many trades

Called after ShadowComparisonEngine.compare() to produce depth insights
that feed directly into the coaching engine and the Intelligence Console UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  DEPTH METRIC STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExitDepthMetrics:
    """Per-trade execution depth analysis."""
    trade_id: str

    # Missed profit: real exited early, shadow hit target
    missed_profit_usd: float = 0.0      # $ left on the table
    missed_profit_r: float = 0.0        # in R-multiples
    was_premature_exit: bool = False     # shadow hit target but real did not

    # Exit efficiency: what % of the available move (entry→max_favourable) did we capture?
    exit_efficiency_pct: float = 0.0    # 0–100% (100 = captured entire favourable move)

    # RR deviation
    planned_rr: Optional[float] = None
    actual_rr: Optional[float] = None
    rr_deviation: Optional[float] = None     # actual - planned (negative = underperformed)
    rr_deviation_pct: Optional[float] = None # % shortfall vs plan

    # Premature exit cost (specific label)
    premature_exit_cost_usd: float = 0.0
    premature_exit_cost_r: float = 0.0

    # Best alternative outcome (from shadow variants)
    best_shadow_type: Optional[str] = None
    best_shadow_pnl: float = 0.0
    best_shadow_r: Optional[float] = None

    # Coaching flags
    exit_graded: str = "N/A"  # EXCELLENT / GOOD / FAIR / POOR / PREMATURE

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "missed_profit_usd": round(self.missed_profit_usd, 2),
            "missed_profit_r": round(self.missed_profit_r, 3),
            "was_premature_exit": self.was_premature_exit,
            "exit_efficiency_pct": round(self.exit_efficiency_pct, 1),
            "planned_rr": self.planned_rr,
            "actual_rr": round(self.actual_rr, 3) if self.actual_rr is not None else None,
            "rr_deviation": round(self.rr_deviation, 3) if self.rr_deviation is not None else None,
            "rr_deviation_pct": round(self.rr_deviation_pct, 1) if self.rr_deviation_pct is not None else None,
            "premature_exit_cost_usd": round(self.premature_exit_cost_usd, 2),
            "premature_exit_cost_r": round(self.premature_exit_cost_r, 3),
            "best_shadow_type": self.best_shadow_type,
            "best_shadow_pnl": round(self.best_shadow_pnl, 2),
            "best_shadow_r": round(self.best_shadow_r, 3) if self.best_shadow_r is not None else None,
            "exit_graded": self.exit_graded,
        }


@dataclass
class ExecutionPattern:
    """A recurring execution pattern detected across multiple trades."""
    pattern_id: str
    label: str
    description: str
    frequency: int           # how many trades exhibit this
    frequency_pct: float     # as % of total trades analysed
    avg_cost_usd: float      # average $ cost per occurrence
    avg_cost_r: float        # average R cost per occurrence
    total_cost_usd: float    # cumulative cost over all occurrences
    severity: str            # LOW / MEDIUM / HIGH / CRITICAL
    coaching_message: str    # what the trader should work on
    example_trade_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "label": self.label,
            "description": self.description,
            "frequency": self.frequency,
            "frequency_pct": round(self.frequency_pct, 1),
            "avg_cost_usd": round(self.avg_cost_usd, 2),
            "avg_cost_r": round(self.avg_cost_r, 3),
            "total_cost_usd": round(self.total_cost_usd, 2),
            "severity": self.severity,
            "coaching_message": self.coaching_message,
            "example_trade_ids": self.example_trade_ids[:3],
        }


@dataclass
class DivergenceReport:
    """Complete depth analysis across a set of trades."""
    n_trades_analysed: int
    metrics: List[ExitDepthMetrics] = field(default_factory=list)
    patterns: List[ExecutionPattern] = field(default_factory=list)

    # Aggregated across all trades
    avg_exit_efficiency_pct: float = 0.0
    avg_rr_deviation: float = 0.0
    total_missed_profit_usd: float = 0.0
    total_missed_profit_r: float = 0.0
    premature_exit_rate: float = 0.0   # % of trades with premature exit

    # Grade distribution
    exit_grade_distribution: Dict[str, int] = field(default_factory=dict)

    def summary_dict(self) -> Dict:
        return {
            "n_trades": self.n_trades_analysed,
            "avg_exit_efficiency_pct": round(self.avg_exit_efficiency_pct, 1),
            "avg_rr_deviation": round(self.avg_rr_deviation, 3),
            "total_missed_profit_usd": round(self.total_missed_profit_usd, 2),
            "total_missed_profit_r": round(self.total_missed_profit_r, 3),
            "premature_exit_rate_pct": round(self.premature_exit_rate * 100, 1),
            "patterns": [p.to_dict() for p in self.patterns],
            "exit_grade_distribution": self.exit_grade_distribution,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  DIVERGENCE TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class ShadowDivergenceTracker:
    """
    Computes execution depth metrics for individual trades and detects
    recurring patterns across a portfolio of trades.

    Input for each trade comes from the comparison.py ShadowComparison objects
    plus the raw trade and shadow trade data.
    """

    # ── Per-trade depth metrics ───────────────────────────────────────────────

    def compute_trade_metrics(
        self,
        trade_id: str,
        real_pnl: float,
        real_r: Optional[float],
        real_max_favourable: Optional[float],    # max unrealised profit during trade ($)
        real_exit_price: Optional[float],
        real_entry_price: float,
        real_stop_loss: float,
        real_direction: str,                     # "long" or "short"
        planned_rr: Optional[float],             # RR set at plan time
        shadow_strategy_pnl: Optional[float],    # strategy shadow outcome
        shadow_strategy_r: Optional[float],
        shadow_strategy_exit_reason: Optional[str],
        best_shadow_pnl: float,
        best_shadow_r: Optional[float],
        best_shadow_type: Optional[str],
        risk_amount: Optional[float],            # $ at risk (entry - stop) × shares
    ) -> ExitDepthMetrics:
        """
        Compute all five depth metrics for a single closed trade.

        Args:
            real_max_favourable: Highest unrealised P&L during the trade.
                                 If not tracked, pass None and efficiency will be estimated.
        """
        m = ExitDepthMetrics(trade_id=trade_id)
        m.best_shadow_pnl  = best_shadow_pnl
        m.best_shadow_r    = best_shadow_r
        m.best_shadow_type = best_shadow_type
        m.planned_rr       = planned_rr
        m.actual_rr        = real_r

        # ── 1. RR Deviation ───────────────────────────────────────────────────
        if real_r is not None and planned_rr is not None and planned_rr > 0:
            m.rr_deviation = real_r - planned_rr
            m.rr_deviation_pct = (m.rr_deviation / planned_rr) * 100

        # ── 2. Missed Profit (strategy shadow hit target, real did not) ───────
        if shadow_strategy_pnl is not None:
            strategy_better = shadow_strategy_pnl > real_pnl
            if strategy_better:
                missed = shadow_strategy_pnl - real_pnl
                m.missed_profit_usd = max(0.0, missed)
                if risk_amount and risk_amount > 0:
                    m.missed_profit_r = m.missed_profit_usd / risk_amount

            # Premature exit: strategy shadow closed via target, real closed early
            if (
                shadow_strategy_exit_reason in ("target", "take_profit", "tp_hit")
                and strategy_better
                and real_r is not None and real_r < (shadow_strategy_r or 0)
            ):
                m.was_premature_exit = True
                m.premature_exit_cost_usd = m.missed_profit_usd
                m.premature_exit_cost_r   = m.missed_profit_r

        # ── 3. Exit Efficiency ────────────────────────────────────────────────
        # Efficiency = real_pnl / max_favourable_pnl (what % of available move we captured)
        # If max_favourable not tracked, use best shadow as proxy
        available_move = real_max_favourable if real_max_favourable else best_shadow_pnl
        if available_move is not None and available_move > 0:
            m.exit_efficiency_pct = min(100.0, (real_pnl / available_move) * 100)
        elif available_move is not None and available_move < 0 and real_pnl < 0:
            # Both negative: we captured part of a losing trade — efficiency is how LITTLE we lost
            m.exit_efficiency_pct = min(100.0, (available_move / real_pnl) * 100)

        # ── 4. Grade the exit ─────────────────────────────────────────────────
        m.exit_graded = self._grade_exit(
            m.exit_efficiency_pct,
            m.was_premature_exit,
            m.rr_deviation,
            m.missed_profit_r,
        )

        return m

    def _grade_exit(
        self,
        efficiency: float,
        premature: bool,
        rr_dev: Optional[float],
        missed_r: float,
    ) -> str:
        if premature and missed_r > 0.5:
            return "PREMATURE"
        if efficiency >= 85:
            return "EXCELLENT"
        if efficiency >= 65:
            return "GOOD"
        if efficiency >= 40:
            return "FAIR"
        return "POOR"

    # ── Pattern detection ─────────────────────────────────────────────────────

    def detect_patterns(
        self,
        trade_metrics: List[ExitDepthMetrics],
        min_frequency: int = 3,
        min_frequency_pct: float = 25.0,
    ) -> List[ExecutionPattern]:
        """
        Detect recurring execution problems across a set of trades.

        A pattern is flagged when it appears in at least `min_frequency` trades
        OR in at least `min_frequency_pct` % of trades.
        """
        n = len(trade_metrics)
        if n == 0:
            return []

        patterns: List[ExecutionPattern] = []

        # ── Pattern: Chronic Premature Exit ───────────────────────────────────
        premature = [m for m in trade_metrics if m.was_premature_exit]
        if len(premature) >= min_frequency or (len(premature) / n) * 100 >= min_frequency_pct:
            avg_cost = sum(m.premature_exit_cost_usd for m in premature) / len(premature)
            avg_r    = sum(m.premature_exit_cost_r for m in premature) / len(premature)
            patterns.append(ExecutionPattern(
                pattern_id="chronic_premature_exit",
                label="Chronic Premature Exit",
                description=(
                    f"Strategy target was hit by the shadow trade in {len(premature)}/{n} trades "
                    f"({len(premature)/n*100:.0f}%) where real exit was earlier."
                ),
                frequency=len(premature),
                frequency_pct=len(premature) / n * 100,
                avg_cost_usd=avg_cost,
                avg_cost_r=avg_r,
                total_cost_usd=sum(m.premature_exit_cost_usd for m in premature),
                severity="HIGH" if avg_r > 0.5 else "MEDIUM",
                coaching_message=(
                    "You are consistently exiting before your planned target. "
                    f"Average cost: {avg_r:.2f}R per trade. "
                    "This is likely driven by fear of giving back profits. "
                    "Set a rule: only exit early if the original thesis is invalidated — "
                    "not because price stalls."
                ),
                example_trade_ids=[m.trade_id for m in premature[:3]],
            ))

        # ── Pattern: Poor Exit Efficiency ─────────────────────────────────────
        poor_eff = [m for m in trade_metrics if m.exit_efficiency_pct < 40 and m.exit_efficiency_pct >= 0]
        if len(poor_eff) >= min_frequency or (len(poor_eff) / n) * 100 >= min_frequency_pct:
            avg_eff = sum(m.exit_efficiency_pct for m in poor_eff) / len(poor_eff)
            patterns.append(ExecutionPattern(
                pattern_id="poor_exit_efficiency",
                label="Poor Exit Efficiency",
                description=(
                    f"Exit efficiency below 40% in {len(poor_eff)}/{n} trades "
                    f"({len(poor_eff)/n*100:.0f}%). "
                    f"Average efficiency: {avg_eff:.0f}%."
                ),
                frequency=len(poor_eff),
                frequency_pct=len(poor_eff) / n * 100,
                avg_cost_usd=sum(m.missed_profit_usd for m in poor_eff) / len(poor_eff),
                avg_cost_r=sum(m.missed_profit_r for m in poor_eff) / len(poor_eff),
                total_cost_usd=sum(m.missed_profit_usd for m in poor_eff),
                severity="MEDIUM",
                coaching_message=(
                    "You are systematically capturing less than 40% of the available move. "
                    "Common causes: exiting at first sign of resistance, no trailing stop, "
                    "or taking profit at arbitrary levels rather than key structure. "
                    "Review your exit methodology and add a trailing stop to let winners run."
                ),
                example_trade_ids=[m.trade_id for m in poor_eff[:3]],
            ))

        # ── Pattern: Consistent RR Underperformance ────────────────────────────
        rr_under = [m for m in trade_metrics if m.rr_deviation is not None and m.rr_deviation < -0.5]
        if len(rr_under) >= min_frequency or (len(rr_under) / n) * 100 >= min_frequency_pct:
            avg_dev = sum(m.rr_deviation for m in rr_under) / len(rr_under)
            patterns.append(ExecutionPattern(
                pattern_id="rr_underperformance",
                label="Consistent RR Underperformance",
                description=(
                    f"Actual RR fell {abs(avg_dev):.1f}R short of plan in "
                    f"{len(rr_under)}/{n} trades ({len(rr_under)/n*100:.0f}%)."
                ),
                frequency=len(rr_under),
                frequency_pct=len(rr_under) / n * 100,
                avg_cost_usd=sum(m.missed_profit_usd for m in rr_under) / len(rr_under),
                avg_cost_r=abs(avg_dev),
                total_cost_usd=sum(m.missed_profit_usd for m in rr_under),
                severity="HIGH" if abs(avg_dev) > 1.0 else "MEDIUM",
                coaching_message=(
                    f"Your actual RR ({abs(avg_dev):.1f}R below plan) is systematically worse than expected. "
                    "This erodes your edge even with a good win rate. "
                    "Root causes: targets set too aggressively OR exits triggered by emotion before target. "
                    "Review target placement — are targets at realistic structure levels?"
                ),
                example_trade_ids=[m.trade_id for m in rr_under[:3]],
            ))

        # ── Pattern: Missed Large Winners ─────────────────────────────────────
        big_misses = [m for m in trade_metrics if m.missed_profit_r > 1.0]
        if len(big_misses) >= 2:
            avg_missed = sum(m.missed_profit_r for m in big_misses) / len(big_misses)
            patterns.append(ExecutionPattern(
                pattern_id="missed_large_winners",
                label="Repeatedly Missing Large Winners",
                description=(
                    f"Exited > 1R short of optimal in {len(big_misses)} trades. "
                    f"Average missed: {avg_missed:.1f}R per trade."
                ),
                frequency=len(big_misses),
                frequency_pct=len(big_misses) / n * 100,
                avg_cost_usd=sum(m.missed_profit_usd for m in big_misses) / len(big_misses),
                avg_cost_r=avg_missed,
                total_cost_usd=sum(m.missed_profit_usd for m in big_misses),
                severity="CRITICAL" if avg_missed > 2.0 else "HIGH",
                coaching_message=(
                    f"You are leaving an average of {avg_missed:.1f}R per trade on the table. "
                    "Large winners are what make discretionary trading profitable — "
                    "cutting them short destroys your expectancy. "
                    "Consider scaling out: take 50% at 1R, trail the rest to target."
                ),
                example_trade_ids=[m.trade_id for m in big_misses[:3]],
            ))

        return sorted(patterns, key=lambda p: p.total_cost_usd, reverse=True)

    # ── Full portfolio analysis ───────────────────────────────────────────────

    def analyse_portfolio(
        self,
        trade_metrics: List[ExitDepthMetrics],
    ) -> DivergenceReport:
        """
        Aggregate metrics and detect patterns across a set of trades.
        Returns a DivergenceReport suitable for the Intelligence Console.
        """
        n = len(trade_metrics)
        report = DivergenceReport(
            n_trades_analysed=n,
            metrics=trade_metrics,
        )

        if n == 0:
            return report

        # Aggregates
        report.avg_exit_efficiency_pct = sum(
            m.exit_efficiency_pct for m in trade_metrics
        ) / n

        valid_rr_devs = [m.rr_deviation for m in trade_metrics if m.rr_deviation is not None]
        if valid_rr_devs:
            report.avg_rr_deviation = sum(valid_rr_devs) / len(valid_rr_devs)

        report.total_missed_profit_usd = sum(m.missed_profit_usd for m in trade_metrics)
        report.total_missed_profit_r   = sum(m.missed_profit_r for m in trade_metrics)
        report.premature_exit_rate     = sum(1 for m in trade_metrics if m.was_premature_exit) / n

        # Grade distribution
        for m in trade_metrics:
            report.exit_grade_distribution[m.exit_graded] = (
                report.exit_grade_distribution.get(m.exit_graded, 0) + 1
            )

        # Pattern detection
        report.patterns = self.detect_patterns(trade_metrics)

        return report

    # ── Coaching summary ──────────────────────────────────────────────────────

    def coaching_summary(self, report: DivergenceReport) -> List[str]:
        """
        Generate prioritised coaching bullet points from the divergence report.
        Sorted by financial impact.
        """
        bullets = []

        if report.premature_exit_rate > 0.3:
            bullets.append(
                f"You exit early in {report.premature_exit_rate*100:.0f}% of trades. "
                f"Total cost: ${report.total_missed_profit_usd:.0f} "
                f"({report.total_missed_profit_r:.1f}R)."
            )

        if report.avg_rr_deviation < -0.5:
            bullets.append(
                f"Average RR is {abs(report.avg_rr_deviation):.1f}R below plan. "
                "Your exits are systematically underperforming your entry plan."
            )

        if report.avg_exit_efficiency_pct < 50:
            bullets.append(
                f"Exit efficiency averages {report.avg_exit_efficiency_pct:.0f}% — "
                "you capture less than half the available move on most trades."
            )

        for pattern in report.patterns[:2]:  # top 2 patterns by cost
            bullets.append(
                f"{pattern.label}: {pattern.description} "
                f"Total cost: ${pattern.total_cost_usd:.0f}."
            )

        return bullets
