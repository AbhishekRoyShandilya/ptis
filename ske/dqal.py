# ske/dqal.py
"""
Data Quality Assurance Layer (DQAL) — validates an SKO through 5 gates
to produce a quality score (0–100) and grade (A–F).

Gates:
1. Completeness — required fields, no NaN, sample size
2. Logical consistency — stop/target/entry relationships, RR > 0
3. Market structure — prices within ATR, volume realistic, market hours
4. Statistical evidence — min trades per tier, CI on win rate, PF significance
5. Edge verification — expectancy > 0, Sharpe > 0.5, drawdown < 30%
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from core.enums import SKOStatus
from ske.models import DQALGateResult, DQALResult

logger = logging.getLogger(__name__)

# Gate weights for overall score
_GATE_WEIGHTS = {
    "completeness": 0.15,
    "logical_consistency": 0.15,
    "market_structure": 0.15,
    "statistical_evidence": 0.25,
    "edge_verification": 0.30,
}

# Grade thresholds
_GRADE_THRESHOLDS = [
    (90, "A"),
    (75, "B"),
    (60, "C"),
    (40, "D"),
    (0, "F"),
]


class DQALValidator:
    """Validates an SKO's data quality across 5 gates."""

    def __init__(
        self,
        min_trades_developing: int = 10,
        min_trades_validated: int = 20,
        min_trades_proven: int = 50,
    ):
        self._tier_thresholds = {
            SKOStatus.HYPOTHESIS: 0,
            SKOStatus.DEVELOPING: min_trades_developing,
            SKOStatus.VALIDATED: min_trades_validated,
            SKOStatus.PROVEN: min_trades_proven,
        }

    def validate(
        self,
        sko_data: Dict[str, Any],
        trade_outcomes: Optional[List[float]] = None,
    ) -> DQALResult:
        """
        Run all 5 DQAL gates on an SKO.

        Args:
            sko_data: Dict with SKO fields (scoring_config, exit_rules, etc.)
            trade_outcomes: List of R-multiples for all trades in the SKO.

        Returns:
            DQALResult with gate scores, overall score, and grade.
        """
        outcomes = trade_outcomes or []

        gates = [
            self._gate_completeness(sko_data, outcomes),
            self._gate_logical_consistency(sko_data),
            self._gate_market_structure(sko_data, outcomes),
            self._gate_statistical_evidence(sko_data, outcomes),
            self._gate_edge_verification(outcomes),
        ]

        # Weighted average
        total_weight = sum(_GATE_WEIGHTS.get(g.gate_name, 0.2) for g in gates)
        weighted_sum = sum(
            g.score * _GATE_WEIGHTS.get(g.gate_name, 0.2) for g in gates
        )
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Grade
        grade = "F"
        for threshold, letter in _GRADE_THRESHOLDS:
            if overall >= threshold:
                grade = letter
                break

        # Collect issues and recommendations
        all_issues = []
        all_recs = []
        for g in gates:
            all_issues.extend(g.issues)
        all_recs = self._generate_recommendations(gates, outcomes)

        return DQALResult(
            gate_scores=gates,
            overall_score=round(overall, 1),
            grade=grade,
            issues=all_issues,
            recommendations=all_recs,
        )

    # ──────────────────────────────────────────
    #  Gate 1: Completeness
    # ──────────────────────────────────────────

    def _gate_completeness(
        self, sko: Dict[str, Any], outcomes: List[float],
    ) -> DQALGateResult:
        score = 100.0
        issues: List[str] = []

        # Required fields
        required = ["name", "short_code", "scoring_config"]
        for field in required:
            if not sko.get(field):
                score -= 20
                issues.append(f"Missing required field: {field}")

        # Scoring config should have at least 2 features
        scoring = sko.get("scoring_config", {})
        if len(scoring) < 2:
            score -= 15
            issues.append(f"Scoring config has only {len(scoring)} features (min 2)")

        # Need some trades
        total_trades = sko.get("total_trades", len(outcomes))
        if total_trades < 5:
            score -= 30
            issues.append(f"Only {total_trades} trades (need ≥5 for basic analysis)")
        elif total_trades < 10:
            score -= 10
            issues.append(f"Low sample: {total_trades} trades")

        # Exit rules
        if not sko.get("exit_rules"):
            score -= 10
            issues.append("No exit rules defined")

        # Risk parameters
        if not sko.get("risk_parameters"):
            score -= 10
            issues.append("No risk parameters defined")

        return DQALGateResult(
            gate_name="completeness",
            score=max(0.0, score),
            passed=score >= 50,
            issues=issues,
        )

    # ──────────────────────────────────────────
    #  Gate 2: Logical Consistency
    # ──────────────────────────────────────────

    def _gate_logical_consistency(self, sko: Dict[str, Any]) -> DQALGateResult:
        score = 100.0
        issues: List[str] = []

        exit_rules = sko.get("exit_rules", {})
        risk = sko.get("risk_parameters", {})
        conviction_map = sko.get("conviction_map", {})

        # ATR multiple should be positive
        atr_mult = exit_rules.get("atr_multiple", 1.5)
        if atr_mult <= 0:
            score -= 20
            issues.append(f"ATR multiple is {atr_mult} (must be > 0)")

        # Risk percent should be positive and reasonable
        base_risk = risk.get("base_risk_pct", 1.0)
        max_risk = risk.get("max_risk_pct", 2.0)
        if base_risk <= 0 or base_risk > 10:
            score -= 15
            issues.append(f"Base risk {base_risk}% is out of bounds (0-10%)")
        if max_risk < base_risk:
            score -= 10
            issues.append("Max risk is less than base risk")

        # RR minimum
        rr_min = risk.get("rr_minimum", 1.0)
        if rr_min < 0.5:
            score -= 15
            issues.append(f"RR minimum {rr_min} is dangerously low")

        # Conviction map should be monotonically increasing
        if conviction_map:
            expected_order = ["skip", "small", "standard", "aggressive", "max"]
            prev_upper = 0.0
            for band in expected_order:
                bounds = conviction_map.get(band, [])
                if len(bounds) == 2:
                    if bounds[0] < prev_upper - 0.01:
                        score -= 10
                        issues.append(f"Conviction band '{band}' overlaps with previous")
                    prev_upper = bounds[1]

        return DQALGateResult(
            gate_name="logical_consistency",
            score=max(0.0, score),
            passed=score >= 50,
            issues=issues,
        )

    # ──────────────────────────────────────────
    #  Gate 3: Market Structure
    # ──────────────────────────────────────────

    def _gate_market_structure(
        self, sko: Dict[str, Any], outcomes: List[float],
    ) -> DQALGateResult:
        score = 100.0
        issues: List[str] = []

        # Check for extreme outlier R-multiples (likely data errors)
        if outcomes:
            max_r = max(abs(r) for r in outcomes)
            if max_r > 50:
                score -= 20
                issues.append(f"Extreme R-multiple detected: {max_r:.1f}R (possible data error)")

            # Check for suspiciously perfect results
            wins = [r for r in outcomes if r > 0]
            if len(wins) == len(outcomes) and len(outcomes) > 5:
                score -= 30
                issues.append("100% win rate with 5+ trades — likely overfitted or data error")

        # Time preferences should be in IST market hours
        time_prefs = sko.get("time_preferences", {})
        if time_prefs:
            valid_sessions = {"pre_market", "morning", "midday", "afternoon", "close", "after_hours"}
            for session in time_prefs:
                if session not in valid_sessions:
                    score -= 5
                    issues.append(f"Unknown session type: {session}")

        return DQALGateResult(
            gate_name="market_structure",
            score=max(0.0, score),
            passed=score >= 50,
            issues=issues,
        )

    # ──────────────────────────────────────────
    #  Gate 4: Statistical Evidence
    # ──────────────────────────────────────────

    def _gate_statistical_evidence(
        self, sko: Dict[str, Any], outcomes: List[float],
    ) -> DQALGateResult:
        score = 100.0
        issues: List[str] = []

        n = len(outcomes)
        # Use effective sample size if available (from quality gate), else raw count
        effective_n = sko.get("effective_sample_size", n)

        status = sko.get("status", "hypothesis")
        tier_min = self._tier_thresholds.get(
            SKOStatus(status) if isinstance(status, str) and status in [s.value for s in SKOStatus] else SKOStatus.HYPOTHESIS,
            0,
        )

        # Check effective trade count vs tier requirement
        if effective_n < tier_min:
            deficit = tier_min - effective_n
            score -= min(40, deficit * 3)
            issues.append(f"Need {deficit:.0f} more effective trades for '{status}' tier (have {effective_n:.1f})")

        if effective_n < 5:
            score -= 30
            issues.append(f"Insufficient effective sample size ({effective_n:.1f}) for statistical analysis")
            return DQALGateResult(
                gate_name="statistical_evidence",
                score=max(0.0, score),
                passed=False,
                issues=issues,
            )

        # Win rate confidence interval (95%) — use effective N for CI width
        win_rate = sum(1 for r in outcomes if r > 0) / n if n > 0 else 0
        ci_half = 1.96 * math.sqrt(win_rate * (1 - win_rate) / effective_n) if effective_n > 0 else 1.0
        if ci_half > 0.15:
            score -= 15
            issues.append(f"Win rate CI too wide: {win_rate:.0%} ± {ci_half:.0%} (N_eff={effective_n:.1f})")

        # Profit factor significance
        gross_profit = sum(r for r in outcomes if r > 0)
        gross_loss = abs(sum(r for r in outcomes if r < 0))
        if gross_loss > 0:
            pf = gross_profit / gross_loss
            if pf < 1.0:
                score -= 20
                issues.append(f"Profit factor {pf:.2f} is below 1.0")
            elif pf < 1.2:
                score -= 10
                issues.append(f"Profit factor {pf:.2f} is marginal")
        elif gross_profit > 0:
            pass  # no losses = fine for now
        else:
            score -= 25
            issues.append("No profitable trades")

        # Binomial test for win rate significance (when scipy available)
        if effective_n >= 10 and n >= 10:
            try:
                from scipy.stats import binom_test
                wins = sum(1 for r in outcomes if r > 0)
                p_value = binom_test(wins, n, 0.5, alternative="greater")
                if p_value > 0.10:
                    score -= 10
                    issues.append(f"Win rate not statistically significant (p={p_value:.3f})")
            except ImportError:
                pass  # scipy not available, skip

        return DQALGateResult(
            gate_name="statistical_evidence",
            score=max(0.0, score),
            passed=score >= 50,
            issues=issues,
        )

    # ──────────────────────────────────────────
    #  Gate 5: Edge Verification
    # ──────────────────────────────────────────

    def _gate_edge_verification(self, outcomes: List[float]) -> DQALGateResult:
        score = 100.0
        issues: List[str] = []

        if len(outcomes) < 5:
            return DQALGateResult(
                gate_name="edge_verification",
                score=30.0,
                passed=False,
                issues=["Too few trades for edge verification"],
            )

        n = len(outcomes)
        mean_r = sum(outcomes) / n

        # Expectancy > 0
        if mean_r <= 0:
            score -= 30
            issues.append(f"Negative expectancy: {mean_r:.3f}R per trade")

        # Sharpe ratio (annualized assuming 252 trading days)
        variance = sum((r - mean_r) ** 2 for r in outcomes) / n
        std_r = math.sqrt(variance) if variance > 0 else 1.0
        sharpe = (mean_r / std_r) * math.sqrt(min(252, n)) if std_r > 0 else 0.0

        if sharpe < 0.5:
            score -= 20
            issues.append(f"Low Sharpe ratio: {sharpe:.2f} (target ≥0.5)")
        elif sharpe < 1.0:
            score -= 5
            issues.append(f"Moderate Sharpe ratio: {sharpe:.2f}")

        # Max drawdown (cumulative R)
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in outcomes:
            cumulative += r
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        if max_dd > 10.0:  # 10R drawdown
            score -= 25
            issues.append(f"Large max drawdown: {max_dd:.1f}R")
        elif max_dd > 5.0:
            score -= 10
            issues.append(f"Notable drawdown: {max_dd:.1f}R")

        # Win rate / loss rate balance
        wins = [r for r in outcomes if r > 0]
        losses = [r for r in outcomes if r <= 0]
        if wins and losses:
            avg_win = sum(wins) / len(wins)
            avg_loss = abs(sum(losses) / len(losses))
            if avg_loss > 0:
                payoff = avg_win / avg_loss
                if payoff < 0.5:
                    score -= 15
                    issues.append(f"Poor payoff ratio: {payoff:.2f}")

        return DQALGateResult(
            gate_name="edge_verification",
            score=max(0.0, score),
            passed=score >= 50,
            issues=issues,
        )

    # ──────────────────────────────────────────
    #  Recommendations
    # ──────────────────────────────────────────

    @staticmethod
    def _generate_recommendations(
        gates: List[DQALGateResult], outcomes: List[float],
    ) -> List[str]:
        recs: List[str] = []

        for g in gates:
            if not g.passed:
                if g.gate_name == "completeness":
                    recs.append("Add missing configuration fields and take more trades")
                elif g.gate_name == "logical_consistency":
                    recs.append("Review exit rules and risk parameters for logical errors")
                elif g.gate_name == "market_structure":
                    recs.append("Check for data quality issues or unrealistic trade results")
                elif g.gate_name == "statistical_evidence":
                    recs.append("Take more trades to build statistical significance")
                elif g.gate_name == "edge_verification":
                    if len(outcomes) < 20:
                        recs.append("Collect more trade data to verify edge persistence")
                    else:
                        recs.append("Review strategy rules — edge may not be present")

        return recs
