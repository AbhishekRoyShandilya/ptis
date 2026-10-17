# improvement_lab/tradeoff_simulator.py
"""
Tradeoff Simulator & Composition Tester
=========================================

Every improvement has a COST.

"Book partial at 1:2" captures profit on reversals BUT reduces profit
on runners. The system must quantify BOTH sides.

TradeoffSimulator:
  - Replays historical trades WITH the proposed improvement
  - Computes trades_improved, trades_hurt, net expectancy change
  - Analyzes tail/runner impact
  - Generates alternative parameter variants

CompositionTester:
  - Tests improvements individually AND in combination
  - Detects harmful interactions (antagonistic effects)
  - Finds optimal subset combination
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from improvement_lab.models import (
    AlternativeImprovement,
    CompositionResult,
    ImprovementHypothesis,
    ImprovementType,
    InteractionEffect,
    TradeoffAnalysis,
)

logger = logging.getLogger(__name__)


class TradeoffSimulator:
    """
    Simulates improvements against historical trades.

    For each improvement type, applies the proposed rule to historical
    trades and computes what would have happened.
    """

    def simulate(
        self,
        hypothesis: ImprovementHypothesis,
        trades: List[Any],
    ) -> TradeoffAnalysis:
        """
        Replay historical trades with the proposed improvement applied.
        Returns full tradeoff analysis.
        """
        if not trades:
            return TradeoffAnalysis(recommendation="insufficient_data")

        imp_type = hypothesis.improvement_type
        rule = hypothesis.proposed_rule or hypothesis.evidence

        if imp_type == ImprovementType.RR_KILLER_FIX.value:
            return self._simulate_partial_exit(trades, rule)
        elif imp_type == ImprovementType.EARLY_EXIT_FILTER.value:
            return self._simulate_early_exit(trades, rule)
        elif imp_type in (
            ImprovementType.ENTRY_FILTER.value,
            ImprovementType.REGIME_FILTER.value,
            ImprovementType.VARIANT_AVOIDANCE.value,
        ):
            return self._simulate_entry_filter(trades, rule)
        else:
            return self._simulate_generic(trades, rule)

    def simulate_multiple(
        self,
        hypotheses: List[ImprovementHypothesis],
        trades: List[Any],
    ) -> TradeoffAnalysis:
        """
        Simulate multiple improvements applied TOGETHER.
        Used for composition testing.
        """
        if not trades or not hypotheses:
            return TradeoffAnalysis(recommendation="insufficient_data")

        original_outcomes = []
        modified_outcomes = []

        for trade in trades:
            original_r = self._get_r(trade)
            modified_r = original_r

            # Apply each improvement in sequence
            for hyp in hypotheses:
                imp_type = hyp.improvement_type
                rule = hyp.proposed_rule or hyp.evidence

                if imp_type in (
                    ImprovementType.ENTRY_FILTER.value,
                    ImprovementType.REGIME_FILTER.value,
                    ImprovementType.VARIANT_AVOIDANCE.value,
                ):
                    # Entry filter: if matched, trade is skipped (R=0)
                    if self._trade_matches_filter(trade, rule):
                        modified_r = 0.0
                        break  # trade skipped, no further rules apply

                elif imp_type == ImprovementType.RR_KILLER_FIX.value:
                    # Partial exit at threshold
                    trigger_r = rule.get("trigger_r", 2.0)
                    partial_pct = rule.get("partial_pct", 0.5)
                    max_r = self._get_max_favorable_r(trade)
                    if max_r and max_r >= trigger_r:
                        modified_r = self._apply_partial_exit(
                            original_r, trigger_r, partial_pct, max_r,
                        )

                elif imp_type == ImprovementType.EARLY_EXIT_FILTER.value:
                    # Exit when warning signals appear
                    if self._matches_early_exit_condition(trade, rule):
                        exit_r = rule.get("exit_r", -0.3)
                        if original_r < exit_r:
                            modified_r = exit_r  # would have exited earlier

            original_outcomes.append(original_r)
            modified_outcomes.append(modified_r)

        return self._compute_tradeoff(original_outcomes, modified_outcomes, trades)

    # ── Improvement-specific simulations ──────────────────────────

    def _simulate_partial_exit(
        self, trades: List[Any], rule: Dict,
    ) -> TradeoffAnalysis:
        """Simulate partial exit at R threshold."""
        trigger_r = rule.get("trigger_r", 2.0)
        partial_pct = rule.get("partial_pct", 0.5)

        original = []
        modified = []

        for trade in trades:
            orig_r = self._get_r(trade)
            max_r = self._get_max_favorable_r(trade)
            original.append(orig_r)

            if max_r and max_r >= trigger_r:
                mod_r = self._apply_partial_exit(orig_r, trigger_r, partial_pct, max_r)
                modified.append(mod_r)
            else:
                modified.append(orig_r)

        analysis = self._compute_tradeoff(original, modified, trades)

        # Generate alternatives with different partial %s and trigger Rs
        alternatives = []
        for pct in [0.33, 0.50, 0.67]:
            for tr in [1.5, 2.0, 2.5, 3.0]:
                if pct == partial_pct and tr == trigger_r:
                    continue
                alt_modified = []
                for i, trade in enumerate(trades):
                    max_r = self._get_max_favorable_r(trade)
                    if max_r and max_r >= tr:
                        alt_modified.append(
                            self._apply_partial_exit(original[i], tr, pct, max_r)
                        )
                    else:
                        alt_modified.append(original[i])

                alt_analysis = self._compute_tradeoff(original, alt_modified, trades)
                if alt_analysis.net_expectancy_change_r > 0:
                    alternatives.append(AlternativeImprovement(
                        description=f"Exit {int(pct*100)}% at {tr}R",
                        net_r=alt_analysis.net_expectancy_change_r,
                        runner_cost_pct=alt_analysis.pct_profits_from_runners,
                        params={"trigger_r": tr, "partial_pct": pct},
                    ))

        alternatives.sort(key=lambda a: a.net_r, reverse=True)
        analysis.alternatives = alternatives[:5]
        return analysis

    def _simulate_early_exit(
        self, trades: List[Any], rule: Dict,
    ) -> TradeoffAnalysis:
        """Simulate early exit when warning conditions are met."""
        exit_r = rule.get("exit_r", -0.3)
        original = []
        modified = []

        for trade in trades:
            orig_r = self._get_r(trade)
            original.append(orig_r)

            if self._matches_early_exit_condition(trade, rule):
                # Would have exited at exit_r instead of full SL
                modified.append(max(orig_r, exit_r))
            else:
                modified.append(orig_r)

        return self._compute_tradeoff(original, modified, trades)

    def _simulate_entry_filter(
        self, trades: List[Any], rule: Dict,
    ) -> TradeoffAnalysis:
        """Simulate skipping trades that match filter condition."""
        original = []
        modified = []

        for trade in trades:
            orig_r = self._get_r(trade)
            original.append(orig_r)

            if self._trade_matches_filter(trade, rule):
                modified.append(0.0)  # trade skipped
            else:
                modified.append(orig_r)

        analysis = self._compute_tradeoff(original, modified, trades)

        # Generate tighter/looser filter alternatives
        # (depends on the specific filter conditions)
        return analysis

    def _simulate_generic(
        self, trades: List[Any], rule: Dict,
    ) -> TradeoffAnalysis:
        """Generic simulation — no modification applied."""
        original = [self._get_r(t) for t in trades]
        return self._compute_tradeoff(original, original, trades)

    # ── Core tradeoff computation ─────────────────────────────────

    def _compute_tradeoff(
        self,
        original: List[float],
        modified: List[float],
        trades: List[Any],
    ) -> TradeoffAnalysis:
        """Compute full tradeoff analysis from original vs modified outcomes."""
        n = len(original)
        if n == 0:
            return TradeoffAnalysis(recommendation="no_data")

        orig = np.array(original)
        mod = np.array(modified)
        diff = mod - orig

        improved_mask = diff > 0.1
        hurt_mask = diff < -0.1
        unchanged_mask = ~(improved_mask | hurt_mask)

        trades_improved = int(improved_mask.sum())
        trades_hurt = int(hurt_mask.sum())

        improvement_on_fixed = float(diff[improved_mask].mean()) if trades_improved > 0 else 0.0
        cost_on_hurt = float(abs(diff[hurt_mask].mean())) if trades_hurt > 0 else 0.0

        net_change = float(mod.mean() - orig.mean())
        orig_expectancy = float(orig.mean())
        net_change_pct = (net_change / abs(orig_expectancy) * 100) if orig_expectancy != 0 else 0.0

        # Win rate change
        orig_wr = float((orig > 0).sum() / max(n, 1))
        mod_wr = float((mod > 0).sum() / max(n, 1))

        # Avg winner/loser change
        orig_winners = orig[orig > 0]
        mod_winners = mod[mod > 0]
        orig_losers = orig[orig < 0]
        mod_losers = mod[mod < 0]

        avg_winner_change = (
            (float(mod_winners.mean()) - float(orig_winners.mean()))
            if len(orig_winners) > 0 and len(mod_winners) > 0 else 0.0
        )
        avg_loser_change = (
            (float(mod_losers.mean()) - float(orig_losers.mean()))
            if len(orig_losers) > 0 and len(mod_losers) > 0 else 0.0
        )

        # Profit factor
        orig_pf = float(orig_winners.sum() / max(abs(orig_losers.sum()), 0.01)) if len(orig_winners) > 0 else 0.0
        mod_pf = float(mod_winners.sum() / max(abs(mod_losers.sum()), 0.01)) if len(mod_winners) > 0 else 0.0

        # Runner / tail impact
        best_orig = float(orig.max()) if n > 0 else 0.0
        best_mod = float(mod.max()) if n > 0 else 0.0

        # Top 10% contribute how much of total profit
        sorted_orig = np.sort(orig)[::-1]
        top_10_count = max(1, n // 10)
        total_profit = float(orig[orig > 0].sum()) if (orig > 0).any() else 1.0
        runner_profit = float(sorted_orig[:top_10_count].sum())
        pct_from_runners = runner_profit / max(total_profit, 0.01)

        # How much do runners lose with the improvement?
        sorted_mod = np.sort(mod)[::-1]
        mod_runner_profit = float(sorted_mod[:top_10_count].sum())
        runner_loss_pct = (
            (runner_profit - mod_runner_profit) / max(runner_profit, 0.01)
        ) if runner_profit > 0 else 0.0

        runner_impact = (
            f"Top {top_10_count} trades contribute {pct_from_runners:.0%} of profits. "
            f"This improvement {'reduces' if runner_loss_pct > 0 else 'preserves'} "
            f"their contribution by {abs(runner_loss_pct):.0%}."
        )

        # Confidence interval via bootstrap
        ci_low, ci_high = self._bootstrap_ci(diff, n_bootstrap=500)

        # Probability net positive
        prob_positive = float((diff > 0).sum() / max(n, 1))

        # Recommendation
        if net_change > 0.15 and prob_positive > 0.55:
            rec = (
                f"RECOMMEND: Net positive by {net_change:.2f}R per trade. "
                f"Improves {trades_improved} trades, hurts {trades_hurt}."
            )
        elif net_change > 0 and prob_positive > 0.5:
            rec = (
                f"MARGINAL: Small positive of {net_change:.2f}R per trade. "
                f"Consider if the {runner_loss_pct:.0%} runner cost is acceptable."
            )
        else:
            rec = (
                f"NOT RECOMMENDED: Net {net_change:+.2f}R per trade. "
                f"Costs outweigh benefits."
            )

        return TradeoffAnalysis(
            trades_improved=trades_improved,
            improvement_on_fixed_trades_r=round(improvement_on_fixed, 3),
            trades_hurt=trades_hurt,
            cost_on_hurt_trades_r=round(cost_on_hurt, 3),
            net_expectancy_change_r=round(net_change, 3),
            net_expectancy_change_pct=round(net_change_pct, 1),
            win_rate_change=round(mod_wr - orig_wr, 3),
            avg_winner_change_r=round(avg_winner_change, 3),
            avg_loser_change_r=round(avg_loser_change, 3),
            profit_factor_change=round(mod_pf - orig_pf, 3),
            best_trade_r_with=round(best_mod, 2),
            best_trade_r_without=round(best_orig, 2),
            pct_profits_from_runners=round(pct_from_runners, 3),
            runner_impact=runner_impact,
            sample_size=n,
            confidence_interval_net_r=(round(ci_low, 3), round(ci_high, 3)),
            probability_net_positive=round(prob_positive, 3),
            recommendation=rec,
        )

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _apply_partial_exit(
        final_r: float, trigger_r: float, partial_pct: float, max_r: float,
    ) -> float:
        """
        Compute modified R with partial exit.
        partial_pct booked at trigger_r, rest runs to final_r.
        """
        booked = trigger_r * partial_pct
        remaining = final_r * (1 - partial_pct)
        return booked + remaining

    @staticmethod
    def _get_r(trade: Any) -> float:
        if isinstance(trade, dict):
            val = trade.get("r_multiple", 0)
        else:
            val = getattr(trade, "r_multiple", 0)
        try:
            return float(val or 0)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _get_max_favorable_r(trade: Any) -> Optional[float]:
        """Get max favorable excursion in R."""
        for field in ["max_favorable_r", "max_favorable_excursion_r", "mfe_r"]:
            if isinstance(trade, dict):
                val = trade.get(field)
            else:
                val = getattr(trade, field, None)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None

    def _trade_matches_filter(self, trade: Any, rule: Dict) -> bool:
        """Check if a trade matches an entry filter condition."""
        condition = rule.get("condition", rule)
        for key, expected in condition.items():
            if key in ("type", "action", "impact", "interaction"):
                continue

            if isinstance(trade, dict):
                actual = trade.get(key)
            else:
                actual = getattr(trade, key, None)

            if actual is None:
                continue

            if hasattr(actual, "value"):
                actual = actual.value

            if isinstance(expected, str):
                if str(actual).lower() != expected.lower():
                    return False
            elif isinstance(expected, (int, float)):
                try:
                    if abs(float(actual) - expected) > 0.1:
                        return False
                except (ValueError, TypeError):
                    return False

        return True

    def _matches_early_exit_condition(self, trade: Any, rule: Dict) -> bool:
        """Check if a trade matches early exit warning conditions."""
        condition = rule.get("condition", rule)
        return self._trade_matches_filter(trade, condition)

    @staticmethod
    def _bootstrap_ci(
        diffs: np.ndarray, n_bootstrap: int = 500, alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """Bootstrap 95% confidence interval for the mean difference."""
        if len(diffs) < 5:
            return (float(diffs.min()), float(diffs.max())) if len(diffs) > 0 else (0.0, 0.0)

        rng = np.random.RandomState(42)
        means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(diffs, size=len(diffs), replace=True)
            means.append(float(sample.mean()))

        means = sorted(means)
        low_idx = int(n_bootstrap * (alpha / 2))
        high_idx = int(n_bootstrap * (1 - alpha / 2))
        return (means[low_idx], means[min(high_idx, len(means) - 1)])


class CompositionTester:
    """
    Tests multiple improvements individually AND together.
    Detects harmful interactions between improvements.
    """

    def __init__(self):
        self._simulator = TradeoffSimulator()

    def test_composition(
        self,
        hypotheses: List[ImprovementHypothesis],
        trades: List[Any],
    ) -> CompositionResult:
        """
        Test improvements individually and in all pairwise combinations.
        Detect synergistic/antagonistic interactions.
        Find optimal subset.
        """
        if not hypotheses or not trades:
            return CompositionResult()

        results: Dict[str, TradeoffAnalysis] = {}

        # 1. Test each independently
        for hyp in hypotheses:
            results[hyp.id] = self._simulator.simulate(hyp, trades)

        # 2. Test pairwise combinations
        interactions = []
        for hyp_a, hyp_b in combinations(hypotheses, 2):
            combo_key = f"{hyp_a.id}+{hyp_b.id}"
            combined = self._simulator.simulate_multiple([hyp_a, hyp_b], trades)
            results[combo_key] = combined

            individual_sum = (
                results[hyp_a.id].net_expectancy_change_r +
                results[hyp_b.id].net_expectancy_change_r
            )
            interaction_effect = combined.net_expectancy_change_r - individual_sum

            if abs(interaction_effect) > 0.1:
                interactions.append(InteractionEffect(
                    improvement_a_id=hyp_a.id,
                    improvement_b_id=hyp_b.id,
                    individual_sum_r=round(individual_sum, 3),
                    combined_r=round(combined.net_expectancy_change_r, 3),
                    interaction_r=round(interaction_effect, 3),
                    interaction_type="synergistic" if interaction_effect > 0 else "antagonistic",
                    note=(
                        f"{'Positive' if interaction_effect > 0 else 'Negative'} "
                        f"interaction of {abs(interaction_effect):.2f}R"
                    ),
                ))

        # 3. Test full combination if >2
        if len(hypotheses) > 2:
            full_key = "+".join(h.id for h in hypotheses)
            results[full_key] = self._simulator.simulate_multiple(hypotheses, trades)

        # 4. Find optimal subset
        best_combo, best_r = self._find_best_subset(hypotheses, results, trades)

        explanation = self._build_explanation(hypotheses, results, interactions, best_combo)

        return CompositionResult(
            individual_results=results,
            interactions=interactions,
            recommended_combination=[h.id for h in best_combo],
            combined_net_r=round(best_r, 3),
            explanation=explanation,
        )

    def _find_best_subset(
        self,
        hypotheses: List[ImprovementHypothesis],
        results: Dict[str, TradeoffAnalysis],
        trades: List[Any],
    ) -> Tuple[List[ImprovementHypothesis], float]:
        """Find the subset of improvements with highest combined net R."""
        best_combo: List[ImprovementHypothesis] = []
        best_r = 0.0

        # Test all subsets (reasonable for up to ~6 improvements)
        for size in range(1, min(len(hypotheses) + 1, 7)):
            for combo in combinations(hypotheses, size):
                combo_key = "+".join(h.id for h in combo)
                if combo_key in results:
                    r = results[combo_key].net_expectancy_change_r
                elif len(combo) == 1:
                    r = results[combo[0].id].net_expectancy_change_r
                else:
                    combined = self._simulator.simulate_multiple(list(combo), trades)
                    r = combined.net_expectancy_change_r

                if r > best_r:
                    best_r = r
                    best_combo = list(combo)

        return best_combo, best_r

    @staticmethod
    def _build_explanation(
        hypotheses: List[ImprovementHypothesis],
        results: Dict[str, TradeoffAnalysis],
        interactions: List[InteractionEffect],
        best_combo: List[ImprovementHypothesis],
    ) -> str:
        """Build human-readable explanation of composition results."""
        lines = []

        # Individual results
        for h in hypotheses:
            r = results.get(h.id)
            if r:
                lines.append(
                    f"- {h.title}: {r.net_expectancy_change_r:+.2f}R/trade "
                    f"({r.trades_improved} improved, {r.trades_hurt} hurt)"
                )

        # Interactions
        if interactions:
            lines.append("")
            lines.append("Interactions:")
            for ix in interactions:
                lines.append(
                    f"  {ix.interaction_type}: {ix.interaction_r:+.2f}R "
                    f"(individual sum={ix.individual_sum_r:.2f}, "
                    f"combined={ix.combined_r:.2f})"
                )

        # Recommendation
        if best_combo:
            combo_names = [h.title for h in best_combo]
            lines.append("")
            lines.append(f"Recommended: {', '.join(combo_names)}")

        return "\n".join(lines)
