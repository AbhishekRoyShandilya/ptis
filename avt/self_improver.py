# avt/self_improver.py (core flow)
"""
AVT Self-Improver
==================

The bridge between Improvement Lab (analysis) and AVT Engine
(execution).

Flow:
1. After market close → ask lab to analyze AVT's trades
2. Lab returns actionable hypotheses
3. Self-improver applies top hypothesis to AVT
4. AVT trades with improvement → accumulates proof
5. Weekly: evaluate proof → keep or revert
6. When proven → notify competition engine → user sees it

RULES:
- Max 1 active unproven improvement per strategy at a time
  (so we can isolate what's working)
- Must have MIN_PROOF_TRADES before evaluating
- Must have MIN_PROOF_WEEKS before evaluating
- Revert immediately if delta < PROOF_REVERT_THRESHOLD

DOMAIN SEPARATION RULES:
- AVT improvements stay inside AVT until proven
- Proven improvements become SUGGESTIONS to user, never auto-applied
- AVT NEVER modifies StrategyProfile (user intent)
- StrategyProfile = USER MIND, AVT = MARKET TRUTH
- They must NEVER overwrite each other
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from improvement_lab.models import (
    ImprovementHypothesis,
    ImprovementLabReport,
    ProofStatus,
)
from improvement_lab.orchestrator import ImprovementLabOrchestrator

# Optional: ContrastLearner for contrast-based edge insights
try:
    from avt.contrast_learner import ContrastLearner
    _CLE_AVAILABLE = True
except ImportError:
    _CLE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Proof requirements
MIN_PROOF_TRADES = 50
MIN_PROOF_WEEKS = 3
MIN_PROOF_DELTA_R = 0.05        # must improve by at least 0.05R/trade
REVERT_THRESHOLD_R = -0.10      # revert if hurting by 0.10R/trade
MAX_ACTIVE_IMPROVEMENTS = 1     # per strategy (for isolation)


class AVTSelfImprover:
    """
    Manages the prove-it-yourself cycle.

    The improvement lab discovers hypotheses.
    The self-improver applies them to AVT.
    AVT trades with them. Evidence accumulates.
    Only proven improvements survive.
    """

    def __init__(
        self,
        lab: ImprovementLabOrchestrator,
        avt_engine: Any,  # AutonomousVirtualTrader
        repository: Any,  # AVT trade repository
        contrast_learners: Optional[Dict[str, Any]] = None,
    ):
        self._lab = lab
        self._avt = avt_engine
        self._repo = repository

        # Per-strategy ContrastLearner instances (from ContinuousLearner)
        self._contrast_learners = contrast_learners or {}

        # Active improvements being proven (strategy_id → hypothesis)
        self._active_proofs: Dict[str, ImprovementHypothesis] = {}

        # Proven improvements (strategy_id → list of proven)
        self._proven: Dict[str, List[ImprovementHypothesis]] = {}

    # ── Daily: Run after market close ─────────────────────────────

    def daily_analysis(self) -> Dict[str, Any]:
        """
        Called after market close each day.

        For each strategy:
        1. If no active proof → run lab → apply top hypothesis
        2. If active proof → check progress
        3. If proof complete → evaluate → keep or revert
        """
        results = {}

        for strategy_id in self._avt.strategies:
            version = self._avt.strategy_versions[strategy_id]

            # Check active proof first
            active = self._active_proofs.get(strategy_id)

            if active is not None:
                # Evaluate progress
                eval_result = self._evaluate_proof(
                    strategy_id, active
                )
                results[strategy_id] = eval_result
            else:
                # No active proof — run analysis, maybe apply new one
                discovery_result = self._discover_and_apply(
                    strategy_id, version
                )
                results[strategy_id] = discovery_result

        return results

    def _discover_and_apply(
        self,
        strategy_id: str,
        version: Any,
    ) -> Dict[str, Any]:
        """
        Run improvement lab on AVT's trades.
        If actionable hypothesis found → apply to AVT.
        """
        # Get AVT's trades for current version only
        trades = self._repo.get_avt_trades(
            strategy_id=strategy_id,
            version=version.version,
            since=version.version_start_date,
        )

        if len(trades) < 30:
            return {
                "action": "waiting",
                "reason": f"Only {len(trades)} trades (need 30)",
            }

        # Run improvement lab
        report = self._lab.analyze_avt_trades(
            trades=trades,
            strategy_id=strategy_id,
            strategy_version=version.version,
        )

        if not report.actionable_hypotheses:
            return {
                "action": "no_improvements_found",
                "trades_analyzed": len(trades),
            }

        # Pick the TOP hypothesis (highest net R)
        best = max(
            report.actionable_hypotheses,
            key=lambda h: (
                h.tradeoff.net_expectancy_change_r
                if h.tradeoff
                else 0
            ),
        )

        # Record performance BEFORE improvement
        perf_before = _compute_performance(trades)
        best.proof_performance_before = perf_before

        # Apply to AVT
        self._apply_to_avt(strategy_id, best, version)

        return {
            "action": "improvement_applied",
            "hypothesis": best.title,
            "expected_impact_r": best.expected_impact_r,
            "new_version": version.version,
            "contrast_insights": self._get_contrast_summary(strategy_id),
        }

    def _apply_to_avt(
        self,
        strategy_id: str,
        hypothesis: ImprovementHypothesis,
        version: Any,
    ) -> None:
        """
        Apply a hypothesis to AVT's strategy.
        Bump version. Start proof period.
        """
        # Bump version
        old_version = version.version
        new_version = _increment_version(old_version)

        # Add improvement to AVT's active rules
        version.active_improvements.append(hypothesis)
        version.version = new_version
        version.version_start_date = datetime.now(timezone.utc)

        # Mark hypothesis
        hypothesis.proof_status = ProofStatus.APPLIED_TO_AVT.value
        hypothesis.applied_to_avt_at = datetime.now(timezone.utc)
        hypothesis.applied_at_version = new_version
        hypothesis.proof_start_date = datetime.now(timezone.utc)

        # Track as active proof
        self._active_proofs[strategy_id] = hypothesis

        # Save
        self._repo.save_improvement(hypothesis)

        logger.info(
            "AVT applied improvement to %s: '%s'. "
            "Version %s → %s. Proof period started.",
            strategy_id,
            hypothesis.title,
            old_version,
            new_version,
        )

    def _evaluate_proof(
        self,
        strategy_id: str,
        hypothesis: ImprovementHypothesis,
    ) -> Dict[str, Any]:
        """
        Check if an active improvement has been proven or disproven.

        CRITERIA:
        - Need MIN_PROOF_TRADES trades since application
        - Need MIN_PROOF_WEEKS elapsed
        - Delta must be > MIN_PROOF_DELTA_R to prove
        - Delta < REVERT_THRESHOLD_R → immediate revert
        """
        trades_after = self._repo.get_avt_trades(
            strategy_id=strategy_id,
            since=hypothesis.proof_start_date,
        )

        hypothesis.proof_trades_count = len(trades_after)
        elapsed_weeks = (
            datetime.now(timezone.utc) - hypothesis.proof_start_date
        ).days / 7

        # Early revert check (even before full proof period)
        if len(trades_after) >= 20:
            perf_after = _compute_performance(trades_after)
            delta = (
                perf_after["expectancy_r"]
                - hypothesis.proof_performance_before["expectancy_r"]
            )

            if delta < REVERT_THRESHOLD_R:
                # Improvement is clearly HURTING. Revert immediately.
                self._revert(
                    strategy_id,
                    hypothesis,
                    reason=(
                        f"After {len(trades_after)} trades, "
                        f"delta is {delta:+.3f}R (threshold: "
                        f"{REVERT_THRESHOLD_R}R). Reverting."
                    ),
                )
                return {
                    "action": "reverted_early",
                    "trades": len(trades_after),
                    "delta_r": round(delta, 3),
                }

        # Full evaluation
        if (
            len(trades_after) >= MIN_PROOF_TRADES
            and elapsed_weeks >= MIN_PROOF_WEEKS
        ):
            perf_after = _compute_performance(trades_after)
            delta = (
                perf_after["expectancy_r"]
                - hypothesis.proof_performance_before["expectancy_r"]
            )

            # Statistical test: Mann-Whitney U (one-sided)
            # Compare after-period R-multiples vs before-period baseline
            before_trades = self._repo.get_avt_trades(
                strategy_id=strategy_id,
                before=hypothesis.proof_start_date,
                limit=MIN_PROOF_TRADES,
            )
            r_before = [
                t.get("r_multiple", 0) if isinstance(t, dict) else getattr(t, "r_multiple", 0)
                for t in before_trades
            ]
            r_after_vals = [
                t.get("r_multiple", 0) if isinstance(t, dict) else getattr(t, "r_multiple", 0)
                for t in trades_after
            ]

            # Need ≥ 20 per group for adequate test power (~60% at medium effect, p<0.05)
            if len(r_before) >= 20 and len(r_after_vals) >= 20:
                from scipy.stats import mannwhitneyu
                _, p_value = mannwhitneyu(r_after_vals, r_before, alternative="greater")
            else:
                p_value = 1.0  # insufficient data → cannot prove

            hypothesis.proof_performance_after = perf_after
            hypothesis.proof_delta_r = round(delta, 4)
            hypothesis.proof_p_value = round(p_value, 4)

            # Proof requires BOTH positive delta AND statistical significance
            proven = (delta >= MIN_PROOF_DELTA_R) and (p_value < 0.05)

            if proven:
                # PROVEN
                hypothesis.proof_status = ProofStatus.PROVEN.value

                # Collect example trades
                hypothesis.proof_example_trades = (
                    _get_best_example_trades(trades_after, hypothesis)
                )

                # Move to proven list
                del self._active_proofs[strategy_id]
                self._proven.setdefault(strategy_id, []).append(
                    hypothesis
                )

                self._repo.save_improvement(hypothesis)

                logger.info(
                    "PROVEN: %s improvement '%s' — "
                    "delta +%.3fR over %d trades",
                    strategy_id,
                    hypothesis.title,
                    delta,
                    len(trades_after),
                )

                return {
                    "action": "proven",
                    "title": hypothesis.title,
                    "delta_r": round(delta, 3),
                    "trades": len(trades_after),
                    "weeks": round(elapsed_weeks, 1),
                }
            else:
                # DISPROVEN — delta insufficient or not statistically significant
                not_significant = p_value >= 0.05 and delta >= MIN_PROOF_DELTA_R
                reason_suffix = (
                    f"p={p_value:.3f} (need p<0.05) — not statistically significant."
                    if not_significant
                    else f"delta is only {delta:+.3f}R (need {MIN_PROOF_DELTA_R}R)."
                )
                self._revert(
                    strategy_id,
                    hypothesis,
                    reason=(
                        f"After {len(trades_after)} trades over "
                        f"{elapsed_weeks:.1f} weeks, {reason_suffix} "
                        f"Not worth keeping."
                    ),
                )
                return {
                    "action": "disproven",
                    "delta_r": round(delta, 3),
                    "p_value": round(p_value, 4),
                    "trades": len(trades_after),
                }

        # Still proving
        hypothesis.proof_status = ProofStatus.PROVING.value
        return {
            "action": "proving",
            "trades_so_far": len(trades_after),
            "trades_needed": MIN_PROOF_TRADES,
            "weeks_elapsed": round(elapsed_weeks, 1),
            "weeks_needed": MIN_PROOF_WEEKS,
        }

    def _revert(
        self,
        strategy_id: str,
        hypothesis: ImprovementHypothesis,
        reason: str,
    ) -> None:
        """Remove a failed improvement from AVT."""
        version = self._avt.strategy_versions[strategy_id]

        # Remove from active improvements
        version.active_improvements = [
            imp for imp in version.active_improvements
            if imp.id != hypothesis.id
        ]

        # Bump version
        version.version = _increment_version(version.version)
        version.version_start_date = datetime.now(timezone.utc)

        # Update hypothesis
        hypothesis.proof_status = ProofStatus.REVERTED.value
        hypothesis.reverted_at = datetime.now(timezone.utc)
        hypothesis.revert_reason = reason

        # Remove from active proofs
        if strategy_id in self._active_proofs:
            del self._active_proofs[strategy_id]

        self._repo.save_improvement(hypothesis)

        logger.info(
            "REVERTED: %s improvement '%s' — %s",
            strategy_id,
            hypothesis.title,
            reason,
        )

    # ── Getters for competition engine ────────────────────────────

    def get_proven_improvements(
        self, strategy_id: str
    ) -> List[ImprovementHypothesis]:
        """
        Get all proven improvements for a strategy.
        These are safe to show to the user with full evidence.
        """
        return self._proven.get(strategy_id, [])

    def get_active_proof(
        self, strategy_id: str
    ) -> Optional[ImprovementHypothesis]:
        """Get the currently-being-proven improvement."""
        return self._active_proofs.get(strategy_id)

    def get_all_status(self) -> Dict[str, Any]:
        """Summary for dashboard."""
        status = {}
        for strategy_id in self._avt.strategies:
            active = self._active_proofs.get(strategy_id)
            proven = self._proven.get(strategy_id, [])
            status[strategy_id] = {
                "proving": (
                    {
                        "title": active.title,
                        "trades": active.proof_trades_count,
                        "needed": active.proof_trades_needed,
                    }
                    if active
                    else None
                ),
                "proven_count": len(proven),
                "proven_total_impact_r": round(
                    sum(p.proof_delta_r or 0 for p in proven), 3
                ),
            }
        return status

    # ── Contrast Learner Integration ──────────────────────────

    def _get_contrast_summary(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get contrast insights for a strategy from the CLE.

        Returns a summary of what features differentiate wins
        from losses, plus pattern stability check.

        Used to enrich daily analysis results with edge insights.
        """
        if not _CLE_AVAILABLE:
            return None

        cle = self._contrast_learners.get(strategy_id)
        if cle is None or not cle.has_sufficient_data:
            return None

        try:
            insight = cle.generate_insight()
            stability = cle.check_pattern_stability()

            return {
                "confidence": insight.confidence,
                "data_sufficient": insight.data_sufficient,
                "n_trades": insight.n_total,
                "top_differentiators": [
                    {
                        "feature": fc.feature_name,
                        "direction": fc.direction,
                        "effect_size": round(fc.effect_size, 3),
                        "reliable": fc.reliable,
                    }
                    for fc in insight.top_features[:5]
                ],
                "pattern_stable": stability.get("stable", True),
                "stability_recommendation": stability.get(
                    "recommendation", "maintain_weights"
                ),
            }
        except Exception as e:
            logger.warning("CLE summary failed for %s: %s", strategy_id, e)
            return None

    def set_contrast_learners(
        self, learners: Dict[str, Any]
    ) -> None:
        """
        Set the ContrastLearner instances (from ContinuousLearner).

        Called during system wiring to share CLE instances:
            self_improver.set_contrast_learners(
                continuous_learner._contrast_learners
            )
        """
        self._contrast_learners = learners


# ── Helpers ───────────────────────────────────────────────────────


def _compute_performance(trades: List[Any]) -> Dict[str, float]:
    """Compute performance metrics from trades."""
    if not trades:
        return {"expectancy_r": 0.0, "win_rate": 0.0, "trade_count": 0}

    r_values = []
    wins = 0
    for t in trades:
        r = t.get("r_multiple", 0) if isinstance(t, dict) else getattr(t, "r_multiple", 0)
        r = float(r) if r is not None else 0.0
        r_values.append(r)
        if r > 0:
            wins += 1

    import numpy as np
    return {
        "expectancy_r": round(float(np.mean(r_values)), 4),
        "win_rate": round(wins / len(trades), 4),
        "trade_count": len(trades),
        "total_r": round(float(np.sum(r_values)), 4),
    }


def _increment_version(version: str) -> str:
    """1.0 → 1.1, 1.1 → 1.2, etc."""
    parts = version.split(".")
    if len(parts) == 2:
        major, minor = int(parts[0]), int(parts[1])
        return f"{major}.{minor + 1}"
    return version + ".1"


def _get_best_example_trades(
    trades: List[Any],
    hypothesis: ImprovementHypothesis,
    max_examples: int = 5,
) -> List[Dict[str, Any]]:
    """
    Select the best example trades that demonstrate
    the improvement working.
    """
    # For now, return top trades by R-multiple
    sorted_trades = sorted(
        trades,
        key=lambda t: (
            t.get("r_multiple", 0)
            if isinstance(t, dict)
            else getattr(t, "r_multiple", 0)
        ),
        reverse=True,
    )
    examples = []
    for t in sorted_trades[:max_examples]:
        if isinstance(t, dict):
            examples.append(t)
        else:
            examples.append({
                "id": str(getattr(t, "id", "")),
                "symbol": getattr(t, "symbol", ""),
                "r_multiple": getattr(t, "r_multiple", 0),
                "entry_time": str(getattr(t, "entry_time", "")),
            })
    return examples