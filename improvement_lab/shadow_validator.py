# improvement_lab/shadow_validator.py
"""
Shadow Validator
================

A/B tests proposed improvements in shadow mode before graduating them
to live suggestions. Each improvement runs as a shadow experiment:

  DETECTED → PROPOSED → SHADOW_TESTING → VALIDATED / REJECTED

Shadow testing:
  - Tracks trades that WOULD have been affected by the filter/rule
  - Compares actual outcome vs what would have happened with the improvement
  - After 50+ shadow trades, evaluates: did the improvement actually help?

Graduation criteria:
  - >= 50 shadow trades
  - Improvement consistent in >= 65% of affected trades
  - Expected R saved > 0.3R per affected trade
  - No adverse side effects (doesn't filter out winners)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from improvement_lab.models import (
    EntryFilter,
    EarlyExitPattern,
    ExperimentStatus,
    ImprovementCandidate,
    ImprovementType,
    RRKillerPattern,
    ValidationProgress,
)

logger = logging.getLogger(__name__)

# Graduation thresholds
MIN_SHADOW_TRADES = 50
MIN_CONSISTENCY = 0.65       # 65% of shadow trades must confirm benefit
MIN_R_SAVED_PER_TRADE = 0.3  # average R saved per affected trade
MAX_FALSE_POSITIVE_RATE = 0.20  # max 20% of filtered trades were actually winners


class ShadowExperiment:
    """Tracks one improvement through shadow validation."""

    def __init__(
        self,
        experiment_id: str,
        candidate: ImprovementCandidate,
        strategy_id: str,
    ):
        self.experiment_id = experiment_id
        self.candidate = candidate
        self.strategy_id = strategy_id
        self.status = ExperimentStatus.SHADOW_TESTING
        self.created_at = datetime.now(timezone.utc)

        # Shadow tracking
        self.shadow_trades: List[Dict[str, Any]] = []
        self.would_have_skipped: int = 0
        self.actual_losers_caught: int = 0
        self.actual_winners_blocked: int = 0  # false positives
        self.r_saved_total: float = 0.0
        self.r_missed_total: float = 0.0  # R from wrongly blocked winners

    @property
    def shadow_count(self) -> int:
        return len(self.shadow_trades)

    @property
    def consistency(self) -> float:
        if self.would_have_skipped == 0:
            return 0.0
        return self.actual_losers_caught / self.would_have_skipped

    @property
    def false_positive_rate(self) -> float:
        if self.would_have_skipped == 0:
            return 0.0
        return self.actual_winners_blocked / self.would_have_skipped

    @property
    def avg_r_saved(self) -> float:
        if self.would_have_skipped == 0:
            return 0.0
        net_r = self.r_saved_total - self.r_missed_total
        return net_r / self.would_have_skipped

    def to_progress(self) -> ValidationProgress:
        return ValidationProgress(
            experiment_id=self.experiment_id,
            strategy_id=self.strategy_id,
            status=self.status.value,
            shadow_trades_count=self.shadow_count,
            trades_needed=max(0, MIN_SHADOW_TRADES - self.shadow_count),
            consistency_rate=round(self.consistency, 3),
            avg_r_saved=round(self.avg_r_saved, 3),
            false_positive_rate=round(self.false_positive_rate, 3),
            ready_to_graduate=self._check_graduation(),
        )

    def _check_graduation(self) -> bool:
        if self.shadow_count < MIN_SHADOW_TRADES:
            return False
        if self.consistency < MIN_CONSISTENCY:
            return False
        if self.avg_r_saved < MIN_R_SAVED_PER_TRADE:
            return False
        if self.false_positive_rate > MAX_FALSE_POSITIVE_RATE:
            return False
        return True


class ShadowValidator:
    """
    Manages shadow experiments for improvement candidates.

    Each candidate gets a shadow experiment. As new trades come in,
    the validator checks whether the improvement would have applied
    and tracks the outcome.
    """

    def __init__(self):
        self._experiments: Dict[str, ShadowExperiment] = {}

    # ── Experiment lifecycle ──────────────────────────────────────

    def create_experiment(
        self,
        candidate: ImprovementCandidate,
        strategy_id: str,
    ) -> ShadowExperiment:
        """Create a new shadow experiment for a candidate."""
        exp_id = f"exp_{uuid.uuid4().hex[:12]}"
        experiment = ShadowExperiment(
            experiment_id=exp_id,
            candidate=candidate,
            strategy_id=strategy_id,
        )
        self._experiments[exp_id] = experiment
        logger.info(
            "Created shadow experiment %s for strategy %s: %s",
            exp_id, strategy_id, candidate.title,
        )
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[ShadowExperiment]:
        return self._experiments.get(experiment_id)

    def get_active_experiments(
        self, strategy_id: Optional[str] = None,
    ) -> List[ShadowExperiment]:
        """Get all active (shadow testing) experiments."""
        exps = [
            e for e in self._experiments.values()
            if e.status == ExperimentStatus.SHADOW_TESTING
        ]
        if strategy_id:
            exps = [e for e in exps if e.strategy_id == strategy_id]
        return exps

    # ── Shadow trade processing ───────────────────────────────────

    def process_trade(
        self,
        trade: Any,
        strategy_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Process a closed trade against all active experiments for this strategy.

        Returns list of shadow results: which experiments would have triggered.
        """
        results = []
        active = self.get_active_experiments(strategy_id)

        for exp in active:
            result = self._evaluate_trade_against_experiment(trade, exp)
            if result:
                results.append(result)
                # Check if ready to graduate after this trade
                if exp.shadow_count >= MIN_SHADOW_TRADES:
                    self._try_graduate(exp)

        return results

    def _evaluate_trade_against_experiment(
        self,
        trade: Any,
        experiment: ShadowExperiment,
    ) -> Optional[Dict[str, Any]]:
        """Check if this trade matches the experiment's filter condition."""
        candidate = experiment.candidate
        condition = candidate.evidence.get("condition", {})

        if not condition:
            return None

        # Check if trade matches the filter condition
        matched = self._trade_matches_condition(trade, condition)

        if not matched:
            # Trade doesn't match — record as unaffected
            experiment.shadow_trades.append({
                "trade_id": str(getattr(trade, "id", "")),
                "matched": False,
                "outcome": self._get_trade_outcome(trade),
            })
            return None

        # Trade WOULD have been filtered/modified
        experiment.would_have_skipped += 1
        outcome = self._get_trade_outcome(trade)
        pnl_r = outcome.get("r_multiple", 0)

        if pnl_r < 0:
            # Loser caught — improvement would have helped
            experiment.actual_losers_caught += 1
            experiment.r_saved_total += abs(pnl_r)
        elif pnl_r > 0:
            # Winner blocked — false positive
            experiment.actual_winners_blocked += 1
            experiment.r_missed_total += pnl_r

        experiment.shadow_trades.append({
            "trade_id": str(getattr(trade, "id", "")),
            "matched": True,
            "outcome": outcome,
            "would_have_helped": pnl_r < 0,
        })

        return {
            "experiment_id": experiment.experiment_id,
            "matched": True,
            "trade_id": str(getattr(trade, "id", "")),
            "would_have_helped": pnl_r < 0,
            "r_impact": round(pnl_r, 2),
            "progress": experiment.to_progress(),
        }

    def _trade_matches_condition(
        self,
        trade: Any,
        condition: Dict[str, Any],
    ) -> bool:
        """Check if a trade matches a filter condition dict."""
        for key, expected in condition.items():
            if key == "interaction":
                continue  # skip meta keys

            actual = self._get_trade_field(trade, key)
            if actual is None:
                return False

            # Handle comparison operators in string values
            if isinstance(expected, str):
                if expected.startswith("<="):
                    try:
                        threshold = float(expected[2:].strip())
                        if float(actual) > threshold:
                            return False
                    except (ValueError, TypeError):
                        return False
                elif expected.startswith(">="):
                    try:
                        threshold = float(expected[2:].strip())
                        if float(actual) < threshold:
                            return False
                    except (ValueError, TypeError):
                        return False
                elif expected in ("active", "inactive", "elevated", "high"):
                    # ML-discovered conditions — fuzzy match
                    continue
                else:
                    # Exact match
                    actual_str = actual.value if hasattr(actual, "value") else str(actual)
                    if actual_str.lower() != expected.lower():
                        return False
            elif isinstance(expected, (int, float)):
                try:
                    if abs(float(actual) - expected) > 0.1:
                        return False
                except (ValueError, TypeError):
                    return False

        return True

    @staticmethod
    def _get_trade_field(trade: Any, field: str) -> Any:
        """Get a field from trade object or dict."""
        if isinstance(trade, dict):
            return trade.get(field)
        return getattr(trade, field, None)

    @staticmethod
    def _get_trade_outcome(trade: Any) -> Dict[str, Any]:
        """Extract outcome metrics from a closed trade."""
        if isinstance(trade, dict):
            pnl = trade.get("pnl", 0)
            r_mult = trade.get("r_multiple", 0)
        else:
            pnl = getattr(trade, "pnl", 0) or 0
            r_mult = getattr(trade, "r_multiple", 0) or 0

        try:
            pnl = float(pnl)
        except (ValueError, TypeError):
            pnl = 0.0

        try:
            r_mult = float(r_mult)
        except (ValueError, TypeError):
            r_mult = 0.0

        return {
            "pnl": pnl,
            "r_multiple": r_mult,
            "was_winner": pnl > 0,
        }

    # ── Graduation ────────────────────────────────────────────────

    def _try_graduate(self, experiment: ShadowExperiment) -> None:
        """Check if experiment meets graduation criteria."""
        progress = experiment.to_progress()

        if progress.ready_to_graduate:
            experiment.status = ExperimentStatus.VALIDATED
            logger.info(
                "Experiment %s VALIDATED: consistency=%.1f%%, avg_r_saved=%.2f, "
                "false_positive_rate=%.1f%%",
                experiment.experiment_id,
                experiment.consistency * 100,
                experiment.avg_r_saved,
                experiment.false_positive_rate * 100,
            )
        elif experiment.shadow_count >= MIN_SHADOW_TRADES * 2:
            # Gave it enough time — reject
            experiment.status = ExperimentStatus.REJECTED
            logger.info(
                "Experiment %s REJECTED after %d trades: consistency=%.1f%%, "
                "avg_r_saved=%.2f",
                experiment.experiment_id,
                experiment.shadow_count,
                experiment.consistency * 100,
                experiment.avg_r_saved,
            )

    def force_evaluate(self, experiment_id: str) -> Optional[ValidationProgress]:
        """Force-evaluate an experiment regardless of trade count."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        if exp.shadow_count < 10:
            return exp.to_progress()  # too few trades to evaluate

        progress = exp.to_progress()
        if progress.ready_to_graduate:
            exp.status = ExperimentStatus.VALIDATED
        elif exp.consistency < 0.5 or exp.false_positive_rate > 0.3:
            exp.status = ExperimentStatus.REJECTED

        return exp.to_progress()

    def graduate(self, experiment_id: str) -> Optional[ShadowExperiment]:
        """Manually graduate a validated experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None
        if exp.status == ExperimentStatus.VALIDATED:
            exp.status = ExperimentStatus.GRADUATED
            logger.info("Experiment %s GRADUATED", experiment_id)
        return exp

    def reject(self, experiment_id: str, reason: str = "") -> Optional[ShadowExperiment]:
        """Manually reject an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None
        exp.status = ExperimentStatus.REJECTED
        logger.info("Experiment %s REJECTED: %s", experiment_id, reason)
        return exp

    # ── Reporting ─────────────────────────────────────────────────

    def get_all_progress(
        self, strategy_id: Optional[str] = None,
    ) -> List[ValidationProgress]:
        """Get progress for all experiments."""
        exps = self._experiments.values()
        if strategy_id:
            exps = [e for e in exps if e.strategy_id == strategy_id]
        return [e.to_progress() for e in exps]

    def get_graduated(
        self, strategy_id: Optional[str] = None,
    ) -> List[ShadowExperiment]:
        """Get all graduated experiments."""
        exps = [
            e for e in self._experiments.values()
            if e.status in (ExperimentStatus.VALIDATED, ExperimentStatus.GRADUATED)
        ]
        if strategy_id:
            exps = [e for e in exps if e.strategy_id == strategy_id]
        return exps

    def summary(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """Summary stats across all experiments."""
        exps = list(self._experiments.values())
        if strategy_id:
            exps = [e for e in exps if e.strategy_id == strategy_id]

        by_status = {}
        for e in exps:
            status = e.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_experiments": len(exps),
            "by_status": by_status,
            "total_shadow_trades": sum(e.shadow_count for e in exps),
            "total_r_saved": round(
                sum(e.r_saved_total for e in exps), 2,
            ),
            "total_r_missed": round(
                sum(e.r_missed_total for e in exps), 2,
            ),
        }
