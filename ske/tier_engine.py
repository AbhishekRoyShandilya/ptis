# ske/tier_engine.py
"""
Tier Progression Engine — determines SKO tier using effective sample size
(Kish's N) instead of raw trade count, ensuring tier advancement reflects
genuine strategy learning rather than volume of noisy trades.

Three count types:
  - raw_count: total trades attributed to this strategy
  - filtered_count: trades passing the quality gate
  - effective_count: Kish's effective N from weighted samples

Tier thresholds (using effective_count):
  - Hypothesis: 0+
  - Developing: 8+ effective trades
  - Validated: 18+ effective trades, DQAL >= 60
  - Proven: 40+ effective trades, DQAL >= 75, win_rate CI < 0.12
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from core.enums import SKOStatus
from ske.models import TierEvaluation

logger = logging.getLogger(__name__)

# Tier thresholds on effective_count
_TIER_THRESHOLDS = {
    SKOStatus.HYPOTHESIS: 0,
    SKOStatus.DEVELOPING: 8,
    SKOStatus.VALIDATED: 18,
    SKOStatus.PROVEN: 40,
}

# Additional requirements per tier
_TIER_REQUIREMENTS = {
    SKOStatus.VALIDATED: {
        "min_dqal": 60.0,
    },
    SKOStatus.PROVEN: {
        "min_dqal": 75.0,
        "max_win_rate_ci": 0.12,  # 95% CI half-width
    },
}


class TierProgressionEngine:
    """
    Evaluates and manages SKO tier progression using effective sample size.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[SKOStatus, int]] = None,
    ):
        self._thresholds = thresholds or dict(_TIER_THRESHOLDS)

    def evaluate(
        self,
        learning_weights: List[float],
        current_tier: str = "hypothesis",
        dqal_score: float = 0.0,
        win_rate: Optional[float] = None,
    ) -> TierEvaluation:
        """
        Evaluate tier eligibility based on effective sample size.

        Args:
            learning_weights: List of learning weights from quality gate (one per trade).
            current_tier: Current SKO tier status.
            dqal_score: Current DQAL score (0-100).
            win_rate: Current win rate (0-1).

        Returns:
            TierEvaluation with counts and eligible tier.
        """
        raw_count = len(learning_weights)
        filtered_count = sum(1 for w in learning_weights if w > 0)
        effective_count = self.compute_effective_n(learning_weights)

        # Determine eligible tier
        eligible_tier = SKOStatus.HYPOTHESIS
        requirements_met: Dict[str, bool] = {}

        for tier in [SKOStatus.PROVEN, SKOStatus.VALIDATED, SKOStatus.DEVELOPING, SKOStatus.HYPOTHESIS]:
            threshold = self._thresholds.get(tier, 0)
            count_ok = effective_count >= threshold
            requirements_met[f"{tier.value}_count"] = count_ok

            if not count_ok:
                continue

            # Check additional requirements
            extra = _TIER_REQUIREMENTS.get(tier, {})
            all_met = True

            if "min_dqal" in extra:
                dqal_ok = dqal_score >= extra["min_dqal"]
                requirements_met[f"{tier.value}_dqal"] = dqal_ok
                if not dqal_ok:
                    all_met = False

            if "max_win_rate_ci" in extra and win_rate is not None:
                ci = self._win_rate_ci(win_rate, effective_count)
                ci_ok = ci <= extra["max_win_rate_ci"]
                requirements_met[f"{tier.value}_ci"] = ci_ok
                if not ci_ok:
                    all_met = False

            if count_ok and all_met:
                eligible_tier = tier
                break

        promoted = self._is_promotion(current_tier, eligible_tier.value)

        if promoted:
            logger.info(
                "Tier promotion: %s → %s (effective_n=%.1f, raw=%d, filtered=%d)",
                current_tier, eligible_tier.value, effective_count, raw_count, filtered_count,
            )

        return TierEvaluation(
            raw_count=raw_count,
            filtered_count=filtered_count,
            effective_count=round(effective_count, 2),
            current_tier=current_tier,
            eligible_tier=eligible_tier.value,
            promoted=promoted,
            requirements_met=requirements_met,
        )

    @staticmethod
    def compute_effective_n(weights: List[float]) -> float:
        """
        Compute Kish's effective sample size from learning weights.

        N_eff = (sum(w_i))^2 / sum(w_i^2)

        This accounts for the fact that down-weighted trades contribute
        less information than full-weight trades.
        """
        if not weights:
            return 0.0

        # Filter out zero weights
        nonzero = [w for w in weights if w > 0]
        if not nonzero:
            return 0.0

        sum_w = sum(nonzero)
        sum_w2 = sum(w * w for w in nonzero)

        if sum_w2 == 0:
            return 0.0

        return (sum_w * sum_w) / sum_w2

    @staticmethod
    def _win_rate_ci(win_rate: float, n_eff: float) -> float:
        """Compute 95% CI half-width for win rate using effective N."""
        if n_eff <= 0:
            return 1.0
        p = max(0.01, min(0.99, win_rate))
        return 1.96 * math.sqrt(p * (1 - p) / n_eff)

    @staticmethod
    def _is_promotion(current: str, eligible: str) -> bool:
        """Check if eligible tier is a promotion from current."""
        order = {
            "hypothesis": 0,
            "developing": 1,
            "validated": 2,
            "proven": 3,
            "archived": -1,
        }
        return order.get(eligible, 0) > order.get(current, 0)
