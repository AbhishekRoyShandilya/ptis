# ske/tqw_computer.py
"""
Adaptive Trade Quality Weight (TQW) Computer.

Computes a composite learning weight (0.0–1.0) for each trade based on
behavioral quality signals. Handles missing components via weight
redistribution (not defaults).

Components:
  - discipline_score (0–100): from StrategyAttribution
  - decision_quality_score (0–100): from DecisionQualityRecord
  - journal_quality_score (0–100): derived from JournalEntry completeness
  - override_penalty (0–1): from OverrideEvent severity
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ske.models import TQWComponents, TQWResult

logger = logging.getLogger(__name__)

# Default weights when ALL components are present
_DEFAULT_WEIGHTS = {
    "discipline_score": 0.35,
    "decision_quality_score": 0.30,
    "journal_quality_score": 0.15,
    "override_penalty": 0.20,
}

# Data confidence mapping: how many components → confidence level
_CONFIDENCE_MAP = {
    0: 0.0,
    1: 0.40,
    2: 0.65,
    3: 0.85,
    4: 1.00,
}


class AdaptiveTQWComputer:
    """
    Computes a composite Trade Quality Weight from available behavioral signals.

    When components are missing, their weights are redistributed proportionally
    among available components — never uses defaults or synthetic values.
    """

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        min_tqw_for_inclusion: float = 0.15,
    ):
        self._base_weights = base_weights or dict(_DEFAULT_WEIGHTS)
        self._min_tqw = min_tqw_for_inclusion

    def compute(self, components: TQWComponents) -> TQWResult:
        """
        Compute composite TQW from available components.

        Returns:
            TQWResult with composite_tqw in [0.0, 1.0], data_confidence,
            weights_used, and exclusion flag.
        """
        available = self._get_available(components)
        n_available = len(available)

        if n_available == 0:
            # No data at all — assign neutral weight with low confidence
            return TQWResult(
                composite_tqw=0.5,
                components=components,
                data_confidence=0.0,
                weights_used={},
                excluded=False,
                exclusion_reason=None,
            )

        # Redistribute weights among available components
        weights_used = self._redistribute_weights(available)

        # Compute weighted score
        weighted_sum = 0.0
        for comp_name, norm_value in available.items():
            weighted_sum += norm_value * weights_used[comp_name]

        # Apply data confidence scaling
        data_confidence = _CONFIDENCE_MAP.get(n_available, 0.5)
        # Scale TQW toward 0.5 (neutral) proportionally to missing confidence
        composite_tqw = data_confidence * weighted_sum + (1 - data_confidence) * 0.5

        # Clamp
        composite_tqw = max(0.0, min(1.0, composite_tqw))

        # Check exclusion
        excluded = composite_tqw < self._min_tqw
        exclusion_reason = (
            f"TQW {composite_tqw:.3f} below threshold {self._min_tqw}"
            if excluded else None
        )

        return TQWResult(
            composite_tqw=round(composite_tqw, 4),
            components=components,
            data_confidence=round(data_confidence, 2),
            weights_used=weights_used,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
        )

    def compute_from_trade(self, trade: Dict[str, Any]) -> TQWResult:
        """
        Convenience: extract components from a trade dict and compute TQW.

        Expected keys (all optional):
            discipline_score, decision_quality_score, journal_quality_score,
            override_count, override_max_severity
        """
        components = self._extract_components(trade)
        return self.compute(components)

    # ──────────────────────────────────────────
    #  Internal
    # ──────────────────────────────────────────

    def _get_available(self, comp: TQWComponents) -> Dict[str, float]:
        """Return normalized [0,1] values for available components."""
        available: Dict[str, float] = {}

        if comp.discipline_score is not None:
            available["discipline_score"] = comp.discipline_score / 100.0

        if comp.decision_quality_score is not None:
            available["decision_quality_score"] = comp.decision_quality_score / 100.0

        if comp.journal_quality_score is not None:
            available["journal_quality_score"] = comp.journal_quality_score / 100.0

        if comp.override_penalty is not None:
            # override_penalty is already 0–1 where 0=good, 1=bad
            # Invert: high quality = low penalty
            available["override_penalty"] = 1.0 - comp.override_penalty

        return available

    def _redistribute_weights(self, available: Dict[str, float]) -> Dict[str, float]:
        """Redistribute missing component weights proportionally among available."""
        total_available_weight = sum(
            self._base_weights.get(k, 0.0) for k in available
        )
        if total_available_weight <= 0:
            # Fallback: equal weights
            n = len(available)
            return {k: 1.0 / n for k in available}

        # Normalize so available weights sum to 1.0
        return {
            k: self._base_weights.get(k, 0.0) / total_available_weight
            for k in available
        }

    @staticmethod
    def _extract_components(trade: Dict[str, Any]) -> TQWComponents:
        """Extract TQW components from a trade dict."""
        discipline = trade.get("discipline_score")
        dq = trade.get("decision_quality_score")
        journal = trade.get("journal_quality_score")

        # Override penalty: compute from count + severity
        override_count = trade.get("override_count", 0)
        max_severity = trade.get("override_max_severity", 0.0)
        override_penalty = None
        if override_count is not None and override_count > 0:
            # More overrides + higher severity = higher penalty
            # Cap at 1.0
            override_penalty = min(1.0, (override_count * 0.15) + (max_severity * 0.3))
        elif override_count == 0:
            override_penalty = 0.0

        return TQWComponents(
            discipline_score=discipline,
            decision_quality_score=dq,
            journal_quality_score=journal,
            override_penalty=override_penalty,
        )

    @staticmethod
    def compute_journal_quality(journal: Dict[str, Any]) -> float:
        """
        Derive a journal quality score (0–100) from JournalEntry fields.
        Rewards completeness and self-reflection.
        """
        score = 0.0
        max_score = 0.0

        # Was it planned? (20 pts)
        max_score += 20
        if journal.get("was_planned") is not None:
            score += 20 if journal["was_planned"] else 5

        # Pre-trade fields (20 pts)
        max_score += 20
        pre_fields = ["plan", "pre_trade_confidence", "pre_trade_emotion"]
        filled = sum(1 for f in pre_fields if journal.get(f) is not None)
        score += (filled / len(pre_fields)) * 20

        # Post-trade fields (30 pts)
        max_score += 30
        post_fields = ["post_trade_emotion", "followed_plan", "would_take_again",
                       "what_went_well", "what_went_wrong"]
        filled = sum(1 for f in post_fields if journal.get(f) is not None)
        score += (filled / len(post_fields)) * 30

        # Lesson learned (20 pts)
        max_score += 20
        lesson = journal.get("lesson")
        if lesson and len(str(lesson)) > 10:
            score += 20
        elif lesson:
            score += 10

        # Follow-through (10 pts)
        max_score += 10
        if journal.get("followed_plan") is True:
            score += 10
        elif journal.get("followed_plan") is False:
            score += 3  # at least they admitted it

        return min(100.0, (score / max_score) * 100) if max_score > 0 else 0.0
