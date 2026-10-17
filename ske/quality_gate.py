# ske/quality_gate.py
"""
Trade Quality Gate — 4-layer pipeline that determines how much each trade
contributes to SKO learning, preventing behavioral noise from polluting
strategy knowledge.

Layers:
  1. Behavioral Hard Filter — immediate exclusion for FOMO, revenge, tilt
  2. Composite TQW — weighted quality score from discipline, DQ, journal, overrides
  3. Strategy Alignment — feature vector similarity to learned centroid
  4. Human Override — manual include/exclude with asymmetric caps

Dual model routing:
  - Every trade feeds the "unfiltered" model (for divergence tracking)
  - Only trades passing the gate feed the "filtered" model (for learning)
  - Divergence = behavioral cost measurement
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ske.alignment_checker import StrategyAlignmentChecker
from ske.models import (
    AlignmentResult, QualityGateResult, TQWComponents, TQWResult,
)
from ske.tqw_computer import AdaptiveTQWComputer

logger = logging.getLogger(__name__)

# Layer 1: Behavioral exclusion patterns (hard filter)
_HARD_EXCLUSION_EMOTIONS = {"fomo", "revenge", "tilt", "panic", "euphoria"}

# Layer 2: TQW threshold below which trade is excluded from filtered model
_DEFAULT_MIN_TQW = 0.15

# Layer 4: Human override caps
_OVERRIDE_INCLUDE_CAP = 0.70   # manual include capped at this weight
_OVERRIDE_EXCLUDE_FLOOR = 0.0  # manual exclude sets weight to 0


class TradeQualityGate:
    """
    4-layer quality gate pipeline for SKO trade learning.

    Usage:
        gate = TradeQualityGate()
        result = gate.evaluate(trade_dict, sko_context)
        # result.learning_weight → how much this trade contributes
        # result.route → "filtered" | "unfiltered_only" | "excluded"
    """

    def __init__(
        self,
        tqw_computer: Optional[AdaptiveTQWComputer] = None,
        alignment_checker: Optional[StrategyAlignmentChecker] = None,
        min_tqw: float = _DEFAULT_MIN_TQW,
        enable_alignment: bool = True,
    ):
        self._tqw = tqw_computer or AdaptiveTQWComputer(min_tqw_for_inclusion=min_tqw)
        self._alignment = alignment_checker or StrategyAlignmentChecker()
        self._min_tqw = min_tqw
        self._enable_alignment = enable_alignment

    def evaluate(
        self,
        trade: Dict[str, Any],
        sko_context: Optional[Dict[str, Any]] = None,
    ) -> QualityGateResult:
        """
        Run the full 4-layer quality gate on a trade.

        Args:
            trade: Trade dict with behavioral metadata:
                - emotion_pre, emotion_post: emotional state
                - discipline_score: 0-100
                - decision_quality_score: 0-100
                - journal_quality_score: 0-100
                - override_count, override_max_severity
                - feature_vector: {feature: value}
                - human_override: Optional float (manual weight override)
                - behavioral_flags: Optional list of flag strings
            sko_context: SKO context for alignment checking:
                - variant_centroids: list of centroid dicts
                - anti_variant_centroids: list of anti-variant centroid dicts
                - filtered_count: number of quality-filtered trades
                - filtered_feature_vectors: list of filtered trade feature vectors

        Returns:
            QualityGateResult with learning_weight and routing decision.
        """
        ctx = sko_context or {}
        exclusion_reasons: List[str] = []
        trade_id = str(trade.get("id", ""))

        # ── Layer 1: Behavioral Hard Filter ──
        layer1_passed = self._layer1_behavioral_filter(trade)
        if not layer1_passed:
            reason = self._get_exclusion_reason(trade)
            exclusion_reasons.append(f"L1: {reason}")
            return QualityGateResult(
                trade_id=trade_id,
                learning_weight=0.0,
                tqw=TQWResult(
                    composite_tqw=0.0,
                    components=TQWComponents(),
                    data_confidence=0.0,
                ),
                alignment=AlignmentResult(activation_mode="disabled"),
                layer1_passed=False,
                layer2_tqw=0.0,
                layer3_aligned=True,
                exclusion_reasons=exclusion_reasons,
                route="excluded",
            )

        # ── Layer 2: Composite TQW ──
        tqw_result = self._tqw.compute_from_trade(trade)
        layer2_weight = tqw_result.composite_tqw

        if tqw_result.excluded:
            exclusion_reasons.append(f"L2: {tqw_result.exclusion_reason}")

        # ── Layer 3: Strategy Alignment ──
        alignment_result = AlignmentResult(activation_mode="disabled")
        alignment_penalty = 0.0

        if self._enable_alignment and trade.get("feature_vector"):
            alignment_result = self._alignment.check_alignment(
                trade_features=trade["feature_vector"],
                sko_centroids=ctx.get("variant_centroids", []),
                filtered_count=ctx.get("filtered_count", 0),
                filtered_feature_vectors=ctx.get("filtered_feature_vectors"),
                anti_variant_centroids=ctx.get("anti_variant_centroids"),
            )
            alignment_penalty = self._alignment.compute_alignment_penalty(
                alignment_result
            )

            if not alignment_result.aligned:
                exclusion_reasons.append(
                    f"L3: similarity {alignment_result.similarity_score:.3f} "
                    f"< threshold {alignment_result.threshold_used:.3f}"
                )
            if alignment_result.anti_variant_match:
                exclusion_reasons.append("L3: matches anti-variant cluster")

        # Combine Layer 2 + Layer 3
        combined_weight = layer2_weight * (1.0 - alignment_penalty)

        # ── Layer 4: Human Override ──
        human_override = trade.get("human_override")
        layer4_override = None

        if human_override is not None:
            layer4_override = float(human_override)
            if layer4_override > 0:
                # Include override: cap at _OVERRIDE_INCLUDE_CAP
                combined_weight = min(layer4_override, _OVERRIDE_INCLUDE_CAP)
            else:
                # Exclude override
                combined_weight = _OVERRIDE_EXCLUDE_FLOOR
                exclusion_reasons.append("L4: human excluded")

        # Final weight clamp
        learning_weight = max(0.0, min(1.0, combined_weight))

        # Determine route
        if learning_weight <= 0.0:
            route = "excluded"
        elif tqw_result.excluded or (not alignment_result.aligned and alignment_result.activation_mode not in ("disabled", "soft")):
            route = "unfiltered_only"
        else:
            route = "filtered"

        return QualityGateResult(
            trade_id=trade_id,
            learning_weight=round(learning_weight, 4),
            tqw=tqw_result,
            alignment=alignment_result,
            human_override=layer4_override,
            layer1_passed=True,
            layer2_tqw=round(layer2_weight, 4),
            layer3_aligned=alignment_result.aligned,
            layer4_override=layer4_override,
            exclusion_reasons=exclusion_reasons,
            route=route,
        )

    def evaluate_batch(
        self,
        trades: List[Dict[str, Any]],
        sko_context: Optional[Dict[str, Any]] = None,
    ) -> List[QualityGateResult]:
        """Evaluate multiple trades through the quality gate."""
        return [self.evaluate(t, sko_context) for t in trades]

    # ──────────────────────────────────────────
    #  Layer 1: Behavioral Hard Filter
    # ──────────────────────────────────────────

    @staticmethod
    def _layer1_behavioral_filter(trade: Dict[str, Any]) -> bool:
        """
        Hard exclusion for trades with clear behavioral contamination.
        Returns True if trade passes (is NOT excluded).
        """
        # Check emotional state
        emotion_pre = str(trade.get("emotion_pre", "")).lower()
        emotion_post = str(trade.get("emotion_post", "")).lower()

        if emotion_pre in _HARD_EXCLUSION_EMOTIONS:
            return False
        if emotion_post in _HARD_EXCLUSION_EMOTIONS and emotion_pre in _HARD_EXCLUSION_EMOTIONS:
            return False

        # Check explicit behavioral flags
        flags = trade.get("behavioral_flags", [])
        if flags:
            hard_flags = {"revenge_trade", "fomo_entry", "tilt_trade", "panic_exit"}
            if set(f.lower() for f in flags) & hard_flags:
                return False

        # Check for extremely low discipline
        discipline = trade.get("discipline_score")
        if discipline is not None and discipline < 15:
            return False

        return True

    @staticmethod
    def _get_exclusion_reason(trade: Dict[str, Any]) -> str:
        """Get human-readable reason for Layer 1 exclusion."""
        emotion_pre = str(trade.get("emotion_pre", "")).lower()
        if emotion_pre in _HARD_EXCLUSION_EMOTIONS:
            return f"pre-trade emotion: {emotion_pre}"

        flags = trade.get("behavioral_flags", [])
        if flags:
            return f"behavioral flags: {', '.join(flags)}"

        discipline = trade.get("discipline_score")
        if discipline is not None and discipline < 15:
            return f"discipline score {discipline} < 15"

        return "behavioral exclusion"
