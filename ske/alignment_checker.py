# ske/alignment_checker.py
"""
Strategy Alignment Checker — Layer 3 of the Trade Quality Gate.

Verifies that a trade's feature vector is consistent with the strategy's
learned centroid (not a random/FOMO trade claiming to be this strategy).

Graduated activation:
  - disabled: SKO has < 10 filtered trades (cold start)
  - soft: 10-19 filtered trades (alignment_penalty applied but not exclusionary)
  - moderate: 20-49 filtered trades (standard threshold)
  - strict: 50+ filtered trades (tighter threshold, high centroid confidence)

Centroid confidence is computed from pairwise cosine similarity among
filtered examples — more coherent examples = higher confidence = stricter
alignment threshold.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ske.models import AlignmentResult

logger = logging.getLogger(__name__)

# Activation mode thresholds (based on filtered trade count)
_MODE_THRESHOLDS = [
    (50, "strict"),
    (20, "moderate"),
    (10, "soft"),
    (0, "disabled"),
]

# Base similarity thresholds per mode
_BASE_THRESHOLDS = {
    "disabled": 0.0,    # no filtering
    "soft": 0.30,       # very lenient
    "moderate": 0.45,   # standard
    "strict": 0.55,     # tight
}

# How much centroid confidence adjusts the threshold
_CONFIDENCE_ADJUSTMENT = 0.15  # max threshold tightening at confidence=1.0


class StrategyAlignmentChecker:
    """
    Checks whether a trade's feature vector aligns with the strategy's
    learned patterns, using graduated activation based on data maturity.
    """

    def __init__(
        self,
        base_thresholds: Optional[Dict[str, float]] = None,
        confidence_adjustment: float = _CONFIDENCE_ADJUSTMENT,
    ):
        self._base_thresholds = base_thresholds or dict(_BASE_THRESHOLDS)
        self._confidence_adj = confidence_adjustment

    def check_alignment(
        self,
        trade_features: Dict[str, float],
        sko_centroids: List[Dict[str, float]],
        filtered_count: int,
        filtered_feature_vectors: Optional[List[Dict[str, float]]] = None,
        anti_variant_centroids: Optional[List[Dict[str, float]]] = None,
    ) -> AlignmentResult:
        """
        Check if a trade's features align with the strategy.

        Args:
            trade_features: Feature vector of the trade being evaluated.
            sko_centroids: List of variant centroids from the SKO.
            filtered_count: Number of quality-filtered trades in the SKO.
            filtered_feature_vectors: All filtered trade feature vectors
                (used for centroid confidence computation).
            anti_variant_centroids: Centroids of known anti-variant clusters.

        Returns:
            AlignmentResult with alignment decision and metadata.
        """
        mode = self._determine_mode(filtered_count)

        if mode == "disabled":
            return AlignmentResult(
                aligned=True,
                similarity_score=0.0,
                centroid_confidence=0.0,
                threshold_used=0.0,
                activation_mode="disabled",
            )

        if not sko_centroids:
            # No centroids yet — can't check alignment
            return AlignmentResult(
                aligned=True,
                similarity_score=0.0,
                centroid_confidence=0.0,
                threshold_used=0.0,
                activation_mode=mode,
            )

        # Compute centroid confidence from filtered examples
        centroid_confidence = 0.5  # default
        if filtered_feature_vectors and len(filtered_feature_vectors) >= 5:
            centroid_confidence = self._compute_centroid_confidence(
                filtered_feature_vectors
            )

        # Compute effective threshold
        base_threshold = self._base_thresholds.get(mode, 0.45)
        threshold = base_threshold + (self._confidence_adj * centroid_confidence)
        threshold = min(threshold, 0.85)  # hard cap

        # Find best similarity to any variant centroid
        best_sim, best_label = self._best_centroid_similarity(
            trade_features, sko_centroids
        )

        # Check anti-variant match
        anti_match = False
        if anti_variant_centroids:
            anti_sim, _ = self._best_centroid_similarity(
                trade_features, anti_variant_centroids
            )
            if anti_sim > best_sim and anti_sim > 0.5:
                anti_match = True

        # Alignment decision
        aligned = best_sim >= threshold and not anti_match

        return AlignmentResult(
            aligned=aligned,
            similarity_score=round(best_sim, 4),
            centroid_confidence=round(centroid_confidence, 4),
            threshold_used=round(threshold, 4),
            activation_mode=mode,
            nearest_variant=best_label,
            anti_variant_match=anti_match,
        )

    def compute_alignment_penalty(self, result: AlignmentResult) -> float:
        """
        Compute a weight penalty (0–1) based on alignment.
        0.0 = no penalty (perfect alignment), 1.0 = full penalty (no alignment).

        In soft mode, this is used as a multiplier rather than hard exclusion.
        """
        if result.activation_mode == "disabled":
            return 0.0

        if result.anti_variant_match:
            return 1.0

        if result.aligned:
            return 0.0

        # Graduated penalty based on how far below threshold
        if result.threshold_used > 0:
            shortfall = max(0, result.threshold_used - result.similarity_score)
            penalty = min(1.0, shortfall / result.threshold_used)
        else:
            penalty = 0.0

        # In soft mode, cap penalty at 0.5 (never fully exclude)
        if result.activation_mode == "soft":
            penalty = min(penalty, 0.5)

        return round(penalty, 4)

    # ──────────────────────────────────────────
    #  Centroid Confidence
    # ──────────────────────────────────────────

    def _compute_centroid_confidence(
        self, feature_vectors: List[Dict[str, float]]
    ) -> float:
        """
        Compute centroid confidence from pairwise cosine similarity
        among filtered examples. More coherent cluster = higher confidence.

        Uses sigmoid mapping: confidence = sigmoid(mean_pairwise_sim * 5 - 2)
        """
        if len(feature_vectors) < 3:
            return 0.3

        # Convert to matrix
        all_features = sorted(
            set(f for fv in feature_vectors for f in fv.keys())
        )
        matrix = np.array([
            [fv.get(f, 0.0) for f in all_features]
            for fv in feature_vectors
        ], dtype=np.float64)

        # Compute pairwise cosine similarities (sample if too many)
        n = len(matrix)
        if n > 100:
            # Random sample of 100 pairs
            rng = np.random.RandomState(42)
            indices = rng.choice(n, size=100, replace=False)
            matrix = matrix[indices]
            n = 100

        # Compute norms
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms

        # Pairwise cosine similarity via dot product
        sim_matrix = normalized @ normalized.T

        # Extract upper triangle (exclude diagonal)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairwise_sims = sim_matrix[mask]

        if len(pairwise_sims) == 0:
            return 0.3

        mean_sim = float(np.mean(pairwise_sims))

        # Sigmoid mapping: maps mean_sim ∈ [0, 1] → confidence ∈ [~0.1, ~0.95]
        confidence = 1.0 / (1.0 + math.exp(-(mean_sim * 5 - 2)))

        return max(0.0, min(1.0, confidence))

    # ──────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _determine_mode(filtered_count: int) -> str:
        for threshold, mode in _MODE_THRESHOLDS:
            if filtered_count >= threshold:
                return mode
        return "disabled"

    @staticmethod
    def _best_centroid_similarity(
        trade_features: Dict[str, float],
        centroids: List[Dict[str, float]],
    ) -> Tuple[float, Optional[str]]:
        """Find the centroid with highest cosine similarity to trade_features."""
        if not centroids:
            return 0.0, None

        # Unified feature set
        all_features = sorted(
            set(trade_features.keys()) |
            set(f for c in centroids for f in c.keys())
        )

        trade_vec = np.array([trade_features.get(f, 0.0) for f in all_features])
        trade_norm = np.linalg.norm(trade_vec)
        if trade_norm == 0:
            return 0.0, None

        best_sim = -1.0
        best_idx = 0

        for i, centroid in enumerate(centroids):
            cent_vec = np.array([centroid.get(f, 0.0) for f in all_features])
            cent_norm = np.linalg.norm(cent_vec)
            if cent_norm == 0:
                continue
            sim = float(np.dot(trade_vec, cent_vec) / (trade_norm * cent_norm))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        label = f"centroid_{best_idx}" if best_sim > -1 else None
        return max(0.0, best_sim), label
