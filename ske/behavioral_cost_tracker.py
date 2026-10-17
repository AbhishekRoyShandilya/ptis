# ske/divergence_tracker.py
"""
Divergence Tracker — computes and tracks divergence between filtered
(quality-gated) and unfiltered (all trades) SKO models.

The divergence represents the "behavioral cost" — how much worse the
strategy performs when including undisciplined/noisy trades.

Publishes events:
  - BEHAVIORAL_COST_IMPROVED: cost decreased (trader improving)
  - BEHAVIORAL_COST_REGRESSED: cost increased (trader getting worse)
  - STRATEGY_DIVERGENCE_HIGH: divergence exceeds threshold
  - DIVERGENCE_HISTORY_UPDATED: new snapshot recorded
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.events import Event, EventType, get_event_bus
from ske.models import DivergenceSnapshot

logger = logging.getLogger(__name__)

_HIGH_DIVERGENCE_THRESHOLD = 0.20  # behavioral cost > 0.2R triggers alert


class DivergenceTracker:
    """
    Computes divergence between filtered and unfiltered trade populations
    for a given SKO, and tracks the divergence over time.
    """

    def __init__(
        self,
        high_divergence_threshold: float = _HIGH_DIVERGENCE_THRESHOLD,
    ):
        self._threshold = high_divergence_threshold

    def compute_divergence(
        self,
        all_outcomes: List[float],
        all_weights: List[float],
        all_feature_vectors: Optional[List[Dict[str, float]]] = None,
    ) -> DivergenceSnapshot:
        """
        Compute divergence snapshot between filtered and unfiltered populations.

        Args:
            all_outcomes: R-multiples for ALL trades.
            all_weights: Learning weights (0=excluded, >0=filtered).
            all_feature_vectors: Optional feature vectors for centroid divergence.

        Returns:
            DivergenceSnapshot with all metrics.
        """
        if not all_outcomes or len(all_outcomes) != len(all_weights):
            return DivergenceSnapshot()

        n = len(all_outcomes)
        filtered_mask = [w > 0 for w in all_weights]
        excluded_mask = [w <= 0 for w in all_weights]

        filtered_outcomes = [o for o, m in zip(all_outcomes, filtered_mask) if m]
        excluded_outcomes = [o for o, m in zip(all_outcomes, excluded_mask) if m]
        filtered_weights = [w for w, m in zip(all_weights, filtered_mask) if m]

        raw_count = n
        filtered_count = len(filtered_outcomes)
        excluded_count = len(excluded_outcomes)

        # Effective sample size (Kish's N)
        effective_n = self._kish_n(filtered_weights)

        # Unfiltered stats (all trades, equal weight)
        unfiltered_wr, unfiltered_exp = self._compute_stats(all_outcomes)

        # Filtered stats (weighted)
        filtered_wr, filtered_exp = self._compute_weighted_stats(
            filtered_outcomes, filtered_weights
        )

        # Behavioral cost per trade
        behavioral_cost = None
        if filtered_exp is not None and unfiltered_exp is not None:
            behavioral_cost = filtered_exp - unfiltered_exp

        # Centroid divergence (cosine distance between filtered/unfiltered centroids)
        centroid_div = None
        feature_div: Dict[str, Any] = {}
        if all_feature_vectors and len(all_feature_vectors) == n:
            centroid_div, feature_div = self._compute_centroid_divergence(
                all_feature_vectors, filtered_mask
            )

        # Interpretation
        interpretation = self._interpret(behavioral_cost, centroid_div)

        return DivergenceSnapshot(
            filtered_win_rate=filtered_wr,
            unfiltered_win_rate=unfiltered_wr,
            filtered_expectancy_r=filtered_exp,
            unfiltered_expectancy_r=unfiltered_exp,
            behavioral_cost_per_trade_r=behavioral_cost,
            centroid_divergence_cosine=centroid_div,
            divergence_interpretation=interpretation,
            effective_sample_size=effective_n,
            feature_divergence=feature_div,
            exclusion_breakdown={
                "total_excluded": excluded_count,
                "total_filtered": filtered_count,
                "total_raw": raw_count,
            },
        )

    def publish_events(
        self,
        sko_id: str,
        current: DivergenceSnapshot,
        previous: Optional[DivergenceSnapshot] = None,
    ) -> None:
        """
        Publish divergence-related events based on current vs previous snapshot.
        """
        bus = get_event_bus()

        # Always publish history update
        bus.publish(Event(
            event_type=EventType.DIVERGENCE_HISTORY_UPDATED,
            source="ske.divergence_tracker",
            data={
                "sko_id": sko_id,
                "behavioral_cost": current.behavioral_cost_per_trade_r,
                "interpretation": current.divergence_interpretation,
            },
        ))

        # Check for high divergence
        if (current.behavioral_cost_per_trade_r is not None
                and abs(current.behavioral_cost_per_trade_r) > self._threshold):
            bus.publish(Event(
                event_type=EventType.STRATEGY_DIVERGENCE_HIGH,
                source="ske.divergence_tracker",
                data={
                    "sko_id": sko_id,
                    "behavioral_cost": current.behavioral_cost_per_trade_r,
                    "threshold": self._threshold,
                },
            ))

        # Compare with previous for improvement/regression
        if previous and previous.behavioral_cost_per_trade_r is not None:
            prev_cost = abs(previous.behavioral_cost_per_trade_r)
            curr_cost = abs(current.behavioral_cost_per_trade_r or 0)

            if curr_cost < prev_cost - 0.01:
                bus.publish(Event(
                    event_type=EventType.BEHAVIORAL_COST_IMPROVED,
                    source="ske.divergence_tracker",
                    data={
                        "sko_id": sko_id,
                        "previous_cost": prev_cost,
                        "current_cost": curr_cost,
                        "improvement": prev_cost - curr_cost,
                    },
                ))
            elif curr_cost > prev_cost + 0.01:
                bus.publish(Event(
                    event_type=EventType.BEHAVIORAL_COST_REGRESSED,
                    source="ske.divergence_tracker",
                    data={
                        "sko_id": sko_id,
                        "previous_cost": prev_cost,
                        "current_cost": curr_cost,
                        "regression": curr_cost - prev_cost,
                    },
                ))

    # ──────────────────────────────────────────
    #  Stats Computation
    # ──────────────────────────────────────────

    @staticmethod
    def _compute_stats(outcomes: List[float]) -> Tuple[Optional[float], Optional[float]]:
        """Compute win rate and expectancy (unweighted)."""
        if not outcomes:
            return None, None
        n = len(outcomes)
        wins = sum(1 for r in outcomes if r > 0)
        win_rate = wins / n
        expectancy = sum(outcomes) / n
        return round(win_rate, 4), round(expectancy, 4)

    @staticmethod
    def _compute_weighted_stats(
        outcomes: List[float], weights: List[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute weighted win rate and expectancy."""
        if not outcomes or not weights:
            return None, None

        total_w = sum(weights)
        if total_w <= 0:
            return None, None

        weighted_wins = sum(w for o, w in zip(outcomes, weights) if o > 0)
        win_rate = weighted_wins / total_w

        weighted_exp = sum(o * w for o, w in zip(outcomes, weights)) / total_w

        return round(win_rate, 4), round(weighted_exp, 4)

    @staticmethod
    def _kish_n(weights: List[float]) -> float:
        """Kish's effective sample size."""
        nonzero = [w for w in weights if w > 0]
        if not nonzero:
            return 0.0
        sum_w = sum(nonzero)
        sum_w2 = sum(w * w for w in nonzero)
        if sum_w2 == 0:
            return 0.0
        return round((sum_w * sum_w) / sum_w2, 2)

    @staticmethod
    def _compute_centroid_divergence(
        feature_vectors: List[Dict[str, float]],
        filtered_mask: List[bool],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Compute cosine distance between filtered and unfiltered centroids,
        plus per-feature divergence.
        """
        all_features = sorted(
            set(f for fv in feature_vectors for f in fv.keys())
        )
        if not all_features:
            return None, {}

        filtered_vecs = []
        unfiltered_vecs = []
        for fv, is_filtered in zip(feature_vectors, filtered_mask):
            vec = [fv.get(f, 0.0) for f in all_features]
            unfiltered_vecs.append(vec)
            if is_filtered:
                filtered_vecs.append(vec)

        if not filtered_vecs or len(filtered_vecs) == len(unfiltered_vecs):
            # No divergence if all or none are filtered
            return 0.0, {}

        filtered_arr = np.array(filtered_vecs, dtype=np.float64)
        unfiltered_arr = np.array(unfiltered_vecs, dtype=np.float64)

        filtered_centroid = np.mean(filtered_arr, axis=0)
        unfiltered_centroid = np.mean(unfiltered_arr, axis=0)

        # Cosine distance = 1 - cosine_similarity
        dot = np.dot(filtered_centroid, unfiltered_centroid)
        norm_f = np.linalg.norm(filtered_centroid)
        norm_u = np.linalg.norm(unfiltered_centroid)

        if norm_f > 0 and norm_u > 0:
            cosine_sim = float(dot / (norm_f * norm_u))
            cosine_dist = 1.0 - cosine_sim
        else:
            cosine_dist = 0.0

        # Per-feature divergence
        feature_div: Dict[str, Any] = {}
        for i, feat in enumerate(all_features):
            f_mean = float(filtered_centroid[i])
            u_mean = float(unfiltered_centroid[i])
            shift = f_mean - u_mean
            # Significance: abs(shift) / pooled_std
            f_std = float(np.std(filtered_arr[:, i])) if len(filtered_arr) > 1 else 0.0
            u_std = float(np.std(unfiltered_arr[:, i])) if len(unfiltered_arr) > 1 else 0.0
            pooled_std = math.sqrt((f_std ** 2 + u_std ** 2) / 2) if (f_std + u_std) > 0 else 1.0
            significance = abs(shift) / pooled_std if pooled_std > 0 else 0.0

            if significance > 0.3:  # only report meaningful shifts
                feature_div[feat] = {
                    "filtered_mean": round(f_mean, 4),
                    "unfiltered_mean": round(u_mean, 4),
                    "shift": round(shift, 4),
                    "significance": round(significance, 4),
                }

        return round(cosine_dist, 4), feature_div

    @staticmethod
    def _interpret(
        behavioral_cost: Optional[float],
        centroid_div: Optional[float],
    ) -> Optional[str]:
        """Interpret the divergence pattern."""
        if behavioral_cost is None:
            return None

        abs_cost = abs(behavioral_cost)

        if abs_cost < 0.02:
            return "stable"
        elif behavioral_cost > 0:
            # Filtered > unfiltered = quality gate is helping
            if abs_cost > 0.10:
                return "improving"
            return "stable"
        else:
            # Filtered < unfiltered = something wrong (rare)
            return "regressing"
