# ske/rebuilder.py
"""
SKO Rebuilder — applies temporal weight decay and triggers full rebuilds
of SKO models at tier boundaries, on schedule, or manually.

Temporal decay:
  - Exponential decay with configurable half-life (default 120 days)
  - Weight floor of 0.05 (old trades never fully vanish)
  - Decay applied multiplicatively on top of quality gate weight

Full rebuild triggers:
  - Tier boundary crossing (promotion/demotion)
  - Weekly scheduled (configurable)
  - Manual request
  - Divergence threshold exceeded
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ske.models import QualityGateResult

logger = logging.getLogger(__name__)

_DEFAULT_HALF_LIFE_DAYS = 120.0
_WEIGHT_FLOOR = 0.05
_DIVERGENCE_REBUILD_THRESHOLD = 0.15  # if behavioral cost > 0.15R, trigger rebuild


class SKORebuilder:
    """
    Manages temporal decay and full rebuilds of SKO models.
    """

    def __init__(
        self,
        default_half_life_days: float = _DEFAULT_HALF_LIFE_DAYS,
        weight_floor: float = _WEIGHT_FLOOR,
    ):
        self._default_half_life = default_half_life_days
        self._weight_floor = weight_floor

    def compute_temporal_decay(
        self,
        trade_time: datetime,
        reference_time: Optional[datetime] = None,
        half_life_days: Optional[float] = None,
    ) -> float:
        """
        Compute temporal decay factor for a trade.

        decay = max(floor, 2^(-age_days / half_life))

        Args:
            trade_time: When the trade was taken.
            reference_time: Reference point (default: now UTC).
            half_life_days: Strategy-specific half-life.

        Returns:
            Decay factor in [floor, 1.0].
        """
        ref = reference_time or datetime.now(timezone.utc)
        half_life = half_life_days or self._default_half_life

        # Ensure timezone-aware comparison
        if trade_time.tzinfo is None:
            trade_time = trade_time.replace(tzinfo=timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)

        age_days = (ref - trade_time).total_seconds() / 86400.0
        if age_days <= 0:
            return 1.0

        decay = math.pow(2, -age_days / half_life)
        return max(self._weight_floor, decay)

    def compute_final_learning_weight(
        self,
        quality_gate_weight: float,
        trade_time: datetime,
        reference_time: Optional[datetime] = None,
        half_life_days: Optional[float] = None,
    ) -> float:
        """
        Compute final learning weight = quality_gate_weight × temporal_decay.

        Args:
            quality_gate_weight: Output from TradeQualityGate (0-1).
            trade_time: When the trade occurred.
            reference_time: Reference point for decay computation.
            half_life_days: Strategy-specific half-life override.

        Returns:
            Final learning weight in [0.0, 1.0].
        """
        if quality_gate_weight <= 0:
            return 0.0

        decay = self.compute_temporal_decay(trade_time, reference_time, half_life_days)
        final = quality_gate_weight * decay
        return max(0.0, min(1.0, round(final, 6)))

    def apply_decay_to_batch(
        self,
        trades_with_weights: List[Dict[str, Any]],
        reference_time: Optional[datetime] = None,
        half_life_days: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply temporal decay to a batch of trades.

        Each trade dict must have:
            - quality_gate_weight: float
            - entry_time or trade_time: datetime

        Returns:
            Updated list with added 'final_learning_weight' and 'temporal_decay' fields.
        """
        ref = reference_time or datetime.now(timezone.utc)
        results = []

        for trade in trades_with_weights:
            qg_weight = trade.get("quality_gate_weight", trade.get("learning_weight", 1.0))
            trade_time = trade.get("entry_time") or trade.get("trade_time")

            if trade_time is None:
                # No time info — use full weight (no decay)
                trade["temporal_decay"] = 1.0
                trade["final_learning_weight"] = qg_weight
            else:
                decay = self.compute_temporal_decay(trade_time, ref, half_life_days)
                final = max(0.0, min(1.0, qg_weight * decay))
                trade["temporal_decay"] = round(decay, 6)
                trade["final_learning_weight"] = round(final, 6)

            results.append(trade)

        return results

    def should_rebuild(
        self,
        sko_data: Dict[str, Any],
        tier_changed: bool = False,
        days_since_last_rebuild: Optional[float] = None,
        behavioral_cost: Optional[float] = None,
        rebuild_interval_days: float = 7.0,
    ) -> bool:
        """
        Determine whether a full SKO rebuild should be triggered.

        Triggers:
            1. Tier boundary crossing
            2. Time since last rebuild exceeds interval
            3. Behavioral cost exceeds threshold
            4. Manual flag
        """
        # 1. Tier changed
        if tier_changed:
            logger.info("Rebuild triggered: tier boundary crossing")
            return True

        # 2. Scheduled interval
        if days_since_last_rebuild is not None and days_since_last_rebuild >= rebuild_interval_days:
            logger.info(
                "Rebuild triggered: %.1f days since last rebuild (interval=%.1f)",
                days_since_last_rebuild, rebuild_interval_days,
            )
            return True

        # 3. Behavioral cost
        if behavioral_cost is not None and behavioral_cost > _DIVERGENCE_REBUILD_THRESHOLD:
            logger.info(
                "Rebuild triggered: behavioral cost %.3fR > threshold %.3fR",
                behavioral_cost, _DIVERGENCE_REBUILD_THRESHOLD,
            )
            return True

        return False

    def prepare_rebuild_weights(
        self,
        trades: List[Dict[str, Any]],
        gate_results: List[QualityGateResult],
        reference_time: Optional[datetime] = None,
        half_life_days: Optional[float] = None,
    ) -> List[float]:
        """
        Prepare final learning weights for a full rebuild by combining
        quality gate results with temporal decay.

        Args:
            trades: Trade dicts with entry_time.
            gate_results: Quality gate results aligned with trades.
            reference_time: Reference time for decay.
            half_life_days: Half-life override.

        Returns:
            List of final learning weights aligned with trades.
        """
        ref = reference_time or datetime.now(timezone.utc)
        weights: List[float] = []

        for trade, gate in zip(trades, gate_results):
            qg_weight = gate.learning_weight
            trade_time = trade.get("entry_time") or trade.get("trade_time")

            if trade_time is None or qg_weight <= 0:
                weights.append(qg_weight)
            else:
                final = self.compute_final_learning_weight(
                    qg_weight, trade_time, ref, half_life_days,
                )
                weights.append(final)

        return weights
