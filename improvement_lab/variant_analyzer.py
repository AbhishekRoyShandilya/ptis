# improvement_lab/variant_analyzer.py
"""
Variant-Aware Analyzer
=======================

Every analysis in the improvement lab runs at TWO levels:
  1. STRATEGY LEVEL — broad pattern detection
  2. VARIANT LEVEL — granular, actionable patterns

Cross-checks against the SKE variant engine:
  - If failing trades are concentrated in one variant →
    check if it's an anti-variant → different action
  - If spread across variants → strategy-wide issue

Before suggesting "add exit rule at 1:2 when RSI > 75":
  → Check: Do the failing trades belong to a specific variant?
  → If YES + anti-variant: "stop trading that variant"
  → If YES + marginal: "variant-specific fix"
  → If NO: "strategy-wide improvement"
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from improvement_lab.models import VariantConcentration

logger = logging.getLogger(__name__)

# If >60% of pattern trades are in one variant, it's concentrated
CONCENTRATION_THRESHOLD = 0.60
MIN_VARIANT_TRADES = 3


class VariantAnalyzer:
    """
    Checks whether detected patterns are concentrated in specific variants.
    Uses the SKE variant engine for classification.
    """

    def __init__(self, db_session: Any = None):
        self._db = db_session

    def check_variant_concentration(
        self,
        pattern_trade_ids: List[str],
        strategy_id: str,
        trades: Optional[List[Any]] = None,
    ) -> VariantConcentration:
        """
        Check if the trades in a detected pattern are concentrated
        in a specific SKE variant.

        Returns VariantConcentration with action recommendation.
        """
        if not pattern_trade_ids:
            return VariantConcentration(is_concentrated=False)

        # Get variant IDs for each trade
        variant_map = self._get_variant_map(pattern_trade_ids, strategy_id, trades)

        if not variant_map:
            return VariantConcentration(
                is_concentrated=False,
                action="all_variants",
            )

        # Count trades per variant
        variant_counts: Dict[str, int] = defaultdict(int)
        for trade_id, variant_id in variant_map.items():
            variant_counts[variant_id or "unclassified"] += 1

        total = len(variant_map)
        if total < MIN_VARIANT_TRADES:
            return VariantConcentration(is_concentrated=False, action="all_variants")

        # Find dominant variant
        dominant_variant = max(variant_counts, key=variant_counts.get)
        dominant_count = variant_counts[dominant_variant]
        concentration_pct = dominant_count / total

        if concentration_pct < CONCENTRATION_THRESHOLD:
            return VariantConcentration(
                is_concentrated=False,
                concentration_pct=round(concentration_pct, 3),
                action="all_variants",
            )

        # Pattern IS concentrated in one variant
        # Look up variant details from SKE
        variant_info = self._get_variant_info(strategy_id, dominant_variant)
        variant_type = self._classify_variant_type(variant_info)

        if variant_type == "anti":
            action = "avoid_variant"
        elif variant_type == "marginal":
            action = "variant_specific_exit"
        else:
            action = "variant_specific_exit"

        return VariantConcentration(
            is_concentrated=True,
            primary_variant_id=dominant_variant,
            primary_variant_label=self._get_variant_label(variant_info),
            variant_type=variant_type,
            concentration_pct=round(concentration_pct, 3),
            action=action,
        )

    def group_trades_by_variant(
        self,
        trades: List[Any],
        strategy_id: str,
    ) -> Dict[Optional[str], List[Any]]:
        """Group trades by their SKE variant classification."""
        groups: Dict[Optional[str], List[Any]] = defaultdict(list)

        for trade in trades:
            variant_id = self._classify_trade_variant(trade, strategy_id)
            groups[variant_id].append(trade)

        return dict(groups)

    # ── Variant lookup ────────────────────────────────────────────

    def _get_variant_map(
        self,
        trade_ids: List[str],
        strategy_id: str,
        trades: Optional[List[Any]] = None,
    ) -> Dict[str, Optional[str]]:
        """Map trade_id → variant_id for a set of trades."""
        result: Dict[str, Optional[str]] = {}

        if trades:
            for trade in trades:
                tid = str(self._get_field(trade, "id", ""))
                if tid in trade_ids or not trade_ids:
                    variant_id = self._classify_trade_variant(trade, strategy_id)
                    result[tid] = variant_id
            return result

        # If no trades provided, try DB
        if self._db is not None:
            return self._load_variant_map_from_db(trade_ids, strategy_id)

        return result

    def _classify_trade_variant(
        self,
        trade: Any,
        strategy_id: str,
    ) -> Optional[str]:
        """Classify a single trade into its variant using SKE."""
        # Check if trade already has variant assigned
        variant_id = self._get_field(trade, "variant_id")
        if variant_id:
            return str(variant_id)

        # Try using the variant engine
        try:
            from ske.variant_engine import classify_trade
            from ske.models import VariantInfo

            # Get variants for this strategy
            variants = self._get_strategy_variants(strategy_id)
            if not variants:
                return None

            # Build feature vector from trade
            feature_vector = self._trade_to_feature_vector(trade)
            if not feature_vector:
                return None

            label, score = classify_trade(feature_vector, variants)
            return label
        except ImportError:
            pass
        except Exception as e:
            logger.debug("Variant classification failed: %s", e)

        return None

    def _get_strategy_variants(self, strategy_id: str) -> List[Any]:
        """Load variants for a strategy from SKE."""
        if self._db is None:
            return []
        try:
            from core.models import SKOVariant
            return self._db.query(SKOVariant).filter(
                SKOVariant.sko_id == strategy_id,
                SKOVariant.is_active == True,
            ).all()
        except Exception:
            pass

        # Try via SKE registry
        try:
            from ske.registry import get_sko
            sko = get_sko(strategy_id)
            if sko:
                return sko.variants or []
        except Exception:
            pass

        return []

    def _get_variant_info(
        self, strategy_id: str, variant_id: str,
    ) -> Optional[Any]:
        """Load detailed variant info."""
        if self._db is None:
            return None
        try:
            from core.models import SKOVariant
            return self._db.query(SKOVariant).filter(
                SKOVariant.sko_id == strategy_id,
                SKOVariant.label == variant_id,
            ).first()
        except Exception:
            return None

    def _classify_variant_type(self, variant_info: Any) -> str:
        """Determine if a variant is positive, marginal, or anti."""
        if variant_info is None:
            return "unknown"

        # Check explicit type
        variant_type = self._get_field(variant_info, "variant_type")
        if variant_type == "anti_variant":
            return "anti"

        # Check by performance metrics
        win_rate = self._safe_float(self._get_field(variant_info, "win_rate"), 0.5)
        expectancy = self._safe_float(self._get_field(variant_info, "expectancy"), 0.0)

        if win_rate < 0.4 and expectancy < 0.0:
            return "anti"
        elif expectancy < 0.1:
            return "marginal"
        else:
            return "positive"

    def _get_variant_label(self, variant_info: Any) -> str:
        """Get human-readable variant label."""
        if variant_info is None:
            return "unknown"
        label = self._get_field(variant_info, "label", "")
        narrative = self._get_field(variant_info, "narrative", "")
        return label or narrative[:50] or "unnamed variant"

    def _trade_to_feature_vector(self, trade: Any) -> Optional[Dict[str, float]]:
        """Convert trade to feature vector for variant classification."""
        try:
            features = {}
            for field in ["conviction_score", "risk_pct"]:
                val = self._get_field(trade, field)
                if val is not None:
                    features[field] = float(val)

            # Regime
            regime = self._get_field(trade, "market_regime", "unknown")
            if hasattr(regime, "value"):
                regime = regime.value
            features["regime"] = hash(str(regime)) % 100 / 100.0

            # Session
            session = self._get_field(trade, "session_type", "morning")
            if hasattr(session, "value"):
                session = session.value
            features["session"] = hash(str(session)) % 100 / 100.0

            return features if features else None
        except Exception:
            return None

    def _load_variant_map_from_db(
        self, trade_ids: List[str], strategy_id: str,
    ) -> Dict[str, Optional[str]]:
        """Load variant assignments from DB."""
        # Implementation depends on how variants are stored per-trade
        # For now, classify each trade on the fly
        result: Dict[str, Optional[str]] = {}
        try:
            from core.models import Trade
            trades = self._db.query(Trade).filter(
                Trade.id.in_(trade_ids)
            ).all()
            for trade in trades:
                tid = str(trade.id)
                result[tid] = self._classify_trade_variant(trade, strategy_id)
        except Exception:
            pass
        return result

    @staticmethod
    def _get_field(obj: Any, field: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(field, default)
        val = getattr(obj, field, default)
        return default if val is None else val

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
