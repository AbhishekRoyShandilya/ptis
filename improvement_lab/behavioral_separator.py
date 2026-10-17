# improvement_lab/behavioral_separator.py
"""
Behavioral vs Market Pattern Separator
========================================

For every detected pattern, determines if it's a MARKET pattern
(market reversed → strategy change needed) or a BEHAVIORAL pattern
(trader panicked → coaching needed, not strategy change).

Uses:
  - Shadow trades: what would have happened WITHOUT trader intervention
  - OverrideEvents: when trader modified the plan
  - Quality gate scores: discipline measurement

If behavioral_pct > 50%: route to coaching, NOT strategy modification.
If market_pct > 70%: route to strategy improvement pipeline.
If mixed: route to BOTH.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from improvement_lab.models import (
    BehavioralClassification,
    PatternClassification,
)

logger = logging.getLogger(__name__)

# Override types that indicate trader intervention in exit
EXIT_INTERVENTION_TYPES = {
    "early_exit", "late_exit", "stop_moved",
    "stop_too_tight", "stop_too_wide", "target_skipped",
}


class BehavioralMarketSeparator:
    """
    For any detected pattern, classify whether it's driven by
    market structure or trader behavior.

    This is critical: market patterns need strategy changes.
    Behavioral patterns need coaching, not strategy changes.
    """

    def __init__(self, db_session: Any = None):
        self._db = db_session

    def classify(
        self,
        trade_ids: List[str],
        trades: Optional[List[Any]] = None,
    ) -> BehavioralClassification:
        """
        Classify a set of trades as market-driven or behavior-driven.

        For each trade, compares actual outcome vs shadow outcome,
        and checks for override events.

        Args:
            trade_ids: IDs of trades in the detected pattern
            trades: optional pre-loaded trade objects (with overrides + shadows)
        """
        if not trade_ids:
            return BehavioralClassification(
                classification=PatternClassification.INSUFFICIENT_DATA,
                note="No trades to classify.",
            )

        market_reversals = 0
        behavioral_reversals = 0
        no_shadow = 0
        market_trade_ids = []
        behavioral_trade_ids = []

        trade_list = trades or self._load_trades(trade_ids)

        for trade in trade_list:
            trade_id = str(self._get_field(trade, "id", ""))
            if trade_id not in trade_ids and trade_ids[0] != "":
                continue

            # Get override events
            overrides = self._get_overrides(trade)
            exit_modifications = [
                ov for ov in overrides
                if self._get_override_type(ov) in EXIT_INTERVENTION_TYPES
            ]

            # Get shadow data
            shadow_r = self._get_shadow_r(trade)
            actual_r = self._safe_float(self._get_field(trade, "r_multiple", 0))

            if shadow_r is None:
                no_shadow += 1
                # No shadow data — assume market pattern (conservative)
                market_reversals += 1
                market_trade_ids.append(trade_id)
                continue

            if exit_modifications:
                # Trader intervened in exit
                if shadow_r > actual_r + 0.3:
                    # Shadow did significantly better → trader's intervention hurt
                    behavioral_reversals += 1
                    behavioral_trade_ids.append(trade_id)
                else:
                    # Shadow also didn't do well → market reversed regardless
                    market_reversals += 1
                    market_trade_ids.append(trade_id)
            else:
                # No override — pure market action
                market_reversals += 1
                market_trade_ids.append(trade_id)

        total = market_reversals + behavioral_reversals
        if total < 5:
            return BehavioralClassification(
                classification=PatternClassification.INSUFFICIENT_DATA,
                market_pct=market_reversals / max(total, 1),
                behavioral_pct=behavioral_reversals / max(total, 1),
                note=(
                    f"Only {total} classifiable trades "
                    f"({no_shadow} without shadow data). "
                    "Treat as market pattern with low confidence."
                ),
            )

        market_pct = market_reversals / total
        behavioral_pct = behavioral_reversals / total

        if market_pct > 0.70:
            return BehavioralClassification(
                classification=PatternClassification.MARKET_PATTERN,
                market_pct=round(market_pct, 3),
                behavioral_pct=round(behavioral_pct, 3),
                confidence=round(market_pct, 3),
                market_trade_ids=market_trade_ids,
                behavioral_trade_ids=behavioral_trade_ids,
                action="strategy_improvement",
                note=(
                    "This pattern is driven by market structure, "
                    "not trader behavior. Strategy modification is appropriate."
                ),
            )
        elif behavioral_pct > 0.50:
            return BehavioralClassification(
                classification=PatternClassification.BEHAVIORAL_PATTERN,
                market_pct=round(market_pct, 3),
                behavioral_pct=round(behavioral_pct, 3),
                confidence=round(behavioral_pct, 3),
                market_trade_ids=market_trade_ids,
                behavioral_trade_ids=behavioral_trade_ids,
                action="coaching_intervention",
                note=(
                    "This pattern is significantly driven by trader "
                    "interventions (stop tightening, early exits). "
                    "The strategy itself may not need changing — "
                    "the trader's behavior does. "
                    "Routing to behavioral intelligence module."
                ),
            )
        else:
            return BehavioralClassification(
                classification=PatternClassification.MIXED,
                market_pct=round(market_pct, 3),
                behavioral_pct=round(behavioral_pct, 3),
                confidence=round(max(market_pct, behavioral_pct), 3),
                market_trade_ids=market_trade_ids,
                behavioral_trade_ids=behavioral_trade_ids,
                action="both",
                note=(
                    "This pattern has both market and behavioral "
                    "components. Suggest partial strategy modification "
                    "AND behavioral coaching."
                ),
            )

    # ── Data access helpers ───────────────────────────────────────

    def _load_trades(self, trade_ids: List[str]) -> List[Any]:
        """Load trades from DB. Returns empty list if no DB session."""
        if self._db is None:
            return []
        try:
            from core.models import Trade
            return self._db.query(Trade).filter(
                Trade.id.in_(trade_ids)
            ).all()
        except Exception:
            return []

    def _get_overrides(self, trade: Any) -> List[Any]:
        """Get override events for a trade."""
        if isinstance(trade, dict):
            return trade.get("overrides", [])
        return getattr(trade, "overrides", []) or []

    def _get_override_type(self, override: Any) -> str:
        """Extract override type string."""
        ov_type = self._get_field(override, "override_type", "")
        if hasattr(ov_type, "value"):
            return ov_type.value
        return str(ov_type)

    def _get_shadow_r(self, trade: Any) -> Optional[float]:
        """
        Get the shadow trade's final R for this trade.
        Shadow trades represent what would have happened without intervention.
        """
        # From pre-loaded context
        if isinstance(trade, dict):
            shadow_r = trade.get("shadow_final_r")
            if shadow_r is not None:
                return float(shadow_r)

        # From shadow relationship
        shadow_sessions = getattr(trade, "shadow_sessions", None)
        if shadow_sessions:
            for session in shadow_sessions:
                shadow_trades = getattr(session, "shadow_trades", []) or []
                for st in shadow_trades:
                    if getattr(st, "shadow_type", "") == "mirror":
                        r = getattr(st, "r_multiple", None)
                        if r is not None:
                            return float(r)

        # From DB if available
        if self._db is not None:
            return self._load_shadow_r_from_db(trade)

        return None

    def _load_shadow_r_from_db(self, trade: Any) -> Optional[float]:
        """Load shadow R from database."""
        try:
            from shadow.models import ShadowSession, ShadowTrade
            trade_id = self._get_field(trade, "id")
            if not trade_id:
                return None
            session = self._db.query(ShadowSession).filter(
                ShadowSession.real_trade_id == trade_id
            ).first()
            if not session:
                return None
            mirror = self._db.query(ShadowTrade).filter(
                ShadowTrade.shadow_session_id == session.id,
                ShadowTrade.shadow_type == "mirror",
            ).first()
            if mirror and mirror.r_multiple is not None:
                return float(mirror.r_multiple)
        except Exception:
            pass
        return None

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
