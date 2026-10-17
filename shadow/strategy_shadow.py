# shadow/strategy_shadow.py
"""
Strategy Shadow — creates a shadow trade that follows EXACT strategy
rules (from StrategyDNA or learned StrategyProfile) independent of
what the trader actually did.

This is the "what the rules say" baseline used by the comparison
engine to measure deviation cost.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session as DBSession

from core.enums import ShadowTradeType, TradeDirection
from core.models import Trade
from shadow.models import ShadowSession, ShadowTrade
from strategy_engine.strategy_dna.schema import StrategyDNA

logger = logging.getLogger(__name__)


class StrategyShadowBuilder:
    """
    Builds a shadow trade that represents what the strategy's RULES
    dictate — stop, targets, trailing, time exit — regardless of
    how the trader actually managed the trade.
    """

    def build(
        self,
        db: DBSession,
        session: ShadowSession,
        trade: Trade,
        dna: StrategyDNA,
        market_snapshot: dict[str, Any] | None = None,
    ) -> ShadowTrade | None:
        """Create a strategy-rule shadow from StrategyDNA."""
        direction = self._resolve_direction(trade)

        strategy_stop = self._compute_stop(trade, dna, market_snapshot)
        if strategy_stop is None:
            logger.warning(
                "Cannot compute strategy stop for %s — skipping strategy shadow",
                trade.id,
            )
            return None

        strategy_targets = self._compute_targets(trade, dna, strategy_stop)

        # Risk amount
        if direction == "long":
            risk_dist = trade.entry_price - strategy_stop
        else:
            risk_dist = strategy_stop - trade.entry_price

        risk_amount = risk_dist * trade.position_size if risk_dist > 0 else None

        # Build config from DNA
        trailing_config = self._build_trailing_config(dna)
        time_exit_config = self._build_time_exit_config(dna)

        partial_plan = []
        if hasattr(dna, "profit_taking_rules") and dna.profit_taking_rules:
            for rule in dna.profit_taking_rules:
                if isinstance(rule, dict) and "r_multiple" in rule:
                    partial_plan.append({
                        "r_multiple": rule["r_multiple"],
                        "size_pct": rule.get("size_pct", 50),
                    })

        shadow = ShadowTrade(
            id=uuid.uuid4(),
            session_id=session.id,
            real_trade_id=trade.id,
            shadow_type=ShadowTradeType.STRATEGY.value,
            label=f"Strategy: {dna.display_name}",
            symbol=trade.symbol,
            direction=direction,
            entry_price=trade.entry_price,
            entry_time=trade.entry_time,
            position_size=trade.position_size,
            stop_loss=strategy_stop,
            original_stop=strategy_stop,
            take_profit=strategy_targets[0] if strategy_targets else None,
            trailing_config=trailing_config,
            risk_amount=risk_amount,
            remaining_size=trade.position_size,
            high_since_entry=trade.entry_price,
            low_since_entry=trade.entry_price,
            strategy_id=dna.name,
            config={
                "source": "strategy_shadow_builder",
                "strategy_name": dna.name,
                "strategy_display": dna.display_name,
                "targets": strategy_targets,
                "partial_exit_plan": partial_plan,
                "trailing": trailing_config,
                "time_exit": time_exit_config,
            },
        )
        db.add(shadow)
        db.flush()

        logger.info(
            "Strategy shadow created for %s: stop=%.2f, targets=%s",
            trade.id, strategy_stop, strategy_targets,
        )
        return shadow

    def build_from_profile(
        self,
        db: DBSession,
        session: ShadowSession,
        trade: Trade,
        profile_data: dict[str, Any],
    ) -> ShadowTrade | None:
        """
        Create a strategy shadow from a learned StrategyProfile's
        exit_rules and risk_parameters (JSON fields).
        """
        exit_rules = profile_data.get("exit_rules", {})
        risk_params = profile_data.get("risk_parameters", {})
        direction = self._resolve_direction(trade)

        # Compute stop from profile
        stop_method = exit_rules.get("stop_method", "atr")
        strategy_stop = None

        if stop_method == "atr" and exit_rules.get("typical_stop_atr"):
            atr_mult = exit_rules["typical_stop_atr"]
            # Estimate ATR from trade's stop if available
            if trade.stop_loss is not None:
                if direction == "long":
                    implied_atr = (trade.entry_price - trade.stop_loss) / max(atr_mult, 0.1)
                else:
                    implied_atr = (trade.stop_loss - trade.entry_price) / max(atr_mult, 0.1)
                if direction == "long":
                    strategy_stop = trade.entry_price - (implied_atr * atr_mult)
                else:
                    strategy_stop = trade.entry_price + (implied_atr * atr_mult)

        if strategy_stop is None:
            strategy_stop = trade.stop_loss  # fallback

        if strategy_stop is None:
            return None

        # Risk amount
        if direction == "long":
            risk_dist = trade.entry_price - strategy_stop
        else:
            risk_dist = strategy_stop - trade.entry_price
        risk_amount = risk_dist * trade.position_size if risk_dist > 0 else None

        # Targets from profile R:R
        typical_rr = risk_params.get("typical_rr", 2.0)
        if risk_dist > 0:
            if direction == "long":
                target = trade.entry_price + (risk_dist * typical_rr)
            else:
                target = trade.entry_price - (risk_dist * typical_rr)
        else:
            target = trade.take_profit

        shadow = ShadowTrade(
            id=uuid.uuid4(),
            session_id=session.id,
            real_trade_id=trade.id,
            shadow_type=ShadowTradeType.STRATEGY.value,
            label=f"Profile: {profile_data.get('name', 'learned')}",
            symbol=trade.symbol,
            direction=direction,
            entry_price=trade.entry_price,
            entry_time=trade.entry_time,
            position_size=trade.position_size,
            stop_loss=strategy_stop,
            original_stop=strategy_stop,
            take_profit=target,
            risk_amount=risk_amount,
            remaining_size=trade.position_size,
            high_since_entry=trade.entry_price,
            low_since_entry=trade.entry_price,
            strategy_id=profile_data.get("short_code"),
            config={
                "source": "profile_shadow",
                "profile_name": profile_data.get("name"),
                "typical_rr": typical_rr,
            },
        )
        db.add(shadow)
        db.flush()
        return shadow

    # ── helpers ──

    @staticmethod
    def _resolve_direction(trade: Trade) -> str:
        if isinstance(trade.direction, TradeDirection):
            return trade.direction.value
        return str(trade.direction).lower()

    def _compute_stop(
        self,
        trade: Trade,
        dna: StrategyDNA,
        market: dict[str, Any] | None,
    ) -> float | None:
        method = dna.stop_method

        if method == "atr":
            atr = None
            if market and "atr_14" in (market.get("indicators") or {}):
                atr = market["indicators"]["atr_14"]
            if atr:
                # Parse multiplier from stop_description or default 1.5
                mult = 1.5
                if dna.personal_insights.learned_statistics:
                    mult = dna.personal_insights.learned_statistics.get(
                        "typical_atr_multiple", 1.5
                    )
                direction = self._resolve_direction(trade)
                if direction == "long":
                    return trade.entry_price - (atr * mult)
                return trade.entry_price + (atr * mult)

        if method == "percentage":
            # Extract percentage from description
            pct = 1.0 / 100
            direction = self._resolve_direction(trade)
            if direction == "long":
                return trade.entry_price * (1 - pct)
            return trade.entry_price * (1 + pct)

        # Fallback: use trade's own stop
        return trade.stop_loss

    def _compute_targets(
        self,
        trade: Trade,
        dna: StrategyDNA,
        stop_price: float,
    ) -> list[float]:
        direction = self._resolve_direction(trade)

        if direction == "long":
            risk_dist = trade.entry_price - stop_price
        else:
            risk_dist = stop_price - trade.entry_price

        if risk_dist <= 0:
            return []

        # Parse R-multiples from DNA target_primary
        r_targets = [2.0]  # default
        target_str = dna.target_primary or ""
        if "R" in target_str or "r" in target_str:
            try:
                r_val = float(target_str.replace("R", "").replace("r", "").strip())
                r_targets = [r_val]
            except ValueError:
                pass

        targets = []
        for r in r_targets:
            if direction == "long":
                targets.append(round(trade.entry_price + (risk_dist * r), 4))
            else:
                targets.append(round(trade.entry_price - (risk_dist * r), 4))

        return targets

    @staticmethod
    def _build_trailing_config(dna: StrategyDNA) -> dict[str, Any]:
        # Check if DNA mentions trailing in exit rules
        for rule in dna.profit_taking_rules or []:
            if isinstance(rule, dict) and "trail" in str(rule).lower():
                return {
                    "enabled": True,
                    "type": "atr",
                    "params": {"atr_multiplier": 1.5},
                    "activate_at_r": 1.0,
                }
        return {}

    @staticmethod
    def _build_time_exit_config(dna: StrategyDNA) -> dict[str, Any]:
        for rule in dna.time_exit_rules or []:
            if isinstance(rule, dict) and rule.get("close_before"):
                return {
                    "enabled": True,
                    "close_before": rule["close_before"],
                }
            if isinstance(rule, str) and ":" in rule:
                return {"enabled": True, "close_before": rule}
        return {}
