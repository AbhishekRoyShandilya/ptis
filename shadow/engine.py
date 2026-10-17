# shadow/engine.py
"""
Core Shadow Trade Engine.

Responsibilities:
1. Create shadow trades when a real trade is entered
2. Update shadow trades with price changes
3. Process exits (stop, target, trailing, time)
4. Close shadow sessions when real trade closes
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session as DBSession

from core.enums import ShadowTradeType, TradeDirection, TradeStatus
from core.events import Event, EventType, get_event_bus
from core.models import Trade
from shadow.models import ShadowPriceSnapshot, ShadowSession, ShadowTrade, TradeDivergence
from strategy_engine.strategy_registry import get_registry
from strategy_engine.templates.loader import StrategyTemplate

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  SHADOW CREATION CONFIG
# ══════════════════════════════════════════════

VARIANT_CONFIGS = {
    ShadowTradeType.VARIANT_TIGHT: {
        "label": "Tight Stop Variant",
        "stop_multiplier": 0.6,
        "target_multiplier": 1.0,
    },
    ShadowTradeType.VARIANT_WIDE: {
        "label": "Wide Stop Variant",
        "stop_multiplier": 1.5,
        "target_multiplier": 1.0,
    },
    ShadowTradeType.VARIANT_TRAIL: {
        "label": "Trailing Exit Variant",
        "stop_multiplier": 1.0,
        "trailing_enabled": True,
        "trailing_atr_mult": 1.5,
    },
}


class ShadowEngine:
    """
    Creates and manages shadow trades for real trade comparison.
    """

    def __init__(self):
        self._registry = get_registry()
        self._event_bus = get_event_bus()

    # ══════════════════════════════════════════
    #  SESSION + SHADOW CREATION
    # ══════════════════════════════════════════

    def create_shadow_session(
        self,
        db: DBSession,
        real_trade: Trade,
        market_snapshot: Optional[dict] = None,
    ) -> ShadowSession:
        """
        Create a full shadow session for a real trade.
        Generates: Mirror + Strategy + Variants.
        """
        strategy_id = real_trade.declared_strategy_id
        template = self._registry.get(strategy_id) if strategy_id else None

        session = ShadowSession(
            id=uuid.uuid4(),
            real_trade_id=real_trade.id,
            strategy_id=strategy_id,
            status="active",
        )
        db.add(session)
        db.flush()

        shadows_created = []

        # 1. Mirror Shadow — exact copy of real trade
        mirror = self._create_mirror_shadow(db, session, real_trade)
        shadows_created.append(mirror)

        # 2. Strategy Shadow — what the strategy rules dictate
        if template:
            strategy_shadow = self._create_strategy_shadow(
                db, session, real_trade, template, market_snapshot
            )
            if strategy_shadow:
                shadows_created.append(strategy_shadow)

        # 3. Variant Shadows — different risk management
        for variant_type, config in VARIANT_CONFIGS.items():
            variant = self._create_variant_shadow(
                db, session, real_trade, variant_type, config, market_snapshot
            )
            if variant:
                shadows_created.append(variant)

        db.commit()

        logger.info(
            "Shadow session created for trade %s: %d shadow trades (%s)",
            real_trade.id,
            len(shadows_created),
            ", ".join(s.shadow_type for s in shadows_created),
        )

        # Publish event
        self._event_bus.publish(Event(
            event_type=EventType.SHADOW_CREATED,
            source="shadow_engine",
            trade_id=real_trade.id,
            data={
                "session_id": str(session.id),
                "shadow_count": len(shadows_created),
                "shadow_ids": [str(s.id) for s in shadows_created],
                "types": [s.shadow_type for s in shadows_created],
            },
        ))

        return session

    def _create_mirror_shadow(
        self, db: DBSession, session: ShadowSession, trade: Trade
    ) -> ShadowTrade:
        """Exact copy of real trade — baseline comparison."""
        shadow = ShadowTrade(
            id=uuid.uuid4(),
            session_id=session.id,
            real_trade_id=trade.id,
            shadow_type=ShadowTradeType.MIRROR.value,
            label="Mirror (Exact Copy)",
            symbol=trade.symbol,
            direction=trade.direction.value,
            entry_price=trade.entry_price,
            entry_time=trade.entry_time,
            position_size=trade.position_size,
            stop_loss=trade.stop_loss,
            original_stop=trade.stop_loss,
            take_profit=trade.take_profit,
            risk_amount=trade.risk_amount,
            remaining_size=trade.position_size,
            high_since_entry=trade.entry_price,
            low_since_entry=trade.entry_price,
            strategy_id=trade.declared_strategy_id,
            config={"source": "mirror", "original_trade_id": str(trade.id)},
        )
        db.add(shadow)
        db.flush()
        return shadow

    def _create_strategy_shadow(
        self,
        db: DBSession,
        session: ShadowSession,
        trade: Trade,
        template: StrategyTemplate,
        market_snapshot: Optional[dict] = None,
    ) -> Optional[ShadowTrade]:
        """
        Shadow trade using EXACT strategy rules.
        Stop/target from template, not from actual trade.
        """
        direction = TradeDirection(trade.direction.value if isinstance(trade.direction, TradeDirection) else trade.direction)

        # Calculate strategy-defined stop
        strategy_stop = self._compute_strategy_stop(
            trade, template, market_snapshot
        )
        # Calculate strategy-defined targets
        strategy_targets = self._compute_strategy_targets(
            trade, template, strategy_stop
        )

        if strategy_stop is None:
            logger.warning("Could not compute strategy stop for shadow — skipping strategy shadow")
            return None

        # Risk amount based on strategy stop
        if direction == TradeDirection.LONG:
            risk_dist = trade.entry_price - strategy_stop
        else:
            risk_dist = strategy_stop - trade.entry_price

        risk_amount = risk_dist * trade.position_size if risk_dist > 0 else None

        # Trailing config
        trailing_config = {}
        if template.exit_rules.trailing.enabled:
            trailing_config = {
                "enabled": True,
                "type": template.exit_rules.trailing.type,
                "params": template.exit_rules.trailing.params,
                "activate_at_r": template.exit_rules.trailing.activate_at_r,
            }

        # Time exit config
        time_exit_config = {}
        if template.exit_rules.time_exit.enabled:
            time_exit_config = {
                "enabled": True,
                "max_duration_minutes": template.exit_rules.time_exit.max_duration_minutes,
                "close_before": template.exit_rules.time_exit.close_before,
            }

        shadow = ShadowTrade(
            id=uuid.uuid4(),
            session_id=session.id,
            real_trade_id=trade.id,
            shadow_type=ShadowTradeType.STRATEGY.value,
            label=f"Strategy: {template.name}",
            symbol=trade.symbol,
            direction=trade.direction.value if isinstance(trade.direction, str) else trade.direction.value,
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
            strategy_id=template.id,
            config={
                "source": "strategy",
                "strategy_id": template.id,
                "targets": strategy_targets,
                "partial_exit_plan": [
                    {"r_multiple": t.r_multiple, "size_pct": t.size_pct}
                    for t in template.exit_rules.targets
                ],
                "trailing": trailing_config,
                "time_exit": time_exit_config,
            },
        )
        db.add(shadow)
        db.flush()
        return shadow

    def _create_variant_shadow(
        self,
        db: DBSession,
        session: ShadowSession,
        trade: Trade,
        variant_type: ShadowTradeType,
        config: dict,
        market_snapshot: Optional[dict] = None,
    ) -> Optional[ShadowTrade]:
        """Create a variant shadow with modified risk parameters."""
        if trade.stop_loss is None:
            return None

        direction = trade.direction.value if isinstance(trade.direction, str) else trade.direction.value

        if direction == "long":
            stop_dist = trade.entry_price - trade.stop_loss
            variant_stop = trade.entry_price - (stop_dist * config.get("stop_multiplier", 1.0))
        else:
            stop_dist = trade.stop_loss - trade.entry_price
            variant_stop = trade.entry_price + (stop_dist * config.get("stop_multiplier", 1.0))

        risk_amount = abs(trade.entry_price - variant_stop) * trade.position_size

        trailing_config = {}
        if config.get("trailing_enabled"):
            trailing_config = {
                "enabled": True,
                "type": "atr",
                "params": {"atr_multiplier": config.get("trailing_atr_mult", 1.5)},
                "activate_at_r": 1.0,
            }

        shadow = ShadowTrade(
            id=uuid.uuid4(),
            session_id=session.id,
            real_trade_id=trade.id,
            shadow_type=variant_type.value,
            label=config.get("label", variant_type.value),
            symbol=trade.symbol,
            direction=direction,
            entry_price=trade.entry_price,
            entry_time=trade.entry_time,
            position_size=trade.position_size,
            stop_loss=variant_stop,
            original_stop=variant_stop,
            take_profit=trade.take_profit,
            trailing_config=trailing_config,
            risk_amount=risk_amount,
            remaining_size=trade.position_size,
            high_since_entry=trade.entry_price,
            low_since_entry=trade.entry_price,
            strategy_id=trade.declared_strategy_id,
            config={
                "source": "variant",
                "variant_type": variant_type.value,
                "stop_multiplier": config.get("stop_multiplier", 1.0),
            },
        )
        db.add(shadow)
        db.flush()
        return shadow

    def _compute_strategy_stop(
        self, trade: Trade, template: StrategyTemplate, market: Optional[dict]
    ) -> Optional[float]:
        """Compute stop loss according to strategy template rules."""
        sl_type = template.exit_rules.stop_loss.type
        params = template.exit_rules.stop_loss.params

        if sl_type == "fixed" and params.get("value"):
            return params["value"]

        if sl_type == "atr":
            atr = None
            if market and "atr_14" in (market.get("indicators") or {}):
                atr = market["indicators"]["atr_14"]
            if atr:
                mult = params.get("multiplier", 1.5)
                if trade.direction == TradeDirection.LONG:
                    return trade.entry_price - (atr * mult)
                else:
                    return trade.entry_price + (atr * mult)

        if sl_type == "percentage":
            pct = params.get("value", 1.0) / 100
            if trade.direction == TradeDirection.LONG:
                return trade.entry_price * (1 - pct)
            else:
                return trade.entry_price * (1 + pct)

        if sl_type == "structure":
            # Use opening range or trade's own stop as fallback
            ref = params.get("reference", "")
            if ref == "opening_range_opposite" and market:
                if trade.direction == TradeDirection.LONG:
                    return market.get("opening_range_low")
                else:
                    return market.get("opening_range_high")
            if ref == "swing_low" and market:
                # Fallback: use provided stop
                return trade.stop_loss

        # Fallback to trade's own stop
        return trade.stop_loss

    def _compute_strategy_targets(
        self,
        trade: Trade,
        template: StrategyTemplate,
        stop_price: Optional[float],
    ) -> list[float]:
        """Compute target prices from strategy R-multiple targets."""
        if not stop_price:
            return []

        if trade.direction == TradeDirection.LONG:
            risk_dist = trade.entry_price - stop_price
        else:
            risk_dist = stop_price - trade.entry_price

        if risk_dist <= 0:
            return []

        targets = []
        for target_def in template.exit_rules.targets:
            if trade.direction == TradeDirection.LONG:
                price = trade.entry_price + (risk_dist * target_def.r_multiple)
            else:
                price = trade.entry_price - (risk_dist * target_def.r_multiple)
            targets.append(round(price, 4))

        return targets

    # ══════════════════════════════════════════
    #  PRICE UPDATES
    # ══════════════════════════════════════════

    def update_price(
        self,
        db: DBSession,
        shadow_trade_id: uuid.UUID,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Update a shadow trade with a new price tick.
        Checks stops, targets, trailing, time exits.
        Returns status update dict.
        """
        timestamp = timestamp or datetime.now(timezone.utc)

        shadow = db.query(ShadowTrade).filter(ShadowTrade.id == shadow_trade_id).first()
        if not shadow or shadow.status != "open":
            return {"status": "skipped", "reason": "not open"}

        direction = shadow.direction

        # Update tracking
        shadow.current_price = price
        shadow.price_updates += 1
        shadow.last_update = timestamp

        if shadow.high_since_entry is None or price > (shadow.high_since_entry or 0):
            shadow.high_since_entry = price
        if shadow.low_since_entry is None or price < (shadow.low_since_entry or float("inf")):
            shadow.low_since_entry = price

        # Compute unrealized P&L
        if direction == "long":
            shadow.unrealized_pnl = (price - shadow.entry_price) * (shadow.remaining_size or shadow.position_size)
            favorable = price - shadow.entry_price
            adverse = shadow.entry_price - price
        else:
            shadow.unrealized_pnl = (shadow.entry_price - price) * (shadow.remaining_size or shadow.position_size)
            favorable = shadow.entry_price - price
            adverse = price - shadow.entry_price

        # Max favorable / adverse excursion
        shadow.max_favorable = max(shadow.max_favorable or 0, max(0, favorable * shadow.position_size))
        shadow.max_adverse = max(shadow.max_adverse or 0, max(0, adverse * shadow.position_size))

        # ── Check exits ──
        exit_result = self._check_exits(shadow, price, timestamp)

        if exit_result:
            self._close_shadow(db, shadow, **exit_result)
            db.commit()
            return {
                "status": "closed",
                "exit_reason": exit_result["exit_reason"],
                "exit_price": exit_result["exit_price"],
                "pnl": shadow.realized_pnl,
            }

        # ── Check partial exits ──
        partial_result = self._check_partial_exits(shadow, price, timestamp)
        if partial_result:
            db.commit()

        db.commit()

        return {
            "status": "open",
            "current_price": price,
            "unrealized_pnl": round(shadow.unrealized_pnl, 2),
            "max_favorable": round(shadow.max_favorable, 2),
            "max_adverse": round(shadow.max_adverse, 2),
        }

    def update_session_prices(
        self,
        db: DBSession,
        real_trade_id: uuid.UUID,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> list[dict]:
        """Update ALL shadow trades for a real trade with a new price."""
        shadows = (
            db.query(ShadowTrade)
            .filter(
                ShadowTrade.real_trade_id == real_trade_id,
                ShadowTrade.status == "open",
            )
            .all()
        )

        results = []
        for shadow in shadows:
            result = self.update_price(db, shadow.id, price, timestamp)
            result["shadow_id"] = str(shadow.id)
            result["shadow_type"] = shadow.shadow_type
            result["label"] = shadow.label
            results.append(result)

        return results

    # ══════════════════════════════════════════
    #  EXIT CHECKS
    # ══════════════════════════════════════════

    def _check_exits(
        self, shadow: ShadowTrade, price: float, timestamp: datetime
    ) -> Optional[dict]:
        """Check if any exit condition is triggered. Returns exit details or None."""
        direction = shadow.direction

        # ── Stop loss ──
        if shadow.stop_loss is not None:
            if direction == "long" and price <= shadow.stop_loss:
                return {
                    "exit_price": shadow.stop_loss,
                    "exit_time": timestamp,
                    "exit_reason": "stop_hit",
                }
            elif direction == "short" and price >= shadow.stop_loss:
                return {
                    "exit_price": shadow.stop_loss,
                    "exit_time": timestamp,
                    "exit_reason": "stop_hit",
                }

        # ── Trailing stop ──
        trailing = shadow.trailing_config or {}
        if trailing.get("enabled"):
            new_trailing = self._compute_trailing_stop(shadow, price)
            if new_trailing is not None:
                shadow.trailing_stop = new_trailing
                # Check if trailing stop hit
                if direction == "long" and price <= new_trailing:
                    return {
                        "exit_price": new_trailing,
                        "exit_time": timestamp,
                        "exit_reason": "trailing_stop",
                    }
                elif direction == "short" and price >= new_trailing:
                    return {
                        "exit_price": new_trailing,
                        "exit_time": timestamp,
                        "exit_reason": "trailing_stop",
                    }

        # ── Full take profit (if no partial exits configured) ──
        config = shadow.config or {}
        partial_plan = config.get("partial_exit_plan", [])
        if not partial_plan and shadow.take_profit is not None:
            if direction == "long" and price >= shadow.take_profit:
                return {
                    "exit_price": shadow.take_profit,
                    "exit_time": timestamp,
                    "exit_reason": "target_hit",
                }
            elif direction == "short" and price <= shadow.take_profit:
                return {
                    "exit_price": shadow.take_profit,
                    "exit_time": timestamp,
                    "exit_reason": "target_hit",
                }

        # ── Time exit ──
        time_exit = config.get("time_exit", {})
        if time_exit.get("enabled"):
            max_dur = time_exit.get("max_duration_minutes")
            if max_dur:
                elapsed = (timestamp - shadow.entry_time).total_seconds() / 60
                if elapsed >= max_dur:
                    return {
                        "exit_price": price,
                        "exit_time": timestamp,
                        "exit_reason": "time_exit",
                    }

            close_before = time_exit.get("close_before")
            if close_before:
                from zoneinfo import ZoneInfo
                from datetime import time as dt_time
                from config.settings import settings
                try:
                    local = timestamp.astimezone(ZoneInfo(settings.MARKET_TIMEZONE))
                    close_t = dt_time(*map(int, close_before.split(":")))
                    if local.time() >= close_t:
                        return {
                            "exit_price": price,
                            "exit_time": timestamp,
                            "exit_reason": "time_exit",
                        }
                except Exception:
                    pass

        return None

    def _check_partial_exits(
        self, shadow: ShadowTrade, price: float, timestamp: datetime
    ) -> Optional[dict]:
        """Check and execute partial exits based on R-multiple targets."""
        config = shadow.config or {}
        partial_plan = config.get("partial_exit_plan", [])
        if not partial_plan:
            return None

        if not shadow.risk_amount or shadow.risk_amount <= 0:
            return None

        remaining = shadow.remaining_size or shadow.position_size
        if remaining <= 0:
            return None

        direction = shadow.direction
        already_exited = shadow.partial_exits or []
        exited_levels = {pe.get("target_level") for pe in already_exited}

        if direction == "long":
            current_r = ((price - shadow.entry_price) * shadow.position_size) / shadow.risk_amount
        else:
            current_r = ((shadow.entry_price - price) * shadow.position_size) / shadow.risk_amount

        for level in partial_plan:
            r_target = level["r_multiple"]
            size_pct = level["size_pct"]

            if r_target in exited_levels:
                continue

            if current_r >= r_target:
                exit_size = shadow.position_size * (size_pct / 100.0)
                exit_size = min(exit_size, remaining)

                partial_exit = {
                    "price": price,
                    "size": exit_size,
                    "r_multiple": round(current_r, 2),
                    "timestamp": timestamp.isoformat(),
                    "target_level": r_target,
                }

                shadow.partial_exits = list(already_exited) + [partial_exit]
                shadow.remaining_size = remaining - exit_size

                logger.info(
                    "Shadow %s partial exit: %.0f shares at %.2f (%.1fR, target %.1fR)",
                    shadow.id, exit_size, price, current_r, r_target,
                )

                # If all exited, close fully
                if shadow.remaining_size <= 0:
                    return {"fully_exited": True}

                # Move stop to breakeven after first target
                if r_target >= 1.0 and shadow.stop_loss != shadow.entry_price:
                    shadow.stop_loss = shadow.entry_price
                    logger.info("Shadow %s: stop moved to breakeven", shadow.id)

                return partial_exit

        return None

    def _compute_trailing_stop(
        self, shadow: ShadowTrade, price: float
    ) -> Optional[float]:
        """Compute trailing stop level based on config."""
        trailing = shadow.trailing_config or {}
        if not trailing.get("enabled"):
            return None

        # Check if activated
        activate_r = trailing.get("activate_at_r", 1.0)
        if shadow.risk_amount and shadow.risk_amount > 0:
            if shadow.direction == "long":
                current_r = ((price - shadow.entry_price) * shadow.position_size) / shadow.risk_amount
            else:
                current_r = ((shadow.entry_price - price) * shadow.position_size) / shadow.risk_amount

            if current_r < activate_r:
                return shadow.trailing_stop  # not yet activated

        trail_type = trailing.get("type", "percentage")
        params = trailing.get("params", {})

        if trail_type == "percentage":
            pct = params.get("percentage", 1.0) / 100
            if shadow.direction == "long":
                new_stop = price * (1 - pct)
            else:
                new_stop = price * (1 + pct)

        elif trail_type == "atr":
            atr_mult = params.get("atr_multiplier", 1.5)
            atr = params.get("atr_value", 1.0)
            if shadow.direction == "long":
                new_stop = price - (atr * atr_mult)
            else:
                new_stop = price + (atr * atr_mult)

        elif trail_type == "swing":
            # Use low_since_entry with buffer for swing trailing
            buffer = params.get("buffer_ticks", 2) * 0.01
            if shadow.direction == "long":
                new_stop = (shadow.low_since_entry or shadow.entry_price) - buffer
            else:
                new_stop = (shadow.high_since_entry or shadow.entry_price) + buffer

        else:
            return shadow.trailing_stop

        # Trailing stop only moves in favorable direction
        current_trailing = shadow.trailing_stop
        if current_trailing is not None:
            if shadow.direction == "long":
                new_stop = max(new_stop, current_trailing)
            else:
                new_stop = min(new_stop, current_trailing)

        return round(new_stop, 4)

    def _close_shadow(
        self,
        db: DBSession,
        shadow: ShadowTrade,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
    ) -> None:
        """Close a shadow trade and compute final P&L."""
        shadow.exit_price = exit_price
        shadow.exit_time = exit_time
        shadow.exit_reason = exit_reason
        shadow.status = exit_reason  # stop_hit, target_hit, trailing_stop, time_exit

        # Compute realized P&L including partials
        total_pnl = 0.0
        partials = shadow.partial_exits or []
        partial_size_exited = sum(p.get("size", 0) for p in partials)

        for partial in partials:
            p_price = partial["price"]
            p_size = partial["size"]
            if shadow.direction == "long":
                total_pnl += (p_price - shadow.entry_price) * p_size
            else:
                total_pnl += (shadow.entry_price - p_price) * p_size

        # Remaining position at exit price
        remaining = shadow.position_size - partial_size_exited
        if remaining > 0:
            if shadow.direction == "long":
                total_pnl += (exit_price - shadow.entry_price) * remaining
            else:
                total_pnl += (shadow.entry_price - exit_price) * remaining

        shadow.realized_pnl = round(total_pnl, 2)

        # R-multiple
        if shadow.risk_amount and shadow.risk_amount > 0:
            shadow.r_multiple = round(total_pnl / shadow.risk_amount, 2)

        logger.info(
            "Shadow %s closed: %s P&L=$%.2f (%.1fR) reason=%s",
            shadow.id, shadow.label, total_pnl,
            shadow.r_multiple or 0.0, exit_reason,
        )

        # Publish event
        self._event_bus.publish(Event(
            event_type=EventType.SHADOW_CLOSED,
            source="shadow_engine",
            trade_id=shadow.real_trade_id,
            data={
                "shadow_id": str(shadow.id),
                "shadow_type": shadow.shadow_type,
                "exit_reason": exit_reason,
                "pnl": shadow.realized_pnl,
                "r_multiple": shadow.r_multiple,
            },
        ))

    # ══════════════════════════════════════════
    #  SESSION CLOSE
    # ══════════════════════════════════════════

    def close_session(
        self,
        db: DBSession,
        real_trade_id: uuid.UUID,
        final_price: float,
        timestamp: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Close all open shadow trades for a real trade.
        Called when the real trade closes.
        """
        timestamp = timestamp or datetime.now(timezone.utc)

        session = (
            db.query(ShadowSession)
            .filter(
                ShadowSession.real_trade_id == real_trade_id,
                ShadowSession.status == "active",
            )
            .first()
        )

        if not session:
            return {"status": "no_session"}

        open_shadows = (
            db.query(ShadowTrade)
            .filter(
                ShadowTrade.session_id == session.id,
                ShadowTrade.status == "open",
            )
            .all()
        )

        results = []
        for shadow in open_shadows:
            self._close_shadow(db, shadow, final_price, timestamp, "real_trade_closed")
            results.append({
                "shadow_id": str(shadow.id),
                "type": shadow.shadow_type,
                "label": shadow.label,
                "pnl": shadow.realized_pnl,
                "r_multiple": shadow.r_multiple,
            })

        # Close session
        session.status = "closed"
        session.closed_at = timestamp
        session.summary = {
            "shadows_closed": len(results),
            "results": results,
        }

        db.commit()

        logger.info(
            "Shadow session closed for trade %s: %d shadows",
            real_trade_id, len(results),
        )

        return {
            "status": "closed",
            "session_id": str(session.id),
            "shadows": results,
        }

    # ══════════════════════════════════════════
    #  SNAPSHOT
    # ══════════════════════════════════════════

    def take_snapshot(
        self,
        db: DBSession,
        shadow_trade_id: uuid.UUID,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a price snapshot for P&L curve building."""
        timestamp = timestamp or datetime.now(timezone.utc)

        shadow = db.query(ShadowTrade).filter(ShadowTrade.id == shadow_trade_id).first()
        if not shadow:
            return

        if shadow.direction == "long":
            pnl = (price - shadow.entry_price) * (shadow.remaining_size or shadow.position_size)
        else:
            pnl = (shadow.entry_price - price) * (shadow.remaining_size or shadow.position_size)

        r_mult = None
        if shadow.risk_amount and shadow.risk_amount > 0:
            r_mult = round(pnl / shadow.risk_amount, 2)

        snapshot = ShadowPriceSnapshot(
            id=uuid.uuid4(),
            shadow_trade_id=shadow_trade_id,
            timestamp=timestamp,
            price=price,
            unrealized_pnl=round(pnl, 2),
            r_multiple=r_mult,
        )
        db.add(snapshot)