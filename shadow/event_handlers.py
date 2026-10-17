# shadow/event_handlers.py
"""
Event handlers that wire the Shadow Engine into the trade lifecycle.
Shadow creation triggers on TRADE_ENTERED.
Shadow comparison triggers on TRADE_CLOSED.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.database import SessionLocal
from core.events import Event, EventType, get_event_bus
from core.models import Trade
from shadow.comparison import ShadowComparisonEngine
from shadow.engine import ShadowEngine

logger = logging.getLogger(__name__)

_shadow_engine: Optional[ShadowEngine] = None
_comparison_engine: Optional[ShadowComparisonEngine] = None


def get_shadow_engine() -> ShadowEngine:
    global _shadow_engine
    if _shadow_engine is None:
        _shadow_engine = ShadowEngine()
    return _shadow_engine


def get_comparison_engine() -> ShadowComparisonEngine:
    global _comparison_engine
    if _comparison_engine is None:
        _comparison_engine = ShadowComparisonEngine()
    return _comparison_engine


def on_trade_entered(event: Event) -> None:
    """Create shadow session when a real trade is entered."""
    trade_id = event.trade_id
    if not trade_id:
        return

    logger.info("Shadow handler: creating shadows for trade %s", trade_id)

    db = SessionLocal()
    try:
        trade = db.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            logger.error("Trade %s not found for shadow creation", trade_id)
            return

        engine = get_shadow_engine()

        # Build market snapshot dict from event data
        market_data = event.data.get("market_snapshot")

        engine.create_shadow_session(db, trade, market_snapshot=market_data)

    except Exception as e:
        logger.error("Shadow creation failed for trade %s: %s", trade_id, e, exc_info=True)
        db.rollback()
    finally:
        db.close()


def on_trade_closed(event: Event) -> None:
    """Close shadow session and run comparison when real trade closes."""
    trade_id = event.trade_id
    if not trade_id:
        return

    exit_price = event.data.get("exit_price")
    if not exit_price:
        logger.warning("No exit_price in TRADE_CLOSED event for %s", trade_id)
        return

    logger.info("Shadow handler: closing shadows for trade %s", trade_id)

    db = SessionLocal()
    try:
        engine = get_shadow_engine()
        comparison = get_comparison_engine()

        # Close all shadow trades at the real exit price
        close_result = engine.close_session(db, trade_id, final_price=exit_price)

        # Run comparison analysis
        report = comparison.compare(db, trade_id)

        if report:
            logger.info(
                "Shadow comparison: real=$%.2f, best_shadow=$%.2f, "
                "execution_quality=%.0f, opp_cost=$%.2f",
                report.real_pnl, report.best_shadow_pnl,
                report.execution_quality, report.opportunity_cost,
            )

            # Publish comparison result event
            bus = get_event_bus()
            bus.publish(Event(
                event_type=EventType.SHADOW_DIVERGENCE,
                source="shadow_comparison",
                trade_id=trade_id,
                data={
                    "report": report.to_dict(),
                    "execution_quality": report.execution_quality,
                    "opportunity_cost": report.opportunity_cost,
                    "insights": report.insights,
                },
            ))

    except Exception as e:
        logger.error("Shadow close/compare failed for trade %s: %s", trade_id, e, exc_info=True)
        db.rollback()
    finally:
        db.close()


def wire_shadow_handlers() -> None:
    """Register shadow event handlers with the event bus."""
    bus = get_event_bus()
    bus.subscribe(EventType.TRADE_ENTERED, on_trade_entered)
    bus.subscribe(EventType.TRADE_CLOSED, on_trade_closed)
    logger.info("Shadow event handlers wired")