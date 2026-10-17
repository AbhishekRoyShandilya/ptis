# api/routes/shadow.py
"""Shadow trading API endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session as DBSession

from api.auth import get_current_user

from api.schemas.shadow_schemas import (
    ComparisonResponse,
    PriceUpdateRequest,
    PriceUpdateResponse,
    ShadowSessionResponse,
    ShadowSummaryResponse,
    ShadowTradeResponse,
)
from core.database import get_db_dependency as get_db
from core.models import Trade
from shadow.comparison import ShadowComparisonEngine
from shadow.engine import ShadowEngine
from shadow.models import ShadowSession, ShadowTrade

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/shadow", tags=["shadow"], dependencies=[Depends(get_current_user)])

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


# ══════════════════════════════════════════════
#  GET /shadow/sessions — List shadow sessions
# ══════════════════════════════════════════════

@router.get("/sessions", response_model=list[ShadowSessionResponse])
def list_shadow_sessions(
    status: Optional[str] = Query(None, pattern="^(active|closed)$"),
    limit: int = Query(20, ge=1, le=100),
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    query = (
        db.query(ShadowSession)
        .join(Trade, ShadowSession.real_trade_id == Trade.id)
        .filter(Trade.trader_id == user.id)
        .order_by(ShadowSession.started_at.desc())
    )

    if status:
        query = query.filter(ShadowSession.status == status)

    sessions = query.limit(limit).all()
    results = []

    for session in sessions:
        shadows = (
            db.query(ShadowTrade)
            .filter(ShadowTrade.session_id == session.id)
            .all()
        )

        results.append(ShadowSessionResponse(
            session_id=str(session.id),
            real_trade_id=str(session.real_trade_id),
            strategy_id=session.strategy_id,
            status=session.status,
            started_at=session.started_at,
            closed_at=session.closed_at,
            shadow_trades=[
                ShadowTradeResponse(
                    id=str(s.id),
                    shadow_type=s.shadow_type,
                    label=s.label,
                    symbol=s.symbol,
                    direction=s.direction,
                    entry_price=s.entry_price,
                    entry_time=s.entry_time,
                    position_size=s.position_size,
                    stop_loss=s.stop_loss,
                    take_profit=s.take_profit,
                    trailing_stop=s.trailing_stop,
                    status=s.status,
                    current_price=s.current_price,
                    unrealized_pnl=s.unrealized_pnl or 0.0,
                    realized_pnl=s.realized_pnl,
                    r_multiple=s.r_multiple,
                    max_favorable=s.max_favorable or 0.0,
                    max_adverse=s.max_adverse or 0.0,
                    exit_price=s.exit_price,
                    exit_time=s.exit_time,
                    exit_reason=s.exit_reason,
                    partial_exits=s.partial_exits or [],
                    remaining_size=s.remaining_size,
                    strategy_id=s.strategy_id,
                    price_updates=s.price_updates or 0,
                )
                for s in shadows
            ],
            summary=session.summary or {},
        ))

    return results


# ══════════════════════════════════════════════
#  GET /shadow/trade/{id} — Shadow trade detail
# ══════════════════════════════════════════════

@router.get("/trade/{shadow_trade_id}", response_model=ShadowTradeResponse)
def get_shadow_trade(
    shadow_trade_id: UUID,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    shadow = (
        db.query(ShadowTrade)
        .join(ShadowSession, ShadowTrade.session_id == ShadowSession.id)
        .join(Trade, ShadowSession.real_trade_id == Trade.id)
        .filter(ShadowTrade.id == shadow_trade_id, Trade.trader_id == user.id)
        .first()
    )
    if not shadow:
        raise HTTPException(status_code=404, detail="Shadow trade not found")

    return ShadowTradeResponse(
        id=str(shadow.id),
        shadow_type=shadow.shadow_type,
        label=shadow.label,
        symbol=shadow.symbol,
        direction=shadow.direction,
        entry_price=shadow.entry_price,
        entry_time=shadow.entry_time,
        position_size=shadow.position_size,
        stop_loss=shadow.stop_loss,
        take_profit=shadow.take_profit,
        trailing_stop=shadow.trailing_stop,
        status=shadow.status,
        current_price=shadow.current_price,
        unrealized_pnl=shadow.unrealized_pnl or 0.0,
        realized_pnl=shadow.realized_pnl,
        r_multiple=shadow.r_multiple,
        max_favorable=shadow.max_favorable or 0.0,
        max_adverse=shadow.max_adverse or 0.0,
        exit_price=shadow.exit_price,
        exit_time=shadow.exit_time,
        exit_reason=shadow.exit_reason,
        partial_exits=shadow.partial_exits or [],
        remaining_size=shadow.remaining_size,
        strategy_id=shadow.strategy_id,
        price_updates=shadow.price_updates or 0,
    )


# ══════════════════════════════════════════════
#  POST /shadow/price — Update shadow trades
# ══════════════════════════════════════════════

@router.post("/price/{real_trade_id}", response_model=PriceUpdateResponse)
def update_shadow_prices(
    real_trade_id: UUID,
    request: PriceUpdateRequest,
    db: DBSession = Depends(get_db),
    engine: ShadowEngine = Depends(get_shadow_engine),
    user=Depends(get_current_user),
):
    """Push a price update to all shadow trades for a real trade."""
    # Verify the real trade belongs to the current user
    trade = db.query(Trade).filter(Trade.id == real_trade_id, Trade.trader_id == user.id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    results = engine.update_session_prices(
        db=db,
        real_trade_id=real_trade_id,
        price=request.price,
        timestamp=request.timestamp,
    )
    return PriceUpdateResponse(results=results)


# ══════════════════════════════════════════════
#  GET /shadow/compare/{trade_id} — Comparison
# ══════════════════════════════════════════════

@router.get("/compare/{real_trade_id}", response_model=ComparisonResponse)
def compare_real_vs_shadow(
    real_trade_id: UUID,
    db: DBSession = Depends(get_db),
    engine: ShadowComparisonEngine = Depends(get_comparison_engine),
    user=Depends(get_current_user),
):
    """Get real vs shadow comparison for a closed trade."""
    # Verify the real trade belongs to the current user
    trade = db.query(Trade).filter(Trade.id == real_trade_id, Trade.trader_id == user.id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    report = engine.compare(db, real_trade_id)
    if not report:
        raise HTTPException(
            status_code=404,
            detail="No comparison available — trade may not be closed or no shadows exist"
        )

    return ComparisonResponse(**report.to_dict())