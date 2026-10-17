# api/routes/trades.py
"""
Trade API endpoints.
Wires HTTP requests to the TradePipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session as DBSession, joinedload

from api.auth import get_current_user

from api.schemas.trade_schemas import (
    TradeCloseRequest,
    TradeCloseResponse,
    TradeDetailResponse,
    TradeEntryRequest,
    TradeEntryResponse,
)
from core.database import get_db_dependency as get_db
from core.enums import EmotionalState, TradeDirection, TradeStatus
from core.models import StrategyAttribution, Trade
from core.trade_pipeline import TradePipeline, TradeInput

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/trades", tags=["trades"], dependencies=[Depends(get_current_user)])

# Pipeline instance — initialized at startup
_pipeline: TradePipeline | None = None


def get_pipeline() -> TradePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TradePipeline()
    return _pipeline


# ══════════════════════════════════════════════
#  POST /trades — Enter a new trade
# ══════════════════════════════════════════════

@router.post("", response_model=TradeEntryResponse, status_code=201)
def enter_trade(
    request: TradeEntryRequest,
    db: DBSession = Depends(get_db),
    pipeline: TradePipeline = Depends(get_pipeline),
    user=Depends(get_current_user),
):
    """
    Enter a new trade.

    Triggers the full attribution pipeline:
    - Strategy matching (Tier 2)
    - Override detection (Tier 3)
    - Behavioral event recording
    - Shadow trade creation (via event)
    """
    # Map request to internal DTO
    trade_input = TradeInput(
        symbol=request.symbol,
        direction=TradeDirection(request.direction),
        entry_price=request.entry_price,
        position_size=request.position_size,
        entry_time=request.entry_time or datetime.now(timezone.utc),
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        declared_strategy_id=request.declared_strategy_id,
        emotion_pre=(
            EmotionalState(request.emotion_pre)
            if request.emotion_pre else None
        ),
        conviction_score=request.conviction_score,
        notes=request.notes,
        account_balance=request.account_balance,
        source=request.source,
        broker=request.broker,
        external_id=request.external_id,
        trader_id=str(user.id),
    )

    result = pipeline.enter_trade(db, trade_input)

    if not result.success:
        raise HTTPException(status_code=400, detail=result.errors)

    return TradeEntryResponse(
        success=True,
        trade_id=str(result.trade_id),
        attribution=(
            result.attribution.to_dict() if result.attribution else None
        ),
        shadow_trade_ids=[str(sid) for sid in result.shadow_ids],
        insights=result.insights,
        warnings=result.warnings,
        errors=result.errors,
    )


# ══════════════════════════════════════════════
#  POST /trades/{id}/close — Close a trade
# ══════════════════════════════════════════════

@router.post("/{trade_id}/close", response_model=TradeCloseResponse)
def close_trade(
    trade_id: UUID,
    request: TradeCloseRequest,
    db: DBSession = Depends(get_db),
    pipeline: TradePipeline = Depends(get_pipeline),
    user=Depends(get_current_user),
):
    """
    Close an open trade.

    Calculates P&L, R-multiple, backfills behavioral event outcomes,
    and publishes TRADE_CLOSED event (triggers shadow comparison).
    """
    # Verify ownership before closing
    trade = db.query(Trade).filter(Trade.id == trade_id, Trade.trader_id == user.id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    result = pipeline.close_trade(
        db=db,
        trade_id=trade_id,
        exit_price=request.exit_price,
        exit_time=request.exit_time,
        emotion_post=(
            EmotionalState(request.emotion_post)
            if request.emotion_post else None
        ),
        exit_notes=request.exit_notes,
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))

    return TradeCloseResponse(**result)


# ══════════════════════════════════════════════
#  GET /trades — List trades
# ══════════════════════════════════════════════

@router.get("", response_model=list[TradeDetailResponse])
def list_trades(
    status: Optional[str] = Query(None, pattern="^(open|closed|cancelled)$"),
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """List trades with optional filters."""
    query = (
        db.query(Trade)
        .options(joinedload(Trade.attribution))
        .filter(Trade.trader_id == user.id)
        .order_by(Trade.entry_time.desc())
    )

    if status:
        query = query.filter(Trade.status == TradeStatus(status))
    if strategy_id:
        query = query.filter(Trade.declared_strategy_id == strategy_id)
    if symbol:
        query = query.filter(Trade.symbol == symbol.upper())

    trades = query.offset(offset).limit(limit).all()

    results = []
    for trade in trades:
        attr = trade.attribution

        detail = TradeDetailResponse(
            id=str(trade.id),
            symbol=trade.symbol,
            direction=trade.direction.value,
            status=trade.status.value,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            position_size=trade.position_size,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            pnl=trade.pnl,
            pnl_pct=trade.pnl_pct,
            r_multiple=trade.r_multiple,
            commission=trade.commission,
            risk_amount=trade.risk_amount,
            risk_pct=trade.risk_pct,
            session_type=trade.session_type.value if trade.session_type else None,
            market_regime=trade.market_regime.value if trade.market_regime else None,
            declared_strategy_id=trade.declared_strategy_id,
            emotion_pre=trade.emotion_pre.value if trade.emotion_pre else None,
            emotion_post=trade.emotion_post.value if trade.emotion_post else None,
            conviction_score=trade.conviction_score,
            notes=trade.notes,
            source=trade.source,
            resolved_strategy=attr.resolved_strategy_id if attr else None,
            discipline_score=attr.discipline_score if attr else None,
            confidence_score=attr.confidence_score if attr else None,
            override_count=attr.tier3_override_count if attr else 0,
        )
        results.append(detail)

    return results


# ══════════════════════════════════════════════
#  GET /trades/{id} — Trade detail
# ══════════════════════════════════════════════

@router.get("/{trade_id}", response_model=TradeDetailResponse)
def get_trade(
    trade_id: UUID,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get full trade detail including attribution."""
    trade = (
        db.query(Trade)
        .options(joinedload(Trade.attribution))
        .filter(Trade.id == trade_id, Trade.trader_id == user.id)
        .first()
    )
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    attr = trade.attribution

    return TradeDetailResponse(
        id=str(trade.id),
        symbol=trade.symbol,
        direction=trade.direction.value,
        status=trade.status.value,
        entry_price=trade.entry_price,
        exit_price=trade.exit_price,
        entry_time=trade.entry_time,
        exit_time=trade.exit_time,
        position_size=trade.position_size,
        stop_loss=trade.stop_loss,
        take_profit=trade.take_profit,
        pnl=trade.pnl,
        pnl_pct=trade.pnl_pct,
        r_multiple=trade.r_multiple,
        risk_amount=trade.risk_amount,
        risk_pct=trade.risk_pct,
        session_type=trade.session_type.value if trade.session_type else None,
        market_regime=trade.market_regime.value if trade.market_regime else None,
        declared_strategy_id=trade.declared_strategy_id,
        emotion_pre=trade.emotion_pre.value if trade.emotion_pre else None,
        emotion_post=trade.emotion_post.value if trade.emotion_post else None,
        conviction_score=trade.conviction_score,
        notes=trade.notes,
        source=trade.source,
        resolved_strategy=attr.resolved_strategy_id if attr else None,
        discipline_score=attr.discipline_score if attr else None,
        confidence_score=attr.confidence_score if attr else None,
        override_count=attr.tier3_override_count if attr else 0,
    )