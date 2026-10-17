# comparison.py
# api/routes/comparison.py
"""
Real vs Shadow comparison endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.auth import get_current_user
from datetime import date, datetime
from typing import Optional

from core.database import get_db_dependency
from core.models import Trade
from shadow.comparison import ComparisonEngine

router = APIRouter(prefix="/comparison", tags=["comparison"], dependencies=[Depends(get_current_user)])


@router.get("/trade/{trade_id}")
def compare_trade(
    trade_id: str,
    db: Session = Depends(get_db_dependency),
    user=Depends(get_current_user),
):
    """Compare a real trade with its shadow trades."""
    trade = db.query(Trade).filter(Trade.id == trade_id, Trade.trader_id == user.id).first()
    if not trade:
        raise HTTPException(404, "Trade not found")
    engine = ComparisonEngine(db)
    comparison = engine.compare_trade(trade_id)

    return {
        "real_trade_id": comparison.real_trade_id,
        "symbol": comparison.symbol,
        "direction": comparison.direction,
        "real": {
            "entry": comparison.real_entry,
            "exit": comparison.real_exit,
            "pnl": comparison.real_pnl,
            "r": comparison.real_r,
            "hold_minutes": comparison.real_hold_minutes,
        },
        "shadows": comparison.shadows,
        "divergences": comparison.divergences,
        "best_outcome": comparison.best_outcome,
        "pnl_left_on_table": comparison.pnl_left_on_table,
        "key_insight": comparison.key_insight,
    }


@router.get("/session/{session_date}")
def compare_session(
    session_date: str,
    db: Session = Depends(get_db_dependency),
):
    """Compare full trading session: real vs shadow."""
    try:
        parsed_date = date.fromisoformat(session_date)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    engine = ComparisonEngine(db)
    comparison = engine.compare_session(parsed_date)

    return {
        "date": comparison.date.isoformat(),
        "real_summary": comparison.real_summary,
        "mirror_summary": comparison.mirror_summary,
        "strategy_summary": comparison.strategy_summary,
        "divergence_summary": comparison.divergence_summary,
        "ai_insights": comparison.ai_insights,
        "trade_count": len(comparison.trade_comparisons),
    }


@router.get("/cumulative")
def cumulative_comparison(
    days: int = 30,
    db: Session = Depends(get_db_dependency),
):
    """Cumulative comparison over N days."""
    engine = ComparisonEngine(db)
    return engine.get_cumulative_comparison(days)


@router.get("/divergence-patterns")
def divergence_patterns(
    days: int = 30,
    db: Session = Depends(get_db_dependency),
):
    """Analyze divergence patterns for learning."""
    engine = ComparisonEngine(db)
    return engine.get_divergence_patterns(days)