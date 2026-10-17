# dashboard.py
# api/routes/dashboard.py
"""
Dashboard data aggregation endpoint.
Single call to get everything the dashboard needs.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.auth import get_current_user
from sqlalchemy import and_
from datetime import datetime, timedelta
from typing import Optional

from core.database import get_db_dependency
from core.models import Trade
from core.enums import TradeStatus
from shadow.models import ShadowTrade
from api.dependencies import get_orchestrator, get_query_engine

router = APIRouter(prefix="/dashboard", tags=["dashboard"], dependencies=[Depends(get_current_user)])


@router.get("/")
def get_dashboard(
    current_regime: Optional[str] = None,
    db: Session = Depends(get_db_dependency),
    user=Depends(get_current_user),
):
    """
    Single endpoint for all dashboard data.
    Frontend calls this on load and refresh.
    """
    today = datetime.combine(datetime.now().date(), datetime.min.time())

    # ── Today's trades ──
    today_trades = db.query(Trade).filter(
        Trade.trader_id == user.id,
        Trade.entry_time >= today,
    ).order_by(Trade.entry_time.desc()).all()

    today_closed = [t for t in today_trades if t.status == TradeStatus.CLOSED]
    today_open = [t for t in today_trades if t.status == TradeStatus.OPEN]
    today_pnl = sum(t.pnl for t in today_closed if t.pnl)
    today_winners = sum(1 for t in today_closed if t.pnl and t.pnl > 0)

    # ── Open shadow trades ──
    open_shadows = db.query(ShadowTrade).filter(
        ShadowTrade.status == TradeStatus.OPEN
    ).all()

    # ── Recent trades (last 7 days) ──
    week_ago = datetime.now() - timedelta(days=7)
    recent_trades = db.query(Trade).filter(
        and_(
            Trade.trader_id == user.id,
            Trade.status == TradeStatus.CLOSED,
            Trade.entry_time >= week_ago,
        )
    ).order_by(Trade.entry_time.desc()).all()

    week_pnl = sum(t.pnl for t in recent_trades if t.pnl)

    # ── Discipline today ──
    today_discipline = []
    for t in today_trades:
        if t.tags and "discipline_score" in t.tags:
            today_discipline.append(t.tags["discipline_score"])

    avg_discipline_today = (
        sum(today_discipline) / len(today_discipline)
        if today_discipline else None
    )

    # ── Consecutive losses ──
    all_recent = db.query(Trade).filter(
        Trade.trader_id == user.id,
        Trade.status == TradeStatus.CLOSED,
    ).order_by(Trade.entry_time.desc()).limit(10).all()

    consecutive_losses = 0
    for t in all_recent:
        if t.pnl and t.pnl < 0:
            consecutive_losses += 1
        else:
            break

    # ── Strategy recommendations for regime ──
    regime_recs = []
    if current_regime:
        query = get_query_engine()
        recs = query.get_strategies_for_regime(current_regime)
        regime_recs = [
            {
                "strategy": r.strategy_name,
                "display_name": r.display_name,
                "fitness": r.fitness_score,
                "sizing": r.sizing_recommendation,
            }
            for r in recs[:5]
        ]

    # ── Intelligence alerts ──
    orchestrator = get_orchestrator(db)
    intelligence = orchestrator.daily_intelligence_summary(current_regime)

    return {
        "timestamp": datetime.now().isoformat(),
        "current_regime": current_regime,

        "today": {
            "total_trades": len(today_trades),
            "open_trades": len(today_open),
            "closed_trades": len(today_closed),
            "pnl": round(today_pnl, 2),
            "winners": today_winners,
            "losers": len(today_closed) - today_winners,
            "win_rate": round(today_winners / len(today_closed) * 100, 1) if today_closed else 0,
            "avg_discipline": round(avg_discipline_today, 1) if avg_discipline_today else None,
        },

        "week": {
            "pnl": round(week_pnl, 2),
            "trades": len(recent_trades),
        },

        "open_trades": [
            {
                "id": t.id,
                "symbol": t.symbol,
                "direction": t.direction.value,
                "entry_price": t.entry_price,
                "strategy": t.strategy_name,
                "entry_time": t.entry_time.isoformat(),
            }
            for t in today_open
        ],

        "open_shadows": len(open_shadows),

        "alerts": {
            "consecutive_losses": consecutive_losses,
            "critical_alerts": intelligence.get("critical_count", 0),
            "warnings": intelligence.get("warning_count", 0),
            "insights": intelligence.get("insights", [])[:5],
        },

        "regime_recommendations": regime_recs,
    }