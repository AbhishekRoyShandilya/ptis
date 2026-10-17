# api/routes/analytics.py (add strategy performance + deviation analysis)
"""Extended analytics endpoints."""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import get_current_user

from sqlalchemy.orm import Session as DBSession

from analytics.strategy_performance import StrategyPerformanceTracker
from behavioral.deviation_attribution import DeviationAttributionEngine
from core.database import get_db_dependency as get_db
from core.models import BehavioralEvent, OverrideEvent, StrategyAttribution, Trade
from core.enums import TradeStatus



logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"], dependencies=[Depends(get_current_user)])


class StrategyPerformanceResponse(BaseModel):
    strategy_id: str
    trade_count: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_r: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    avg_discipline: float = 0.0
    edge_trend: str = "stable"
    streaks: dict = {}
    recent_vs_historical: dict = {}
    regime_breakdown: dict = {}
    session_breakdown: dict = {}


class DeviationAnalysisResponse(BaseModel):
    trade_id: str
    primary_cause: str
    cause_confidence: float = 0.0
    breakdown: dict = {}
    emotional_evidence: list[str] = []
    market_evidence: list[str] = []
    strategy_evidence: list[str] = []
    regime: dict = {}
    risk_context: dict = {}
    verdict: dict = {}
    action_items: list[str] = []

# ══════════════════════════════════════════════
#  GET /analytics/strategy/{id}
# ══════════════════════════════════════════════

@router.get("/strategy/{strategy_id}", response_model=StrategyPerformanceResponse)
def get_strategy_performance(
    strategy_id: str,
    lookback_days: int = 180,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get detailed performance metrics for a specific strategy."""
    tracker = StrategyPerformanceTracker()
    metrics = tracker.compute(db, strategy_id, lookback_days)

    if metrics.trade_count == 0:
        raise HTTPException(status_code=404, detail="No trades found for this strategy")

    return StrategyPerformanceResponse(**metrics.to_dict())


# ══════════════════════════════════════════════
#  GET /analytics/strategies — All strategies
# ══════════════════════════════════════════════

@router.get("/strategies", response_model=list[StrategyPerformanceResponse])
def get_all_strategy_performance(
    lookback_days: int = 180,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get performance metrics for all strategies."""
    tracker = StrategyPerformanceTracker()
    all_metrics = tracker.compute_all(db, lookback_days)
    return [StrategyPerformanceResponse(**m.to_dict()) for m in all_metrics.values()]


# ══════════════════════════════════════════════
#  GET /analytics/deviation/{trade_id}
# ══════════════════════════════════════════════

@router.get("/deviation/{trade_id}", response_model=DeviationAnalysisResponse)
def analyze_trade_deviation(
    trade_id: UUID,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Analyze WHY a trade deviated from expected outcome.
    
    Determines if the cause was:
    - Emotional (trader broke rules)
    - Market condition (regime changed)
    - Strategy edge loss (strategy losing edge)
    - Random variance (normal loss)
    """
    trade = db.query(Trade).filter(Trade.id == trade_id, Trade.trader_id == user.id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    if trade.pnl is None:
        raise HTTPException(status_code=400, detail="Trade must be closed for deviation analysis")

    attribution = (
        db.query(StrategyAttribution)
        .filter(StrategyAttribution.trade_id == trade_id)
        .first()
    )

    overrides = (
        db.query(OverrideEvent)
        .filter(OverrideEvent.trade_id == trade_id)
        .all()
    )

    behavioral = (
        db.query(BehavioralEvent)
        .filter(BehavioralEvent.trade_id == trade_id)
        .all()
    )

    # Get recent strategy performance for context
    recent_perf = None
    if trade.declared_strategy_id:
        tracker = StrategyPerformanceTracker()
        recent_perf = tracker.get_recent_performance(db, trade.declared_strategy_id)

    engine = DeviationAttributionEngine()
    analysis = engine.analyze(
        trade=trade,
        attribution=attribution,
        overrides=overrides,
        behavioral_events=behavioral,
        entry_market=None,  # Would be fetched from enrichment cache
        exit_market=None,
        recent_strategy_performance=recent_perf,
    )

    return DeviationAnalysisResponse(**analysis.to_dict())


# ══════════════════════════════════════════════
#  GET /analytics/performance — Overall performance
# ══════════════════════════════════════════════

@router.get("/performance")
def get_overall_performance(
    lookback_days: int = 180,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Overall performance summary across all strategies."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import func

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    trades = (
        db.query(Trade)
        .filter(
            Trade.trader_id == user.id,
            Trade.status == TradeStatus.CLOSED,
            Trade.entry_time >= cutoff,
        )
        .all()
    )

    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "total_pnl": 0,
            "avg_r": 0, "profit_factor": 0, "best_trade": None, "worst_trade": None,
        }

    winners = [t for t in trades if t.pnl and t.pnl > 0]
    losers = [t for t in trades if t.pnl and t.pnl < 0]
    total_pnl = sum(t.pnl or 0 for t in trades)
    gross_profit = sum(t.pnl for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
    r_vals = [t.r_multiple for t in trades if t.r_multiple is not None]

    # ── R-multiple distribution (buckets) ──
    def _r_bucket(r: float) -> str:
        if r < -2: return "<-2R"
        if r < -1: return "-2R to -1R"
        if r < 0: return "-1R to 0R"
        if r < 1: return "0R to 1R"
        if r < 2: return "1R to 2R"
        return ">2R"

    _bucket_order = ["<-2R", "-2R to -1R", "-1R to 0R", "0R to 1R", "1R to 2R", ">2R"]
    _rdist = {b: 0 for b in _bucket_order}
    for r in r_vals:
        _rdist[_r_bucket(r)] += 1
    r_distribution = [{"r": b, "count": _rdist[b]} for b in _bucket_order]

    # ── Session breakdown by IST time-of-day ──
    def _session(dt) -> str:
        ist = dt + timedelta(hours=5, minutes=30)
        h = ist.hour
        if h < 9: return "Pre-mkt"
        if h < 11: return "Morning"
        if h < 14: return "Midday"
        return "Afternoon"

    _sess: dict = {}
    for t in trades:
        if not t.entry_time:
            continue
        s = _session(t.entry_time)
        d = _sess.setdefault(s, {"trades": 0, "wins": 0})
        d["trades"] += 1
        if t.pnl and t.pnl > 0:
            d["wins"] += 1
    sessions = [
        {
            "session": s,
            "trades": _sess[s]["trades"],
            "wr": round(_sess[s]["wins"] / _sess[s]["trades"] * 100) if _sess[s]["trades"] else 0,
        }
        for s in ["Pre-mkt", "Morning", "Midday", "Afternoon"] if s in _sess
    ]

    return {
        "total_trades": len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": len(winners) / len(trades) if trades else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_r": round(sum(r_vals) / len(r_vals), 2) if r_vals else 0,
        "expectancy": round(total_pnl / len(trades), 2) if trades else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "best_trade": max((t.pnl or 0) for t in trades),
        "worst_trade": min((t.pnl or 0) for t in trades),
        "r_distribution": r_distribution,
        "sessions": sessions,
        "lookback_days": lookback_days,
    }


# ══════════════════════════════════════════════
#  GET /analytics/equity — Equity curve data
# ══════════════════════════════════════════════

@router.get("/equity")
def get_equity_curve(
    lookback_days: int = 180,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Cumulative PnL series for equity curve charting."""
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    trades = (
        db.query(Trade)
        .filter(
            Trade.trader_id == user.id,
            Trade.status == TradeStatus.CLOSED,
            Trade.exit_time >= cutoff,
        )
        .order_by(Trade.exit_time.asc())
        .all()
    )

    cumulative = 0.0
    points = []
    for t in trades:
        cumulative += t.pnl or 0
        points.append({
            "date": t.exit_time.isoformat() if t.exit_time else None,
            "pnl": round(t.pnl or 0, 2),
            "cumulative_pnl": round(cumulative, 2),
            "symbol": t.symbol,
        })

    return {"points": points, "total_pnl": round(cumulative, 2)}


# ══════════════════════════════════════════════
#  GET /analytics/session — Current session stats
# ══════════════════════════════════════════════

@router.get("/session")
def get_session_analytics(
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Today's trading session analytics."""
    from datetime import datetime, date, timezone

    today = date.today()
    today_start = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)

    trades = (
        db.query(Trade)
        .filter(Trade.trader_id == user.id, Trade.entry_time >= today_start)
        .all()
    )

    open_trades = [t for t in trades if t.status == TradeStatus.OPEN]
    closed = [t for t in trades if t.status == TradeStatus.CLOSED]
    winners = [t for t in closed if t.pnl and t.pnl > 0]

    return {
        "date": today.isoformat(),
        "total_trades": len(trades),
        "open": len(open_trades),
        "closed": len(closed),
        "winners": len(winners),
        "losers": len(closed) - len(winners),
        "win_rate": len(winners) / len(closed) if closed else 0,
        "session_pnl": round(sum(t.pnl or 0 for t in closed), 2),
    }


# ══════════════════════════════════════════════
#  GET /analytics/streak — Win/loss streak
# ══════════════════════════════════════════════

@router.get("/streak")
def get_streak_analytics(
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Current and historical win/loss streaks."""
    trades = (
        db.query(Trade)
        .filter(
            Trade.trader_id == user.id,
            Trade.status == TradeStatus.CLOSED,
        )
        .order_by(Trade.exit_time.desc())
        .limit(100)
        .all()
    )

    current_streak = 0
    streak_type = "none"
    if trades:
        first_pnl = trades[0].pnl or 0
        streak_type = "win" if first_pnl > 0 else "loss"
        for t in trades:
            pnl = t.pnl or 0
            if (streak_type == "win" and pnl > 0) or (streak_type == "loss" and pnl <= 0):
                current_streak += 1
            else:
                break

    # Max streaks
    max_win, max_loss, cur_w, cur_l = 0, 0, 0, 0
    for t in reversed(trades):
        if (t.pnl or 0) > 0:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_win = max(max_win, cur_w)
        max_loss = max(max_loss, cur_l)

    return {
        "current_streak": current_streak,
        "streak_type": streak_type,
        "max_win_streak": max_win,
        "max_loss_streak": max_loss,
    }


# ══════════════════════════════════════════════
#  GET /analytics/behavior — Behavioral summary
# ══════════════════════════════════════════════

@router.get("/behavior")
def get_behavior_analytics(
    lookback_days: int = 90,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Behavioral analytics: discipline, overrides, clean execution rate."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import func

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    trade_ids = [
        t[0] for t in
        db.query(Trade.id)
        .filter(Trade.trader_id == user.id, Trade.entry_time >= cutoff)
        .all()
    ]

    if not trade_ids:
        return {"trades_analyzed": 0, "avg_discipline": None, "override_count": 0, "clean_rate": None}

    scores = (
        db.query(StrategyAttribution.discipline_score)
        .filter(StrategyAttribution.trade_id.in_(trade_ids))
        .all()
    )
    valid = [s[0] for s in scores if s[0] is not None]

    override_count = (
        db.query(OverrideEvent)
        .filter(OverrideEvent.trade_id.in_(trade_ids))
        .count()
    )

    return {
        "trades_analyzed": len(trade_ids),
        "avg_discipline": round(sum(valid) / len(valid), 2) if valid else None,
        "override_count": override_count,
        "override_rate": round(override_count / len(trade_ids), 2) if trade_ids else 0,
    }


# ══════════════════════════════════════════════
#  GET /analytics/trades/{trade_id}/decision-quality
# ══════════════════════════════════════════════

@router.get("/trades/{trade_id}/decision-quality")
def get_trade_decision_quality(
    trade_id: UUID,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Alias for deviation analysis — frontend compatibility."""
    trade = db.query(Trade).filter(Trade.id == trade_id, Trade.trader_id == user.id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    if trade.pnl is None:
        raise HTTPException(status_code=400, detail="Trade must be closed")

    attribution = (
        db.query(StrategyAttribution)
        .filter(StrategyAttribution.trade_id == trade_id)
        .first()
    )
    overrides = db.query(OverrideEvent).filter(OverrideEvent.trade_id == trade_id).all()
    behavioral = db.query(BehavioralEvent).filter(BehavioralEvent.trade_id == trade_id).all()

    recent_perf = None
    if trade.declared_strategy_id:
        tracker = StrategyPerformanceTracker()
        recent_perf = tracker.get_recent_performance(db, trade.declared_strategy_id)

    engine = DeviationAttributionEngine()
    analysis = engine.analyze(
        trade=trade,
        attribution=attribution,
        overrides=overrides,
        behavioral_events=behavioral,
        entry_market=None,
        exit_market=None,
        recent_strategy_performance=recent_perf,
    )
    return DeviationAnalysisResponse(**analysis.to_dict())


# ══════════════════════════════════════════════
#  GET /analytics/strategies/{id}/performance
# ══════════════════════════════════════════════

@router.get("/strategies/{strategy_id}/performance", response_model=StrategyPerformanceResponse)
def get_strategy_performance_by_id(
    strategy_id: str,
    lookback_days: int = 180,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get performance for a specific strategy by ID (frontend alias)."""
    tracker = StrategyPerformanceTracker()
    metrics = tracker.compute(db, strategy_id, lookback_days)

    if metrics.trade_count == 0:
        raise HTTPException(status_code=404, detail="No trades found for this strategy")

    return StrategyPerformanceResponse(**metrics.to_dict())
