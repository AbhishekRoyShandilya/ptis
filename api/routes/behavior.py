# behavior.py
# api/routes/behavior.py
"""Behavioral intelligence API endpoints."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession

from api.auth import get_current_user

from api.schemas.analytics_schemas import (
    BaselineResponse,
    BehavioralDashboardResponse,
    BiasDetectionResponse,
    DeviationAlertResponse,
    ImprovementResponse,
)
from behavioral.baseline_modeler import BaselineModeler
from behavioral.cognitive_bias_detector import CognitiveBiasDetector
from behavioral.deviation_detector import DeviationDetector
from behavioral.improvement_tracker import ImprovementTracker
from core.database import get_db_dependency as get_db
from core.session_manager import get_session_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/behavior", tags=["behavioral intelligence"], dependencies=[Depends(get_current_user)])

_baseline_cache: Optional[dict] = None


# ══════════════════════════════════════════════
#  GET /behavior/baseline
# ══════════════════════════════════════════════

@router.get("/baseline", response_model=BaselineResponse)
def get_baseline(
    lookback_days: int = 90,
    db: DBSession = Depends(get_db),
):
    """Get behavioral baseline computed from historical trades."""
    modeler = BaselineModeler(lookback_days=lookback_days)
    baseline = modeler.compute_baseline(db)
    return BaselineResponse(**baseline.to_dict())


# ══════════════════════════════════════════════
#  GET /behavior/improvement
# ══════════════════════════════════════════════

@router.get("/improvement", response_model=ImprovementResponse)
def get_improvement(
    weeks: int = 12,
    db: DBSession = Depends(get_db),
):
    """Get improvement tracking report with weekly trends."""
    tracker = ImprovementTracker(weeks_lookback=weeks)
    report = tracker.compute_report(db)
    return ImprovementResponse(**report.to_dict())


# ══════════════════════════════════════════════
#  GET /behavior/biases
# ══════════════════════════════════════════════

@router.get("/biases", response_model=list[BiasDetectionResponse])
def detect_biases(
    lookback_days: int = 60,
    db: DBSession = Depends(get_db),
):
    """Detect cognitive biases from trade history."""
    detector = CognitiveBiasDetector(lookback_days=lookback_days)
    detections = detector.detect_all(db)
    return [BiasDetectionResponse(**d.to_dict()) for d in detections]


# ══════════════════════════════════════════════
#  GET /behavior/alerts
# ══════════════════════════════════════════════

@router.get("/alerts", response_model=list[DeviationAlertResponse])
def get_session_alerts(
    db: DBSession = Depends(get_db),
):
    """Get active behavioral deviation alerts for current session."""
    session_mgr = get_session_manager()
    session_ctx = session_mgr.to_session_context()

    # Compute baseline for deviation detection
    modeler = BaselineModeler(lookback_days=60)
    baseline = modeler.compute_baseline(db)

    detector = DeviationDetector(baseline)
    alerts = detector.check_session(session_ctx)

    return [DeviationAlertResponse(**a.to_dict()) for a in alerts]


# ══════════════════════════════════════════════
#  GET /behavior/dashboard
# ══════════════════════════════════════════════

@router.get("/dashboard", response_model=BehavioralDashboardResponse)
def get_behavioral_dashboard(
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Comprehensive behavioral intelligence dashboard data."""
    # Baseline
    modeler = BaselineModeler(lookback_days=90)
    baseline = modeler.compute_baseline(db)

    # Improvement
    tracker = ImprovementTracker(weeks_lookback=12)
    improvement = tracker.compute_report(db)

    # Biases
    bias_detector = CognitiveBiasDetector(lookback_days=60)
    biases = bias_detector.detect_all(db)

    # Session alerts
    session_mgr = get_session_manager()
    session_ctx = session_mgr.to_session_context()
    deviation_detector = DeviationDetector(baseline)
    alerts = deviation_detector.check_session(session_ctx)

    # Today's discipline
    from datetime import datetime, date, timezone
    from sqlalchemy import func
    from core.models import StrategyAttribution, Trade
    from core.enums import TradeStatus

    today = date.today()
    today_start = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)

    today_trades = (
        db.query(Trade.id)
        .filter(Trade.trader_id == user.id, Trade.entry_time >= today_start)
        .all()
    )
    today_ids = [t[0] for t in today_trades]

    discipline_today = None
    clean_rate_today = None

    if today_ids:
        scores = (
            db.query(StrategyAttribution.discipline_score)
            .filter(StrategyAttribution.trade_id.in_(today_ids))
            .all()
        )
        valid_scores = [s[0] for s in scores if s[0] is not None]
        if valid_scores:
            discipline_today = sum(valid_scores) / len(valid_scores)

        from core.enums import BehavioralEventType
        from core.models import BehavioralEvent

        clean_count = (
            db.query(func.count(BehavioralEvent.id))
            .filter(
                BehavioralEvent.trade_id.in_(today_ids),
                BehavioralEvent.event_type == BehavioralEventType.CLEAN_EXECUTION,
            )
            .scalar()
        )
        clean_rate_today = clean_count / len(today_ids)

    return BehavioralDashboardResponse(
        baseline=BaselineResponse(**baseline.to_dict()),
        improvement=ImprovementResponse(**improvement.to_dict()),
        active_biases=[BiasDetectionResponse(**b.to_dict()) for b in biases[:5]],
        session_alerts=[DeviationAlertResponse(**a.to_dict()) for a in alerts],
        discipline_score_today=discipline_today,
        clean_execution_rate_today=clean_rate_today,
    )


# ══════════════════════════════════════════════
#  GET /behavior/deviations
# ══════════════════════════════════════════════

@router.get("/deviations")
def get_deviations(
    lookback_days: int = 30,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get recent trade deviations from baseline behavior."""
    modeler = BaselineModeler(lookback_days=max(lookback_days, 60))
    baseline = modeler.compute_baseline(db)

    from datetime import datetime, timedelta, timezone
    from core.models import Trade
    from core.enums import TradeStatus

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    trades = (
        db.query(Trade)
        .filter(
            Trade.trader_id == user.id,
            Trade.status == TradeStatus.CLOSED,
            Trade.entry_time >= cutoff,
        )
        .order_by(Trade.entry_time.desc())
        .limit(50)
        .all()
    )

    deviations = []
    for trade in trades:
        issues = []
        # Check R-multiple deviation
        if trade.r_multiple is not None and baseline.to_dict().get("avg_r"):
            avg_r = baseline.to_dict()["avg_r"]
            if trade.r_multiple < avg_r - 1.5:
                issues.append(f"R-multiple ({trade.r_multiple:.1f}) well below average ({avg_r:.1f})")
        # Check override count
        if trade.override_count and trade.override_count > 0:
            issues.append(f"{trade.override_count} rule override(s)")
        if issues:
            deviations.append({
                "trade_id": str(trade.id),
                "symbol": trade.symbol,
                "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
                "pnl": trade.pnl,
                "r_multiple": trade.r_multiple,
                "issues": issues,
            })

    return {"deviations": deviations, "total": len(deviations)}


# ══════════════════════════════════════════════
#  GET /behavior/discipline
# ══════════════════════════════════════════════

@router.get("/discipline")
def get_discipline_score(
    lookback_days: int = 30,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get discipline score summary."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import func
    from core.models import StrategyAttribution, Trade
    from core.enums import TradeStatus

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    trade_ids = [
        t[0] for t in
        db.query(Trade.id)
        .filter(
            Trade.trader_id == user.id,
            Trade.entry_time >= cutoff,
        )
        .all()
    ]

    if not trade_ids:
        return {"score": None, "trades_scored": 0, "trend": "insufficient_data"}

    scores = (
        db.query(StrategyAttribution.discipline_score)
        .filter(StrategyAttribution.trade_id.in_(trade_ids))
        .all()
    )
    valid = [s[0] for s in scores if s[0] is not None]

    if not valid:
        return {"score": None, "trades_scored": 0, "trend": "insufficient_data"}

    avg_score = sum(valid) / len(valid)

    # Trend: compare first half vs second half
    mid = len(valid) // 2
    trend = "stable"
    if mid > 2:
        first_half = sum(valid[:mid]) / mid
        second_half = sum(valid[mid:]) / (len(valid) - mid)
        if second_half > first_half + 0.05:
            trend = "improving"
        elif second_half < first_half - 0.05:
            trend = "declining"

    return {
        "score": round(avg_score, 2),
        "trades_scored": len(valid),
        "trend": trend,
        "high": round(max(valid), 2),
        "low": round(min(valid), 2),
    }