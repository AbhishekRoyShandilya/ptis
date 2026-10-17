# regime.py
# api/routes/regime.py
"""
Market regime endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.auth import get_current_user
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from core.database import get_db_dependency
from core.enums import MarketRegime
from core.models import Trade
from shadow.models import ShadowTrade
from strategy_engine.regime_detector import MarketRegimeDetector
from api.dependencies import get_query_engine
import pandas as pd

router = APIRouter(prefix="/regime", tags=["regime"], dependencies=[Depends(get_current_user)])


class RegimeSetRequest(BaseModel):
    """Manual regime setting."""
    regime: str
    confidence: Optional[float] = 0.8
    notes: Optional[str] = None


@router.post("/set")
def set_current_regime(
    request: RegimeSetRequest,
    db: Session = Depends(get_db_dependency),
):
    """
    Manually set current market regime.
    For Phase 1, you identify the regime yourself.
    System learns which strategies work in each.
    """
    from core.models import MarketRegimeLog

    log = MarketRegimeLog(
        timestamp=datetime.now(),
        regime=MarketRegime(request.regime),
        confidence=request.confidence,
        indicators={"source": "manual", "notes": request.notes},
    )
    db.add(log)
    db.commit()

    # Get strategy recommendations
    query = get_query_engine()
    recs = query.get_strategies_for_regime(request.regime)

    return {
        "regime": request.regime,
        "recommendations": [
            {
                "strategy": r.strategy_name,
                "display_name": r.display_name,
                "fitness": r.fitness_score,
                "sizing": r.sizing_recommendation,
                "notes": r.notes,
            }
            for r in recs
        ],
    }


@router.post("/detect")
def auto_detect_regime(
    symbol: str = "NIFTY",
    timeframe: str = "1d",
    db: Session = Depends(get_db_dependency),
):
    """
    Auto-detect current market regime using Angel One historical data.
    Saves result to DB so /regime/current returns it.
    """
    from core.models import MarketRegimeLog
    try:
        from ingestion.market_data_service import get_market_data_service
        from datetime import date, timedelta
        svc = get_market_data_service()
        if not svc or not svc.has_historical:
            return {"error": "No historical data service available"}

        candles = svc._fetch_candles_raw(
            symbol=symbol,
            interval=timeframe,
            from_date=date.today() - timedelta(days=120),
            to_date=date.today(),
        )
        if len(candles) < 50:
            return {"error": f"Insufficient candles: {len(candles)} (need 50+)"}

        df = pd.DataFrame(candles)
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "datetime"})

        detector = MarketRegimeDetector()
        analysis = detector.detect(df)
        if not analysis or not analysis.regime:
            return {"error": "Regime detection returned no result"}

        import numpy as np
        def _native(v):
            if isinstance(v, (np.integer,)): return int(v)
            if isinstance(v, (np.floating,)): return float(v)
            if isinstance(v, (np.bool_,)): return bool(v)
            return v

        indicators = {k: _native(v) for k, v in (analysis.indicators or {}).items()}
        indicators.update({"symbol": symbol, "timeframe": timeframe, "source": "auto_detect"})

        log = MarketRegimeLog(
            timestamp=datetime.now(),
            regime=analysis.regime,
            confidence=analysis.confidence,
            indicators=indicators,
        )
        db.add(log)
        db.commit()

        return {
            "regime": analysis.regime.value,
            "confidence": analysis.confidence,
            "symbol": symbol,
            "candles_used": len(candles),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/current")
def get_current_regime(db: Session = Depends(get_db_dependency)):
    """Get the most recently set regime."""
    from core.models import MarketRegimeLog

    latest = db.query(MarketRegimeLog).order_by(
        MarketRegimeLog.timestamp.desc()
    ).first()

    if not latest:
        return {"regime": "unknown", "message": "No regime set yet"}

    return {
        "regime": latest.regime.value,
        "confidence": latest.confidence,
        "set_at": latest.timestamp.isoformat(),
        "indicators": latest.indicators,
    }


@router.get("/history")
def get_regime_history(
    days: int = 30,
    db: Session = Depends(get_db_dependency),
):
    """Get regime history."""
    from core.models import MarketRegimeLog
    from datetime import timedelta

    cutoff = datetime.now() - timedelta(days=days)
    logs = db.query(MarketRegimeLog).filter(
        MarketRegimeLog.timestamp >= cutoff
    ).order_by(MarketRegimeLog.timestamp.desc()).all()

    return {
        "period_days": days,
        "entries": [
            {
                "regime": l.regime.value,
                "confidence": l.confidence,
                "timestamp": l.timestamp.isoformat(),
            }
            for l in logs
        ],
    }


@router.get("/options")
def get_regime_options():
    """Get all available regime options for frontend dropdown."""
    return {
        "regimes": [
            {"value": r.value, "label": r.value.replace("_", " ").title()}
            for r in MarketRegime
        ]
    }