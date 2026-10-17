# api/routes/coplanner.py
"""
Co-Planner Intelligence — API Endpoints.

Engine-driven trading intelligence: conviction scoring, regime matrix,
cross-stock comparison, opportunity ranking, behavioral guard.

ENGINE computes → LLM explains. No LLM scoring. No LLM ranking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from api.auth import get_current_user
from core.database import get_db_dependency as get_db

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/coplanner",
    tags=["coplanner"],
    dependencies=[Depends(get_current_user)],
)


# ── Pydantic schemas ────────────────────────────────────────────

class SetupInput(BaseModel):
    symbol: str
    strategy_name: str
    timeframe: str = "15m"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    current_volume: float = 0.0
    avg_volume: float = 0.0
    current_regime: str = "unknown"
    pattern_score: float = 0.0
    cluster_id: Optional[str] = None
    pattern_data: Dict[str, Any] = Field(default_factory=dict)


class OpenPositionInput(BaseModel):
    symbol: str
    strategy: str = ""
    sector: str = ""


class PlanSessionRequest(BaseModel):
    setups: List[SetupInput]
    open_positions: Optional[List[OpenPositionInput]] = None


class EvaluateSingleRequest(BaseModel):
    setup: SetupInput


class SuggestTargetsRequest(BaseModel):
    entry: float
    stop_loss: float


class RebuildMatrixRequest(BaseModel):
    force: bool = True


# ── Live market data enrichment ───────────────────────────────

def _enrich_setup_with_live_data(setup: SetupInput) -> SetupInput:
    """
    Fill in missing price/volume fields from live market data (Upstox)
    if the user didn't provide them. Never overwrites user-provided values.
    """
    try:
        from ingestion.market_data_service import get_market_data_service
        svc = get_market_data_service()
        if not svc:
            return setup
    except Exception:
        return setup

    symbol = setup.symbol.upper()

    # Fetch LTP if entry_price is missing
    if setup.entry_price == 0.0 and svc.has_live:
        try:
            ltp = svc.get_ltp(symbol)
            if ltp and ltp > 0:
                setup.entry_price = ltp
        except Exception:
            pass

    # Fetch full quote for volume if missing
    if (setup.current_volume == 0.0 or setup.avg_volume == 0.0) and svc.has_live:
        try:
            quote = svc.get_market_quote(symbol)
            if quote.get("volume") and setup.current_volume == 0.0:
                setup.current_volume = float(quote["volume"])
        except Exception:
            pass

    # Detect current regime if unknown and historical data is available
    if setup.current_regime == "unknown" and svc.has_historical:
        try:
            from datetime import date as _date, timedelta as _td
            from strategy_engine.regime_detector import MarketRegimeDetector
            candles = svc._fetch_candles_raw(
                symbol=symbol,
                interval="1d",
                from_date=_date.today() - _td(days=60),
                to_date=_date.today(),
            )
            if len(candles) >= 50:
                import pandas as pd
                df = pd.DataFrame(candles)
                if "timestamp" in df.columns:
                    df = df.rename(columns={"timestamp": "datetime"})
                detector = MarketRegimeDetector()
                analysis = detector.detect(df)
                if analysis and analysis.regime:
                    setup.current_regime = analysis.regime.value
        except Exception:
            pass

    return setup


# ── Engine factory ──────────────────────────────────────────────

def _build_engine(db: DBSession):
    """Create CoPlannerEngine wired to the current DB session."""
    from coplanner.db_interface import CoPlannerDB
    from coplanner.engine import CoPlannerEngine

    cp_db = CoPlannerDB(db)

    # pattern_learner is strategy-specific; wired per-strategy when needed
    pattern_learner = None

    # Try to get LLM client
    llm_client = None
    try:
        from agents.llm_client import LLMClient
        from config.settings import settings
        if settings.LLM_API_KEY:
            llm_client = LLMClient(
                provider=settings.LLM_PROVIDER,
                model=settings.LLM_MODEL,
                api_key=settings.LLM_API_KEY,
            )
    except Exception:
        pass

    return CoPlannerEngine(
        db=cp_db,
        pattern_learner=pattern_learner,
        llm_client=llm_client,
    )


def _to_setup_data(inp: SetupInput):
    """Convert Pydantic input → engine SetupData."""
    from coplanner.models import SetupData
    return SetupData(
        symbol=inp.symbol,
        strategy_name=inp.strategy_name,
        timeframe=inp.timeframe,
        entry_price=inp.entry_price,
        stop_loss=inp.stop_loss,
        target_price=inp.target_price,
        current_volume=inp.current_volume,
        avg_volume=inp.avg_volume,
        current_regime=inp.current_regime,
        pattern_score=inp.pattern_score,
        cluster_id=inp.cluster_id,
        pattern_data=inp.pattern_data,
    )


def _serialize_result(result) -> Dict[str, Any]:
    """Serialize PlanSessionResult to JSON-safe dict."""
    from dataclasses import asdict
    import json

    def _default(obj):
        if hasattr(obj, "value"):
            return obj.value
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    raw = asdict(result)
    # Round-trip through JSON to ensure all nested enums/dates are serializable
    return json.loads(json.dumps(raw, default=_default))


# ══════════════════════════════════════════════════════════════════
#  MAIN PLANNING SESSION
# ══════════════════════════════════════════════════════════════════

@router.post("/plan")
def plan_session(
    request: PlanSessionRequest,
    db: DBSession = Depends(get_db),
):
    """
    Main entry point — run full planning session.

    Executes the 7-step flow:
    1. Behavioral guard
    2. Cross-stock comparison (conviction + regime + RR)
    3. Multi-factor ranking
    4. Regime warnings
    5. Recommendation
    6. LLM explanation (optional)
    """
    if not request.setups:
        raise HTTPException(status_code=400, detail="At least one setup required")

    engine = _build_engine(db)
    # Enrich setups with live market data where user didn't provide values
    enriched_inputs = [_enrich_setup_with_live_data(s) for s in request.setups]
    setups = [_to_setup_data(s) for s in enriched_inputs]

    open_positions = None
    if request.open_positions:
        open_positions = [p.model_dump() for p in request.open_positions]
    else:
        # Auto-fetch open positions from connected broker
        try:
            from ingestion.market_data_service import get_market_data_service
            svc = get_market_data_service()
            client = None
            if svc and svc._angel:
                client = svc._angel
            elif svc and getattr(svc, "_zerodha", None):
                client = svc._zerodha
            else:
                from ingestion.trade_sync import get_trade_sync_service
                groww_svc = get_trade_sync_service()
                if groww_svc and groww_svc._groww:
                    client = groww_svc._groww
            if client:
                broker_positions = client.get_positions()
                if broker_positions:
                    open_positions = [
                        {"symbol": p["symbol"], "strategy": "", "sector": ""}
                        for p in broker_positions
                    ]
        except Exception:
            pass  # Not critical — plan works without positions too

    result = engine.plan_session(setups, open_positions=open_positions)
    return _serialize_result(result)


# ══════════════════════════════════════════════════════════════════
#  SINGLE SETUP EVALUATION
# ══════════════════════════════════════════════════════════════════

@router.post("/evaluate")
def evaluate_single(
    request: EvaluateSingleRequest,
    db: DBSession = Depends(get_db),
):
    """Quick evaluation of a single setup without full ranking."""
    engine = _build_engine(db)
    enriched = _enrich_setup_with_live_data(request.setup)
    setup = _to_setup_data(enriched)
    return engine.evaluate_single(setup)


# ══════════════════════════════════════════════════════════════════
#  REGIME MATRIX
# ══════════════════════════════════════════════════════════════════

@router.get("/regime-matrix")
def get_regime_matrix(db: DBSession = Depends(get_db)):
    """Full strategy × regime matrix display."""
    engine = _build_engine(db)
    return engine.get_regime_matrix()


@router.get("/regime-matrix/{strategy_name}")
def get_strategy_profile(
    strategy_name: str,
    db: DBSession = Depends(get_db),
):
    """Cross-regime profile for a single strategy."""
    engine = _build_engine(db)
    return engine.get_strategy_profile(strategy_name)


@router.post("/regime-matrix/rebuild")
def rebuild_matrix(db: DBSession = Depends(get_db)):
    """Force rebuild of strategy × regime matrix from trade data."""
    engine = _build_engine(db)
    return engine.rebuild_matrix()


# ══════════════════════════════════════════════════════════════════
#  RR TOOLS
# ══════════════════════════════════════════════════════════════════

@router.post("/rr/suggest-targets")
def suggest_targets(
    request: SuggestTargetsRequest,
    db: DBSession = Depends(get_db),
):
    """Multi-target exit suggestions based on entry and stop-loss."""
    if request.entry <= 0 or request.stop_loss <= 0:
        raise HTTPException(status_code=400, detail="Entry and stop_loss must be positive")
    engine = _build_engine(db)
    return engine.suggest_targets(request.entry, request.stop_loss)


@router.get("/rr/config")
def get_rr_config(db: DBSession = Depends(get_db)):
    """Current RR enforcement configuration."""
    engine = _build_engine(db)
    return engine.get_rr_config()


# ══════════════════════════════════════════════════════════════════
#  DATA QUALITY & EXECUTION PROFILE
# ══════════════════════════════════════════════════════════════════

@router.get("/data-quality")
def get_data_quality(db: DBSession = Depends(get_db)):
    """Data trust layer quality report — execution grade, clean ratio, etc."""
    engine = _build_engine(db)
    return engine.get_data_quality_report()


@router.get("/data-availability")
def get_data_availability(db: DBSession = Depends(get_db)):
    """Cold start / warming up / AVT ready status."""
    engine = _build_engine(db)
    return engine.get_data_availability()


@router.get("/execution-profile")
def get_execution_profile(db: DBSession = Depends(get_db)):
    """Full execution quality profile — strengths, weaknesses, growth areas."""
    engine = _build_engine(db)
    return engine.get_execution_profile()


@router.get("/evolution-timeline")
def get_evolution_timeline(
    window_size: int = 20,
    db: DBSession = Depends(get_db),
):
    """Quality evolution over time in rolling windows."""
    engine = _build_engine(db)
    return engine.get_evolution_timeline(window_size)


@router.get("/behavioral-patterns")
def get_behavioral_patterns(db: DBSession = Depends(get_db)):
    """Recurring behavioral patterns detected from trade quality analysis."""
    engine = _build_engine(db)
    return engine.detect_behavioral_patterns()


@router.post("/refresh-trust-weights")
def refresh_trust_weights(db: DBSession = Depends(get_db)):
    """Recompute adaptive trust weights based on current execution quality."""
    engine = _build_engine(db)
    engine.refresh_trust_weights()
    return {"status": "ok", "message": "Trust weights refreshed"}


@router.get("/learned-strategies")
def get_learned_strategies(db: DBSession = Depends(get_db)):
    """
    List all strategies the co-planner knows about from StrategyProfile.
    Includes regime preferences, confidence scores, and entry criteria.
    """
    engine = _build_engine(db)
    strategies = engine.get_learned_strategies()
    return {
        "count": len(strategies),
        "strategies": strategies,
    }


@router.get("/strategies-for-regime/{regime}")
def get_strategies_for_regime(regime: str, db: DBSession = Depends(get_db)):
    """
    Get user's strategies ranked by suitability for the given market regime.
    Combines trade-history performance with learned regime preferences.
    """
    engine = _build_engine(db)
    ranked = engine.get_strategies_for_regime(regime)
    return {
        "regime": regime,
        "strategies": ranked,
        "count": len(ranked),
    }
