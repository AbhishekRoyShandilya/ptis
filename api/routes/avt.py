# api/routes/avt.py
"""
Autonomous Virtual Trader — API Endpoints
===========================================

Control AVT, view trades, manage proven improvements,
track competition, and chat with AVT.

NO unproven hypotheses are exposed. NO pre-trade-check
(that was the old improvement_lab pattern). The user only
sees what AVT has proven with its own virtual trading.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from typing import Any, Dict, List

from api.auth import get_current_user
from api.schemas.avt_schemas import (
    AVTStartRequest,
    AVTStartResponse,
    AVTStatusResponse,
    AVTStopResponse,
    AdoptImprovementRequest,
    AdoptImprovementResponse,
    CompetitionHistoryResponse,
    CompetitionReportResponse,
    ImprovementListResponse,
    ImprovementPresentationResponse,
    ImprovementResponse,
    ScanLogListResponse,
    VersionHistoryResponse,
    VirtualTradeListResponse,
    VirtualTradeResponse,
)
from core.database import get_db_dependency as get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/avt", tags=["avt"], dependencies=[Depends(get_current_user)])


# ── Engine singleton access ─────────────────────────────────────

def _get_avt(user=None):
    """
    The relevant AVT engine. With a user → that user's engine; without → the sole
    engine (back-compat for secondary introspection routes not yet per-user
    scoped; correct for single-user). 503 if none.
    """
    from avt.engine import get_avt
    avt = get_avt(str(user.id)) if (user is not None and hasattr(user, "id")) else get_avt()
    if avt is None:
        raise HTTPException(
            status_code=503,
            detail="AVT not started for this user — POST /api/v1/avt/start first",
        )
    return avt


def _get_avt_dep(user=Depends(get_current_user)):
    """FastAPI dependency: the CURRENT USER's AVT engine (per-user scoped)."""
    return _get_avt(user)


# ══════════════════════════════════════════════════════════════════
#  AVT CONTROL (per-user)
# ══════════════════════════════════════════════════════════════════

@router.post("/start", response_model=AVTStartResponse)
async def start_avt(
    request: AVTStartRequest,
    user=Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """Start THIS user's autonomous virtual trader (creating their engine on first start)."""
    from avt.engine import get_avt
    from avt.integrations import setup_avt_for_user

    avt = get_avt(str(user.id)) or setup_avt_for_user(str(user.id))
    if avt is None:
        raise HTTPException(
            status_code=503,
            detail="AVT unavailable — market-data deps not configured",
        )

    if avt.status == "running":
        return AVTStartResponse(
            status="running",
            message="AVT is already running",
            strategies_loaded=len(avt.strategy_versions),
        )

    try:
        import asyncio
        avt.status = "running"
        avt._started_at = datetime.now(timezone.utc)
        asyncio.create_task(avt.run())
        return AVTStartResponse(
            status=avt.status,
            message="AVT started successfully",
            strategies_loaded=len(avt.strategy_versions),
        )
    except Exception as e:
        logger.error("Failed to start AVT: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to start AVT: {e}")


@router.post("/stop", response_model=AVTStopResponse)
async def stop_avt(avt=Depends(_get_avt_dep)):
    """Stop THIS user's autonomous virtual trader."""
    open_count = len(avt.portfolio.open_positions)
    avt.status = "stopped"

    return AVTStopResponse(
        status=avt.status,
        message="AVT stopped",
        open_positions_at_stop=open_count,
    )


class ReplayStartRequest(BaseModel):
    from_date: Optional[str] = None    # ISO; default 2022-01-01
    to_date: Optional[str] = None      # ISO; default today
    universe: str = "nifty100"         # nifty100 | fno (FIXED — not the live watchlist)
    max_symbols: int = 150
    strategy: Optional[str] = None     # restrict to one sko_id / short_code
    mode: str = "benchmark"            # benchmark (frozen) | learning (not yet available)


@router.post("/replay/start")
def start_avt_replay(req: ReplayStartRequest, user=Depends(get_current_user)):
    """
    Build THIS user's strategy baseline by replaying their taught strategies over
    lake history (BENCHMARK mode: frozen v1.0, no learning). Populates the clean
    edge matrix + per-(strategy,stock) ranking baseline so a freshly-taught
    strategy isn't ranked blind. Runs in the background (minutes); poll
    GET /avt/replay/status. Idempotent — re-calling while running returns the
    live status instead of starting a second run.
    """
    if req.mode != "benchmark":
        raise HTTPException(
            status_code=400,
            detail="Only benchmark replay is available. Learning (adaptive) replay "
                   "is a separate research tool and is not built yet.",
        )
    from datetime import date as _date
    from avt.replay import start_replay_job
    from core.database import SessionLocal

    start = datetime.fromisoformat(req.from_date) if req.from_date else datetime(2022, 1, 1)
    end = (datetime.fromisoformat(req.to_date) if req.to_date
           else datetime.combine(_date.today(), datetime.min.time()))
    status = start_replay_job(
        str(user.id), SessionLocal, start=start, end=end,
        only_strategy=req.strategy, universe=req.universe,
        max_symbols=req.max_symbols, mode=req.mode,
    )
    return status


@router.get("/replay/status")
def get_avt_replay_status(user=Depends(get_current_user)):
    """This user's benchmark-replay job status (state / message / progress / results)."""
    from avt.replay import replay_job_status
    st = replay_job_status(str(user.id))
    return st or {"state": "idle", "message": "no replay has been run yet"}


@router.get("/status", response_model=AVTStatusResponse)
def get_avt_status(avt=Depends(_get_avt_dep), db: DBSession = Depends(get_db)):
    """Current AVT state, open positions, portfolio snapshot (this user)."""
    uptime = 0.0
    if hasattr(avt, "_started_at") and avt._started_at:
        uptime = (datetime.now(timezone.utc) - avt._started_at).total_seconds()

    recent_errors: list = []
    for lg in getattr(avt, "scan_logs", [])[-3:]:
        recent_errors.extend(getattr(lg, "errors", [])[:3])

    return AVTStatusResponse(
        status=avt.status,
        uptime_seconds=uptime,
        current_cycle=getattr(avt, "_cycle_count", 0),
        open_positions=len(avt.portfolio.open_positions),
        total_trades=avt.portfolio.trade_count,
        portfolio_equity=avt.portfolio.equity,
        portfolio_drawdown_pct=avt.portfolio.current_drawdown_pct * 100,
        active_strategies=list(avt.strategy_versions.keys()),
        last_scan_at=getattr(avt, "_last_scan_at", None),
        errors=recent_errors[:5],
        capital_sleeves=(avt._allocator.summary() if getattr(avt, "_allocator", None) else []),
        partitioned=(avt._allocator.is_partitioned if getattr(avt, "_allocator", None) else False),
    )


# ══════════════════════════════════════════════════════════════════
#  CAPITAL SLEEVES (B.1) — the user's account split (their money decision)
# ══════════════════════════════════════════════════════════════════

class SleeveSpec(BaseModel):
    name: str
    capital: float
    instruments: List[str]
    max_positions: int = 0        # 0 → inherit the engine's default max_open_positions
    size_mode: str = "risk_pct"   # how trades in this sleeve are sized: risk_pct | lots | amount
    size_value: float = 0.0       # lots count (lots) OR rupees-per-trade (amount); ignored for risk_pct


class SleeveConfigRequest(BaseModel):
    virtual_capital: Optional[float] = None   # total account (defaults to the engine's default if unset)
    sleeves: List[SleeveSpec] = []


@router.get("/sleeves")
def get_avt_sleeves(user=Depends(get_current_user)):
    """This user's configured capital split (futures / options / equity pools). Empty → shared pool."""
    from avt.engine import get_avt
    from avt.sleeve_store import get_config
    cfg = get_config(str(user.id))
    avt = get_avt(str(user.id))
    return {
        "virtual_capital": cfg.get("virtual_capital"),
        "sleeves": cfg.get("sleeves", []),
        "running": bool(avt is not None and getattr(avt, "status", "") == "running"),
        # live per-sleeve state if the engine is up (so the editor can show current usage)
        "live": (avt._allocator.summary() if (avt is not None and getattr(avt, "_allocator", None)) else []),
        "valid_instruments": ["future", "equity", "option"],
    }


@router.put("/sleeves")
def set_avt_sleeves(req: SleeveConfigRequest, user=Depends(get_current_user)):
    """Persist this user's account split. Applies on the next AVT (re)start; if the engine is idle it is dropped
    so it rebuilds with the new pools on next start. Refuses to change the split while AVT is running (stop first —
    reallocating capital under open positions is unsafe)."""
    from avt.engine import get_avt, remove_avt
    from avt.sleeve_store import set_config

    avt = get_avt(str(user.id))
    if avt is not None and getattr(avt, "status", "") == "running":
        raise HTTPException(status_code=409, detail="Stop AVT before changing capital sleeves.")

    try:
        stored = set_config(str(user.id), req.virtual_capital,
                            [s.model_dump() for s in req.sleeves])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Drop the idle engine so the next /start rebuilds it with the new sleeves.
    if avt is not None:
        remove_avt(str(user.id))

    return {"status": "saved", "message": "Capital sleeves saved — (re)start AVT to apply.",
            "virtual_capital": stored["virtual_capital"], "sleeves": stored["sleeves"]}


@router.delete("/sleeves")
def clear_avt_sleeves(user=Depends(get_current_user)):
    """Revert to the shared pool (no sleeves). Applies on next (re)start."""
    from avt.engine import get_avt, remove_avt
    from avt.sleeve_store import clear_config
    avt = get_avt(str(user.id))
    if avt is not None and getattr(avt, "status", "") == "running":
        raise HTTPException(status_code=409, detail="Stop AVT before changing capital sleeves.")
    clear_config(str(user.id))
    if avt is not None:
        remove_avt(str(user.id))
    return {"status": "cleared", "message": "Reverted to shared pool — (re)start AVT to apply."}


@router.get("/health")
async def get_avt_health(user=Depends(get_current_user)):
    """
    This user's AVT health snapshot: is AVT configured and actually working?

    Returns health = healthy | quiet | not_ready, with a reason and the latest
    scan funnel (scanned → candidates → ranked → filtered → executed) so the UI
    can distinguish a quiet market from a misconfigured engine. Reports
    'not_ready' (not a 503) when the user hasn't started AVT yet.
    """
    from avt.engine import get_avt
    avt = get_avt(str(user.id))
    if avt is None:
        return {
            "health": "not_ready",
            "reason": "AVT not started — POST /api/v1/avt/start to begin",
            "strategies_loaded": 0,
            "strategies_with_scoring_config": 0,
            "watchlist_size": 0,
            "signal_gen_attached": False,
            "market_open": False,
            "last_scan_at": None,
            "last_cycle": {},
            "recent_errors": [],
        }
    return await avt.get_health()


# ══════════════════════════════════════════════════════════════════
#  TRADES
# ══════════════════════════════════════════════════════════════════

@router.get("/trades", response_model=VirtualTradeListResponse)
def list_trades(
    status: Optional[str] = Query(None, pattern="^(open|closed)$"),
    strategy_id: Optional[UUID] = None,
    limit: int = Query(100, ge=1, le=500),
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """List THIS user's virtual trades with optional filters."""
    from avt.repository import AVTTradeRepository
    repo = AVTTradeRepository(db)
    tid = str(user.id)

    if status == "open":
        trades = repo.get_open_trades(strategy_id=strategy_id, trader_id=tid)
    elif status == "closed":
        trades = repo.get_closed_trades(strategy_id=strategy_id, limit=limit, trader_id=tid)
    else:
        open_trades = repo.get_open_trades(strategy_id=strategy_id, trader_id=tid)
        closed_trades = repo.get_closed_trades(strategy_id=strategy_id, limit=limit, trader_id=tid)
        trades = open_trades + closed_trades

    return VirtualTradeListResponse(
        trades=[_trade_to_response(t) for t in trades],
        total=len(trades),
    )


@router.get("/trades/open", response_model=VirtualTradeListResponse)
def list_open_trades(
    strategy_id: Optional[UUID] = None,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """This user's current open positions."""
    from avt.repository import AVTTradeRepository
    repo = AVTTradeRepository(db)
    trades = repo.get_open_trades(strategy_id=strategy_id, trader_id=str(user.id))

    return VirtualTradeListResponse(
        trades=[_trade_to_response(t) for t in trades],
        total=len(trades),
    )


@router.get("/trades/{trade_id}", response_model=VirtualTradeResponse)
def get_trade(trade_id: UUID, db: DBSession = Depends(get_db)):
    """Single trade detail."""
    from avt.repository import AVTTradeRepository
    repo = AVTTradeRepository(db)
    trade = repo.get_by_id(trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return _trade_to_response(trade)


# ══════════════════════════════════════════════════════════════════
#  IMPROVEMENTS (only proven, with evidence)
# ══════════════════════════════════════════════════════════════════

@router.get("/improvements", response_model=ImprovementListResponse)
def list_improvements(
    strategy_id: Optional[UUID] = None,
    status: Optional[str] = Query(None, pattern="^(proving|proven|reverted|adopted)$"),
    db: DBSession = Depends(get_db),
):
    """
    List improvements. By default returns proven only.
    Users can also see 'proving' (in-progress) for transparency.
    """
    from avt.repository import AVTImprovementRepository
    repo = AVTImprovementRepository(db)

    if strategy_id:
        improvements = repo.get_by_strategy(strategy_id, status=status)
    elif status:
        improvements = repo.get_proven() if status == "proven" else []
    else:
        improvements = repo.get_proven()

    return ImprovementListResponse(
        improvements=[_improvement_to_response(i) for i in improvements],
        total=len(improvements),
    )


@router.get("/improvements/{improvement_id}", response_model=ImprovementResponse)
def get_improvement(improvement_id: UUID, db: DBSession = Depends(get_db)):
    """Single improvement detail with proof evidence."""
    from avt.repository import AVTImprovementRepository
    repo = AVTImprovementRepository(db)
    imp = repo.get_by_id(improvement_id)
    if not imp:
        raise HTTPException(status_code=404, detail="Improvement not found")
    return _improvement_to_response(imp)


@router.post(
    "/improvements/{improvement_id}/adopt",
    response_model=AdoptImprovementResponse,
)
def adopt_improvement(
    improvement_id: UUID,
    request: AdoptImprovementRequest,
    db: DBSession = Depends(get_db),
):
    """User adopts a proven improvement into their real trading rules."""
    from avt.repository import AVTImprovementRepository
    repo = AVTImprovementRepository(db)

    imp = repo.get_by_id(improvement_id)
    if not imp:
        raise HTTPException(status_code=404, detail="Improvement not found")
    if imp.status != "proven":
        raise HTTPException(
            status_code=400,
            detail=f"Can only adopt proven improvements. Current status: {imp.status}",
        )

    imp.status = "adopted"
    if hasattr(imp, "adoption_notes"):
        imp.adoption_notes = request.notes

    logger.info(
        "User adopted improvement %s: %s (notes: %s)",
        improvement_id, imp.description, request.notes,
    )

    db.commit()

    # Notify downstream systems (co-planner, coaching, etc.)
    try:
        from core.events import Event, EventType, get_event_bus
        get_event_bus().publish(Event(
            event_type=EventType.AVT_IMPROVEMENT_PROVEN,
            source="avt.adoption",
            strategy_id=str(getattr(imp, "strategy_id", "")),
            data={
                "improvement_id": str(improvement_id),
                "strategy_id": str(getattr(imp, "strategy_id", "")),
                "description": imp.description,
                "action": "adopted_by_user",
            },
        ))
    except Exception as e:
        logger.warning("Failed to publish adoption event: %s", e)

    return AdoptImprovementResponse(
        improvement_id=improvement_id,
        status="adopted",
        message=f"Improvement '{imp.description}' adopted into your trading rules.",
    )


@router.get(
    "/improvements/{improvement_id}/presentation",
    response_model=ImprovementPresentationResponse,
)
async def get_improvement_presentation(
    improvement_id: UUID,
    db: DBSession = Depends(get_db),
):
    """Get the storytelling presentation for a proven improvement."""
    from avt.repository import AVTImprovementRepository
    repo = AVTImprovementRepository(db)

    imp = repo.get_by_id(improvement_id)
    if not imp:
        raise HTTPException(status_code=404, detail="Improvement not found")

    from avt.presenter import AVTPresenter
    presenter = AVTPresenter()

    try:
        avt = _get_avt()
        presentation = await presenter.generate_improvement_presentation(imp, avt)
        return ImprovementPresentationResponse(
            title=presentation.title,
            narrative=presentation.narrative,
            summary_stats=presentation.summary_stats,
            examples=presentation.examples,
            user_action_options=presentation.user_action_options,
            avt_confidence=presentation.avt_confidence,
        )
    except Exception as e:
        logger.error("Presentation generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate presentation")


# ══════════════════════════════════════════════════════════════════
#  COMPETITION
# ══════════════════════════════════════════════════════════════════

@router.get(
    "/competition/{strategy_id}",
    response_model=CompetitionReportResponse,
)
def get_competition_report(
    strategy_id: UUID,
    period_days: int = Query(30, ge=7, le=365),
    db: DBSession = Depends(get_db),
):
    """Current AVT vs user comparison for a strategy."""
    from avt.competition import CompetitionEngine
    from avt.repository import AVTCompetitionRepository

    comp_repo = AVTCompetitionRepository(db)
    engine = CompetitionEngine(db=db)

    try:
        report = engine.generate_comparison(
            strategy_id=str(strategy_id),
            period_days=period_days,
        )
    except Exception as e:
        logger.error("Competition report failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate competition report")

    try:
        comp_repo.save_snapshot(report, strategy_id)
        db.commit()
    except Exception as e:
        logger.warning("Failed to save competition snapshot: %s", e)
        db.rollback()

    return CompetitionReportResponse(
        strategy_name=report.strategy_name,
        period_days=report.period_days,
        user_stats=report.user_stats,
        avt_stats=report.avt_stats,
        performance_gap=report.performance_gap,
        differences=[d.model_dump() for d in report.differences],
        proven_improvements=[i.model_dump(mode="json") for i in report.proven_improvements],
        overall_narrative=report.overall_narrative,
    )


@router.get(
    "/competition/{strategy_id}/history",
    response_model=CompetitionHistoryResponse,
)
def get_competition_history(
    strategy_id: UUID,
    limit: int = Query(30, ge=1, le=90),
    db: DBSession = Depends(get_db),
):
    """Historical competition snapshots for trend tracking."""
    from avt.repository import AVTCompetitionRepository
    repo = AVTCompetitionRepository(db)
    snapshots = repo.get_history(strategy_id, limit=limit)

    return CompetitionHistoryResponse(
        strategy_id=strategy_id,
        snapshots=snapshots,
    )


# ══════════════════════════════════════════════════════════════════
#  VERSIONS
# ══════════════════════════════════════════════════════════════════

@router.get(
    "/versions/{strategy_id}",
    response_model=VersionHistoryResponse,
)
def get_version_history(
    strategy_id: UUID,
    db: DBSession = Depends(get_db),
):
    """Version history with performance for a strategy."""
    from avt.repository import AVTVersionRepository
    repo = AVTVersionRepository(db)
    versions = repo.get_history(strategy_id)

    return VersionHistoryResponse(
        strategy_id=strategy_id,
        versions=versions,
    )


# ══════════════════════════════════════════════════════════════════
#  SCAN LOGS
# ══════════════════════════════════════════════════════════════════

@router.get("/scan-logs", response_model=ScanLogListResponse)
def list_scan_logs(
    limit: int = Query(50, ge=1, le=200),
    db: DBSession = Depends(get_db),
):
    """Recent scan cycle logs for transparency."""
    from avt.repository import AVTScanLogRepository
    repo = AVTScanLogRepository(db)
    logs = repo.get_recent(limit=limit)

    return ScanLogListResponse(
        logs=logs,
        total=len(logs),
    )


# ══════════════════════════════════════════════════════════════════
#  CHAT (NL conversation with AVT)
# ══════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    intent: str
    pending_messages: list = []


@router.post("/chat", response_model=ChatResponse)
async def chat_with_avt(request: ChatRequest, user=Depends(get_current_user)):
    """Natural language conversation with AVT."""
    try:
        avt = _get_avt()
        conversation = _get_conversation(avt, user_id=str(user.id))
        result = await conversation.chat(request.message)
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("AVT chat error: %s", e)
        raise HTTPException(status_code=500, detail="Chat failed")


@router.get("/chat/history")
async def get_chat_history(limit: int = Query(20, ge=1, le=100), user=Depends(get_current_user)):
    """Get conversation history."""
    try:
        avt = _get_avt()
        conversation = _get_conversation(avt, user_id=str(user.id))
        return {"history": conversation.get_history(limit)}
    except HTTPException:
        raise
    except Exception:
        return {"history": []}


@router.get("/chat/proactive")
async def get_proactive_message(user=Depends(get_current_user)):
    """Check if AVT has something to tell the user."""
    try:
        avt = _get_avt()
        conversation = _get_conversation(avt, user_id=str(user.id))
        msg = conversation.get_proactive_message()
        return {"has_message": msg is not None, "message": msg}
    except Exception:
        return {"has_message": False, "message": None}


# ══════════════════════════════════════════════════════════════════
#  OBSERVATIONS (behavioral only)
# ══════════════════════════════════════════════════════════════════

@router.get("/observations")
async def get_observations(limit: int = Query(50, ge=1, le=200)):
    """Recent behavioral observations from AVT observer."""
    try:
        avt = _get_avt()
        observer = getattr(avt, "_observer", None)
        if not observer:
            return {"observations": []}
        return {"observations": observer.get_recent_observations(limit=limit)}
    except HTTPException:
        raise
    except Exception:
        return {"observations": []}


@router.get("/observations/today")
async def get_today_observations():
    """Today's behavioral observation summary."""
    try:
        avt = _get_avt()
        observer = getattr(avt, "_observer", None)
        if observer:
            return observer.get_today_summary()
        return {"date": "", "total_observations": 0, "observations": []}
    except HTTPException:
        raise
    except Exception:
        return {"date": "", "total_observations": 0, "observations": []}


@router.get("/messages")
async def get_triage_messages(limit: int = Query(20, ge=1, le=100)):
    """Unacknowledged messages for the user."""
    try:
        avt = _get_avt()
        triage = getattr(avt, "_triage", None)
        if triage:
            return {
                "messages": triage.get_pending_messages(limit),
                "unread_count": triage.get_unread_count(),
                "critical_count": triage.get_critical_count(),
            }
        return {"messages": [], "unread_count": 0, "critical_count": 0}
    except HTTPException:
        raise
    except Exception:
        return {"messages": [], "unread_count": 0, "critical_count": 0}


@router.post("/messages/{message_id}/ack")
async def acknowledge_message(message_id: str):
    """Acknowledge a triage message."""
    try:
        avt = _get_avt()
        triage = getattr(avt, "_triage", None)
        if triage:
            ok = triage.acknowledge_message(message_id)
            return {"acknowledged": ok}
        return {"acknowledged": False}
    except HTTPException:
        raise
    except Exception:
        return {"acknowledged": False}


# ══════════════════════════════════════════════════════════════════
#  STRATEGY LIFECYCLE
# ══════════════════════════════════════════════════════════════════

class ValidateRequest(BaseModel):
    trade_history: Optional[list] = None


@router.post("/strategies/{strategy_id}/validate")
def validate_strategy(
    strategy_id: UUID,
    request: ValidateRequest = None,
    db: DBSession = Depends(get_db),
):
    """Run health gate on a strategy. Returns grade + issues."""
    avt = _get_avt()
    lm = getattr(avt, "_lifecycle_manager", None)
    if not lm:
        raise HTTPException(status_code=503, detail="Lifecycle manager not available")

    strategy_dict = avt.strategies.get(str(strategy_id))
    if not strategy_dict:
        raise HTTPException(status_code=404, detail="Strategy not found")

    from avt.strategy_lifecycle import StrategyHealthGate
    gate = StrategyHealthGate()
    trade_history = request.trade_history if request else None
    result = gate.validate(strategy_dict, trade_history)

    return {
        "strategy_id": str(strategy_id),
        "grade": result.grade,
        "overall_health_score": result.overall_health_score,
        "structural_score": result.structural_score,
        "coherence_score": result.coherence_score,
        "issues": [
            {
                "category": i.category,
                "severity": i.severity,
                "field": i.field,
                "message": i.message,
                "suggestion": i.suggestion,
            }
            for i in result.issues
        ],
    }


@router.get("/strategies/{strategy_id}/lifecycle")
def get_lifecycle_status(strategy_id: UUID):
    """Full lifecycle state for a strategy."""
    avt = _get_avt()
    lm = getattr(avt, "_lifecycle_manager", None)
    if not lm:
        raise HTTPException(status_code=503, detail="Lifecycle manager not available")

    info = lm.get_lifecycle_info(str(strategy_id))
    return {
        "strategy_id": str(strategy_id),
        "state": info.state,
        "grade": info.grade,
        "health_score": info.health_score,
        "allocation_multiplier": info.allocation_multiplier,
        "health_issues": info.health_issues,
        "calibration_alignment": info.calibration_alignment,
        "calibration_signals": getattr(info, "calibration_signals", 0),
        "calibration_comparisons": getattr(info, "calibration_comparisons", 0),
        "degradation_level": info.degradation_level,
        "rolling_expectancy": info.rolling_expectancy,
        "translation_drift": info.translation_drift,
        "baseline_expectancy": getattr(info, "baseline_expectancy", None),
    }


@router.get("/strategies/lifecycle/all")
def get_all_lifecycle():
    """Summary of all strategy lifecycle states."""
    avt = _get_avt()
    lm = getattr(avt, "_lifecycle_manager", None)
    if not lm:
        raise HTTPException(status_code=503, detail="Lifecycle manager not available")

    all_info = lm.get_all_lifecycle_info()
    return {
        "strategies": {
            sid: {
                "state": info.state,
                "grade": info.grade,
                "health_score": info.health_score,
                "allocation_multiplier": info.allocation_multiplier,
                "degradation_level": info.degradation_level,
            }
            for sid, info in all_info.items()
        },
        "total": len(all_info),
    }


class CalibrationAnnotateRequest(BaseModel):
    mismatch_id: str
    reason: str


@router.post("/strategies/{strategy_id}/calibration/start")
def start_calibration(strategy_id: UUID):
    """Enter calibration mode for a strategy."""
    avt = _get_avt()
    lm = getattr(avt, "_lifecycle_manager", None)
    if not lm:
        raise HTTPException(status_code=503, detail="Lifecycle manager not available")

    sid = str(strategy_id)

    # Create calibration tracker if not exists
    cal = lm.get_calibration_tracker(sid)
    if not cal:
        from avt.calibration_tracker import SignalCalibrationTracker
        cal = SignalCalibrationTracker(strategy_id=sid)
        lm.set_calibration_tracker(sid, cal)

    ok = lm.start_calibration(sid)
    if not ok:
        state = lm.get_state(sid).value
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start calibration from state '{state}'",
        )

    return {"strategy_id": sid, "state": "calibrating", "message": "Calibration started"}


@router.post("/strategies/{strategy_id}/calibration/skip")
def skip_calibration(strategy_id: UUID):
    """Skip calibration, go straight to autonomous."""
    avt = _get_avt()
    lm = getattr(avt, "_lifecycle_manager", None)
    if not lm:
        raise HTTPException(status_code=503, detail="Lifecycle manager not available")

    ok = lm.skip_calibration(str(strategy_id))
    if not ok:
        raise HTTPException(status_code=400, detail="Strategy is not in calibration")

    return {"strategy_id": str(strategy_id), "state": "autonomous", "message": "Calibration skipped"}


@router.get("/strategies/{strategy_id}/calibration/report")
def get_calibration_report(strategy_id: UUID):
    """Alignment score, mismatches, suggestions."""
    avt = _get_avt()
    lm = getattr(avt, "_lifecycle_manager", None)
    if not lm:
        raise HTTPException(status_code=503, detail="Lifecycle manager not available")

    cal = lm.get_calibration_tracker(str(strategy_id))
    if not cal:
        raise HTTPException(status_code=404, detail="No calibration tracker for this strategy")

    report = cal.compute_alignment()
    can_grad, grad_msg = cal.should_graduate()
    mismatches = cal.get_mismatches(limit=20)

    return {
        "strategy_id": str(strategy_id),
        "alignment_score": report.alignment_score,
        "true_positives": report.true_positives,
        "false_positives": report.false_positives,
        "false_negatives": report.false_negatives,
        "false_positive_rate": report.false_positive_rate,
        "false_negative_rate": report.false_negative_rate,
        "total_comparisons": report.total_comparisons,
        "signals_generated": report.signals_generated,
        "user_trades": report.user_trades,
        "ready_to_graduate": report.ready_to_graduate,
        "graduation_blockers": report.graduation_blockers,
        "graduation_message": grad_msg,
        "top_mismatch_reasons": report.top_mismatch_reasons,
        "mismatches": [
            {
                "id": m.id,
                "type": m.type,
                "symbol": m.symbol,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                "user_annotation": m.user_annotation,
            }
            for m in mismatches
        ],
    }


@router.post("/strategies/{strategy_id}/calibration/annotate")
def annotate_calibration_mismatch(
    strategy_id: UUID,
    request: CalibrationAnnotateRequest,
):
    """User explains why a mismatch occurred."""
    avt = _get_avt()
    lm = getattr(avt, "_lifecycle_manager", None)
    if not lm:
        raise HTTPException(status_code=503, detail="Lifecycle manager not available")

    cal = lm.get_calibration_tracker(str(strategy_id))
    if not cal:
        raise HTTPException(status_code=404, detail="No calibration tracker")

    ok = cal.annotate_mismatch(request.mismatch_id, request.reason)
    if not ok:
        raise HTTPException(status_code=404, detail="Mismatch not found")

    return {"annotated": True, "mismatch_id": request.mismatch_id}


class UserTradeRequest(BaseModel):
    symbol: str
    direction: str  # "long" | "short"
    timestamp: Optional[datetime] = None
    trade_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@router.post("/strategies/{strategy_id}/calibration/user-trade")
def record_calibration_user_trade(
    strategy_id: UUID,
    request: UserTradeRequest,
    db: DBSession = Depends(get_db),
):
    """
    Record a user's manual trade for calibration comparison.

    Call this whenever you take a real/paper trade during calibration mode.
    The system will match it against AVT signals within ±15 minutes on the
    same symbol + direction to compute alignment (true positive vs false negative).

    Mismatches (AVT signalled but you didn't trade, or you traded but AVT didn't)
    are recorded and shown in the calibration report with annotation support.
    """
    avt = _get_avt()
    lm = getattr(avt, "_lifecycle_manager", None)
    if not lm:
        raise HTTPException(status_code=503, detail="Lifecycle manager not available")

    sid = str(strategy_id)
    cal = lm.get_calibration_tracker(sid)
    if not cal:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No calibration tracker for strategy {sid}. "
                "Call POST /calibration/start first."
            ),
        )

    trade_id = cal.record_user_trade(
        symbol=request.symbol,
        direction=request.direction,
        timestamp=request.timestamp,
        trade_id=request.trade_id,
        details=request.details or {},
    )

    # Compute updated alignment
    report = cal.compute_alignment()

    return {
        "recorded": True,
        "trade_id": trade_id,
        "strategy_id": sid,
        "symbol": request.symbol,
        "direction": request.direction,
        "alignment_score": report.alignment_score,
        "total_comparisons": report.total_comparisons,
        "true_positives": report.true_positives,
        "false_negatives": report.false_negatives,
    }


@router.post("/strategies/{strategy_id}/calibration/graduate")
def graduate_calibration(strategy_id: UUID):
    """Graduate from calibration to autonomous trading."""
    avt = _get_avt()
    lm = getattr(avt, "_lifecycle_manager", None)
    if not lm:
        raise HTTPException(status_code=503, detail="Lifecycle manager not available")

    sid = str(strategy_id)
    cal = lm.get_calibration_tracker(sid)
    if cal:
        can_grad, msg = cal.should_graduate()
        if not can_grad:
            raise HTTPException(status_code=400, detail=f"Cannot graduate: {msg}")

    ok = lm.graduate_calibration(sid)
    if not ok:
        raise HTTPException(status_code=400, detail="Strategy is not in calibration")

    return {"strategy_id": sid, "state": "autonomous", "message": "Graduated to autonomous"}


@router.get("/strategies/{strategy_id}/edge-monitor")
def get_edge_monitor(strategy_id: UUID):
    """Current edge monitor status: degradation, rolling metrics, drift."""
    avt = _get_avt()
    lm = getattr(avt, "_lifecycle_manager", None)
    if not lm:
        raise HTTPException(status_code=503, detail="Lifecycle manager not available")

    mon = lm.get_edge_monitor(str(strategy_id))
    if not mon:
        return {
            "strategy_id": str(strategy_id),
            "status": "no_monitor",
            "message": "Edge monitor not initialized for this strategy",
        }

    status = mon.get_status()
    return {
        "strategy_id": str(strategy_id),
        **status,
    }


@router.post("/strategies/{strategy_id}/resume")
def resume_strategy(strategy_id: UUID):
    """Resume a paused or degraded strategy."""
    avt = _get_avt()
    lm = getattr(avt, "_lifecycle_manager", None)
    if not lm:
        raise HTTPException(status_code=503, detail="Lifecycle manager not available")

    ok = lm.resume_strategy(str(strategy_id))
    if not ok:
        state = lm.get_state(str(strategy_id)).value
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume from state '{state}'",
        )

    return {"strategy_id": str(strategy_id), "state": "autonomous", "message": "Strategy resumed"}


# ══════════════════════════════════════════════════════════════════
#  SELF-IMPROVER STATUS (what AVT is proving)
# ══════════════════════════════════════════════════════════════════

@router.get("/proving")
async def get_proving_status():
    """What improvements AVT is currently proving."""
    try:
        avt = _get_avt()
        improver = getattr(avt, "_self_improver", None)
        if improver:
            return improver.get_all_status()
        return {}
    except HTTPException:
        raise
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════
#  CONVERSATION — PER-USER INSTANCES
# ══════════════════════════════════════════════════════════════════

_conversation_instances: dict[str, "AVTConversation"] = {}


def _get_conversation(avt, user_id: str = "default"):
    """Get or create a conversation instance for this user."""
    if user_id not in _conversation_instances:
        from avt.conversation import AVTConversation
        _conversation_instances[user_id] = AVTConversation(
            avt_engine=avt,
            self_improver=getattr(avt, "_self_improver", None),
            competition=getattr(avt, "_competition", None),
            observer=getattr(avt, "_observer", None),
            triage=getattr(avt, "_triage", None),
        )
    return _conversation_instances[user_id]


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _trade_to_response(t) -> VirtualTradeResponse:
    return VirtualTradeResponse(
        id=t.id,
        strategy_id=t.strategy_id,
        strategy_version=t.strategy_version,
        variant_id=t.variant_id,
        symbol=t.symbol,
        direction=t.direction,
        entry_time=t.entry_time,
        entry_price=t.entry_price,
        quantity=t.quantity,
        stop_price=t.stop_price,
        conviction_at_entry=t.conviction_at_entry,
        regime_at_entry=t.regime_at_entry,
        ranking_position=t.ranking_position,
        ranking_score=t.ranking_score,
        max_favorable_r=t.max_favorable_r,
        exit_time=t.exit_time,
        exit_price=t.exit_price,
        exit_type=t.exit_type,
        # domain VirtualTrade uses realized_pnl / r_multiple; the response uses pnl_absolute / pnl_r (getattr keeps
        # it robust whether the object is a domain trade or a DB-hydrated one).
        exit_reason=getattr(t, "exit_reason", None) or t.exit_type,
        pnl_r=(getattr(t, "pnl_r", None) if getattr(t, "pnl_r", None) is not None else getattr(t, "r_multiple", None)),
        pnl_pct=getattr(t, "pnl_pct", None),
        pnl_absolute=(getattr(t, "pnl_absolute", None) if getattr(t, "pnl_absolute", None) is not None else getattr(t, "realized_pnl", None)),
        active_improvements=t.active_improvements,
        improvement_exits=getattr(t, "improvement_exits", None) or [],
        status=t.status,
        instrument=getattr(t, "instrument", "future"),
        tradeable=getattr(t, "tradeable", None),
        option_strike=getattr(t, "option_strike", None),
        option_expiry=getattr(t, "option_expiry", None),
        option_right=getattr(t, "option_right", None),
        option_lots=getattr(t, "option_lots", None),
        option_lot_size=getattr(t, "option_lot_size", None),
        underlying_stop=getattr(t, "underlying_stop", None),
    )


def _improvement_to_response(i) -> ImprovementResponse:
    return ImprovementResponse(
        id=i.id,
        strategy_id=i.strategy_id,
        type=i.type,
        description=i.description,
        params=i.params,
        status=i.status,
        applied_at=i.applied_at,
        applied_at_version=i.applied_at_version,
        evidence_strength=i.evidence_strength,
        performance_before=i.performance_before,
        performance_after=i.performance_after,
        validation_result=i.validation_result,
        reverted_at=i.reverted_at,
        revert_reason=i.revert_reason,
    )


# ══════════════════════════════════════════════════════════════
#  JUDGMENT LEARNING — User setup decisions
# ══════════════════════════════════════════════════════════════


class SetupDecisionRequest(BaseModel):
    """User's accept/reject decision on a presented setup."""
    judgment_name: str  # e.g., "compression_quality_judgment"
    accepted: bool
    candles: List[Dict[str, Any]] = []  # OHLCV data around the setup
    compression_start: Optional[int] = None
    compression_end: Optional[int] = None
    reason: str = ""  # optional explanation


class JudgmentStatusResponse(BaseModel):
    strategy_id: str
    initialized: bool
    learners: List[str]
    total_decisions: int
    explanations: Dict[str, str]


@router.post("/strategies/{strategy_id}/setup-decision")
def record_setup_decision(
    strategy_id: UUID,
    request: SetupDecisionRequest,
):
    """
    Record user's YES/NO decision on a presented setup.

    THIS IS THE MOST VALUABLE DATA IN THE SYSTEM.

    Each decision teaches the JudgmentManager what the user's
    implicit judgment model looks like.
    """
    try:
        from bootstrap.live_connector import get_judgment_connector
        connector = get_judgment_connector()
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Judgment learning not available",
        )

    sid = str(strategy_id)
    if not connector.has_manager(sid):
        raise HTTPException(
            status_code=404,
            detail=f"No JudgmentManager for strategy {sid}. Run bootstrap first.",
        )

    ok = connector.record_decision(
        strategy_id=sid,
        judgment_name=request.judgment_name,
        candles=request.candles,
        user_accepted=request.accepted,
        compression_start=request.compression_start,
        compression_end=request.compression_end,
    )

    if not ok:
        raise HTTPException(status_code=500, detail="Failed to record decision")

    # Get updated status
    manager = connector.get_manager(sid)
    total = manager.total_decisions(request.judgment_name) if manager else 0

    return {
        "recorded": True,
        "strategy_id": sid,
        "judgment_name": request.judgment_name,
        "accepted": request.accepted,
        "total_decisions_for_factor": total,
    }


@router.get("/strategies/{strategy_id}/judgment-status")
def get_judgment_status(strategy_id: UUID):
    """
    Get judgment learning status for a strategy.

    Shows: active learners, total decisions, explanations of what
    the system has learned about the user's judgment.
    """
    try:
        from bootstrap.live_connector import get_judgment_connector
        connector = get_judgment_connector()
    except ImportError:
        return {
            "strategy_id": str(strategy_id),
            "initialized": False,
            "learners": [],
            "total_decisions": 0,
            "explanations": {},
        }

    sid = str(strategy_id)
    manager = connector.get_manager(sid)
    if not manager:
        return {
            "strategy_id": sid,
            "initialized": False,
            "learners": [],
            "total_decisions": 0,
            "explanations": {},
        }

    learners = list(manager._learners.keys()) if hasattr(manager, "_learners") else []
    total = manager.total_decisions_all() if hasattr(manager, "total_decisions_all") else 0

    explanations = connector.get_judgment_explanations(sid)
    reductions = connector.get_uncertainty_reductions(sid)

    return {
        "strategy_id": sid,
        "initialized": True,
        "learners": learners,
        "total_decisions": total,
        "explanations": explanations,
        "uncertainty_reductions": {
            k: round(v, 3) for k, v in reductions.items()
        },
    }
