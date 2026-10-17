# api/routes/learning.py
"""
Strategy Learning API — Teaching sessions, profiles, trade analysis, co-planning.

Replaces hard-coded strategy rules with AI-learned understanding.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.dependencies import (
    get_db_dependency,
    get_llm_client,
    get_screenshot_store,
    get_vision_analyzer,
)
from config.settings import settings
from core.models import LearningExample, LearningSession, StrategyProfile


def _require_llm():
    """Raise 503 if LLM is not configured — teaching requires LLM."""
    if not settings.LLM_API_KEY:
        raise HTTPException(
            503,
            "LLM API key is not configured. Strategy teaching requires an LLM provider. "
            "Set LLM_API_KEY in your .env file.",
        )
from ingestion.screenshot_handler import get_media_type, resize_if_needed, validate_screenshot
from strategy_learning.models import (
    BatchTrainingReport,
    CoplanRequest,
    CoplanResponse,
    ProfileConfidence,
    StrategyProfileSummary,
    TradeAnalysisRequest,
    TradeAnalysisResponse,
    UserAnnotation,
    VisionAnalysis,
)
from strategy_learning.screenshot_store import ScreenshotStore
from strategy_learning.session_manager import LearningSessionManager
from strategy_learning.vision_analyzer import ChartVisionAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/learning", tags=["strategy-learning"], dependencies=[Depends(get_current_user)])


# ══════════════════════════════════════════════
#  Request/Response Schemas
# ══════════════════════════════════════════════

class CreateProfileRequest(BaseModel):
    name: str
    short_code: str
    tags: list[str] = []
    preferred_timeframes: list[str] = []   # e.g. ["5m"], ["15m"], ["5m","15m"]


class StartSessionRequest(BaseModel):
    profile_id: UUID
    session_type: str = "initial_teaching"
    notes: str = ""


class AnnotateRequest(BaseModel):
    example_id: UUID
    annotations: UserAnnotation


class ProfileResponse(BaseModel):
    id: UUID
    name: str
    short_code: str
    status: str
    narrative: Optional[str] = None
    entry_criteria: list[dict[str, Any]] = []
    exit_rules: dict[str, Any] = {}
    risk_parameters: dict[str, Any] = {}
    regime_preferences: dict[str, Any] = {}
    time_preferences: dict[str, Any] = {}
    confidence_scores: dict[str, Any] = {}
    examples_count: int = 0
    sessions_count: int = 0
    tags: list[str] = []
    preferred_timeframes: list[str] = []


class SessionResponse(BaseModel):
    id: UUID
    strategy_profile_id: UUID
    status: str
    session_type: str
    examples_count: int = 0
    notes: Optional[str] = None
    ai_summary: Optional[str] = None


class ExampleResponse(BaseModel):
    id: UUID
    screenshot_path: str
    vision_analysis: dict[str, Any] = {}
    user_annotations: dict[str, Any] = {}
    ai_interpretation: Optional[str] = None
    is_positive_example: bool = True
    market_context: dict[str, Any] = {}


# ══════════════════════════════════════════════
#  PROFILE CRUD
# ══════════════════════════════════════════════

@router.post("/profiles", response_model=ProfileResponse)
async def create_profile(
    req: CreateProfileRequest,
    db: Session = Depends(get_db_dependency),
    user=Depends(get_current_user),
):
    """Create a new strategy profile to start teaching (owned by the caller)."""
    # Short code must be unique within the caller's own strategies.
    existing = db.execute(
        select(StrategyProfile).where(
            StrategyProfile.short_code == req.short_code,
            StrategyProfile.trader_id == user.id,
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(400, f"Short code '{req.short_code}' already exists")

    tfs = req.preferred_timeframes or ["15m"]
    profile = StrategyProfile(
        id=uuid4(),
        name=req.name,
        short_code=req.short_code,
        trader_id=user.id,
        status="draft",
        tags=req.tags,
        profile_metadata={"preferred_timeframes": tfs},
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)

    return _profile_to_response(profile)


@router.get("/profiles", response_model=list[ProfileResponse])
async def list_profiles(
    db: Session = Depends(get_db_dependency),
    user=Depends(get_current_user),
):
    """List the caller's own strategy profiles."""
    result = db.execute(
        select(StrategyProfile)
        .where(StrategyProfile.trader_id == user.id)
        .order_by(StrategyProfile.created_at.desc())
    )
    profiles = result.scalars().all()
    return [_profile_to_response(p) for p in profiles]


@router.get("/profiles/{profile_id}", response_model=ProfileResponse)
async def get_profile(
    profile_id: UUID,
    db: Session = Depends(get_db_dependency),
    user=Depends(get_current_user),
):
    """Get full strategy profile with learned understanding (caller-owned)."""
    profile = db.get(StrategyProfile, profile_id)
    if not profile or profile.trader_id != user.id:
        raise HTTPException(404, "Profile not found")
    return _profile_to_response(profile)


@router.delete("/profiles/{profile_id}")
async def archive_profile(
    profile_id: UUID,
    db: Session = Depends(get_db_dependency),
    user=Depends(get_current_user),
):
    """Archive a strategy profile (caller-owned)."""
    profile = db.get(StrategyProfile, profile_id)
    if not profile or profile.trader_id != user.id:
        raise HTTPException(404, "Profile not found")
    profile.status = "archived"
    db.commit()
    return {"status": "archived", "id": str(profile_id)}


# ══════════════════════════════════════════════
#  LEARNING SESSIONS
# ══════════════════════════════════════════════

@router.post("/sessions", response_model=SessionResponse)
async def start_session(
    req: StartSessionRequest,
    db: Session = Depends(get_db_dependency),
):
    """Start a new teaching session for a profile."""
    mgr = LearningSessionManager(db)
    try:
        session = mgr.start_session(
            profile_id=req.profile_id,
            session_type=req.session_type,
            notes=req.notes,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _session_to_response(session)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    db: Session = Depends(get_db_dependency),
):
    """Get session details."""
    session = db.get(LearningSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return _session_to_response(session)


@router.post("/sessions/{session_id}/screenshot", response_model=ExampleResponse)
async def upload_screenshot(
    session_id: UUID,
    file: UploadFile = File(...),
    context: str = Form(""),
    db: Session = Depends(get_db_dependency),
    analyzer: ChartVisionAnalyzer = Depends(get_vision_analyzer),
    store: ScreenshotStore = Depends(get_screenshot_store),
):
    """Upload a chart screenshot for AI analysis."""
    _require_llm()

    session = db.get(LearningSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.status != "active":
        raise HTTPException(400, f"Session is {session.status}, not active")

    # Read and validate file
    file_bytes = await file.read()
    try:
        validate_screenshot(file_bytes, file.filename or "screenshot.png")
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Resize if needed
    file_bytes = resize_if_needed(file_bytes)

    # Save screenshot to disk
    screenshot_path = store.save(
        profile_id=session.strategy_profile_id,
        session_id=session_id,
        file_bytes=file_bytes,
        filename=file.filename or "screenshot.png",
    )
    screenshot_hash = store.compute_hash(file_bytes)

    # Run AI vision analysis
    media_type = get_media_type(file.filename or "screenshot.png")
    vision_analysis = await analyzer.analyze_screenshot_bytes(
        image_bytes=file_bytes,
        media_type=media_type,
        context=context,
    )

    # Store example (without user annotations yet — those come via /annotate)
    mgr = LearningSessionManager(db)
    example = mgr.add_example(
        session_id=session_id,
        screenshot_path=screenshot_path,
        screenshot_hash=screenshot_hash,
        vision_analysis=vision_analysis,
        user_annotations=None,
        market_context={"context_note": context} if context else {},
    )

    return _example_to_response(example)


@router.post("/sessions/{session_id}/annotate", response_model=ExampleResponse)
async def annotate_example(
    session_id: UUID,
    req: AnnotateRequest,
    db: Session = Depends(get_db_dependency),
):
    """Add user annotations to a learning example."""
    session = db.get(LearningSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.status != "active":
        raise HTTPException(400, f"Session is {session.status}")

    mgr = LearningSessionManager(db)
    try:
        example = mgr.update_annotations(req.example_id, req.annotations)
    except ValueError as e:
        raise HTTPException(404, str(e))

    return _example_to_response(example)


@router.post("/sessions/{session_id}/complete", response_model=SessionResponse)
async def complete_session(
    session_id: UUID,
    db: Session = Depends(get_db_dependency),
):
    """Complete a teaching session and trigger profile rebuild."""
    mgr = LearningSessionManager(db)
    try:
        session = mgr.complete_session(session_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Trigger async profile rebuild
    from strategy_learning.strategy_profile_builder import ProfileBuilder
    builder = ProfileBuilder(db=db, llm=get_llm_client())
    try:
        await builder.build_profile(session.strategy_profile_id)
    except Exception as e:
        logger.error("Profile rebuild failed: %s", e)
        # Session is still completed even if rebuild fails

    # Promote to an active "my strategy" and wire it into the intelligence
    # registry so the scanner / co-planner / behavioral matcher become aware of
    # it — same end-state as a finalized Bootstrap strategy.
    try:
        profile = db.get(StrategyProfile, session.strategy_profile_id)
        if profile is not None:
            if profile.status in ("draft", "learning"):
                profile.status = "active"
                db.commit()
            from strategy_engine.templates.profile_to_template import save_template_yaml
            save_template_yaml(profile)
            from strategy_engine.strategy_registry import get_registry
            get_registry().reload()
    except Exception as e:
        logger.warning("Wiring taught strategy into intelligence failed: %s", e)

    return _session_to_response(session)


@router.delete("/sessions/{session_id}")
async def abandon_session(
    session_id: UUID,
    db: Session = Depends(get_db_dependency),
):
    """Abandon a teaching session."""
    mgr = LearningSessionManager(db)
    try:
        mgr.abandon_session(session_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "abandoned"}


# ══════════════════════════════════════════════
#  EXAMPLES & ANALYSIS
# ══════════════════════════════════════════════

@router.get("/profiles/{profile_id}/examples", response_model=list[ExampleResponse])
async def get_profile_examples(
    profile_id: UUID,
    db: Session = Depends(get_db_dependency),
):
    """List all learning examples for a profile."""
    result = db.execute(
        select(LearningExample)
        .where(LearningExample.strategy_profile_id == profile_id)
        .order_by(LearningExample.created_at.desc())
    )
    examples = result.scalars().all()
    return [_example_to_response(ex) for ex in examples]


@router.get("/profiles/{profile_id}/questions")
async def get_next_questions(
    profile_id: UUID,
    db: Session = Depends(get_db_dependency),
):
    """Get AI-generated questions for areas needing more examples."""
    from strategy_learning.strategy_profile_builder import ProfileBuilder
    builder = ProfileBuilder(db=db, llm=get_llm_client())
    questions = await builder.get_next_questions(profile_id)
    return {"questions": questions}


@router.post("/profiles/{profile_id}/analyze-trade", response_model=TradeAnalysisResponse)
async def analyze_trade_against_profile(
    profile_id: UUID,
    req: TradeAnalysisRequest,
    db: Session = Depends(get_db_dependency),
):
    """Analyze a trade against the learned profile."""
    profile = db.get(StrategyProfile, profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")

    from strategy_learning.trade_analyzer import TradeAnalyzer
    analyzer = TradeAnalyzer(db=db, llm=get_llm_client())
    result = await analyzer.analyze_trade(trade_request=req, profile=profile)
    return result


@router.post("/coplan", response_model=CoplanResponse)
async def coplan_trade(
    profile_id: UUID = Form(...),
    context: str = Form(""),
    symbol: str = Form(""),
    timeframe: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db_dependency),
    analyzer: ChartVisionAnalyzer = Depends(get_vision_analyzer),
    store: ScreenshotStore = Depends(get_screenshot_store),
):
    """Co-plan a trade: upload current chart + get AI recommendations."""
    profile = db.get(StrategyProfile, profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")

    # Read and analyze screenshot
    file_bytes = await file.read()
    try:
        validate_screenshot(file_bytes, file.filename or "chart.png")
    except ValueError as e:
        raise HTTPException(400, str(e))

    file_bytes = resize_if_needed(file_bytes)
    media_type = get_media_type(file.filename or "chart.png")

    vision_analysis = await analyzer.analyze_screenshot_bytes(
        image_bytes=file_bytes,
        media_type=media_type,
        context=f"Symbol: {symbol}, Timeframe: {timeframe}. {context}",
    )

    from strategy_learning.coplan_engine import CoplanEngine
    engine = CoplanEngine(db=db, llm=get_llm_client())
    result = await engine.coplan(
        vision_analysis=vision_analysis,
        profile=profile,
        context=context,
        symbol=symbol,
        timeframe=timeframe,
    )
    return result


# ══════════════════════════════════════════════
#  BATCH TRAINING (multi-screenshot)
# ══════════════════════════════════════════════


class BatchUploadResponse(BaseModel):
    session_id: UUID
    total_uploaded: int
    analysed_successfully: int
    failed: int
    examples: list[ExampleResponse] = []


class BatchTrainRequest(BaseModel):
    context: str = ""


class BatchTrainingReportResponse(BaseModel):
    session_id: UUID
    report: BatchTrainingReport
    profile_updated: bool = False


@router.post(
    "/sessions/{session_id}/batch-upload",
    response_model=BatchUploadResponse,
)
async def batch_upload_screenshots(
    session_id: UUID,
    files: list[UploadFile] = File(...),
    context: str = Form(""),
    db: Session = Depends(get_db_dependency),
    analyzer: ChartVisionAnalyzer = Depends(get_vision_analyzer),
    store: ScreenshotStore = Depends(get_screenshot_store),
):
    """
    Upload 15-20 chart screenshots at once for batch strategy training.

    Each screenshot is validated, stored, and analysed by Vision AI.
    After upload, call /batch-train to cross-reference all screenshots
    and synthesise a unified strategy understanding.
    """
    session = db.get(LearningSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.status != "active":
        raise HTTPException(400, f"Session is {session.status}, not active")

    if len(files) < 3:
        raise HTTPException(400, "Batch training requires at least 3 screenshots (10-20 recommended)")
    if len(files) > 30:
        raise HTTPException(400, "Maximum 30 screenshots per batch")

    mgr = LearningSessionManager(db)
    examples = []
    n_ok = 0
    n_fail = 0

    for file in files:
        try:
            file_bytes = await file.read()
            filename = file.filename or "screenshot.png"
            validate_screenshot(file_bytes, filename)
            file_bytes = resize_if_needed(file_bytes)

            # Save to disk
            screenshot_path = store.save(
                profile_id=session.strategy_profile_id,
                session_id=session_id,
                file_bytes=file_bytes,
                filename=filename,
            )
            screenshot_hash = store.compute_hash(file_bytes)

            # Vision analysis
            media_type = get_media_type(filename)
            vision_analysis = await analyzer.analyze_screenshot_bytes(
                image_bytes=file_bytes,
                media_type=media_type,
                context=context,
            )

            # Store example
            example = mgr.add_example(
                session_id=session_id,
                screenshot_path=screenshot_path,
                screenshot_hash=screenshot_hash,
                vision_analysis=vision_analysis,
                user_annotations=None,
                market_context={"context_note": context, "batch_upload": True},
            )
            examples.append(_example_to_response(example))
            n_ok += 1

        except Exception as e:
            logger.warning("Batch upload: failed to process %s: %s", file.filename, e)
            n_fail += 1

    return BatchUploadResponse(
        session_id=session_id,
        total_uploaded=len(files),
        analysed_successfully=n_ok,
        failed=n_fail,
        examples=examples,
    )


@router.post(
    "/sessions/{session_id}/batch-train",
    response_model=BatchTrainingReportResponse,
)
async def batch_train(
    session_id: UUID,
    req: BatchTrainRequest = BatchTrainRequest(),
    db: Session = Depends(get_db_dependency),
):
    """
    Cross-reference all screenshots in a session to synthesise strategy understanding.

    This is the core batch training step — analyses all uploaded screenshots,
    finds consistent patterns, identifies contradictions, and generates
    follow-up questions. Like training an autonomous vehicle from labeled data.

    Call this AFTER batch-upload has completed.
    """
    session = db.get(LearningSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Load all examples for this session
    result = db.execute(
        select(LearningExample)
        .where(LearningExample.session_id == session_id)
        .order_by(LearningExample.created_at.asc())
    )
    examples = result.scalars().all()

    if len(examples) < 3:
        raise HTTPException(
            400,
            f"Only {len(examples)} examples in session. Need at least 3 for batch training (10-20 recommended).",
        )

    # Reconstruct VisionAnalysis objects from stored JSON
    analyses: list[VisionAnalysis] = []
    annotations: list[dict] = []
    for ex in examples:
        try:
            va = VisionAnalysis(**(ex.vision_analysis or {}))
            analyses.append(va)
        except Exception:
            analyses.append(VisionAnalysis())
        annotations.append(ex.user_annotations or {})

    _require_llm()

    # Run batch training
    from strategy_learning.batch_trainer import BatchTrainer

    trainer = BatchTrainer(llm_client=get_llm_client())
    report = await trainer.train(
        analyses=analyses,
        user_context=req.context,
        annotations=annotations,
    )

    # Store the report on the session
    session.ai_summary = report.synthesis_narrative
    if hasattr(session, "batch_training_report"):
        session.batch_training_report = report.model_dump()

    # If sufficient quality, trigger profile update
    profile_updated = False
    if report.is_sufficient:
        try:
            from strategy_learning.strategy_profile_builder import ProfileBuilder

            builder = ProfileBuilder(db=db, llm=get_llm_client())
            await builder.build_profile(session.strategy_profile_id)
            profile_updated = True
        except Exception as e:
            logger.error("Profile rebuild after batch train failed: %s", e)

    db.commit()

    return BatchTrainingReportResponse(
        session_id=session_id,
        report=report,
        profile_updated=profile_updated,
    )


@router.get(
    "/sessions/{session_id}/training-report",
    response_model=BatchTrainingReport,
)
async def get_training_report(
    session_id: UUID,
    db: Session = Depends(get_db_dependency),
):
    """
    Get the batch training report for a session.

    Returns the cross-referenced analysis with consistent patterns,
    contradictions, synthesised rules, and follow-up questions.
    """
    session = db.get(LearningSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Re-run synthesis from stored examples
    result = db.execute(
        select(LearningExample)
        .where(LearningExample.session_id == session_id)
        .order_by(LearningExample.created_at.asc())
    )
    examples = result.scalars().all()

    if not examples:
        raise HTTPException(404, "No examples in this session")

    analyses = []
    annotations = []
    for ex in examples:
        try:
            va = VisionAnalysis(**(ex.vision_analysis or {}))
            analyses.append(va)
        except Exception:
            analyses.append(VisionAnalysis())
        annotations.append(ex.user_annotations or {})

    from strategy_learning.batch_trainer import BatchTrainer

    trainer = BatchTrainer(llm_client=get_llm_client())
    report = await trainer.train(
        analyses=analyses,
        annotations=annotations,
    )
    return report


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════

def _profile_to_response(p: StrategyProfile) -> ProfileResponse:
    return ProfileResponse(
        id=p.id,
        name=p.name,
        short_code=p.short_code,
        status=p.status,
        narrative=p.narrative,
        entry_criteria=p.entry_criteria or [],
        exit_rules=p.exit_rules or {},
        risk_parameters=p.risk_parameters or {},
        regime_preferences=p.regime_preferences or {},
        time_preferences=p.time_preferences or {},
        confidence_scores=p.confidence_scores or {},
        examples_count=p.examples_count or 0,
        sessions_count=p.sessions_count or 0,
        tags=p.tags or [],
        preferred_timeframes=(p.profile_metadata or {}).get("preferred_timeframes", ["15m"]),
    )


def _session_to_response(s: LearningSession) -> SessionResponse:
    return SessionResponse(
        id=s.id,
        strategy_profile_id=s.strategy_profile_id,
        status=s.status,
        session_type=s.session_type,
        examples_count=s.examples_count or 0,
        notes=s.notes,
        ai_summary=s.ai_summary,
    )


def _example_to_response(ex: LearningExample) -> ExampleResponse:
    return ExampleResponse(
        id=ex.id,
        screenshot_path=ex.screenshot_path,
        vision_analysis=ex.vision_analysis or {},
        user_annotations=ex.user_annotations or {},
        ai_interpretation=ex.ai_interpretation,
        is_positive_example=ex.is_positive_example,
        market_context=ex.market_context or {},
    )
