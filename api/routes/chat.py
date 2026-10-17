"""
Chat API Routes — WebSocket and REST endpoints for the coaching agent.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agents.coaching_agent import (
    CoachingAgent,
    CoachingContext,
    CoachingMode,
    CoachingPersonality,
    CoachingResponse,
)
from agents.conversation_manager import ConversationManager
from agents.llm_client import LLMClient
from api.auth import get_current_user
from api.dependencies import (
    get_db,
    get_llm_client,
    get_rag_retriever,
    get_current_trader_id,
)
from api.schemas.chat_schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionResponse,
    ChatSessionListResponse,
    ChatHistoryResponse,
    ProactiveBriefingRequest,
    ProactiveBriefingResponse,
    TradeReviewRequest,
    TradeReviewResponse,
    CoachingInsightResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["coaching"], dependencies=[Depends(get_current_user)])


# ─── REST Endpoints ──────────────────────────────────────────────


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    db: AsyncSession = Depends(get_db),
    llm_client: LLMClient = Depends(get_llm_client),
    rag_retriever=Depends(get_rag_retriever),
    trader_id: str = Depends(get_current_trader_id),
):
    """
    Send a message to the AI coaching agent and get a response.

    This is the primary chat endpoint. It:
    1. Gets or creates a conversation session
    2. Assembles full trading context
    3. Sends the message through the coaching agent
    4. Persists both the message and response
    5. Returns the structured coaching response
    """
    # Initialize components
    conversation_mgr = ConversationManager(db)
    coaching_agent = CoachingAgent(
        llm_client=llm_client,
        personality=CoachingPersonality(request.personality or "balanced"),
    )

    # Get or create conversation session
    conversation = await conversation_mgr.get_or_create_session(
        trader_id=trader_id,
        session_id=request.session_id,
    )

    # Record the user message
    await conversation_mgr.add_user_message(
        session_id=conversation.session_id,
        content=request.message,
        metadata={"trade_id": request.trade_id} if request.trade_id else {},
    )

    # Build coaching context
    coaching_context = CoachingContext(
        coaching_personality=CoachingPersonality(request.personality or "balanced"),
    )

    # Determine mode override if specified
    mode_override = None
    if request.mode:
        try:
            mode_override = CoachingMode(request.mode)
        except ValueError:
            pass

    # Generate coaching response
    coaching_response = await coaching_agent.process(
        user_message=request.message,
        context=coaching_context,
        mode=mode_override,
    )

    # Persist the assistant response
    await conversation_mgr.add_assistant_message(
        session_id=conversation.session_id,
        content=coaching_response.message,
        metadata={
            "mode": coaching_response.mode.value,
            "confidence": coaching_response.confidence,
            "insight_count": len(coaching_response.insights),
            "action_count": len(coaching_response.action_items),
        },
    )

    return ChatMessageResponse(
        session_id=conversation.session_id,
        message=coaching_response.message,
        mode=coaching_response.mode.value,
        insights=[
            CoachingInsightResponse(
                id=i.id,
                priority=i.priority.value,
                category=i.category,
                title=i.title,
                content=i.content,
                evidence=i.evidence,
                action_items=i.action_items,
            )
            for i in coaching_response.insights
        ],
        action_items=coaching_response.action_items,
        follow_up_questions=coaching_response.follow_up_questions,
        suggested_focus=coaching_response.suggested_focus,
        confidence=coaching_response.confidence,
    )


@router.post("/trade-review", response_model=TradeReviewResponse)
async def review_trade(
    request: TradeReviewRequest,
    db: AsyncSession = Depends(get_db),
    llm_client: LLMClient = Depends(get_llm_client),
    rag_retriever=Depends(get_rag_retriever),
    trader_id: str = Depends(get_current_trader_id),
):
    """
    Request a detailed AI coaching review of a specific trade.

    This assembles the full context for the trade (DQ report, attribution,
    shadow comparison, behavioral data) and generates a comprehensive review.
    """
    conversation_mgr = ConversationManager(db)
    coaching_agent = CoachingAgent(
        llm_client=llm_client,
        personality=CoachingPersonality(request.personality or "balanced"),
    )

    # Create a new session specifically for this trade review
    conversation = await conversation_mgr.get_or_create_session(
        trader_id=trader_id,
        session_id=request.session_id,
    )

    # Build coaching context
    coaching_context = CoachingContext(
        coaching_personality=CoachingPersonality(request.personality or "balanced"),
    )

    # Construct the review prompt
    user_message = request.message or "Please review my recent trade. What did I do well and what should I improve?"

    if request.focus_areas:
        user_message += f" Please focus especially on: {', '.join(request.focus_areas)}."

    # Record and process
    await conversation_mgr.add_user_message(
        session_id=conversation.session_id,
        content=user_message,
        metadata={"trade_id": request.trade_id, "type": "trade_review"},
    )

    coaching_response = await coaching_agent.process(
        user_message=user_message,
        context=coaching_context,
        mode=CoachingMode.TRADE_REVIEW,
    )

    await conversation_mgr.add_assistant_message(
        session_id=conversation.session_id,
        content=coaching_response.message,
        metadata={
            "mode": "trade_review",
            "trade_id": request.trade_id,
        },
    )

    trade_data: dict = {}
    dq_data: dict = {}

    return TradeReviewResponse(
        session_id=conversation.session_id,
        trade_id=request.trade_id,
        message=coaching_response.message,
        trade_summary={
            "symbol": trade_data.get("symbol"),
            "direction": trade_data.get("direction"),
            "pnl": trade_data.get("pnl"),
            "r_multiple": trade_data.get("r_multiple"),
            "strategy": trade_data.get("strategy"),
        },
        decision_quality={
            "composite_score": dq_data.get("composite_score"),
            "grade": dq_data.get("grade"),
            "dimensions": dq_data.get("dimensions", {}),
        },
        insights=[
            CoachingInsightResponse(
                id=i.id,
                priority=i.priority.value,
                category=i.category,
                title=i.title,
                content=i.content,
                evidence=i.evidence,
                action_items=i.action_items,
            )
            for i in coaching_response.insights
        ],
        action_items=coaching_response.action_items,
        follow_up_questions=coaching_response.follow_up_questions,
    )


@router.post("/briefing", response_model=ProactiveBriefingResponse)
async def get_proactive_briefing(
    request: ProactiveBriefingRequest,
    db: AsyncSession = Depends(get_db),
    llm_client: LLMClient = Depends(get_llm_client),
    rag_retriever=Depends(get_rag_retriever),
    trader_id: str = Depends(get_current_trader_id),
):
    """
    Get a proactive coaching briefing.

    Used for:
    - Pre-session preparation
    - End-of-day debrief
    - Behavioral alert responses
    - Pattern insight notifications
    """
    coaching_agent = CoachingAgent(
        llm_client=llm_client,
        personality=CoachingPersonality(request.personality or "balanced"),
    )

    coaching_context = CoachingContext(
        coaching_personality=CoachingPersonality(request.personality or "balanced"),
    )

    coaching_response = await coaching_agent.generate_proactive_briefing(
        context=coaching_context,
    )

    return ProactiveBriefingResponse(
        message=coaching_response.message,
        mode=coaching_response.mode.value,
        insights=[
            CoachingInsightResponse(
                id=i.id,
                priority=i.priority.value,
                category=i.category,
                title=i.title,
                content=i.content,
                evidence=i.evidence,
                action_items=i.action_items,
            )
            for i in coaching_response.insights
        ],
        action_items=coaching_response.action_items,
        suggested_focus=coaching_response.suggested_focus,
    )


@router.get("/sessions", response_model=ChatSessionListResponse)
async def list_sessions(
    limit: int = Query(default=10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    trader_id: str = Depends(get_current_trader_id),
):
    """List recent coaching conversation sessions."""
    conversation_mgr = ConversationManager(db)
    sessions = await conversation_mgr.get_recent_sessions(
        trader_id=trader_id,
        limit=limit,
    )

    return ChatSessionListResponse(
        sessions=[
            ChatSessionResponse(
                session_id=s["session_id"],
                mode=s.get("mode"),
                preview=s.get("preview"),
                turn_count=s["turn_count"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
            )
            for s in sessions
        ]
    )


@router.get("/sessions/{session_id}/history", response_model=ChatHistoryResponse)
async def get_session_history(
    session_id: str,
    max_turns: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    trader_id: str = Depends(get_current_trader_id),
):
    """Get the full conversation history for a session."""
    conversation_mgr = ConversationManager(db)
    history = await conversation_mgr.get_conversation_history(
        session_id=session_id,
        max_turns=max_turns,
    )

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            {"role": h["role"], "content": h["content"]}
            for h in history
        ],
        total_turns=len(history),
    )


@router.post("/sessions/{session_id}/end")
async def end_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    trader_id: str = Depends(get_current_trader_id),
):
    """End a conversation session."""
    conversation_mgr = ConversationManager(db)
    await conversation_mgr.end_session(session_id)
    return {"status": "ok", "session_id": session_id}


# ─── WebSocket Endpoint ─────────────────────────────────────────


@router.websocket("/ws/{trader_id}")
async def coaching_websocket(
    websocket: WebSocket,
    trader_id: str,
    db: AsyncSession = Depends(get_db),
    llm_client: LLMClient = Depends(get_llm_client),
    rag_retriever=Depends(get_rag_retriever),
):
    """
    WebSocket endpoint for real-time coaching chat.

    Protocol:
    - Client sends JSON: {"message": "...", "session_id": "...", "trade_id": "..."}
    - Server responds with JSON: {"type": "response", "data": {...}}
    - Server can push proactive insights: {"type": "insight", "data": {...}}
    - Server sends typing indicator: {"type": "typing", "data": {"is_typing": true}}
    """
    await websocket.accept()
    logger.info(f"Coaching WebSocket connected for trader {trader_id}")

    conversation_mgr = ConversationManager(db)

    session_id: Optional[str] = None

    try:
        while True:
            # Receive message from client
            raw_data = await websocket.receive_text()

            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid JSON"},
                })
                continue

            message = data.get("message", "").strip()
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Empty message"},
                })
                continue

            # Get or create session
            request_session_id = data.get("session_id", session_id)
            conversation = await conversation_mgr.get_or_create_session(
                trader_id=trader_id,
                session_id=request_session_id,
            )
            session_id = conversation.session_id

            # Send typing indicator
            await websocket.send_json({
                "type": "typing",
                "data": {"is_typing": True},
            })

            # Record user message
            await conversation_mgr.add_user_message(
                session_id=session_id,
                content=message,
                metadata={"trade_id": data.get("trade_id")},
            )

            # Assemble context
            history = await conversation_mgr.get_conversation_history(
                session_id
            )

            personality = CoachingPersonality(
                data.get("personality", "balanced")
            )

            coaching_context = CoachingContext(coaching_personality=personality)

            # Initialize coaching agent
            coaching_agent = CoachingAgent(
                llm_client=llm_client,
                personality=personality,
            )

            # Parse mode override
            mode_override = None
            if data.get("mode"):
                try:
                    mode_override = CoachingMode(data["mode"])
                except ValueError:
                    pass

            # Generate response
            coaching_response = await coaching_agent.process(
                user_message=message,
                context=coaching_context,
                mode=mode_override,
            )

            # Persist assistant response
            await conversation_mgr.add_assistant_message(
                session_id=session_id,
                content=coaching_response.message,
                metadata={
                    "mode": coaching_response.mode.value,
                    "confidence": coaching_response.confidence,
                },
            )

            # Stop typing indicator
            await websocket.send_json({
                "type": "typing",
                "data": {"is_typing": False},
            })

            # Send main response
            await websocket.send_json({
                "type": "response",
                "data": {
                    "session_id": session_id,
                    "message": coaching_response.message,
                    "mode": coaching_response.mode.value,
                    "action_items": coaching_response.action_items,
                    "follow_up_questions": coaching_response.follow_up_questions,
                    "suggested_focus": coaching_response.suggested_focus,
                    "confidence": coaching_response.confidence,
                },
            })

            # Send insights as separate messages
            for insight in coaching_response.insights:
                await websocket.send_json({
                    "type": "insight",
                    "data": insight.to_dict(),
                })

    except WebSocketDisconnect:
        logger.info(f"Coaching WebSocket disconnected for trader {trader_id}")
    except Exception as e:
        logger.error(
            f"Coaching WebSocket error for trader {trader_id}: {e}",
            exc_info=True,
        )
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": "Internal server error"},
            })
        except Exception:
            logger.debug("WS send failed after coaching error for trader %s", trader_id)
class DiscretionaryTradeRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    symbol: str
    direction: Optional[int] = Field(None, ge=-1, le=1)
    timeframe: str = "5m"
    todays_trade_count: int = Field(default=0, ge=0)
    current_pnl: float = 0.0
    session_id: Optional[str] = None
    personality: Optional[str] = "balanced"

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What do you think about this NVDA breakout above 880?",
                "symbol": "NVDA",
                "direction": 1,
                "todays_trade_count": 2,
                "current_pnl": 150.0,
            }
        }


@router.post("/evaluate-trade", response_model=ChatMessageResponse)
async def evaluate_trade_with_coaching(
    request: DiscretionaryTradeRequest,
    db: AsyncSession = Depends(get_db),
    llm_client: LLMClient = Depends(get_llm_client),
    rag_retriever=Depends(get_rag_retriever),
    trader_id: str = Depends(get_current_trader_id),
):
    """
    Evaluate a discretionary trade idea with AI coaching.

    Combines the quantitative intelligence assessment with the
    coaching agent's natural language response. This is the primary
    endpoint for "What do you think about this trade?" queries.

    Flow:
    1. Intelligence layer analyzes market + matches strategies
    2. Coaching agent wraps analysis in personalized guidance
    3. Behavioral guardrails are checked and included
    """
    from intelligence.realtime_service import RealtimeIntelligenceService
    from intelligence.price_fetcher import get_price_fetcher
    from config.settings import settings
    from strategy_engine.strategy_registry import StrategyRegistry
    from strategy_engine.conditions.registry import ConditionRegistry

    # Initialize intelligence service
    registry = StrategyRegistry()
    registry.initialize()          # load YAML templates
    condition_registry = ConditionRegistry()
    price_fetcher = get_price_fetcher(settings)

    intel_service = RealtimeIntelligenceService(
        db=db,
        strategy_registry=registry,
        condition_registry=condition_registry,
        price_fetcher=price_fetcher,
    )

    # Get quantitative assessment
    try:
        assessment = await intel_service.evaluate_discretionary(
            query=request.query,
            symbol=request.symbol,
            trader_id=trader_id,
            direction=request.direction,
            todays_trade_count=request.todays_trade_count,
            current_pnl=request.current_pnl,
            timeframe=request.timeframe,
        )
    except Exception as e:
        logger.error(f"Intelligence assessment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Trade evaluation failed")

    # Initialize coaching components
    conversation_mgr = ConversationManager(db)
    coaching_agent = CoachingAgent(
        llm_client=llm_client,
        personality=CoachingPersonality(request.personality or "balanced"),
    )

    # Get or create session
    conversation = await conversation_mgr.get_or_create_session(
        trader_id=trader_id,
        session_id=request.session_id,
    )

    # Record user message
    await conversation_mgr.add_user_message(
        session_id=conversation.session_id,
        content=request.query,
        metadata={
            "type": "discretionary_query",
            "symbol": request.symbol,
            "direction": request.direction,
        },
    )

    # Build coaching context
    coaching_context = CoachingContext(
        coaching_personality=CoachingPersonality(request.personality or "balanced"),
    )

    # Generate coaching response with assessment
    coaching_response = await coaching_agent.process_discretionary_query(
        user_message=request.query,
        context=coaching_context,
        assessment=assessment,
    )

    # Persist response
    await conversation_mgr.add_assistant_message(
        session_id=conversation.session_id,
        content=coaching_response.message,
        metadata={
            "type": "discretionary_evaluation",
            "symbol": request.symbol,
            "recommendation": assessment.recommendation,
            "confidence": assessment.confidence,
        },
    )

    return ChatMessageResponse(
        session_id=conversation.session_id,
        message=coaching_response.message,
        mode="discretionary_evaluation",
        insights=[
            CoachingInsightResponse(
                id=i.id,
                priority=i.priority.value,
                category=i.category,
                title=i.title,
                content=i.content,
                evidence=i.evidence,
                action_items=i.action_items,
            )
            for i in coaching_response.insights
        ],
        action_items=coaching_response.action_items,
        follow_up_questions=coaching_response.follow_up_questions,
        suggested_focus=coaching_response.suggested_focus,
        confidence=coaching_response.confidence,
    )


# ── Co-Planner (context-aware) ────────────────────────────────────


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
TEXT_EXTENSIONS = {".csv", ".txt", ".json"}
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10 MB


def _get_media_type(filename: str) -> str:
    """Return MIME type for image files."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(ext, "image/png")


async def _process_attachments(
    files: list[UploadFile],
    llm_client: LLMClient,
    symbol: Optional[str],
) -> tuple[str, list[dict]]:
    """
    Process uploaded files and return (text_context, attachment_metadata).

    - Images -> run through ChartVisionAnalyzer -> structured chart analysis
    - Text files (CSV, TXT, JSON) -> read content and inject as text
    - PDFs -> read first 5000 chars as text
    """
    from strategy_learning.vision_analyzer import ChartVisionAnalyzer
    from ingestion.screenshot_handler import resize_if_needed

    text_parts: list[str] = []
    metadata: list[dict] = []

    for f in files:
        file_bytes = await f.read()
        fname = f.filename or "attachment"

        if len(file_bytes) > MAX_ATTACHMENT_SIZE:
            text_parts.append(f"[Skipped {fname}: exceeds 10 MB limit]")
            continue

        ext = ("." + fname.rsplit(".", 1)[-1].lower()) if "." in fname else ""

        if ext in IMAGE_EXTENSIONS:
            # ── Chart screenshot analysis ──
            try:
                file_bytes = resize_if_needed(file_bytes)
                media_type = _get_media_type(fname)
                analyzer = ChartVisionAnalyzer()   # builds its OWN vision client (llm_client is text-only)
                vision = await analyzer.analyze_screenshot_bytes(
                    image_bytes=file_bytes,
                    media_type=media_type,
                    context=f"Symbol: {symbol}" if symbol else "",
                )
                parts = [f"\n## Chart Analysis — {fname}"]
                if vision.overall_context:
                    parts.append(vision.overall_context)
                if vision.candle_patterns:
                    names = ", ".join(
                        cp.pattern for cp in vision.candle_patterns
                    )
                    parts.append(f"**Patterns detected:** {names}")
                if vision.price_levels:
                    levels = ", ".join(
                        f"₹{pl.level:,.0f} ({pl.label})"
                        for pl in vision.price_levels[:5]
                    )
                    parts.append(f"**Key levels:** {levels}")
                if vision.support_resistance:
                    parts.append(
                        "**S/R zones:** " + ", ".join(vision.support_resistance[:5])
                    )
                if vision.trend_direction:
                    parts.append(f"**Trend:** {vision.trend_direction}")
                if vision.market_structure:
                    parts.append(f"**Structure:** {vision.market_structure}")
                if vision.key_observations:
                    parts.append(
                        "**Observations:** " + "; ".join(vision.key_observations[:5])
                    )
                text_parts.append("\n".join(parts))
                metadata.append({"name": fname, "type": media_type, "analyzed": True})
            except Exception as exc:
                logger.warning("Vision analysis failed for %s: %s", fname, exc)
                text_parts.append(f"[Chart image {fname} uploaded but analysis failed]")
                metadata.append({"name": fname, "type": _get_media_type(fname), "analyzed": False})

        elif ext in TEXT_EXTENSIONS:
            # ── Text/CSV/JSON file ──
            try:
                content = file_bytes.decode("utf-8", errors="replace")
                # Truncate very large text files
                if len(content) > 8000:
                    content = content[:8000] + "\n... [truncated]"
                text_parts.append(f"\n## Attached File — {fname}\n```\n{content}\n```")
                metadata.append({"name": fname, "type": f.content_type or "text/plain", "analyzed": True})
            except Exception as exc:
                logger.warning("Text read failed for %s: %s", fname, exc)
                metadata.append({"name": fname, "type": "text/plain", "analyzed": False})

        elif ext == ".pdf":
            # ── PDF — extract raw text ──
            try:
                content = file_bytes.decode("utf-8", errors="replace")[:5000]
                text_parts.append(f"\n## Attached PDF — {fname}\n```\n{content}\n```")
                metadata.append({"name": fname, "type": "application/pdf", "analyzed": True})
            except Exception:
                metadata.append({"name": fname, "type": "application/pdf", "analyzed": False})
        else:
            text_parts.append(f"[Unsupported file type: {fname}]")
            metadata.append({"name": fname, "type": "unknown", "analyzed": False})

    return "\n".join(text_parts), metadata


@router.post("/co-plan", response_model=ChatMessageResponse)
async def co_plan(
    query: str = Form(...),
    symbol: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    personality: str = Form("supportive"),
    files: list[UploadFile] = File(default=[]),
    db: AsyncSession = Depends(get_db),
    trader_id: str = Depends(get_current_trader_id),
    llm_client: LLMClient = Depends(get_llm_client),
    rag_retriever=Depends(get_rag_retriever),
):
    """
    Context-enriched co-planning with optional file/screenshot attachments.

    Accepts multipart form data:
    - query: the trader's question
    - symbol: optional stock symbol for context
    - files: optional chart screenshots, CSVs, text files
    - Images are analyzed via ChartVisionAnalyzer (patterns, levels, structure)
    - Text files are read and injected as context
    """
    try:
        return await _co_plan_inner(
            query=query, symbol=symbol, session_id=session_id,
            personality=personality, files=files, db=db,
            trader_id=trader_id, llm_client=llm_client, rag_retriever=rag_retriever,
        )
    except Exception as _top_exc:
        logger.error("co-plan top-level crash: %s", _top_exc, exc_info=True)
        return ChatMessageResponse(
            session_id=session_id or "error",
            message=(
                "Something went wrong on my end. Please try again.\n\n"
                f"_(Error: {type(_top_exc).__name__}: {str(_top_exc)[:120]})_"
            ),
            mode="co_planner",
            insights=[],
            action_items=[],
            follow_up_questions=[],
            suggested_focus=None,
            confidence=0.0,
        )


def _format_strategy_details(template) -> str:
    """
    Build a human-readable summary of a strategy template's actual parameters.
    Used to inject real EMA periods, RSI thresholds, time windows, etc. into the LLM context
    so it can answer specific questions like 'which EMA?', 'what timeframe?', 'what RSI range?'.
    """
    lines = [f"### {template.name}"]
    if getattr(template, "description", None):
        lines.append(f"  Edge: {template.description}")
    if getattr(template, "direction", None):
        lines.append(f"  Direction: {template.direction}")

    # Time window
    tw = getattr(template, "time_window", None)
    if tw:
        lines.append(f"  Time window: {tw.start} - {tw.end} IST")

    # Preferred timeframe
    exec_prefs = getattr(template, "execution", None)
    if exec_prefs and getattr(exec_prefs, "preferred_timeframe", None):
        lines.append(f"  Timeframe: {exec_prefs.preferred_timeframe}")

    # Regimes
    pref = [r.value if hasattr(r, "value") else str(r) for r in getattr(template, "preferred_regimes", [])]
    if pref:
        lines.append(f"  Preferred regimes: {', '.join(pref)}")
    avoid = [r.value if hasattr(r, "value") else str(r) for r in getattr(template, "avoid_regimes", [])]
    if avoid:
        lines.append(f"  Avoid regimes: {', '.join(avoid)}")

    # Entry confirmations — the most important part (contains EMA periods, RSI ranges, etc.)
    confs = getattr(template, "entry_confirmations", [])
    if confs:
        lines.append("  Entry conditions:")
        for c in confs:
            req = "REQUIRED" if c.required else "supporting"
            params = c.params or {}
            param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
            param_str = f" ({param_str})" if param_str else ""
            desc = c.description or c.type
            lines.append(f"    - {desc} [{req}]{param_str}")

    # Exit rules
    exit_rules = getattr(template, "exit_rules", None)
    if exit_rules:
        sl = getattr(exit_rules, "stop_loss", None)
        if sl:
            sl_params = ", ".join(f"{k}={v}" for k, v in (sl.params or {}).items())
            lines.append(f"  Stop loss: {sl.type}" + (f" ({sl_params})" if sl_params else ""))
        targets = getattr(exit_rules, "targets", []) or []
        if targets:
            t_strs = [f"{t.r_multiple}R ({t.size_pct}%)" for t in targets]
            lines.append(f"  Targets: {' / '.join(t_strs)}")
        trail = getattr(exit_rules, "trailing", None)
        if trail and getattr(trail, "enabled", False):
            t_params = ", ".join(f"{k}={v}" for k, v in (trail.params or {}).items())
            activate = getattr(trail, "activate_at_r", None)
            lines.append(f"  Trailing: {trail.type}" + (f" after {activate}R" if activate else "") + (f" ({t_params})" if t_params else ""))

    # Risk
    risk = getattr(template, "risk_rules", None)
    if risk:
        risk_bits = []
        if getattr(risk, "max_risk_pct", None):
            risk_bits.append(f"risk {risk.max_risk_pct}%/trade")
        if getattr(risk, "max_daily_trades", None):
            risk_bits.append(f"max {risk.max_daily_trades} trades/day")
        if risk_bits:
            lines.append(f"  Risk: {', '.join(risk_bits)}")

    return "\n".join(lines)


def _build_coplan_insights(
    reply: str,
    intent_type: str,
    intel_data: Optional[dict] = None,
) -> tuple[list, list, str]:
    """
    Parse a structured co-plan LLM reply and extract:
      - insights:     list of CoachingInsightResponse dicts (verdict, conditions, trade plan)
      - action_items: list[str] — what needs to change for a GO
      - voice_text:   the reply stripped of structured sections (just the voice sentences)

    Expected LLM output format:
      VERDICT: [GO/WATCH/WAIT/NO-GO] — [reason]
      CONDITIONS:
      ✅/❌/⚠️ [condition] — [value]
      TRADE PLAN (optional):
      Entry: ... | SL: ... | T1: ... | R:R ...
      [voice sentences]
      Valid until ...
    """
    insights: list = []
    action_items: list[str] = []

    if intent_type != "setup_analysis":
        return insights, action_items, reply

    lines = reply.splitlines()
    voice_lines: list[str] = []
    i = 0
    in_conditions = False
    in_trade_plan = False
    cond_lines: list[str] = []
    trade_plan_lines: list[str] = []
    verdict_captured = False
    trade_plan_captured = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── VERDICT line ──
        if stripped.upper().startswith("VERDICT:") and not verdict_captured:
            rest = stripped[len("VERDICT:"):].strip()
            parts = rest.split("—", 1)
            verdict_word = parts[0].strip().upper()  # GO / WATCH / WAIT / NO-GO
            verdict_reason = parts[1].strip() if len(parts) > 1 else rest
            severity_map = {
                "GO": "positive",
                "WATCH": "info",
                "WAIT": "warning",
                "NO-GO": "critical",
                "NO_GO": "critical",
            }
            severity = severity_map.get(verdict_word, "warning")
            insights.append(CoachingInsightResponse(
                id="verdict",
                severity=severity,
                priority=severity,
                category="verdict",
                title=f"VERDICT: {verdict_word}",
                message=verdict_reason,
                content=verdict_reason,
            ))
            verdict_captured = True
            in_conditions = False
            in_trade_plan = False
            i += 1
            continue

        # ── CONDITIONS section header ──
        if stripped.upper().startswith("CONDITIONS") and ":" in stripped:
            in_conditions = True
            in_trade_plan = False
            i += 1
            continue

        # ── TRADE PLAN section ──
        if stripped.upper().startswith("TRADE PLAN") and not trade_plan_captured:
            in_conditions = False
            in_trade_plan = True
            trade_plan_lines = []
            i += 1
            continue

        # ── Condition lines ✅❌⚠️ ──
        if in_conditions and stripped and stripped[0] in ("✅", "❌", "⚠️"):
            # Safety net: strip any "(Strategy Name)" suffix the LLM may have left in
            # despite the prompt rule. Pattern: " (Compression Breakout)" → ""
            import re as _re
            cleaned = _re.sub(r"\s*\([A-Z][A-Za-z ]+(?:Reversal|Breakout|Pullback|Bounce|Fade|Fill)?\)\s*", " ", stripped).strip()
            # Collapse double spaces
            cleaned = _re.sub(r"\s{2,}", " ", cleaned)
            cond_lines.append(cleaned)
            if cleaned.startswith("❌"):
                clean = cleaned.lstrip("❌").strip().split("—")[0].strip()
                action_items.append(f"Missing: {clean}")
            i += 1
            continue

        # ── Trade plan content ──
        if in_trade_plan and stripped:
            if stripped[0] in ("✅", "❌", "⚠️") or stripped.upper().startswith("VALID"):
                in_trade_plan = False
            else:
                trade_plan_lines.append(stripped)
                i += 1
                continue

        # ── End of conditions block (blank line or non-condition) ──
        if in_conditions and stripped and stripped[0] not in ("✅", "❌", "⚠️"):
            in_conditions = False

        # ── Everything else = voice text ──
        voice_lines.append(line)
        i += 1

    # Build conditions insight card
    if cond_lines:
        met = sum(1 for l in cond_lines if l.startswith("✅"))
        total = len(cond_lines)
        missing_count = sum(1 for l in cond_lines if l.startswith("❌"))
        cond_severity = "positive" if missing_count == 0 else ("warning" if missing_count <= 2 else "critical")
        insights.append(CoachingInsightResponse(
            id="conditions",
            severity=cond_severity,
            priority=cond_severity,
            category="conditions",
            title=f"Conditions: {met}/{total} met",
            message="\n".join(cond_lines),
            content="\n".join(cond_lines),
        ))

    # Build trade plan insight card — but ONLY if conditions justify it.
    # Three guards to prevent LLM hallucination of unactionable trade plans:
    #   (1) ≥75% of conditions met
    #   (2) NO entry-trigger condition is in the failed (❌) list
    #       (entry-trigger keywords: "near", "above", "below", "break", "reach", "trigger",
    #        EMA/VWAP/BB/ORB level mentions)
    #   (3) Entry price within ~0.6% of current price (intraday-actionable)
    if trade_plan_lines and cond_lines:
        met_count = sum(1 for l in cond_lines if l.startswith("✅"))
        total_count = len(cond_lines)
        ratio = met_count / total_count if total_count else 0
        suppress_reason = None

        # Guard 1: ratio
        if ratio < 0.75:
            suppress_reason = f"only {met_count}/{total_count} conditions met (need ≥75%)"

        # Guard 2: entry-trigger unmet?
        # Entry-trigger conditions are critical — if the price isn't AT the level the strategy
        # requires, the strategy isn't actionable yet.
        if not suppress_reason:
            _ENTRY_TRIGGER_KEYWORDS = (
                "near", "above", "below", "break", "reach",
                "ema", "vwap", "bb", "bollinger", "orb", "opening range",
                "trigger", "compression", "breakout",
            )
            for cond in cond_lines:
                if not cond.startswith("❌"):
                    continue
                # Strip emoji + lowercase for keyword check
                cond_lower = cond.lstrip("❌").strip().lower()
                if any(kw in cond_lower for kw in _ENTRY_TRIGGER_KEYWORDS):
                    suppress_reason = f"entry-trigger condition NOT met: '{cond_lower[:60]}'"
                    break

        # Guard 3: entry price too far from current — extract Entry: ₹X and current price proxy
        # If we can't parse, skip this guard (don't be over-strict).
        if not suppress_reason and intel_data:
            try:
                import re as _re
                entry_text = " ".join(trade_plan_lines)
                m_entry = _re.search(r"entry[:\s]*₹?\s*([0-9]+(?:[.,][0-9]+)?)", entry_text, _re.I)
                # Current price comes from intel_data — try common keys
                cur = intel_data.get("ltp") or intel_data.get("close") or intel_data.get("current_price")
                if m_entry and cur:
                    entry_price = float(m_entry.group(1).replace(",", ""))
                    cur_price = float(cur)
                    if cur_price > 0:
                        pct_away = abs(entry_price - cur_price) / cur_price
                        # 0.6% threshold for intraday — beyond this it's swing territory
                        if pct_away > 0.006:
                            suppress_reason = (
                                f"entry ₹{entry_price:.2f} is {pct_away*100:.2f}% from current "
                                f"₹{cur_price:.2f} — not intraday-actionable"
                            )
            except Exception as _e:
                logger.debug("Entry-distance check skipped: %s", _e)

        if suppress_reason is None:
            trade_plan_captured = True
            insights.append(CoachingInsightResponse(
                id="trade_plan",
                severity="positive",
                priority="positive",
                category="trade_plan",
                title="Trade Plan",
                message="\n".join(trade_plan_lines),
                content="\n".join(trade_plan_lines),
            ))
        else:
            logger.info(
                "Trade plan suppressed: %s. LLM tried: %s",
                suppress_reason, " | ".join(trade_plan_lines)[:150],
            )
            # Convert the suppressed plan into an action_item so trader sees what would have
            # been suggested (if conditions evolve)
            action_items.append(
                f"Plan suppressed ({suppress_reason}). LLM-proposed entry held back."
            )
    elif trade_plan_lines:
        # No conditions present — treat as ungrounded, suppress
        logger.info("Trade plan suppressed: no CONDITIONS block to validate against")

    # Voice text = stripped of structured sections
    voice_text = "\n".join(voice_lines).strip()
    return insights, action_items, voice_text


async def _co_plan_inner(
    query: str,
    symbol: Optional[str],
    session_id: Optional[str],
    personality: str,
    files,
    db,
    trader_id: str,
    llm_client,
    rag_retriever,
):
    from agents.coplanner_context import CoPlannerContextAssembler
    from core.database import SessionLocal
    from agents.llm_client import LLMMessage as _IntentMsg
    import json as _json

    # ── Step 0: LLM-based intent extraction ──────────────────────────
    # Ask a fast LLM call to understand what the user actually wants.
    # This replaces brittle regex and handles all natural language forms.
    effective_symbol = symbol  # Use explicit symbol from frontend if provided
    intent_type = "general"    # setup_analysis | price_query | portfolio_review | behavioral | watchlist_scan | general
    intent_timeframe = "15m"
    intent_direction: Optional[str] = None  # long | short | None
    _intent_data: dict = {}

    # Watchlist-scan heuristic: run intent extraction EVEN when a symbol was passed
    # from the frontend, so that cross-watchlist queries ("any breakout in watchlist?",
    # "scan all stocks", "which setups are forming?") are correctly identified and not
    # locked to the currently active symbol.
    _WATCHLIST_TRIGGERS = {
        "watchlist", "all stocks", "any stocks", "any breakout", "any setup",
        "any signal", "any trade", "scan all", "which stock", "which stocks",
        "which ones", "top pick", "top picks", "best setup", "any forming",
    }
    _query_lower = query.lower()
    _might_be_watchlist = any(t in _query_lower for t in _WATCHLIST_TRIGGERS)

    # Always run intent extraction:
    # - Needed when user asks about a DIFFERENT stock than the one focused in the watchlist panel
    #   e.g. user focused on NESTLEIND but asks "is there any entry in Angel One?"
    # - Also needed for watchlist scan detection
    # If LLM extracts a symbol → it overrides the frontend-passed active symbol
    # If LLM returns null → frontend-passed symbol is kept as-is
    # Fetch last assistant message so the intent extractor can resolve contextual queries
    # like "yes give the template", "show me", "what about that one" — without these,
    # the LLM has no anchor and falls back to the frontend's activeSymbol incorrectly.
    _last_assistant_snippet = ""
    _last_user_snippet = ""
    if session_id:
        try:
            _ctx_mgr = ConversationManager(db)
            _recent = await _ctx_mgr.get_conversation_history(session_id, max_turns=2)
            for _m in reversed(_recent):
                if not _last_assistant_snippet and _m.get("role") == "assistant":
                    _last_assistant_snippet = (_m.get("content") or "")[:400]
                elif not _last_user_snippet and _m.get("role") == "user":
                    if (_m.get("content") or "").strip() != query.strip():
                        _last_user_snippet = (_m.get("content") or "")[:200]
                if _last_assistant_snippet and _last_user_snippet:
                    break
        except Exception as _ce:
            logger.debug("Intent context fetch skipped: %s", _ce)

    if True:
        try:
            # Gather available strategy names for the LLM to recognise
            from strategy_engine.strategy_registry import get_registry as _ireg
            _ireg_inst = _ireg()
            _strat_list = [s.name for s in _ireg_inst.get_all_active()] if _ireg_inst.is_initialized else []
            _strat_hint = ", ".join(_strat_list) if _strat_list else "none configured yet"

            _ctx_block = ""
            if _last_user_snippet or _last_assistant_snippet:
                _ctx_block = (
                    "Recent conversation (use to resolve pronouns/contextual queries):\n"
                    + (f"  PREV USER: {_last_user_snippet}\n" if _last_user_snippet else "")
                    + (f"  PREV ASSISTANT: {_last_assistant_snippet}\n" if _last_assistant_snippet else "")
                    + "\n"
                )

            _intent_prompt = (
                "You are a parser for a trading co-planner. Extract structured intent from the trader's message.\n\n"
                f"{_ctx_block}"
                f"Configured strategy names: {_strat_hint}\n\n"
                "Return ONLY valid JSON (no explanation, no markdown):\n"
                "{\n"
                '  "symbol": "<NSE ticker if mentioned, e.g. RELIANCE, HDFCBANK, INFY — null if none>",\n'
                '  "intent_type": "<one of: setup_analysis, price_query, portfolio_review, behavioral, watchlist_scan, general>",\n'
                '  "timeframe": "<e.g. 5m, 15m, 1h, 1d — default 15m>",\n'
                '  "direction": "<long, short, or null>"\n'
                "}\n\n"
                "Rules:\n"
                "- symbol must be an NSE ticker like RELIANCE, HDFCBANK, TCS, NIFTY — NOT a word like 'strategies', 'market', 'portfolio'\n"
                "- If the user mentions a company name, return its NSE ticker (e.g. 'Reliance'→RELIANCE, 'hdfc bank'→HDFCBANK, 'Angel One'→ANGELONE, 'angel one'→ANGELONE, 'sbi'→SBIN, 'bajaj finance'→BAJFINANCE, 'kotak'→KOTAKBANK)\n"
                "- If no specific stock is mentioned, return null for symbol\n"
                "- setup_analysis: user asks if ONE specific stock is ready/setting up for a trade\n"
                "- price_query: user asks for price, LTP, change of a specific stock\n"
                "- portfolio_review: user asks about their trades, P&L, performance\n"
                "- behavioral: user asks about their patterns, discipline, psychology\n"
                "- watchlist_scan: user asks about ALL or ANY stocks in their watchlist — phrases like\n"
                "  'any breakout in watchlist', 'any stocks', 'scan all', 'which stocks', 'any setup forming',\n"
                "  'any signals', 'top pick', 'which ones are ready' — return null for symbol\n"
                "- general: market overview, strategy explanation, asking about strategy templates/parameters,\n"
                "  follow-up questions ('which EMA?', 'what timeframe?', 'show me the template', 'yes give it')\n\n"
                "CONTEXT RESOLUTION (CRITICAL):\n"
                "- If the message is short/contextual ('yes', 'show me', 'give me the template', 'what about it',\n"
                "  'which EMA', 'which timeframe', 'and the stop'), DO NOT switch symbol or context — \n"
                "  use the PREV ASSISTANT message above to understand what the user is referring to.\n"
                "- If PREV ASSISTANT discussed TCS + EMA Pullback and user says 'give me the template',\n"
                "  the symbol stays TCS (or null if they're asking about the strategy in general),\n"
                "  intent_type becomes 'general' (asking about strategy details, not a new setup).\n"
                "- NEVER hallucinate a symbol the user didn't mention and the prior context didn't reference.\n"
            )
            _intent_resp = await llm_client.chat(
                messages=[
                    _IntentMsg(role="system", content=_intent_prompt),
                    _IntentMsg(role="user", content=query),
                ],
                temperature=0.0,
                max_tokens=80,
            )
            _intent_raw = _intent_resp.content.strip()
            # Strip markdown fences if present
            if _intent_raw.startswith("```"):
                _intent_raw = _intent_raw.split("```")[1]
                if _intent_raw.startswith("json"):
                    _intent_raw = _intent_raw[4:]
            _intent_data = _json.loads(_intent_raw)
            _sym = _intent_data.get("symbol")
            if _sym and isinstance(_sym, str) and len(_sym) >= 2:
                effective_symbol = _sym.upper().strip()
            intent_type = _intent_data.get("intent_type", "general")
            intent_timeframe = _intent_data.get("timeframe") or "15m"
            intent_direction = _intent_data.get("direction")
            # Watchlist-scan: override any symbol that was pre-set by the frontend
            # (active symbol from WatchlistPanel) — the user is asking about ALL stocks,
            # not just the one they last clicked.
            if intent_type == "watchlist_scan":
                effective_symbol = None
            logger.info(
                "co-plan intent: symbol=%s type=%s tf=%s dir=%s (query=%r)",
                effective_symbol, intent_type, intent_timeframe, intent_direction, query,
            )
        except Exception as _ie:
            logger.warning("Intent extraction LLM failed (%s) — using regex fallback", type(_ie).__name__)
            # Regex fallback: first non-stop-word all-caps token in the query
            import re as _re
            _STOP = {
                "I","MY","DO","YOU","THE","OF","IN","AT","FOR","TO","AND","OR","IS","IT",
                "A","AN","BE","ME","WE","US","ON","UP","AS","BY","IF","SO","NO","ANY","ALL","ITS",
                "HI","HELLO","HEY","OK","OKAY","YES","YEP","SURE","THANKS","THANK","PLEASE","HELP","NEED","WANT",
                "WHAT","HOW","WHY","WHEN","WHERE","WHO","WHICH","KNOW","TELL","SHOW","GIVE","FIND","CHECK",
                "GET","BUY","CAN","HAVE","SHOULD","WOULD","COULD","WILL","MAY","MIGHT","DOES","DID","WAS","ARE","HAD",
                "LOOK","LOOKS","THINK","FEEL","SEEMS","SEEM","TAKE","MAKE","COME","KEEP","HOLD","MOVE",
                "WAIT","LIKE","JUST","ALSO","EVEN","BACK","NOW","NEW","OLD","NEXT","LAST","BEST","MORE","LESS",
                "VERY","MUCH","MANY","MOST","SOME","EACH","BOTH","THEN","THAN","THEM","THEY","THEIR","THERE","THESE",
                "INTO","OVER","AFTER","BEFORE","STILL","SINCE","BEEN","BEING","HAVING","DOING","GOING","SEEING",
                "LTP","NSE","BSE","NIFTY","SENSEX","PE","CE","SL","TP","RR","EMA","SMA","RSI","MACD","ATR","VWAP",
                "OI","IV","STOCK","SHARE","PRICE","TRADE","MARKET","TODAY","ABOUT","THAT","WITH","FROM","THIS","LEVEL",
                "SETUP","PLAN","GOOD","NEAR","HIGH","LOW","OPEN","CLOSE","LONG","SHORT","CALL","PUT","GAIN","LOSS",
                "STOP","ENTRY","EXIT","RISK","SIZE","POSITION","TREND","REGIME","SESSION","DAY","WEEK","MONTH",
                "STRATEGY","STRATEGIES","ANALYSIS","SUPPORT","RESISTANCE","TARGET","CANDLE","CHART","VOLUME",
                "BREAKOUT","PULLBACK","REVERSAL","MOMENTUM","SQUEEZE","RANGE","SWING","SETTING","LOOKING",
                "CURRENT","CURRENTLY","RIGHT","READY","ACTIVE","RECENT","LATEST","USING","BASED","SIGNAL",
                "SIGNALS","ALERTS","ALERT","SYSTEM","PROFILE","OVERVIEW","REVIEW","UPDATE","REPORT","STATUS",
                "TRADES","TRADE","PROFIT","LOSSES","GAINS","SECTOR","INDEX","INDICES","WATCH","WATCHLIST",
            }
            _fb_candidates = [w for w in _re.findall(r'\b([A-Z]{2,12})\b', query.upper()) if w not in _STOP]
            if _fb_candidates:
                effective_symbol = _fb_candidates[0]  # First non-stop word = most likely ticker
            # Classify intent from keywords
            _ql = query.lower()
            if any(w in _ql for w in _WATCHLIST_TRIGGERS) and not _fb_candidates:
                intent_type = "watchlist_scan"
                effective_symbol = None
            elif any(w in _ql for w in ["price","ltp","trading at","current price","how much"]):
                intent_type = "price_query"
            elif any(w in _ql for w in ["setup","setting up","strategy","ready","breakout","pullback","pattern"]):
                intent_type = "setup_analysis"
            elif any(w in _ql for w in ["p&l","pnl","trades","portfolio","performance","profit","loss","how am i doing"]):
                intent_type = "portfolio_review"
            elif any(w in _ql for w in ["discipline","psychology","overtrading","emotion","fear","greed","pattern"]):
                intent_type = "behavioral"
            logger.info("Regex fallback: symbol=%s intent=%s", effective_symbol, intent_type)

    # Deterministic override (router-on-Llama is the #1 product risk): "where's the money move / what's
    # moving / which stock" style queries MUST trigger the brain watchlist scan, even if the LLM intent
    # parse called it 'general'. This is exactly the misroute that gave the generic non-answer.
    _ql_ovr = (query or "").lower()
    if intent_type in ("general", "setup_analysis", "price_query") and any(p in _ql_ovr for p in (
        "money move", "big money", "big move", "where is the move", "where's the move", "where is the action",
        "what is moving", "what's moving", "which stock", "which symbol", "anything building", "any setup",
        "any setups", "scan", "where is money building", "what is building", "what's building", "money is building")):
        intent_type = "watchlist_scan"
        logger.info("intent override -> watchlist_scan (money-move scan phrasing)")

    logger.info("co-plan effective_symbol=%s intent=%s (from query=%r)", effective_symbol, intent_type, query)

    # ── Watchlist scan (multi-symbol) ────────────────────────────────
    # When intent is watchlist_scan, run a lightweight evaluation across all
    # watchlist stocks and build a summary block. The per-symbol single-stock
    # intelligence path is skipped (effective_symbol stays None).
    watchlist_scan_block = ""
    _scan_failed = False
    _scan_diag = ""
    if intent_type == "watchlist_scan":
        try:
            from automation.setup_scanner import get_scanner_loop

            # Collect symbols: scanner watchlist first, then watchlist DB
            _ws_symbols: list[str] = []
            try:
                _scanner = get_scanner_loop()
                _ws_symbols = list(_scanner.symbols)[:12]
            except Exception:
                pass
            if not _ws_symbols:
                # the user's ACTUAL Smart Picks (what the sidebar shows), top by match_score — the same query
                # the watchlist API uses. This is the list that was missing → empty scan → hallucination.
                _sync_db_ws = SessionLocal()
                try:
                    from core.models import Watchlist as _WL, WatchlistItem as _WLI
                    _smart = _sync_db_ws.query(_WL).filter(
                        _WL.trader_id == trader_id, _WL.name == "Smart Picks",
                        _WL.watchlist_type == "strategy", _WL.is_active == True,
                    ).first()
                    if _smart:
                        _rows = (_sync_db_ws.query(_WLI.symbol)
                                 .filter(_WLI.watchlist_id == _smart.id, _WLI.status.in_(["matched", "active"]))
                                 .order_by(_WLI.match_score.desc().nullslast()).limit(12).all())
                        _ws_symbols = [r[0] for r in _rows]
                    if not _ws_symbols:
                        _ws_symbols = [r[0] for r in _sync_db_ws.query(_WLI.symbol).distinct().limit(12).all()]
                except Exception:
                    logger.warning("co-plan smart-picks query failed", exc_info=True)
                finally:
                    _sync_db_ws.close()
            if not _ws_symbols:
                # FINAL fallback — the scanner watchlist & DB are often empty; use the FNO universe so the
                # scan has REAL symbols to read (the empty list is exactly what made the scan fail → hallucinate).
                try:
                    from api.routes.watchlist import _FNO_NIFTY50
                    _ws_symbols = list(_FNO_NIFTY50)
                except Exception:
                    pass

            if _ws_symbols:
                # BRAIN-backed scan: the SAME eye every face uses (state / money-move stage / bias),
                # building moves ranked first. Bounded (~10s/symbol) + cached (10 min) for chat latency;
                # the fully-live version is the deferred proactive precompute.
                from intelligence.watchlist_scan import scan_watchlist, render_scan
                import asyncio as _aio_ws
                # run in a worker thread — the eye calls asyncio.run() internally, which RAISES inside
                # FastAPI's running event loop (this is why every read failed in the app but not standalone)
                _scan = await _aio_ws.to_thread(scan_watchlist, _ws_symbols, 4)
                _scan_ok = sum(1 for r in _scan if r.get("ok"))
                if _scan_ok:
                    watchlist_scan_block = render_scan(_scan)
                else:
                    _scan_failed = True   # NO real data → must NOT let the LLM invent (the hallucination bug)
                    _scan_diag = (f"tried {len(_scan)}/{len(_ws_symbols)} "
                                  f"[{', '.join(r.get('symbol','?') for r in _scan)}], eye read 0 OK")
                logger.info("Watchlist brain-scan: %d symbols, %d read OK, %d building",
                            len(_ws_symbols), _scan_ok, sum(1 for r in _scan if r.get("building")))
            else:
                _scan_failed = True
                _scan_diag = "no symbols found (scanner empty + Smart Picks empty + FNO fallback failed)"
        except Exception as _wse:
            logger.warning("Watchlist scan failed: %s", _wse, exc_info=True)
            _scan_failed = True
            _scan_diag = f"exception: {type(_wse).__name__}: {str(_wse)[:100]}"

    # 1. Assemble trader context — use a sync session to avoid async/sync mismatch
    context_block = "[Trader context unavailable]"
    _sync_db = SessionLocal()
    try:
        assembler = CoPlannerContextAssembler(_sync_db)
        snapshot = assembler.assemble_sync(trader_id, symbol=effective_symbol)
        context_block = CoPlannerContextAssembler.format_for_prompt(snapshot, symbol=effective_symbol)

        # Append last-week mistake patterns (SQL aggregation, no LLM)
        try:
            from analytics.mistake_summarizer import LastWeekMistakeSummarizer
            _mistake_summary = LastWeekMistakeSummarizer(_sync_db).summarize(trader_id)
            if _mistake_summary:
                context_block += "\n\n" + _mistake_summary
        except Exception as _me:
            logger.debug("Mistake summarizer skipped: %s", _me)
    except Exception as exc:
        logger.warning("CoPlannerContext assembly failed: %s", exc)
    finally:
        _sync_db.close()

    # 2. Fetch live market data via Upstox (with source tagging)
    live_market_block = ""
    ltp_source = None
    if effective_symbol:
        try:
            from ingestion.market_data_service import get_market_data_service, init_market_data_service
            _mds = get_market_data_service()
            if not _mds:
                # Service not yet initialized at startup — init now
                logger.info("MarketDataService not initialized — initializing on-demand")
                _mds = init_market_data_service()
            if not _mds:
                logger.warning("MarketDataService unavailable (no broker credentials configured)")
            else:
                _quote = _mds.get_market_quote(effective_symbol)
                # If full quote returned no LTP, try lightweight LTP call as fallback
                if not (_quote and _quote.get("ltp")):
                    logger.info("Full quote returned no LTP for %s, trying get_ltp fallback", effective_symbol)
                    _fallback_ltp = _mds.get_ltp(effective_symbol)
                    if _fallback_ltp:
                        _quote = {"ltp": _fallback_ltp}
                if _quote and _quote.get("ltp"):
                    ltp = _quote["ltp"]
                    # prev_close = previous session's close (for day-change calc)
                    # close = today's closing price (same as ltp after close)
                    prev_close = _quote.get("prev_close")
                    ltp_source = "Upstox" if _mds.has_live else "Angel One"
                    change_str = ""
                    if prev_close and prev_close > 0 and abs(ltp - prev_close) > 0.01:
                        chg = ltp - prev_close
                        chg_pct = chg / prev_close * 100
                        direction = "▲" if chg >= 0 else "▼"
                        change_str = f" {direction} {abs(chg):,.2f} ({chg_pct:+.2f}%)"
                    # Detect if market is currently open (NSE: Mon-Fri 09:15-15:30 IST)
                    import pytz as _pytz
                    _ist = _pytz.timezone("Asia/Kolkata")
                    _now_ist = datetime.now(_ist)
                    _is_market_open = (
                        _now_ist.weekday() < 5  # Mon-Fri
                        and (9, 15) <= (_now_ist.hour, _now_ist.minute) <= (15, 30)
                    )
                    _data_label = "Live LTP" if _is_market_open else "Last Traded Price (market closed)"

                    parts = [f"## Market Data — {effective_symbol} (source: {ltp_source})"]
                    parts.append(f"{_data_label}: ₹{ltp:,.2f}{change_str}")
                    if _quote.get("open"):
                        parts.append(f"Open: ₹{_quote['open']:,.2f}")
                    if _quote.get("high") and _quote.get("low"):
                        parts.append(f"Day Range: ₹{_quote['low']:,.2f} – ₹{_quote['high']:,.2f}")
                    if _quote.get("volume"):
                        parts.append(f"Volume: {_quote['volume']:,}")
                    if prev_close:
                        parts.append(f"Prev Close: ₹{prev_close:,.2f}")
                    if not _is_market_open:
                        parts.append("Note: Market is currently closed. This is the last traded price from the previous session.")
                    live_market_block = "\n".join(parts)
                    logger.info(
                        "Live quote %s: ₹%s (source=%s, market_open=%s)",
                        effective_symbol, ltp, ltp_source, _is_market_open,
                    )
                else:
                    logger.warning(
                        "No LTP returned for %s — quote=%r (has_live=%s)",
                        effective_symbol, _quote, _mds.has_live,
                    )
        except Exception as exc:
            logger.error("Live quote fetch for %s failed: %s", effective_symbol, exc, exc_info=True)

    # 3. Process attachments
    attachment_context = ""
    attachment_meta: list[dict] = []
    if files:
        attachment_context, attachment_meta = await _process_attachments(
            files, llm_client, effective_symbol,
        )

    # 4. Symbol intelligence — MIGRATED to the eye (cognition_service). The old GEN-1
    #    RealtimeIntelligenceService `symbol_intel` block was REMOVED (App Reality Audit #1);
    #    the active-symbol read now comes solely from the eye_block below.
    symbol_intel = ""

    # 4b. TRADE REVIEW — the user's OWN recent closed trades + the eye's read at entry (grounded
    #     mistake-finding). Fires when they ask about their trades/positions/mistakes.
    trades_block = ""
    _ql_tr = (query or "").lower()
    _wants_trades = intent_type in ("behavioral", "portfolio_review") or any(w in _ql_tr for w in (
        "my trade", "my position", "my closed", "mistake", "analyze my", "review my", "what did i do",
        "my pnl", "my p&l", "my loss", "closed position", "my trades", "i traded", "today's trade",
        "todays trade", "my call", "my put", "current position"))
    if _wants_trades:
        try:
            from core.models import Trade
            _tdb = SessionLocal()
            try:
                _trades = (_tdb.query(Trade)
                           .filter(Trade.trader_id == trader_id)
                           .order_by(Trade.entry_time.desc()).limit(6).all())
            finally:
                _tdb.close()
            if _trades:
                import asyncio as _aio_tr
                from intelligence.trade_review import review_for_coplan
                # max_eye=4: deep eye-review the 4 most recent trades (entry+exit reads ≈ 20s each,
                # ~80s for 4 — under the frontend's 120s timeout). Older trades get the P&L line.
                # (Scaling past this needs concurrent eye reads / Layer-2 incremental narration.)
                trades_block = await _aio_tr.to_thread(review_for_coplan, _trades, 4)
            else:
                trades_block = "## Your Recent Trades\nNo trades found for your account — nothing to analyze yet."
        except Exception as _te:
            logger.warning("co-plan trade review failed: %s", _te, exc_info=True)

    # ── BRAIN: the EYE's live read (cognition_service) — the SAME move/state/bias/money-move-stage the rest
    #    of PTIS uses. This grounds the co-planner in the actual brain instead of generic LLM blather.
    #    ~10s (eye narration), so gated on a trading-intent keyword. Honest if the eye can't read.
    eye_block = ""
    _eye_verdict = ""    # FAITHFULNESS GUARD: the eye's deterministic read, prepended VERBATIM to the answer
    # The eye is now the SOLE symbol intelligence (GEN-1 symbol_intel removed) → fire it for ANY analytical
    # query on a symbol. Skip only: watchlist_scan (the scan already reads it) and a pure price_query
    # (just wants the number — the eye's ~10s isn't worth it).
    _needs_eye = bool(effective_symbol) and intent_type not in ("watchlist_scan", "price_query")
    if _needs_eye:
        try:
            from intelligence import cognition_service as _cog
            import asyncio as _aio_eye
            # worker thread: the eye calls asyncio.run() internally → raises inside FastAPI's running loop
            _r = await _aio_eye.to_thread(_cog.read, effective_symbol, "15m")
            if getattr(_r, "ok", False):
                # deterministic verdict — shown verbatim so the LLM's prose can never silently contradict it
                _eye_verdict = (f"**{effective_symbol} — the eye reads: {_r.state}"
                                + (f"/{_r.substate}" if _r.substate else "")
                                + (f" · {_r.bias}" if _r.bias else "")
                                + (f" · stage {_r.stage}" if _r.stage else "") + "**")
                _el = [f"## Eye Read — {effective_symbol} (PTIS's OWN analytical read of this symbol — make this the BACKBONE of your answer; cite its state / money-move stage / bias)"]
                for _lab, _val in (
                    ("State", _r.state and (_r.state + (f" / {_r.substate}" if _r.substate else ""))),
                    ("Money-move stage", _r.stage and (_r.stage + (f" — {_r.stage_note}" if _r.stage_note else ""))),
                    ("Move story", _r.move_story), ("Directional bias", _r.bias),
                    ("Why (mechanism)", _r.mechanism), ("Invalidation", _r.invalidation), ("Read", _r.read),
                ):
                    if _val:
                        _el.append(f"{_lab + ': ' if _lab else ''}{_val}")
                eye_block = "\n".join(_el)
            elif getattr(_r, "note", ""):
                eye_block = (f"## Eye Read — {effective_symbol}\nThe eye could not read this symbol right now "
                             f"({_r.note}). Be honest about that — do NOT invent a read.")
        except Exception as _ee:
            logger.warning("co-plan eye read failed for %s: %s", effective_symbol, _ee)

    # Compute current IST time and NSE session status (always injected into context)
    try:
        import pytz as _pytz_cp
        _ist_cp = _pytz_cp.timezone("Asia/Kolkata")
        _now_cp = datetime.now(_ist_cp)
    except Exception:
        from datetime import timezone as _tz
        _now_cp = datetime.now(_tz.utc)
    _day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    _t_cp = (_now_cp.hour, _now_cp.minute)
    _is_weekday_cp = _now_cp.weekday() < 5
    _is_open_cp = _is_weekday_cp and (9, 15) <= _t_cp <= (15, 30)
    if not _is_weekday_cp:
        _session_label = "CLOSED (weekend)"
    elif _t_cp < (9, 15):
        _mins_to_open = (9 * 60 + 15) - (_now_cp.hour * 60 + _now_cp.minute)
        _session_label = f"PRE-MARKET — opens in {_mins_to_open} min"
    elif _t_cp > (15, 30):
        _session_label = "CLOSED — session ended for today"
    else:
        _mins_left = (15 * 60 + 30) - (_now_cp.hour * 60 + _now_cp.minute)
        _session_label = f"OPEN — {_mins_left} min remaining in session"
    _session_block = (
        f"## Current Session\n"
        f"Time (IST): {_now_cp.strftime('%I:%M %p')} {_day_names[_now_cp.weekday()]}, "
        f"{_now_cp.strftime('%d %b %Y')}\n"
        f"NSE Market: {_session_label}"
    )

    # 5. Build the system context block (goes to LLM as silent input, NOT shown to user)
    #    Sections: capabilities -> session -> trader snapshot -> live data -> strategy intel -> attachments
    system_context_parts = [
        "## SYSTEM CAPABILITIES",
        "You are a live trading co-planner with access to:",
        "- Real-time market data via Upstox (LTP, OHLC, volume)",
        "- Historical OHLCV via Angel One (15-min candles)",
        "- Trader's full history, P&L, strategy performance, and behavioral data",
        "",
        "CRITICAL RULES:",
        "- Use the data below silently. Do NOT repeat headers or raw context in your reply.",
        "- Respond like a knowledgeable human co-trader — clean, direct, decision-focused.",
        "- If live data is present, use it. Never say you lack real-time access.",
        "- For simple price queries: answer in 1-2 sentences, then add brief insight if relevant.",
        "- For trade planning: give setup conditions, risk levels, strategy fit from the data.",
        "- If an 'Eye Read' or 'Watchlist Money-Move Scan' block is present, it is PTIS's OWN analytical",
        "  read (state / money-move stage / bias / invalidation) — make it the BACKBONE of your answer and",
        "  cite it. Do NOT analyze or name any symbol that is NOT in the provided data blocks.",
        "  Give the trader concrete guidance: which strategy fits NOW, what to watch for entry.",
        "- NEVER say 'when the market opens' if the session block says the market is OPEN.",
        "- NEVER INVENT STATISTICS. If the trader context shows 0 trades or no symbol history,",
        "  you MUST say 'no historical data yet' — do NOT make up win rates, R multiples,",
        "  trade sizes, or strategy performance. Invented numbers in a trading tool cause real harm.",
        "",
        "---",
        _session_block,
        "---",
        context_block,
    ]
    if live_market_block:
        system_context_parts += ["", "---", live_market_block]
    elif effective_symbol:
        # No live data — tell LLM not to guess prices but still give historical analysis
        system_context_parts += [
            "", "---",
            f"## Market Data — {effective_symbol}",
            "STATUS: Live price feed unavailable (market closed or broker disconnected).",
            "HARD RULE: Do NOT write ANY price number (₹XXX or $XXX) for this symbol.",
            "Do NOT say 'last traded price was ₹...' — you do not have this data.",
            "Instead: acknowledge price is unavailable in ONE sentence, then give a",
            "COMPLETE analysis using only the Trader Context above:",
            "  - Historical win rate and avg R on this symbol",
            "  - Whether this symbol fits the trader's strategy profile",
            "  - Risk/sizing guidance based on past trades",
            "  - Whether the trader should consider this setup when market opens",
        ]
    # Inject configured strategy templates with FULL parameter details so LLM
    # can answer specific questions like "which EMA?", "what timeframe?", "what RSI range?"
    try:
        from strategy_engine.strategy_registry import get_registry as _get_reg
        _reg = _get_reg()
        _strat_templates = list(_reg.get_all_active()) if _reg else []
        if _strat_templates:
            _details_blocks = [_format_strategy_details(t) for t in _strat_templates]
            system_context_parts += [
                "", "---",
                "## Configured Strategy Templates (with actual parameters)",
                "These are the trader's configured strategies WITH their real parameters.",
                "When asked specific questions like 'which EMA?', 'what timeframe?', 'what RSI?', 'what stop?',",
                "USE THESE EXACT VALUES — do NOT say 'I don't have a specific EMA configured'.",
                "",
                *_details_blocks,
            ]
    except Exception as _se:
        logger.debug("Strategy details injection failed: %s", _se)
    if watchlist_scan_block:
        system_context_parts += ["", "---", watchlist_scan_block]
    if eye_block:
        system_context_parts += ["", "---", eye_block]
    if trades_block:
        system_context_parts += ["", "---", trades_block]
    if symbol_intel:
        system_context_parts += ["", "---", symbol_intel]
    if attachment_context:
        system_context_parts += ["", "---", "## Chart / File Analysis", attachment_context]

    full_system_context = "\n".join(system_context_parts)

    # ── HALLUCINATION GUARD ── a watchlist scan was asked for but produced NO real data → return an honest
    # message and DO NOT call the LLM. Without this, the LLM invents stocks/levels not in the data — the
    # "INFY breakout at 1500" failure (INFY was at 1016, falling; INFY wasn't even in the watchlist).
    if intent_type == "watchlist_scan" and _scan_failed:
        logger.warning("co-plan: watchlist scan produced no data (%s) — honest no-data reply (no LLM)", _scan_diag)
        return ChatMessageResponse(
            session_id=session_id or "co_planner",
            message=("I can't read your watchlist right now, so I won't guess or name stocks I haven't actually "
                     "read.\n\nTo fix: make sure your watchlist is populated and the candle feed is fresh "
                     "(`python -m marketdata.backfill --update`, check the broker token), then ask again."
                     + (f"\n\n_diagnostic: {_scan_diag}_" if _scan_diag else "")),
            mode="co_planner", insights=[], action_items=[], follow_up_questions=[],
            suggested_focus=None, confidence=0.0,
        )

    # 5. Call LLM directly with system context + clean user message.
    # We do this by calling the LLM directly rather than going through CoachingAgent's
    # context builder (which only adds knowledge sections conditionally).
    from agents.llm_client import LLMMessage as _LLMMsg

    # Build intent-specific response instruction
    _intent_instructions = {
        "watchlist_scan": (
            "The trader is asking about their ENTIRE WATCHLIST — not one stock.\n"
            "Check the 'NSE Market' line in Current Session:\n"
            "- If market is OPEN: use the Watchlist Scan block to give a ranked GO/WATCH/SKIP summary "
            "(name the stock, strategy, key reason). 5-10 lines.\n"
            "- If market is CLOSED or WEEKEND: market is closed — tell them clearly, then give a "
            "'Prepare for Tomorrow' response: which strategies to watch for, what price levels to set alerts on, "
            "what regime conditions to check at open. Use their strategy profile and trader context. "
            "Sound like their experienced self preparing the night before. 5-8 lines."
        ),
        "setup_analysis": (
            f"The trader is asking about the situation/setup for {effective_symbol or 'a stock'}.\n"
            "Ground your answer ENTIRELY in the 'Eye Read' block above — that is PTIS's own cognitive read "
            "(state / move-story / why / bias / invalidation). NEVER invent anything not in it.\n\n"
            "Read it the way a sharp discretionary trader reads a chart — a STORY, not a mechanical checklist:\n"
            "1) THE READ — what state the move is in (balance / building / alive / exhausted) and WHERE it is "
            "in its life (early, mature, extended; pullbacks; how it's held VWAP), in plain language.\n"
            "2) WHY — the mechanism: what the market is actually doing and why (the eye's 'Why' field).\n"
            "3) THE THREAT — the observable that would prove the read WRONG (the eye's invalidation). This is "
            "your RISK and your stop: a STRUCTURAL level (where you're wrong), never a fixed % or a made-up number.\n"
            "4) THE CALL — the eye's bias: go-with / wait-for-acceptance / stand-aside / already-late — and what "
            "you'd wait to SEE (acceptance, a reclaim, a real break) before acting.\n\n"
            "HARD RULES — this IS the philosophy, do NOT break them:\n"
            "- NEVER invent price targets (T1/T2), R:R ratios, or a fixed SL %. A real move is HELD while it's "
            "alive and exited when the STORY CHANGES (the invalidation) — let the tail run; there is no capped target.\n"
            "- The ONLY risk level is the eye's structural invalidation. If you state a price, it must be that.\n"
            "- Do NOT produce a 'Conditions X/Y met' checklist or name strategy templates — that's the old "
            "mechanical engine. Read the MOVE, not a rulebook.\n"
            "- If the eye says EXHAUSTED / STORY_CHANGED / already-late, say plainly the move is late or done — "
            "do NOT manufacture a fresh entry just because price popped.\n"
            "- 3-6 short sentences, trader-to-trader ('you'/'we'). No invented expiry times, no sub-headers."
        ),
        "price_query": (
            f"The trader is asking for the CURRENT PRICE of {effective_symbol or 'a stock'}.\n"
            "Answer in 1-2 sentences: state the LTP from market data, add one brief context line "
            "(e.g., where it is vs VWAP or ORB). Do not elaborate unless asked."
        ),
        "portfolio_review": (
            "The trader is asking about their OWN TRADES / P&L / positions / mistakes.\n"
            "Ground EVERYTHING in the 'Your Recent Trades' block (exact facts + the eye's read of the "
            "UNDERLYING at entry/exit) and the Trader Context — never invent a trade, price, or number.\n"
            "BE BETTER THAN A GENERIC P&L ANALYZER: a generic tool only knows prices + textbook ('theta is "
            "brutal', 'buy high sell low'). YOU know what the MARKET was actually doing — use the eye reads as "
            "the BACKBONE of every point. For each meaningful trade, say in plain language: WHAT the underlying "
            "was doing at entry (the eye's state/mechanism), whether the option bet was WITH or AGAINST that "
            "move, the structural level that mattered (the eye's exit-trigger), and the EXIT timing (cut while "
            "alive / held past the turn / exited into exhaustion). THEN synthesize the one repeated BEHAVIORAL "
            "mistake across the trades (e.g. 'you keep buying options AGAINST the prevailing move / into "
            "balance, then cutting fast' — grounded in the eye reads, not a guess).\n"
            "When a 'MEMORY:' line is present, CITE IT — it's the grounded historical edge a generic tool "
            "CANNOT produce ('setups like this resolved X% vs Y% base, n=Z'). Use it to judge whether the odds "
            "actually favored the bet, beyond just 'it lost'.\n"
            "You MAY add real options context (expiry-day theta, IV) but ONLY alongside the market read, never "
            "instead of it. Direct, specific, non-judgmental, 'you'/'we'. End with ONE concrete change.\n"
            "If a trade's eye read says 'data not in the lake / can't read yet', say so honestly (the feed is "
            "refreshing) — do NOT invent what the eye saw."
        ),
        "behavioral": (
            "The trader is asking about their TRADING PSYCHOLOGY / PATTERNS / DISCIPLINE / their own trades.\n"
            "Ground EVERYTHING in the 'Your Recent Trades' block + Trader Context behavioral data — never "
            "invent. Be direct, honest, non-judgmental. Name the real repeated mistake (e.g. option-buying "
            "into the wrong side, chasing, revenge trades, clustered losses) and give ONE concrete change."
        ),
        "general": (
            "The trader has a GENERAL question about the market, strategies, or concepts.\n"
            "Answer concisely using available context. Reference live data if relevant."
        ),
    }
    _intent_instr = _intent_instructions.get(intent_type, _intent_instructions["general"])

    _system_prompt = (
        "You are the live trading co-planner in a Personal Trading Intelligence System (PTIS).\n"
        "You have real-time market data (Upstox LTP, OHLC) and historical candle analysis (Angel One).\n"
        "All relevant data has already been fetched and is injected into the context below.\n\n"
        f"## This Query's Intent: {intent_type.upper()}\n"
        f"{_intent_instr}\n\n"
        "CORE RULES:\n"
        "- NEVER CONTRADICT the Eye Read's state/bias. If the eye says BALANCE, do NOT call it a breakout or "
        "trend; if it says EXHAUSTED, do NOT call it fresh momentum; if the bias is wait-for-acceptance / "
        "stand-aside, do NOT say 'go long now'. The eye is the ground truth — narrate around it, never over it.\n"
        "- NEVER say you lack real-time access — you have it. Use it.\n"
        "- NEVER say you cannot access trade history — it is in 'Trader Context'.\n"
        "- NEVER repeat back raw data headers or dump sections verbatim.\n"
        "- NEVER invent statistics (win %, avg R, trade counts). If context shows 0 trades, say so.\n"
        "- If Live Intelligence block is present: it IS the market analysis. Use regime, trend, VWAP, strategy readiness from it.\n"
        "- If market is OPEN (see Current Session): give actionable NOW guidance.\n"
        "- For setup_analysis: follow the EXACT output format in the intent section above — structured first, voice after.\n"
        "- For all other intents: be concise (3-6 sentences). Write like the trader's experienced self advising their current self.\n\n"
        "---\n"
        f"{full_system_context}"
    )
    # Inject last N conversation turns for true multi-turn memory.
    # Fetch from DB using the async session — trim to last 5 turns (10 messages)
    # so we don't blow the token budget. Each turn = 1 user + 1 assistant message.
    _history_msgs: list[_LLMMsg] = []
    if session_id:
        try:
            _hist_mgr = ConversationManager(db)
            _prior = await _hist_mgr.get_conversation_history(session_id, max_turns=10)
            # Drop the last user message if it duplicates the current query
            # (can happen if the message was persisted before this call completes)
            if _prior and _prior[-1]["role"] == "user" and _prior[-1]["content"].strip() == query.strip():
                _prior = _prior[:-1]
            _history_msgs = [
                _LLMMsg(role=m["role"], content=m["content"])
                for m in _prior
                if m["role"] in ("user", "assistant")
            ]
            if _history_msgs:
                logger.debug("Co-plan: injecting %d prior turns into context", len(_history_msgs) // 2)
        except Exception as _he:
            logger.debug("History fetch skipped (non-fatal): %s", _he)

    _messages = [
        _LLMMsg(role="system", content=_system_prompt),
        *_history_msgs,
        _LLMMsg(role="user", content=query),
    ]

    logger.info("Co-plan symbol=%s live_block=%r history_turns=%d", effective_symbol, live_market_block[:120] if live_market_block else "EMPTY", len(_history_msgs) // 2)
    logger.debug("Co-plan system prompt (first 500): %s", _system_prompt[:500])

    try:
        _raw = await llm_client.chat(messages=_messages, temperature=0.65, max_tokens=800)
        _reply = _raw.content
    except Exception as exc:
        _exc_str = str(exc)
        logger.error("Co-plan LLM call failed: %s", exc)
        _prov = getattr(llm_client, "provider", "?")
        _mdl = getattr(llm_client, "model", "?")
        if "429" in _exc_str or "quota" in _exc_str.lower() or "RESOURCE_EXHAUSTED" in _exc_str or "rate" in _exc_str.lower():
            _reply = (
                f"**AI rate limit / quota hit** — the `{_prov}` model (`{_mdl}`) is temporarily rate-limited "
                f"or over quota (429). Market data and analysis are fine; only the written reply is paused.\n\n"
                "- Wait ~30–60s and retry, or\n"
                f"- Check your `{_prov}` API limits / `LLM_API_KEY` in your env."
            )
        else:
            _reply = f"AI response failed ({_prov}/{_mdl}): {type(exc).__name__}. Try again in a moment."

    # 6. Conversation persistence
    # The intelligence evaluation may have left the db transaction in a failed state
    # (r_estimator or trader_profile queries can error). Roll back before new queries.
    try:
        await db.rollback()
    except Exception:
        pass

    conversation_mgr = ConversationManager(db)
    _fallback_session_id = session_id or f"coplan-{trader_id[:8]}"
    try:
        conversation = await conversation_mgr.get_or_create_session(
            trader_id=trader_id, session_id=session_id,
        )
        await conversation_mgr.add_user_message(
            session_id=conversation.session_id, content=query,
            metadata={"type": "co_plan", "symbol": effective_symbol, "attachments": attachment_meta},
        )
        await conversation_mgr.add_assistant_message(
            session_id=conversation.session_id, content=_reply,
            metadata={"type": "co_plan", "symbol": effective_symbol},
        )
        _fallback_session_id = conversation.session_id
    except Exception as _conv_exc:
        logger.warning("Co-plan conversation persistence failed (non-fatal): %s", _conv_exc)

    # Parse structured reply into insight cards + clean voice text.
    # Pass live ltp so the entry-distance guard can suppress un-actionable trade plans
    # (e.g., entry 2% below current price for a 15m timeframe — that's swing, not intraday).
    _intel_data: dict = {}
    _local_ltp = locals().get("ltp")
    if _local_ltp is not None:
        try:
            _intel_data["ltp"] = float(_local_ltp)
        except (TypeError, ValueError):
            pass
    _insights, _action_items, _voice_text = _build_coplan_insights(
        _reply, intent_type, intel_data=_intel_data,
    )

    # FAITHFULNESS GUARD: lead with the eye's DETERMINISTIC verdict (verbatim from the organ) so the LLM's
    # prose can never silently contradict it (it once narrated "BREAKOUT" when the eye said BALANCE). The
    # eye is the truth; the narrative elaborates around it.
    if _eye_verdict and _voice_text and not _voice_text.startswith("**"):
        _voice_text = f"{_eye_verdict}\n\n{_voice_text}"

    return ChatMessageResponse(
        session_id=_fallback_session_id,
        message=_voice_text,         # just the voice sentences
        mode="co_planner",
        insights=_insights,          # verdict card, conditions card, trade plan card
        action_items=_action_items,  # missing conditions as action items
        follow_up_questions=[],
        suggested_focus=effective_symbol,
        confidence=0.8,
    )


# ── Session Briefing ──────────────────────────────────────────────


class SessionBriefingRequestBody(BaseModel):
    symbols: list[str] = Field(
        default_factory=list,
        description="Watchlist symbols to evaluate. Empty = use scanner's current watchlist.",
    )
    session_id: Optional[str] = None


@router.post("/session-briefing")
async def session_briefing(
    request: SessionBriefingRequestBody,
    db: AsyncSession = Depends(get_db),
    llm_client: LLMClient = Depends(get_llm_client),
    trader_id: str = Depends(get_current_trader_id),
):
    """
    Prepare a full session battle plan via the WorkflowCopilot pipeline.

    Steps (all rule-based except the final narrative):
      1. Behavioral context + last-week mistake patterns  (SQL)
      2. Per-symbol condition checklist via readiness_scanner  (rule-based)
      3. Grade A/B/C by regime alignment + conditions met + R estimate  (maths)
      4. One LLM call to write the narrative battle plan  (synthesis only)

    Triggered by the "Prepare Session" button in the co-planner UI.
    """
    from agents.workflow_copilot import WorkflowCopilot
    from core.database import SessionLocal

    symbols = request.symbols

    # If no symbols provided, pull from scanner's live watchlist
    if not symbols:
        try:
            from automation.setup_scanner import get_scanner_loop
            scanner = get_scanner_loop()
            symbols = list(scanner.symbols)[:15]
        except Exception:
            symbols = []

    # WorkflowCopilot needs a sync DB session for SQL queries
    _sync_db = SessionLocal()
    try:
        copilot = WorkflowCopilot(sync_db=_sync_db, llm=llm_client)
        briefing = await copilot.prepare(trader_id=trader_id, symbols=symbols)
        return briefing.to_dict()
    except Exception as e:
        logger.error("Session briefing failed: %s", e, exc_info=True)
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "trader_id": trader_id,
            "top_opportunities": [],
            "skipped_symbols": [],
            "behavioral_summary": "",
            "mistake_patterns": "",
            "session_limit_reached": False,
            "battle_plan": f"Session briefing failed: {type(e).__name__}. Try again in a moment.",
            "plan_generated": False,
        }
    finally:
        _sync_db.close()

