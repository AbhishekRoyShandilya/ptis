"""
PTIS LLM Coaching API
======================

REST API endpoints for personalized LLM coaching with RAG.

Endpoints:
- POST /api/v1/coaching/query - Personalized coaching query
- POST /api/v1/coaching/strategy - Strategy recommendations
- GET /api/v1/coaching/context/{trader_id} - Personalized context
- POST /api/v1/knowledge/add - Add knowledge entry
- GET /api/v1/knowledge/{trader_id} - Get trader knowledge
- GET /api/v1/knowledge/summary/{trader_id} - Knowledge summary

Features:
- Real-time personalized coaching
- RAG-enhanced responses
- Knowledge base management
- Context-aware recommendations
- Performance tracking

NOTE: These endpoints require optional packages (anthropic, faiss,
sentence-transformers).  If those are not installed the router still
loads, but endpoints return 503 Service Unavailable.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel, Field

from api.auth import get_current_user
import uuid

logger = logging.getLogger(__name__)

# Create router
llm_router = APIRouter(prefix="/api/v1", tags=["LLM Coaching"], dependencies=[Depends(get_current_user)])


# ── Local enum stubs so Pydantic models work even without optional deps ──

class QueryType(str, Enum):
    ENTRY_DECISION = "entry_decision"
    EXIT_DECISION = "exit_decision"
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGY_REVIEW = "strategy_review"
    PSYCHOLOGY_COACHING = "psychology_coaching"
    POST_TRADE_REVIEW = "post_trade_review"
    GENERAL_TRADING = "general_trading"


class KnowledgeType(str, Enum):
    TRADE_EXPERIENCE = "trade_experience"
    STRATEGY_INSIGHT = "strategy_insight"
    MISTAKE_LEARNING = "mistake_learning"
    SUCCESS_PATTERN = "success_pattern"
    MARKET_OBSERVATION = "market_observation"


# ── Lazy service getters ──

def _get_llm_service():
    try:
        from intelligence.llm_integration import get_llm_service
        return get_llm_service()
    except Exception as e:
        logger.warning("LLM service unavailable: %s", e)
        return None


def _get_rag_service():
    try:
        from intelligence.personalized_rag import get_personalized_rag
        return get_personalized_rag()
    except Exception as e:
        logger.warning("RAG service unavailable: %s", e)
        return None


def _require_llm():
    svc = _get_llm_service()
    if svc is None:
        raise HTTPException(503, "LLM service unavailable (missing optional packages)")
    return svc


def _require_rag():
    svc = _get_rag_service()
    if svc is None:
        raise HTTPException(503, "RAG service unavailable (missing optional packages)")
    return svc


# ── Pydantic Models ──

class CoachingQueryRequest(BaseModel):
    trader_id: str = Field(..., description="Trader ID")
    query_type: QueryType = Field(..., description="Type of coaching query")
    question: str = Field(..., description="Coaching question")
    current_trade: Optional[Dict[str, Any]] = Field(None)
    chart_context: Optional[Dict[str, Any]] = Field(None)
    market_data: Optional[Dict[str, Any]] = Field(None)
    urgency: str = Field("normal")


class CoachingResponse(BaseModel):
    query_id: str
    trader_id: str
    response: str
    personalization_factors: List[str]
    confidence_score: float
    relevant_patterns: List[str]
    recommended_actions: List[str]
    risk_warnings: List[str]
    follow_up_questions: List[str]
    response_time_ms: float
    timestamp: str


class StrategyRecommendationRequest(BaseModel):
    trader_id: str
    market_regime: str
    setup_type: str
    additional_context: Optional[Dict[str, Any]] = None


class KnowledgeAddRequest(BaseModel):
    trader_id: str
    knowledge_type: KnowledgeType
    title: str
    description: str
    key_insights: List[str] = Field(default_factory=list)
    market_conditions: Optional[Dict[str, Any]] = None
    emotional_state: str = "neutral"
    strategy_used: Optional[str] = None
    outcome: str = "neutral"
    impact_score: float = 0.0
    confidence_score: float = 0.5
    tags: List[str] = Field(default_factory=list)


class PersonalizedContextRequest(BaseModel):
    trader_id: str
    current_situation: Dict[str, Any]


# ── API Endpoints ──

@llm_router.post("/coaching/query", response_model=CoachingResponse)
async def process_coaching_query(request: CoachingQueryRequest):
    """Process personalized coaching query using LLM with RAG."""
    try:
        service = _require_llm()
        from intelligence.llm_integration import CoachingQuery as _CQ

        query = _CQ(
            query_id=str(uuid.uuid4()),
            trader_id=request.trader_id,
            query_type=request.query_type,
            question=request.question,
            current_trade=request.current_trade,
            chart_context=request.chart_context,
            market_data=request.market_data,
            urgency=request.urgency,
        )
        response = await service.process_coaching_query(query)

        return CoachingResponse(
            query_id=response.query_id,
            trader_id=response.trader_id,
            response=response.response,
            personalization_factors=response.personalization_factors,
            confidence_score=response.confidence_score,
            relevant_patterns=response.relevant_patterns,
            recommended_actions=response.recommended_actions,
            risk_warnings=response.risk_warnings,
            follow_up_questions=response.follow_up_questions,
            response_time_ms=response.response_time_ms,
            timestamp=response.timestamp.isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Coaching query failed: %s", e)
        raise HTTPException(500, f"Coaching query failed: {e}")


@llm_router.post("/coaching/strategy")
async def get_strategy_recommendations(request: StrategyRecommendationRequest):
    """Get personalized strategy recommendations."""
    try:
        service = _require_llm()
        recs = await service.get_strategy_recommendations(
            trader_id=request.trader_id,
            market_regime=request.market_regime,
            setup_type=request.setup_type,
        )
        return {
            "trader_id": request.trader_id,
            "market_regime": request.market_regime,
            "setup_type": request.setup_type,
            "recommendations": recs.get("recommendations", []),
            "personalization_factors": recs.get("trader_context", {}),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Strategy recommendations failed: %s", e)
        raise HTTPException(500, f"Strategy recommendations failed: {e}")


@llm_router.post("/coaching/context/{trader_id}")
async def get_personalized_context(
    trader_id: str = Path(...),
    request: Optional[PersonalizedContextRequest] = None,
):
    """Get personalized context for current trading situation."""
    try:
        rag = _require_rag()
        situation = request.current_situation if request else {}
        context = await rag.get_personalized_context(trader_id=trader_id, current_situation=situation)
        return {"trader_id": trader_id, "context": context, "timestamp": datetime.utcnow().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Context retrieval failed: %s", e)
        raise HTTPException(500, f"Context retrieval failed: {e}")


@llm_router.get("/coaching/context/{trader_id}")
async def get_trader_context(trader_id: str = Path(...)):
    """Get trader's current context summary."""
    try:
        rag = _require_rag()
        summary = rag.get_trader_knowledge_summary(trader_id)
        return {"trader_id": trader_id, "knowledge_summary": summary, "timestamp": datetime.utcnow().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Context summary failed: %s", e)
        raise HTTPException(500, f"Context summary failed: {e}")


@llm_router.post("/knowledge/add")
async def add_knowledge_entry(request: KnowledgeAddRequest):
    """Add knowledge entry to trader's personalized knowledge base."""
    try:
        rag = _require_rag()
        entry_id = ""
        if request.knowledge_type == KnowledgeType.STRATEGY_INSIGHT:
            entry_id = await rag.add_strategy_insight(
                trader_id=request.trader_id,
                strategy_name=request.strategy_used or "unknown",
                insight=request.description,
                performance_data={"impact": request.impact_score, "confidence": request.confidence_score},
            )
        elif request.knowledge_type == KnowledgeType.MISTAKE_LEARNING:
            entry_id = await rag.add_mistake_learning(
                trader_id=request.trader_id,
                mistake_description=request.title,
                lesson_learned=request.description,
                context={"market_conditions": request.market_conditions, "emotional_state": request.emotional_state, "strategy": request.strategy_used},
            )
        return {
            "entry_id": entry_id,
            "trader_id": request.trader_id,
            "knowledge_type": request.knowledge_type.value,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Knowledge addition failed: %s", e)
        raise HTTPException(500, f"Knowledge addition failed: {e}")


@llm_router.get("/knowledge/{trader_id}")
async def get_trader_knowledge(
    trader_id: str = Path(...),
    knowledge_type: Optional[str] = Query(None),
    limit: int = Query(20),
):
    """Get trader's knowledge entries."""
    try:
        rag = _require_rag()
        from intelligence.personalized_rag import KnowledgeType as _KT
        knowledge_types = None
        if knowledge_type:
            try:
                knowledge_types = [_KT(knowledge_type)]
            except ValueError:
                raise HTTPException(400, f"Invalid knowledge type: {knowledge_type}")
        results = await rag.retrieve_relevant_knowledge(
            trader_id=trader_id, query="", knowledge_types=knowledge_types, limit=limit
        )
        return {
            "trader_id": trader_id,
            "knowledge_type": knowledge_type,
            "entries": [r.to_dict() for r in results],
            "total_count": len(results),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Knowledge retrieval failed: %s", e)
        raise HTTPException(500, f"Knowledge retrieval failed: {e}")


@llm_router.get("/knowledge/summary/{trader_id}")
async def get_knowledge_summary(trader_id: str = Path(...)):
    """Get summary of trader's knowledge base."""
    try:
        rag = _require_rag()
        summary = rag.get_trader_knowledge_summary(trader_id)
        return {"trader_id": trader_id, "summary": summary, "timestamp": datetime.utcnow().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Knowledge summary failed: %s", e)
        raise HTTPException(500, f"Knowledge summary failed: {e}")


@llm_router.delete("/knowledge/cache/{trader_id}")
async def clear_knowledge_cache(trader_id: str = Path(...)):
    """Clear cached data for a trader."""
    try:
        llm = _require_llm()
        llm.clear_context_cache(trader_id)
        return {"trader_id": trader_id, "status": "success", "message": "Cache cleared", "timestamp": datetime.utcnow().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Cache clearing failed: %s", e)
        raise HTTPException(500, f"Cache clearing failed: {e}")


@llm_router.get("/health/llm")
async def get_llm_health():
    """Get LLM service health status."""
    svc = _get_llm_service()
    if svc is None:
        return {"status": "unavailable", "timestamp": datetime.utcnow().isoformat()}
    return {
        "status": "healthy",
        "llm_provider": svc.llm_provider.value,
        "model_name": svc.model_name,
        "context_cache_size": len(svc.context_cache),
        "timestamp": datetime.utcnow().isoformat(),
    }


@llm_router.get("/health/rag")
async def get_rag_health():
    """Get RAG service health status."""
    svc = _get_rag_service()
    if svc is None:
        return {"status": "unavailable", "timestamp": datetime.utcnow().isoformat()}
    return {
        "status": "healthy",
        "total_knowledge_entries": len(svc.knowledge_entries),
        "trader_count": len(svc.trader_indices),
        "timestamp": datetime.utcnow().isoformat(),
    }


@llm_router.get("/tokens/today")
async def get_token_usage_today(user=Depends(get_current_user)):
    """Get today's LLM token usage breakdown by caller and model."""
    from agents.llm_client import get_token_tracker
    tracker = get_token_tracker()
    return tracker.get_today()


@llm_router.get("/tokens/summary")
async def get_token_usage_summary(
    days: int = Query(7, ge=1, le=90),
    user=Depends(get_current_user),
):
    """Get multi-day LLM token usage summary."""
    from agents.llm_client import get_token_tracker
    tracker = get_token_tracker()
    return tracker.get_summary(days=days)


@llm_router.get("/tokens/recent")
async def get_recent_llm_calls(
    limit: int = Query(20, ge=1, le=100),
    user=Depends(get_current_user),
):
    """Get the most recent LLM calls with token details (for debugging)."""
    from agents.llm_client import get_token_tracker
    tracker = get_token_tracker()
    return {"calls": tracker.get_recent_calls(limit=limit)}


@llm_router.get("/examples/coaching")
async def get_coaching_examples():
    """Get examples of coaching queries and responses."""
    return {
        "examples": [
            {"query_type": "entry_decision", "question": "Should I enter this breakout on AAPL?"},
            {"query_type": "risk_assessment", "question": "Is my position size appropriate for current volatility?"},
            {"query_type": "psychology_coaching", "question": "I'm frustrated after 3 losses. What should I do?"},
        ]
    }
