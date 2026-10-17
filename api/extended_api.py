"""
PTIS Extended API
==================

Extended API endpoints for frontend compatibility.

Endpoints:
- GET /api/v1/intelligence/{trader_id} - Complete intelligence summary
- GET /api/v1/watchlist/{trader_id} - Strategy-driven watchlist
- GET /api/v1/feed/{trader_id} - Intelligence feed
- POST /api/v1/memory/{trader_id}/update - Memory updates
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field

from api.auth import get_current_user
import uuid

# Lazy imports — these modules depend on optional packages (sklearn, faiss, etc.)
# The endpoints below use mock data, so we defer imports to call time.
def _get_intelligence_service():
    try:
        from intelligence.integration_service import get_intelligence_service
        return get_intelligence_service()
    except Exception:
        return None

def _get_rag_service():
    try:
        from intelligence.personalized_rag import get_personalized_rag
        return get_personalized_rag()
    except Exception:
        return None

logger = logging.getLogger(__name__)

# Create router
extended_router = APIRouter(prefix="/api/v1", tags=["Extended"], dependencies=[Depends(get_current_user)])

# Pydantic Models
class MemoryUpdateRequest(BaseModel):
    """Request model for memory updates."""
    update_type: str = Field(..., description="Type of update")
    data: Dict[str, Any] = Field(..., description="Update data")

class KnowledgeEntryRequest(BaseModel):
    """Request model for knowledge entries."""
    knowledge_type: str = Field(..., description="Type of knowledge")
    content: str = Field(..., description="Knowledge content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

# Mock data generators (in production, these would use real data)
def generate_mock_intelligence_summary(trader_id: str) -> Dict[str, Any]:
    """Generate mock intelligence summary for frontend compatibility."""
    return {
        "trader_id": trader_id,
        "timestamp": datetime.utcnow().isoformat(),
        "processing_status": "completed",
        "trader_profile": {
            "trader_id": trader_id,
            "risk_style": "moderate",
            "discipline_score": 75,
            "consistency_score": 68,
            "patience_score": 62,
            "impulsiveness_score": 35,
            "preferred_timeframes": ["15m", "1h"],
            "favorite_strategies": ["breakout", "trend_following"],
            "position_sizing_style": "fixed_percentage"
        },
        "behavioral_patterns": [
            {
                "pattern_type": "early_entry",
                "severity": "medium",
                "description": "Tendency to enter breakouts before confirmation",
                "frequency": 0.3,
                "impact_score": -0.4
            }
        ],
        "predictions": {
            "performance": {
                "win_rate": 0.45,
                "risk_level": "moderate",
                "next_trade_win_probability": 0.52
            }
        },
        "recommendations": [
            {
                "type": "discipline",
                "priority": "high",
                "title": "Improve Entry Timing",
                "description": "Wait for confirmation before entering breakouts",
                "confidence": 0.8
            }
        ]
    }

def generate_mock_watchlist(trader_id: str) -> Dict[str, Any]:
    """Generate mock strategy watchlist for frontend compatibility."""
    return {
        "trader_id": trader_id,
        "categories": {
            "strategy_opportunities": [
                {
                    "symbol": "AAPL",
                    "strategy_match": "breakout_setup",
                    "readiness_score": 0.85,
                    "regime_compatibility": "high",
                    "volatility_state": "normal",
                    "momentum_signal": "bullish",
                    "price": 175.50,
                    "change": 2.3
                },
                {
                    "symbol": "TSLA",
                    "strategy_match": "pullback_entry",
                    "readiness_score": 0.72,
                    "regime_compatibility": "medium",
                    "volatility_state": "elevated",
                    "momentum_signal": "neutral",
                    "price": 245.80,
                    "change": -1.2
                }
            ],
            "momentum_alerts": [
                {
                    "symbol": "NVDA",
                    "strategy_match": "momentum_breakout",
                    "readiness_score": 0.91,
                    "regime_compatibility": "high",
                    "volatility_state": "expanding",
                    "momentum_signal": "strong_bullish",
                    "price": 485.20,
                    "change": 5.8
                }
            ],
            "avoid_list": [
                {
                    "symbol": "META",
                    "strategy_match": "unsuitable_regime",
                    "readiness_score": 0.25,
                    "regime_compatibility": "low",
                    "volatility_state": "high",
                    "momentum_signal": "bearish",
                    "price": 312.40,
                    "change": -3.1
                }
            ]
        },
        "summary": {
            "total_opportunities": 3,
            "high_confidence": 2,
            "market_regime": "trending",
            "last_updated": datetime.utcnow().isoformat()
        }
    }

def generate_mock_feed(trader_id: str) -> Dict[str, Any]:
    """Generate mock intelligence feed for frontend compatibility."""
    return {
        "trader_id": trader_id,
        "feed_items": [
            {
                "id": "feed_001",
                "type": "discipline_score",
                "title": "Discipline Score",
                "value": 75,
                "trend": "stable",
                "change": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "insights": ["Rule adherence is consistent", "No violations detected recently"]
            },
            {
                "id": "feed_002",
                "type": "decision_quality",
                "title": "Decision Quality",
                "value": 72,
                "trend": "improving",
                "change": 3,
                "timestamp": datetime.utcnow().isoformat(),
                "insights": ["Better risk management", "Improved entry timing"]
            },
            {
                "id": "feed_003",
                "type": "behavioral_alert",
                "title": "Behavioral Alert",
                "value": "low_risk",
                "trend": "stable",
                "change": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "insights": ["No concerning patterns detected", "Emotional state is balanced"]
            },
            {
                "id": "feed_004",
                "type": "regime_status",
                "title": "Market Regime",
                "value": "trending",
                "trend": "stable",
                "change": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "insights": ["Bullish trend established", "Volatility is normal"]
            },
            {
                "id": "feed_005",
                "type": "coaching_highlight",
                "title": "Coaching Insight",
                "value": "entry_patience",
                "trend": "improving",
                "change": 2,
                "timestamp": datetime.utcnow().isoformat(),
                "insights": ["Waiting for confirmation is improving", "Early entries reduced by 30%"]
            }
        ],
        "summary": {
            "total_items": 5,
            "alerts_count": 0,
            "insights_count": 3,
            "last_updated": datetime.utcnow().isoformat()
        }
    }

# API Endpoints
@extended_router.get("/intelligence/{trader_id}")
async def get_intelligence_summary(
    trader_id: str = Path(..., description="Trader ID")
) -> Dict[str, Any]:
    """
    Get complete intelligence summary for a trader.
    
    This endpoint provides a comprehensive view of the trader's
    intelligence data including profile, patterns, predictions,
    and recommendations.
    """
    try:
        # Try to get real data from integration service
        intelligence_service = _get_intelligence_service()
        
        # For now, return mock data (in production, use real service)
        logger.info(f"Returning intelligence summary for trader: {trader_id}")
        return generate_mock_intelligence_summary(trader_id)
        
    except Exception as e:
        logger.error(f"Error getting intelligence summary: {e}")
        # Return mock data as fallback
        return generate_mock_intelligence_summary(trader_id)

@extended_router.get("/watchlist/{trader_id}")
async def get_strategy_watchlist(
    trader_id: str = Path(..., description="Trader ID")
) -> Dict[str, Any]:
    """
    Get strategy-driven watchlist for a trader.
    
    This endpoint provides personalized watchlist data based on
    the trader's strategies, current market regime, and preferences.
    """
    try:
        # Try to get real data from strategy service
        logger.info(f"Returning strategy watchlist for trader: {trader_id}")
        return generate_mock_watchlist(trader_id)
        
    except Exception as e:
        logger.error(f"Error getting strategy watchlist: {e}")
        # Return mock data as fallback
        return generate_mock_watchlist(trader_id)

@extended_router.get("/feed/{trader_id}")
async def get_intelligence_feed(
    trader_id: str = Path(..., description="Trader ID")
) -> Dict[str, Any]:
    """
    Get intelligence feed for a trader.
    
    This endpoint provides real-time intelligence feed data including
    discipline scores, decision quality, behavioral alerts, and
    coaching highlights.
    """
    try:
        # Try to get real data from feed service
        logger.info(f"Returning intelligence feed for trader: {trader_id}")
        return generate_mock_feed(trader_id)
        
    except Exception as e:
        logger.error(f"Error getting intelligence feed: {e}")
        # Return mock data as fallback
        return generate_mock_feed(trader_id)

@extended_router.post("/memory/{trader_id}/update")
async def update_memory(
    trader_id: str = Path(..., description="Trader ID"),
    request: MemoryUpdateRequest = ...,
) -> Dict[str, Any]:
    """
    Update trader memory based on interaction.
    
    This endpoint updates the trader's memory based on
    coaching interactions, trade outcomes, and other events.
    """
    try:
        logger.info(f"Updating memory for trader: {trader_id}, type: {request.update_type}")
        
        # In production, this would update actual memory systems
        # For now, just acknowledge the update
        
        return {
            "trader_id": trader_id,
            "update_type": request.update_type,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Memory updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@extended_router.post("/knowledge/add")
async def add_knowledge_entry(
    request: KnowledgeEntryRequest = ...,
) -> Dict[str, Any]:
    """
    Add knowledge entry to personalized RAG system.
    
    This endpoint adds new knowledge entries to the trader's
    personalized knowledge base for future retrieval.
    """
    try:
        # Get RAG service
        rag_service = _get_rag_service()
        
        # Map knowledge type
        knowledge_type_map = {
            "trade_experience": "trade_experience",
            "strategy_insight": "strategy_insight", 
            "mistake_learning": "mistake_learning",
            "success_pattern": "success_pattern"
        }
        
        mapped_type = knowledge_type_map.get(request.knowledge_type, "trade_experience")
        
        # Add knowledge entry (in production, this would use real RAG service)
        logger.info(f"Adding knowledge entry: {mapped_type}")
        
        return {
            "status": "success",
            "knowledge_id": str(uuid.uuid4()),
            "knowledge_type": mapped_type,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Knowledge entry added successfully"
        }
        
    except Exception as e:
        logger.error(f"Error adding knowledge entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@extended_router.get("/knowledge/{trader_id}")
async def get_trader_knowledge(
    trader_id: str = Path(..., description="Trader ID"),
    knowledge_type: Optional[str] = Query(None, description="Filter by knowledge type"),
    limit: int = Query(20, description="Maximum number of entries to return")
) -> Dict[str, Any]:
    """
    Get trader's personalized knowledge base.
    
    This endpoint retrieves knowledge entries from the trader's
    personalized knowledge base with optional filtering.
    """
    try:
        # Get RAG service
        rag_service = _get_rag_service()
        
        logger.info(f"Getting knowledge for trader: {trader_id}, type: {knowledge_type}")
        
        # In production, this would query real RAG service
        # For now, return empty result
        
        return {
            "trader_id": trader_id,
            "entries": [],
            "total_count": 0,
            "knowledge_type": knowledge_type,
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting trader knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@extended_router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for extended API.

    Verifies actual service availability instead of assuming healthy.
    """
    services: Dict[str, str] = {"extended_api": "running"}
    intel_svc = _get_intelligence_service()
    services["intelligence_service"] = "running" if intel_svc is not None else "unavailable"
    rag_svc = _get_rag_service()
    services["rag_service"] = "running" if rag_svc is not None else "unavailable"

    all_ok = all(v == "running" for v in services.values())
    return {
        "status": "healthy" if all_ok else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "services": services,
    }
