# api/routes/profile.py
"""Trader DNA profile and strategy intelligence endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth import get_current_user

from sqlalchemy.orm import Session as DBSession

from behavioral.trader_profile import TraderDNAProfiler
from core.database import get_db
from strategy_engine.strategy_recommender import StrategyDiscoveryEngine
from core.models import StrategyProfile

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/profile", tags=["trader profile"], dependencies=[Depends(get_current_user)])


class TraderDNAResponse(BaseModel):
    trader_type: str = ""
    trader_type_confidence: float = 0.0
    risk: dict = {}
    timing: dict = {}
    thinking: dict = {}
    emotional: dict = {}
    regime_adaptability: dict = {}
    strategy_summary: dict = {}
    scores: dict = {}
    computed_at: str | None = None
    trade_count_used: int = 0


class StrategyCandidateResponse(BaseModel):
    name: str
    description: str
    type: str
    scores: dict = {}
    target_regimes: list[str] = []
    fills_gap_for: list[str] = []
    expected_metrics: dict = {}
    reasons: list[str] = []
    key_rules: list[str] = []
    similar_to: str | None = None


# ══════════════════════════════════════════════
#  GET /profile/dna — Full trader DNA
# ══════════════════════════════════════════════

@router.get("/dna", response_model=TraderDNAResponse)
def get_trader_dna(
    lookback_days: int = 120,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Compute full trader DNA profile.
    
    Analyzes your strategies, trading history, and behavior to determine:
    - Risk style (conservative/moderate/aggressive/adaptive)
    - Timing preferences (early_bird/confirmation/momentum/patient)
    - Thinking style (momentum/structure/probability/volatility)
    - Emotional patterns and triggers
    - Regime adaptability
    """
    profiler = TraderDNAProfiler(lookback_days=lookback_days)
    dna = profiler.compute(db, trader_id=user.id)
    return TraderDNAResponse(**dna.to_dict())


# ══════════════════════════════════════════════
#  GET /profile/strategies — Personal strategies
# ══════════════════════════════════════════════

@router.get("/strategies")
def get_learned_strategies(db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """List all learned strategy profiles."""
    profiles = db.query(StrategyProfile).filter(StrategyProfile.trader_id == user.id, StrategyProfile.status == "active").all()
    strategies = [
        {"name": p.name, "short_code": p.short_code, "confidence": p.confidence_scores, "examples": p.examples_count}
        for p in profiles
    ]
    return {
        "strategies": strategies,
        "count": len(strategies),
    }


# ══════════════════════════════════════════════
#  GET /profile/discover — Strategy suggestions
# ══════════════════════════════════════════════

@router.get("/discover", response_model=list[StrategyCandidateResponse])
def discover_strategies(
    max_suggestions: int = 5,
    db: DBSession = Depends(get_db),
):
    """
    Discover complementary strategies based on your trader DNA.
    
    Suggests strategies that:
    - Fill regime gaps in your playbook
    - Match your risk/thinking/timing style
    - Cover market conditions where you underperform
    """
    profiler = TraderDNAProfiler(lookback_days=120)
    dna = profiler.compute(db)

    engine = StrategyDiscoveryEngine()
    candidates = engine.discover(dna, max_suggestions=max_suggestions)

    return [StrategyCandidateResponse(**c.to_dict()) for c in candidates]