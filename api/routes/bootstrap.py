# api/routes/bootstrap.py
"""
Bootstrap Understanding System API routes.

Endpoints:
  GET    /api/v1/bootstrap/patterns           — List all pattern templates
  GET    /api/v1/bootstrap/patterns/search     — Search patterns by keyword
  GET    /api/v1/bootstrap/patterns/{id}       — Get full pattern template
  POST   /api/v1/bootstrap/start               — Start bootstrap session
  GET    /api/v1/bootstrap/sessions/{id}       — Get session state
  POST   /api/v1/bootstrap/sessions/{id}/correct  — Apply corrections
  POST   /api/v1/bootstrap/sessions/{id}/finalize  — Save profile + DNA YAML
  POST   /api/v1/bootstrap/interpret           — Standalone interpretation (preview)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/bootstrap",
    tags=["bootstrap"],
    dependencies=[Depends(get_current_user)],
)


# ── Singleton orchestrator ────────────────────────────────────

_orchestrator = None


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from bootstrap.orchestrator import BootstrapOrchestrator
        _orchestrator = BootstrapOrchestrator()
    return _orchestrator


# ── Request/Response schemas ──────────────────────────────────


class StartRequest(BaseModel):
    """Start a new bootstrap session."""
    description: str = Field(
        ..., min_length=10,
        description="Natural language description of your trading strategy",
    )
    selected_pattern_ids: List[str] = Field(
        default_factory=list,
        description="Pattern template IDs to use as starting point",
    )
    timeframe: str = Field("5min", description="Primary timeframe")
    direction: str = Field("long", description="Trade direction: long, short, both")
    instruments: List[str] = Field(
        default_factory=list,
        description="Instruments to backtest (default: FNO universe)",
    )
    user_claimed_wr: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Your claimed win rate (0.0-1.0)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "description": (
                    "I trade compression breakouts on 5-min Nifty stocks. "
                    "I look for tight range with decreasing volume, then enter "
                    "on breakout with volume surge. Stop below compression low. "
                    "I skip ugly compressions with big wicks."
                ),
                "selected_pattern_ids": ["compression_breakout"],
                "timeframe": "5min",
                "direction": "long",
                "user_claimed_wr": 0.55,
            }
        }


class CorrectionRequest(BaseModel):
    """Apply corrections to a bootstrap session."""
    corrections: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of corrections to apply",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "corrections": [
                    {
                        "type": "modify_rule",
                        "rule_index": 0,
                        "field": "value",
                        "new_value": 50,
                    },
                    {
                        "type": "answer_question",
                        "answer": "I use 1.5 ATR stop loss, tighter in low vol",
                    },
                ]
            }
        }


class InterpretRequest(BaseModel):
    """Standalone interpretation (preview without full pipeline)."""
    description: str = Field(..., min_length=10)
    pattern_ids: List[str] = Field(default_factory=list)


class PatternSearchRequest(BaseModel):
    """Search patterns by keyword."""
    q: str = Field(..., min_length=2)


# ── Pattern Library Endpoints ─────────────────────────────────


@router.get("/patterns")
async def list_patterns():
    """List all available pattern templates."""
    orch = _get_orchestrator()
    return {"patterns": orch.list_patterns()}


@router.get("/patterns/search")
async def search_patterns(q: str = ""):
    """Search patterns by keyword."""
    if len(q) < 2:
        raise HTTPException(400, "Query must be at least 2 characters")
    orch = _get_orchestrator()
    return {"results": orch.search_patterns(q)}


@router.get("/patterns/{pattern_id}")
async def get_pattern(pattern_id: str):
    """Get full details of a pattern template."""
    orch = _get_orchestrator()
    result = orch.get_pattern(pattern_id)
    if result is None:
        raise HTTPException(404, f"Pattern '{pattern_id}' not found")
    return result


# ── Session Endpoints ─────────────────────────────────────────


@router.post("/start")
async def start_bootstrap(req: StartRequest):
    """
    Start a new bootstrap session.

    Interprets the NL description, runs fast backtest,
    and warm-starts downstream systems.
    """
    orch = _get_orchestrator()
    try:
        session = await orch.start(
            description=req.description,
            selected_pattern_ids=req.selected_pattern_ids,
            timeframe=req.timeframe,
            direction=req.direction,
            instruments=req.instruments,
            user_claimed_wr=req.user_claimed_wr,
        )
        return session.to_dict()
    except Exception as e:
        logger.error("Bootstrap start failed: %s", e)
        raise HTTPException(500, f"Bootstrap failed: {e}")


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get current state of a bootstrap session."""
    orch = _get_orchestrator()
    session = orch.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return session.to_dict()


@router.post("/sessions/{session_id}/correct")
async def correct_session(session_id: str, req: CorrectionRequest):
    """
    Apply corrections and re-run pipeline.

    Supported correction types:
    - modify_rule: Change a hard rule's value
    - add_rule / remove_rule: Add/remove hard rules
    - modify_narrative: Re-interpret with new description
    - answer_question: Answer a teaching question
    - modify_exit: Change exit logic parameters
    """
    orch = _get_orchestrator()
    session = orch.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")

    try:
        session = await orch.correct(session_id, req.corrections)
        return session.to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Correction failed: %s", e)
        raise HTTPException(500, f"Correction failed: {e}")


@router.post("/sessions/{session_id}/finalize")
async def finalize_session(session_id: str, user=Depends(get_current_user)):
    """
    Finalize: save StrategyProfile to DB + generate DNA YAML.

    After this, the strategy is active and all downstream systems
    (AVT, coaching, SKE) can work with it immediately.
    """
    orch = _get_orchestrator()
    session = orch.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")

    try:
        result = await orch.finalize(session_id, trader_id=user.id)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Finalization failed: %s", e)
        raise HTTPException(500, f"Finalization failed: {e}")


# ── Standalone Interpretation ─────────────────────────────────


@router.post("/interpret")
async def interpret_only(req: InterpretRequest):
    """
    Preview: interpret NL description without running full pipeline.
    Useful for UI to show interpretation before committing.
    """
    orch = _get_orchestrator()
    try:
        from bootstrap.nl_interpreter import LLMStrategyInterpreter
        draft = await orch._interpreter.interpret(
            user_text=req.description,
            selected_template_ids=req.pattern_ids,
        )
        return {
            "strategy_name": draft.strategy_name,
            "understanding_confidence": draft.understanding_confidence,
            "hard_rules": draft.hard_rules,
            "soft_scorers": draft.soft_scorers,
            "feature_keys": draft.feature_keys,
            "matched_templates": [
                getattr(t, "name", str(t))
                for t in (draft.matched_templates or [])
            ],
            "gaps": draft.gaps,
            "questions": draft.questions_to_ask,
            "warnings": draft.warnings,
        }
    except Exception as e:
        logger.error("Interpretation failed: %s", e)
        raise HTTPException(500, f"Interpretation failed: {e}")
