# api/routes/discovery.py
"""
Pattern Discovery API routes.

Endpoints:
  POST   /api/v1/discovery/discover     — Launch a discovery from NL input
  GET    /api/v1/discovery/{id}          — Get discovery report by ID
  GET    /api/v1/discovery/              — List all discoveries
  POST   /api/v1/discovery/{id}/accept   — Accept a discovered strategy
  POST   /api/v1/discovery/{id}/reject   — Reject a discovered strategy
  POST   /api/v1/discovery/parse         — Parse NL input (Stage 1 only)
  POST   /api/v1/discovery/fingerprint   — Extract fingerprint (Stage 2 only)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import get_current_user

from discovery.models import DiscoveryStatus
from discovery.orchestrator import get_discovery_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/discovery", tags=["discovery"], dependencies=[Depends(get_current_user)])


# ── Request/Response schemas ──────────────────────────────────────

class DiscoverRequest(BaseModel):
    """Natural-language pattern discovery request."""
    input: str = Field(..., description="Natural language description of the pattern", min_length=5)
    similarity_threshold: Optional[float] = Field(None, ge=0.5, le=1.0, description="Similarity threshold (0.5-1.0)")
    lookback_years: Optional[int] = Field(None, ge=1, le=10, description="Years of history to scan")
    scope: Optional[str] = Field(None, description="Stock universe: fno, nifty50, or comma-separated tickers")

    class Config:
        json_schema_extra = {
            "example": {
                "input": "Look at Reliance 15-min chart, today's momentum breakout at 10:30 with solid volume. Find similar across FNO stocks",
                "similarity_threshold": 0.90,
                "lookback_years": 3,
                "scope": "fno",
            }
        }


class ParseRequest(BaseModel):
    """Test the intent parser without running full pipeline."""
    input: str = Field(..., min_length=3)


class AcceptRequest(BaseModel):
    """Accept a discovered strategy."""
    notes: str = Field("", description="Reviewer notes")


class RejectRequest(BaseModel):
    """Reject a discovered strategy."""
    reason: str = Field("", description="Rejection reason")


# ── Routes ────────────────────────────────────────────────────────

@router.post("/discover")
async def discover_pattern(req: DiscoverRequest):
    """
    Launch the full 7-stage pattern discovery pipeline.

    1. Parses your natural language input
    2. Extracts the pattern fingerprint from market data
    3. Scans FNO universe for similar historical patterns
    4. Backtests all matches
    5. Analyses regime and time-of-day performance
    6. Generates a Strategy DNA YAML
    7. Returns the report for your review (accept/reject)
    """
    orch = get_discovery_orchestrator()

    # Build override query params
    from discovery.similarity_scanner import ScanConfig
    scan_cfg = None
    if req.similarity_threshold is not None:
        scan_cfg = ScanConfig(similarity_threshold=req.similarity_threshold)

    report = await orch.discover(
        raw_input=req.input,
        scan_config=scan_cfg,
    )

    return report.to_dict()


@router.get("/")
async def list_discoveries():
    """List all pattern discoveries (current session)."""
    orch = get_discovery_orchestrator()
    return {"discoveries": orch.list_discoveries()}


@router.get("/{discovery_id}")
async def get_discovery(discovery_id: str):
    """Get a specific discovery report by ID."""
    orch = get_discovery_orchestrator()
    report = orch.get_discovery(discovery_id)
    if not report:
        raise HTTPException(status_code=404, detail="Discovery not found")
    return report.to_dict()


@router.post("/{discovery_id}/accept")
async def accept_discovery(discovery_id: str, req: AcceptRequest):
    """
    Accept a discovered strategy.
    This saves the YAML to config/strategies/ and indexes it in RAG.
    """
    orch = get_discovery_orchestrator()
    report = orch.get_discovery(discovery_id)
    if not report:
        raise HTTPException(status_code=404, detail="Discovery not found")
    if report.status != DiscoveryStatus.REVIEW:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot accept discovery in status: {report.status.value}"
        )

    try:
        path = await orch.accept_and_index(report, notes=req.notes)
        return {
            "status": "accepted",
            "strategy_name": report.stored_strategy_id,
            "file_path": path,
            "message": f"Strategy '{report.generated_strategy.display_name}' saved and indexed",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{discovery_id}/reject")
async def reject_discovery(discovery_id: str, req: RejectRequest):
    """Reject a discovered strategy."""
    orch = get_discovery_orchestrator()
    report = orch.get_discovery(discovery_id)
    if not report:
        raise HTTPException(status_code=404, detail="Discovery not found")

    orch.reject(report, reason=req.reason)
    return {
        "status": "rejected",
        "reason": req.reason,
    }


@router.post("/parse")
async def parse_intent(req: ParseRequest):
    """
    Test the intent parser — returns the structured PatternQuery
    without running the full pipeline.
    """
    from discovery.intent_parser import IntentParser
    parser = IntentParser()
    query = await parser.parse(req.input)
    return query.to_dict()


@router.post("/fingerprint")
async def extract_fingerprint(req: ParseRequest):
    """
    Extract a pattern fingerprint from market data.
    Runs Stages 1 + 2 only (parse + extract).
    """
    orch = get_discovery_orchestrator()
    from discovery.intent_parser import IntentParser
    parser = IntentParser()
    query = await parser.parse(req.input)

    try:
        fp = await orch._extractor.extract(query)
        return {
            "query": query.to_dict(),
            "fingerprint": fp.to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
