"""
Research API Routes
====================

Endpoints for the autonomous research agent:
  - GET  /research/findings              list findings (filter by status, type, strategy)
  - GET  /research/findings/{id}         finding detail
  - POST /research/findings/{id}/accept  trader accepts → enters AVT proof cycle
  - POST /research/findings/{id}/reject  trader rejects → status=rejected
  - POST /research/run                   manual research run (admin/dev)
  - GET  /research/status                last run summary

Real-money safety:
  - Accept does NOT auto-apply changes to live strategies
  - Accept marks finding as 'in_proof_cycle' which feeds the existing AVT proof gate
  - Reject is permanent — won't resurface
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session as DBSession

from api.auth import get_current_user
from core.database import get_db_dependency as get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["research"], dependencies=[Depends(get_current_user)])


# ─── Schemas ──────────────────────────────────────────────────────


class ResearchFindingItem(BaseModel):
    id: str
    type: str
    strategy_id: Optional[str]
    symbol: Optional[str]
    detected_at: datetime
    confidence: str
    p_value: Optional[float]
    sample_size: Optional[int]
    title: str
    summary: str
    rationale: Optional[str]
    suggested_action: Optional[str]
    status: str
    payload: dict = Field(default_factory=dict)
    related_finding_ids: Optional[list] = None
    created_at: datetime
    updated_at: datetime


class FindingActionRequest(BaseModel):
    note: Optional[str] = Field(None, max_length=500, description="Optional note from trader")


class RunSummary(BaseModel):
    started_at: str
    ended_at: str
    duration_seconds: float
    n_strategies_analyzed: int
    n_decay_findings: int
    n_emergence_findings: int
    n_meta_findings: int
    persist_result: dict


# ─── List & Get ───────────────────────────────────────────────────


@router.get("/findings", response_model=list[ResearchFindingItem])
def list_findings(
    status: Optional[str] = Query(None, description="Filter by status (pending_review, accepted, rejected, in_proof_cycle, proven, expired)"),
    type: Optional[str] = Query(None, description="Filter by type (decay_detected, edge_emerging, regime_shift, etc.)"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy"),
    limit: int = Query(50, ge=1, le=500),
    db: DBSession = Depends(get_db),
):
    from core.models import ResearchFinding

    q = db.query(ResearchFinding)
    if status:
        q = q.filter(ResearchFinding.status == status)
    if type:
        q = q.filter(ResearchFinding.type == type)
    if strategy_id:
        q = q.filter(ResearchFinding.strategy_id == strategy_id)
    rows = q.order_by(desc(ResearchFinding.detected_at)).limit(limit).all()
    return [ResearchFindingItem.model_validate(r, from_attributes=True) for r in rows]


@router.get("/findings/{finding_id}", response_model=ResearchFindingItem)
def get_finding(finding_id: str, db: DBSession = Depends(get_db)):
    from core.models import ResearchFinding

    row = db.query(ResearchFinding).filter_by(id=finding_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Finding not found")
    return ResearchFindingItem.model_validate(row, from_attributes=True)


# ─── Accept / Reject ──────────────────────────────────────────────


def _append_status_history(row, *, frm: str, to: str, by: str, note: Optional[str] = None):
    """Append an audit entry to status_history JSON column."""
    history = list(row.status_history or [])
    history.append({
        "from": frm,
        "to": to,
        "at": datetime.now().isoformat(),
        "by": by,
        "note": note,
    })
    row.status_history = history


@router.post("/findings/{finding_id}/accept", response_model=ResearchFindingItem)
def accept_finding(
    finding_id: str,
    body: FindingActionRequest,
    db: DBSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Trader accepts a finding. Status moves to 'in_proof_cycle' — the AVT proof
    pipeline will accumulate evidence (50 trades + 3 weeks + Mann-Whitney) before
    it's promoted to 'proven' and shown as a live recommendation.

    DOES NOT auto-apply the suggested action — the trader must still execute it
    themselves (e.g., reduce sizing, enable strategy). This endpoint just signals
    intent and unlocks the proof cycle.
    """
    from core.models import ResearchFinding

    row = db.query(ResearchFinding).filter_by(id=finding_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Finding not found")
    if row.status != "pending_review":
        raise HTTPException(status_code=400, detail=f"Cannot accept finding in status '{row.status}'")

    by_user = getattr(current_user, "id", None) or getattr(current_user, "email", None) or "unknown"
    _append_status_history(row, frm=row.status, to="in_proof_cycle", by=str(by_user), note=body.note)
    row.status = "in_proof_cycle"
    row.updated_at = datetime.now()
    db.commit()
    db.refresh(row)
    return ResearchFindingItem.model_validate(row, from_attributes=True)


@router.post("/findings/{finding_id}/reject", response_model=ResearchFindingItem)
def reject_finding(
    finding_id: str,
    body: FindingActionRequest,
    db: DBSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Trader rejects a finding. Permanent — won't resurface in pending_review filter.
    """
    from core.models import ResearchFinding

    row = db.query(ResearchFinding).filter_by(id=finding_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Finding not found")
    if row.status not in ("pending_review", "in_proof_cycle"):
        raise HTTPException(status_code=400, detail=f"Cannot reject finding in status '{row.status}'")

    by_user = getattr(current_user, "id", None) or getattr(current_user, "email", None) or "unknown"
    _append_status_history(row, frm=row.status, to="rejected", by=str(by_user), note=body.note)
    row.status = "rejected"
    row.updated_at = datetime.now()
    db.commit()
    db.refresh(row)
    return ResearchFindingItem.model_validate(row, from_attributes=True)


# ─── Manual run + status ──────────────────────────────────────────


@router.post("/run", response_model=RunSummary)
def run_now(db: DBSession = Depends(get_db)):
    """
    Manually trigger a research agent run. Useful for:
      - Initial deploy / smoke test
      - Calibration runs after threshold changes
      - On-demand check-ins between scheduled runs

    Returns the run summary (counts of findings produced).
    """
    from core.database import SessionLocal
    from research_agent.scheduler import ResearchAgentScheduler

    scheduler = ResearchAgentScheduler(db_factory=SessionLocal)
    summary = scheduler.run_once()
    return summary


@router.get("/status")
def status():
    """Quick liveness/info endpoint."""
    from research_agent.scheduler import KILL_SWITCH_ENV, ResearchAgentScheduler
    import os
    return {
        "kill_switch_env": KILL_SWITCH_ENV,
        "kill_switch_active": ResearchAgentScheduler.kill_switch_active(),
        "kill_switch_value": os.getenv(KILL_SWITCH_ENV, ""),
    }


# ═══════════════════════════════════════════════════════════════════
#   USER-INITIATED CONTINUOUS RESEARCH (Phase 4 enhanced)
# ═══════════════════════════════════════════════════════════════════
#
# Workflow:
#   POST /research/discover-continuous  → returns task_id immediately
#   GET  /research/tasks                → list active tasks for the trader
#   GET  /research/tasks/{id}           → progress + counts
#   POST /research/tasks/{id}/stop      → graceful stop
#   GET  /research/drafts               → list validated strategies awaiting decision
#   GET  /research/drafts/{id}          → draft detail (full YAML + stats)
#   POST /research/drafts/{id}/send-to-avt → write YAML, AVT picks up via hot-reload
#   POST /research/drafts/{id}/discard  → mark discarded
#
# NO automatic trading anywhere in this loop.


# ─── Schemas (continuous research) ────────────────────────────────


class StartContinuousResearchRequest(BaseModel):
    scope: str = Field(..., description="fno | nifty50 | comma-separated tickers", min_length=2)
    theme: Optional[str] = Field(None, max_length=255, description="optional human hint, e.g. 'momentum on FNO'")
    pattern_types: Optional[list[str]] = Field(
        None,
        description="optional list of pattern types to test; defaults to library set",
    )
    lookback_years: int = Field(2, ge=1, le=10)
    similarity_threshold: float = Field(0.85, ge=0.5, le=0.99)
    max_candidates: int = Field(100, ge=1, le=500)

    # Validation thresholds
    min_sample_size: int = Field(30, ge=10, le=500)
    min_win_rate: float = Field(0.55, ge=0.4, le=0.9)
    min_avg_r: float = Field(0.5, ge=0.0, le=3.0)


class TaskItem(BaseModel):
    id: str
    trader_id: Optional[str]
    theme: Optional[str]
    scope: str
    symbols: list[str] = Field(default_factory=list)
    pattern_types: list[str] = Field(default_factory=list)
    lookback_years: int
    similarity_threshold: float
    min_sample_size: int
    min_win_rate: float
    min_avg_r: float
    status: str
    candidates_planned: int
    candidates_completed: int
    candidates_validated: int
    current_candidate: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    stopped_at: Optional[datetime]


class DraftItem(BaseModel):
    id: str
    task_id: Optional[str]
    symbol: str
    pattern_type: str
    sample_size: int
    win_rate: float
    avg_r: float
    profit_factor: Optional[float]
    best_regime: Optional[str]
    best_time_slot: Optional[str]
    htf_alignment: Optional[str]
    optimal_target_r: Optional[float]
    reversal_cliff_r: Optional[float]
    strategy_name: str
    status: str
    sent_to_avt_at: Optional[datetime]
    avt_strategy_id: Optional[str]
    yaml_path: Optional[str]
    user_note: Optional[str]
    created_at: datetime
    updated_at: datetime


class DraftDetail(DraftItem):
    strategy_yaml: str
    strategy_dna: dict


class SendToAvtRequest(BaseModel):
    note: Optional[str] = Field(None, max_length=500)


# ─── Continuous research endpoints ────────────────────────────────


@router.post("/discover-continuous", response_model=TaskItem)
def start_continuous_research(
    body: StartContinuousResearchRequest,
    db: DBSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Start a long-running continuous discovery task.

    Enumerates (symbol × pattern_type) candidates from the scope, runs the existing
    7-stage discovery pipeline on each, and persists validated results as DiscoveryDraft
    rows. Returns the task immediately — poll GET /research/tasks/{id} for progress.

    Drafts are inert until user clicks "Send to AVT" on each one individually.
    """
    from core.database import SessionLocal
    from core.models import ResearchTask
    from research_agent.continuous_research import (
        ResearchTaskManager, ValidationThresholds, get_task_manager,
    )

    trader_id = getattr(current_user, "id", None) or getattr(current_user, "email", None) or "unknown"

    manager = get_task_manager(db_factory=SessionLocal)
    thresholds = ValidationThresholds(
        min_sample_size=body.min_sample_size,
        min_win_rate=body.min_win_rate,
        min_avg_r=body.min_avg_r,
    )

    try:
        task = manager.start_task(
            scope=body.scope,
            theme=body.theme,
            pattern_types=body.pattern_types,
            lookback_years=body.lookback_years,
            similarity_threshold=body.similarity_threshold,
            thresholds=thresholds,
            trader_id=str(trader_id),
            max_candidates=body.max_candidates,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to spawn task: {e}")

    # Read the persisted row to return canonical state
    row = db.query(ResearchTask).filter_by(id=task.task_id).first()
    if not row:
        # Race — task hasn't persisted yet. Return summary.
        s = task.summary()
        return TaskItem(
            id=task.task_id,
            trader_id=str(trader_id),
            theme=body.theme,
            scope=body.scope,
            symbols=task.symbols,
            pattern_types=task.pattern_types,
            lookback_years=body.lookback_years,
            similarity_threshold=body.similarity_threshold,
            min_sample_size=body.min_sample_size,
            min_win_rate=body.min_win_rate,
            min_avg_r=body.min_avg_r,
            status=s["status"],
            candidates_planned=s["candidates_planned"],
            candidates_completed=s["candidates_completed"],
            candidates_validated=s["candidates_validated"],
            current_candidate=s["current_candidate"],
            error_message=s["error_message"],
            created_at=datetime.now(),
            started_at=None,
            stopped_at=None,
        )
    return TaskItem.model_validate(row, from_attributes=True)


@router.get("/tasks", response_model=list[TaskItem])
def list_tasks(
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: DBSession = Depends(get_db),
):
    from core.models import ResearchTask
    q = db.query(ResearchTask)
    if status:
        q = q.filter(ResearchTask.status == status)
    rows = q.order_by(desc(ResearchTask.created_at)).limit(limit).all()
    return [TaskItem.model_validate(r, from_attributes=True) for r in rows]


@router.get("/tasks/{task_id}", response_model=TaskItem)
def get_task(task_id: str, db: DBSession = Depends(get_db)):
    from core.models import ResearchTask
    row = db.query(ResearchTask).filter_by(id=task_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskItem.model_validate(row, from_attributes=True)


@router.post("/tasks/{task_id}/stop", response_model=TaskItem)
def stop_task(task_id: str, db: DBSession = Depends(get_db)):
    from core.models import ResearchTask
    from research_agent.continuous_research import get_task_manager
    from core.database import SessionLocal

    manager = get_task_manager(db_factory=SessionLocal)
    if not manager.stop_task(task_id):
        # Task may have completed already or process restarted — mark in DB regardless
        row = db.query(ResearchTask).filter_by(id=task_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        if row.status == "running":
            row.status = "stopped"
            row.stopped_at = datetime.now()
            db.commit()

    row = db.query(ResearchTask).filter_by(id=task_id).first()
    return TaskItem.model_validate(row, from_attributes=True)


# ─── Draft endpoints ──────────────────────────────────────────────


@router.get("/drafts", response_model=list[DraftItem])
def list_drafts(
    task_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None, description="pending_review | sent_to_avt | discarded"),
    limit: int = Query(50, ge=1, le=500),
    db: DBSession = Depends(get_db),
):
    from core.models import DiscoveryDraft
    q = db.query(DiscoveryDraft)
    if task_id:
        q = q.filter(DiscoveryDraft.task_id == task_id)
    if status:
        q = q.filter(DiscoveryDraft.status == status)
    rows = q.order_by(desc(DiscoveryDraft.created_at)).limit(limit).all()
    return [DraftItem.model_validate(r, from_attributes=True) for r in rows]


@router.get("/drafts/{draft_id}", response_model=DraftDetail)
def get_draft(draft_id: str, db: DBSession = Depends(get_db)):
    from core.models import DiscoveryDraft
    row = db.query(DiscoveryDraft).filter_by(id=draft_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Draft not found")
    return DraftDetail.model_validate(row, from_attributes=True)


@router.post("/drafts/{draft_id}/send-to-avt", response_model=DraftDetail)
def send_draft_to_avt(
    draft_id: str,
    body: SendToAvtRequest,
    db: DBSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Promote a discovery draft to an active strategy YAML.

    Writes YAML to config/strategies/, publishes STRATEGY_DISCOVERED event
    (which existing hot-reload handler picks up), and updates draft status.

    AVT will start virtual-trading the strategy on its next scan cycle.
    Trader still controls everything — virtual trades are NOT real money.
    """
    from pathlib import Path
    from core.events import Event, EventType, get_event_bus
    from core.models import DiscoveryDraft

    row = db.query(DiscoveryDraft).filter_by(id=draft_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Draft not found")
    if row.status != "pending_review":
        raise HTTPException(status_code=400, detail=f"Cannot send draft in status '{row.status}'")

    if not row.strategy_yaml:
        raise HTTPException(status_code=400, detail="Draft has no YAML content — cannot promote")

    # Write YAML to config/strategies/
    strategies_dir = Path("config/strategies")
    strategies_dir.mkdir(parents=True, exist_ok=True)
    yaml_filename = f"{row.strategy_name}.yaml"
    yaml_path = strategies_dir / yaml_filename

    header = (
        f"# {yaml_filename}\n"
        f"# Promoted from continuous research draft {row.id}\n"
        f"# Symbol: {row.symbol} | Pattern: {row.pattern_type}\n"
        f"# Backtest: {row.sample_size} samples | WR {row.win_rate:.0%} | Avg R {row.avg_r:.2f}\n"
        f"# Promoted at: {datetime.now().isoformat()}\n"
        f"# User note: {body.note or '(none)'}\n"
        f"# AVT will virtual-trade this strategy. NOT live trading.\n\n"
    )
    yaml_path.write_text(header + row.strategy_yaml, encoding="utf-8")

    # Update draft state
    row.status = "sent_to_avt"
    row.sent_to_avt_at = datetime.now()
    row.yaml_path = str(yaml_path)
    row.avt_strategy_id = row.strategy_name
    row.user_note = body.note
    db.commit()

    # Publish event so hot-reload handler picks it up
    try:
        bus = get_event_bus()
        bus.publish(Event(
            event_type=EventType.STRATEGY_DISCOVERED,
            source="research.send_to_avt",
            data={
                "draft_id": row.id,
                "strategy_name": row.strategy_name,
                "yaml_path": str(yaml_path),
                "symbol": row.symbol,
                "pattern": row.pattern_type,
                "win_rate": row.win_rate,
                "avg_r": row.avg_r,
                "sample_size": row.sample_size,
            },
        ))
        logger.info("Draft %s sent to AVT: %s", row.id, yaml_path)
    except Exception as e:
        logger.warning("Failed to publish STRATEGY_DISCOVERED event for draft %s: %s", row.id, e)

    db.refresh(row)
    return DraftDetail.model_validate(row, from_attributes=True)


@router.post("/drafts/{draft_id}/discard", response_model=DraftItem)
def discard_draft(
    draft_id: str,
    body: SendToAvtRequest,
    db: DBSession = Depends(get_db),
):
    from core.models import DiscoveryDraft
    row = db.query(DiscoveryDraft).filter_by(id=draft_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Draft not found")
    row.status = "discarded"
    row.user_note = body.note
    db.commit()
    db.refresh(row)
    return DraftItem.model_validate(row, from_attributes=True)
