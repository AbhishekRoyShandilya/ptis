"""
Behavioral Agent API Routes
============================

Endpoints for the autonomous Behavioral Agent (Phase 2 of behavioral intelligence):
  - POST /behavioral-agent/run             — kick off analysis (returns immediately)
  - GET  /behavioral-agent/status/{tid}    — task progress
  - GET  /behavioral-agent/profile         — current trader's behavior profile
  - GET  /behavioral-agent/replay/{tid}    — drill-down per trade
  - GET  /behavioral-agent/insights        — top 1-3 actionable insights

Real-money safety:
  - Read-only: never places trades, never modifies broker accounts
  - Insights capped at 1-3 (per "no lecture" rule)
  - Per-trader scoping via auth dependency
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from api.auth import get_current_user
from core.database import get_db_dependency as get_db

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/behavioral-agent",
    tags=["behavioral-agent"],
    dependencies=[Depends(get_current_user)],
)


# ─── Request / response schemas ────────────────────────────────────


class RunRequest(BaseModel):
    """Optional config for a BA run."""
    since_days: int = Field(90, ge=7, le=365, description="Trades from last N days")
    force_replay: bool = Field(False, description="Re-replay even cached trades")


class RunResponse(BaseModel):
    task_id: str
    status: str
    message: str
    trader_id: str
    started_at: str


class TaskStatus(BaseModel):
    task_id: str
    status: str               # pending | running | completed | failed
    trader_id: str
    started_at: Optional[str]
    finished_at: Optional[str]
    duration_seconds: Optional[float]
    trades_processed: Optional[int]
    replays_created: Optional[int]
    matches_attributed: Optional[int]
    causes_linked: Optional[int]
    biggest_issue_pattern: Optional[str]
    error: Optional[str]


class ProfileResponse(BaseModel):
    trader_id: str
    strengths: List[Dict[str, Any]] = Field(default_factory=list)
    weaknesses: List[Dict[str, Any]] = Field(default_factory=list)
    biggest_issue: Optional[Dict[str, Any]]
    best_strategy: Optional[str]
    worst_pattern: Optional[str]
    discipline_score: Optional[float]
    causal_summary: List[Dict[str, Any]] = Field(default_factory=list)
    archetype: Optional[str]
    trades_analyzed_count: int
    last_analyzed_at: Optional[str]


class ReplayResponse(BaseModel):
    trade_id: str
    symbol: str
    tier: str
    entry_time: str
    exit_time: Optional[str]
    regime_at_entry: Optional[str]
    regime_at_exit: Optional[str]
    trend_alignment: Optional[str]
    peak_r: Optional[float]
    final_r: Optional[float]
    early_exit_severity: Optional[float]
    move_pct_complete_at_entry: Optional[float]
    rsi_at_entry: Optional[float]
    payload: Dict[str, Any]
    causal_sentence: Optional[str] = None


class InsightItem(BaseModel):
    rank: int
    severity: str             # critical | warning | info
    pattern: str
    title: str
    explanation: str
    action: str
    sample_size: int
    frequency_pct: float


# ─── In-memory task registry (process-local, lost on restart) ──────


_tasks: Dict[str, Dict[str, Any]] = {}


def _make_task_id() -> str:
    return f"ba_{uuid.uuid4().hex[:12]}"


def _trader_id_str(current_user: Any) -> str:
    """Extract a stable trader id string from the current user."""
    tid = getattr(current_user, "id", None) or getattr(current_user, "trader_id", None)
    if tid is None:
        raise HTTPException(status_code=401, detail="No trader id on current user")
    return str(tid)


# ─── Run + status ──────────────────────────────────────────────────


@router.post("/run", response_model=RunResponse)
def run_behavioral_agent(
    body: RunRequest,
    bg: BackgroundTasks,
    current_user=Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """
    Kick off a behavioral analysis run for the current trader.
    Runs in background; poll /status/{task_id} for completion.
    """
    trader_id = _trader_id_str(current_user)
    task_id = _make_task_id()
    started_at = datetime.utcnow().isoformat()

    _tasks[task_id] = {
        "task_id": task_id,
        "trader_id": trader_id,
        "status": "pending",
        "started_at": started_at,
        "since_days": body.since_days,
        "force_replay": body.force_replay,
    }

    bg.add_task(_run_in_background, task_id, trader_id, body.since_days, body.force_replay)

    return RunResponse(
        task_id=task_id,
        status="pending",
        message="Behavioral analysis kicked off; poll /status/{task_id}",
        trader_id=trader_id,
        started_at=started_at,
    )


def _run_in_background(task_id: str, trader_id: str, since_days: int, force_replay: bool):
    """Synchronous run wrapped in background task."""
    from agents.behavioral_agent import BehavioralAgent
    from core.database import SessionLocal

    state = _tasks.setdefault(task_id, {})
    state["status"] = "running"

    try:
        # Resolve market data service (optional — replay degrades gracefully without it)
        try:
            from ingestion.market_data_service import get_market_data_service
            mds = get_market_data_service()
        except Exception as e:
            logger.warning("MarketDataService unavailable: %s", e)
            mds = None

        if mds is None:
            state["status"] = "failed"
            state["error"] = "MarketDataService not available — cannot replay"
            state["finished_at"] = datetime.utcnow().isoformat()
            return

        agent = BehavioralAgent(db_session_factory=SessionLocal, market_data_service=mds)
        summary = agent.run(
            trader_id=trader_id,
            since_days=since_days,
            force_replay=force_replay,
        )
        state.update({
            "status": "completed",
            "finished_at": summary.finished_at,
            "duration_seconds": summary.duration_seconds,
            "trades_processed": summary.trades_processed,
            "replays_created": summary.replays_created,
            "replays_reused": summary.replays_reused,
            "matches_attributed": summary.matches_attributed,
            "causes_linked": summary.causes_linked,
            "biggest_issue_pattern": summary.biggest_issue_pattern,
        })
        if summary.errors:
            state["error"] = "; ".join(summary.errors[:3])
    except Exception as e:
        logger.error("BA background run failed (%s): %s", task_id, e, exc_info=True)
        state["status"] = "failed"
        state["finished_at"] = datetime.utcnow().isoformat()
        state["error"] = str(e)


@router.get("/status/{task_id}", response_model=TaskStatus)
def get_task_status(task_id: str, current_user=Depends(get_current_user)):
    state = _tasks.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")
    # Per-trader scoping
    if state.get("trader_id") != _trader_id_str(current_user):
        raise HTTPException(status_code=403, detail="Not your task")
    return TaskStatus(
        task_id=task_id,
        status=state.get("status", "unknown"),
        trader_id=state.get("trader_id", ""),
        started_at=state.get("started_at"),
        finished_at=state.get("finished_at"),
        duration_seconds=state.get("duration_seconds"),
        trades_processed=state.get("trades_processed"),
        replays_created=state.get("replays_created"),
        matches_attributed=state.get("matches_attributed"),
        causes_linked=state.get("causes_linked"),
        biggest_issue_pattern=state.get("biggest_issue_pattern"),
        error=state.get("error"),
    )


# ─── Profile ───────────────────────────────────────────────────────


@router.get("/profile", response_model=ProfileResponse)
def get_profile(current_user=Depends(get_current_user), db: DBSession = Depends(get_db)):
    """Return the current trader's behavior profile (synthesized by latest BA run)."""
    from core.models import BehaviorProfile

    trader_id = _trader_id_str(current_user)
    row = db.query(BehaviorProfile).filter_by(trader_id=trader_id).first()
    if row is None:
        # Empty profile (BA hasn't run yet)
        return ProfileResponse(
            trader_id=trader_id,
            strengths=[],
            weaknesses=[],
            biggest_issue=None,
            best_strategy=None,
            worst_pattern=None,
            discipline_score=None,
            causal_summary=[],
            archetype=None,
            trades_analyzed_count=0,
            last_analyzed_at=None,
        )
    return ProfileResponse(
        trader_id=trader_id,
        strengths=row.strengths or [],
        weaknesses=row.weaknesses or [],
        biggest_issue=row.biggest_issue,
        best_strategy=row.best_strategy,
        worst_pattern=row.worst_pattern,
        discipline_score=row.discipline_score,
        causal_summary=row.causal_summary or [],
        archetype=row.archetype,
        trades_analyzed_count=row.trades_analyzed_count or 0,
        last_analyzed_at=row.last_analyzed_at.isoformat() if row.last_analyzed_at else None,
    )


# ─── Per-trade replay (drill-down) ─────────────────────────────────


@router.get("/replay/{trade_id}", response_model=ReplayResponse)
def get_replay(
    trade_id: str,
    current_user=Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """Return the replay data + causal sentence for a single trade."""
    from core.models import TradeReplay, Trade

    trader_id = _trader_id_str(current_user)

    # Verify trade belongs to current trader
    try:
        trade_uuid = uuid.UUID(trade_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid trade_id")

    trade = db.query(Trade).filter_by(id=trade_uuid).first()
    if trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    if str(trade.trader_id) != trader_id:
        raise HTTPException(status_code=403, detail="Not your trade")

    row = db.query(TradeReplay).filter_by(trade_id=trade_uuid).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Replay not found — run BA first")

    # Optionally compute causal sentence on the fly using stored payload
    causal_sentence = None
    try:
        from behavioral.causal_linker import CausalLinker
        from agents.behavioral_agent import BehavioralAgent
        # Reconstruct ReplayedTrade from the row
        replay = BehavioralAgent._hydrate_replay(trade, row)

        # Match to current strategies (fresh matcher; cheap)
        from behavioral.trade_matcher import TradeStrategyMatcher
        match_result = TradeStrategyMatcher().match(
            replay, declared_strategy_id=getattr(trade, "declared_strategy_id", None),
        )
        explanation = CausalLinker().link(
            replay, match_result, getattr(trade, "declared_strategy_id", None),
            trade_pnl=getattr(trade, "pnl", None),
            trade_r_multiple=getattr(trade, "r_multiple", None),
        )
        causal_sentence = explanation.sentence
    except Exception as e:
        logger.debug("Causal-sentence regeneration failed: %s", e)

    return ReplayResponse(
        trade_id=trade_id,
        symbol=row.symbol,
        tier=row.tier,
        entry_time=row.entry_time.isoformat() if row.entry_time else "",
        exit_time=row.exit_time.isoformat() if row.exit_time else None,
        regime_at_entry=row.regime_at_entry,
        regime_at_exit=row.regime_at_exit,
        trend_alignment=row.trend_alignment,
        peak_r=row.peak_r,
        final_r=row.final_r,
        early_exit_severity=row.early_exit_severity,
        move_pct_complete_at_entry=row.move_pct_complete_at_entry,
        rsi_at_entry=row.rsi_at_entry,
        payload=row.payload or {},
        causal_sentence=causal_sentence,
    )


# ─── Top insights ──────────────────────────────────────────────────


@router.get("/insights", response_model=List[InsightItem])
def get_insights(
    max_items: int = Query(3, ge=1, le=5),
    current_user=Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """
    Top 1-3 actionable insights. Capped per the "no lecture" rule.
    Returns weaknesses ranked by frequency × severity.
    """
    from core.models import BehaviorProfile

    trader_id = _trader_id_str(current_user)
    row = db.query(BehaviorProfile).filter_by(trader_id=trader_id).first()
    if row is None or not (row.weaknesses or []):
        return []

    items: List[InsightItem] = []
    for i, w in enumerate(row.weaknesses[: max_items]):
        sev = (
            "critical" if w.get("frequency_pct", 0) >= 40
            else "warning" if w.get("frequency_pct", 0) >= 20
            else "info"
        )
        pattern = w.get("pattern", "")
        title = _humanize_pattern(pattern)
        items.append(InsightItem(
            rank=i + 1,
            severity=sev,
            pattern=pattern,
            title=title,
            explanation=(
                f"{title} occurred in {w.get('frequency')} of "
                f"{row.trades_analyzed_count} trades ({w.get('frequency_pct')}%). "
                f"{w.get('example', '')}"
            ),
            action=_suggest_action(pattern),
            sample_size=row.trades_analyzed_count,
            frequency_pct=w.get("frequency_pct", 0.0),
        ))

    return items


# ─── Helpers ───────────────────────────────────────────────────────


def _humanize_pattern(pattern: str) -> str:
    """Convert internal cause tag into a human-readable phrase."""
    return {
        "LATE_ENTRY": "Late entries",
        "EARLY_EXIT": "Early exits",
        "RR_KILLED": "Letting winners turn into losers",
        "REGIME_MISMATCH": "Trading in unfavorable regime",
        "REGIME_SHIFT": "Regime shifting against trade",
        "AGAINST_TREND": "Counter-trend entries",
        "OVEREXTENDED": "Entering at extreme RSI",
        "STRATEGY_MISMATCH": "Trade doesn't match declared strategy",
        "NO_STRATEGY_MATCH": "Discretionary / no-strategy trades",
    }.get(pattern, pattern.replace("_", " ").title())


def _suggest_action(pattern: str) -> str:
    """Short, concrete action per pattern. Single sentence — no lecture."""
    return {
        "LATE_ENTRY":
            "Set a hard rule: don't enter if the move is already >65% complete. "
            "Wait for the next setup or skip.",
        "EARLY_EXIT":
            "Pre-define exit at the entry — don't exit on emotion. "
            "Use scale-out at 1R/2R/3R if the strategy supports it.",
        "RR_KILLED":
            "Move stop to break-even at +1R, then trail. Don't let winners give back >50% of peak.",
        "REGIME_MISMATCH":
            "Skip trades when current regime doesn't match your strategy's preferred regime.",
        "REGIME_SHIFT":
            "Check higher-timeframe regime before holding through volatility expansion.",
        "AGAINST_TREND":
            "Add a trend filter: don't take long entries below 1h-50EMA (or short above).",
        "OVEREXTENDED":
            "Don't enter at RSI >75 (long) or <25 (short). Wait for reset.",
        "STRATEGY_MISMATCH":
            "Pre-tag strategy before entry. If conditions don't match the strategy, don't take it.",
        "NO_STRATEGY_MATCH":
            "Reduce discretionary trades. Only take entries with a clear matching strategy.",
    }.get(pattern, "Review the drill-down on these trades to identify the pattern.")
