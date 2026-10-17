"""
Continuous Research Task — User-Initiated
==========================================

User clicks "Start Continuous Research" with a theme/scope. Background loop
enumerates (symbol, pattern_type) candidates and runs the existing 7-stage
discovery pipeline on each. Validated results become DiscoveryDraft rows
that the user reviews and "Sends to AVT" individually.

Design decisions:
  - User-initiated only — no autonomous nightly cron (use ResearchAgentScheduler
    for that, which handles decay/emergence monitoring on existing strategies)
  - Long-running asyncio task — does NOT block API requests
  - In-memory task registry + DB persistence for status (survives restarts)
  - Per-candidate validation gate prevents weak strategies from becoming drafts
  - Stop-anytime via task.stop() or POST /research/tasks/{id}/stop

NO automatic trading anywhere in this loop. Drafts are inert until user
explicitly clicks "Send to AVT".
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────

# Pattern types we'll iterate through (matches built-in library + common variations)
DEFAULT_PATTERN_TYPES: List[str] = [
    "compression_breakout",
    "ema_pullback",
    "vwap_bounce",
    "vwap_reversal",
    "orb_breakout",
    "momentum_exhaustion",
    "gap_fill",
    "pullback_trend",
]

# Per-candidate candle timeframe used for discovery prompts
DEFAULT_TIMEFRAME: str = "15m"

# Pause between candidates so we don't slam the candle fetcher
INTER_CANDIDATE_PAUSE_SECONDS: float = 1.5


# ── Validation thresholds ──────────────────────────────────────────

@dataclass
class ValidationThresholds:
    """Drafts must beat ALL these to be saved (else discarded silently)."""
    min_sample_size: int = 30
    min_win_rate: float = 0.55
    min_avg_r: float = 0.5
    min_profit_factor: float = 1.2


# ── Candidate enumerator ───────────────────────────────────────────

def expand_scope_to_symbols(scope: str) -> List[str]:
    """
    Resolve scope string to a concrete symbol list.
    Reuses the FNO/Nifty50 universes from the existing orchestrator.
    """
    from discovery.orchestrator import FNO_UNIVERSE, NIFTY50

    s = (scope or "").lower().strip()
    if s in ("fno", "fno_universe", ""):
        return list(FNO_UNIVERSE)
    if s in ("nifty50", "nifty_50"):
        return list(NIFTY50)
    # Otherwise treat as comma-separated tickers
    return [t.strip().upper() for t in scope.split(",") if t.strip()]


def enumerate_candidates(
    symbols: List[str],
    pattern_types: List[str],
    timeframe: str = DEFAULT_TIMEFRAME,
    max_candidates: int = 200,
) -> List[Tuple[str, str, str]]:
    """
    Build the candidate list as (symbol, pattern_type, raw_input_for_discovery).
    Returns list capped at max_candidates.
    """
    candidates: List[Tuple[str, str, str]] = []
    for sym in symbols:
        for pat in pattern_types:
            raw_input = (
                f"Find historical {pat.replace('_', ' ')} patterns on {sym} "
                f"({timeframe}). Backtest forward and report the edge."
            )
            candidates.append((sym, pat, raw_input))
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


# ── Persistence helpers ────────────────────────────────────────────

def _build_draft_id(task_id: str, symbol: str, pattern: str) -> str:
    """Stable ID per task+symbol+pattern (avoids duplicate drafts on rerun)."""
    digest = hashlib.sha1(f"{task_id}|{symbol}|{pattern}".encode()).hexdigest()[:12]
    return f"draft_{digest}"


def _validate_report(report: Any, thresholds: ValidationThresholds) -> bool:
    """
    Quick gate: does the discovery report meet our minimum bars to become a draft?

    Reads aggregated stats from DiscoveryReport. Returns True only when all
    thresholds met.
    """
    try:
        # Aggregate stats live on the report (set during backtest stage)
        agg = getattr(report, "aggregate_stats", None) or {}
        sample = int(agg.get("total_trades", 0) or len(getattr(report, "backtest_results", []) or []))
        wr = float(agg.get("win_rate", 0.0) or 0.0)
        avg_r = float(agg.get("avg_r", 0.0) or 0.0)
        pf = float(agg.get("profit_factor", 0.0) or 0.0)
    except Exception:
        return False

    if sample < thresholds.min_sample_size:
        return False
    if wr < thresholds.min_win_rate:
        return False
    if avg_r < thresholds.min_avg_r:
        return False
    if pf and pf < thresholds.min_profit_factor:
        return False
    return True


def _persist_draft(
    db,
    task_id: str,
    trader_id: Optional[str],
    symbol: str,
    pattern: str,
    report: Any,
) -> Optional[str]:
    """Persist a validated DiscoveryReport as a DiscoveryDraft row. Returns draft_id."""
    from core.models import DiscoveryDraft

    draft_id = _build_draft_id(task_id, symbol, pattern)
    existing = db.query(DiscoveryDraft).filter_by(id=draft_id).first()

    # Extract stats safely
    agg = getattr(report, "aggregate_stats", None) or {}
    correlation = getattr(report, "correlation", None)
    best_regime = ""
    best_time_slot = ""
    if correlation:
        # Best regime = highest win_rate regime breakdown
        regimes = getattr(correlation, "regime_breakdowns", []) or []
        if regimes:
            top = max(regimes, key=lambda r: getattr(r, "win_rate", 0))
            best_regime = getattr(top, "regime", "") or ""
        slots = getattr(correlation, "time_slot_breakdowns", []) or []
        if slots:
            top = max(slots, key=lambda s: getattr(s, "win_rate", 0))
            best_time_slot = getattr(top, "slot", "") or ""

    # HTF alignment + Phase 2 optimal target — only present if Multi-TF + target optimizer ran
    backtest_results = getattr(report, "backtest_results", None) or []
    htf_alignment = ""
    if backtest_results:
        try:
            # Most common composite regime among backtested trades
            from collections import Counter
            counts = Counter(getattr(r, "htf_regime", "") for r in backtest_results)
            counts.pop("", None)
            if counts:
                htf_alignment = counts.most_common(1)[0][0]
        except Exception:
            pass

    optimal_target_r = None
    reversal_cliff_r = None
    try:
        from discovery.target_optimizer import optimize_all_regimes
        if backtest_results:
            opt_reports = optimize_all_regimes(backtest_results, min_trades_per_regime=10)
            if best_regime and best_regime in opt_reports:
                rep = opt_reports[best_regime]
                optimal_target_r = rep.optimal_target_r
                reversal_cliff_r = rep.reversal_cliff_r
            elif opt_reports:
                # Fall back to overall best
                rep = max(opt_reports.values(), key=lambda r: r.optimal_profit_factor)
                optimal_target_r = rep.optimal_target_r
                reversal_cliff_r = rep.reversal_cliff_r
    except Exception as e:
        logger.debug("Target optimizer unavailable for draft: %s", e)

    # Strategy YAML + DNA from generation stage
    gen = getattr(report, "generated_strategy", None)
    strategy_yaml = getattr(gen, "yaml_text", "") if gen else ""
    strategy_dna = getattr(gen, "dna_dict", {}) if gen else {}
    strategy_name = getattr(gen, "strategy_name", "") or f"{pattern}_{symbol.lower()}"

    if existing:
        existing.sample_size = int(agg.get("total_trades", existing.sample_size) or existing.sample_size)
        existing.win_rate = float(agg.get("win_rate", existing.win_rate) or existing.win_rate)
        existing.avg_r = float(agg.get("avg_r", existing.avg_r) or existing.avg_r)
        existing.profit_factor = float(agg.get("profit_factor", existing.profit_factor or 0)) or existing.profit_factor
        existing.best_regime = best_regime or existing.best_regime
        existing.best_time_slot = best_time_slot or existing.best_time_slot
        existing.htf_alignment = htf_alignment or existing.htf_alignment
        existing.optimal_target_r = optimal_target_r or existing.optimal_target_r
        existing.reversal_cliff_r = reversal_cliff_r if reversal_cliff_r is not None else existing.reversal_cliff_r
        existing.strategy_yaml = strategy_yaml or existing.strategy_yaml
        existing.strategy_dna = strategy_dna or existing.strategy_dna
        existing.updated_at = datetime.now()
        return existing.id

    row = DiscoveryDraft(
        id=draft_id,
        task_id=task_id,
        trader_id=trader_id,
        discovery_id=getattr(report, "discovery_id", None),
        symbol=symbol,
        pattern_type=pattern,
        sample_size=int(agg.get("total_trades", 0) or 0),
        win_rate=float(agg.get("win_rate", 0.0) or 0.0),
        avg_r=float(agg.get("avg_r", 0.0) or 0.0),
        profit_factor=float(agg.get("profit_factor", 0.0) or 0.0) or None,
        best_regime=best_regime or None,
        best_time_slot=best_time_slot or None,
        htf_alignment=htf_alignment or None,
        optimal_target_r=optimal_target_r,
        reversal_cliff_r=reversal_cliff_r,
        strategy_name=strategy_name,
        strategy_yaml=strategy_yaml,
        strategy_dna=strategy_dna,
        status="pending_review",
    )
    db.add(row)
    return row.id


# ── Continuous research task ───────────────────────────────────────

class ContinuousResearchTask:
    """
    Long-running background task wrapping the existing discovery pipeline.

    Lifecycle:
      created → started → (looping over candidates) → stopped/completed/failed

    Public API:
      task = ContinuousResearchTask(...)
      asyncio.create_task(task.run())
      task.stop()                       # graceful — finishes current candidate
      task.summary()                    # dict snapshot for status endpoint
    """

    def __init__(
        self,
        task_id: str,
        trader_id: Optional[str],
        scope: str,
        theme: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        pattern_types: Optional[List[str]] = None,
        lookback_years: int = 2,
        similarity_threshold: float = 0.85,
        thresholds: Optional[ValidationThresholds] = None,
        db_factory: Optional[Callable[[], Any]] = None,
        orchestrator: Any = None,
        timeframe: str = DEFAULT_TIMEFRAME,
        max_candidates: int = 200,
        on_status_change: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.task_id = task_id
        self.trader_id = trader_id
        self.scope = scope
        self.theme = theme
        self.symbols = symbols or expand_scope_to_symbols(scope)
        self.pattern_types = pattern_types or list(DEFAULT_PATTERN_TYPES)
        self.lookback_years = lookback_years
        self.similarity_threshold = similarity_threshold
        self.thresholds = thresholds or ValidationThresholds()
        self.db_factory = db_factory
        self._orchestrator = orchestrator
        self.timeframe = timeframe
        self.max_candidates = max_candidates
        self._on_status_change = on_status_change

        self._stop_event = asyncio.Event()
        self._candidates: List[Tuple[str, str, str]] = []
        self._completed = 0
        self._validated = 0
        self._current: Optional[str] = None
        self._error_message: Optional[str] = None
        self._status: str = "pending"     # pending|running|stopped|completed|failed

    # ── Public controls ──

    def stop(self) -> None:
        """Request graceful stop — finishes current candidate then exits."""
        self._stop_event.set()
        logger.info("Task %s: stop requested", self.task_id)

    def summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self._status,
            "candidates_planned": len(self._candidates),
            "candidates_completed": self._completed,
            "candidates_validated": self._validated,
            "current_candidate": self._current,
            "error_message": self._error_message,
            "scope": self.scope,
            "theme": self.theme,
            "symbols": self.symbols,
            "pattern_types": self.pattern_types,
        }

    # ── Main loop ──

    async def run(self) -> None:
        """Run the task to completion (or until stopped)."""
        try:
            self._set_status("running", started_at=datetime.now())
            self._candidates = enumerate_candidates(
                self.symbols, self.pattern_types,
                timeframe=self.timeframe, max_candidates=self.max_candidates,
            )
            logger.info(
                "Task %s started: %d candidates (%d symbols × %d patterns, capped at %d)",
                self.task_id, len(self._candidates),
                len(self.symbols), len(self.pattern_types), self.max_candidates,
            )
            self._persist_progress()

            for symbol, pattern, raw_input in self._candidates:
                if self._stop_event.is_set():
                    break

                self._current = f"{symbol} | {pattern}"
                logger.info("Task %s candidate: %s", self.task_id, self._current)
                self._persist_progress()

                try:
                    await self._run_one_candidate(symbol, pattern, raw_input)
                except Exception as e:
                    logger.warning("Task %s candidate %s failed: %s", self.task_id, self._current, e)
                finally:
                    self._completed += 1
                    self._persist_progress()

                # Polite pause between candidates
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=INTER_CANDIDATE_PAUSE_SECONDS)
                except asyncio.TimeoutError:
                    pass

            if self._stop_event.is_set():
                self._set_status("stopped", stopped_at=datetime.now())
                logger.info("Task %s stopped after %d/%d candidates (%d validated)",
                            self.task_id, self._completed, len(self._candidates), self._validated)
            else:
                self._set_status("completed", stopped_at=datetime.now())
                logger.info("Task %s completed: %d candidates tested, %d validated",
                            self.task_id, self._completed, self._validated)

        except Exception as e:
            self._error_message = str(e)
            self._set_status("failed", stopped_at=datetime.now())
            logger.error("Task %s failed: %s", self.task_id, e, exc_info=True)

    async def _run_one_candidate(
        self,
        symbol: str,
        pattern: str,
        raw_input: str,
    ) -> None:
        """Run discovery on a single candidate; persist as draft if validated."""
        if not self._orchestrator:
            from discovery.orchestrator import get_discovery_orchestrator
            self._orchestrator = get_discovery_orchestrator()

        # Apply task-level overrides via scan_config + backtest_config
        from discovery.similarity_scanner import ScanConfig
        scan_cfg = ScanConfig(
            similarity_threshold=self.similarity_threshold,
            lookback_years=self.lookback_years,
            scope=self.scope,
        )

        try:
            report = await self._orchestrator.discover(
                raw_input=raw_input,
                scan_config=scan_cfg,
            )
        except Exception as e:
            logger.debug("Discovery failed for %s/%s: %s", symbol, pattern, e)
            return

        # Validate
        if not _validate_report(report, self.thresholds):
            logger.debug("Candidate %s/%s did not validate (below thresholds)", symbol, pattern)
            return

        # Persist as draft
        if self.db_factory is None:
            logger.warning("Task %s: no db_factory — cannot persist draft", self.task_id)
            return

        db = self.db_factory()
        try:
            draft_id = _persist_draft(
                db, self.task_id, self.trader_id,
                symbol, pattern, report,
            )
            db.commit()
            if draft_id:
                self._validated += 1
                logger.info("Task %s validated: %s/%s → draft %s",
                            self.task_id, symbol, pattern, draft_id)
        except Exception as e:
            logger.warning("Failed to persist draft for %s/%s: %s", symbol, pattern, e)
            db.rollback()
        finally:
            try:
                db.close()
            except Exception:
                pass

    # ── DB persistence of progress ──

    def _set_status(self, status: str, started_at: Optional[datetime] = None,
                    stopped_at: Optional[datetime] = None) -> None:
        self._status = status
        self._persist_progress(started_at=started_at, stopped_at=stopped_at)
        if self._on_status_change:
            try:
                self._on_status_change(status, self.summary())
            except Exception:
                pass

    def _persist_progress(
        self,
        started_at: Optional[datetime] = None,
        stopped_at: Optional[datetime] = None,
    ) -> None:
        """Update the research_tasks row with current progress."""
        if self.db_factory is None:
            return
        from core.models import ResearchTask

        db = self.db_factory()
        try:
            row = db.query(ResearchTask).filter_by(id=self.task_id).first()
            if not row:
                # First call — insert
                row = ResearchTask(
                    id=self.task_id,
                    trader_id=self.trader_id,
                    theme=self.theme,
                    scope=self.scope,
                    symbols=list(self.symbols),
                    pattern_types=list(self.pattern_types),
                    lookback_years=self.lookback_years,
                    similarity_threshold=self.similarity_threshold,
                    min_sample_size=self.thresholds.min_sample_size,
                    min_win_rate=self.thresholds.min_win_rate,
                    min_avg_r=self.thresholds.min_avg_r,
                    status=self._status,
                )
                db.add(row)

            row.status = self._status
            row.candidates_planned = len(self._candidates)
            row.candidates_completed = self._completed
            row.candidates_validated = self._validated
            row.current_candidate = self._current
            row.error_message = self._error_message
            if started_at:
                row.started_at = started_at
            if stopped_at:
                row.stopped_at = stopped_at
            db.commit()
        except Exception as e:
            logger.debug("Failed to persist progress for task %s: %s", self.task_id, e)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            try:
                db.close()
            except Exception:
                pass


# ── Task manager (in-memory registry) ──────────────────────────────

class ResearchTaskManager:
    """
    In-memory registry of running ContinuousResearchTask objects.

    Survives across API requests within the same backend process. On restart,
    in-flight tasks are marked 'stopped' (since the asyncio task is gone).
    """

    def __init__(self, db_factory: Optional[Callable[[], Any]] = None):
        self.db_factory = db_factory
        self._tasks: Dict[str, ContinuousResearchTask] = {}
        self._asyncio_tasks: Dict[str, asyncio.Task] = {}

    def start_task(
        self,
        scope: str,
        theme: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        pattern_types: Optional[List[str]] = None,
        lookback_years: int = 2,
        similarity_threshold: float = 0.85,
        thresholds: Optional[ValidationThresholds] = None,
        trader_id: Optional[str] = None,
        max_candidates: int = 200,
    ) -> ContinuousResearchTask:
        """Create + spawn a new continuous research task."""
        task_id = f"rtask_{uuid.uuid4().hex[:12]}"
        task = ContinuousResearchTask(
            task_id=task_id,
            trader_id=trader_id,
            scope=scope,
            theme=theme,
            symbols=symbols,
            pattern_types=pattern_types,
            lookback_years=lookback_years,
            similarity_threshold=similarity_threshold,
            thresholds=thresholds,
            db_factory=self.db_factory,
            max_candidates=max_candidates,
        )
        self._tasks[task_id] = task

        # Spawn the background asyncio task
        try:
            loop = asyncio.get_event_loop()
            self._asyncio_tasks[task_id] = loop.create_task(task.run())
        except RuntimeError:
            # No running loop (synchronous caller) — schedule on a fresh loop
            logger.error("No event loop available to spawn task %s — must be called from async context", task_id)
            raise

        return task

    def stop_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.stop()
        return True

    def get_task(self, task_id: str) -> Optional[ContinuousResearchTask]:
        return self._tasks.get(task_id)

    def list_running(self) -> List[ContinuousResearchTask]:
        return [t for t in self._tasks.values() if t._status == "running"]


# ── Module singleton ───────────────────────────────────────────────

_manager_singleton: Optional[ResearchTaskManager] = None


def get_task_manager(db_factory: Optional[Callable[[], Any]] = None) -> ResearchTaskManager:
    """Module-level singleton."""
    global _manager_singleton
    if _manager_singleton is None:
        _manager_singleton = ResearchTaskManager(db_factory=db_factory)
    elif db_factory is not None and _manager_singleton.db_factory is None:
        _manager_singleton.db_factory = db_factory
    return _manager_singleton
