"""
Research Agent Scheduler
=========================

Coordinates the nightly research run:
  1. Loads all active strategies
  2. Runs decay detector on each (Phase 5A)
  3. Runs emergence detector on each (Phase 5B)
  4. Persists findings + computes cross-correlation meta-findings (aggregator)
  5. (Future, when Phase 4 explorer is fully built) runs discovery exploration

Real-money safety:
  - Findings are written with status='pending_review'
  - NEVER auto-applies anything to live strategies
  - Kill switch: env var DISABLE_AUTO_RESEARCH=true halts the run

Usage:
    # Manual one-shot (recommended for first deploy)
    python -m research_agent.scheduler --once

    # Background loop (call from worker.py during normal operation)
    await ResearchAgentScheduler(db_factory).run_loop()
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from research_agent.aggregator import aggregate_and_persist
from research_agent.decay_detector import DecayDetector
from research_agent.emergence_detector import EmergenceDetector
from research_agent.models import DecayFinding, EmergenceFinding

logger = logging.getLogger(__name__)


# ── Defaults (configurable via env) ────────────────────────────────

DEFAULT_RUN_INTERVAL_HOURS: int = 24      # nightly run
DEFAULT_RUN_HOUR_IST: int = 23            # 11 PM IST
KILL_SWITCH_ENV: str = "DISABLE_AUTO_RESEARCH"


# ── Default loaders (callers can override) ─────────────────────────

def default_strategy_id_loader(db) -> List[str]:
    """Get all active strategy IDs from the registry."""
    try:
        from strategy_engine.strategy_registry import get_registry
        reg = get_registry()
        if reg and reg.is_initialized:
            return [s.id for s in reg.get_all_active() if s.id]
    except Exception as e:
        logger.warning("Could not load strategy IDs from registry: %s", e)
    return []


def default_r_history_loader_factory(db_factory: Callable[[], Any]) -> Callable[[str], List[float]]:
    """
    Returns a closure that loads R-multiple history from the trades table for a strategy.
    Trades must have an `r_multiple` column and a `strategy_id` filter.
    """
    def _load(strategy_id: str) -> List[float]:
        from sqlalchemy import select
        try:
            from core.models import Trade
        except ImportError:
            return []

        db = db_factory()
        try:
            stmt = (
                select(Trade.r_multiple)
                .where(Trade.strategy_id == strategy_id)
                .where(Trade.r_multiple.isnot(None))
                .order_by(Trade.entry_time.asc())
            )
            try:
                rows = db.execute(stmt).scalars().all()
            except Exception:
                # Fallback: maybe attribution table is the link
                return []
            return [float(r) for r in rows if r is not None]
        finally:
            try:
                db.close()
            except Exception:
                pass

    return _load


# ── Scheduler ──────────────────────────────────────────────────────

class ResearchAgentScheduler:
    """
    Orchestrator for the research agent's periodic runs.

    Args:
        db_factory: callable returning a NEW SQLAlchemy session per run
        strategy_id_loader: callable(db) -> list of active strategy IDs
        r_history_loader: callable(strategy_id) -> list of R-multiples
        run_interval_hours: how often to run (default: nightly = 24h)
        emergence_loaders: dict with keys 'trade_loader', 'regime_freq_loader',
                          'strategy_regime_loader' for EmergenceDetector
                          (optional — emergence is skipped if not provided)
    """

    def __init__(
        self,
        db_factory: Callable[[], Any],
        strategy_id_loader: Optional[Callable[[Any], List[str]]] = None,
        r_history_loader: Optional[Callable[[str], List[float]]] = None,
        run_interval_hours: int = DEFAULT_RUN_INTERVAL_HOURS,
        emergence_loaders: Optional[Dict[str, Callable]] = None,
    ):
        self.db_factory = db_factory
        self._strat_id_loader = strategy_id_loader or default_strategy_id_loader
        self._r_loader = r_history_loader or default_r_history_loader_factory(db_factory)
        self.run_interval_hours = run_interval_hours
        self._emergence_loaders = emergence_loaders or {}
        self._stop_event: Optional[asyncio.Event] = None
        self._last_run_at: Optional[datetime] = None

    @staticmethod
    def kill_switch_active() -> bool:
        return os.getenv(KILL_SWITCH_ENV, "").lower() in ("true", "1", "yes")

    def run_once(self) -> Dict[str, Any]:
        """
        Single synchronous research run. Idempotent — safe to call manually.
        Returns summary dict.
        """
        if self.kill_switch_active():
            logger.warning("Research agent kill switch active (env %s). Skipping run.", KILL_SWITCH_ENV)
            return {"skipped": True, "reason": "kill_switch_active"}

        run_started = datetime.now()
        logger.info("Research agent run starting at %s", run_started.isoformat())

        # 1. Load active strategies
        db = self.db_factory()
        try:
            strategy_ids = self._strat_id_loader(db)
        finally:
            try:
                db.close()
            except Exception:
                pass

        if not strategy_ids:
            logger.info("No active strategies — research run is a no-op")
            return {"skipped": True, "reason": "no_active_strategies"}

        logger.info("Research agent: analyzing %d strategies: %s", len(strategy_ids), strategy_ids[:5])

        # 2. Phase 5A — decay detection
        decay_detector = DecayDetector(trade_loader=self._r_loader)
        decay_findings: List[DecayFinding] = decay_detector.detect_all(strategy_ids)
        logger.info("Decay findings: %d", len(decay_findings))

        # 3. Phase 5B — emergence detection (if loaders provided)
        emergence_findings: List[EmergenceFinding] = []
        if all(k in self._emergence_loaders for k in ("trade_loader", "regime_freq_loader", "strategy_regime_loader")):
            try:
                emergence_detector = EmergenceDetector(
                    trade_loader=self._emergence_loaders["trade_loader"],
                    regime_freq_loader=self._emergence_loaders["regime_freq_loader"],
                    strategy_regime_loader=self._emergence_loaders["strategy_regime_loader"],
                )
                emergence_findings = emergence_detector.detect_all(strategy_ids)
                logger.info("Emergence findings: %d", len(emergence_findings))
            except Exception as e:
                logger.warning("Emergence detection failed: %s", e)
        else:
            logger.debug("Skipping emergence detection — loaders not configured")

        # 4. Persist + cross-correlate
        db = self.db_factory()
        persist_result: Dict[str, Any] = {}
        try:
            persist_result = aggregate_and_persist(
                db, decay_findings, emergence_findings, commit=True,
            )
        except Exception as e:
            logger.error("Aggregator failed: %s", e, exc_info=True)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            try:
                db.close()
            except Exception:
                pass

        run_ended = datetime.now()
        self._last_run_at = run_ended

        summary = {
            "started_at": run_started.isoformat(),
            "ended_at": run_ended.isoformat(),
            "duration_seconds": (run_ended - run_started).total_seconds(),
            "n_strategies_analyzed": len(strategy_ids),
            "n_decay_findings": len(decay_findings),
            "n_emergence_findings": len(emergence_findings),
            "n_meta_findings": persist_result.get("n_meta", 0),
            "persist_result": persist_result,
        }
        logger.info(
            "Research agent run complete: %d decay, %d emergence, %d meta findings (%.1fs)",
            summary["n_decay_findings"], summary["n_emergence_findings"],
            summary["n_meta_findings"], summary["duration_seconds"],
        )
        return summary

    async def run_loop(self, stop_event: Optional[asyncio.Event] = None) -> None:
        """
        Long-running background loop. Runs once every `run_interval_hours`.
        Call from worker.py to keep research running while the system is up.
        """
        self._stop_event = stop_event or asyncio.Event()
        sleep_seconds = self.run_interval_hours * 3600

        logger.info(
            "ResearchAgentScheduler started — running every %d hours",
            self.run_interval_hours,
        )

        while not self._stop_event.is_set():
            try:
                # Run synchronously inside the async loop (research is mostly DB + math, not IO)
                self.run_once()
            except Exception as e:
                logger.error("Research run failed: %s", e, exc_info=True)

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_seconds)
            except asyncio.TimeoutError:
                continue  # normal — time to run again

        logger.info("ResearchAgentScheduler stopped")


# ── Manual CLI entry point ─────────────────────────────────────────

def main() -> None:
    """CLI: `python -m research_agent.scheduler --once`"""
    import argparse

    parser = argparse.ArgumentParser(description="Research agent CLI")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    )

    # Build a sync DB factory from core.database
    try:
        from core.database import SessionLocal
        db_factory = SessionLocal
    except ImportError:
        logger.error("core.database.SessionLocal not available — cannot run")
        return

    scheduler = ResearchAgentScheduler(db_factory=db_factory)

    if args.once:
        result = scheduler.run_once()
        print()
        print("=== Run Summary ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        # No --once: explain how to use
        print("Use --once for a manual run.")
        print("For background scheduling, integrate with worker.py")


if __name__ == "__main__":
    main()
