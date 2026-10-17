# avt/integrations.py
"""
AVT Integration Layer
======================

Wires the in-memory AVT engine to:
  1. Database persistence (repository layer)
  2. Event bus (trade opened, closed, scan cycle, improvements)
  3. WebSocket broadcast (real-time push to frontend)
  4. Background scheduling (self-improver daily, eval weekly, competition daily)

The engine itself stays clean — this module hooks into its callbacks
and lifecycle events to persist and broadcast state changes.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional

from avt.models import ActiveImprovement, AVTConfig, ScanCycleLog, VirtualTrade

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  DB PERSISTENCE HOOKS
# ══════════════════════════════════════════════════════════════════

class AVTPersistenceHook:
    """
    Hooks into engine callbacks to persist trades and scan logs to DB.

    Attached to engine._on_trade_opened and _on_trade_closed callbacks.
    Also called after each scan cycle to log it.
    """

    def __init__(self, db_factory):
        """
        Args:
            db_factory: callable that yields a DB session (e.g., get_db or SessionLocal)
        """
        self._db_factory = db_factory

    def on_trade_opened(self, trade: VirtualTrade) -> None:
        """Persist a newly opened virtual trade."""
        try:
            db = self._get_db()
            try:
                from avt.repository import AVTTradeRepository, AVTVersionRepository
                AVTTradeRepository(db).save(trade)
                AVTVersionRepository(db).increment_trade_count(trade.strategy_id)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to persist trade open %s: %s", trade.id, e)

    def on_trade_closed(self, trade: VirtualTrade) -> None:
        """Update the trade row with exit data."""
        try:
            db = self._get_db()
            try:
                from avt.repository import AVTTradeRepository
                AVTTradeRepository(db).save(trade)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to persist trade close %s: %s", trade.id, e)

    def on_scan_cycle(self, log: ScanCycleLog, strategy_id=None) -> None:
        """Persist a scan cycle log."""
        try:
            db = self._get_db()
            try:
                from avt.repository import AVTScanLogRepository
                AVTScanLogRepository(db).save(log, strategy_id=strategy_id)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to persist scan log: %s", e)

    def on_improvement_changed(self, improvement: ActiveImprovement) -> None:
        """Persist improvement state change (applied, proven, reverted)."""
        try:
            db = self._get_db()
            try:
                from avt.repository import AVTImprovementRepository
                AVTImprovementRepository(db).save(improvement)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to persist improvement %s: %s", improvement.id, e)

    def on_version_bumped(self, strategy_id, version: str, improvement_ids: List) -> None:
        """Record a version bump."""
        try:
            db = self._get_db()
            try:
                from avt.repository import AVTVersionRepository
                AVTVersionRepository(db).save_version(
                    strategy_id=strategy_id,
                    version=version,
                    improvement_ids=improvement_ids,
                )
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to persist version bump: %s", e)

    def _get_db(self):
        """Get a fresh DB session."""
        if callable(self._db_factory):
            db = self._db_factory()
            # If it's a generator (from get_db), advance it
            if hasattr(db, "__next__"):
                return next(db)
            return db
        return self._db_factory


# ══════════════════════════════════════════════════════════════════
#  EVENT BUS HOOKS
# ══════════════════════════════════════════════════════════════════

class AVTEventHook:
    """
    Publishes AVT events to the core event bus.
    These events are picked up by the WebSocket handler for real-time push.
    """

    def __init__(self):
        try:
            from core.events import get_event_bus
            self._bus = get_event_bus()
        except Exception:
            self._bus = None

    def on_trade_opened(self, trade: VirtualTrade) -> None:
        self._publish("AVT_TRADE_OPENED", data={
            "trade_id": str(trade.id),
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "quantity": trade.quantity,
            "stop_price": trade.stop_price,
            "conviction": trade.conviction_at_entry,
            "strategy_id": str(trade.strategy_id),
            "strategy_version": trade.strategy_version,
            "rank": trade.ranking_position,
        })

    def on_trade_closed(self, trade: VirtualTrade) -> None:
        self._publish("AVT_TRADE_CLOSED", data={
            "trade_id": str(trade.id),
            "symbol": trade.symbol,
            "direction": trade.direction,
            "pnl_r": trade.pnl_r,
            "pnl_absolute": trade.pnl_absolute,
            "exit_type": trade.exit_type,
            "exit_reason": trade.exit_reason,
            "strategy_id": str(trade.strategy_id),
        })

    def on_scan_cycle(self, log: ScanCycleLog) -> None:
        self._publish("AVT_SCAN_CYCLE", data={
            "stocks_scanned": log.stocks_scanned,
            "candidates_found": log.candidates_found,
            "executed": log.executed_count,
            "exits": log.exits_triggered,
            "duration_ms": log.duration_ms,
            "errors": log.errors[:5],  # cap to avoid bloat
        })

    def on_improvement_applied(self, imp: ActiveImprovement) -> None:
        self._publish("AVT_IMPROVEMENT_APPLIED", data={
            "improvement_id": str(imp.id),
            "description": imp.description,
            "type": imp.type,
            "strategy_id": str(imp.strategy_id),
        })

    def on_improvement_proven(self, imp: ActiveImprovement) -> None:
        self._publish("AVT_IMPROVEMENT_PROVEN", data={
            "improvement_id": str(imp.id),
            "description": imp.description,
            "type": imp.type,
            "evidence_strength": imp.evidence_strength,
            "strategy_id": str(imp.strategy_id),
        })

    def on_improvement_reverted(self, imp: ActiveImprovement) -> None:
        self._publish("AVT_IMPROVEMENT_REVERTED", data={
            "improvement_id": str(imp.id),
            "description": imp.description,
            "revert_reason": imp.revert_reason,
            "strategy_id": str(imp.strategy_id),
        })

    def on_started(self, strategies: List[str]) -> None:
        self._publish("AVT_STARTED", data={"strategies": strategies})

    def on_stopped(self, cycle_count: int) -> None:
        self._publish("AVT_STOPPED", data={"cycles_completed": cycle_count})

    def _publish(self, event_type_name: str, data: Dict[str, Any]) -> None:
        if not self._bus:
            return
        try:
            from core.events import Event, EventType
            et = getattr(EventType, event_type_name, None)
            if et:
                self._bus.publish(Event(
                    event_type=et,
                    source="avt.integrations",
                    data=data,
                ))
        except Exception as e:
            logger.debug("AVT event publish failed (%s): %s", event_type_name, e)


# ══════════════════════════════════════════════════════════════════
#  WEBSOCKET WIRING
# ══════════════════════════════════════════════════════════════════

def wire_avt_to_websocket():
    """
    Subscribe AVT events to the session WebSocket channel.
    Called once during app startup after event handlers are wired.
    """
    try:
        from core.events import Event, EventType, get_event_bus
        from api.routes.websocket import manager

        bus = get_event_bus()

        avt_event_types = [
            EventType.AVT_TRADE_OPENED,
            EventType.AVT_TRADE_CLOSED,
            EventType.AVT_SCAN_CYCLE,
            EventType.AVT_IMPROVEMENT_APPLIED,
            EventType.AVT_IMPROVEMENT_PROVEN,
            EventType.AVT_IMPROVEMENT_REVERTED,
            EventType.AVT_STARTED,
            EventType.AVT_STOPPED,
        ]

        async def _broadcast_avt_event(event_data: dict):
            await manager.broadcast("session_live", event_data)

        def on_avt_event(event: Event):
            event_data = {
                "type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data,
            }
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(_broadcast_avt_event(event_data))
            except RuntimeError:
                pass

        for et in avt_event_types:
            bus.subscribe(et, on_avt_event)

        logger.info("AVT events wired to WebSocket (session_live channel)")

    except Exception as e:
        logger.warning("Failed to wire AVT to WebSocket: %s", e)


# ══════════════════════════════════════════════════════════════════
#  ENGINE WIRING — attach all hooks to engine + self-improver
# ══════════════════════════════════════════════════════════════════

def _feed_breakdown_hotness(engine, trade) -> None:
    """On a BREAKDOWN trade close, update the LIVE ice-cold selector (`HotnessTracker` last-5 realized-R) + persist it,
    so the deployed skip-ice-cold gate advances across the session/days. Updates the SAME tracker instance the
    breakdown generator reads. No-op for non-breakdown trades / any error (never blocks the close path)."""
    try:
        if not str(getattr(trade, "strategy_id", "")).upper().startswith("BREAKDOWN"):
            return
        bd = getattr(getattr(engine, "signal_gen", None), "breakdown", None)
        ht = getattr(bd, "hotness", None)
        sym = getattr(trade, "symbol", None); r = getattr(trade, "r_multiple", None)
        if ht is None or not sym or r is None:
            return
        ht.on_trade_close(sym, float(r)); ht.save()
    except Exception:
        pass


def wire_avt_engine(avt_engine, self_improver=None, db_factory=None):
    """
    Attach persistence + event hooks to the AVT engine and self-improver.

    Call this once after init_avt() during app startup.
    """
    persistence = AVTPersistenceHook(db_factory) if db_factory else None
    events = AVTEventHook()

    # ── Engine callbacks ──

    original_open = avt_engine._on_trade_opened
    original_close = avt_engine._on_trade_closed

    def on_trade_opened(trade: VirtualTrade):
        if persistence:
            persistence.on_trade_opened(trade)
        events.on_trade_opened(trade)
        if original_open:
            original_open(trade)

    def on_trade_closed(trade: VirtualTrade):
        if persistence:
            persistence.on_trade_closed(trade)
        events.on_trade_closed(trade)
        _feed_breakdown_hotness(avt_engine, trade)   # advance the deployed skip-ice-cold selector on breakdown closes
        if original_close:
            original_close(trade)

    avt_engine._on_trade_opened = on_trade_opened
    avt_engine._on_trade_closed = on_trade_closed

    # ── Wrap scan cycle to persist log + publish event ──

    _original_run_scan = avt_engine.run_scan_cycle

    async def wrapped_run_scan_cycle():
        log = await _original_run_scan()
        if persistence:
            persistence.on_scan_cycle(log)
        events.on_scan_cycle(log)
        return log

    avt_engine.run_scan_cycle = wrapped_run_scan_cycle

    # ── Wrap start/stop ──

    _original_run = avt_engine.run

    async def wrapped_run(stop_event=None):
        avt_engine._started_at = datetime.now(timezone.utc)
        events.on_started(list(avt_engine.strategy_versions.keys()))
        try:
            await _original_run(stop_event=stop_event)
        finally:
            events.on_stopped(avt_engine._cycle_count)

    avt_engine.run = wrapped_run

    # ── Self-improver callbacks ──

    if self_improver:
        def on_imp_applied(imp: ActiveImprovement):
            if persistence:
                persistence.on_improvement_changed(imp)
            events.on_improvement_applied(imp)

        def on_imp_proven(imp: ActiveImprovement):
            if persistence:
                persistence.on_improvement_changed(imp)
            events.on_improvement_proven(imp)

        def on_imp_reverted(imp: ActiveImprovement):
            if persistence:
                persistence.on_improvement_changed(imp)
            events.on_improvement_reverted(imp)

        self_improver._on_improvement_applied = on_imp_applied
        self_improver._on_improvement_proven = on_imp_proven
        self_improver._on_improvement_reverted = on_imp_reverted

    # ── Wire calibration DB persistence into lifecycle manager ──

    lm = getattr(avt_engine, "_lifecycle_manager", None)
    if lm is not None and db_factory is not None:
        lm._calibration_db_factory = db_factory
        logger.info("Calibration persistence wired to lifecycle manager")

    logger.info("AVT engine wired: persistence=%s, events=True", persistence is not None)


# ══════════════════════════════════════════════════════════════════
#  STATE RELOAD FROM DB
# ══════════════════════════════════════════════════════════════════

def reload_avt_state(avt_engine, db_factory):
    """
    On startup, reload open positions and improvement history from DB
    into the in-memory engine.
    """
    try:
        db = db_factory()
        if hasattr(db, "__next__"):
            db = next(db)

        try:
            from avt.repository import AVTImprovementRepository, AVTTradeRepository

            trade_repo = AVTTradeRepository(db)
            imp_repo = AVTImprovementRepository(db)

            # Reload open positions
            open_trades = trade_repo.get_open_trades()
            for trade in open_trades:
                avt_engine.portfolio.open_positions.append(trade)
                avt_engine.portfolio.trade_count += 1

            # Reload recent closed trades (for self-improver analysis)
            closed_trades = trade_repo.get_closed_trades(limit=500)
            avt_engine.portfolio.closed_trades = closed_trades

            # Update trade count
            avt_engine.portfolio.trade_count += len(closed_trades)

            # Reload active improvements into strategy versions
            for strategy_id, version in avt_engine.strategy_versions.items():
                active_imps = imp_repo.get_by_strategy(
                    strategy_id=strategy_id if hasattr(strategy_id, 'hex') else None,
                    status="active",
                )
                proven_imps = imp_repo.get_by_strategy(
                    strategy_id=strategy_id if hasattr(strategy_id, 'hex') else None,
                    status="proven",
                )
                version.active_improvements = active_imps + proven_imps

            # Restore calibration tracker state from DB
            lm = getattr(avt_engine, "_lifecycle_manager", None)
            calibration_restored = 0
            if lm is not None:
                from avt.repository import AVTCalibrationRepository
                cal_repo = AVTCalibrationRepository(db)
                for strategy_id in avt_engine.strategy_versions:
                    cal = lm.get_calibration_tracker(strategy_id)
                    if cal is not None:
                        n = cal_repo.restore_tracker(strategy_id, cal)
                        calibration_restored += n
                        if n:
                            logger.info(
                                "Calibration restored: strategy=%s, %d records",
                                strategy_id, n,
                            )

            logger.info(
                "AVT state reloaded: %d open, %d closed, %d improvements, %d calibration records",
                len(open_trades), len(closed_trades),
                sum(len(v.active_improvements) for v in avt_engine.strategy_versions.values()),
                calibration_restored,
            )

        finally:
            db.close()

    except Exception as e:
        logger.warning("AVT state reload failed (starting fresh): %s", e)


def checkpoint_avt_state(db_factory) -> None:
    """
    Persist all open virtual positions to DB before shutdown.
    Called from lifespan shutdown to ensure no state is lost.
    """
    try:
        from avt.engine import get_avt
        avt = get_avt()
        if not avt:
            return

        open_positions = avt.portfolio.open_positions
        if not open_positions:
            logger.info("AVT checkpoint: no open positions to save")
            return

        db = db_factory()
        if hasattr(db, "__next__"):
            db = next(db)

        try:
            from avt.repository import AVTTradeRepository
            repo = AVTTradeRepository(db)
            saved = 0
            for trade in open_positions:
                try:
                    repo.save(trade)
                    saved += 1
                except Exception:
                    pass  # already persisted by hook — ignore duplicate
            db.commit()
            logger.info("AVT checkpoint: %d/%d open positions saved", saved, len(open_positions))
        finally:
            db.close()

    except Exception as e:
        logger.warning("AVT checkpoint failed: %s", e)


# ══════════════════════════════════════════════════════════════════
#  BACKGROUND SCHEDULER
# ══════════════════════════════════════════════════════════════════

class AVTScheduler:
    """
    Runs periodic background tasks:
      - Daily (after market close): self-improver analysis
      - Weekly (Friday): improvement evaluation
      - Daily: competition snapshot save
    """

    def __init__(
        self,
        self_improver=None,
        competition_engine=None,
        db_factory=None,
    ):
        self._improver = self_improver
        self._competition = competition_engine
        self._db_factory = db_factory
        self._running = False

    async def run(self, stop_event: Optional[asyncio.Event] = None):
        """Run the scheduler loop. Checks every 5 minutes if a task is due."""
        self._running = True
        logger.info("AVT scheduler started")

        last_daily_date = None
        last_weekly_date = None

        while self._running:
            if stop_event and stop_event.is_set():
                break

            now = datetime.now(timezone.utc)
            today = now.date()

            # IST market close ~15:30 IST = 10:00 UTC
            ist_hour = now.hour + 5 + (1 if now.minute >= 30 else 0)

            # ── Daily tasks (after 16:00 IST = market closed) ──
            if ist_hour >= 16 and last_daily_date != today:
                last_daily_date = today
                await self._run_daily_tasks()

            # ── Weekly tasks (Friday after market close) ──
            # now.weekday() == 4 is Friday
            if ist_hour >= 16 and now.weekday() == 4 and last_weekly_date != today:
                last_weekly_date = today
                await self._run_weekly_tasks()

            # Sleep 5 minutes
            await asyncio.sleep(300)

        logger.info("AVT scheduler stopped")

    def stop(self):
        self._running = False

    async def _run_daily_tasks(self):
        """Post-market daily analysis."""
        # 1. Self-improver analysis
        if self._improver:
            try:
                results = self._improver.daily_analysis()
                logger.info("AVT daily analysis: %s", results)
            except Exception as e:
                logger.error("AVT daily analysis failed: %s", e)

        # 2. Edge amplifier re-learning (updates A+ edges from latest trades)
        try:
            from improvement_lab.edge_amplifier import get_edge_amplifier
            from avt.engine import get_avt

            avt = get_avt()
            edge_amp = get_edge_amplifier()
            if avt and edge_amp:
                for sid in avt.strategies:
                    closed = avt.get_closed_trades(strategy_id=sid)
                    if len(closed) >= 20:
                        profile = edge_amp.learn(closed, sid)
                        logger.info(
                            "Edge amplifier re-learned %s: %d edges (from %d trades)",
                            sid, len(profile.edges) if profile else 0, len(closed),
                        )
        except Exception as e:
            logger.debug("Edge amplifier daily re-learn skipped: %s", e)

        # 3. Competition snapshot
        if self._competition and self._db_factory:
            try:
                await self._save_competition_snapshots()
            except Exception as e:
                logger.error("AVT competition snapshot failed: %s", e)

    async def _run_weekly_tasks(self):
        """
        Friday: run daily_analysis again (which evaluates active proofs).
        The new self_improver handles both discovery and evaluation
        in a single daily_analysis() call.
        """
        if self._improver:
            try:
                results = self._improver.daily_analysis()
                logger.info("AVT weekly eval: %s", results)
            except Exception as e:
                logger.error("AVT weekly evaluation failed: %s", e)

    async def _save_competition_snapshots(self):
        """Save daily competition snapshot for each strategy."""
        from avt.engine import get_avt
        avt = get_avt()
        if not avt:
            return

        db = self._db_factory()
        if hasattr(db, "__next__"):
            db = next(db)

        try:
            from avt.repository import AVTCompetitionRepository

            comp_repo = AVTCompetitionRepository(db)

            for strategy_id in avt.strategy_versions:
                try:
                    report = self._competition.generate_comparison(
                        strategy_id=strategy_id,
                        period_days=30,
                    )
                    # strategy_id might be string, try to cast
                    from uuid import UUID as _UUID
                    sid = strategy_id if isinstance(strategy_id, _UUID) else _UUID(strategy_id)
                    comp_repo.save_snapshot(report, sid)
                except Exception as e:
                    logger.debug("Competition snapshot for %s failed: %s", strategy_id, e)

            db.commit()
        finally:
            db.close()


# ══════════════════════════════════════════════════════════════════
#  STRATEGY + WATCHLIST SOURCING FOR AVT
# ══════════════════════════════════════════════════════════════════

def _load_tradeable_skos(db_factory, trader_id: Any = None) -> List[Dict[str, Any]]:
    """
    Load compiled, tradeable SKOs (non-empty scoring_config) from the DB as
    dicts suitable for SignalGenerator.evaluate. These are the strategies the
    user has taught/onboarded — AVT trades only these.

    trader_id: when provided, load ONLY that user's strategies (multi-user
    correctness). None = all (single-user / legacy global mode).
    """
    out: List[Dict[str, Any]] = []
    try:
        db = db_factory()
        if hasattr(db, "__next__"):
            db = next(db)
        try:
            from core.models import StrategyKnowledgeObject
            from core.enums import SKOStatus

            q = db.query(StrategyKnowledgeObject).filter(
                StrategyKnowledgeObject.status != SKOStatus.ARCHIVED
            )
            if trader_id is not None:
                q = q.filter(StrategyKnowledgeObject.trader_id == trader_id)
            rows = q.all()
            for s in rows:
                if not s.scoring_config:
                    continue  # not compiled → SignalGenerator can't fire
                out.append({
                    "id": str(s.id),
                    "short_code": s.short_code,
                    "name": s.name,
                    "status": s.status.value if hasattr(s.status, "value") else str(s.status),
                    "scoring_config": s.scoring_config or {},
                    "conviction_map": s.conviction_map or {},
                    "entry_zones": s.entry_zones or [],
                    "exit_rules": s.exit_rules or {},
                    "risk_parameters": s.risk_parameters or {},
                    "regime_fingerprint": s.regime_fingerprint or {},
                    "time_preferences": s.time_preferences or {},
                })
        finally:
            db.close()
    except Exception as e:
        logger.warning("Failed to load tradeable SKOs for AVT: %s", e)
    return out


def _build_avt_watchlist(db_factory, trader_id=None, max_symbols: int = 60) -> List[str]:
    """This user's master-watchlist symbols, falling back to the FNO universe."""
    symbols: List[str] = []
    try:
        db = db_factory()
        if hasattr(db, "__next__"):
            db = next(db)
        try:
            from core.models import Watchlist, WatchlistItem
            master_q = db.query(Watchlist).filter(
                Watchlist.watchlist_type == "master",
                Watchlist.is_active == True,  # noqa: E712
            )
            if trader_id is not None:
                master_q = master_q.filter(Watchlist.trader_id == trader_id)
            master = master_q.first()
            if master:
                items = (
                    db.query(WatchlistItem)
                    .filter(
                        WatchlistItem.watchlist_id == master.id,
                        WatchlistItem.status.in_(["active", "matched"]),
                    )
                    .all()
                )
                symbols = [it.symbol for it in items if it.symbol]
        finally:
            db.close()
    except Exception as e:
        logger.debug("AVT master watchlist load failed: %s", e)

    if not symbols:
        try:
            from automation.fno_universe import FNOUniverseFetcher
            f = FNOUniverseFetcher()
            symbols = f.get_symbols() or (f.get_nifty50() + f.get_nifty_next50())
        except Exception as e:
            logger.debug("AVT FNO universe load failed: %s", e)

    deduped: List[str] = []
    for s in symbols:
        if s and s not in deduped:
            deduped.append(s)
    return deduped[:max_symbols]


def _build_strategy_watchlists(db_factory, strategy_dicts) -> Dict[str, List[str]]:
    """
    Map each loaded SKO (by its engine id) → its per-strategy watchlist symbols,
    reusing the SAME strategy watchlists the co-planner/scanner use.

    Join path: SKO.short_code == StrategyProfile.short_code → the strategy
    Watchlist (watchlist_type="strategy") → its active/matched WatchlistItems.
    Strategies without a strategy watchlist are omitted (engine falls back to
    the global watchlist for those).
    """
    out: Dict[str, List[str]] = {}
    if not strategy_dicts:
        return out
    try:
        db = db_factory()
        if hasattr(db, "__next__"):
            db = next(db)
        try:
            from core.models import StrategyProfile, Watchlist, WatchlistItem
            for s in strategy_dicts:
                sid = str(s.get("id", "") or "")
                short_code = s.get("short_code")
                if not sid or not short_code:
                    continue
                profile = (
                    db.query(StrategyProfile)
                    .filter(StrategyProfile.short_code == short_code)
                    .first()
                )
                if not profile:
                    continue
                wl = (
                    db.query(Watchlist)
                    .filter(
                        Watchlist.watchlist_type == "strategy",
                        Watchlist.strategy_profile_id == profile.id,
                    )
                    .first()
                )
                if not wl:
                    continue
                items = (
                    db.query(WatchlistItem)
                    .filter(
                        WatchlistItem.watchlist_id == wl.id,
                        WatchlistItem.status.in_(["active", "matched"]),
                    )
                    .all()
                )
                syms = [it.symbol for it in items if it.symbol]
                if syms:
                    out[sid] = syms
        finally:
            db.close()
    except Exception as e:
        logger.warning("Failed to build per-strategy watchlists: %s", e)
    return out


def _wire_sko_hot_reload(avt_engine, db_factory) -> None:
    """Hot-load newly compiled SKOs into the running AVT engine (no restart)."""
    try:
        from core.events import Event, EventType, get_event_bus
        from avt.models import StrategyVersion

        bus = get_event_bus()

        def on_sko_created(event: Event):
            try:
                if not str(event.data.get("sko_id", "")):
                    return
                # Only react to SKOs owned by THIS engine's user (multi-user).
                evt_trader = event.data.get("trader_id")
                if (avt_engine.trader_id is not None and evt_trader is not None
                        and str(evt_trader) != str(avt_engine.trader_id)):
                    return
                # Load only this engine owner's strategies.
                for s in _load_tradeable_skos(db_factory, avt_engine.trader_id):
                    sid = str(s.get("id"))
                    if sid in avt_engine.strategies:
                        continue
                    avt_engine.strategies[sid] = s
                    avt_engine.strategy_versions[sid] = StrategyVersion(
                        base_sko=s,
                        version="1.0",
                        version_start_date=datetime.now(timezone.utc),
                    )
                    lm = getattr(avt_engine, "_lifecycle_manager", None)
                    if lm is not None:
                        try:
                            lm.ingest_strategy(s)
                        except Exception:
                            pass
                    logger.info(
                        "AVT hot-loaded new strategy %s (%s)",
                        s.get("short_code"), sid,
                    )
            except Exception as e:
                logger.debug("SKO hot-reload handler failed: %s", e)

        bus.subscribe(EventType.SKO_CREATED, on_sko_created)
        logger.info("AVT subscribed to SKO_CREATED for hot-reload")
    except Exception as e:
        logger.debug("Failed to wire SKO hot-reload: %s", e)


# ══════════════════════════════════════════════════════════════════
#  FULL STARTUP WIRING — called from api/main.py lifespan
# ══════════════════════════════════════════════════════════════════

# ── Per-user lazy setup ─────────────────────────────────────────
# AVT engines are created lazily per user when they hit POST /avt/start. The app
# stores the shared deps (db factory + market data) at startup so the route can
# build a user's engine on demand.

_AVT_DB_FACTORY = None
_AVT_MARKET_DATA = None


def set_avt_deps(db_factory, market_data=None) -> None:
    """Store shared deps at app startup for lazy per-user engine creation."""
    global _AVT_DB_FACTORY, _AVT_MARKET_DATA
    _AVT_DB_FACTORY = db_factory
    _AVT_MARKET_DATA = market_data


def setup_avt_for_user(trader_id: str) -> Any:
    """
    Lazily create + fully wire a user's AVT engine (idempotent — returns the
    existing engine if already set up). Returns the engine, or None if deps
    aren't available.
    """
    from avt.engine import get_avt
    existing = get_avt(str(trader_id))
    if existing is not None:
        return existing
    if _AVT_DB_FACTORY is None:
        logger.warning("AVT deps not set — cannot create engine for trader %s", trader_id)
        return None
    setup_avt_full(_AVT_DB_FACTORY, market_data=_AVT_MARKET_DATA, trader_id=str(trader_id))
    return get_avt(str(trader_id))


def setup_avt_full(db_factory, market_data=None, trader_id=None) -> Optional[asyncio.Task]:
    """
    Full AVT setup for ONE user: init their engine, wire hooks, reload state,
    run health gate per strategy, start scheduler. Returns the scheduler task.

    trader_id: the engine's owner — its SKOs, watchlist and virtual trades are
    all scoped to this user. None = legacy single-user/global mode.

    Called lazily from POST /api/v1/avt/start (per-user) — see setup_avt_for_user.
    """
    try:
        from avt.engine import get_avt, init_avt
        from ske.signal_generator import SignalGenerator
        from avt.stock_ranker import StockSelectionRanker

        # Load ONLY this user's tradeable, compiled strategies (non-empty
        # scoring_config). AVT trades only strategies the user taught/onboarded.
        # May be empty until they finalize onboarding (health banner reports it).
        strategy_dicts = _load_tradeable_skos(db_factory, trader_id)

        # Build the scan watchlist (this user's master watchlist → FNO fallback)
        watchlist = _build_avt_watchlist(db_factory, trader_id)

        # Inject the user's ACTIVATED system templates (cascade, …) alongside their taught strategies. These are
        # the system's OWN strategies (no scoring_config → not in the taught DB); selection lives in template_store.
        from avt.template_store import activated_skos
        template_dicts = activated_skos(trader_id, watchlist)
        all_strategies = strategy_dicts + template_dicts

        # Signal-gen: with any template active, route per-SKO (cascade → CascadeSignalGenerator, self-priming live;
        # taught → SignalGenerator). Without templates, the plain taught generator (default engine unchanged).
        if template_dicts:
            from avt.dispatch_signal import DispatchingSignalGenerator
            sig_gen = DispatchingSignalGenerator(taught=SignalGenerator(), live_prime=True)
        else:
            sig_gen = SignalGenerator()

        # Capital sleeves (B.1): this user's account split (₹2L futures / ₹50k options / rest equity …), if they
        # configured one — feeds AVTConfig so strategies run concurrently off separate pools. Unset → no config →
        # the engine's shared pool (byte-identical default).
        cfg: Optional[AVTConfig] = None
        try:
            from avt.sleeve_store import get_config as _sleeve_cfg
            _sc = _sleeve_cfg(trader_id)
            if _sc.get("sleeves"):
                cfg = AVTConfig()
                if _sc.get("virtual_capital"):
                    cfg.virtual_capital = float(_sc["virtual_capital"])
                cfg.sleeves = list(_sc["sleeves"])
        except Exception:
            logger.exception("AVT sleeve config load failed for trader %s — using shared pool", trader_id)
            cfg = None

        # Init engine with the full fuel line: signal generator + ranker +
        # watchlist + live market data. Always init (even with 0 strategies)
        # so /start works and the health endpoint reports honestly.
        avt = init_avt(
            strategies=all_strategies,
            config=cfg,
            trader_id=trader_id,
            market_data=market_data,
            signal_generator=sig_gen,
            stock_ranker=StockSelectionRanker(market_data=market_data),
            watchlist=watchlist,
        )

        # Real F&O charges (STT-dominated) when a system template is active — the cascade's validated result is
        # net of these. Default engines (no templates) keep their prior slippage-only behaviour untouched.
        if template_dicts:
            try:
                from analytics.futures_charges import FuturesCostModel
                avt.portfolio.cost_model = FuturesCostModel()
            except Exception:
                logger.warning("could not attach FuturesCostModel to AVT portfolio", exc_info=True)

        # Two-tier attention (B.2): this is the LIVE per-user engine → turn on peripheral-vision gating so a
        # ~100-symbol watchlist is feasible (cheap recognition on all; expensive cascade priming only on lit-up
        # symbols). Fail-open + cascade-only → safe even before the warm live-candle cache is populated (it just
        # no-ops until then). Replay/research engines never call this path, so the golden stays byte-identical.
        avt._two_tier_attention = True
        logger.info(
            "AVT engine initialized: %d taught + %d system-template SKOs, %d watchlist "
            "symbols, signal_gen=%s, market_data=%s",
            len(strategy_dicts), len(template_dicts), len(watchlist),
            type(sig_gen).__name__,
            type(market_data).__name__ if market_data else "replay",
        )

        # Per-strategy auto-scanning: each strategy scans only its own matched
        # stocks (same source the co-planner uses). The provider reads the
        # engine's CURRENT strategies each cycle, so hot-loaded strategies and
        # newly distributed/matched stocks are picked up automatically.
        def _strategy_watchlist_provider():
            return _build_strategy_watchlists(db_factory, list(avt.strategies.values()))

        avt.set_strategy_watchlist_provider(_strategy_watchlist_provider)
        try:
            avt.set_strategy_watchlists(_strategy_watchlist_provider())
        except Exception as e:
            logger.debug("Initial per-strategy watchlist build failed: %s", e)

        # Create improvement lab (AVT's internal analysis engine)
        # Pass db_factory so TwoSpeedLearner models are persisted to avt_learner_state
        from improvement_lab.orchestrator import ImprovementLabOrchestrator
        lab = ImprovementLabOrchestrator(db_factory=db_factory)

        # Create AVT trade repository
        from avt.repository import AVTTradeRepository, AVTImprovementRepository
        db = db_factory()
        if hasattr(db, "__next__"):
            db = next(db)
        repo = AVTTradeRepository(db)

        # Load the per-(strategy,stock) ranking baseline (benchmark replay) into
        # the ranker. Empty until the user builds a baseline → ranking unaffected.
        try:
            if getattr(avt, "_stock_ranker", None) is not None:
                avt._stock_ranker.load_symbol_performance(db, trader_id)
        except Exception as e:
            logger.debug("symbol-performance load skipped: %s", e)

        # Create self-improver (bridges lab → engine → proof)
        from avt.self_improver import AVTSelfImprover
        self_improver = AVTSelfImprover(
            lab=lab,
            avt_engine=avt,
            repository=repo,
        )

        # Create observer (behavioral checks) and triage (message queue)
        from avt.observer import AVTObserver
        from avt.triage import AVTTriage
        observer = AVTObserver()
        triage = AVTTriage(observer)

        # Create competition engine
        competition = None
        try:
            from avt.competition import CompetitionEngine
            competition = CompetitionEngine(db=db)
        except Exception as e:
            logger.debug("Competition engine not available: %s", e)

        # ── Strategy Lifecycle: health gate + lifecycle manager ──
        lifecycle_manager = None
        try:
            from avt.strategy_lifecycle import StrategyHealthGate, StrategyLifecycleManager

            health_gate = StrategyHealthGate()
            lifecycle_manager = StrategyLifecycleManager()

            for strategy_dict in strategy_dicts:
                sid = str(strategy_dict.get("id", strategy_dict.get("short_code", "")))
                try:
                    lifecycle_manager.ingest_strategy(strategy_dict)
                    info = lifecycle_manager.get_lifecycle_info(sid)
                    if info:
                        logger.info(
                            "Strategy %s: state=%s grade=%s health=%.0f",
                            sid, info.state, info.grade, info.health_score,
                        )
                except Exception as e:
                    logger.warning("Lifecycle init for %s failed: %s", sid, e)

            logger.info(
                "Lifecycle manager: %d strategies ingested",
                len(lifecycle_manager._lifecycles),
            )
        except Exception as e:
            logger.warning("Strategy lifecycle setup failed (non-fatal): %s", e)

        # Wire hooks (persistence + events)
        wire_avt_engine(avt, self_improver=self_improver, db_factory=db_factory)

        # Wire AVT events to WebSocket
        wire_avt_to_websocket()

        # Wire lifecycle events to WebSocket
        _wire_lifecycle_events()

        # Hot-load strategies compiled while the server is running
        _wire_sko_hot_reload(avt, db_factory)

        # Reload state from DB
        reload_avt_state(avt, db_factory)

        # ── Edge Amplifier: learn A+ setups from AVT's trade history ──
        try:
            from improvement_lab.edge_amplifier import get_edge_amplifier
            edge_amp = get_edge_amplifier()
            for sid in avt.strategies:
                closed = avt.get_closed_trades(strategy_id=sid)
                if len(closed) >= 20:
                    profile = edge_amp.learn(closed, sid)
                    if profile.edges:
                        logger.info(
                            "Edge amplifier: %s → %d edges (%d validated), best=%s",
                            sid, len(profile.edges),
                            sum(1 for e in profile.edges if e.split_half_valid),
                            profile.best_edge.grade if profile.best_edge else "none",
                        )
        except Exception as e:
            logger.debug("Edge amplifier init skipped: %s", e)

        # Start scheduler
        scheduler = AVTScheduler(
            self_improver=self_improver,
            competition_engine=competition,
            db_factory=db_factory,
        )
        scheduler_task = asyncio.create_task(scheduler.run())
        logger.info("AVT scheduler started (daily analysis + weekly eval + competition snapshots)")

        # Store references for API access and shutdown
        avt._self_improver = self_improver
        avt._observer = observer
        avt._triage = triage
        avt._competition = competition
        avt._scheduler = scheduler
        avt._scheduler_task = scheduler_task
        avt._lifecycle_manager = lifecycle_manager

        return scheduler_task

    except Exception as e:
        logger.warning("AVT full setup failed: %s", e)
        return None


def _wire_lifecycle_events():
    """Subscribe lifecycle events to WebSocket broadcast."""
    try:
        from core.events import Event, EventType, get_event_bus

        bus = get_event_bus()
        lifecycle_event_types = [
            EventType.STRATEGY_HEALTH_CHECKED,
            EventType.STRATEGY_CALIBRATION_STARTED,
            EventType.STRATEGY_CALIBRATION_GRADUATED,
            EventType.STRATEGY_DEGRADATION_DETECTED,
            EventType.STRATEGY_PAUSED,
            EventType.STRATEGY_RESUMED,
        ]

        def on_lifecycle_event(event: Event):
            try:
                from api.routes.websocket import manager
                event_data = {
                    "type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "data": event.data,
                }
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        manager.broadcast("session_live", event_data)
                    )
            except Exception:
                pass

        for et in lifecycle_event_types:
            bus.subscribe(et, on_lifecycle_event)

        logger.info("Lifecycle events wired to WebSocket")
    except Exception as e:
        logger.debug("Failed to wire lifecycle events: %s", e)
