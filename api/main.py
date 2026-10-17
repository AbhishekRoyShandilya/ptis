# api/main.py
"""
FastAPI application factory.
Initializes all subsystems on startup.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import os

from config.logging_config import setup_logging
from config.settings import settings
from core.database import engine, get_db, SessionLocal
from core.events import EventType, get_event_bus
from core.session_manager import get_session_manager
from strategy_engine.strategy_registry import get_registry

logger = logging.getLogger(__name__)

# When True (default), the API process owns background loops (scanner, AVT, trade-sync).
# Set PTIS_RUN_BACKGROUND_LOOPS=false in .env to disable — then run worker.py separately.
# WARNING: never run worker.py alongside the API when this is True; loops will duplicate.
_RUN_BACKGROUND_LOOPS = os.environ.get("PTIS_RUN_BACKGROUND_LOOPS", "true").lower() == "true"

# Smart-pick / setup watchlist scanning is OFF by default — it continuously scans watchlist stocks (heavy) and
# is USER-ACTIONED. Turn it on with PTIS_SETUP_SCANNER=true. The feed-refresh (lake updater), AVT (user-started),
# and trade-sync loops are unaffected.
_SETUP_SCANNER_ON = os.environ.get("PTIS_SETUP_SCANNER", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    # ── STARTUP ──
    setup_logging()
    logger.info("=== PTIS Starting ===")

    # Startup config validation
    config_warnings = settings.validate_config()
    for w in config_warnings:
        logger.warning("CONFIG: %s", w)

    # Security check: reject default JWT secret in production
    if not settings.DEBUG and settings.PTIS_JWT_SECRET == "ptis-dev-secret-change-in-production-2026":
        raise RuntimeError(
            "PTIS_JWT_SECRET must be set to a secure value when DEBUG=False. "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )

    # Sentry error tracking (if configured)
    if settings.SENTRY_DSN:
        try:
            import sentry_sdk
            sentry_sdk.init(dsn=settings.SENTRY_DSN, traces_sample_rate=0.1)
            logger.info("Sentry error tracking initialized")
        except ImportError:
            logger.warning("SENTRY_DSN set but sentry-sdk not installed")

    app.state.start_time = datetime.now(timezone.utc)

    # 1. Initialize strategy registry
    try:
        registry = get_registry()
        count = registry.initialize()
        logger.info("Strategy registry: %d strategies loaded", count)
    except Exception as e:
        logger.error("Strategy registry failed to initialize: %s", e)

    # 2. Rebuild session state from DB
    try:
        session_mgr = get_session_manager()
        db = SessionLocal()
        try:
            session_mgr.rebuild_from_db(db)
            logger.info("Session state rebuilt: %s", session_mgr.summary())
        finally:
            db.close()
    except Exception as e:
        logger.error("Session rebuild failed: %s", e)

    # 3. Wire event handlers
    try:
        _wire_event_handlers()
        logger.info("Event handlers wired")
    except Exception as e:
        logger.error("Event handler wiring failed: %s", e)

    # 3b. Wire strategy hot-reload (propagates new strategies to scanner + co-planner)
    try:
        from automation.strategy_hotreload import register_hotreload_handler
        register_hotreload_handler()
        logger.info("Strategy hot-reload handler registered")
    except Exception as e:
        logger.warning("Strategy hot-reload handler failed: %s", e)

    # 3c. Start continuous learner (learns from every closed trade)
    try:
        from strategy_learning.continuous_learner import get_continuous_learner
        learner = get_continuous_learner(db_factory=get_db)
        learner.subscribe()
        logger.info("Continuous learner subscribed to trade events")
    except Exception as e:
        logger.warning(f"Continuous learner not started: {e}")

    # 3d. Initialize market data service (Angel One historical + Upstox live)
    market_data_svc = None
    try:
        from ingestion.market_data_service import init_market_data_service
        market_data_svc = init_market_data_service()
        if market_data_svc:
            parts = []
            if market_data_svc.has_historical:
                parts.append("Angel One (historical)")
            if market_data_svc.has_live:
                parts.append("Upstox (live)")
            logger.info("Market data service ready: %s", " + ".join(parts))
        else:
            logger.info("Market data service: not configured (need Angel One and/or Upstox creds)")
    except Exception as e:
        logger.warning("Market data service: %s", e)

    # 4. Start setup scanner background loop
    # Background loops are owned by this process when PTIS_RUN_BACKGROUND_LOOPS=true (default).
    # Set to false and run worker.py separately for a two-process deployment.
    # Never run worker.py alongside the API when this is true — loops will duplicate.
    scanner = None
    scanner_task = None
    feed_refresh_task = None
    if _RUN_BACKGROUND_LOOPS and _SETUP_SCANNER_ON:
        try:
            from automation.setup_scanner import get_scanner_loop
            scanner = get_scanner_loop(
                symbols=[],          # populated at runtime via /api/v1/automation/scanner/*
                scan_interval=60,
            )
            # Inject candle fetcher if available (Angel One historical + Upstox intraday)
            if market_data_svc and market_data_svc.has_historical:
                scanner.set_fetcher(market_data_svc.fetch_candles_for_scanner)
                logger.info("Scanner: using live candle feed")
            # Wire DB factory so scanner auto-populates from StrategyProfile watchlists
            scanner.set_db_factory(SessionLocal)
            scanner.sync_from_strategy_profiles()
            scanner_task = asyncio.create_task(scanner.run())
            logger.info("Setup scanner loop started")
        except Exception as e:
            logger.warning(f"Setup scanner not started: {e}")
    elif _RUN_BACKGROUND_LOOPS:
        logger.info("Setup scanner OFF (default, user-actioned) — set PTIS_SETUP_SCANNER=true to enable smart-pick scanning")

    if _RUN_BACKGROUND_LOOPS:

        # 4a. Feed auto-refresh — keep the OHLCV lake fresh so the eye is never stale (startup catch-up +
        # periodic during market hours). The "set and forget" so the user never runs backfill manually.
        if os.getenv("PTIS_AUTO_FEED_REFRESH", "true").lower() in ("1", "true", "yes") and market_data_svc:
            try:
                from marketdata.auto_refresh import feed_refresh_loop
                feed_refresh_task = asyncio.create_task(feed_refresh_loop())
                logger.info("Feed auto-refresh loop started (lake stays current automatically)")
            except Exception as e:
                logger.warning(f"Feed auto-refresh not started: {e}")

        # 4b. Proactive partner loop — auto-surfaces money moves / position alerts into the live feed
        # (event bus → /ws/session → LiveAlertsPage/chat). OFF by default: per-symbol narration is heavy
        # until the streaming eye lands. Enable with PTIS_PROACTIVE_LOOP=true during market hours.
        # Watchlist: PTIS_PROACTIVE_WATCHLIST (comma-list) for instant dogfooding, else the scanner's
        # synced symbols (StrategyProfiles / Watchlist / Smart Picks).
        if os.getenv("PTIS_PROACTIVE_LOOP", "false").lower() == "true":
            try:
                from intelligence.proactive_loop import ProactiveLoop
                from intelligence.proactive_monitor import live_open_positions

                def _proactive_watchlist():
                    env = [s.strip().upper() for s in os.getenv("PTIS_PROACTIVE_WATCHLIST", "").split(",") if s.strip()]
                    # Breadth is now CHEAP: scan_watchlist runs cheap recognition over ALL these symbols and
                    # pays the full-eye LLM only on the MOVERS (recognition/understanding split), so the cap can
                    # be wide. PTIS_PROACTIVE_MAX (default 40) just bounds the recognition sweep; movers are few
                    # so the LLM budget is self-limiting. Open positions are monitored separately regardless.
                    syms = env or list(getattr(scanner, "symbols", []) or [])
                    return syms[: int(os.getenv("PTIS_PROACTIVE_MAX", "40"))]

                _ploop = ProactiveLoop(
                    watchlist_fn=_proactive_watchlist,
                    positions_fn=live_open_positions,   # monitor the user's OPEN broker positions too
                    interval_sec=int(os.getenv("PTIS_PROACTIVE_INTERVAL", "300")),
                )
                asyncio.create_task(_ploop.run())
                logger.info("Proactive partner loop started")
            except Exception as e:
                logger.warning(f"Proactive loop not started: {e}")
    else:
        logger.info("Background loops disabled (PTIS_RUN_BACKGROUND_LOOPS=false) — worker.py owns scanner/AVT/trade-sync")

    # 4a. Startup watchlist scan — distribute master watchlist → strategy watchlists.
    # The heavy candle-fetch/analysis is SYNCHRONOUS, so it runs in a worker
    # thread (run_in_executor); otherwise it blocks the event loop and starves
    # every other request for the duration of the scan.
    def _do_watchlist_scan():
        db = SessionLocal()
        try:
            from core.models import Watchlist
            from strategy_learning.watchlist_analyzer import WatchlistAnalyzer

            masters = db.query(Watchlist).filter(
                Watchlist.watchlist_type == "master",
                Watchlist.is_active == True,  # noqa: E712
            ).all()
            if not masters:
                return

            analyzer = WatchlistAnalyzer(db)
            try:
                from ingestion.market_data_service import get_market_data_service
                mds = get_market_data_service()
                if mds and mds.has_historical:
                    analyzer.set_fetcher(mds.fetch_candles_for_scanner)
            except Exception:
                pass

            for master in masters:
                try:
                    results = analyzer.scan_and_distribute(master.id)
                    matched_counts = {
                        k: v.get("total_matched", 0)
                        for k, v in results.items() if isinstance(v, dict)
                    }
                    logger.info(
                        "Startup watchlist scan: '%s' -> %s", master.name, matched_counts
                    )
                except Exception as e:
                    logger.warning("Startup scan failed for '%s': %s", master.name, e)
        except Exception as e:
            logger.warning("Startup watchlist scan aborted: %s", e)
        finally:
            db.close()

    if _SETUP_SCANNER_ON:
        try:
            async def _startup_watchlist_scan():
                await asyncio.sleep(20)   # let market data authenticate
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, _do_watchlist_scan)

            asyncio.create_task(_startup_watchlist_scan())
            logger.info("Startup watchlist scan scheduled (after 20s, off the event loop)")
        except Exception as e:
            logger.warning("Could not schedule startup watchlist scan: %s", e)
    else:
        logger.info("Startup watchlist scan skipped (setup scanner off by default — PTIS_SETUP_SCANNER=true to enable)")

    # 4a-ii. Smart Picks auto-seed — DISABLED.
    # The system no longer auto-scores the FNO universe or auto-creates
    # watchlists. Watchlists are purely user-driven (the user adds symbols),
    # and scanning is per-user based on their taught strategies + their own
    # watchlists. The on-demand POST /api/v1/watchlist/smart-seed endpoint
    # remains for users who explicitly choose to populate Smart Picks.
    logger.info("Smart Picks auto-seed disabled (watchlists are user-driven only)")

    # 4b. Wire market data to Discovery + Intelligence
    if market_data_svc:
        if market_data_svc.has_historical:
            try:
                from discovery.orchestrator import get_discovery_orchestrator
                disc_orch = get_discovery_orchestrator()
                disc_orch.set_fetcher(market_data_svc.fetch_candles_for_discovery)
                logger.info("Discovery: using Angel One historical candle feed")
            except Exception as e:
                logger.debug("Discovery fetcher wiring skipped: %s", e)

        try:
            from intelligence.market_context import MarketContextEnricher, get_market_enricher
            enricher = get_market_enricher()
            enricher.data_provider = market_data_svc.as_intelligence_provider()
            logger.info("Intelligence: using live market data provider")
        except Exception as e:
            logger.debug("Intelligence provider wiring skipped: %s", e)

    # 5. AVT is PER-USER now — each user's engine is created lazily when they hit
    #    POST /api/v1/avt/start. Here we just register the shared deps (db factory
    #    + market data) so the route can build a user's engine on demand.
    avt_scheduler_task = None  # per-user schedulers are created on /start
    if _RUN_BACKGROUND_LOOPS:
        try:
            from avt.integrations import set_avt_deps
            avt_market_data = None
            if market_data_svc and market_data_svc.has_live:
                from avt.market_data import UpstoxMarketData
                avt_market_data = UpstoxMarketData(market_data_svc)
            set_avt_deps(db_factory=SessionLocal, market_data=avt_market_data)
            logger.info("AVT deps registered (per-user engines created on POST /api/v1/avt/start)")
        except Exception as e:
            logger.warning(f"AVT deps not registered: {e}")

    # 6. Start Groww trade sync background loop
    trade_sync_task = None
    if _RUN_BACKGROUND_LOOPS:
        try:
            from ingestion.trade_sync import init_trade_sync_service
            sync_svc = init_trade_sync_service(db_factory=SessionLocal)
            if sync_svc:
                trade_sync_task = asyncio.create_task(sync_svc.run())
                logger.info("Groww trade sync started (every %ds)", sync_svc._sync_interval)
            else:
                logger.info("Groww trade sync: not configured (GROWW_API_KEY missing)")
        except Exception as e:
            logger.warning("Trade sync: %s", e)

    # 7. Initialize RAG memory ingestor
    try:
        from rag.ingestor import init_rag_ingestor
        rag_ingestor = init_rag_ingestor(db_factory=SessionLocal)
        logger.info("RAG memory ingestor subscribed to events")
    except Exception as e:
        logger.warning("RAG ingestor not started: %s", e)

    logger.info("=== PTIS Ready ===")
    yield

    # ── SHUTDOWN ──
    logger.info("=== PTIS Shutting Down ===")

    # Stop ALL per-user AVT engines (checkpoint + scheduler + run loop)
    try:
        from avt.engine import all_avt_engines
        engines = all_avt_engines()
        if engines:
            try:
                from avt.integrations import checkpoint_avt_state
                checkpoint_avt_state(db_factory=SessionLocal)
            except Exception as e:
                logger.warning("AVT checkpoint on shutdown failed: %s", e)
            for tid, avt in engines.items():
                try:
                    if getattr(avt, "_scheduler", None):
                        avt._scheduler.stop()
                    task = getattr(avt, "_scheduler_task", None)
                    if task:
                        task.cancel()
                    avt.status = "stopped"
                except Exception as e:
                    logger.warning("AVT shutdown error for %s: %s", tid, e)
            logger.info("Stopped %d AVT engine(s)", len(engines))
    except Exception as e:
        logger.warning("AVT shutdown error: %s", e)

    if scanner_task:
        scanner_task.cancel()
        try:
            await scanner_task
        except asyncio.CancelledError:
            pass
        logger.info("Setup scanner stopped")

    if feed_refresh_task:
        feed_refresh_task.cancel()
        try:
            await feed_refresh_task
        except asyncio.CancelledError:
            pass
        logger.info("Feed auto-refresh stopped")

    if trade_sync_task:
        trade_sync_task.cancel()
        try:
            await trade_sync_task
        except asyncio.CancelledError:
            pass
        logger.info("Trade sync stopped")


def _wire_event_handlers():
    """Connect event bus to subsystem handlers."""
    from shadow.event_handlers import wire_shadow_handlers
    wire_shadow_handlers()
    bus = get_event_bus()

    # Shadow engine listens for trade entries
    def on_trade_entered(event):
        logger.info(
            "→ Shadow engine: trade entered %s %s @ %s",
            event.data.get("symbol"),
            event.data.get("direction"),
            event.data.get("entry_price"),
        )
        # shadow/engine.py handler will be wired here in Priority 3

    # Shadow engine listens for trade closures
    def on_trade_closed(event):
        logger.info(
            "→ Shadow comparison: trade closed %s P&L=$%s",
            event.data.get("symbol"),
            event.data.get("pnl"),
        )
        # shadow/comparison.py handler will be wired here

    # Log behavioral events
    def on_override_detected(event):
        logger.warning(
            "WARNING Override detected on trade %s: %d overrides, discipline=%.0f",
            event.trade_id,
            event.data.get("total_overrides", 0),
            event.data.get("discipline_score", 0),
        )

    bus.subscribe(EventType.TRADE_ENTERED, on_trade_entered)
    bus.subscribe(EventType.TRADE_CLOSED, on_trade_closed)
    bus.subscribe(EventType.OVERRIDE_DETECTED, on_override_detected)

    # User-research record: freeze a STATE+STRATEGY+USER-BEHAVIOR sample at entry,
    # fill the OUTCOME at close (immutable, point-in-time). Behavioural/coaching
    # substrate ONLY — NOT strategy edge (that comes from AVT). Best-effort.
    try:
        from marketdata.user_research import register_user_research_capture
        register_user_research_capture(SessionLocal)
    except Exception as e:
        logger.warning("User-research capture not registered: %s", e)

    # Bootstrap → Judgment Learning: auto-initialize JudgmentManager
    def on_bootstrap_completed(event):
        strategy_id = event.data.get("strategy_name", "")
        uncertainty_factors = event.data.get("uncertainty_factors", [])
        if not strategy_id or not uncertainty_factors:
            return
        try:
            from bootstrap.live_connector import get_judgment_connector
            connector = get_judgment_connector()
            connector.initialize_for_strategy(strategy_id, uncertainty_factors)
            logger.info(
                "JudgmentManager initialized for '%s' (%d learnable factors)",
                strategy_id, len(uncertainty_factors),
            )
        except Exception as e:
            logger.warning("JudgmentManager init failed for %s: %s", strategy_id, e)

    bus.subscribe(EventType.BOOTSTRAP_COMPLETED, on_bootstrap_completed)

    # AVT improvement adopted → refresh co-planner's strategy-regime matrix
    def on_improvement_adopted(event):
        if event.data.get("action") != "adopted_by_user":
            return
        strategy_id = event.data.get("strategy_id", "")
        logger.info(
            "AVT improvement adopted for '%s' — refreshing co-planner matrix",
            strategy_id,
        )
        try:
            from core.database import SessionLocal
            from coplanner.db_interface import CoPlannerDB
            from coplanner.engine import CoPlannerEngine
            db = SessionLocal()
            try:
                cp_db = CoPlannerDB(db)
                engine = CoPlannerEngine(db=cp_db)
                engine.rebuild_matrix()
                engine.refresh_trust_weights()
                logger.info("Co-planner matrix rebuilt after AVT adoption")
            finally:
                db.close()
        except Exception as e:
            logger.warning("Co-planner rebuild after adoption failed: %s", e)

    bus.subscribe(EventType.AVT_IMPROVEMENT_PROVEN, on_improvement_adopted)


def create_app() -> FastAPI:
    app = FastAPI(
        title="PTIS — Personal Trading Intelligence System",
        version="0.2.0",
        description="Behavioral trading intelligence with strategy attribution, "
                    "shadow trading, trader DNA profiling, and strategy discovery",
        lifespan=lifespan,
    )

    cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting + request logging + metrics
    from api.middleware import RequestLoggingMiddleware, get_request_metrics
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/metrics")
    def metrics():
        return get_request_metrics().summary()

    # Core routes
    from api.routes.trades import router as trades_router
    from api.routes.shadow import router as shadow_router
    from api.routes.behavior import router as behavior_router
    from api.routes.profile import router as profile_router
    from api.routes.analytics import router as analytics_router
    from api.routes.websocket import router as ws_router

    # Feature routes
    from api.routes.journal import router as journal_router
    from api.routes.automation import router as automation_router
    from api.routes.strategies import router as strategies_router
    from api.routes.comparison import router as comparison_router
    from api.routes.dashboard import router as dashboard_router
    from api.routes.regime import router as regime_router
    from api.routes.knowledge import router as knowledge_router
    from api.routes.learning import router as learning_router
    from api.routes.chat import router as chat_router
    from api.routes.intelligence import router as intelligence_router
    from api.routes.discovery import router as discovery_router
    from api.routes.watchlist import router as watchlist_router
    from api.routes.ske import router as ske_router
    from api.routes.competition import router as competition_router
    # improvement_lab routes absorbed into AVT — api/routes/avt.py now handles
    # pre-trade checks, observations, triage, and conversation alongside improvements
    from api.routes.avt import router as avt_router
    from api.routes.bootstrap import router as bootstrap_router
    from api.routes.export import router as export_router
    from api.routes.broker import router as broker_router
    from api.routes.coplanner import router as coplanner_router
    from api.routes.research import router as research_router
    from api.routes.behavioral_agent import router as behavioral_agent_router

    # Auth router (self-prefixed /api/v1/auth)
    from api.auth import auth_router

    # Extended API routes for frontend compatibility
    from api.llm_coaching_api import llm_router
    from api.extended_api import extended_router

    app.include_router(auth_router)  # already has /api/v1/auth prefix
    app.include_router(trades_router, prefix="/api/v1")
    app.include_router(shadow_router, prefix="/api/v1")
    app.include_router(behavior_router, prefix="/api/v1")
    app.include_router(profile_router, prefix="/api/v1")
    app.include_router(analytics_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/api/v1")
    app.include_router(journal_router, prefix="/api/v1")
    app.include_router(automation_router, prefix="/api/v1")
    app.include_router(strategies_router, prefix="/api/v1")
    app.include_router(comparison_router, prefix="/api/v1")
    app.include_router(dashboard_router, prefix="/api/v1")
    app.include_router(regime_router, prefix="/api/v1")
    app.include_router(knowledge_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(intelligence_router, prefix="/api/v1")
    app.include_router(discovery_router, prefix="/api/v1")
    app.include_router(learning_router, prefix="/api/v1")
    app.include_router(watchlist_router, prefix="/api/v1")
    app.include_router(ske_router, prefix="/api/v1")
    app.include_router(competition_router, prefix="/api/v1")
    app.include_router(avt_router, prefix="/api/v1")
    app.include_router(bootstrap_router, prefix="/api/v1")
    app.include_router(export_router, prefix="/api/v1")
    app.include_router(broker_router, prefix="/api/v1")
    app.include_router(coplanner_router, prefix="/api/v1")
    app.include_router(research_router, prefix="/api/v1")
    app.include_router(behavioral_agent_router, prefix="/api/v1")
    app.include_router(llm_router)
    app.include_router(extended_router)

    # ── Global exception handler ──
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ── Health check ──
    @app.get("/health")
    def health():
        from sqlalchemy import text as sa_text
        from api.routes.websocket import manager

        registry = get_registry()
        session = get_session_manager()

        # DB connectivity check
        db_ok = False
        db = SessionLocal()
        try:
            db.execute(sa_text("SELECT 1"))
            db_ok = True
        except Exception as e:
            logger.warning("Health DB check failed: %s", e)
        finally:
            db.close()

        uptime = 0.0
        if hasattr(app.state, "start_time"):
            uptime = (datetime.now(timezone.utc) - app.state.start_time).total_seconds()

        overall = "healthy" if db_ok else "degraded"
        return {
            "status": overall,
            "version": "0.2.0",
            "db": "connected" if db_ok else "unreachable",
            "uptime_seconds": round(uptime, 1),
            "yaml_strategies": len(registry.list_ids()),
            "session": session.summary(),
            "websocket_connections": manager.active_connections,
        }

    return app



app = create_app()