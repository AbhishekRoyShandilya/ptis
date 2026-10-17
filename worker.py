# worker.py
"""
Background worker process — runs setup scanner, trade sync, and AVT scheduler
outside the API process.

Usage:
    python worker.py

This separates long-running tasks from the API server so they:
- Don't die on API restart
- Don't duplicate on horizontal scaling
- Don't block HTTP request handling

For beta: run alongside the API server as a separate process.
For production: use a proper task queue (Celery/ARQ/Dramatiq).
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

from config.logging_config import setup_logging
from config.settings import settings
from core.database import SessionLocal, get_db
from core.events import get_event_bus

setup_logging()
logger = logging.getLogger("ptis.worker")

# Safety guard: if the API process is configured to own background loops
# (PTIS_RUN_BACKGROUND_LOOPS=true, the default), starting worker.py simultaneously
# will duplicate scanner, AVT, and trade-sync — creating race conditions and
# potential double order processing.
# Fix: set PTIS_RUN_BACKGROUND_LOOPS=false in your .env before running worker.py.
import os as _os
if _os.environ.get("PTIS_RUN_BACKGROUND_LOOPS", "true").lower() == "true":
    print(
        "\n[PTIS WORKER] ERROR: PTIS_RUN_BACKGROUND_LOOPS=true (default).\n"
        "The API process is configured to run background loops (scanner, AVT, trade-sync).\n"
        "Starting worker.py alongside the API will DUPLICATE these loops.\n"
        "To use a separate worker process:\n"
        "  1. Set PTIS_RUN_BACKGROUND_LOOPS=false in your .env\n"
        "  2. Restart the API\n"
        "  3. Then run: python worker.py\n"
    )
    sys.exit(1)

_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    logger.info("Shutdown signal received, stopping gracefully...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


async def run_worker():
    logger.info("=== PTIS Worker Starting ===")
    tasks = []

    # 1. Continuous learner
    try:
        from strategy_learning.continuous_learner import get_continuous_learner
        learner = get_continuous_learner(db_factory=get_db)
        learner.subscribe()
        logger.info("Continuous learner subscribed")
    except Exception as e:
        logger.warning("Continuous learner: %s", e)

    # 2. Market data service
    market_data_svc = None
    try:
        from ingestion.market_data_service import init_market_data_service
        market_data_svc = init_market_data_service()
        if market_data_svc:
            logger.info("Market data service ready")
    except Exception as e:
        logger.warning("Market data: %s", e)

    # 3. Setup scanner
    try:
        from automation.setup_scanner import get_scanner_loop
        scanner = get_scanner_loop(symbols=[], scan_interval=60)
        if market_data_svc and market_data_svc.has_historical:
            scanner.set_fetcher(market_data_svc.fetch_candles_for_scanner)
        scanner.set_db_factory(SessionLocal)
        scanner.sync_from_strategy_profiles()
        tasks.append(asyncio.create_task(scanner.run()))
        logger.info("Setup scanner started")
    except Exception as e:
        logger.warning("Setup scanner: %s", e)

    # 4. AVT engine
    try:
        from avt.integrations import setup_avt_full
        avt_market_data = None
        if market_data_svc and market_data_svc.has_live:
            from avt.market_data import UpstoxMarketData
            avt_market_data = UpstoxMarketData(market_data_svc)
        avt_task = setup_avt_full(db_factory=SessionLocal, market_data=avt_market_data)
        if avt_task:
            tasks.append(avt_task)
        logger.info("AVT engine initialized")
    except Exception as e:
        logger.warning("AVT: %s", e)

    # 5. Trade sync (Groww)
    try:
        from ingestion.trade_sync import init_trade_sync_service
        sync_svc = init_trade_sync_service(db_factory=SessionLocal)
        if sync_svc:
            tasks.append(asyncio.create_task(sync_svc.run()))
            logger.info("Trade sync started")
    except Exception as e:
        logger.warning("Trade sync: %s", e)

    # 6. RAG ingestor
    try:
        from rag.ingestor import init_rag_ingestor
        init_rag_ingestor(db_factory=SessionLocal)
        logger.info("RAG ingestor subscribed")
    except Exception as e:
        logger.warning("RAG ingestor: %s", e)

    logger.info("=== PTIS Worker Ready (%d background tasks) ===", len(tasks))

    # Keep running until shutdown
    while not _shutdown:
        await asyncio.sleep(1)

    # Graceful shutdown
    logger.info("Shutting down worker tasks...")

    # Checkpoint AVT
    try:
        from avt.integrations import checkpoint_avt_state
        checkpoint_avt_state(db_factory=SessionLocal)
        logger.info("AVT state checkpointed")
    except Exception:
        pass

    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    logger.info("=== PTIS Worker Stopped ===")


if __name__ == "__main__":
    asyncio.run(run_worker())
