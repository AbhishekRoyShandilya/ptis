"""
marketdata.auto_refresh — keep the OHLCV lake fresh automatically so the eye is never stale.

Runs `backfill --update --all` once at app startup (catch up since the last session) and then periodically
during market hours, in a worker thread (the backfill is sync — never block the event loop). This is the
"set and forget" the user asked for: no manual `python -m marketdata.backfill` before each session.

Gated in the lifespan by PTIS_AUTO_FEED_REFRESH (default on) + a connected broker.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


def _in_refresh_window() -> bool:
    """Weekday, ~08:45–16:00 IST — pre-open catch-up, intraday tail, and post-close settle."""
    try:
        import datetime
        import pytz
        ist = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
    except Exception:
        import datetime as _dt
        ist = _dt.datetime.now()
    if ist.weekday() >= 5:                       # Sat/Sun
        return False
    return (8, 45) <= (ist.hour, ist.minute) <= (16, 0)


def _run_backfill() -> None:
    """Run the backfill as a SEPARATE PROCESS (its own GIL) so the heavy 104-symbol refresh never starves
    the API event loop or contends with the app's lake reads — the app stays responsive (coaching/chat work
    after hours) while the feed refreshes in the background."""
    import subprocess
    import sys
    import os
    subprocess.run(
        [sys.executable, "-m", "marketdata.backfill", "--update", "--all"],
        env={**os.environ}, timeout=1800, check=False,
    )


def _lake_is_stale() -> bool:
    """True only if a refresh would actually fetch something (lake's newest bar is before today). Avoids
    re-hammering the broker on every restart — especially AFTER HOURS, when today's data is already in and
    nothing new exists (the proactive/live work isn't needed after close, only stored-data coaching is)."""
    try:
        import pandas as pd
        from marketdata.candle_store import get_candle_store
        df = get_candle_store().get("INFY", "15m", pd.Timestamp.now() - pd.Timedelta(days=5), pd.Timestamp.now())
        if df is None or df.empty:
            return True
        return pd.to_datetime(df["ts"]).max().date() < pd.Timestamp.now().date()
    except Exception:
        return True


async def feed_refresh_loop(interval_min: int = 30) -> None:
    """Startup catch-up refresh (only if stale), then every `interval_min` during market hours. The backfill
    runs in a SUBPROCESS so it never blocks the API — coaching/behavioral/trade-analysis stay responsive."""
    # Startup catch-up — only if the lake is actually behind (skip the pointless after-hours re-hammer).
    if _lake_is_stale():
        try:
            logger.info("auto feed refresh: startup catch-up (lake stale) — backfill --update --all (subprocess)…")
            await asyncio.to_thread(_run_backfill)
            logger.info("auto feed refresh: startup catch-up done")
        except Exception as e:
            logger.warning("auto feed refresh (startup) failed: %s", e, exc_info=True)
    else:
        logger.info("auto feed refresh: lake already current — skipping startup catch-up")

    # Periodic — market hours only (skip overnight/weekend idle).
    while True:
        try:
            await asyncio.sleep(interval_min * 60)
        except asyncio.CancelledError:
            logger.info("auto feed refresh: stopped")
            raise
        if not _in_refresh_window():
            continue
        try:
            logger.info("auto feed refresh: periodic update…")
            await asyncio.to_thread(_run_backfill)
            logger.info("auto feed refresh: periodic done")
        except Exception as e:
            logger.warning("auto feed refresh (periodic) failed: %s", e, exc_info=True)
