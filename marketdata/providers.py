"""
marketdata.providers — system-level historical candle providers.

Primary: Upstox v3 historical (Jan-2022 1-min, daily to 2000), chunked per the
v3 retrieval limits. Fallback: Angel One / Zerodha / yfinance (indices), reusing
the existing low-level fetchers in ingestion.market_data_service.

IMPORTANT: the fallback calls the *low-level* `_fetch_candles_angel` /
`_fetch_candles_zerodha` (and `fetch_index_candles_yfinance`) directly — NOT
`_fetch_candles_raw`, which is being rewired to read through CandleStore. Going
through `_fetch_candles_raw` here would create an infinite loop.

Returns provider-native rows: [{timestamp, open, high, low, close, volume, oi}].
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

# interval → (v3 unit, v3 interval, chunk size in days)
_V3 = {
    "1m": ("minutes", "1", 28),    # 1-min limit: 1 month/request
    "1d": ("days", "1", 3650),     # daily limit: 1 decade/request
}

# Polite throttle between chunk calls during bulk backfill, + one retry on 429.
_THROTTLE_S = 0.25
_RATELIMIT_BACKOFF_S = 1.5


def _upstox_client():
    """Reuse the running MarketDataService's Upstox client, else build from token."""
    try:
        from ingestion.market_data_service import get_market_data_service
        svc = get_market_data_service()
        if svc and getattr(svc, "_upstox", None):
            return svc._upstox
    except Exception:
        pass
    try:
        from ingestion.token_store import get_upstox_token
        from ingestion.broker_adapters.upstox import UpstoxClient
        token = get_upstox_token()
        if token:
            return UpstoxClient(access_token=token, api_key=settings.UPSTOX_API_KEY)
    except Exception as e:
        logger.debug("Upstox client unavailable: %s", e)
    return None


def _fetch_upstox_v3(
    symbol: str, interval: str, from_date: date, to_date: date
) -> tuple[list[dict[str, Any]], bool]:
    """
    Returns (rows, complete). `complete` is False if any chunk failed (rate
    limit / network) so the caller won't mark the range covered — it gets
    retried on the next run instead of leaving a permanent hole.
    """
    from ingestion.broker_adapters.upstox import UpstoxRateLimitError

    cfg = _V3.get(interval)
    if cfg is None:
        return [], True
    unit, v3_interval, chunk_days = cfg
    client = _upstox_client()
    if client is None:
        return [], False  # couldn't even attempt → not covered

    # Prefer the marketdata dimension key (NSE-correct ISIN); fall back to the
    # adapter's resolver. If we only have the invalid plain-symbol fallback
    # (e.g. NSE_EQ|TATAMOTORS — symbol delisted/renamed, not in Upstox's master),
    # skip cleanly instead of making N doomed 400-ing calls. Mark complete=True so
    # it isn't retried every run (there is genuinely no data to fetch).
    from marketdata import instruments
    key = instruments.instrument_key(symbol) or client.resolve_instrument_key(symbol)
    if not key or key.upper().endswith("|" + symbol.upper()):
        logger.warning("No valid Upstox instrument key for %s — skipping (not in master)", symbol)
        return [], True

    out: list[dict[str, Any]] = []
    complete = True
    cursor = from_date
    while cursor <= to_date:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), to_date)
        fetched = False
        for attempt in range(2):
            try:
                out.extend(client.get_historical_candles_v3(key, unit, v3_interval, cursor, chunk_end))
                fetched = True
                break
            except UpstoxRateLimitError:
                if attempt == 0:
                    time.sleep(_RATELIMIT_BACKOFF_S)
                    continue
                logger.warning("Upstox v3 rate-limited %s %s %s→%s", symbol, interval, cursor, chunk_end)
            except Exception as e:
                logger.warning("Upstox v3 fetch failed %s %s %s→%s: %s",
                               symbol, interval, cursor, chunk_end, e)
                break
        if not fetched:
            complete = False
        time.sleep(_THROTTLE_S)
        cursor = chunk_end + timedelta(days=1)
    return out, complete


def _fetch_fallback(symbol: str, interval: str, from_date: date, to_date: date) -> list[dict[str, Any]]:
    """Angel One / Zerodha historical, then yfinance for indices."""
    try:
        from ingestion.market_data_service import (
            get_market_data_service, fetch_index_candles_yfinance,
        )
    except Exception:
        return []

    svc = get_market_data_service()
    if svc is None:
        # 2026-09-17: in a fresh process (m4_eod, backfill) the service was never initialised, so this returned
        # None and the fetch fell through to yfinance -> 0 rows -> the lake never got the CURRENT day and
        # m4_eod skipped it with "no complete sessions". Initialise on demand.
        try:
            from ingestion.market_data_service import init_market_data_service
            svc = init_market_data_service()
        except Exception as e:
            logger.debug("market data service init failed: %s", e)
    if svc is not None:
        try:
            if getattr(svc, "_angel", None):
                rows = svc._fetch_candles_angel(symbol, interval, from_date, to_date)
                if rows:
                    return rows
            if getattr(svc, "_zerodha", None):
                rows = svc._fetch_candles_zerodha(symbol, interval, from_date, to_date)
                if rows:
                    return rows
        except Exception as e:
            logger.debug("Angel/Zerodha fallback failed for %s: %s", symbol, e)

    # Last resort: yfinance (indices only)
    try:
        return fetch_index_candles_yfinance(symbol, interval, from_date, to_date)
    except Exception:
        return []


def fetch(
    symbol: str, interval: str, from_date: date, to_date: date
) -> tuple[list[dict[str, Any]], bool]:
    """
    Fetch historical candles for [from_date, to_date] (inclusive). interval ∈ {"1m","1d"}.
    Upstox v3 primary; falls back to Angel One / Zerodha / yfinance on miss.

    Returns (rows, complete). `complete=False` means the fetch hit an error
    (rate limit / no provider), so the range should NOT be marked covered.
    """
    rows: list[dict[str, Any]] = []
    complete = True
    if (settings.MARKET_DATA_PROVIDER or "upstox").lower() == "upstox":
        rows, complete = _fetch_upstox_v3(symbol, interval, from_date, to_date)
    if not rows:
        fb = _fetch_fallback(symbol, interval, from_date, to_date)
        if fb:
            rows, complete = fb, True
    logger.debug("providers.fetch %s %s %s→%s → %d rows (complete=%s)",
                 symbol, interval, from_date, to_date, len(rows), complete)
    return rows, complete
