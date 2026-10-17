"""
intelligence.warm_candles — keep per-symbol candles WARM in memory for the live proactive loop.

The recognition sweep's only real per-symbol cost is the COLD lake read (~0.7s to pull + resample the 10-day
window). In a continuous loop that window is INVARIANT except for today's forming bars, so re-reading it every
pass is waste. This caches the accumulated frame per (symbol, interval) and, on a warm pass, fetches ONLY from
the last cached day forward (new + the updating forming bar), merging it in. A short min-refresh guard also
collapses the intra-pass double-touch (recognize + read of the same symbol) to a single fetch.

LIVE use only (today, as_of=None). Historical/point-in-time reads bypass this and load their exact window.
"""
from __future__ import annotations

import time as _time

import pandas as pd

_WARM: dict = {}        # (symbol, interval) -> accumulated candle DataFrame
_FETCHED: dict = {}     # (symbol, interval) -> monotonic ts of last fetch
_MIN_REFRESH_S = 30.0   # within this of a fetch, reuse (kills the intra-pass recognize+read double-read)


def warm_candles(symbol: str, interval: str = "5m", lookback_days: int = 10):
    """The warm 10-day window for `symbol`, refreshing only the tail since the last fetch. Returns a
    candle DataFrame (ts/open/high/low/close/volume) or None. Best-effort — falls back to the cache."""
    from marketdata.candle_store import get_candle_store
    key = (symbol, interval)
    now_m = _time.monotonic()
    cached = _WARM.get(key)
    if cached is not None and len(cached) and (now_m - _FETCHED.get(key, 0.0)) < _MIN_REFRESH_S:
        return cached                                    # recently warmed → reuse, no lake touch
    cs = get_candle_store()
    day = pd.Timestamp.now().normalize()
    lo = day - pd.Timedelta(days=lookback_days)
    hi = day + pd.Timedelta(days=1)
    try:
        if cached is None or len(cached) == 0:
            df = cs.get(symbol, interval, lo, hi, live=True)          # cold: full window
        else:
            last = pd.to_datetime(cached["ts"]).max()
            tail = cs.get(symbol, interval, last.normalize(), hi, live=True)   # warm: only new/updated bars
            if tail is not None and len(tail):
                keep = cached[pd.to_datetime(cached["ts"]) < last.normalize()]
                df = pd.concat([keep, tail], ignore_index=True)
            else:
                df = cached
    except Exception:
        return cached
    if df is not None and len(df):
        df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
        df = df[pd.to_datetime(df["ts"]) >= lo].reset_index(drop=True)
        _WARM[key] = df
        _FETCHED[key] = now_m
    return df


def clear_warm(symbol: str | None = None, interval: str = "5m") -> None:
    """Drop the warm cache (a symbol, or all) — e.g. on a new trading day."""
    if symbol is None:
        _WARM.clear(); _FETCHED.clear()
    else:
        _WARM.pop((symbol, interval), None); _FETCHED.pop((symbol, interval), None)
