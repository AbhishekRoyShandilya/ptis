"""
marketdata.candle_store — the single read-through interface every consumer uses.

    CandleStore.get(symbol, interval, start, end) -> pandas.DataFrame

Behavior:
  - 1m / 1d  : lake-first; fetch only the missing date range from providers,
               write to the Parquet lake, then return the requested slice.
  - 5m/15m/30m/1h : derived by resampling 1m IN THIS METHOD (DuckDB time_bucket).
               Resampling lives here so a future `cached_resamples/` materialized
               layer drops in transparently (check cache → else resample) with no
               consumer changes.

Schema returned: marketdata.lake.CANDLE_COLS, chronological.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from marketdata import lake, instruments, providers

# Rate-limit guard for the always-refetch-today logic: re-fetch the current (forming) day at most
# once per _TODAY_TTL_S seconds per (symbol, interval), so a burst of live reads (a watchlist scan,
# the proactive loop) doesn't hammer the broker into "exceeding access rate".
_TODAY_FETCHED: dict = {}
_TODAY_TTL_S = 45.0

logger = logging.getLogger(__name__)

# Normalise interval aliases → canonical tokens.
_ALIAS = {
    "1minute": "1m", "1min": "1m", "1m": "1m",
    "5minute": "5m", "5min": "5m", "5m": "5m",
    "15minute": "15m", "15min": "15m", "15m": "15m",
    "30minute": "30m", "30min": "30m", "30m": "30m",
    "60minute": "1h", "60min": "1h", "60m": "1h", "1h": "1h", "1hour": "1h",
    "day": "1d", "daily": "1d", "1day": "1d", "1d": "1d", "d": "1d",
}
# Resample targets → DuckDB bucket interval.
_RESAMPLE = {"5m": "5 minutes", "15m": "15 minutes", "30m": "30 minutes", "1h": "1 hour"}
_BASE = {"1m", "1d"}


def _norm(interval: str) -> str:
    return _ALIAS.get(str(interval).lower().strip(), str(interval).lower().strip())


class CandleStore:
    def get(
        self, symbol: str, interval: str, start, end, live: bool = False
    ) -> pd.DataFrame:
        """Read candles for [start, end]. `live=True` appends today's forming bars
        from the real-time feed (the lake's historical endpoint lags during the
        session) — callers gate this on "querying now" to avoid lookahead."""
        iv = _norm(interval)
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        # The lake stores tz-naive timestamps; a tz-aware bound (e.g. a
        # pd.Timestamp.utcnow() "now" from the analog engine, or a tz-aware
        # trade entry_time) would make DuckDB refuse the ts comparison. Strip tz
        # (keep wall-clock) so every caller compares naive-to-naive.
        if getattr(start, "tzinfo", None) is not None:
            start = start.tz_localize(None)
        if getattr(end, "tzinfo", None) is not None:
            end = end.tz_localize(None)

        if iv in _RESAMPLE:
            base = self.get(symbol, "1m", start, end)        # warm-up history (no live here)
            df = self._resample(base, iv, _RESAMPLE[iv])
        else:
            if iv not in _BASE:
                logger.warning("CandleStore: unsupported interval %r → treating as 1d", interval)
                iv = "1d"
            self._ensure(symbol, iv, start, end)
            df = lake.read_candles(symbol, iv, start, end)

        if live:
            df = append_live_tail(df, symbol, interval)      # today's bars at the requested interval
        return df

    # ── internals ──────────────────────────────────────────────

    def _ensure(self, symbol: str, interval: str, start: datetime, end: datetime) -> None:
        """
        Fetch + cache only the date ranges not already FETCHED (per the coverage
        ledger). Each fetched range is then marked covered — even if it yielded no
        bars (holidays/weekends) — so identical re-requests are a true no-op.
        """
        import pandas as _pd
        from datetime import timedelta as _td
        sd, ed = lake.to_date(start), lake.to_date(end)
        # IST "today" — the current, still-forming session. Two live-data invariants hinge on it:
        _today = _pd.Timestamp.now(tz="Asia/Kolkata").normalize().date()
        ranges = [(gs, ge) for gs, ge in lake.missing_ranges(symbol, interval, sd, ed) if gs <= ge]
        # (1) ALWAYS re-fetch today, even if the ledger says "covered". An early/0-bar fetch at the open
        # otherwise marks today covered and FREEZES every live read (and the proactive partner) at
        # yesterday's close for the whole session — the live-blindness bug.
        if sd <= _today <= ed and not any(gs <= _today <= ge for gs, ge in ranges):
            ranges.append((_today, _today))
        import time as _time
        for gs, ge in ranges:
            # Rate-limit guard: a range covering today is re-fetched at most once per _TODAY_TTL_S
            # per symbol/interval — otherwise a burst of live reads hammers the broker. Serve lake.
            if gs <= _today <= ge:
                _k = (symbol, interval)
                if _time.time() - _TODAY_FETCHED.get(_k, 0.0) < _TODAY_TTL_S:
                    continue
                _TODAY_FETCHED[_k] = _time.time()
            # A provider fetch/write failure (e.g. the broken index-intraday path,
            # whose yfinance fallback returns tz-aware bars that clash with the
            # tz-naive lake) must NOT abort the read — 4 years of 1-min may be
            # sitting in the lake already. Log and fall back to stored data; the
            # range stays uncovered so a later run retries it.
            try:
                rows, complete = providers.fetch(symbol, interval, gs, ge)
                if rows:
                    df = self._enrich(rows, symbol, interval)
                    # SEALED SESSIONS ARE IMMUTABLE (2026-09-25). A fetch range spans TODAY *and prior days*, so
                    # refreshing today's forming bars used to re-fetch and OVERWRITE completed sessions — proven by
                    # a read alone moving a partition's mtime, and by an Upstox repair being silently replaced with
                    # Angel rows minutes later. Replays of the same session then returned different answers, and
                    # 2026-09-22 even changed decision (['COALINDIA'] -> []). Drop rows for frozen dates; today and
                    # unsealed sessions still refresh exactly as before. Repairs set PTIS_ALLOW_REWRITE=1.
                    from marketdata import session_seal as _seal
                    df, _dropped = _seal.drop_sealed(df)
                    if _dropped:
                        logger.debug("candle_store: %s/%s dropped %d rows for sealed sessions",
                                     symbol, interval, _dropped)
                    if df is not None and len(df):
                        inst_type, _ = instruments.classify(symbol)
                        lake.write_candles(df, inst_type, symbol, interval)
                # Only mark covered if the fetch SUCCEEDED (even with 0 bars on a holiday/weekend).
                # (2) NEVER mark TODAY covered — it's still forming, so cap coverage at yesterday.
                if complete:
                    cov_end = min(ge, _today - _td(days=1))
                    if cov_end >= gs:
                        lake.add_coverage(symbol, interval, gs, cov_end)
            except Exception as e:
                logger.warning("candle fetch/write failed %s/%s %s..%s (serving lake data): %s",
                               symbol, interval, gs, ge, e)
                continue

    def _enrich(self, rows: list[dict], symbol: str, interval: str) -> pd.DataFrame:
        inst_type, underlying = instruments.classify(symbol)
        key = instruments.instrument_key(symbol) or ""
        df = pd.DataFrame(rows).rename(columns={"timestamp": "ts"})
        # NORMALIZE timezone → tz-naive IST wall-clock, to match the lake. Angel One returns tz-aware
        # (+05:30) bars; writing them straight to the tz-naive lake raised "Cannot compare tz-naive and
        # tz-aware timestamps" → the write aborted and the lake stayed stale (the feed-freshness bug).
        df["ts"] = pd.to_datetime(df["ts"])
        if getattr(df["ts"].dt, "tz", None) is not None:
            df["ts"] = df["ts"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df["symbol"] = symbol
        df["underlying_symbol"] = underlying
        df["instrument_key"] = key
        df["interval"] = interval
        if "oi" not in df.columns:
            df["oi"] = 0
        return df

    def _resample(self, df_1m: pd.DataFrame, target: str, bucket: str) -> pd.DataFrame:
        if df_1m is None or df_1m.empty:
            return pd.DataFrame(columns=lake.CANDLE_COLS)
        con = lake._con()
        con.register("_rs_src", df_1m)
        try:
            out = con.execute(
                f"""
                SELECT
                    any_value(instrument_key)    AS instrument_key,
                    any_value(symbol)            AS symbol,
                    any_value(underlying_symbol) AS underlying_symbol,
                    time_bucket(INTERVAL '{bucket}', ts) AS ts,
                    '{target}'                   AS interval,
                    arg_min(open, ts)            AS open,
                    max(high)                    AS high,
                    min(low)                     AS low,
                    arg_max(close, ts)           AS close,
                    sum(volume)                  AS volume,
                    arg_max(oi, ts)              AS oi
                FROM _rs_src
                GROUP BY time_bucket(INTERVAL '{bucket}', ts)
                ORDER BY ts
                """
            ).fetchdf()
        finally:
            con.unregister("_rs_src")
        return out[lake.CANDLE_COLS]


def append_live_tail(lake_df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """
    Append today's LIVE forming bars (MarketDataService — the same intraday feed
    AVT scans on) to lake history. The lake's historical endpoint doesn't publish
    today's intraday until after close, so during the session a "current" query
    needs the live feed; the lake supplies indicator warm-up. ONE shared
    implementation for every live consumer (analog engine, regime store).
    Best-effort: feed unconfigured/empty → returns lake_df unchanged.
    """
    try:
        from ingestion.market_data_service import get_market_data_service
        svc = get_market_data_service()
        if svc is None:
            return lake_df
        live = svc.fetch_candles_for_scanner(symbol, 120, interval)
        if not live:
            return lake_df
        ldf = pd.DataFrame(live).rename(columns={"timestamp": "ts"})
        ldf["ts"] = pd.to_datetime(ldf["ts"])
        if getattr(ldf["ts"].dt, "tz", None) is not None:
            ldf["ts"] = ldf["ts"].dt.tz_localize(None)   # IST wall-clock, match the naive lake
        if lake_df is None or len(lake_df) == 0:
            return ldf
        cutoff = pd.to_datetime(lake_df["ts"]).max()
        ldf = ldf[ldf["ts"] > cutoff]                    # only bars newer than the lake
        if ldf.empty:
            return lake_df
        for col in lake_df.columns:                      # align to lake schema
            if col not in ldf.columns:
                ldf[col] = symbol if col == "symbol" else (
                    interval if col == "interval" else (0 if col == "oi" else None))
        return pd.concat([lake_df, ldf[lake_df.columns]], ignore_index=True)
    except Exception as e:
        logger.debug("live tail append failed %s/%s: %s", symbol, interval, e)
        return lake_df


_store: Optional[CandleStore] = None


def get_candle_store() -> CandleStore:
    global _store
    if _store is None:
        _store = CandleStore()
    return _store
