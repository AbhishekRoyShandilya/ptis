"""
marketdata.lake — Parquet partition layout + DuckDB read/write for the OHLCV lake.

Layout (options-ready; Phase 1 writes EQ + INDEX only):
    {LAKE}/candles/type={EQ|INDEX|OPT}/symbol={SYM}/interval={1m|1d}/{YYYY}.parquet

Each candle row carries the full uniform schema so equity, index and (later)
option bars share one shape:
    instrument_key, symbol, underlying_symbol, ts, interval,
    open, high, low, close, volume, oi

Writes are idempotent: a year's file is read, merged on `ts` (keep last),
re-sorted and rewritten — so re-running a backfill never duplicates rows.
Reads go through a single in-process DuckDB connection over the parquet glob.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# Canonical column order for every candle row in the lake.
CANDLE_COLS = [
    "instrument_key", "symbol", "underlying_symbol", "ts", "interval",
    "open", "high", "low", "close", "volume", "oi",
]

_duck = None                      # base connection — owns the in-memory DB and its memory cap
_duck_lock = threading.Lock()     # guards first-time creation
_tls = threading.local()          # per-THREAD cursor over that same DB


def _con():
    """A per-THREAD DuckDB cursor over ONE shared in-memory database.

    ROOT-CAUSE FIX (2026-07-20, found live via py-spy): a single connection shared across threads is NOT
    thread-safe in DuckDB — concurrent queries serialise and contend. The live scan runs the cascade prewarm on
    a thread pool, so every worker piled into `read_candles` on the same connection and the scan ground to a halt
    (minutes per cycle, no broker involved). `.cursor()` is DuckDB's documented pattern: an independent connection
    over the SAME database.

    Keeps the 2026-07-06 RAM fix intact — memory_limit/threads are set on the DATABASE, so every cursor inherits
    the 512MB cap (that cap exists because DuckDB otherwise grows its buffer pool toward 80% of RAM and starves
    this 8GB box).
    """
    global _duck
    if _duck is None:
        with _duck_lock:
            if _duck is None:                     # double-checked: only one thread builds it
                import duckdb
                d = duckdb.connect(database=":memory:")
                try:
                    d.execute("SET memory_limit='512MB'")
                    d.execute("SET threads=2")    # stop each worker grabbing all cores
                except Exception:
                    pass
                _duck = d
    cur = getattr(_tls, "cur", None)
    if cur is None:
        cur = _duck.cursor()
        _tls.cur = cur
    return cur


def _lake_root() -> Path:
    return Path(settings.LAKE_PATH)


def _sanitize(s: str) -> str:
    """Filesystem-safe partition token (symbols may contain spaces, e.g. 'INDIA VIX')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).upper()).strip("_") or "UNKNOWN"


def candles_dir(inst_type: str, symbol: str, interval: str) -> Path:
    return (
        _lake_root() / "candles"
        / f"type={_sanitize(inst_type)}"
        / f"symbol={_sanitize(symbol)}"
        / f"interval={interval}"
    )


def _year_file(inst_type: str, symbol: str, interval: str, year: int) -> Path:
    return candles_dir(inst_type, symbol, interval) / f"{year}.parquet"


def _files_for(symbol: str, interval: str) -> list[str]:
    """All parquet files for a symbol+interval across instrument types."""
    root = _lake_root() / "candles"
    if not root.exists():
        return []
    pattern = f"type=*/symbol={_sanitize(symbol)}/interval={interval}/*.parquet"
    return [str(p) for p in root.glob(pattern)]


def write_candles(df: pd.DataFrame, inst_type: str, symbol: str, interval: str) -> int:
    """
    Write candle rows (full CANDLE_COLS schema) into the lake, partitioned by
    year, idempotently. Returns number of rows written across partitions.
    """
    if df is None or df.empty:
        return 0

    df = df.copy()
    # Guarantee schema + dtypes
    for col in CANDLE_COLS:
        if col not in df.columns:
            df[col] = None
    df["ts"] = pd.to_datetime(df["ts"])
    df = df[CANDLE_COLS]

    total = 0
    for year, grp in df.groupby(df["ts"].dt.year):
        path = _year_file(inst_type, symbol, interval, int(year))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            grp = pd.concat([existing, grp], ignore_index=True)
        grp = (
            grp.drop_duplicates(subset=["ts"], keep="last")
               .sort_values("ts")
               .reset_index(drop=True)
        )
        grp.to_parquet(path, index=False)
        total += len(grp)
    logger.info("lake: wrote %s %s (%s) → %d rows", symbol, interval, inst_type, total)
    return total


def read_candles(
    symbol: str, interval: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """Read candles for symbol+interval within [start, end] (inclusive), chronological."""
    files = _files_for(symbol, interval)
    if not files:
        return pd.DataFrame(columns=CANDLE_COLS)

    rows = _con().execute(
        """
        SELECT instrument_key, symbol, underlying_symbol, ts, interval,
               open, high, low, close, volume, oi
        FROM read_parquet(?)
        WHERE ts >= ? AND ts <= ?
        ORDER BY ts
        """,
        [files, start, end],
    ).fetchdf()
    return rows


def stored_minmax(symbol: str, interval: str) -> Optional[tuple[datetime, datetime]]:
    """(min_ts, max_ts) stored for symbol+interval, or None if nothing stored."""
    files = _files_for(symbol, interval)
    if not files:
        return None
    row = _con().execute(
        "SELECT min(ts), max(ts) FROM read_parquet(?)", [files]
    ).fetchone()
    if not row or row[0] is None:
        return None
    return (row[0], row[1])


def to_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.to_datetime(d).date()


# ── Coverage ledger ─────────────────────────────────────────────────
# Records which date ranges have actually been FETCHED per (symbol, interval),
# independent of whether bars exist (holidays/weekends have none). Without this,
# a request that starts on a non-trading day or ends "today" would re-fetch the
# edges forever, because the data min/max never equals the requested bounds.

def _coverage_file(symbol: str, interval: str) -> Path:
    return _lake_root() / "coverage" / f"{_sanitize(symbol)}__{interval}.json"


def get_coverage(symbol: str, interval: str) -> list[tuple[date, date]]:
    p = _coverage_file(symbol, interval)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text())
        return [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in raw]
    except Exception:
        return []


def _merge(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Merge overlapping/adjacent (≤1 day gap) date ranges."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    out = [ranges[0]]
    for s, e in ranges[1:]:
        ls, le = out[-1]
        if s <= le + timedelta(days=1):
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def add_coverage(symbol: str, interval: str, frm: date, to: date) -> None:
    merged = _merge(get_coverage(symbol, interval) + [(frm, to)])
    p = _coverage_file(symbol, interval)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([[s.isoformat(), e.isoformat()] for s, e in merged]))


def forget_coverage_after(symbol: str, interval: str, cutoff: date) -> None:
    """
    Drop coverage on/after `cutoff` so the next fetch RE-REQUESTS that window.
    Heals the "marked covered before the bars published" trap: e.g. an --update
    fired over the weekend gets 0 bars + complete=True for Friday and marks it
    covered; when Friday's data later publishes it would be skipped forever.
    Re-opening a rolling recent tail on each update guarantees it gets caught.
    """
    trimmed: list[tuple[date, date]] = []
    for frm, to in get_coverage(symbol, interval):
        if to < cutoff:
            trimmed.append((frm, to))
        elif frm < cutoff:
            trimmed.append((frm, cutoff - timedelta(days=1)))
        # else: range lies fully on/after cutoff → drop it
    p = _coverage_file(symbol, interval)
    if trimmed:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([[s.isoformat(), e.isoformat()] for s, e in trimmed]))
    elif p.exists():
        p.unlink()


def missing_ranges(symbol: str, interval: str, frm: date, to: date) -> list[tuple[date, date]]:
    """Sub-ranges of [frm, to] not yet covered by the ledger."""
    covered = get_coverage(symbol, interval)
    gaps: list[tuple[date, date]] = []
    cursor = frm
    for cs, ce in covered:
        if ce < cursor or cs > to:
            continue
        if cs > cursor:
            gaps.append((cursor, min(cs - timedelta(days=1), to)))
        cursor = max(cursor, ce + timedelta(days=1))
        if cursor > to:
            break
    if cursor <= to:
        gaps.append((cursor, to))
    return gaps
