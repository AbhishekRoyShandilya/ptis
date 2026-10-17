"""
marketdata.instruments — the append-only instrument dimension.

One row per tradable contract (equity, index, future, option), sourced from
Upstox's public instrument master (`complete.csv.gz`). Loaded in full now —
including options — so option-chain ingestion later needs no dimension work.

Dimension schema (data/lake/instruments.parquet):
    instrument_key, symbol, name, instrument_type (EQ|INDEX|FUT|OPT),
    underlying_symbol, expiry, strike, option_type, lot_size, tick_size,
    exchange, active

Also provides cheap symbol classification used by the candle write path:
    classify(symbol) -> (inst_type, underlying_symbol)
    instrument_key(symbol) -> Optional[str]
"""
from __future__ import annotations

import csv
import gzip
import io
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import settings
from ingestion.broker_adapters.upstox import UpstoxClient

logger = logging.getLogger(__name__)

DIM_COLS = [
    "instrument_key", "symbol", "name", "instrument_type",
    "underlying_symbol", "expiry", "strike", "option_type",
    "lot_size", "tick_size", "exchange", "active",
]

# Spot indices we treat as type=INDEX (underlying = itself). Used both for
# classification and as a fallback when the dimension parquet isn't built yet.
INDEX_SYMBOLS = {
    "NIFTY", "NIFTY 50", "NIFTY50", "BANKNIFTY", "NIFTY BANK", "FINNIFTY",
    "NIFTY FIN SERVICE", "MIDCPNIFTY", "NIFTY MID SELECT", "INDIA VIX", "VIX",
    "SENSEX", "BANKEX", "NIFTY IT", "NIFTY AUTO", "NIFTY PHARMA", "NIFTY FMCG",
    "NIFTY METAL", "NIFTY REALTY",
}

_cache_df: Optional[pd.DataFrame] = None
_cache_lookup: Optional[dict] = None


def _parquet_path() -> Path:
    return Path(settings.LAKE_PATH) / "instruments.parquet"


def _map_type(raw_type: str, option_type: str) -> tuple[str, str]:
    """Map Upstox (instrument_type, option_type) → (our type, option_type)."""
    t = (raw_type or "").upper()
    ot = (option_type or "").upper()
    if t == "EQUITY":
        return "EQ", ""
    if t == "INDEX":
        return "INDEX", ""
    if t in ("CE", "PE") or ot in ("CE", "PE"):
        return "OPT", (ot or t)
    if t in ("FUT", "FUTIDX", "FUTSTK"):
        return "FUT", ""
    return t or "EQ", ot


def _parse_expiry(raw) -> Optional[date]:
    if raw is None or raw == "":
        return None
    s = str(raw).strip()
    # Epoch milliseconds?
    if s.isdigit():
        try:
            return datetime.fromtimestamp(int(s) / 1000).date()
        except Exception:
            return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def refresh_instruments(url: Optional[str] = None) -> int:
    """Download + parse the Upstox instrument master → instruments.parquet. Returns row count."""
    import httpx

    url = url or UpstoxClient.INSTRUMENTS_URL
    logger.info("instruments: downloading master from %s", url)
    resp = httpx.get(url, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    content = gzip.decompress(resp.content).decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))

    rows = []
    for r in reader:
        key = r.get("instrument_key", "")
        tsym = (r.get("tradingsymbol", "") or "").upper()
        if not key or not tsym:
            continue
        inst_type, opt_type = _map_type(r.get("instrument_type", ""), r.get("option_type", ""))
        name = (r.get("name", "") or "").upper()
        exchange = r.get("exchange", "")

        if inst_type in ("EQ", "INDEX"):
            symbol = tsym.removesuffix("-EQ")
            underlying = symbol
        else:  # FUT / OPT — name is the underlying
            symbol = tsym
            underlying = name or tsym

        try:
            strike = float(r.get("strike") or 0) or None
        except (TypeError, ValueError):
            strike = None
        try:
            lot_size = int(float(r.get("lot_size") or 0)) or None
        except (TypeError, ValueError):
            lot_size = None
        try:
            tick_size = float(r.get("tick_size") or 0) or None
        except (TypeError, ValueError):
            tick_size = None

        rows.append({
            "instrument_key": key,
            "symbol": symbol,
            "name": name,
            "instrument_type": inst_type,
            "underlying_symbol": underlying,
            "expiry": _parse_expiry(r.get("expiry")),
            "strike": strike,
            "option_type": opt_type or None,
            "lot_size": lot_size,
            "tick_size": tick_size,
            "exchange": exchange,
            "active": True,
        })

    df = pd.DataFrame(rows, columns=DIM_COLS)
    out = _parquet_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    global _cache_df, _cache_lookup
    _cache_df, _cache_lookup = None, None  # invalidate
    counts = df["instrument_type"].value_counts().to_dict()
    logger.info("instruments: wrote %d rows → %s | by type: %s", len(df), out, counts)
    return len(df)


def _load() -> pd.DataFrame:
    global _cache_df
    if _cache_df is None:
        p = _parquet_path()
        _cache_df = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=DIM_COLS)
    return _cache_df


def _lookup() -> dict:
    """symbol(upper) -> (instrument_type, underlying_symbol, instrument_key) for EQ+INDEX."""
    global _cache_lookup
    if _cache_lookup is None:
        df = _load()
        m: dict = {}
        if not df.empty:
            eqi = df[df["instrument_type"].isin(["EQ", "INDEX"])]
            for _, row in eqi.iterrows():
                sym = str(row["symbol"]).upper()
                exch = str(row["exchange"] or "")
                entry = (row["instrument_type"], row["underlying_symbol"], row["instrument_key"], exch)
                prev = m.get(sym)
                # Prefer NSE over BSE for the canonical key (NSE = liquid history).
                if prev is None or (exch.startswith("NSE") and not str(prev[3]).startswith("NSE")):
                    m[sym] = entry
        _cache_lookup = m
    return _cache_lookup


def classify(symbol: str) -> tuple[str, str]:
    """
    (instrument_type, underlying_symbol) for a spot symbol. Falls back to a
    hardcoded index set + EQ default when the dimension isn't built yet.
    """
    up = symbol.upper()
    hit = _lookup().get(up)
    if hit:
        return hit[0], hit[1]
    if up in INDEX_SYMBOLS:
        return "INDEX", symbol
    return "EQ", symbol


def instrument_key(symbol: str) -> Optional[str]:
    hit = _lookup().get(symbol.upper())
    return hit[2] if hit else None
