"""
marketdata.features — enriched feature store (Layer 1 of the quant lakehouse).

For each (symbol, timeframe, bar) we PRE-COMPUTE indicators ONCE and store them,
so nothing recomputes MA(200)/ATR/RSI on every query. Read by backtests, the
feature/judgment layers, regime detection and research.

QUANT-GRADE INVARIANTS:
  • Point-in-time / no-lookahead: every indicator here is *causal* — ewm/rolling/
    cumsum use only the current and prior bars, so a row for ts T contains only
    what was knowable at T's close. No future bar ever leaks in.
  • Versioned: every row carries FEATURE_VERSION so results reproduce when the
    feature definition changes (bump the version → new rows, old ones preserved).
  • Immutable/append-only: stored in the Parquet lake (lake/features/...),
    partitioned like candles; idempotent year-file merge on ts.

This is the ONE place enriched indicators are computed for storage. The 12-dim
similarity *fingerprint* (discovery.event_extractor.compute_fingerprint) is a
separate, on-demand layer; FINGERPRINT_DIMS is re-exported here so callers share
one canonical name list.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata import lake
from strategy_engine.indicators import IndicatorEngine

logger = logging.getLogger(__name__)

# Bump when the feature definition changes. Old rows keep their version.
FEATURE_VERSION = "1.0"

# Canonical 12-dim similarity fingerprint dimension names (must match
# discovery.event_extractor.compute_fingerprint ordering). Shared so the
# feature store, snapshots, events and judgment layer agree on one vocabulary.
FINGERPRINT_DIMS = [
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "volume_surge",
    "atr_expansion", "momentum_score", "consolidation", "range_compression",
    "trend_alignment", "rsi_norm", "vwap_norm", "rel_volume",
]

# Stored feature columns (the enriched indicator set).
FEATURE_COLS = [
    "symbol", "interval", "ts", "feature_version",
    "ema9", "ema20", "ema50", "ema200", "sma20", "sma50",
    "atr", "atr_pct", "rsi", "adx",
    "bb_upper", "bb_mid", "bb_lower", "bb_width_pct",
    "supertrend_dir",
    "rvol", "vwap", "vwap_dist_atr",
    "mom_5", "mom_20",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
]


def _session_vwap(df: pd.DataFrame, ts: pd.Series | None = None) -> pd.Series:
    """Intraday VWAP that resets each trading day (causal cumsum within session).
    PERF 2026-09-23: `.dt.date` builds one Python date OBJECT per bar (160k iterations per slot across the universe,
    0.8 s). `.dt.normalize()` gives the SAME day groups as datetime64, vectorised. `ts` lets the caller pass the
    already-converted column instead of re-running to_datetime."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    t = ts if ts is not None else pd.to_datetime(df["ts"])
    day = t.to_numpy().astype("datetime64[D]")     # same day groups, no Series/accessor overhead
    cum_tpv = (tp * df["volume"]).groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return cum_tpv / cum_vol


def compute_features(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """
    Compute the enriched, point-in-time feature row for every bar in df.
    df must have columns: ts, open, high, low, close, volume (chronological).
    Returns a DataFrame with FEATURE_COLS.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    df = df.sort_values("ts").reset_index(drop=True)
    ts_conv = pd.to_datetime(df["ts"])                     # PERF 2026-09-23: convert ONCE, reuse everywhere below
    close, high, low, vol, opn = df["close"], df["high"], df["low"], df["volume"], df["open"]
    ie = IndicatorEngine

    # PERF 2026-09-23: build every column into a dict and construct the frame ONCE. Assigning 31 columns one by one
    # made pandas insert a block per column (2,759 __setitem__ calls per slot across the universe, ~0.8 s). Values and
    # order are unchanged - the identity test (git HEAD vs this file, all columns, exact) passes.
    ie = IndicatorEngine
    cols = {}
    cols["ts"] = ts_conv.values

    cols["ema9"] = ie.ema(close, 9)
    cols["ema20"] = ie.ema(close, 20)
    cols["ema50"] = ie.ema(close, 50)
    cols["ema200"] = ie.ema(close, 200)
    cols["sma20"] = ie.sma(close, 20)
    cols["sma50"] = ie.sma(close, 50)

    atr = ie.atr(df, 14)
    cols["atr"] = atr
    cols["atr_pct"] = atr / close.replace(0, np.nan)
    cols["rsi"] = ie.rsi(close, 14)
    cols["adx"] = ie.adx(df, 14)

    bb_u, bb_m, bb_l, bb_w = ie.bollinger_bands(close, 20, 2.0)
    cols["bb_upper"], cols["bb_mid"], cols["bb_lower"], cols["bb_width_pct"] = bb_u, bb_m, bb_l, bb_w

    try:
        st = ie.supertrend(df, 10, 3.0)
        cols["supertrend_dir"] = st["st_dir"].values
    except Exception:
        cols["supertrend_dir"] = np.zeros(len(df), dtype=int)

    cols["rvol"] = vol / vol.rolling(20).mean().replace(0, np.nan)
    vwap = _session_vwap(df, ts_conv)
    cols["vwap"] = vwap
    cols["vwap_dist_atr"] = (close - vwap) / atr.replace(0, np.nan)

    cols["mom_5"] = close.pct_change(5)
    cols["mom_20"] = close.pct_change(20)

    rng = (high - low).replace(0, np.nan)
    cols["body_ratio"] = (close - opn).abs() / rng
    cols["upper_wick_ratio"] = (high - np.maximum(opn, close)) / rng
    cols["lower_wick_ratio"] = (np.minimum(opn, close) - low) / rng

    out = pd.DataFrame(cols, index=df.index)
    out["symbol"] = symbol
    out["interval"] = interval
    out["feature_version"] = FEATURE_VERSION
    return out[FEATURE_COLS]


# ── Lake persistence (lake/features/type=.../symbol=.../interval=.../{year}.parquet) ──

def _features_dir(inst_type: str, symbol: str, interval: str) -> Path:
    return (
        lake._lake_root() / "features"
        / f"type={lake._sanitize(inst_type)}"
        / f"symbol={lake._sanitize(symbol)}"
        / f"interval={interval}"
    )


def _feature_files(symbol: str, interval: str) -> list[str]:
    root = lake._lake_root() / "features"
    if not root.exists():
        return []
    pattern = f"type=*/symbol={lake._sanitize(symbol)}/interval={interval}/*.parquet"
    return [str(p) for p in root.glob(pattern)]


def write_features(feat: pd.DataFrame, inst_type: str, symbol: str, interval: str) -> int:
    """Idempotent year-partitioned write (merge on ts, keep last)."""
    if feat is None or feat.empty:
        return 0
    feat = feat.copy()
    feat["ts"] = pd.to_datetime(feat["ts"])
    total = 0
    for year, grp in feat.groupby(feat["ts"].dt.year):
        path = _features_dir(inst_type, symbol, interval) / f"{int(year)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            grp = pd.concat([pd.read_parquet(path), grp], ignore_index=True)
        grp = grp.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)
        grp.to_parquet(path, index=False)
        total += len(grp)
    logger.info("features: wrote %s %s → %d rows (v%s)", symbol, interval, total, FEATURE_VERSION)
    return total


def read_features(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    files = _feature_files(symbol, interval)
    if not files:
        return pd.DataFrame(columns=FEATURE_COLS)
    return lake._con().execute(
        "SELECT * FROM read_parquet(?) WHERE ts >= ? AND ts <= ? ORDER BY ts",
        [files, start, end],
    ).fetchdf()


def compute_and_store(symbol: str, interval: str, start, end) -> int:
    """
    Pull candles via the CandleStore, compute enriched features, store to lake.
    Returns rows written. (Indicator warm-up: we fetch from `start`; early rows
    will have NaN long-window indicators — expected and correct.)
    """
    from marketdata.candle_store import get_candle_store
    from marketdata import instruments

    candles = get_candle_store().get(symbol, interval, start, end)
    if candles is None or candles.empty:
        return 0
    feat = compute_features(candles, symbol, interval)
    inst_type, _ = instruments.classify(symbol)
    return write_features(feat, inst_type, symbol, interval)


# The intervals consumers actually read (the eye trades 5m off 1h/1d HTF; scanner/AVT use 15m/1h).
FEATURE_INTERVALS = ["5m", "15m", "1h", "1d"]


def backfill_features(symbols, intervals=None, start=None, to=None, progress: int = 25) -> int:
    """BULK-populate the Feature Store so consumers READ instead of recompute (the cross-consumer payoff
    of the store). For every (symbol, interval): pull candles, compute features, persist. Idempotent
    (write keeps-last) and safe to re-run. Warm-up sits at the genuine start of history (frm = the data
    floor), so persisted rows are fully warmed for any window a consumer asks for. Returns total rows."""
    from datetime import date
    intervals = list(intervals or FEATURE_INTERVALS)
    to = to or date.today()
    symbols = list(symbols)
    total, done, n = 0, 0, len(symbols) * len(intervals)
    for sym in symbols:
        for iv in intervals:
            frm = start or (date(2000, 1, 1) if iv == "1d" else date(2022, 1, 1))
            try:
                total += compute_and_store(sym, iv, frm, to)
            except Exception:
                logger.exception("feature backfill failed: %s %s", sym, iv)
            done += 1
            if progress and done % progress == 0:
                logger.info("features backfill … %d/%d done (%d rows)", done, n, total)
    logger.info("features backfill complete: %d (symbol,interval) pairs → %d rows", n, total)
    return total


def get_features(candles: pd.DataFrame, symbol: str, interval: str,
                 inst_type: str | None = None, persist: bool = True) -> pd.DataFrame:
    """THE read-through entry to the Feature Store (see project_brain/08). Returns the feature rows for
    `candles`, row-aligned to it. Serves PERSISTED rows at the current FEATURE_VERSION (no compute); on
    a miss/stale-version the Feature ENGINE (`compute_features`) computes — authoritative — persists,
    and returns. Bit-identical to `compute_features(candles)` by construction.

    Invariant (08): the Engine is the source of truth; the Store is a cache of its output. Consumers
    call THIS, never `compute_features` directly. (Caches CLOSED bars; a forming bar simply misses and
    is computed live until it closes and a later pass persists it.)"""
    if candles is None or len(candles) == 0:
        return compute_features(candles, symbol, interval)
    c = candles.sort_values("ts").reset_index(drop=True)
    ts = pd.to_datetime(c["ts"])
    try:
        stored = read_features(symbol, interval, ts.iloc[0], ts.iloc[-1])
        if not stored.empty and "feature_version" in stored.columns:
            stored = stored[stored["feature_version"].astype(str) == str(FEATURE_VERSION)]
        if not stored.empty:
            stored = stored.drop_duplicates(subset=["ts"], keep="last")
            have = pd.to_datetime(stored["ts"])
            if set(ts).issubset(set(have)):                       # FULL HIT — serve, no compute
                sidx = stored.set_index(have)
                return sidx.reindex(ts.values).reset_index(drop=True)[FEATURE_COLS]
    except Exception:
        logger.exception("feature store read failed — falling back to compute")
    feat = compute_features(c, symbol, interval)                 # MISS → authoritative compute
    if persist:
        try:
            from marketdata import instruments
            it = inst_type or instruments.classify(symbol)[0]
            write_features(feat, it, symbol, interval)
        except Exception:
            logger.exception("feature store persist failed (non-fatal)")
    return feat
