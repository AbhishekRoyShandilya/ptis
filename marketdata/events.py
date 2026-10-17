"""
marketdata.events — market event store (Layer 4 of the lakehouse).

Humans think "a breakout happened", not "candle #43 formed". This persists
discrete, queryable market EVENTS (compression, expansion, breakout, gap,
volume spike, trend flip, wick reversal) so research/AVT can ask "what happened
after similar events?".

REUSE, not rebuild: events are COMPOSED from the already-computed enriched
feature store (marketdata/features.py — rvol, bb_width, atr_pct, supertrend_dir,
body/wick ratios) plus raw candles. No new indicator math. Richer detectors
(intelligence/breakout_detector.py, strategy_engine/candle_patterns.py) remain
available to layer in later.

QUANT-GRADE: point-in-time (every rule input is causal), EVENT_VERSION stamped,
immutable append-only lake (lake/events/).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata import lake, features as F

logger = logging.getLogger(__name__)

EVENT_VERSION = "1.0"

EVENT_COLS = [
    "symbol", "interval", "ts", "event_type", "direction",
    "confidence", "metadata", "event_version",
]

# Thresholds (transparent, tunable).
_VOL_SPIKE = 2.0          # rvol
_GAP_PCT = 0.01           # 1% gap vs prev close
_WICK_REVERSAL = 0.60     # wick ratio
_BREAKOUT_LOOKBACK = 20


def detect_events(candles: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """
    Compose discrete events from candles + the enriched feature row. Returns a
    long DataFrame (one row per detected event) with EVENT_COLS.
    """
    if candles is None or len(candles) < 25:
        return pd.DataFrame(columns=EVENT_COLS)

    c = candles.sort_values("ts").reset_index(drop=True)
    feat = F.compute_features(c, symbol, interval)

    ts = pd.to_datetime(c["ts"])
    close, high, low, openp = c["close"], c["high"], c["low"], c["open"]
    prev_close = close.shift(1)

    rvol = feat["rvol"]
    bb_w = feat["bb_width_pct"]
    atr_pct = feat["atr_pct"]
    st_dir = feat["supertrend_dir"]
    upw, low_w = feat["upper_wick_ratio"], feat["lower_wick_ratio"]

    rows: list[dict] = []

    def emit(mask, event_type, direction_fn, conf):
        idx = np.where(mask.fillna(False).values)[0]
        for i in idx:
            rows.append({
                "symbol": symbol, "interval": interval, "ts": ts.iloc[i],
                "event_type": event_type,
                "direction": direction_fn(i) if callable(direction_fn) else direction_fn,
                "confidence": float(np.clip(conf.iloc[i] if hasattr(conf, "iloc") else conf, 0, 1)),
                "metadata": {}, "event_version": EVENT_VERSION,
            })

    # Volume spike
    emit(rvol >= _VOL_SPIKE, "volume_spike", "neutral", (rvol / 4.0).clip(0, 1))

    # Gaps (vs prev close)
    gap = (openp - prev_close) / prev_close.replace(0, np.nan)
    emit(gap >= _GAP_PCT, "gap_up", "bullish", (gap / 0.03).clip(0, 1))
    emit(gap <= -_GAP_PCT, "gap_down", "bearish", (gap.abs() / 0.03).clip(0, 1))

    # Compression: BB width in the bottom 20% of its rolling history
    bb_lo = bb_w.rolling(60, min_periods=20).quantile(0.20)
    emit(bb_w <= bb_lo, "compression", "neutral", (1.0 - bb_w / bb_lo.replace(0, np.nan)).clip(0, 1))

    # Volatility expansion: ATR% well above its rolling mean
    atr_mean = atr_pct.rolling(20, min_periods=10).mean()
    atr_ratio = atr_pct / atr_mean.replace(0, np.nan)
    emit(atr_pct >= 1.5 * atr_mean, "atr_expansion", "neutral", (atr_pct / (2 * atr_mean.replace(0, np.nan))).clip(0, 1))

    # Breakout: close clears prior N-bar high/low, CONFIRMED by volume OR a
    # volatility/ATR thrust. Indices report no intraday volume, so the ATR thrust
    # is the confirmation there — without this, breakouts never fire on NIFTY/etc.
    prior_high = high.shift(1).rolling(_BREAKOUT_LOOKBACK).max()
    prior_low = low.shift(1).rolling(_BREAKOUT_LOOKBACK).min()
    confirmed = (rvol >= 1.5) | (atr_ratio >= 1.3)
    brk_conf = pd.concat([rvol / 3.0, atr_ratio / 2.0], axis=1).max(axis=1).clip(0, 1)
    emit((close > prior_high) & confirmed, "breakout", "bullish", brk_conf)
    emit((close < prior_low) & confirmed, "breakout", "bearish", brk_conf)

    # Trend flip (supertrend direction change)
    flip = st_dir != st_dir.shift(1)
    emit(flip & (st_dir == 1), "trend_flip", "bullish", pd.Series(0.6, index=c.index))
    emit(flip & (st_dir == -1), "trend_flip", "bearish", pd.Series(0.6, index=c.index))

    # Wick reversal (rejection)
    emit(upw >= _WICK_REVERSAL, "wick_reversal", "bearish", upw)
    emit(low_w >= _WICK_REVERSAL, "wick_reversal", "bullish", low_w)

    if not rows:
        return pd.DataFrame(columns=EVENT_COLS)
    return pd.DataFrame(rows)[EVENT_COLS]


# ── Lake persistence (lake/events/symbol=.../interval=.../{year}.parquet) ──

def _events_dir(symbol: str, interval: str) -> Path:
    return (
        lake._lake_root() / "events"
        / f"symbol={lake._sanitize(symbol)}"
        / f"interval={interval}"
    )


def write_events(ev: pd.DataFrame, symbol: str, interval: str) -> int:
    if ev is None or ev.empty:
        return 0
    ev = ev.copy()
    ev["ts"] = pd.to_datetime(ev["ts"])
    # metadata must be JSON-serializable for parquet; store as JSON string
    import json
    ev["metadata"] = ev["metadata"].apply(lambda m: json.dumps(m) if not isinstance(m, str) else m)
    total = 0
    for year, grp in ev.groupby(ev["ts"].dt.year):
        path = _events_dir(symbol, interval) / f"{int(year)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            grp = pd.concat([pd.read_parquet(path), grp], ignore_index=True)
        grp = grp.drop_duplicates(subset=["ts", "event_type", "direction"], keep="last").sort_values("ts")
        grp.to_parquet(path, index=False)
        total += len(grp)
    logger.info("events: wrote %s %s → %d events", symbol, interval, total)
    return total


def read_events(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    d = _events_dir(symbol, interval)
    files = [str(p) for p in d.glob("*.parquet")] if d.exists() else []
    if not files:
        return pd.DataFrame(columns=EVENT_COLS)
    return lake._con().execute(
        "SELECT * FROM read_parquet(?) WHERE ts >= ? AND ts <= ? ORDER BY ts",
        [files, start, end],
    ).fetchdf()


def detect_and_store(symbol: str, interval: str, start, end) -> int:
    from marketdata.candle_store import get_candle_store
    candles = get_candle_store().get(symbol, interval, start, end)
    ev = detect_events(candles, symbol, interval)
    return write_events(ev, symbol, interval)
