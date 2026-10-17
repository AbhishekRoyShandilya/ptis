"""
intelligence.story_context — V2.2: the ambient CONTEXT-STATE axis.

A human setup is "(context) + sequence + trigger": the context is a STATE that
holds *during* the episode, not an event that fires. Frozen, coarse, point-in-time
booleans sampled at the trigger bar (plan principle 5 — reused definitions, never
tuned against outcomes):

  HTF_UP / HTF_DOWN  — higher-timeframe trend (completed HTF bars only)
  TIGHT_RANGE        — bb-width percentile ≤ 0.25 (the user's "tiny range")
  BETWEEN_MA_ST      — close trapped between EMA50 and the SuperTrend(10,3) LINE
                       (the user's "best to avoid" chop state)
  SESSION_OPEN       — first hour of the session
  SESSION_LAST       — last hour of the session
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONTEXT_STATES = ["HTF_UP", "HTF_DOWN", "TIGHT_RANGE", "BETWEEN_MA_ST",
                  "SESSION_OPEN", "SESSION_LAST"]


def htf_trend_series(candles: pd.DataFrame, symbol: str, interval: str) -> np.ndarray:
    """Per-bar higher-TF trend as 'up'/'down'/'flat' — completed HTF bars only (point-in-time, no
    lookahead). THE single source of HTF trend: both `context_frame` and the Eye consume this, so HTF
    is computed ONCE (not refetched per bar). Owner of the HTF capability (project_brain/01)."""
    from intelligence.setup_features import _htf_trend
    c = candles.sort_values("ts").reset_index(drop=True)
    ts = pd.to_datetime(c["ts"])
    htf = _htf_trend(symbol, interval, c["ts"].iloc[0], c["ts"].iloc[-1])
    if htf is not None and not htf.empty:
        left = pd.DataFrame({"ts": ts.astype("datetime64[ns]")})
        htf = htf.copy()
        htf["htf_complete"] = pd.to_datetime(htf["htf_complete"]).astype("datetime64[ns]")
        merged = pd.merge_asof(left.sort_values("ts"), htf,
                               left_on="ts", right_on="htf_complete", direction="backward")
        return merged["htf_trend"].fillna("flat").to_numpy()
    return np.array(["flat"] * len(c))


def context_frame(candles: pd.DataFrame, symbol: str, interval: str,
                  feat: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-bar context-state frame (ts + CONTEXT_STATES as 0/1). Point-in-time. `feat` may be supplied
    (already computed on this SAME sorted `candles`) to skip recompute — bit-identical."""
    from marketdata import features as F
    from intelligence.setup_features import _htf_trend, HTF_DELTA, HTF

    if candles is None or len(candles) < 80:
        return pd.DataFrame()
    if feat is None:
        c = candles.sort_values("ts").reset_index(drop=True)
        feat = F.compute_features(c, symbol, interval)   # window-warmed (validated); not the shared store (08)
    else:
        c = candles
    ts = pd.to_datetime(c["ts"])

    out = pd.DataFrame({"ts": ts})

    # ── HTF trend (completed higher-TF bars only — no lookahead) — via the single source ──
    trend = htf_trend_series(c, symbol, interval)
    out["HTF_UP"] = trend == "up"
    out["HTF_DOWN"] = trend == "down"

    # ── tight range ──
    bb_pctile = feat["bb_width_pct"].rolling(100, min_periods=30).rank(pct=True)
    out["TIGHT_RANGE"] = bb_pctile <= 0.25

    # ── trapped between EMA50 and the SuperTrend line ──
    try:
        from strategy_engine.indicators import IndicatorEngine
        st = IndicatorEngine.supertrend(c[["open", "high", "low", "close"]].astype(float),
                                        period=10, multiplier=3.0)
        st_line = st["st_value"].to_numpy(float)
    except Exception as e:
        logger.warning("supertrend line failed %s: %s", symbol, e)
        st_line = np.full(len(c), np.nan)
    close = c["close"].to_numpy(float)
    ema50 = feat["ema50"].to_numpy(float)
    lo = np.minimum(ema50, st_line); hi = np.maximum(ema50, st_line)
    out["BETWEEN_MA_ST"] = (close >= lo) & (close <= hi)

    # ── session time ──
    mins = (ts - ts.dt.normalize()).dt.total_seconds() / 60.0
    start = 9 * 60 + 15; end = 15 * 60 + 30
    out["SESSION_OPEN"] = (mins >= start) & (mins < start + 60)
    out["SESSION_LAST"] = (mins >= end - 60) & (mins < end)

    for col in CONTEXT_STATES:
        out[col] = out[col].fillna(False).astype(int)
    return out
