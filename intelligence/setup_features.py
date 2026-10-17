"""
intelligence.setup_features — the CHARACTERIZE layer (rich, human-legible precursor).

For move-first discovery we don't cluster on continuous features; we describe each
bar with the **interpretable conditions a trader actually checks** before a move:
MTF regime, current regime, support/resistance & clear-air, breakout-from-
consolidation, higher-high-range-break-with-volume, MA position, momentum, candle.

Every condition is a 0/1 flag computed VECTORISED over all bars (the miner needs
them everywhere), strictly point-in-time. The higher-timeframe trend uses only
*completed* HTF bars (merged on the HTF close-time) — no lookahead.

Returns (frame, condition_cols). UP_CONDITIONS / DOWN_CONDITIONS are the
direction-relevant subsets the miner enumerates over (keeps combinatorics sane).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from marketdata import features as F

logger = logging.getLogger(__name__)

SETUP_FEATURE_VERSION = "1.0"

# base interval → higher timeframe used for MTF context
HTF = {"5m": "1h", "15m": "1h", "1h": "1d"}   # 5m/15m trade off the 1h regime (user's framing)
HTF_DELTA = {"1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1)}

# direction-relevant condition subsets (what the miner combines)
UP_CONDITIONS = [
    "above_50ma", "above_200ma", "st_up", "htf_up", "mtf_aligned_up",
    "broke_hh", "clear_air_above", "at_range_high", "volume_surge",
    "expanding", "coiled", "strong_trend", "mom_up", "bull_bar",
    "hammer_like", "rsi_oversold", "rsi_bull_div", "near_support",
]
DOWN_CONDITIONS = [
    "below_50ma", "below_200ma", "st_down", "htf_down", "mtf_aligned_down",
    "broke_ll", "clear_air_below", "at_range_low", "volume_surge",
    "expanding", "coiled", "strong_trend", "mom_down", "bear_bar",
    "shooting_like", "rsi_overbought", "rsi_bear_div", "near_resistance",
]


def _htf_trend(symbol: str, interval: str, span_start, span_end):
    """Higher-TF trend per HTF bar, with the bar's CLOSE-time (for point-in-time join)."""
    htf_iv = HTF.get(interval)
    if not htf_iv:
        return None
    from marketdata.candle_store import get_candle_store
    h = get_candle_store().get(symbol, htf_iv, pd.Timestamp(span_start) - pd.Timedelta(days=120),
                               pd.Timestamp(span_end))
    if h is None or len(h) < 60:
        return None
    h = h.sort_values("ts").reset_index(drop=True)
    hf = F.compute_features(h, symbol, htf_iv)
    ema = hf["ema50"]; slope = ema.diff()
    up = (h["close"] > ema) & (slope > 0)
    dn = (h["close"] < ema) & (slope < 0)
    trend = np.where(up, "up", np.where(dn, "down", "flat"))
    out = pd.DataFrame({"htf_trend": trend})
    out["htf_complete"] = pd.to_datetime(h["ts"]) + HTF_DELTA[htf_iv]   # only usable once closed
    return out.sort_values("htf_complete")


def setup_features(candles: pd.DataFrame, symbol: str, interval: str):
    """Per-bar legible condition frame + the list of condition columns."""
    if candles is None or len(candles) < 80:
        return pd.DataFrame(), []
    c = candles.sort_values("ts").reset_index(drop=True)
    feat = F.compute_features(c, symbol, interval)
    close, high, low, op = c["close"], c["high"], c["low"], c["open"]
    atr = feat["atr"].replace(0, np.nan)
    ema50, ema200 = feat["ema50"], feat["ema200"]

    bb_pctile = feat["bb_width_pct"].rolling(100, min_periods=30).rank(pct=True)
    atr_mean = feat["atr_pct"].rolling(20, min_periods=10).mean().replace(0, np.nan)
    atr_trend = feat["atr_pct"] / atr_mean
    rvol = feat["rvol"]
    rmin, rmax = low.rolling(20).min(), high.rolling(20).max()
    range_pos = (close - rmin) / (rmax - rmin).replace(0, np.nan)
    prior_hh = high.rolling(20).max().shift(1)
    prior_ll = low.rolling(20).min().shift(1)
    room_above = (prior_hh - close) / atr
    room_below = (close - prior_ll) / atr
    macc = feat["mom_5"] - feat["mom_5"].shift(5)
    rsi = feat["rsi"]
    body_ratio = feat["body_ratio"]; low_w = feat["lower_wick_ratio"]; up_w = feat["upper_wick_ratio"]

    df = pd.DataFrame({"symbol": symbol, "interval": interval, "ts": pd.to_datetime(c["ts"])})
    # ── MA position / trend ──
    df["above_50ma"] = close > ema50
    df["below_50ma"] = close < ema50
    df["above_200ma"] = close > ema200
    df["below_200ma"] = close < ema200
    df["st_up"] = feat["supertrend_dir"] == 1
    df["st_down"] = feat["supertrend_dir"] == -1
    df["strong_trend"] = feat["adx"] >= 25
    # ── volatility regime ──
    df["coiled"] = bb_pctile <= 0.20                       # consolidation
    df["expanding"] = atr_trend >= 1.10                    # volatility igniting
    # ── volume ──
    df["volume_surge"] = rvol >= 1.8
    # ── breakout / range structure ──
    df["broke_hh"] = close > prior_hh                      # higher-high range break
    df["broke_ll"] = close < prior_ll
    df["at_range_high"] = range_pos >= 0.80
    df["at_range_low"] = range_pos <= 0.20
    # ── support/resistance & clear-air ──
    df["clear_air_above"] = room_above >= 3.0
    df["clear_air_below"] = room_below >= 3.0
    df["near_resistance"] = room_above <= 1.0
    df["near_support"] = room_below <= 1.0
    # ── momentum / OB-OS ──
    df["mom_up"] = macc > 0
    df["mom_down"] = macc < 0
    df["rsi_overbought"] = rsi >= 70
    df["rsi_oversold"] = rsi <= 30
    df["rsi_bull_div"] = (low <= low.rolling(14).min()) & (rsi > rsi.rolling(14).min() + 5)
    df["rsi_bear_div"] = (high >= high.rolling(14).max()) & (rsi < rsi.rolling(14).max() - 5)
    # ── candle (vectorised proxies from wick/body ratios) ──
    df["bull_bar"] = (close > op) & (body_ratio >= 0.5)
    df["bear_bar"] = (close < op) & (body_ratio >= 0.5)
    df["hammer_like"] = (low_w >= 0.5) & (body_ratio <= 0.35)
    df["shooting_like"] = (up_w >= 0.5) & (body_ratio <= 0.35)

    # ── MTF: higher-TF trend, point-in-time (only completed HTF bars) ──
    htf = _htf_trend(symbol, interval, c["ts"].iloc[0], c["ts"].iloc[-1])
    if htf is not None and not htf.empty:
        left = df[["ts"]].copy()
        left["ts"] = pd.to_datetime(left["ts"]).astype("datetime64[ns]")
        htf = htf.copy()
        htf["htf_complete"] = pd.to_datetime(htf["htf_complete"]).astype("datetime64[ns]")
        merged = pd.merge_asof(left.sort_values("ts"), htf,
                               left_on="ts", right_on="htf_complete", direction="backward")
        htf_trend = merged["htf_trend"].fillna("flat").to_numpy()
    else:
        htf_trend = np.array(["flat"] * len(df))
    df["htf_up"] = htf_trend == "up"
    df["htf_down"] = htf_trend == "down"
    df["mtf_aligned_up"] = df["htf_up"] & df["st_up"] & df["above_50ma"]
    df["mtf_aligned_down"] = df["htf_down"] & df["st_down"] & df["below_50ma"]

    cond_cols = [col for col in df.columns if col not in ("symbol", "interval", "ts")]
    for col in cond_cols:
        df[col] = df[col].fillna(False).astype(int)
    return df, cond_cols


def conditions_for(direction: str) -> list:
    return UP_CONDITIONS if direction == "up" else DOWN_CONDITIONS
