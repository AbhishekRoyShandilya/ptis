"""
intelligence.eye_levels — the Eye's complete level-structure gatherer.

Salvaged VERBATIM from the retired GEN-1 `situation_assembler` (whose `assemble`/`narrate` were the superseded
parallel-evaluator, replaced by `cognition_service`). The Eye (`state_narrator._overhead_facts` via `_gather_levels`)
is the ONLY live consumer, so this keeps just that: every relevant structural level from every source (prior-day H/L,
opening range, VWAP, round numbers, multi-day swings), which rules can't enumerate but the read needs. Point-in-time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Level:
    name: str            # "prior-day high", "VWAP", "round 1500", "swing high"…
    price: float
    dist_atr: float      # signed: +above current price, −below
    side: str            # "above" | "below"


def _round_step(price: float) -> float:
    if price >= 20000: return 100.0
    if price >= 5000:  return 50.0
    if price >= 1000:  return 10.0
    if price >= 200:   return 5.0
    return 1.0


def _daily_swing_prices(c: pd.DataFrame, i: int, mult: float = 1.5) -> list[float]:
    """Multi-day swing highs/lows, computed at DAILY scale (resampled bars + DAILY ATR) — NOT the 5m
    zigzag. BUG FIX (2026-07-22, user-flagged): `atr_zigzag` retraces against the LOCAL bar's ATR, so
    feeding it a multi-week 5m series produced ~69 'structural' pivots for one SENSEX window (nearly all
    intraday noise), burying the one level a human would actually draw (opening-range high, ~78339,
    matching the user's hand-marked 78351.68 resistance) at position #28 by distance. A pivot worth
    remembering across DAYS needs a threshold computed on DAILY data, not the 5m ATR — "compression for
    5min is not compression for 1hr or 1day" applies here too. Only pivots CONFIRMED by `i`'s timestamp
    are used (causal)."""
    from intelligence.move_nature import atr_zigzag
    d = c.copy(); d["ts"] = pd.to_datetime(d["ts"]); d["day"] = d["ts"].dt.normalize()
    cutoff = d["ts"].iloc[i]
    daily = d.groupby("day").agg(open=("open", "first"), high=("high", "max"),
                                 low=("low", "min"), close=("close", "last")).reset_index()
    if len(daily) < 4:
        return []
    dh = daily["high"].to_numpy(float); dl = daily["low"].to_numpy(float); dc = daily["close"].to_numpy(float)
    tr = np.maximum(dh - dl, np.maximum(np.abs(dh - np.roll(dc, 1)), np.abs(dl - np.roll(dc, 1))))
    tr[0] = dh[0] - dl[0]
    datr = pd.Series(tr).rolling(5, min_periods=2).mean().shift(1).to_numpy()   # causal daily ATR
    datr = np.where(np.isfinite(datr), datr, np.nanmedian(tr) if np.isfinite(np.nanmedian(tr)) else np.nan)
    piv = atr_zigzag(dh, dl, datr, mult=mult)
    out = []
    for p in piv:
        # a daily pivot is knowable once its own day's bar has closed -- approximate conf as day-end
        conf_day = daily["day"].iloc[min(p.conf_idx, len(daily) - 1)]
        if conf_day + pd.Timedelta(days=1) <= cutoff.normalize() or (conf_day == cutoff.normalize()):
            out.append(p.price)
    return out


def _gather_levels(c: pd.DataFrame, feat: pd.DataFrame, i: int, price: float, a: float) -> list[Level]:
    """EVERY relevant level from EVERY source — the completeness that rules can't enumerate but
    the reasoner needs: prior-day H/L, multi-day swings, round numbers, VWAP, opening range."""
    h = c["high"].to_numpy(float); l = c["low"].to_numpy(float); cl = c["close"].to_numpy(float)
    ts = pd.to_datetime(c["ts"]); date = ts.dt.date.to_numpy()
    out: list[Level] = []

    def add(name, p):
        if p is None or not np.isfinite(p) or not np.isfinite(a) or a <= 0:
            return
        d = (p - price) / a
        out.append(Level(name, round(float(p), 2), round(float(d), 2), "above" if d >= 0 else "below"))

    days = list(dict.fromkeys(date.tolist())); today = date[i]
    # prior-day high/low
    if today in days and days.index(today) > 0:
        pm = date == days[days.index(today) - 1]
        add("prev-day high", float(h[pm].max())); add("prev-day low", float(l[pm].min()))
    # opening range (first ~6 bars of today)
    tm = np.where(date == today)[0]
    if len(tm) >= 3:
        add("opening-range high", float(h[tm[:6]].max())); add("opening-range low", float(l[tm[:6]].min()))
    # VWAP
    if "vwap" in feat:
        add("VWAP", float(feat["vwap"].to_numpy(float)[i]))
    # round numbers (nearest above & below)
    step = _round_step(price)
    add("round %g" % (np.ceil(price / step) * step), float(np.ceil(price / step) * step))
    add("round %g" % (np.floor(price / step) * step), float(np.floor(price / step) * step))
    # multi-day swing levels — DAILY-scale pivots (see _daily_swing_prices), not a 5m zigzag
    for p in _daily_swing_prices(c, i):
        add("swing (daily)", float(p))
    # dedupe near-identical levels (within 0.15 ATR), keep nearest-to-price naming
    out.sort(key=lambda z: z.price)
    merged: list[Level] = []
    for z in out:
        if merged and abs(z.price - merged[-1].price) <= 0.15 * a:
            continue
        merged.append(z)
    merged.sort(key=lambda z: abs(z.dist_atr))
    return merged
