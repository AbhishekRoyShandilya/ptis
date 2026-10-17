"""
intelligence.lifecycle — the Lifecycle Observer (Stage 1).

For every bar, answer "what STAGE of the move am I looking at?" — pure observation. These are
the trader-level abstractions situation discovery clusters on (NOT raw ATR/RVOL/ADX).

CRITICAL: the energy/fuel read must MATCH move_read (the discovery that lifecycle separates a
winner from a trap depends on it). So we use the SAME run spine move_read uses
(momentum_run.detect_up_runs) and measure fuel the SAME way (early-vs-recent rvol on the run),
not a generic 5/20 ratio. Verified by eye against move_read.

Axes per bar (point-in-time):
  energy    : BUILDING · RELEASE · EXPANDING · MATURE · EXHAUSTING · DORMANT
  structure : HH_HL · LH_LL · COMPRESSION · EXPANSION · RANGE · BREAKDOWN
  fuel      : EXPANDING · STABLE · DRYING
  context   : htf (bull/bear/flat) + location (NEAR_RES · CLEAR_AIR · NEAR_SUP · MID)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LIFECYCLE_VERSION = "1.1"

ENERGY = ["DORMANT", "BUILDING", "RELEASE", "EXPANDING", "MATURE", "EXHAUSTING"]
STRUCTURE = ["RANGE", "COMPRESSION", "EXPANSION", "HH_HL", "LH_LL", "BREAKDOWN"]
FUEL = ["DRYING", "STABLE", "EXPANDING"]
LOC = ["NEAR_SUP", "MID", "NEAR_RES", "CLEAR_AIR"]


def lifecycle_frame(candles: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """Per-bar lifecycle card (ts + energy/structure/fuel/htf/loc + the numerics behind them)."""
    from marketdata import features as F
    from intelligence.move_nature import atr_zigzag
    from intelligence.momentum_run import detect_up_runs

    c = candles.sort_values("ts").reset_index(drop=True)
    if len(c) < 80:
        return pd.DataFrame()
    feat = F.compute_features(c, symbol, interval)
    ts = pd.to_datetime(c["ts"])
    h = c["high"].to_numpy(float); l = c["low"].to_numpy(float); cl = c["close"].to_numpy(float)
    o = c["open"].to_numpy(float)
    atr = feat["atr"].to_numpy(float); rsi = feat["rsi"].to_numpy(float)
    adx = feat["adx"].to_numpy(float) if "adx" in feat else np.full(len(c), np.nan)
    ema20 = feat["ema20"].to_numpy(float); ema50 = feat["ema50"].to_numpy(float)
    rvol = feat["rvol"].to_numpy(float)
    bb_pctile = feat["bb_width_pct"].rolling(100, min_periods=30).rank(pct=True).to_numpy()
    atr_pct = feat["atr_pct"].to_numpy(float)
    atr_mean = pd.Series(atr_pct).rolling(20, min_periods=10).mean().to_numpy()
    atr_trend = atr_pct / np.where(atr_mean > 0, atr_mean, np.nan)
    n = len(c)
    green = cl > o

    # ── the SAME up-run spine move_read uses (so energy/fuel match) ──
    _, _, S = detect_up_runs(c, symbol, interval)
    in_run, run_high, bars_in = S["in_run"], S["run_high"], S["bars_in"]

    # ── swing legs (for structure + down/neutral energy) ──
    piv = atr_zigzag(h, l, atr)
    confs = np.array([p.conf_idx for p in piv]) if piv else np.array([], int)

    energy = np.array(["DORMANT"] * n, dtype=object)
    fuel = np.array(["STABLE"] * n, dtype=object)
    struct = np.array(["RANGE"] * n, dtype=object)
    leg_dir = np.zeros(n, int); leg_ext = np.zeros(n); leg_bars = np.zeros(n, int)
    # computed ONCE (was rebuilt from the full series on every down-leg bar -> O(n^2) on long history)
    rvol_ma5 = pd.Series(rvol).rolling(5, min_periods=2).mean().to_numpy()
    # last-two H / L pivot prices for every prefix piv[:k] — built ONCE (was an O(k) list scan on every
    # bar -> O(bars x pivots), ~2 billion ops per stock on full 5m history). Same values, same comparisons.
    npv = len(piv)
    hN = np.zeros(npv + 1, int); lN = np.zeros(npv + 1, int)
    hA = np.full(npv + 1, np.nan); hB = np.full(npv + 1, np.nan)      # hA = latest H, hB = the one before
    lA = np.full(npv + 1, np.nan); lB = np.full(npv + 1, np.nan)
    for j, p in enumerate(piv):
        hN[j + 1], hA[j + 1], hB[j + 1] = hN[j], hA[j], hB[j]
        lN[j + 1], lA[j + 1], lB[j + 1] = lN[j], lA[j], lB[j]
        if p.kind == "H":
            hN[j + 1] += 1; hB[j + 1] = hA[j]; hA[j + 1] = p.price
        elif p.kind == "L":
            lN[j + 1] += 1; lB[j + 1] = lA[j]; lA[j + 1] = p.price

    for i in range(n):
        a = atr[i]
        if not (np.isfinite(a) and a > 0):
            continue
        # structure from confirmed pivots
        k = int(np.searchsorted(confs, i, side="right")) if len(confs) else 0
        if k:
            last = piv[k - 1]
            leg_dir[i] = 1 if last.kind == "L" else -1
            leg_ext[i] = (cl[i] - last.price) / a * leg_dir[i]
            leg_bars[i] = i - last.idx
            if hN[k] >= 2 and lN[k] >= 2:
                if hA[k] > hB[k] and lA[k] > lB[k]:
                    struct[i] = "HH_HL"
                elif hA[k] < hB[k] and lA[k] < lB[k]:
                    struct[i] = "LH_LL"
        if np.isfinite(bb_pctile[i]) and bb_pctile[i] <= 0.20:
            struct[i] = "COMPRESSION"
        elif np.isfinite(atr_trend[i]) and atr_trend[i] >= 1.6 and struct[i] not in ("HH_HL", "LH_LL"):
            struct[i] = "EXPANSION"
        if struct[i] == "LH_LL" and np.isfinite(ema50[i]) and cl[i] < ema50[i]:
            struct[i] = "BREAKDOWN"

        # ── energy + fuel from the up-run (faithful to move_read) ──
        if in_run[i]:
            start = i - int(bars_in[i])
            ext = (run_high[i] - l[start]) / a
            stretch = (cl[i] - ema20[i]) / a if np.isfinite(ema20[i]) else 0.0
            r = rsi[i] if np.isfinite(rsi[i]) else 50.0
            # fuel = early-vs-recent rvol on the run's GREEN pushes (move_read's definition)
            push = [j for j in range(start, i + 1) if green[j] and np.isfinite(rvol[j])]
            ratio = 1.0
            if len(push) >= 2:
                half = max(1, len(push) // 2)
                early = np.nanmean([rvol[j] for j in push[:half]])
                recent = np.nanmean([rvol[j] for j in push[half:]])
                ratio = recent / early if early > 0 else 1.0
            fuel[i] = "EXPANDING" if ratio >= 1.15 else ("DRYING" if ratio < 0.85 else "STABLE")
            has_coil = np.any(bb_pctile[max(0, start - 10):start] <= 0.25)
            late = ext >= 5 or stretch >= 3 or r >= 75 or (ratio < 0.85 and ext >= 3)
            early_ph = ext < 2 and bars_in[i] <= 4
            if late:
                energy[i] = "EXHAUSTING" if (ratio < 0.85 or r >= 72) else "MATURE"
            elif early_ph and np.isfinite(atr_trend[i]) and atr_trend[i] >= 1.6:
                energy[i] = "RELEASE"
            elif has_coil and ratio >= 1.0 and stretch < 3:
                energy[i] = "RELEASE"
            elif early_ph:
                energy[i] = "BUILDING"
            else:
                energy[i] = "EXPANDING"
        elif leg_dir[i] < 0 and leg_ext[i] >= 1.0:           # down leg (cruder — no validated down spine)
            r = rsi[i] if np.isfinite(rsi[i]) else 50.0
            energy[i] = "EXHAUSTING" if (leg_ext[i] >= 4.5 and r <= 28) else (
                "MATURE" if leg_ext[i] >= 4.5 else "EXPANDING")
            # fuel via simple vol trend for down legs
            vt = rvol_ma5[i]
            fuel[i] = "EXPANDING" if (np.isfinite(vt) and vt >= 1.3) else (
                "DRYING" if (np.isfinite(vt) and vt < 0.8) else "STABLE")
        else:
            energy[i] = "DORMANT"

    # ── context: htf + cheap location vs 50-bar structural hi/lo ──
    try:
        from intelligence.story_context import context_frame
        ctxf = context_frame(c, symbol, interval)
        htf = np.where(ctxf["HTF_UP"].to_numpy() == 1, "bull",
                       np.where(ctxf["HTF_DOWN"].to_numpy() == 1, "bear", "flat")) \
            if (ctxf is not None and not ctxf.empty) else np.array(["flat"] * n)
    except Exception:
        htf = np.array(["flat"] * n)
    hi50 = pd.Series(h).shift(1).rolling(50, min_periods=20).max().to_numpy()
    lo50 = pd.Series(l).shift(1).rolling(50, min_periods=20).min().to_numpy()
    d_res = (hi50 - cl) / np.where(atr > 0, atr, np.nan)
    d_sup = (cl - lo50) / np.where(atr > 0, atr, np.nan)
    loc = np.array(["MID"] * n, dtype=object)
    loc[(d_res >= 0) & (d_res <= 1.0)] = "NEAR_RES"
    loc[(d_sup >= 0) & (d_sup <= 1.0)] = "NEAR_SUP"
    loc[(d_res > 3.0) & (d_sup > 3.0)] = "CLEAR_AIR"

    return pd.DataFrame({
        "ts": ts, "energy": energy, "structure": struct, "fuel": fuel, "htf": htf, "loc": loc,
        "leg_dir": leg_dir, "leg_ext_atr": np.round(leg_ext, 2), "leg_bars": leg_bars,
        "in_run": in_run, "rsi": np.round(rsi, 1),
        "d_res_atr": np.round(d_res, 2), "d_sup_atr": np.round(d_sup, 2),
    })


def lifecycle_card(symbol: str, interval: str, as_of) -> dict:
    """The single-moment lifecycle card (live / verification). Point-in-time."""
    from marketdata.candle_store import get_candle_store
    as_of = pd.Timestamp(as_of)
    c = get_candle_store().get(symbol, interval, as_of.normalize() - pd.Timedelta(days=12), as_of)
    if c is None or len(c) < 80:
        return {}
    c = c[pd.to_datetime(c["ts"]) <= as_of]
    lf = lifecycle_frame(c, symbol, interval)
    if lf.empty:
        return {}
    r = lf.iloc[-1]
    return {"symbol": symbol, "ts": r["ts"], "energy": r["energy"], "structure": r["structure"],
            "fuel": r["fuel"], "htf": r["htf"], "loc": r["loc"],
            "leg_ext_atr": r["leg_ext_atr"], "leg_bars": int(r["leg_bars"])}
