"""
intelligence.momentum_run — the LIVE momentum-move read (the user's real ask).

Not swing-level post-hoc structure. This tracks a momentum move WHILE IT RUNS and, the
moment a small red candle appears mid-move, answers the trader's gut question:
    "is it turning, or just a breather I shouldn't fear?"

An up-run ignites on a strong green push / breakout and is 'active' while price keeps
making new highs above the EMA with green dominance. Each RED candle inside an active
run is a pullback-bar, classified point-in-time as BENIGN (hold) vs REVERSAL_ONSET
(the move is ending), then validated by whether the run makes a NEW HIGH afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_RUN_VERSION = "1.0"
IGNITION_BODY_ATR = 0.8      # a green bar with body ≥0.8 ATR (or a fresh 10-bar-high green) ignites
HIGH_LOOKBACK = 10
RESUME_BARS = 8             # bars after the red to judge "did the run make a new high"
GIVEBACK_ATR = 1.0         # ...before giving back this much from the red's low → not resumed


def detect_up_runs(candles: pd.DataFrame, symbol: str, interval: str, feat: pd.DataFrame | None = None):
    """Return arrays: in_run (bool), run_high (float), bars_in_run (int), and the per-bar
    features needed to classify intra-run red candles. Point-in-time. `feat` may be supplied by a
    caller that already computed it on this SAME (already-sorted) `candles` — avoids recomputing the
    full feature set (the per-bar narration hotspot); bit-identical to computing it here."""
    from marketdata import features as F

    if feat is None:
        c = candles.sort_values("ts").reset_index(drop=True)
        feat = F.compute_features(c, symbol, interval)   # window-warmed (validated); not the shared store (08)
    else:
        c = candles
    o = c["open"].to_numpy(float); h = c["high"].to_numpy(float)
    l = c["low"].to_numpy(float); cl = c["close"].to_numpy(float)
    atr = feat["atr"].to_numpy(float); ema = feat["ema20"].to_numpy(float) if "ema20" in feat else feat["ema50"].to_numpy(float)
    n = len(c)
    green = cl > o
    body = np.abs(cl - o)
    prior_high = pd.Series(h).shift(1).rolling(HIGH_LOOKBACK).max().to_numpy()
    # session boundary — a momentum run is intraday and CANNOT carry across the overnight
    # / weekend gap (the gap is a discontinuity, not a continuation).
    day = pd.to_datetime(c["ts"]).dt.date.to_numpy()
    new_session = np.zeros(n, bool)
    new_session[1:] = day[1:] != day[:-1]

    in_run = np.zeros(n, bool); run_high = np.full(n, np.nan); bars_in = np.zeros(n, int)
    active = False; rh = np.nan; start = 0; last_new_high = 0
    for i in range(n):
        if new_session[i]:           # new day → any prior run is over
            active = False
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            in_run[i] = active; run_high[i] = rh; continue
        ignite = green[i] and (body[i] >= IGNITION_BODY_ATR * a or
                               (np.isfinite(prior_high[i]) and h[i] > prior_high[i]))
        if not active:
            if ignite:
                active = True; rh = h[i]; start = i; last_new_high = i
        else:
            if h[i] > rh:
                rh = h[i]; last_new_high = i
            # run dies: closes below EMA, or no new high for a while AND structure lost
            if cl[i] < ema[i] or (i - last_new_high) > 6:
                active = False
        in_run[i] = active; run_high[i] = rh; bars_in[i] = (i - start) if active else 0
    return c, feat, dict(in_run=in_run, run_high=run_high, bars_in=bars_in,
                         green=green, body=body, atr=atr, ema=ema, o=o, h=h, l=l, cl=cl)


def detect_down_runs(candles: pd.DataFrame, symbol: str, interval: str, feat: pd.DataFrame | None = None):
    """Mirror of detect_up_runs for BEARISH moves: a down-run ignites on a strong RED push /
    breakdown and is 'active' while price keeps making new LOWS below the EMA with red dominance.
    Returns run_low (analog of run_high). Point-in-time, intraday (no carry across the gap). `feat`
    may be supplied (already computed on this SAME sorted `candles`) to skip recompute — bit-identical."""
    from marketdata import features as F

    if feat is None:
        c = candles.sort_values("ts").reset_index(drop=True)
        feat = F.compute_features(c, symbol, interval)   # window-warmed (validated); not the shared store (08)
    else:
        c = candles
    o = c["open"].to_numpy(float); h = c["high"].to_numpy(float)
    l = c["low"].to_numpy(float); cl = c["close"].to_numpy(float)
    atr = feat["atr"].to_numpy(float); ema = feat["ema20"].to_numpy(float) if "ema20" in feat else feat["ema50"].to_numpy(float)
    n = len(c)
    red = cl < o
    body = np.abs(cl - o)
    prior_low = pd.Series(l).shift(1).rolling(HIGH_LOOKBACK).min().to_numpy()
    day = pd.to_datetime(c["ts"]).dt.date.to_numpy()
    new_session = np.zeros(n, bool)
    new_session[1:] = day[1:] != day[:-1]

    in_run = np.zeros(n, bool); run_low = np.full(n, np.nan); bars_in = np.zeros(n, int)
    active = False; rl = np.nan; start = 0; last_new_low = 0
    for i in range(n):
        if new_session[i]:
            active = False
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            in_run[i] = active; run_low[i] = rl; continue
        ignite = red[i] and (body[i] >= IGNITION_BODY_ATR * a or
                             (np.isfinite(prior_low[i]) and l[i] < prior_low[i]))
        if not active:
            if ignite:
                active = True; rl = l[i]; start = i; last_new_low = i
        else:
            if l[i] < rl:
                rl = l[i]; last_new_low = i
            if cl[i] > ema[i] or (i - last_new_low) > 6:   # closes back above EMA, or no new low
                active = False
        in_run[i] = active; run_low[i] = rl; bars_in[i] = (i - start) if active else 0
    return c, feat, dict(in_run=in_run, run_low=run_low, bars_in=bars_in,
                         red=red, body=body, atr=atr, ema=ema, o=o, h=h, l=l, cl=cl)


def classify_red_candles(candles: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """Each RED candle inside an active up-run → BENIGN vs REVERSAL_ONSET + forward outcome."""
    c, feat, S = detect_up_runs(candles, symbol, interval)
    ts = pd.to_datetime(c["ts"])
    n = len(c)
    o, h, l, cl, atr, ema = S["o"], S["h"], S["l"], S["cl"], S["atr"], S["ema"]
    in_run, run_high, bars_in, green = S["in_run"], S["run_high"], S["bars_in"], S["green"]
    rows = []
    for i in range(3, n - 1):
        if not in_run[i] or green[i]:           # only red candles inside an active run
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        pb_depth = (run_high[i] - l[i]) / a               # how far below the run high
        red_body = (o[i] - cl[i]) / a
        broke_prev_low = l[i] < l[i - 1]
        broke_3low = l[i] < np.min(l[i - 3:i])
        closed_below_ema = cl[i] < ema[i]
        ext_atr = (run_high[i] - ema[i]) / a              # how stretched the run is (exhaustion)
        # ── FROZEN classification (the 'should I fear it?' rule) ──
        if (broke_3low or closed_below_ema or pb_depth >= 1.5 or red_body >= 1.0):
            label = "REVERSAL_ONSET"
        elif (not broke_prev_low and pb_depth <= 0.8 and red_body <= 0.6):
            label = "BENIGN"
        else:
            label = "WATCH"
        # forward: does the run make a NEW HIGH (> run_high) before giving back GIVEBACK ATR?
        end = min(i + RESUME_BARS, n - 1)
        fh, fl = h[i + 1:end + 1], l[i + 1:end + 1]
        new_high = np.where(fh > run_high[i])[0]
        give_back = np.where(fl <= l[i] - GIVEBACK_ATR * a)[0]
        th = new_high[0] if len(new_high) else np.inf
        tg = give_back[0] if len(give_back) else np.inf
        outcome = "resumed" if th < tg else ("failed" if tg < th else "neither")
        rows.append({"ts": ts.iloc[i], "symbol": symbol, "bar": i, "label": label,
                     "pb_depth_atr": round(float(pb_depth), 2), "red_body_atr": round(float(red_body), 2),
                     "broke_3low": bool(broke_3low), "below_ema": bool(closed_below_ema),
                     "ext_atr": round(float(ext_atr), 2), "bars_in_run": int(bars_in[i]),
                     "outcome": outcome})
    return pd.DataFrame(rows)


def validate(labelled: pd.DataFrame) -> dict:
    out = {}
    for lab in ("BENIGN", "WATCH", "REVERSAL_ONSET"):
        d = labelled[labelled["label"] == lab]
        if len(d) < 20:
            out[lab] = {"n": len(d), "note": "thin"}; continue
        out[lab] = {"n": len(d), "resumed": round((d["outcome"] == "resumed").mean(), 3),
                    "failed": round((d["outcome"] == "failed").mean(), 3)}
    return out
