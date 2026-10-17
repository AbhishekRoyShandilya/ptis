"""
intelligence.move_nature — classify the NATURE of a move (the user's core ask).

A move is not just "up" or "down" — a trader reads WHAT KIND it is:
  • is this counter-move a PULLBACK (trend resumes) or a real REVERSAL (trend ends)?
  • is the move ADDING FUEL (expanding range/momentum) or STRUCTURE WEAKENING?

This module segments price into swing legs (ATR zig-zag, point-in-time confirmation),
identifies the prevailing trend, and at each completed swing pivot classifies the
counter-move as PULLBACK vs REVERSAL from FROZEN structural features, plus a
fuel-vs-weakening read on the trend leg. Nothing is asserted as truth until
`validate_pullback_reversal` shows the labels actually separate forward behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MOVE_NATURE_VERSION = "1.0"
ZIGZAG_ATR = 1.5            # a swing reverses when price retraces ≥1.5 ATR from the extreme
PULLBACK_MAX_RETRACE = 0.618    # Fibonacci: deeper than this leans reversal
RESUME_BARS = 16           # forward window to judge resume vs reverse


@dataclass
class Pivot:
    idx: int            # bar of the extreme
    conf_idx: int       # bar where it was CONFIRMED (point-in-time knowable)
    price: float
    kind: str           # "H" or "L"


def atr_zigzag(high, low, atr, mult: float = ZIGZAG_ATR):
    """Confirmed swing pivots. A high/low is confirmed once price retraces mult·ATR
    from it; conf_idx is the bar that retracement completes (no lookahead)."""
    n = len(high)
    pivots = []
    direction = 0          # +1 up-swing (seeking high), -1 down-swing (seeking low)
    ext_idx, ext_price = 0, high[0]
    for i in range(1, n):
        a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else np.nan
        if not np.isfinite(a):
            continue
        if direction >= 0:                      # in up-swing, tracking the high
            if high[i] >= ext_price:
                ext_idx, ext_price = i, high[i]
            elif ext_price - low[i] >= mult * a:    # retraced enough → confirm HIGH
                pivots.append(Pivot(ext_idx, i, ext_price, "H"))
                direction = -1; ext_idx, ext_price = i, low[i]
        else:                                   # in down-swing, tracking the low
            if low[i] <= ext_price:
                ext_idx, ext_price = i, low[i]
            elif high[i] - ext_price >= mult * a:    # retraced enough → confirm LOW
                pivots.append(Pivot(ext_idx, i, ext_price, "L"))
                direction = +1; ext_idx, ext_price = i, high[i]
    return pivots


def classify_pullback_reversal(candles: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """At each confirmed pivot that STARTS a counter-move, classify PULLBACK vs REVERSAL.

    Setup: in an uptrend (prior pivots HH/HL), a confirmed HIGH starts a down counter-move.
    We read it at the moment the NEXT (low) pivot confirms — purely from structure known
    then — and label the forward resolution for validation."""
    from marketdata import features as F

    c = candles.sort_values("ts").reset_index(drop=True)
    feat = F.compute_features(c, symbol, interval)
    high = c["high"].to_numpy(float); low = c["low"].to_numpy(float); close = c["close"].to_numpy(float)
    atr = feat["atr"].to_numpy(float); ema50 = feat["ema50"].to_numpy(float)
    ts = pd.to_datetime(c["ts"])
    piv = atr_zigzag(high, low, atr)
    rows = []
    for k in range(3, len(piv)):
        p_lo = piv[k]                       # the counter-move's end pivot (a LOW after an uptrend)
        p_hi = piv[k - 1]                   # the trend extreme (HIGH) the counter-move fell from
        p_prev_lo = piv[k - 2]              # prior swing low (structure to break)
        p_prev_hi = piv[k - 3]
        if not (p_hi.kind == "H" and p_lo.kind == "L"):
            continue
        # prevailing trend = uptrend if HH and HL into the high
        uptrend = p_hi.price > p_prev_hi.price and p_prev_lo.price > piv[k - 4].price if k >= 4 else p_hi.price > p_prev_hi.price
        if not uptrend:
            continue
        i = p_lo.conf_idx                   # point-in-time: we know this at confirmation
        if i >= len(c) - 2:
            continue
        trend_leg = p_hi.price - p_prev_lo.price
        counter_leg = p_hi.price - p_lo.price
        if trend_leg <= 0:
            continue
        retrace = counter_leg / trend_leg
        broke_prior_low = p_lo.price < p_prev_lo.price
        below_ema = close[i] < ema50[i]
        a = atr[i] if atr[i] > 0 else np.nan
        # fuel vs weakening on the TREND leg: range expanding into the high?
        seg = slice(max(0, p_prev_lo.idx), p_hi.idx + 1)
        rng = (high[seg] - low[seg])
        weakening = len(rng) >= 4 and np.nanmean(rng[-len(rng)//2:]) < np.nanmean(rng[:len(rng)//2])
        # ── FROZEN classification ──
        if retrace <= PULLBACK_MAX_RETRACE and not broke_prior_low and not below_ema:
            label = "PULLBACK"
        elif broke_prior_low or (retrace > PULLBACK_MAX_RETRACE and below_ema):
            label = "REVERSAL"
        else:
            label = "AMBIGUOUS"
        # forward resolution: did the uptrend RESUME (new high above p_hi) before making a
        # lower low (below p_lo) within RESUME_BARS?
        end = min(i + RESUME_BARS, len(c) - 1)
        fh, fl = high[i + 1:end + 1], low[i + 1:end + 1]
        resumed = np.inf; reversed_ = np.inf
        if len(fh):
            up_hit = np.where(fh >= p_hi.price)[0]
            dn_hit = np.where(fl <= p_lo.price - 0.1 * (a if np.isfinite(a) else 0))[0]
            resumed = up_hit[0] if len(up_hit) else np.inf
            reversed_ = dn_hit[0] if len(dn_hit) else np.inf
        outcome = ("resumed" if resumed < reversed_ else
                   "reversed" if reversed_ < resumed else "neither")
        rows.append({"ts": ts.iloc[i], "symbol": symbol, "pivot_idx": p_lo.idx,
                     "label": label, "retrace": round(retrace, 2),
                     "broke_prior_low": broke_prior_low, "below_ema": below_ema,
                     "weakening": bool(weakening), "outcome": outcome,
                     "high_price": p_hi.price, "low_price": p_lo.price})
    return pd.DataFrame(rows)


def validate_pullback_reversal(labelled: pd.DataFrame) -> dict:
    """Do the labels separate forward behaviour? PULLBACK should RESUME more than REVERSAL."""
    out = {}
    for lab in ("PULLBACK", "REVERSAL", "AMBIGUOUS"):
        d = labelled[labelled["label"] == lab]
        if len(d) < 20:
            out[lab] = {"n": len(d), "note": "thin"}; continue
        out[lab] = {"n": len(d),
                    "resumed": round((d["outcome"] == "resumed").mean(), 3),
                    "reversed": round((d["outcome"] == "reversed").mean(), 3)}
    return out
