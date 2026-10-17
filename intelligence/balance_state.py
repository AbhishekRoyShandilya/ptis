"""
intelligence.balance_state — the MISSING STATE fact: is the market in BALANCE or has it
TRANSITIONED?

The cognition test showed the reasoner flip-flops on the borderline (a small push inside the
day's range) because nothing told it "price is still INSIDE the balance, not accepted outside."
This supplies that fact: the prevailing balance BAND, how well-formed it is, and where price sits
relative to it (inside / poking / accepted-outside). The trader's unit isn't "did price move" —
it's "what STATE are we in, and has it transitioned?"

Principle held all session: CODE measures the candidate band + its facts (boundaries, touches,
containment, acceptance); the REASONER decides if it's THE relevant balance and whether a
transition is real. Candidate from code, significance from cognition. Causal / point-in-time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BALANCE_STATE_VERSION = "1.0"


@dataclass
class BalanceState:
    band_lo: float = np.nan
    band_hi: float = np.nan
    width_atr: float = np.nan
    touches_lo: int = 0
    touches_hi: int = 0
    containment: float = np.nan      # fraction of window bars whose CLOSE is inside the band
    duration_bars: int = 0
    well_formed: bool = False        # a real two-sided balance (vs trending / no structure)
    price: float = np.nan
    pos: str = ""                    # inside | poking_above | poking_below | accepted_above | accepted_below
    dist_atr: float = 0.0            # signed ATRs outside the band (0 if inside)
    bars_outside: int = 0            # consecutive recent closes beyond the edge (acceptance evidence)
    expansion: float = np.nan        # recent range vs the balance-period typical (transition needs it)
    state: str = "UNFORMED"          # BALANCE | TRANSITION_UP | TRANSITION_DOWN | TREND_UP | TREND_DOWN | UNFORMED
    note: str = ""


def read_balance(symbol: str, interval: str, as_of, candles: pd.DataFrame | None = None,
                 lookback: int = 48, feat: pd.DataFrame | None = None) -> BalanceState:
    """Identify the prevailing balance band over the last `lookback` bars and locate price
    relative to it. Point-in-time (only bars <= as_of). `feat` may be supplied (already computed on
    this SAME `candles`, row-aligned) to skip recompute — bit-identical (features are causal)."""
    from marketdata import features as F
    as_of = pd.Timestamp(as_of)
    if candles is None:
        from marketdata.candle_store import get_candle_store
        start = as_of.normalize() - pd.Timedelta(days=8)
        candles = get_candle_store().get(symbol, interval, start, as_of)
    c = candles[pd.to_datetime(candles["ts"]) <= as_of].sort_values("ts").reset_index(drop=True)
    R = BalanceState()
    if c is None or len(c) < 30:
        R.note = "insufficient data"; return R
    if feat is None or len(feat) != len(c):
        feat = F.compute_features(c, symbol, interval)   # window-warmed (validated); not the shared store (08)
    atr = feat["atr"].to_numpy(float)
    h = c["high"].to_numpy(float); l = c["low"].to_numpy(float); cl = c["close"].to_numpy(float)
    i = len(c) - 1
    a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else np.nan
    if not np.isfinite(a):
        R.note = "no atr"; return R
    w = slice(max(0, i - lookback + 1), i + 1)
    H, L, C = h[w], l[w], cl[w]
    n = len(C)
    R.price = round(float(cl[i]), 2)
    R.duration_bars = n

    # candidate band = the bulk rotation range (trim wick spikes via percentiles of closes/extremes)
    band_hi = float(np.percentile(H, 85)); band_lo = float(np.percentile(L, 15))
    if band_hi <= band_lo:
        R.note = "degenerate band"; return R
    R.band_hi = round(band_hi, 2); R.band_lo = round(band_lo, 2)
    R.width_atr = round((band_hi - band_lo) / a, 2)

    # how well-formed: two-sided touches + containment + small net drift (balance, not trend)
    R.touches_hi = int(np.sum(H >= band_hi - 0.2 * a))
    R.touches_lo = int(np.sum(L <= band_lo + 0.2 * a))
    R.containment = round(float(np.mean((C >= band_lo) & (C <= band_hi))), 2)
    # "moving NOW" = directional EFFICIENCY over a SHORT recent window (does price follow a line right
    # now, or oscillate?) — kept separate from the band, so a trend doesn't get hidden by the wider
    # window wrapping it, and yesterday's balance doesn't dilute today's move.
    rec_n = min(15, n)
    Cr = C[-rec_n:]
    net_r = float(Cr[-1] - Cr[0]); net_r_atr = abs(net_r) / a
    path_r = float(np.sum(np.abs(np.diff(Cr)))) or 1e-9
    eff_r = abs(net_r) / path_r                        # ~1 = clean directional push, ~0 = chop
    moving_now = bool(eff_r >= 0.40 and net_r_atr >= 1.5)
    # band QUALITY only (decoupled from motion) — is this a real two-sided cage?
    R.well_formed = bool(R.touches_hi >= 2 and R.touches_lo >= 2 and R.containment >= 0.6
                         and 0.8 <= R.width_atr <= 10)

    # price position relative to the band + acceptance evidence
    px = cl[i]
    if px > band_hi:
        R.dist_atr = round((px - band_hi) / a, 2)
        k = 0
        for j in range(i, max(0, i - 12) - 1, -1):
            if cl[j] > band_hi: k += 1
            else: break
        R.bars_outside = k
        R.pos = "accepted_above" if (k >= 3 and R.dist_atr >= 0.3) else "poking_above"
    elif px < band_lo:
        R.dist_atr = round((px - band_lo) / a, 2)
        k = 0
        for j in range(i, max(0, i - 12) - 1, -1):
            if cl[j] < band_lo: k += 1
            else: break
        R.bars_outside = k
        R.pos = "accepted_below" if (k >= 3 and abs(R.dist_atr) >= 0.3) else "poking_below"
    else:
        R.pos = "inside"; R.dist_atr = 0.0; R.bars_outside = 0

    # expansion: recent bar ranges vs the balance-period typical (a transition shows expansion)
    rng = H - L
    typ = float(np.median(rng[:-3])) if n > 6 else float(np.median(rng))
    rec = float(np.mean(rng[-3:]))
    R.expansion = round(rec / typ, 2) if typ > 0 else np.nan

    # STATE synthesis — combine WHERE price is (band position) with whether it's MOVING now.
    # A directional push while price is MID-CAGE = rotation (balance); the same push AT/THROUGH the
    # edge = a move. This is what separates the 1.6-ATR pop inside the cage from a real breakout.
    at_edge = bool(px >= band_hi - 0.25 * a or px <= band_lo + 0.25 * a)
    if R.pos.startswith("accepted"):
        direction = "UP" if "above" in R.pos else "DOWN"
        recent = R.bars_outside <= 10
        expanding = np.isfinite(R.expansion) and R.expansion >= 1.2
        R.state = f"TRANSITION_{direction}" if (recent and (expanding or abs(R.dist_atr) <= 4)) else f"TREND_{direction}"
    elif R.pos.startswith("poking"):
        # at/over the edge but not yet accepted: a transition only if actually moving now
        direction = "UP" if "above" in R.pos else "DOWN"
        R.state = f"TRANSITION_{direction}" if moving_now else "BALANCE"
    elif moving_now and (at_edge or not R.well_formed):
        R.state = "TREND_UP" if net_r > 0 else "TREND_DOWN"
    elif R.well_formed:
        R.state = "BALANCE"            # price inside a real cage — pops within it are still balance
    else:
        R.state = "UNFORMED"

    R.note = _note(R)
    return R


def _note(R: BalanceState) -> str:
    if R.state == "BALANCE":
        return (f"rotating inside {R.band_lo}-{R.band_hi} ({R.width_atr} ATR wide, {R.duration_bars} bars, "
                f"{R.touches_lo}/{R.touches_hi} touches lo/hi, {R.containment:.0%} contained); "
                f"price {R.price} {('INSIDE' if R.pos=='inside' else R.pos)} — NOT accepted outside → no transition")
    if R.state.startswith("TRANSITION"):
        d = "above" if R.state.endswith("UP") else "below"
        return (f"ACCEPTED {d} the {R.band_lo}-{R.band_hi} balance: {R.bars_outside} bars held outside, "
                f"{abs(R.dist_atr)} ATR out, range x{R.expansion} → left the cage (state transition)")
    if R.state.startswith("TREND"):
        return (f"trending {R.state.split('_')[1].lower()} — already out of / no balance "
                f"(band {R.band_lo}-{R.band_hi}, {R.containment:.0%} contained)")
    if R.pos.startswith("poking"):
        return (f"{R.pos.replace('_',' ')} the {R.band_lo}-{R.band_hi} band by {abs(R.dist_atr)} ATR, "
                f"only {R.bars_outside} bars — not yet accepted (still effectively balance)")
    return f"no well-formed balance (band {R.band_lo}-{R.band_hi}, {R.containment:.0%} contained, width {R.width_atr} ATR)"


def narrate_balance(R: BalanceState) -> str:
    return f"STATE: {R.state} — {R.note}"


if __name__ == "__main__":
    POINTS = [
        ("HAL", "2025-07-09 15:20", "user: chop/no move"),
        ("HAL", "2024-02-20 15:15", "user: range/no move"),
        ("HAL", "2023-07-27 13:25", "user: inside cage (detector over-proposed a move)"),
        ("HAL", "2023-07-28 09:45", "user: real move begun"),
        ("HAL", "2023-07-28 10:30", "user: real move, now extended"),
    ]
    for sym, t, ur in POINTS:
        R = read_balance(sym, "5m", pd.Timestamp(t))
        print(f"\n{sym} @ {t}  [{ur}]")
        print("   " + narrate_balance(R))
