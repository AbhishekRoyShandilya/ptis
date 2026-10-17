"""
intelligence.level_dynamics — the DYNAMIC-level organ (Stage A of the archetype recognizer).

The eye's existing level machinery (`eye_levels._gather_levels`) is STATIC: it lists levels and their
nearest-distance. The user's teaching (batch-3 topup, 2026-07-23) requires levels understood DYNAMICALLY:
  · a tested level is NOT automatically a wall — repeated tests mean STRENGTHENING (flat/failing) OR
    ABSORPTION (rising+compressing → breakout more likely); the system must tell them apart.
  · APPROACH quality — price arriving FAST at a level is fragile; a slow approach that spends time +
    accumulates + shows weakness then breaks with a big body is the real thing.
  · RECENCY is TIMEFRAME-RELATIVE (in SESSIONS, never absolute calendar; the scale-mistake that bit
    room_atr twice): recent/older buckets scale per TF. An old level re-enters via the DAILY read.

This organ is PER-TIMEFRAME: call it on 5m, 1h, 1d candles separately, each computes levels on ITS OWN
recency scale. `archetype_facts` stacks them into the cumulative TF-weighted room. Strictly causal /
point-in-time (only bars <= the last row are used; pivots are atr_zigzag-confirmed, no lookahead).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LEVEL_DYNAMICS_VERSION = "1.0"

# Recency buckets in SESSIONS (trading days), per timeframe — TIMEFRAME-RELATIVE, not absolute.
# 5m intraday: recent = today/yesterday/day-before (~3-4 sessions), older ~2 weeks. Daily: weeks-months.
RECENCY_SESSIONS = {
    "5m":  {"recent": 4,  "older": 10},
    "15m": {"recent": 6,  "older": 15},
    "1h":  {"recent": 10, "older": 30},
    "1d":  {"recent": 30, "older": 120},
}
# TF hurdle weight (a daily swing is a bigger obstacle to an intraday move than a 5m micro-pivot).
TF_WEIGHT = {"5m": 1.0, "15m": 1.5, "1h": 2.5, "1d": 5.0}


@dataclass
class DynLevel:
    price: float
    kind: str                 # "resistance" (from swing highs) | "support" (from swing lows)
    dist_atr: float           # signed: +above current price, -below
    tf: str                   # the timeframe this level is significant on
    n_rejections: int         # swing pivots clustered at this price within the recency window
    age_sessions: float       # sessions since the MOST RECENT test/pivot at this level
    recency: str              # "recent" | "older" | "thin" (on THIS tf's scale)
    weight: float             # TF_WEIGHT[tf] * recency factor — how much of a hurdle it is


@dataclass
class LevelContext:
    tf: str
    price: float
    atr: float
    levels: list[DynLevel] = field(default_factory=list)          # all relevant levels, sorted by |dist|
    # the level price is currently nearest/interacting with (within 1 ATR), if any:
    active_price: float = float("nan")
    active_kind: str = ""                 # resistance | support | ""
    active_dist_atr: float = float("nan")
    active_n_rejections: int = 0
    # room beyond, per direction (nearest-first), for the cumulative TF-weighted stack in archetype_facts:
    room_up: list[DynLevel] = field(default_factory=list)
    room_down: list[DynLevel] = field(default_factory=list)
    # NOTE: absorption-vs-strengthening and approach-speed are computed in archetype_facts OVER THE COIL
    # WINDOW (from balance_state), not here — a fixed-lookback pivot read was unreliable (coarse
    # atr_zigzag misses the fine higher-lows/lower-highs the user reads; see git history 2026-07-23).


def _session_ordinals(ts: pd.Series) -> np.ndarray:
    """Map each bar to a session ordinal (0-based over distinct trading dates), so 'age' is measured in
    SESSIONS not bars/calendar — the correct, timeframe-consistent unit."""
    dates = pd.to_datetime(ts).dt.normalize()
    uniq = {d: k for k, d in enumerate(sorted(dates.unique()))}
    return dates.map(uniq).to_numpy()


def _cluster_levels(prices: list[float], conf_sess: list[float], kind: str, atr: float,
                    last_sess: float, tol_atr: float = 0.4) -> list[tuple]:
    """Cluster pivot prices within tol_atr*ATR into levels. Returns (price, n, age_sessions) per level —
    n = pivots at that price (rejection count), age = sessions since the most recent pivot there."""
    if not prices:
        return []
    order = np.argsort(prices)
    ps = np.array(prices)[order]; ss = np.array(conf_sess)[order]
    out = []
    cur_p = [ps[0]]; cur_s = [ss[0]]
    for k in range(1, len(ps)):
        if ps[k] - cur_p[-1] <= tol_atr * atr:
            cur_p.append(ps[k]); cur_s.append(ss[k])
        else:
            out.append((float(np.mean(cur_p)), len(cur_p), float(last_sess - max(cur_s))))
            cur_p = [ps[k]]; cur_s = [ss[k]]
    out.append((float(np.mean(cur_p)), len(cur_p), float(last_sess - max(cur_s))))
    return out


def _recency_bucket(age_sessions: float, interval: str) -> str:
    r = RECENCY_SESSIONS.get(interval, RECENCY_SESSIONS["5m"])
    if age_sessions <= r["recent"]:
        return "recent"
    if age_sessions <= r["older"]:
        return "older"
    return "thin"


def _touch_episodes(h, l, cl, level: float, kind: str, atr: float, win_start: int, i: int,
                    tol: float = 0.35) -> int:
    """Count DISTINCT touch-and-reject episodes at a level (a bar reaches the level then closes back on
    the origin side, having been AWAY the prior bar). This fixes the atr_zigzag pivot undercount — a
    tight coil touches its floor many times without ever confirming a zigzag pivot (20-Dec-24: 1 pivot
    but ~13 touches). Point-in-time."""
    n = 0; prev_touch = False
    for k in range(max(0, win_start), i + 1):
        if kind == "resistance":
            touch = h[k] >= level - tol * atr
            reject = cl[k] < level - 0.05 * atr
        else:
            touch = l[k] <= level + tol * atr
            reject = cl[k] > level + 0.05 * atr
        if touch and reject and not prev_touch:
            n += 1
        prev_touch = touch
    return n


def level_context(candles: pd.DataFrame, symbol: str, interval: str,
                  feat: pd.DataFrame | None = None) -> LevelContext:
    """The dynamic-level read for `candles` at `interval`, as of the LAST bar. Point-in-time."""
    from marketdata import features as F
    c = candles.sort_values("ts").reset_index(drop=True)
    if len(c) < 40:
        return LevelContext(tf=interval, price=float("nan"), atr=float("nan"))
    if feat is None:
        feat = F.compute_features(c, symbol, interval)
    i = len(c) - 1
    atr = float(feat["atr"].to_numpy(float)[i])
    price = float(c["close"].to_numpy(float)[i])
    if not (np.isfinite(atr) and atr > 0):
        return LevelContext(tf=interval, price=price, atr=float("nan"))

    sess = _session_ordinals(c["ts"])
    last_sess = float(sess[i])
    rec = RECENCY_SESSIONS.get(interval, RECENCY_SESSIONS["5m"])
    # only pivots within the 'older' recency window count as relevant levels (TF-scaled)
    win_start = np.searchsorted(sess, last_sess - rec["older"] + 1, side="left")

    h = c["high"].to_numpy(float); l = c["low"].to_numpy(float); cl = c["close"].to_numpy(float)
    from intelligence.move_nature import atr_zigzag
    atr_arr = feat["atr"].to_numpy(float)
    piv = atr_zigzag(h, l, atr_arr)
    piv = [p for p in piv if p.conf_idx >= win_start and p.conf_idx <= i]

    hi_prices = [p.price for p in piv if p.kind == "H"]; hi_sess = [float(sess[p.idx]) for p in piv if p.kind == "H"]
    lo_prices = [p.price for p in piv if p.kind == "L"]; lo_sess = [float(sess[p.idx]) for p in piv if p.kind == "L"]

    def _mk(pr, n_piv, age, kind):
        # rejections = max(zigzag pivots, touch-and-reject episodes) — pivots undercount tight coils
        n = max(n_piv, _touch_episodes(h, l, cl, pr, kind, atr, win_start, i))
        recb = _recency_bucket(age, interval)
        recf = 1.0 if recb == "recent" else 0.6 if recb == "older" else 0.3
        w = TF_WEIGHT.get(interval, 1.0) * recf * min(3, n) / 3
        return DynLevel(round(pr, 2), kind, round((pr - price) / atr, 2), interval, n, round(age, 1), recb, round(w, 2))

    levels: list[DynLevel] = []
    for (pr, n_piv, age) in _cluster_levels(hi_prices, hi_sess, "resistance", atr, last_sess):
        levels.append(_mk(pr, n_piv, age, "resistance"))
    for (pr, n_piv, age) in _cluster_levels(lo_prices, lo_sess, "support", atr, last_sess):
        levels.append(_mk(pr, n_piv, age, "support"))
    levels.sort(key=lambda z: abs(z.dist_atr))

    ctx = LevelContext(tf=interval, price=round(price, 2), atr=round(atr, 2), levels=levels)
    near = [z for z in levels if abs(z.dist_atr) <= 1.0]
    if near:
        a = min(near, key=lambda z: abs(z.dist_atr))
        ctx.active_price = a.price; ctx.active_kind = a.kind
        ctx.active_dist_atr = a.dist_atr; ctx.active_n_rejections = a.n_rejections
    ctx.room_up = [z for z in levels if z.dist_atr > 0.2]
    ctx.room_down = [z for z in levels if z.dist_atr < -0.2]
    return ctx
