"""
intelligence.archetype_v2 — REBUILT candidate layer (2026-07-24), after the user showed v1 was contaminated.

WHY. v1 fired on "strong body + close beyond a 20-bar ROLLING high/low" — a birth detector, the exact
candle-marker the user said to drop. Confirmed failures on HAL 2026: a ONE-BAR "coil" (07-Jul 09:45);
three firings inside one move (24-Jun 11:55/12:05/12:10, where 12:10 "broke" the low 12:05 had just made);
and a break straight into a wall the facts had already measured (13-Jul 14:30). Its close_decisiveness
also scored a half-range close as 1.0.

GATES (each is one of the user's rules; every gate reports pass/fail so a firing is fully explainable):
  G0 OPEN      — the 09:15 opening candle is never itself the break (it belongs to the coil; entry 09:20+).
  G1 REAL LEVEL— closes beyond a level with real prior rejections, from level_dynamics, never a rolling extreme.
  G2 FRESH     — MOVE MEMORY: fires only if no QUALIFYING break already fired since price left the coil.
                 A failed probe does NOT consume the signal (24-Jun 11:55 probe -> 12:05 real break; 08-Jul
                 13:50 blocked by support -> 13:55 fires). Still kills 12:10 and 09:45 as continuations.
  G3 REAL COIL — a genuine consolidation preceded it, never a 1-bar "coil".
  G4 WALL      — a weighty level close ahead, or simply no room. NOT daily-only (DIXON has ZERO 1d bars,
                 which made a daily-only rule a silent no-op and let breaks run into resistance).
  G5 CANDLE    — body >> coil average AND the close sits near the candle's extreme (not mid-range).
  G6 CONTEXT   — the HIGHER timeframe must not OPPOSE the break.
  G7 MATURITY  — not a mature higher-TF continuation.

OBJECTIVE (user, 2026-07-24): ASYMMETRIC — minimise HARD FAILURES; a missed move is cheap. Gates are
deliberately strict and will cost valid breaks.

The v1 champion/challenger (`archetype_facts.archetype_verdict*`) are untouched, for comparison.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ARCHETYPE_V2_VERSION = "2.1"

MIN_COIL_BARS = 10          # a real consolidation (>=50 min on 5m), never a 1-bar "coil"
MIN_REJECTIONS = 2          # the level must have been tested/respected before
MIN_BODY_VS_COIL = 2.0      # break body vs the AVERAGE of the coil candles
MIN_CLOSE_POS = 0.65        # close near the candle's extreme (the user: "not half like this")
MIN_ROOM_ATR = 0.0          # raw room floor DISABLED: it cost HAL 24-Jun 12:05 (room 0.44, a
                            # user FIRE case) while adding nothing — the DIXON rejects it was meant to
                            # catch are already blocked by G6_context. Walls are handled by _hurdle_blocks.
HURDLE_BLOCK_ATR = 0.8      # a weighty level this close blocks
HURDLE_HTF_WEIGHT = 2.5     # a higher-TF (1h/1d) level blocks from further out
HURDLE_HTF_ATR = 1.5
MAX_LOOKBACK_BARS = 10      # how far back to look for an already-fired break in this move
COIL_UNDER_LEVEL_BARS = 18  # a coil this long UNDER a tested level = absorption, not continuation (~90min)
# 09:15 handling (user, 2026-07-24). Blanket-blocking the opening candle was the single WORST gate
# (+19.5pp hard-fail on the test half): that cohort has 63% clean-runs and ~2x the reward-to-heat ratio
# (FAV/ADV 1.30 vs 0.64). Time-of-day was never the real cause — DIXON 05-29 was bad because of the
# prev-day support it ran into, which G4/G6 already catch. So 09:15 is allowed, with two conditions the
# user set: it needs MORE room to run, and a real GAP disqualifies it (a slight 0.2-0.3 is fine).
OPEN_MIN_ROOM_ATR = 1.5     # a 09:15 break needs more space ahead than a mid-session one
MAX_OPEN_GAP_PCT = 0.3      # PERCENT of price (user, 2026-07-24: 0.2-0.3% is fine; a real gap -> leave it)

_HURDLE_RE = re.compile(r"at ([\d.]+)ATR \((\w+), w([\d.]+)\)")

# FAILED HYPOTHESES for fresh-vs-continuation (ABB 22-Aug REJECT vs 20-Dec-23 FIRE — identical on every
# fact we compute). All tested and REJECTED by data; do NOT re-try without new evidence:
#  (a) "coil is a one-way DRIFT": accepted cases drift MORE (+4.00, +4.66) than the rejected (+2.09).
#  (b) "broken level must sit at/above the coil edge": rejects 24-Jun 12:05 (level 4 ATR below the coil).
#  (c) "a real coil is made of SMALL candles": separated a 12-case subset cleanly, then FAILED on the full
#      set — accepted maxCoilBody 1.23-3.41 ATR vs rejected 0.89-4.31. Complete overlap; no threshold.
# The distinction is real to the eye but is NOT a scalar on any current fact.


@dataclass
class BreakVerdict:
    symbol: str
    as_of: str
    direction: str
    fires: bool = False
    gates: dict = field(default_factory=dict)
    level: float = float("nan")
    level_rejections: int = 0
    level_tf: str = ""
    coil_bars: int = 0
    body_vs_coil: float = float("nan")
    close_pos: float = float("nan")
    room_atr: float = float("nan")
    hurdle: str = ""

    def why(self) -> str:
        out = [f"{self.symbol} {self.as_of} {self.direction} -> {'FIRE' if self.fires else 'no'}"]
        for g, (ok, detail) in self.gates.items():
            out.append(f"   [{'PASS' if ok else 'FAIL'}] {g:14s} {detail}")
        return "\n".join(out)


def _close_pos(o, h, l, c, direction) -> float:
    """Where the close sits inside the candle's own range: 1.0 = at the extreme in the break direction,
    0.5 = mid-candle (the user's rejected 'closed at half its high')."""
    rng = h - l
    if rng <= 0:
        return 0.0
    return float((h - c) / rng) if direction == "down" else float((c - l) / rng)


def _hurdle_blocks(hidden_hurdle: str) -> bool:
    if not hidden_hurdle:
        return False
    m = _HURDLE_RE.search(hidden_hurdle)
    if not m:
        return False
    dist, w = float(m.group(1)), float(m.group(3))
    return dist <= HURDLE_BLOCK_ATR or (w >= HURDLE_HTF_WEIGHT and dist <= HURDLE_HTF_ATR)


def _stateless_gates(symbol: str, as_of, direction: str | None = None):
    """G1/G3/G4/G5/G6/G7 — point-in-time gates (no move memory). Returns (verdict, ctx)."""
    from intelligence.archetype_facts import build_facts
    from intelligence.level_dynamics import level_context
    from marketdata import features as F
    from marketdata.candle_store import get_candle_store

    as_of = pd.Timestamp(as_of)
    cs = get_candle_store()
    decision_close = as_of + pd.Timedelta(minutes=5)
    raw = cs.get(symbol, "5m", as_of.normalize() - pd.Timedelta(days=12), as_of + pd.Timedelta(hours=6))
    t = pd.to_datetime(raw["ts"])
    c5 = raw[t + pd.Timedelta(minutes=5) <= decision_close].sort_values("ts").reset_index(drop=True)
    f5 = F.compute_features(c5, symbol, "5m")
    i = len(c5) - 1
    atr = float(f5["atr"].to_numpy(float)[i])
    o = c5["open"].to_numpy(float); h = c5["high"].to_numpy(float)
    l = c5["low"].to_numpy(float); cl = c5["close"].to_numpy(float)

    ff = build_facts(symbol, as_of, direction=direction)
    d = ff.direction
    v = BreakVerdict(symbol=symbol, as_of=f"{as_of:%Y-%m-%d %H:%M}", direction=d)
    ctx = dict(c5=c5, f5=f5, i=i, ff=ff, atr=atr, o=o, h=h, l=l, cl=cl)
    if not (np.isfinite(atr) and atr > 0) or i < 40:
        v.gates["data"] = (False, "insufficient history")
        return v, ctx
    v.coil_bars = ff.coil_bars; v.body_vs_coil = ff.body_vs_coil
    v.close_pos = round(_close_pos(o[i], h[i], l[i], cl[i], d), 2)
    v.room_atr = ff.room_atr_to_next; v.hurdle = ff.hidden_hurdle

    # ---- G3 REAL COIL ----
    v.gates["G3_coil"] = (ff.coil_bars >= MIN_COIL_BARS,
                          f"coil_bars={ff.coil_bars} (need >={MIN_COIL_BARS}); width={ff.coil_width_atr}ATR")

    # ---- G1 REAL LEVEL ----
    want = "support" if d == "down" else "resistance"
    prev_c = cl[i - 1]
    crossed = []
    try:
        for L in level_context(c5, symbol, "5m", feat=f5).levels:
            if L.kind != want or L.n_rejections < MIN_REJECTIONS or L.recency == "thin":
                continue
            if (d == "down" and cl[i] < L.price <= prev_c + 0.25 * atr) or \
               (d == "up" and cl[i] > L.price >= prev_c - 0.25 * atr):
                crossed.append(L)
    except Exception:
        pass
    if crossed:
        best = max(crossed, key=lambda L: (L.n_rejections, L.weight))
        v.level, v.level_rejections, v.level_tf = best.price, best.n_rejections, best.tf
        v.gates["G1_level"] = (True, f"broke tested {best.kind} {best.price:.1f} "
                                     f"({best.n_rejections} rejections, {best.recency})")
    else:
        v.gates["G1_level"] = (False, f"no tested {want} (>={MIN_REJECTIONS} rejections) crossed by this candle")

    # ---- G4 WALL: weighty level close ahead, or simply no room ----
    # NOT daily-only: DIXON has ZERO 1d bars in the store, so a daily-only rule was a silent NO-OP for it
    # and let breaks run straight into resistance ("that candle directly ran to resistance").
    walls = []
    if _hurdle_blocks(ff.hidden_hurdle):
        walls.append(ff.hidden_hurdle)
    if np.isfinite(ff.room_atr_to_next) and ff.room_atr_to_next < MIN_ROOM_ATR:
        walls.append(f"only {ff.room_atr_to_next}ATR of room ahead")
    v.gates["G4_wall"] = (not walls, f"{'; '.join(walls) if walls else 'clear'} "
                                     f"(blocks: weighty level within {HURDLE_BLOCK_ATR}ATR, or room <{MIN_ROOM_ATR}ATR)")

    # ---- G5 CANDLE QUALITY ----
    body_ok = bool(np.isfinite(ff.body_vs_coil) and ff.body_vs_coil >= MIN_BODY_VS_COIL)
    v.gates["G5_candle"] = (body_ok and v.close_pos >= MIN_CLOSE_POS,
                            f"body_vs_coil={ff.body_vs_coil} (need >={MIN_BODY_VS_COIL}); "
                            f"close_pos={v.close_pos} (need >={MIN_CLOSE_POS})")

    # ---- G6 CONTEXT: the HIGHER timeframe must not OPPOSE the break ----
    # BUG FIXED: build_facts' aligned_context is an OR ("5m structure is HH_HL *or* htf is up"), so 5m
    # micro-structure overruled the trend and produced "up-trade in a bearish structure" twice (DIXON
    # 06-08 and 03-30: 5m HH_HL with htf_1h=down -> declared "aligned", both hard failures).
    opposes = (d == "up" and ff.htf_trend_1h == "down") or (d == "down" and ff.htf_trend_1h == "up")
    v.gates["G6_context"] = (not opposes,
                             f"htf_1h={ff.htf_trend_1h or 'n/a'} vs {d}-break "
                             f"({'OPPOSED — against the higher-TF structure' if opposes else 'not opposed'}); "
                             f"1d={ff.lc_1d or 'MISSING'}")

    # ---- G7 MATURITY ----
    # The prior-same-direction-leg veto was REMOVED: it contradicts the archetype (an aligned prior move is
    # REQUIRED, not disqualifying) and caused the user's flagged misses (DIXON 06-03, leg 7.2 ATR after a
    # genuine 2h consolidation; 05-22, leg 3.7). That a real coil intervened is already enforced by G3.
    # ABSORPTION OVERRIDE (user, 2026-07-24, DIXON 22-May): a 15m HH_HL structure is NOT a continuation
    # when price already FACED the level and failed to break it, then COILED beneath it. "it directly
    # doesn't broke the resistance after the 1st move... it faced resistance... makes coil candle near
    # resistance slightly pushing price upside to the resistance and it was there for more than 3 hours
    # then break next day." That is the archetype's coil-under-resistance form (cf. HAL 17-Jun), not
    # maturity. So the maturity veto is waived when the broken level has prior rejections AND a real coil
    # sat under it.
    absorbed = (v.level_rejections >= MIN_REJECTIONS and ff.coil_bars >= COIL_UNDER_LEVEL_BARS)
    v.gates["G7_maturity"] = (not ff.maturity_flag or absorbed,
                              f"15m maturity={ff.maturity_flag or 'ok'}"
                              + (f" — WAIVED: {ff.coil_bars}-bar coil under a level with "
                                 f"{v.level_rejections} rejections = absorption, not continuation" if (ff.maturity_flag and absorbed) else ""))
    return v, ctx


def _prior_qualifying_break(symbol: str, ctx: dict, d: str):
    """MOVE MEMORY. Walk back over the bars since price left the coil; if one was itself a QUALIFYING
    break, this candle is a continuation of a move already signalled. A weak probe (half-body close) or a
    blocked candle does NOT count, so the real break still gets its shot."""
    ff = ctx["ff"]; cl = ctx["cl"]; o = ctx["o"]; h = ctx["h"]; l = ctx["l"]; i = ctx["i"]
    ts = pd.to_datetime(ctx["c5"]["ts"])
    edge = ff.coil_lo if d == "down" else ff.coil_hi
    avg = ff.coil_avg_body
    if not (np.isfinite(edge) and avg and avg > 0):
        return None
    for j in range(i - 1, max(0, i - MAX_LOOKBACK_BARS) - 1, -1):
        beyond = (cl[j] < edge) if d == "down" else (cl[j] > edge)
        if not beyond:
            return None
        if abs(cl[j] - o[j]) < MIN_BODY_VS_COIL * avg:
            continue
        if _close_pos(o[j], h[j], l[j], cl[j], d) < MIN_CLOSE_POS:
            continue
        vj, _ = _stateless_gates(symbol, ts.iloc[j], d)
        if vj.gates and all(ok for ok, _ in vj.gates.values()):
            return ts.iloc[j]
    return None


def evaluate_break(symbol: str, as_of, direction: str | None = None) -> BreakVerdict:
    """Is the bar at `as_of` a genuine compression-break of a REAL level? Causal, fully explainable."""
    v, ctx = _stateless_gates(symbol, as_of, direction)
    if "data" in v.gates:
        return v
    ts0 = pd.Timestamp(as_of)
    if ts0.hour == 9 and ts0.minute == 15:
        ts_all = pd.to_datetime(ctx["c5"]["ts"])
        prev = ts_all.dt.normalize() < ts0.normalize()
        prev_close = float(ctx["cl"][prev.values][-1]) if prev.any() else float("nan")
        gap = abs(ctx["o"][ctx["i"]] - prev_close) / prev_close * 100.0 if np.isfinite(prev_close) and prev_close else float("nan")
        room = ctx["ff"].room_atr_to_next
        gap_ok = (not np.isfinite(gap)) or gap <= MAX_OPEN_GAP_PCT
        room_ok = np.isfinite(room) and room >= OPEN_MIN_ROOM_ATR
        v.gates["G0_open915"] = (gap_ok and room_ok,
                                 f"09:15 break — gap={gap:.2f}% (need <={MAX_OPEN_GAP_PCT}%), "
                                 f"room={room}ATR (need >={OPEN_MIN_ROOM_ATR})")
    else:
        v.gates["G0_open915"] = (True, "not the 09:15 candle")
    prior = _prior_qualifying_break(symbol, ctx, v.direction)
    v.gates["G2_fresh"] = (prior is None,
                           "first qualifying break since leaving the coil" if prior is None
                           else f"a qualifying break already fired at {prior:%H:%M} — this is a continuation")
    v.fires = all(ok for ok, _ in v.gates.values())
    return v


# ── MULTI-TIMEFRAME, NARROW FORM. The user's teaching is "the 5m break looks weak but the 15m shows
# strength" — 15m CONFIRMS CANDLE STRENGTH; it is NOT a second chance to fire. A naive 5m-OR-15m was tried
# and measurably degraded quality (labelled cases 8/8 -> 6/8): the 15m path re-fired 24-Jun 12:10 (already
# signalled on 5m) and 08-Jul 13:50 (whose wall the 15m read missed). Every STRUCTURAL gate stays
# 5m-authoritative; the 15m may only satisfy G5_candle. A 15m bar starting 13:15 completes at 13:30, so a
# 15m-confirmed signal arrives up to 15 min later — recorded as lag, not hidden.
TF_MINUTES = {"5m": 5, "15m": 15}
MIN_COIL_BARS_TF = {"5m": 10, "15m": 6}


def _gates_on_tf(symbol: str, as_of, tf: str, direction: str | None = None) -> BreakVerdict:
    """Candle-strength read natively on `tf` (its own coil, its own candle)."""
    from intelligence.balance_state import read_balance
    from marketdata import features as F
    from marketdata.candle_store import get_candle_store

    as_of = pd.Timestamp(as_of)
    td = pd.Timedelta(minutes=TF_MINUTES[tf])
    bar_start = as_of.floor(f"{TF_MINUTES[tf]}min")
    cs = get_candle_store()
    raw = cs.get(symbol, tf, bar_start.normalize() - pd.Timedelta(days=20), bar_start + pd.Timedelta(hours=8))
    t = pd.to_datetime(raw["ts"])
    c = raw[t + td <= bar_start + td].sort_values("ts").reset_index(drop=True)
    v = BreakVerdict(symbol=symbol, as_of=f"{bar_start:%Y-%m-%d %H:%M}", direction=direction or "")
    if c is None or len(c) < 40:
        v.gates["data"] = (False, f"insufficient {tf} history"); return v
    feat = F.compute_features(c, symbol, tf)
    i = len(c) - 1
    atr = float(feat["atr"].to_numpy(float)[i])
    o = c["open"].to_numpy(float); h = c["high"].to_numpy(float)
    l = c["low"].to_numpy(float); cl = c["close"].to_numpy(float)
    if not (np.isfinite(atr) and atr > 0):
        v.gates["data"] = (False, "no atr"); return v
    bal = read_balance(symbol, tf, c["ts"].iloc[i], candles=c, feat=feat)
    d = direction or ("up" if cl[i] >= o[i] else "down")
    v.direction = d
    lo_b, hi_b = bal.band_lo - 0.6 * atr, bal.band_hi + 0.6 * atr
    start = i
    for k in range(i - 1, max(0, i - 60) - 1, -1):
        if lo_b <= cl[k] <= hi_b:
            start = k
        else:
            break
    coil_bars = max(0, i - start)
    bodies = np.abs(cl[start:i] - o[start:i])
    avg = float(np.mean(bodies)) if len(bodies) else float("nan")
    bvc = abs(cl[i] - o[i]) / avg if avg and avg > 0 else float("nan")
    v.coil_bars = coil_bars
    v.body_vs_coil = round(bvc, 2) if np.isfinite(bvc) else float("nan")
    v.close_pos = round(_close_pos(o[i], h[i], l[i], cl[i], d), 2)
    v.gates["G5_candle"] = (coil_bars >= MIN_COIL_BARS_TF[tf] and bool(np.isfinite(bvc) and bvc >= MIN_BODY_VS_COIL)
                            and v.close_pos >= MIN_CLOSE_POS,
                            f"[{tf}] coil={coil_bars} body_vs_coil={v.body_vs_coil} close_pos={v.close_pos}")
    v.fires = all(ok for ok, _ in v.gates.values())
    return v


def evaluate_break_mtf(symbol: str, as_of, direction: str | None = None):
    """Returns (verdict, tf, lag_minutes). 15m may only rescue a marginal CANDLE, never a structural gate."""
    v5 = evaluate_break(symbol, as_of, direction)
    if v5.fires:
        return v5, "5m", 0
    failed = [g for g, (ok, _) in v5.gates.items() if not ok]
    if failed != ["G5_candle"]:
        return v5, "5m", 0
    try:
        v15 = _gates_on_tf(symbol, as_of, "15m", v5.direction or direction)
    except Exception:
        return v5, "5m", 0
    ok15, det15 = v15.gates.get("G5_candle", (False, "n/a"))
    if not ok15:
        return v5, "5m", 0
    bar_start = pd.Timestamp(as_of).floor("15min")
    lag = int(((bar_start + pd.Timedelta(minutes=15)) - (pd.Timestamp(as_of) + pd.Timedelta(minutes=5))).total_seconds() // 60)
    v5.gates["G5_candle"] = (True, f"5m marginal, CONFIRMED on 15m -> {det15}")
    v5.fires = True
    return v5, "15m-confirm", max(0, lag)
