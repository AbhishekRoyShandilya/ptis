"""
intelligence.concept_kb — the CONCEPT layer (component 3): "what-matters-when" as CHECKABLE,
FALSIFIABLE knowledge, not hardcoded rules.

A concept is a trader's question made measurable + grounded:
  1. MEASURE  — a causal, point-in-time function over the candle window -> a graded bucket
                (e.g. Freshness -> FRESH / MID / STRETCHED). No lookahead.
  2. VALIDATE — split the real operating-process trades (structural risk + let-run R) by the
                concept's buckets and check it SEPARATES outcomes, stably across BOTH halves
                (2022-23 vs 2024-25) AND WITHIN sector (so it isn't a sector proxy). The
                both-halves/within-sector gate is the confabulation guard pointed at concepts.
  3. WEIGH    — only concepts that pass carry weight; the cognition agent cites the validated
                effect ("FRESH + CLEAR + EARLY -> let-run expR +0.18 [stable]"), never theory.

Sector is treated as CONTEXT, never a veto (user direction 2026-06-15): the discriminator is
the CONDITION (fresh/room/story-position), not the label — a rare Pharma trend phase that reads
EARLY/FRESH/CLEAR should qualify like HAL. The within-sector validation is exactly what proves a
concept is a real condition and not the sector in disguise.

The user's five, in priority order: Freshness, Room, Narrative Position, Compression->Release,
Acceptance/Absorption.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONCEPT_KB_VERSION = "1.0"
_VALIDATION = Path("data/lake/situations/concept_validation.json")

# causal lookback windows (bars)
_W_LEG = 30      # current-leg origin for freshness
_K_ROOM = 240    # overhead/underfoot structure for room
_W_PUSH = 40     # prior-push count for narrative position
_W_COMP = 15     # pre-move window for compression


@dataclass
class ConceptReading:
    concept: str
    bucket: str          # the graded label, e.g. "FRESH"
    value: float         # the raw measured number behind the bucket


# ── MEASUREMENTS (causal, point-in-time; d = +1 long / -1 short) ──────────────

def freshness(h, l, c, atr, i, d) -> ConceptReading:
    """How extended is THIS leg — fresh (just started) vs stretched (chasing)."""
    a = atr[i] if atr[i] > 0 else 1e-9
    lo = max(0, i - _W_LEG)
    origin = l[lo:i + 1].min() if d > 0 else h[lo:i + 1].max()
    ext = (c[i] - origin) / a if d > 0 else (origin - c[i]) / a
    bucket = "FRESH" if ext < 2 else ("MID" if ext < 4 else "STRETCHED")
    return ConceptReading("freshness", bucket, round(float(ext), 2))


def room(h, l, c, atr, i, d) -> ConceptReading:
    """Clear air to the next structure the move must fight — capped vs room to run."""
    a = atr[i] if atr[i] > 0 else 1e-9
    lo = max(0, i - _K_ROOM)
    if d > 0:
        ahead = h[lo:i]; ahead = ahead[ahead > c[i]]
        gap = (ahead.min() - c[i]) / a if ahead.size else 99.0
    else:
        ahead = l[lo:i]; ahead = ahead[ahead < c[i]]
        gap = (c[i] - ahead.max()) / a if ahead.size else 99.0
    bucket = "CAPPED" if gap < 1 else ("SOME" if gap < 3 else "CLEAR")
    return ConceptReading("room", bucket, round(float(gap), 2))


def narrative_position(h, l, c, atr, i, d) -> ConceptReading:
    """WHERE in the story — beginning of expansion (1st break) vs last push of a mature move.
    Measured from the SEQUENCE (count of prior pushes in the leg) + leg maturity, NOT this bar."""
    a = atr[i] if atr[i] > 0 else 1e-9
    lo = max(20, i - _W_PUSH)
    # count prior breakouts in the same direction over the window = which push is this
    pushes = 0
    for j in range(lo, i):
        if d > 0 and c[j] > h[j - 20:j].max():
            pushes += 1
        elif d < 0 and c[j] < l[j - 20:j].min():
            pushes += 1
    # leg maturity (reuse freshness extension)
    o = l[max(0, i - _W_LEG):i + 1].min() if d > 0 else h[max(0, i - _W_LEG):i + 1].max()
    ext = (c[i] - o) / a if d > 0 else (o - c[i]) / a
    if pushes <= 1 and ext < 3:
        bucket = "EARLY"          # first break / beginning of expansion
    elif pushes >= 3 or ext >= 4:
        bucket = "LATE"           # third+ push / last gasp of a mature move
    else:
        bucket = "MID"
    return ConceptReading("narrative_position", bucket, float(pushes))


def compression_release(h, l, c, atr, i, d) -> ConceptReading:
    """Did this move erupt out of a coil (energy storage -> release) or out of noise."""
    a = atr[i] if atr[i] > 0 else 1e-9
    lo = max(0, i - _W_COMP - 3)
    pre = c[lo:i - 2]
    if pre.size < 6:
        return ConceptReading("compression_release", "NA", np.nan)
    coil = (pre.max() - pre.min()) / a              # pre-move range in ATR (small = coiled)
    bucket = "POST_COIL" if coil < 1.6 else "NO_COIL"
    return ConceptReading("compression_release", bucket, round(float(coil), 2))


def acceptance(h, l, c, atr, i, d) -> ConceptReading:
    """Did the market ACCEPT price past the level (held, closed strong) or absorb/reject it."""
    rng = (h[i] - l[i]) or 1e-9
    close_pos = (c[i] - l[i]) / rng if d > 0 else (h[i] - c[i]) / rng   # 1 = closed at the extreme
    lvl = h[i - 20:i].max() if d > 0 else l[i - 20:i].min()
    beyond = sum(1 for j in (i - 2, i - 1, i)
                 if (c[j] > lvl if d > 0 else c[j] < lvl))
    if close_pos >= 0.6 and beyond >= 2:
        bucket = "ACCEPTED"
    elif close_pos < 0.4 or beyond == 0:
        bucket = "REJECTED"
    else:
        bucket = "NEUTRAL"
    return ConceptReading("acceptance", bucket, round(float(close_pos), 2))


CONCEPTS = {
    "freshness": freshness,
    "room": room,
    "narrative_position": narrative_position,
    "compression_release": compression_release,
    "acceptance": acceptance,
}
# preferred bucket order (good -> bad) for reporting/monotonicity
BUCKET_ORDER = {
    "freshness": ["FRESH", "MID", "STRETCHED"],
    "room": ["CLEAR", "SOME", "CAPPED"],
    "narrative_position": ["EARLY", "MID", "LATE"],
    "compression_release": ["POST_COIL", "NO_COIL"],
    "acceptance": ["ACCEPTED", "NEUTRAL", "REJECTED"],
}


def read_all(h, l, c, atr, i, d) -> dict[str, ConceptReading]:
    out = {}
    for name, fn in CONCEPTS.items():
        try:
            out[name] = fn(h, l, c, atr, i, d)
        except Exception:
            out[name] = ConceptReading(name, "NA", np.nan)
    return out


# ── VALIDATION (falsify each concept against the outcome IT claims) ────────────
# Each concept is measured as a raw value (v_<concept>); we tercile it into a GOOD third and a
# BAD third (data-driven, so buckets are balanced — no hand-tuned thresholds), then test the
# claim against the RIGHT target:
#   good_high = does a HIGHER raw value mean the GOOD condition?
#   target    = the outcome the concept actually claims to move
# LESSON (2026-06-15, partner's catch): validate against REALIZED return, never MFE-in-R.
# MFE-in-R is denominator-inflated — FRESH entries have tighter stops (smaller R), so MFE/R looks
# huge while the ABSOLUTE move (ATR/%) is identical and realized return is ~zero (denom_check.py,
# 272k trades: FRESH mfe 6.44 ATR vs STRETCHED 6.40 ATR; hold_stop effect +0.001). A concept earns
# weight only if it moves the TRADEABLE outcome (hold_stop / net). MFE-in-ATR (denominator-free) is
# kept only as a run-distance signal for EXITS, never for selection validation.
GOOD_HIGH = {"freshness": False, "room": True, "narrative_position": False,
             "compression_release": False, "acceptance": True}
TARGET = {"freshness": "hold_stop", "room": "hold_stop", "narrative_position": "hold_stop",
          "compression_release": "hold_stop", "acceptance": "hold_stop"}
TARGET_LABEL = {"mfe_R": "realized MFE (R) — how far it ran",
                "reversed": "reversal rate — ended negative",
                "net_R": "net follow-through (R)"}


@dataclass
class ConceptValidation:
    concept: str
    n: int
    target: str
    good: dict             # {n, val, A, B}   (target metric for the GOOD tercile)
    bad: dict
    effect: float          # improvement of GOOD over BAD on the target (sign-normalised: +good)
    both_halves: bool
    within_sector: float
    validated: bool
    note: str


def _terciles(df: pd.DataFrame, vcol: str, good_high: bool):
    q1, q2 = df[vcol].quantile([1 / 3, 2 / 3])
    if q1 == q2:                                   # degenerate value distribution
        return None, None
    hi = df[df[vcol] >= q2]; lo = df[df[vcol] <= q1]
    return (hi, lo) if good_high else (lo, hi)     # (good, bad)


def _metric(g: pd.DataFrame, target: str) -> float:
    # normalise so HIGHER = better for every target (reversal is inverted)
    return -g[target].mean() if target == "reversed" else g[target].mean()


def validate(trades: pd.DataFrame, concept: str, min_n: int = 3000) -> ConceptValidation:
    vcol = f"v_{concept}"; target = TARGET[concept]
    df = trades[trades[vcol].notna()]
    if len(df) < min_n:
        return ConceptValidation(concept, len(df), target, {}, {}, 0.0, False, 0.0, False,
                                 "insufficient data")
    good, bad = _terciles(df, vcol, GOOD_HIGH[concept])
    if good is None or len(good) < 200 or len(bad) < 200:
        return ConceptValidation(concept, len(df), target, {}, {}, 0.0, False, 0.0, False,
                                 "degenerate value distribution")

    def side(g):
        a = g[g.year.isin((2022, 2023))]; b = g[g.year.isin((2024, 2025))]
        return {"n": int(len(g)), "val": round(float(_metric(g, target)), 3),
                "A": round(float(_metric(a, target)), 3) if len(a) else None,
                "B": round(float(_metric(b, target)), 3) if len(b) else None}

    gs, bs = side(good), side(bad)
    effect = gs["val"] - bs["val"]
    both = (gs["A"] is not None and bs["A"] is not None and gs["A"] > bs["A"] and gs["B"] > bs["B"])
    secs = []
    for s, grp in df.groupby("sector"):
        gg, bb = _terciles(grp, vcol, GOOD_HIGH[concept])
        if gg is not None and len(gg) >= 60 and len(bb) >= 60:
            secs.append(_metric(gg, target) > _metric(bb, target))
    within = round(float(np.mean(secs)), 2) if secs else 0.0
    thr = 0.05 if target != "reversed" else 0.03      # reversal-rate deltas are smaller in scale
    validated = bool(effect > thr and both and within >= 0.6)
    note = ("VALIDATED — separates the claimed outcome, stable both halves, holds within sector"
            if validated else
            "weak/unstable" if effect > thr / 2 else "does not separate the claimed outcome")
    return ConceptValidation(concept, len(df), target, gs, bs, round(effect, 3), both, within,
                             validated, note)


def validate_all(trades: pd.DataFrame) -> list[ConceptValidation]:
    res = [validate(trades, name) for name in CONCEPTS]
    res.sort(key=lambda v: -v.effect)
    _VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    _VALIDATION.write_text(json.dumps(
        {v.concept: {"target": v.target, "effect": v.effect, "validated": v.validated,
                     "both_halves": v.both_halves, "within_sector": v.within_sector,
                     "n": v.n, "good": v.good, "bad": v.bad, "note": v.note} for v in res},
        indent=2), encoding="utf-8")
    return res


def load_validation() -> dict:
    return json.loads(_VALIDATION.read_text(encoding="utf-8")) if _VALIDATION.exists() else {}


def narrate_validations(res: list[ConceptValidation]) -> str:
    L = [f"CONCEPT VALIDATION — each concept falsified against the outcome IT claims "
         f"(GOOD vs BAD tercile; A=2022-23 B=2024-25)\n{'='*94}"]
    for v in res:
        if not v.good:
            L.append(f"\n[{v.concept}]  n={v.n:,}  -> · {v.note}"); continue
        flag = "✓ VALIDATED" if v.validated else "· " + v.note
        L.append(f"\n[{v.concept}]  target = {TARGET_LABEL[v.target]}   "
                 f"(higher=better; reversal shown as −rate)")
        L.append(f"    n={v.n:,}  effect={v.effect:+.3f}  both_halves={'Y' if v.both_halves else 'N'}"
                 f"  within_sector={v.within_sector:.0%}  -> {flag}")
        L.append(f"    GOOD  n={v.good['n']:>6}  {v.good['val']:+.3f}  "
                 f"(A {v.good['A']:+.3f} / B {v.good['B']:+.3f})")
        L.append(f"    BAD   n={v.bad['n']:>6}  {v.bad['val']:+.3f}  "
                 f"(A {v.bad['A']:+.3f} / B {v.bad['B']:+.3f})")
    return "\n".join(L)
