"""
intelligence.directional_bias — the EARLY, graded directional LEAN (one of the two clocks).

The user's distinction (2026-06-18): BIAS ≠ REGIME.
  · BIAS    — fast, graded, cheap, ACCUMULATING. "Sellers are gaining control." Built from evidence
              that piles up BEFORE any break. Can lean bearish at 10:00 with MEDIUM confidence.
  · REGIME  — slow, binary, EXPENSIVE/earned (the lifecycle htf flip). CONFIRMS the bias, late.

CRUCIALLY MULTI-TIMEFRAME (the v1 5m-only version got seduced by a morning higher-high on JSWSTEEL and
missed the topping). The lean is read where the trader reads it:
  · DAILY   — extended after a big run, then PAUSING/rolling at the top = bearish-vulnerable (topping);
              the mirror for a washed-out bottom. (NOT "drift up = bullish" — a run-then-pause is risk.)
  · 15m     — the structure backbone: lower-highs / failing to reclaim the high (rolling over) vs HH/HL.
  · 5m      — the fine triggers: intraday support lost / high reclaimed, recent momentum.
  · 1h      — the regime word (lagging, low weight).
Reported as FACTS (direction + confidence + the reasons that fired); the reasoner weighs it. It is the
TRUE shakeout-vs-reversal discriminator: a counter-trend birth WITH an already-leaning bias is an
evidenced transition; one AGAINST the bias is a shakeout.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Bias:
    """A LAYERED multi-TF read — NOT a collapsed scalar. Each timeframe is stated as its own fact; the
    reasoner weighs the hierarchy (a 15m pullback inside a bull day = 'up, pulling back'; a 15m rollover
    + daily topping + 1h turning = 'bearish'). Collapsing to one weighted number mis-ranks timeframes."""
    daily: str = ""
    htf: str = ""
    struct: str = ""               # prevailing INTRADAY structure (from lifecycle: HH_HL/LH_LL/RANGE…)
    tf5: str = ""                  # the IMMEDIATE 5m action (pullback / lost support / momentum)

    def layers(self) -> list[tuple[str, str]]:
        return [(tf, v) for tf, v in (("daily", self.daily), ("1h", self.htf),
                                      ("intraday", self.struct), ("now-5m", self.tf5)) if v]

    def any(self) -> bool:
        return bool(self.daily or self.htf or self.struct or self.tf5)


def resample_15m(c: pd.DataFrame) -> pd.DataFrame:
    """5m → 15m OHLC (right-labelled: a 15m bar's stamp is the END of its window, so it is 'completed'
    once price time reaches it). Adds a simple ATR15. Computed ONCE per session."""
    x = c.copy()
    x.index = pd.to_datetime(x["ts"])
    o = x.resample("15min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    if len(o):
        tr = (o["high"] - o["low"]).to_numpy(float)
        o["atr15"] = pd.Series(tr, index=o.index).rolling(14, min_periods=3).mean()
    return o.reset_index().rename(columns={"index": "ts"})


def daily_context(c_daily: pd.DataFrame, as_of) -> str:
    """Daily lean (causal — only bars strictly BEFORE today), as a descriptive fact. The signals that
    matter: ROLLING OVER NEAR HIGHS (failing to make a new daily high + closing weak after an advance =
    distribution/topping, bearish-vulnerable) vs TRENDING UP (making/holding new highs); mirror for the
    lows. NOT 'drift-up = bullish' — a roll-over at the top is the warning, not continuation."""
    if c_daily is None or len(c_daily) < 6:
        return ""
    d = c_daily[pd.to_datetime(c_daily["ts"]) < pd.Timestamp(as_of).normalize()].sort_values("ts")
    if len(d) < 6:
        return ""
    cl = d["close"].to_numpy(float); hi = d["high"].to_numpy(float); lo = d["low"].to_numpy(float)
    rng = float(np.nanmean(hi[-10:] - lo[-10:])) or 1e-9
    last_hi = hi[-1]; ref_hi = float(np.nanmax(hi[-5:-1]))   # last daily bar vs the recent run of highs
    last_lo = lo[-1]; ref_lo = float(np.nanmin(lo[-5:-1]))
    chg4 = (cl[-1] - cl[-4]) / rng                           # close trend over ~3 prior days
    # FACTS, not a thresholded verdict — describe the daily high/low sequence + close trend; reasoner judges
    hi_part = ("a new high" if last_hi > ref_hi + 0.1 * rng else
               "a LOWER high (failed to make a new high)" if last_hi < ref_hi - 0.2 * rng else
               "near the recent high")
    cl_part = ("closing up" if chg4 >= 0.4 else "closing weak" if chg4 <= -0.4 else "closing flat")
    if last_hi < ref_hi - 0.2 * rng or last_lo < ref_lo - 0.2 * rng or last_hi > ref_hi + 0.1 * rng:
        return f"last bar {hi_part}, {cl_part} (vs ~3d ago)"
    return ""


def opening_context(o, h, l, cl, atr, today_idx, prev_idx) -> str:
    """How the day OPENED, read against how YESTERDAY ENDED — a SESSION fact, NOT a rule. The user's
    direction (2026-06-19): don't gate the open on a '>2 ATR = provisional' threshold; OBSERVE it from
    what we already see — the gap, the prior close quality, and the last-hour behaviour (was yesterday
    ending on a live breakout/breakdown, or fading?). Stated descriptively so the reasoner can tell a
    CONTINUATION gap (gap WITH a strong directional close still running into the bell) from an
    EXHAUSTION/TRAP gap (gap AGAINST a faded/reversed close). Combine with the prevailing trend/structure
    (already in SESSION memory) — those carry the regime; this carries the overnight handoff."""
    today_idx = np.asarray(today_idx); prev_idx = np.asarray(prev_idx)
    if len(today_idx) == 0 or len(prev_idx) < 3:
        return ""
    a = float(atr[today_idx[0]])
    if not (a and a > 0):
        return ""
    today_open = float(o[today_idx[0]]); prev_close = float(cl[prev_idx[-1]])
    gap = (today_open - prev_close) / a
    gap_part = (f"gapped UP {gap:.1f} ATR" if gap >= 0.3 else
                f"gapped DOWN {abs(gap):.1f} ATR" if gap <= -0.3 else "opened flat (no gap)")
    # where did yesterday close inside its own day range?
    ph = float(np.nanmax(h[prev_idx])); pl = float(np.nanmin(l[prev_idx]))
    cpos = (prev_close - pl) / ((ph - pl) or 1e-9)
    close_part = ("closed near yesterday's HIGH" if cpos >= 0.75 else
                  "closed near yesterday's LOW" if cpos <= 0.25 else "closed mid-range yesterday")
    # last-hour behaviour (final ~12 prior 5m bars ≈ 1h): carried into the close, or faded/reversed?
    lh = prev_idx[-12:] if len(prev_idx) >= 12 else prev_idx
    net = (float(cl[lh[-1]]) - float(cl[lh[0]])) / a
    hh = float(np.nanmax(h[lh])); ll = float(np.nanmin(l[lh]))
    end_pos = (prev_close - ll) / ((hh - ll) or 1e-9)
    if net >= 0.6:
        hour_part = ("last hour pushed UP into the close (a move still running into the bell)" if end_pos >= 0.7
                     else "last hour rallied then FADED off the high (gave it back into the close)")
    elif net <= -0.6:
        hour_part = ("last hour sold DOWN into the close (weakness running into the bell)" if end_pos <= 0.3
                     else "last hour fell then BOUNCED off the low (reclaimed into the close)")
    else:
        hour_part = "last hour quiet/balanced into the close"
    return f"{gap_part}; {close_part}; {hour_part}"


_STRUCT_DESC = {
    "HH_HL": "uptrend — clear higher highs/lows",
    "LH_LL": "downtrend — lower highs/lows",
    "RANGE": "range — no clear trend (a breakout here is unproven; could be a failed try)",
    "BREAKOUT": "breaking out (up)",
    "BREAKDOWN": "breaking down",
}


def read_bias(c: pd.DataFrame, feat: pd.DataFrame, i: int, day_date,
              daily_ctx: str, cur_htf: str = "", cur_struct: str = "") -> Bias:
    """LAYERED multi-timeframe read at 5m bar i (point-in-time) — each TF stated separately, NO scalar
    collapse. The prevailing INTRADAY structure is taken from lifecycle (`cur_struct`: HH_HL/LH_LL/RANGE
    — the validated broader-trend read, NOT a myopic local window); the 5m layer carries only the
    IMMEDIATE action so the reasoner can tell a pullback-in-an-uptrend from a break-of-a-range."""
    h = c["high"].to_numpy(float); l = c["low"].to_numpy(float); cl = c["close"].to_numpy(float)
    atr = feat["atr"].to_numpy(float); a = atr[i]
    if not (a and a > 0) or i < 9:
        return Bias()
    b = Bias(daily=daily_ctx or "",
             htf=(f"1h regime {cur_htf}" if cur_htf in ("bull", "bear") else ""),
             struct=_STRUCT_DESC.get(cur_struct, "") if cur_struct else "")

    # ── 5m IMMEDIATE action: intraday support lost / high reclaimed; recent momentum ──
    tidx = np.where(pd.to_datetime(c["ts"]).dt.date.to_numpy() == day_date)[0]
    tidx = tidx[tidx <= i]
    parts = []
    if len(tidx) >= 8:
        early = tidx[:-3]
        ilo = float(np.nanmin(l[early])); ihi = float(np.nanmax(h[early]))
        if cl[i] < ilo - 0.1 * a:
            parts.append(f"lost intraday support ({ilo:.2f})")
        elif cl[i] > ihi + 0.1 * a:
            parts.append(f"reclaimed intraday high ({ihi:.2f})")
    mom = (cl[i] - cl[i - 5]) / a
    if mom <= -0.6:
        parts.append("downside momentum")
    elif mom >= 0.6:
        parts.append("upside momentum")
    b.tf5 = "; ".join(parts)
    return b
