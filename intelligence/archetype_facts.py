"""
intelligence.archetype_facts — Stage A fact assembler for the compression→decisive-break archetype.

Combines the EXISTING organs (no rebuild) into ONE causal, point-in-time fact block the LLM recognizer
reads: balance_state (the coil: band/touches/duration/acceptance/break), level_dynamics (levels/recency/
room), lifecycle (compression/energy/structure per TF), htf trend — PLUS the archetype-specific facts
computed HERE because they need the coil window: coil-relative candle strength, close-decisiveness,
absorption-vs-strengthening (fine-grained slope OVER THE COIL, not coarse zigzag), approach speed (the
leg BEFORE the coil), and the TF-weighted cumulative room (recent-intraday → older-daily, dedup, weight).

Everything is strictly <= as_of. This is the eye's rich fact block; its correctness caps the LLM ceiling
(garbage-in), so every field here is eye-audited against the user's labeled charts before Stage B.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

ARCHETYPE_FACTS_VERSION = "1.0"


@dataclass
class ArchetypeFacts:
    symbol: str
    as_of: str
    price: float
    atr: float
    direction: str = ""                   # "up" | "down" — the break direction being evaluated

    # ── the coil (from balance_state, 5m) ──
    coil_lo: float = float("nan")
    coil_hi: float = float("nan")
    coil_width_atr: float = float("nan")
    coil_bars: int = 0                    # bars in the CURRENT consolidation (walked back while in-band)
    coil_touches_broken_edge: int = 0     # touches at the edge being broken (rejection count)
    coil_containment: float = float("nan")
    coil_tight: bool = False              # width + lifecycle COMPRESSION both agree it's tight
    balance_state: str = ""               # BALANCE / TRANSITION_UP/DOWN / TREND / UNFORMED
    accepted_beyond: bool = False         # balance says price accepted beyond the band (real break)
    expansion_on_break: float = float("nan")

    # ── the break candle (coil-relative) ──
    broke_level: float = float("nan")     # the edge the candle broke
    break_body: float = float("nan")
    coil_avg_body: float = float("nan")
    body_vs_coil: float = float("nan")    # break_body / coil_avg_body (the user's #1 criterion)
    close_decisiveness: float = float("nan")  # fraction of the break body BEYOND the level (0..1)

    # ── absorption / approach (coil-aware) ──
    interaction: str = ""                 # absorption | strengthening | neutral
    interaction_why: str = ""
    approach: str = ""                    # slow | gradual | fast
    approach_atr_per_bar: float = float("nan")

    # ── levels / room (TF-weighted cumulative) ──
    active_level: float = float("nan")
    active_kind: str = ""
    room_atr_to_next: float = float("nan")     # distance to the next significant level in break dir
    room_next_tf: str = ""                     # which TF that next level is significant on
    hidden_hurdle: str = ""                    # a level just beyond that could stall it (13-Dec-23 lesson)
    room_ahead: list = field(default_factory=list)   # [(price, kind, dist_atr, tf, weight), ...]

    # ── multi-TF context + maturity/exclusion ──
    lc_5m: str = ""; lc_15m: str = ""; lc_1h: str = ""; lc_1d: str = ""   # "energy/structure" per TF
    htf_trend_1h: str = ""; htf_trend_1d: str = ""
    aligned_context: str = ""             # is prior structure aligned with the break direction?
    maturity_flag: str = ""               # "" | "possible_HH_continuation" (exclusion hint)

    def render(self) -> str:
        d = asdict(self)
        d["room_ahead"] = "; ".join(f"{p:.1f}({k[:3]},{dist:+.1f}ATR,{tf},w{w})" for (p, k, dist, tf, w) in self.room_ahead[:5])
        lines = [f"{self.symbol} @ {self.as_of}  price={self.price} atr={self.atr}  break={self.direction}"]
        lines.append(f"COIL: {self.coil_lo}-{self.coil_hi} ({self.coil_width_atr}ATR, {self.coil_bars}bars, "
                     f"tight={self.coil_tight}, touches@edge={self.coil_touches_broken_edge}, contain={self.coil_containment})")
        lines.append(f"BREAK CANDLE: body {self.break_body} vs coil-avg {self.coil_avg_body} = {self.body_vs_coil}x ; "
                     f"close_decisiveness {self.close_decisiveness} (frac body beyond {self.broke_level}) ; "
                     f"balance={self.balance_state} accepted={self.accepted_beyond} expansion={self.expansion_on_break}")
        lines.append(f"INTERACTION: {self.interaction} — {self.interaction_why}  | APPROACH: {self.approach} ({self.approach_atr_per_bar} atr/bar)")
        lines.append(f"ROOM: next {self.active_kind}→ {self.room_atr_to_next}ATR ({self.room_next_tf})  hidden_hurdle: {self.hidden_hurdle or 'none'}")
        lines.append(f"  ahead: {d['room_ahead']}")
        lines.append(f"CONTEXT: 5m[{self.lc_5m}] 15m[{self.lc_15m}] 1h[{self.lc_1h}] 1d[{self.lc_1d}] | htf 1h={self.htf_trend_1h} 1d={self.htf_trend_1d}")
        lines.append(f"  aligned={self.aligned_context}  maturity={self.maturity_flag or 'ok'}")
        return "\n".join(lines)


def archetype_verdict(ff: "ArchetypeFacts") -> tuple[bool, float]:
    """The TRANSPARENT recognition rule, derived by Claude reasoning over the gold-set fact blocks
    (2026-07-24), NOT a trained model. 5 discriminators, each grounded in the user's teaching:
      · close_decisiveness >= 0.7   — the break candle closed decisively beyond the level
      · body_vs_coil       >= 2.0   — break body strongly exceeds the coil candles
      · interaction        != strengthening — not supply/demand FAILING on the wrong side
      · maturity_flag      not set  — not a mature HH/LL continuation
      · expansion_on_break >= 1.7   — real range expansion, not a slow drift
    In-sample: 14/15 clean gold cases. The 1 miss (fresh-vs-continuation) is a KNOWN gap needing a new
    fact, deliberately NOT patched with a per-case threshold (user: don't tighten at each step).
    Returns (is_archetype, strength) — strength for ranking/grading, not a probability."""
    ok = (ff.close_decisiveness >= 0.7 and ff.body_vs_coil >= 2.0
          and ff.interaction != "strengthening" and not ff.maturity_flag
          and ff.expansion_on_break >= 1.7)
    strength = 0.0
    if ok and np.isfinite(ff.body_vs_coil) and np.isfinite(ff.expansion_on_break):
        strength = round(min(ff.close_decisiveness, 1.0)
                         + min(ff.body_vs_coil, 8) / 8
                         + min(ff.expansion_on_break, 4) / 4
                         + (0.5 if ff.coil_tight else 0.0)
                         + (0.5 if ff.interaction == "absorption" else 0.0), 2)
    return ok, strength


_HURDLE_RE = re.compile(r"at ([\d.]+)ATR \(\w+, w([\d.]+)\)")


def archetype_verdict_challenger(ff: "ArchetypeFacts") -> tuple[bool, float]:
    """CHALLENGER rule (2026-07-24, well-powered OOS on 951 recognized). The champion
    (`archetype_verdict`) is UNTOUCHED — this is additive so BOTH are runnable and comparable later.
    Champion PLUS the only two levers that ROBUSTLY generalized out-of-sample:
      · time-of-day >= 09:45 — skip OPENING breaks ONLY (an opening candle breaks a 20-bar range that
        still includes the prior day's bars = gap/vol noise; opening breaks stop out 50% vs 28% after).
        LATE breaks are KEPT: they have the LOWEST stop-rate (13%) and the same FAV/ADV — an earlier
        [09:45,14:30] over-filter RAISED the stop-rate 28→35% and merely cut frequency, the opposite of
        the goal. Frequency is an OUTPUT, not a target (user, 2026-07-24: "trade better quality, don't
        force frequency down"). Keeps ~59% of champion → ~97/yr.
      · NO weighty CLOSE hidden-hurdle (w>=2 within 1.5 ATR) — the 13-Dec-23 hidden-support lesson (a
        break that immediately meets weighty opposing support/resistance stalls). Small statistical
        effect but a real, user-taught market mechanism (not a data-mined filter).
    body_vs_coil>=3 / expansion<=5 / daily!=DORMANT were DROPPED — they helped on the first 544 but
    deflated to inert/negative on 2x data (in-sample optimism). Net: ~71/yr, TEST stop-outs 41%→33%,
    FAV/ADV 2.42→3.26 (the tradeable win is drawdown, not hit-rate: CLEAN_WIN lift ~+5pp, not sig)."""
    ok, strength = archetype_verdict(ff)
    if not ok:
        return False, strength
    try:                                              # as_of = "YYYY-MM-DD HH:MM" (the break candle)
        hhmm = ff.as_of.split(" ")[1]
        if int(hhmm[:2]) + int(hhmm[3:5]) / 60.0 < 9.75:   # skip OPENING breaks only (see docstring)
            return False, strength
    except Exception:
        pass
    if ff.hidden_hurdle:                              # e.g. "support 1132.5 at 1.0ATR (1h, w2.5)"
        m = _HURDLE_RE.search(ff.hidden_hurdle)
        if m and float(m.group(2)) >= 2.0 and float(m.group(1)) <= 1.5:
            return False, strength
    return True, strength


def _coil_window(cl, band_lo, band_hi, i, atr, scan=60):
    """The CONSOLIDATION that PRECEDED the break: the break candle (bar i) has closed OUTSIDE the band,
    so walk back past it (up to 4 break bars) to the last in-band bar = coil END, then back while in-band
    = coil START. Returns (start, end) with end < i normally (the break candle is excluded from the coil)."""
    lo = band_lo - 0.6 * atr; hi = band_hi + 0.6 * atr
    end = i; steps = 0
    while end >= 0 and not (lo <= cl[end] <= hi) and steps < 4:
        end -= 1; steps += 1
    if end < 0:
        return i, i
    start = end
    for k in range(end, max(0, end - scan) - 1, -1):
        if lo <= cl[k] <= hi:
            start = k
        else:
            break
    return start, end


def _slope_per_bar(y) -> float:
    """OLS slope of y vs bar index (per-bar). Fine-grained — uses ALL bars, unlike coarse zigzag."""
    y = np.asarray(y, float); n = len(y)
    if n < 3:
        return 0.0
    x = np.arange(n)
    return float(np.polyfit(x, y, 1)[0])


def _absorption_over_coil(c, feat, band_lo, band_hi, start, i, direction, atr):
    """Fine-grained absorption over the coil: up-break → slope of bar LOWS (rising = higher lows =
    demand absorbing supply); down-break → slope of bar HIGHS (falling = lower highs = supply absorbing
    demand). Compression = bb-width tightening. Flat coil → neutral."""
    if i - start < 4:
        return "neutral", "coil too short to read absorption"
    low = c["low"].to_numpy(float)[start:i + 1]; high = c["high"].to_numpy(float)[start:i + 1]
    bbw = feat["bb_width_pct"].to_numpy(float)
    comp = np.isfinite(bbw[i]) and np.isfinite(bbw[start]) and bbw[i] < bbw[start] * 0.9
    per = atr / max(1, (i - start))       # a meaningful slope ~ rises/falls ~>=0.5 ATR across the coil
    if direction == "up":
        s = _slope_per_bar(low)
        if s > 0.4 * per:
            return "absorption", f"higher lows into resistance{' + compression' if comp else ''} (demand absorbing supply)"
        if s < -0.4 * per:
            return "strengthening", "lower lows — demand failing under resistance"
    else:
        s = _slope_per_bar(high)
        if s < -0.4 * per:
            return "absorption", f"lower highs into support{' + compression' if comp else ''} (supply absorbing demand)"
        if s > 0.4 * per:
            return "strengthening", "higher highs — supply failing at support"
    return "neutral", "flat/tight coil — decision rests on the break candle" if comp else "flat coil, no directional absorption"


def _approach_read(cl, start, atr, leg=12):
    """Approach speed = ATR/bar of the FINAL leg that brought price INTO the coil (the ~12 bars before
    the coil start). Fast rush = fragile; slow drift = accumulation. A short leg (not a 60-bar average)
    so the metric isn't diluted — the 60-bar version read 91% 'slow' at scale (no discrimination)."""
    j = max(0, start - leg)
    if start <= j:
        return "gradual", float("nan")
    travel = abs(cl[start] - cl[j]); n = start - j
    apb = (travel / n) / atr if atr > 0 else float("nan")
    if not np.isfinite(apb):
        return "gradual", float("nan")
    lab = "fast" if apb >= 0.35 else ("slow" if apb < 0.12 else "gradual")
    return lab, round(float(apb), 3)


def _close_decisiveness(o, cl_, level, direction) -> float:
    """Fraction of the break candle's BODY that is beyond the level in the break direction (0..1).
    The user's 'more body below/above the range than within'."""
    body = abs(cl_ - o)
    if body <= 0:
        return 0.0
    if direction == "up":
        beyond = cl_ - level
    else:
        beyond = level - cl_
    return float(np.clip(beyond / body, 0.0, 1.0))


def build_facts(symbol: str, as_of, direction: str | None = None) -> ArchetypeFacts:
    """Assemble the archetype fact block at `as_of`, point-in-time. `direction` may be given; else it is
    inferred from the balance-state transition / the last candle."""
    from marketdata import features as F
    from marketdata.candle_store import get_candle_store
    from intelligence.balance_state import read_balance
    from intelligence.level_dynamics import level_context
    from intelligence.lifecycle import lifecycle_frame
    from intelligence.story_context import htf_trend_series

    as_of = pd.Timestamp(as_of)
    cs = get_candle_store()
    # CAUSAL convention: decide at the 5m break candle's CLOSE. A bar (any TF) is usable iff its close
    # time (ts + interval) <= that decision moment — so the 5m break candle is COMPLETE, and higher-TF
    # bars are only COMPLETED ones (never a partial current 1h/1d bar, which would leak future data:
    # the exact cascade lookahead class of bug). `as_of + 1min` before this wrongly truncated the 5m
    # break candle to a partial bar (ABB 20-Dec 12:20 read 7258.45 partial vs 7227.95 complete).
    _TD = {"5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15),
           "1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1)}
    decision_close = as_of + _TD["5m"]

    def _causal(interval, back_days):
        raw = cs.get(symbol, interval, as_of.normalize() - pd.Timedelta(days=back_days),
                     as_of + pd.Timedelta(hours=6))
        if raw is None or len(raw) == 0:
            return raw
        t = pd.to_datetime(raw["ts"])
        return raw[t + _TD[interval] <= decision_close].sort_values("ts").reset_index(drop=True)

    c5 = _causal("5m", 12)
    f5 = F.compute_features(c5, symbol, "5m")
    i = len(c5) - 1
    atr = float(f5["atr"].to_numpy(float)[i]); price = float(c5["close"].to_numpy(float)[i])
    ff = ArchetypeFacts(symbol=symbol, as_of=f"{as_of:%Y-%m-%d %H:%M}", price=round(price, 2), atr=round(atr, 2))
    if not (np.isfinite(atr) and atr > 0) or i < 40:
        return ff

    bal = read_balance(symbol, "5m", as_of, candles=c5, feat=f5)
    ff.balance_state = bal.state
    ff.coil_lo, ff.coil_hi, ff.coil_width_atr = bal.band_lo, bal.band_hi, bal.width_atr
    ff.coil_containment = bal.containment
    ff.accepted_beyond = bal.pos.startswith("accepted")
    ff.expansion_on_break = bal.expansion

    # infer direction: balance transition > last-candle sign
    if direction is None:
        if bal.state.endswith("UP"):
            direction = "up"
        elif bal.state.endswith("DOWN"):
            direction = "down"
        else:
            o_i = float(c5["open"].to_numpy(float)[i]); direction = "up" if price >= o_i else "down"
    ff.direction = direction
    broke_level = bal.band_hi if direction == "up" else bal.band_lo
    ff.broke_level = round(broke_level, 2)

    cl = c5["close"].to_numpy(float); op = c5["open"].to_numpy(float)
    start, coil_end = _coil_window(cl, bal.band_lo, bal.band_hi, i, atr)
    ff.coil_bars = int(coil_end - start + 1)
    # coil-relative candle strength: break candle body vs the AVG body of the COIL candles (start..coil_end,
    # the break candle bar i is excluded — it closed outside the band)
    bodies = np.abs(cl[start:coil_end + 1] - op[start:coil_end + 1])
    ff.coil_avg_body = round(float(np.mean(bodies)), 2) if len(bodies) else float("nan")
    ff.break_body = round(float(abs(cl[i] - op[i])), 2)
    ff.body_vs_coil = round(ff.break_body / ff.coil_avg_body, 2) if ff.coil_avg_body and ff.coil_avg_body > 0 else float("nan")
    ff.close_decisiveness = round(_close_decisiveness(op[i], cl[i], broke_level, direction), 2)

    # touches at the broken edge (rejection count) from balance
    ff.coil_touches_broken_edge = bal.touches_hi if direction == "up" else bal.touches_lo

    # tightness: width small AND lifecycle says COMPRESSION recently
    lf5 = lifecycle_frame(c5, symbol, "5m")
    lc5 = lf5.iloc[-1] if not lf5.empty else None
    recent_struct = set(lf5["structure"].iloc[max(0, len(lf5) - ff.coil_bars - 1):].tolist()) if not lf5.empty else set()
    ff.coil_tight = bool(np.isfinite(bal.width_atr) and bal.width_atr <= 3.5 and "COMPRESSION" in recent_struct)

    # absorption + approach (coil-aware): absorption OVER THE COIL [start, coil_end]; approach = leg before
    ff.interaction, ff.interaction_why = _absorption_over_coil(c5, f5, bal.band_lo, bal.band_hi, start, coil_end, direction, atr)
    ff.approach, ff.approach_atr_per_bar = _approach_read(cl, start, atr)

    # levels / room (5m + higher TFs, TF-weighted cumulative)
    lvl5 = level_context(c5, symbol, "5m", feat=f5)
    ff.active_level, ff.active_kind = lvl5.active_price, lvl5.active_kind
    room_side = lvl5.room_up if direction == "up" else lvl5.room_down
    cum = [(z.price, z.kind, z.dist_atr, z.tf, z.weight) for z in room_side]
    # stack higher-TF levels for the older/major hurdles
    for tf in ("1h", "1d"):
        try:
            ctf = _causal(tf, 400 if tf == "1d" else 60)
            lvltf = level_context(ctf, symbol, tf)
            side = lvltf.room_up if direction == "up" else lvltf.room_down
            for z in side:
                cum.append((z.price, z.kind, round((z.price - price) / atr, 2), z.tf, z.weight))
        except Exception:
            pass
    # dedup across TFs (same price within 0.3 ATR → keep highest-weight/TF), order by |dist|
    cum.sort(key=lambda t: abs(t[2]))
    dedup = []
    for t in cum:
        if any(abs(t[0] - u[0]) <= 0.3 * atr for u in dedup):
            continue
        dedup.append(t)
    ff.room_ahead = dedup
    ahead = [t for t in dedup if (t[2] > 0.2 if direction == "up" else t[2] < -0.2)]
    if ahead:
        nxt = ahead[0]
        ff.room_atr_to_next = abs(nxt[2]); ff.room_next_tf = nxt[3]
        # hidden hurdle (the 13-Dec-23 lesson) = an OPPOSING barrier in the path that could stall the
        # move: a SUPPORT below a breakdown / a RESISTANCE above a breakout, WEIGHTY (w>=1.0 = recent or
        # higher-TF significant, not a minor w0.33 pivot) and CLOSE (<=1.5 ATR). Tightened from "any
        # level w>=0.5" which fired on 80% of points (no discrimination).
        hurdle_kind = "support" if direction == "down" else "resistance"
        close_beyond = [t for t in ahead if abs(t[2]) <= 1.5 and t[4] >= 1.0 and t[1] == hurdle_kind]
        if close_beyond:
            h = close_beyond[0]
            ff.hidden_hurdle = f"{h[1]} {h[0]:.1f} at {abs(h[2]):.1f}ATR ({h[3]}, w{h[4]})"

    # multi-TF lifecycle + htf trend + alignment
    ff.lc_5m = f"{lc5['energy']}/{lc5['structure']}" if lc5 is not None else ""
    for tf, attr in (("15m", "lc_15m"), ("1h", "lc_1h"), ("1d", "lc_1d")):
        try:
            ctf = _causal(tf, 400 if tf == "1d" else 40)
            lf = lifecycle_frame(ctf, symbol, tf)
            if not lf.empty:
                r = lf.iloc[-1]; setattr(ff, attr, f"{r['energy']}/{r['structure']}")
        except Exception:
            pass
    try:
        ff.htf_trend_1h = str(htf_trend_series(c5, symbol, "5m")[-1])   # 1h regime read on the 5m spine
    except Exception:
        pass
    # aligned context: for a down-break, prior structure LH_LL/BREAKDOWN or htf down = aligned
    struct5 = lc5["structure"] if lc5 is not None else ""
    if direction == "down":
        ff.aligned_context = "aligned" if struct5 in ("LH_LL", "BREAKDOWN") or ff.htf_trend_1h == "down" else "not-clearly-aligned"
    else:
        ff.aligned_context = "aligned" if struct5 in ("HH_HL",) or ff.htf_trend_1h == "up" else "not-clearly-aligned"
    # maturity / HH-continuation exclusion hint: a TRENDING higher-TF structure (15m HH_HL for up /
    # LH_LL for down) with a NON-tight coil = a trend-continuation, not a tight coil-break. Using the
    # 15m TREND structure + not-tight discriminates the exclusion (7-May 12:35: 15m HH_HL, wide coil)
    # from a valid coil-break (17-Jun: 15m RANGE) — the 5m-structure version wrongly flagged valid ones.
    trend_struct = "HH_HL" if direction == "up" else "LH_LL"
    s15 = ff.lc_15m.split("/")[-1] if ff.lc_15m else ""
    if s15 == trend_struct and not ff.coil_tight:
        ff.maturity_flag = "possible_HH_continuation"
    return ff
