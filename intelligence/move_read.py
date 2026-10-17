"""
intelligence.move_read — the LIVE, CUMULATIVE read of a move (the trader's running mind).

NOT a per-candle classifier and NOT a probability. This is the organ that the user kept
asking for: it carries ONE evolving understanding of a move across its whole life and
speaks it the way a trader sitting in the seat would — "how the day opened, how this move
started, what's fuelled it since, what zone it's walking into, how its pullbacks resolved,
and what it's doing right now."

A single candle is only ever a TRIGGER inside this running context (the breakout bar, the
engulf) — never the unit of meaning. Everything here is built cumulatively from the day's
open up to `as_of`, strictly point-in-time (no lookahead, no future bars).

Reuses the live-awareness blocks already built:
  session_clock   — where in the day we are (open drive vs midday drift vs EOD/BTST)
  momentum_run    — the move-tracking spine (ignition bar, run-high, bars-in-run)
  story_context   — the ambient state holding during the move (HTF 1h, tight-range, etc.)
  marketdata.features — vwap / ema / atr / rsi / rvol for fuel + stretch + zones

Output is a READ object (structured facts) + narrate() (plain trader English). No edge
claim is attached — this is OBSERVATION, the thing the user wants to verify by eye first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MOVE_READ_VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Zone:
    name: str
    price: float
    dist_atr: float          # +above current price, -below
    side: str                # "above" | "below"


@dataclass
class Pullback:
    start_ts: pd.Timestamp
    depth_atr: float
    bars: int
    resolved: str            # "resumed" | "failed" | "live" (still in it / unresolved)


@dataclass
class MoveRead:
    symbol: str
    interval: str
    as_of: pd.Timestamp
    direction: str           # "up" | "down" | "none"
    # day context (DAY MEMORY — how today started)
    phase: str = ""
    bars_since_open: int = 0
    gap_atr: float = 0.0
    open_type: str = ""             # "gap-up open-drive" | "flat range open" | ...
    vwap_session: str = ""          # "above VWAP all session" | "reclaimed VWAP" | ...
    htf_1h: str = "flat"
    pos_vs_pdh: float = np.nan      # (close - prev_day_high)/atr
    pos_vs_vwap_atr: float = np.nan
    # STRUCTURAL MEMORY — what happened BEFORE the move began (the earlier chapters)
    structure_before: list[str] = field(default_factory=list)   # TODAY's intraday structure
    structure_prior: list[str] = field(default_factory=list)    # PRIOR sessions (daily/HTF), labeled as prior
    # the move
    active: bool = False
    ignition_ts: pd.Timestamp | None = None
    ignition_type: str = ""
    bars_in_run: int = 0
    ext_atr: float = 0.0            # favourable travel from ignition, in ATR (up: run-high; down: run-low)
    run_high: float = np.nan        # = run_extreme (price extreme of the run, back-compat)
    run_extreme: float = np.nan     # the run's favourable price extreme (high for up, low for down)
    # fuel / structure
    vol_early: float = np.nan       # rvol on the early pushes
    vol_recent: float = np.nan      # rvol on the recent pushes
    fuel: str = ""                  # "expanding" | "steady" | "fading"
    rsi: float = np.nan
    stretch_atr: float = np.nan     # close above ema20, in ATR (exhaustion proxy)
    pullbacks: list[Pullback] = field(default_factory=list)
    # zones around price
    zones: list[Zone] = field(default_factory=list)
    # the trader's mental summary (synthesis)
    move_source: list[str] = field(default_factory=list)   # WHY THIS MOVE EXISTS (the ✓ list)
    primary_driver: str = ""                               # the single dominant reason
    threats: list[str] = field(default_factory=list)       # CURRENT THREATS (what's against it now)
    # the move's life-stage + the tension behind it (describe → understand)
    move_phase: str = ""           # EARLY | MIDDLE | LATE — where in its life the move is
    energy: str = ""               # energy release | building | healthy continuation | exhaustion risk | unproven
    energy_reasons: list[str] = field(default_factory=list)
    # the running state read
    state: str = ""                 # EXTENDING | PULLING_BACK | STALLING | WEAKENING | DORMANT
    notes: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
def _round_step(price: float) -> float:
    """A 'psychological' round-number step that scales with price."""
    if price >= 20000:  return 100.0
    if price >= 5000:   return 50.0
    if price >= 1000:   return 10.0
    if price >= 200:    return 5.0
    return 1.0


def read_move(symbol: str, interval: str, as_of, candles: pd.DataFrame | None = None,
              lookback_days: int = 6, feat: pd.DataFrame | None = None,
              htf_1h: str | None = None, runs: tuple | None = None,
              prepared: bool = False) -> MoveRead:
    """Build the cumulative read of the move active at `as_of`, point-in-time.

    `candles` may be supplied (already-loaded df with ts/open/high/low/close/volume);
    otherwise pulled from the lake. Only bars with ts <= as_of are ever used. `feat` may be supplied
    (already computed on this SAME `candles`, row-aligned) to skip recompute — bit-identical (features
    are causal and unaffected by add_session_features)."""
    from marketdata import features as F
    from intelligence.session_clock import add_session_features, session_phase
    from intelligence.momentum_run import detect_up_runs, detect_down_runs

    as_of = pd.Timestamp(as_of)
    if candles is None:
        from marketdata.candle_store import get_candle_store
        start = as_of.normalize() - pd.Timedelta(days=lookback_days + 4)
        candles = get_candle_store().get(symbol, interval, start, as_of)
    # STREAMING fast-path: when the caller (narrate_session) guarantees `candles` is ALREADY the clean
    # slice — bars ts<=as_of, sorted, reset-index, session-featured — skip the per-bar preamble
    # (datetime-cast + filter + sort + reset + add_session_features), which is redundant O(i) work rebuilt
    # every bar. All of it is causal/point-in-time, so the fast-path is bit-identical (verified).
    if prepared:
        c = candles
    else:
        c = candles[pd.to_datetime(candles["ts"]) <= as_of].sort_values("ts").reset_index(drop=True)
    R = MoveRead(symbol=symbol, interval=interval, as_of=as_of, direction="none")
    if c is None or len(c) < 60:
        R.state = "DORMANT"; R.notes.append("not enough history to read"); return R

    if not prepared:
        c = add_session_features(c)
    if feat is None or len(feat) != len(c):
        feat = F.compute_features(c, symbol, interval)   # window-warmed (validated); not the shared store (08)
    # move-tracking spine — BOTH directions (symmetric: bullish AND bearish moves). `feat` is computed
    # ONCE above and threaded in (detect_up/down_runs + context_frame would otherwise each recompute the
    # full feature set on this identical `c` — the per-bar narration hotspot). Bit-identical.
    # STREAMING: the caller may thread the run-detectors precomputed ONCE on the full session (they are
    # CAUSAL — detect(full)[i] == detect(prefix<=i)[last], verified — so a per-bar slice is bit-identical).
    # This kills the O(n^2) per-bar recompute (the narration hotspot). Falls back to per-slice detect.
    if runs is not None and len(runs[0]["in_run"]) == len(c) and len(runs[1]["in_run"]) == len(c):
        Sup, Sdn = runs
    else:
        _, _, Sup = detect_up_runs(c, symbol, interval, feat=feat)
        _, _, Sdn = detect_down_runs(c, symbol, interval, feat=feat)
    i = len(c) - 1                                    # the as_of bar
    ts = pd.to_datetime(c["ts"])
    o, h, l, cl = (c[k].to_numpy(float) for k in ["open", "high", "low", "close"])
    vol = c["volume"].to_numpy(float)
    atr = feat["atr"].to_numpy(float)
    ema20 = feat["ema20"].to_numpy(float); rsi = feat["rsi"].to_numpy(float)
    vwap = feat["vwap"].to_numpy(float); rvol = feat["rvol"].to_numpy(float)
    a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else np.nan
    px = cl[i]

    # ── DAY CONTEXT ──────────────────────────────────────────────────────────
    R.phase = session_phase(ts.iloc[i])
    R.bars_since_open = int(c["bars_since_open"].iloc[i])
    date = ts.dt.date.to_numpy()
    today = date[i]
    day_mask = date == today
    prior_mask = date < today
    if prior_mask.any() and np.isfinite(a):
        prev_day = date[prior_mask][-1]
        pm = date == prev_day
        pdh, pdl, pclose = h[pm].max(), l[pm].min(), cl[pm][-1]
        today_open = o[day_mask][0]
        R.gap_atr = float((today_open - pclose) / a)
        R.pos_vs_pdh = float((px - pdh) / a)
    else:
        pdh = pdl = np.nan
    # HTF 1h trend — threaded in by the caller (computed ONCE per session via htf_trend_series), else
    # compute here. Threading skips context_frame entirely (which refetches 120d of HTF + recomputes
    # SuperTrend per call) — the per-bar narration hotspot. Bit-identical (HTF is point-in-time).
    if htf_1h is not None:
        R.htf_1h = htf_1h
    else:
        try:
            from intelligence.story_context import context_frame
            ctx = context_frame(c, symbol, interval, feat=feat)
            if not ctx.empty:
                R.htf_1h = "up" if ctx["HTF_UP"].iloc[i] else ("down" if ctx["HTF_DOWN"].iloc[i] else "flat")
        except Exception:
            pass
    if np.isfinite(a) and np.isfinite(vwap[i]):
        R.pos_vs_vwap_atr = float((px - vwap[i]) / a)

    # ── DAY MEMORY: how today opened + the session's VWAP relationship ──
    di = np.where(day_mask)[0]
    if len(di) >= 3 and np.isfinite(a):
        d0 = di[0]
        gap = "gap-up" if R.gap_atr > 0.4 else ("gap-down" if R.gap_atr < -0.4 else "flat")
        # open drive: first ~6 bars travel ≥1.5 ATR directionally from the open
        ow = di[:min(6, len(di))]
        drive = (cl[ow[-1]] - o[d0]) / a
        if drive >= 1.5:      shape = "open-drive up"
        elif drive <= -1.5:   shape = "open-drive down"
        else:                 shape = "balanced/range open"
        R.open_type = f"{gap} {shape}".strip()
        # VWAP session relationship (today only, point-in-time)
        above = (cl[di] > vwap[di])
        frac = float(np.nanmean(above))
        if frac >= 0.9:      R.vwap_session = "above VWAP all session"
        elif frac <= 0.1:    R.vwap_session = "below VWAP all session"
        elif above[-1] and not above[0]:  R.vwap_session = "reclaimed VWAP intraday"
        elif (not above[-1]) and above[0]: R.vwap_session = "lost VWAP intraday"
        else:                R.vwap_session = "chopping around VWAP"

    # ── STRUCTURAL MEMORY (PRIOR sessions): read the DAILY bars, not the 5m tape across
    #    the gap. Labeled as prior so it's never glued into today's intraday story. ──
    prior_days = [d for d in dict.fromkeys(date.tolist()) if d < today]
    if len(prior_days) >= 3 and np.isfinite(a):
        gdf = pd.DataFrame({"d": date, "o": o, "h": h, "l": l, "c": cl})
        gg = gdf[gdf["d"].isin(prior_days)].groupby("d").agg(op=("o", "first"), hi=("h", "max"),
                                                             lo=("l", "min"), cl=("c", "last"))
        gg = gg.loc[[d for d in prior_days if d in gg.index]]
        today_open = o[np.where(day_mask)[0][0]]
        R.structure_prior = _structure_prior(gg, today_open, pdh, pdl, a)

    # ── THE MOVE (symmetric) — pick the active run; if BOTH active, the more recent ignition ──
    up_in, dn_in = Sup["in_run"], Sdn["in_run"]
    if up_in[i] and dn_in[i]:
        d = 1 if (i - int(Sup["bars_in"][i])) >= (i - int(Sdn["bars_in"][i])) else -1
    elif up_in[i]:
        d = 1
    elif dn_in[i]:
        d = -1
    else:
        d = 0
    extreme = Sup["run_high"] if d >= 0 else Sdn["run_low"]    # favourable price extreme of the run
    R.active = d != 0
    if R.active:
        S = Sup if d > 0 else Sdn
        bars_in = S["bars_in"]
        R.direction = "up" if d > 0 else "down"
        start = i - int(bars_in[i])
        R.ignition_ts = ts.iloc[start]
        R.bars_in_run = int(bars_in[i])
        R.run_extreme = float(extreme[i]); R.run_high = R.run_extreme
        if np.isfinite(a):
            origin = l[start] if d > 0 else h[start]
            R.ext_atr = float((extreme[i] - origin) * d / a)  # favourable travel (always ≥0)
        # ignition type (mirror for down)
        mfo = c["mins_from_open"].iloc[start]
        if d > 0:
            prior = pd.Series(h).shift(1).rolling(10).max().to_numpy()
            broke = np.isfinite(prior[start]) and h[start] > prior[start]
            R.ignition_type = "open-drive" if mfo < 35 else ("breakout of prior high" if broke else "momentum push")
        else:
            prior = pd.Series(l).shift(1).rolling(10).min().to_numpy()
            broke = np.isfinite(prior[start]) and l[start] < prior[start]
            R.ignition_type = "open-drive down" if mfo < 35 else ("breakdown of prior low" if broke else "momentum drop")

        # ── FUEL across the move (pushes = green for up / red for down) ──
        push_bars = [j for j in range(start, i + 1) if ((cl[j] > o[j]) if d > 0 else (cl[j] < o[j]))]
        if len(push_bars) >= 2:
            half = len(push_bars) // 2
            early = [rvol[j] for j in push_bars[:max(1, half)] if np.isfinite(rvol[j])]
            recent = [rvol[j] for j in push_bars[half:] if np.isfinite(rvol[j])]
            R.vol_early = float(np.nanmean(early)) if early else np.nan
            R.vol_recent = float(np.nanmean(recent)) if recent else np.nan
            if np.isfinite(R.vol_early) and np.isfinite(R.vol_recent):
                ratio = R.vol_recent / R.vol_early if R.vol_early > 0 else 1.0
                R.fuel = "expanding" if ratio > 1.15 else ("fading" if ratio < 0.85 else "steady")
        R.rsi = float(rsi[i]) if np.isfinite(rsi[i]) else np.nan
        if np.isfinite(a):
            R.stretch_atr = float((px - ema20[i]) * d / a)     # favourable stretch from EMA

        # ── PULLBACKS SO FAR (counter-moves against the run, cumulative) ──
        R.pullbacks = _pullbacks_in_run(start, i, h, l, cl, o, extreme, atr, ts, d)

        # ── STRUCTURAL MEMORY: what built the launch pad (SAME-SESSION bars before ignition) ──
        day_start = int(np.where(date == date[start])[0][0])
        R.structure_before = _structure_before(start, day_start, h, l, cl, o, vol, atr,
                                               feat["bb_width_pct"].to_numpy(float), d)

    else:
        R.direction = "none"

    # ── ZONES around current price (point-in-time levels) ──
    R.zones = _zones(px, a, h, l, cl, vwap[i], pdh, pdl, day_mask, ts.iloc[i])

    # ── THE RUNNING STATE READ ──
    R.state, extra = _state_read(R, i, h, l, cl, o, extreme, atr)
    R.notes.extend(extra)
    # ── SYNTHESIS: why this move exists + what threatens it now ──
    _synthesize(R)
    # ── UNDERSTAND: what phase the move is in + the tension behind it ──
    _phase_energy(R)
    return R


def _phase_energy(R: MoveRead) -> None:
    """The describe→understand layer. Where in its life is the move (EARLY/MIDDLE/LATE),
    and what's the tension behind the same surface facts — energy release vs exhaustion?
    A coiled-spring breakout (early, from compression, volume expanding) and a chased late
    breakout (6+ ATR, climax/fading, stretched) print the SAME 'breakout+volume' — phase is
    what tells them apart."""
    if not R.active:
        R.move_phase = "—"; R.energy = "no active move"; return
    d = 1 if R.direction == "up" else -1
    npull = len(R.pullbacks)
    stretch = R.stretch_atr if np.isfinite(R.stretch_atr) else 0.0     # favourable (signed by d)
    rsi = R.rsi if np.isfinite(R.rsi) else 50.0
    rsi_exhausted = rsi >= 75 if d > 0 else rsi <= 25                  # overbought up / oversold down
    has_coil = (any("compression" in s for s in R.structure_prior)
                or any(s.startswith("compression") or s.startswith("range narrowing")
                       or s.startswith("sideways") for s in R.structure_before))

    # ── life-stage ──
    late_signs = []
    if R.ext_atr >= 5:           late_signs.append(f"travelled {R.ext_atr:.1f} ATR")
    if npull >= 3:               late_signs.append(f"{npull} pullbacks deep")
    if stretch >= 3:             late_signs.append(f"{stretch:.1f} ATR {'above' if d>0 else 'below'} EMA20")
    if rsi_exhausted:            late_signs.append(f"RSI {rsi:.0f}")
    if R.fuel == "fading" and R.ext_atr >= 3: late_signs.append("volume fading after the push")
    early = R.ext_atr < 2.0 and R.bars_in_run <= 4 and npull <= 1
    if late_signs:
        R.move_phase = "LATE"
    elif early:
        R.move_phase = "EARLY"
    else:
        R.move_phase = "MIDDLE"

    # ── the tension ──
    reasons: list[str] = []
    if R.move_phase == "LATE":
        R.energy = "exhaustion risk"
        reasons = late_signs + (["chasing late into the move"] if R.ext_atr >= 5 else [])
    elif has_coil and R.move_phase in ("EARLY", "MIDDLE") and stretch < 3 and (rsi < 72 if d > 0 else rsi > 28):
        R.energy = "energy release"
        reasons.append("broke from a coil / compression — energy was stored")
        if R.fuel == "expanding":
            reasons.append("volume expanding into the break (earned, not abrupt)")
    elif R.fuel == "expanding" and R.move_phase in ("EARLY", "MIDDLE") and stretch < 3:
        R.energy = "building momentum"
        reasons.append("volume expanding, move still has room")
    elif R.move_phase == "MIDDLE":
        R.energy = "healthy continuation"
        reasons.append("orderly — pullbacks resolving, not yet extended")
    else:
        R.energy = "unproven"
        reasons.append("just ignited with no compression behind it — could go either way")
    R.energy_reasons = reasons


def _structure_before(start, day_start, h, l, cl, o, vol, atr, bbw, d=1, look=30) -> list[str]:
    """Name the launch-pad structure in the bars BEFORE ignition, WITHIN today's session
    (the earlier chapters). Observational, point-in-time — only same-day bars < start.
    A move that ignites early (open-drive) simply has no intraday structure before it —
    the gap/overnight IS the story, and we honestly return nothing here."""
    out: list[str] = []
    lo = max(day_start, start - look)        # never cross the overnight boundary
    if start - lo < 8:                        # too few same-session bars → no intraday structure
        return out
    a = atr[start - 1] if np.isfinite(atr[start - 1]) and atr[start - 1] > 0 else np.nan
    if not np.isfinite(a):
        return out
    pre_h, pre_l, pre_c, pre_o = h[lo:start], l[lo:start], cl[lo:start], o[lo:start]
    pre_bbw = bbw[lo:start]
    rng_atr = (np.nanmax(pre_h) - np.nanmin(pre_l)) / a
    nb = start - lo

    # compression / coil: the pre-window range is tight relative to its length
    if rng_atr <= max(2.5, 0.18 * nb):
        out.append(f"compression — coiled in a ~{rng_atr:.1f} ATR range for {nb} bars")
    elif np.isfinite(np.nanmean(pre_bbw)) and np.nanmean(pre_bbw[-8:]) < np.nanmedian(bbw[max(0, lo-60):lo] if lo > 60 else bbw[:lo]) * 0.8:
        out.append("range narrowing into the launch (BB squeeze)")

    if d > 0:
        # failed breakdown / reclaim: dipped under support then closed back above (up launch pad)
        sup = np.nanmin(pre_l[: max(3, nb // 2)])
        dipped = np.nanmin(pre_l[max(3, nb // 2):])
        if dipped < sup - 0.2 * a and pre_c[-1] > sup + 0.1 * a:
            out.append("failed breakdown — swept below support then reclaimed it")
        # support defense: ≥3 lows clustering at a level that held
        base = np.nanmin(pre_l)
        if int(np.sum(pre_l <= base + 0.3 * a)) >= 3:
            out.append(f"support defended — {int(np.sum(pre_l <= base + 0.3 * a))} taps of ~{base:.1f} held")
    else:
        # failed breakout / rejection: poked above resistance then closed back below (down launch pad)
        res = np.nanmax(pre_h[: max(3, nb // 2)])
        poked = np.nanmax(pre_h[max(3, nb // 2):])
        if poked > res + 0.2 * a and pre_c[-1] < res - 0.1 * a:
            out.append("failed breakout — poked above resistance then rejected")
        # resistance defense: ≥3 highs clustering at a level that capped price
        top = np.nanmax(pre_h)
        if int(np.sum(pre_h >= top - 0.3 * a)) >= 3:
            out.append(f"resistance defended — {int(np.sum(pre_h >= top - 0.3 * a))} taps of ~{top:.1f} capped")

    # accumulation: many bars, small net drift, decent volume (sideways absorption)
    drift = abs(pre_c[-1] - pre_c[0]) / a
    if nb >= 12 and drift <= 1.5 and rng_atr <= 3.5:
        out.append("sideways accumulation — absorbed without trending")

    return out[:3]


def _structure_prior(gg, today_open, pdh, pdl, atr) -> list[str]:
    """Prior-session / higher-TF structure from DAILY bars (completed days only).
    Always tagged as prior — a trader knows 'that was yesterday', never glued to today.

    Compression here means the trader's eye of a COIL: the last sessions STACK in a shared
    tight band — they overlap heavily, none of them is a wide/spike day, and the union of
    their ranges is barely bigger than a single day. A spike-then-drift is rejected."""
    out: list[str] = []
    dr = (gg["hi"] - gg["lo"])
    med = float(dr.tail(10).median()) if len(dr) else np.nan
    if not np.isfinite(med) or med <= 0:
        return out
    last3 = gg.tail(3)
    ops = last3["op"].to_numpy(float); his = last3["hi"].to_numpy(float)
    los = last3["lo"].to_numpy(float); cls = last3["cl"].to_numpy(float)
    drs = his - los
    bodies = cls - ops                               # each day's net open→close move
    union = float(his.max() - los.min())
    med3 = float(np.median(drs)) if len(drs) else np.nan
    shared = float(his.min() - los.max())            # price band common to ALL the days
    drift = float(abs(cls[-1] - cls[0]))             # net progress across the sessions
    n_days = int(len(last3))

    # ── YESTERDAY's character first — the freshest prior move, never miss it.
    #    A strong directional day (big body, closed toward the extreme) is the memory the
    #    user said was getting lost behind a coarse 'compression' label. ──
    yb, yr = float(bodies[-1]), float(drs[-1])
    y_dir_close = abs(yb) >= 0.45 * yr if yr > 0 else False
    if abs(yb) >= 0.7 * med and y_dir_close:
        where = "near the low" if yb < 0 else "near the high"
        out.append(f"yesterday: strong {'DOWN' if yb < 0 else 'UP'} day, closed {where} (prior)")

    # ── multi-day COIL: every day internally BALANCED (closed mid-range, a real range day —
    #    NOT a big directional move that merely flattened) + stacked sideways + contained. ──
    internally_balanced = bool(np.all(np.abs(bodies) <= 0.55 * np.maximum(drs, 1e-9)))
    no_spike = bool(np.all(drs <= 1.4 * med))
    contained = union <= 1.8 * med
    sideways = union > 0 and drift <= 0.45 * union
    overlap = med3 > 0 and shared >= 0.2 * med3
    if internally_balanced and no_spike and contained and sideways and overlap:
        out.append(f"{n_days}-day compression (prior) — balanced sessions stacked sideways "
                   f"(~{union/med:.1f}× a normal day, no net progress)")

    # where did today open relative to that prior structure?
    if np.isfinite(pdh) and np.isfinite(pdl) and pdl <= today_open <= pdh:
        out.append("opened inside yesterday's range (prior)")
    elif los.min() <= today_open <= his.max():
        out.append("opened inside the multi-day range (prior)")
    return out[:3]


def _synthesize(R: MoveRead) -> None:
    """Build the trader's two summaries: WHY THIS MOVE EXISTS + CURRENT THREATS (direction-aware)."""
    if not R.active:
        R.primary_driver = "no active move"
        return
    up = R.direction == "up"; d = 1 if up else -1
    src: list[str] = []
    if R.ignition_type.startswith("open-drive"):
        src.append("Open-drive ignition")
    if up and R.gap_atr > 0.4:
        src.append("Gap-up continuation")
    elif (not up) and R.gap_atr < -0.4:
        src.append("Gap-down continuation")
    if up and R.vwap_session == "above VWAP all session":
        src.append("Above VWAP entire session")
    elif up and R.vwap_session == "reclaimed VWAP intraday":
        src.append("Reclaimed VWAP")
    elif (not up) and R.vwap_session == "below VWAP all session":
        src.append("Below VWAP entire session")
    elif (not up) and R.vwap_session == "lost VWAP intraday":
        src.append("Lost VWAP")
    for s in R.structure_prior:
        if "compression" in s:
            src.append("Multi-day compression (prior)")
        elif "opened inside" in s:
            src.append("Opened inside prior range")
    for s in R.structure_before:
        if s.startswith("compression") or s.startswith("range narrowing"):
            src.append("Compression resolved " + ("upward" if up else "downward"))
        elif s.startswith("failed breakdown"):
            src.append("Failed breakdown / reclaim")
        elif s.startswith("failed breakout"):
            src.append("Failed breakout / rejection")
        elif s.startswith("support defended"):
            src.append("Support defended below")
        elif s.startswith("resistance defended"):
            src.append("Resistance defended above")
        elif s.startswith("sideways"):
            src.append("Absorption before the break")
    if R.ignition_type == "breakout of prior high" or (up and np.isfinite(R.pos_vs_pdh) and R.pos_vs_pdh >= 0):
        src.append("Broke above prior-day high")
    elif R.ignition_type == "breakdown of prior low":
        src.append("Broke below prior-day low")
    if R.fuel == "expanding":
        src.append("Volume expanded into the move")
    seen = set(); R.move_source = [x for x in src if not (x in seen or seen.add(x))]

    # primary driver — the dominant chapter
    brk = "Breakout" if up else "Breakdown"
    if any("compression" in s for s in R.structure_prior):
        R.primary_driver = f"{brk} from multi-day compression"
    elif any(s.startswith(("compression", "range narrowing", "sideways")) for s in R.structure_before):
        R.primary_driver = f"{brk} from accumulated energy"
    elif R.ignition_type.startswith("open-drive") and abs(R.gap_atr) > 0.4:
        R.primary_driver = ("Gap-up" if up else "Gap-down") + " open-drive continuation"
    elif R.ignition_type in ("breakout of prior high", "breakdown of prior low"):
        R.primary_driver = f"{brk} {'above prior-day high' if up else 'below prior-day low'}"
    elif any(s.startswith("failed breakout") for s in R.structure_before):
        R.primary_driver = "Reversal down off a failed breakout"
    elif any(s.startswith("failed breakdown") for s in R.structure_before):
        R.primary_driver = "Reversal up off a failed breakdown"
    else:
        R.primary_driver = "Momentum continuation"

    # threats — what's against it right now (the level it's walking INTO is on the move's side)
    th: list[str] = []
    side = "above" if up else "below"; word = "overhead" if up else "below"
    into = [z for z in R.zones if z.side == side and 0 <= abs(z.dist_atr) <= 1.2]
    for z in into[:2]:
        th.append(f"{z.name} {abs(z.dist_atr):.1f} ATR {word}")
    if R.fuel == "fading":
        th.append("volume fading vs early pushes")
    if len(R.pullbacks) >= 3:
        th.append(f"{len(R.pullbacks)} pullbacks already (move getting choppy)")
    if np.isfinite(R.rsi) and (R.rsi >= 75 if up else R.rsi <= 25):
        th.append(f"RSI stretched ({R.rsi:.0f})")
    if np.isfinite(R.stretch_atr) and R.stretch_atr >= 3:
        th.append(f"{R.stretch_atr:.1f} ATR extended {side} EMA20")
    if R.ext_atr >= 5:
        th.append(f"run already ~{R.ext_atr:.1f} ATR — mature")
    if R.pullbacks and R.pullbacks[-1].resolved == "live" and R.pullbacks[-1].depth_atr >= 1.5:
        th.append(f"live pullback {R.pullbacks[-1].depth_atr:.1f} ATR deep")
    R.threats = th


def _pullbacks_in_run(start, i, h, l, cl, o, extreme, atr, ts, d=1) -> list[Pullback]:
    """Walk the run from ignition to as_of, group COUNTER-moves into pullback EVENTS (red clusters
    for an up-run, green bounces for a down-run), record each one's depth and how it resolved
    (or 'live' if it's the current one). Direction-aware via d (+1 up, −1 down)."""
    out: list[Pullback] = []
    in_pb = False; pb_start = start; pb_ext = (np.inf if d > 0 else -np.inf); ref = extreme[start]
    for j in range(start + 1, i + 1):
        a = atr[j]
        if not np.isfinite(a) or a <= 0:
            continue
        new_extreme = (h[j] > ref) if d > 0 else (l[j] < ref)
        if new_extreme:
            if in_pb:                                  # the prior pullback RESUMED
                out[-1].resolved = "resumed"; in_pb = False
            ref = h[j] if d > 0 else l[j]
            continue
        is_counter = (cl[j] <= o[j]) if d > 0 else (cl[j] >= o[j])     # red vs up / green vs down
        broke_back = (l[j] < ref) if d > 0 else (h[j] > ref)
        trail = (h[j] < h[j - 1]) if d > 0 else (l[j] > l[j - 1])
        starting = (not new_extreme) and (is_counter or trail) and broke_back
        if starting and not in_pb:
            in_pb = True; pb_start = j; pb_ext = l[j] if d > 0 else h[j]
            depth = (ref - l[j]) / a if d > 0 else (h[j] - ref) / a
            out.append(Pullback(start_ts=ts.iloc[j], depth_atr=float(depth), bars=1, resolved="live"))
        elif in_pb:
            pb_ext = min(pb_ext, l[j]) if d > 0 else max(pb_ext, h[j])
            depth = (ref - pb_ext) / a if d > 0 else (pb_ext - ref) / a
            out[-1].depth_atr = float(depth)
            out[-1].bars = j - pb_start + 1
    # inside an ACTIVE run, every completed pullback resumed (else the run had died);
    # only the last one — if we're still in it at as_of — stays 'live'.
    for p in out[:-1]:
        if p.resolved == "live":
            p.resolved = "resumed"
    return out


def _zones(px, a, h, l, cl, vwap_now, pdh, pdl, day_mask, now_ts) -> list[Zone]:
    zones: list[Zone] = []

    def add(name, price):
        if price is None or not np.isfinite(price) or not np.isfinite(a) or a <= 0:
            return
        d = (price - px) / a
        zones.append(Zone(name=name, price=float(price),
                          dist_atr=float(d), side="above" if d >= 0 else "below"))

    add("prev-day high", pdh)
    add("prev-day low", pdl)
    add("VWAP", vwap_now)
    # opening range (first 30m of today)
    if day_mask.any():
        # first ~6 bars of the day on 5m ≈ 30m; approximate with bars_since_open<6 by slicing
        td_h = h[day_mask]; td_l = l[day_mask]
        if len(td_h) >= 3:
            add("opening-range high", float(np.max(td_h[:6])))
            add("opening-range low", float(np.min(td_l[:6])))
    # nearest round numbers
    step = _round_step(px)
    add("round %g" % (np.ceil(px / step) * step), float(np.ceil(px / step) * step))
    add("round %g" % (np.floor(px / step) * step), float(np.floor(px / step) * step))
    # keep only the closest few each side, sorted by absolute distance
    zones = [z for z in zones if abs(z.dist_atr) <= 6]
    zones.sort(key=lambda z: abs(z.dist_atr))
    return zones[:6]


def _state_read(R: MoveRead, i, h, l, cl, o, extreme, atr) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not R.active:
        return "DORMANT", ["no momentum run active right now"]
    a = atr[i]
    d = 1 if R.direction == "up" else -1
    # is the last bar making a new extreme (extending) or off it (pulling back)?
    making_extreme = np.isfinite(extreme[i]) and (h[i] >= extreme[i] - 1e-9 if d > 0 else l[i] <= extreme[i] + 1e-9)
    live_pb = R.pullbacks and R.pullbacks[-1].resolved == "live"
    weakening = []
    if R.fuel == "fading":
        weakening.append("volume fading vs early pushes")
    if np.isfinite(R.rsi) and (R.rsi >= 75 if d > 0 else R.rsi <= 25):
        weakening.append("RSI stretched (%.0f)" % R.rsi)
    if np.isfinite(R.stretch_atr) and R.stretch_atr >= 3:
        weakening.append("price %.1f ATR %s EMA20 (extended)" % (R.stretch_atr, "above" if d > 0 else "below"))
    if R.ext_atr >= 5:
        weakening.append("run already ~%.1f ATR — mature" % R.ext_atr)
    # nearest level the move is walking INTO (resistance above for up, support below for down)?
    side = "above" if d > 0 else "below"
    ahead = [z for z in R.zones if z.side == side and 0 <= abs(z.dist_atr) <= 1.0]
    if ahead:
        notes.append("walking into %s (%.1f ATR %s)" % (ahead[0].name, abs(ahead[0].dist_atr), side))

    if making_extreme and not live_pb:
        state = "EXTENDING"
        notes.append("printing new %s — move is in control" % ("highs" if d > 0 else "lows"))
    elif live_pb:
        depth = R.pullbacks[-1].depth_atr
        if depth >= 1.5 or (np.isfinite(R.stretch_atr) and cl[i] < o[i] and len(weakening) >= 2):
            state = "WEAKENING"
        else:
            state = "PULLING_BACK"
        notes.append("pullback #%d in progress, %.1f ATR deep so far" % (len(R.pullbacks), depth))
    else:
        state = "STALLING"
        notes.append("not making new highs, not yet pulling back — coiling")
    if weakening:
        notes.append("caution: " + "; ".join(weakening))
    return state, notes


# ─────────────────────────────────────────────────────────────────────────────
def narrate(R: MoveRead) -> str:
    """Speak the read as a trader's running mental summary (no probabilities — observation).
    Order = the chapters: DAY → STORY BEFORE → THE MOVE → WHY IT EXISTS → THREATS → STATE."""
    L: list[str] = []
    head = f"{R.symbol} {R.interval} @ {R.as_of:%Y-%m-%d %H:%M}  ·  {R.phase}, bar {R.bars_since_open} of the day"
    L.append(head); L.append("─" * len(head))

    # DAY MEMORY
    day = []
    if R.open_type:
        day.append(R.open_type + (f" ({R.gap_atr:+.1f} ATR)" if abs(R.gap_atr) > 0.4 else ""))
    if R.vwap_session:
        day.append(R.vwap_session)
    day.append(f"1h {R.htf_1h.upper()}")
    if np.isfinite(R.pos_vs_pdh):
        day.append(f"{abs(R.pos_vs_pdh):.1f} ATR {'above' if R.pos_vs_pdh>=0 else 'below'} prev-day high")
    L.append("DAY:     " + " · ".join(day))

    # STRUCTURAL MEMORY (the story before the move) — prior sessions kept separate from today
    if R.structure_prior:
        L.append("PRIOR:   " + "; ".join(R.structure_prior))
    if R.structure_before:
        L.append("TODAY:   " + "; ".join(R.structure_before))

    if not R.active:
        L.append("MOVE:    no momentum run active right now")
        L.append(f"STATE:   {R.state}")
        for nt in R.notes:
            L.append("         • " + nt)
        return "\n".join(L)

    # THE MOVE
    L.append(f"MOVE:    {R.direction}-run ignited {R.ignition_ts:%H:%M} ({R.ignition_type}); "
             f"{R.bars_in_run} bars in, travelled ~{R.ext_atr:.1f} ATR to {R.run_extreme:.1f}")

    # PHASE + ENERGY (the understanding: where in its life + the tension behind it)
    if R.move_phase:
        L.append(f"PHASE:   {R.move_phase} — {R.energy}")
        for r in R.energy_reasons:
            L.append(f"           · {r}")

    # WHY THIS MOVE EXISTS
    if R.move_source:
        L.append("WHY:     why this move exists —")
        for s in R.move_source:
            L.append(f"           ✓ {s}")
        L.append(f"         → primary driver: {R.primary_driver}")

    # FUEL
    fuel = R.fuel or "n/a"
    fl = f"FUEL:    volume {fuel}"
    if np.isfinite(R.vol_early) and np.isfinite(R.vol_recent):
        fl += f" (early rvol {R.vol_early:.1f} → recent {R.vol_recent:.1f})"
    if np.isfinite(R.rsi):
        fl += f"; RSI {R.rsi:.0f}"
    if np.isfinite(R.stretch_atr):
        fl += f"; {abs(R.stretch_atr):.1f} ATR {'above' if R.direction == 'up' else 'below'} EMA20"
    L.append(fl)
    if R.pullbacks:
        res = ", ".join(f"#{k+1} {p.depth_atr:.1f}ATR·{p.resolved}" for k, p in enumerate(R.pullbacks))
        L.append(f"PULLS:   {len(R.pullbacks)} so far — {res}")

    # ZONES
    if R.zones:
        L.append("ZONES:   " + ", ".join(f"{z.name} {z.dist_atr:+.1f}ATR" for z in R.zones))

    # CURRENT THREATS
    if R.threats:
        L.append("THREATS: what's against it now —")
        for t in R.threats:
            L.append(f"           • {t}")

    # STATE
    L.append(f"STATE:   {R.state}")
    for nt in R.notes:
        L.append("         • " + nt)
    return "\n".join(L)


def render_read(R: MoveRead, candles: pd.DataFrame, out_path, before: int = 50):
    """Draw the move's candles with annotations (ignition, run-high, pullbacks, zones,
    as_of) and place the trader-summary text panel beside it — the VISUAL verification
    the user asked for: read the words against the chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from pathlib import Path

    c = candles[pd.to_datetime(candles["ts"]) <= R.as_of].sort_values("ts").reset_index(drop=True)
    i = len(c) - 1
    start = i - R.bars_in_run if R.active else i
    # widen the window to the last ~3 sessions (today + up to 2 prior) so the multi-session
    # story is visible — overnight gaps become day-separator lines, not glued tape.
    dd_all = pd.to_datetime(c["ts"]).dt.date
    udays = list(dict.fromkeys(dd_all.tolist()))
    # show the 3 prior sessions used by structure_prior + today = 4, so the PRIOR
    # multi-day claims (compression / yesterday's range) are fully on-screen to verify.
    keep_days = udays[-4:] if len(udays) >= 4 else udays
    lo = int(np.where(dd_all == keep_days[0])[0][0])
    w = c.iloc[lo:i + 1].reset_index(drop=True)
    o = w["open"].to_numpy(float); h = w["high"].to_numpy(float)
    l = w["low"].to_numpy(float); cl = w["close"].to_numpy(float)
    ts = pd.to_datetime(w["ts"]); dd = ts.dt.date
    x = np.arange(len(w))

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.04)
    ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1]); axt.axis("off")

    for k in range(len(w)):
        up = cl[k] >= o[k]
        col = "#26a69a" if up else "#ef5350"
        ax.plot([k, k], [l[k], h[k]], color=col, linewidth=0.8, zorder=2)
        ax.add_patch(Rectangle((k - 0.3, min(o[k], cl[k])), 0.6, abs(cl[k] - o[k]) + 1e-9,
                               facecolor=col, edgecolor=col, zorder=3))
    # day separators — overnight gaps become labeled dividers, not glued tape
    ymax = h.max()
    for k in range(1, len(w)):
        if dd.iloc[k] != dd.iloc[k - 1]:
            ax.axvline(k - 0.5, color="#90a4ae", linestyle="-", linewidth=0.8, alpha=0.5)
            ax.text(k + 0.5, ymax, f" {dd.iloc[k]:%a %m-%d}", fontsize=7, color="#455a64",
                    va="top", ha="left", alpha=0.8)
    ax.text(0.5, ymax, f" {dd.iloc[0]:%a %m-%d}", fontsize=7, color="#455a64",
            va="top", ha="left", alpha=0.8)
    # ignition + as_of
    ig = start - lo
    if R.active and 0 <= ig < len(w):
        ax.axvline(ig, color="#2e7d32", linestyle="-", linewidth=1.4, alpha=0.7, label="ignition")
        ax.axhline(R.run_high, color="#1565c0", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(len(w) - 1, color="#d62728", linestyle="-", linewidth=1.4, alpha=0.8, label="now (as_of)")
    # pullback start markers
    for p in R.pullbacks:
        j = (pd.to_datetime(w["ts"]) - p.start_ts).abs().idxmin()
        ax.scatter([j], [h[j] + 0.15 * (h.max() - l.min()) / 10], marker="v",
                   color="#fb8c00", s=40, zorder=5)
    # zones as horizontal lines
    for z in R.zones:
        ax.axhline(z.price, color="#9e9e9e", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.text(len(w) - 0.5, z.price, f" {z.name}", fontsize=7, color="#616161", va="center")
    ax.set_xlim(-1, len(w) + 6)
    step = max(1, len(w) // 8)
    ax.set_xticks(x[::step]); ax.set_xticklabels([t.strftime("%m-%d %H:%M") for t in ts[::step]],
                                                 rotation=30, fontsize=7, ha="right")
    ax.set_title(f"{R.symbol} {R.interval} — move read @ {R.as_of:%Y-%m-%d %H:%M}", fontsize=10)
    ax.legend(loc="upper left", fontsize=7); ax.grid(alpha=0.15)

    axt.text(0.0, 1.0, narrate(R), fontsize=8.5, family="monospace", va="top", ha="left",
             transform=axt.transAxes)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=95, bbox_inches="tight"); plt.close(fig)
    return out_path
