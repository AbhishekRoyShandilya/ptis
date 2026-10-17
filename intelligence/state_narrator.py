"""
intelligence.state_narrator — STATE RECOGNITION over a replayed session (CURRENT PHASE).

The milestone: watch a session bar-by-bar and narrate the EVOLVING STATE the way the trained eye
would — BALANCE → TRANSITION → ALIVE → EXHAUSTED (+ honest sub-states) — each step giving the
state, WHY, and WHAT CHANGED. NO entries, no exits, no prediction. The edge is COGNITIVE
RECOGNITION ("this is the kind of move/state I have studied"), never a price/probability forecast.

This is the thin REUSE-heavy orchestrator the plan calls for: it walks the session point-in-time
and reads the existing organs each step —
    balance_state.read_balance  → BALANCE / TRANSITION / TREND (band + acceptance)   [the STATE fact]
    move_read.read_move         → move id, phase, energy, pullbacks, threats, running state
    marketdata.features         → rvol (participation) + bb_width (compression/tightening)
— then maps their FACTS into the 4-state vocabulary as a CANDIDATE read. The candidate is what a
later step hands to the LLM reasoner (at state-change points) to OWN the verdict; here we first
prove the cheap structured track matches the user's eye (no LLM cost).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STATE_NARRATOR_VERSION = "0.1"

# PERF: precompute the prevailing-structure (lifecycle) ONCE over the full window and index per bar, instead of
# recomputing lifecycle_frame(prefix) every ~6 bars (the O(n^2) live-narration hotspot — ~5.5s of a ~6s narrate).
# Bit-identical for SESSION bars: lifecycle_frame is causal (row i == lifecycle_frame(prefix<=i).iloc[-1], verified),
# and _refresh_structure is only ever called on session bars (always past the lifecycle warm-up). Kill-switch: set
# PTIS_NARRATE_LFFULL=0 to fall back to the exact per-bar recompute.
_USE_LF_FULL = os.environ.get("PTIS_NARRATE_LFFULL", "1") == "1"

# the 4-state vocabulary (the user's words)
BALANCE, TRANSITION, ALIVE, EXHAUSTED = "BALANCE", "TRANSITION", "ALIVE", "EXHAUSTED"


@dataclass
class StateStep:
    ts: pd.Timestamp
    state: str
    substate: str = ""
    why: list[str] = field(default_factory=list)
    changed: str = ""              # what changed vs the previous step (the transition note)
    birth: str = ""                # birth/rebirth/continuation event fired at/around this step (if any)
    price: float = float("nan")
    atr: float = float("nan")
    tags: list[str] = field(default_factory=list)   # recognition signature (episodic LIBRARY) — set at a birth
    move_ext: float = float("nan")                   # extension-from-birth (ATR) while a move is active
    memory: dict = field(default_factory=dict)   # scoped working-memory snapshot (SESSION/MOVE/BAR) at this bar
    # ── RICH numeric state (populated only when narrate_session(rich=True); harvest uses it) ──────────
    # The eye COMPUTES all of this and then throws it away: `day_story._COLS` kept only the prose, so every
    # downstream study (move_funnel WHY_PHRASES, expansion_phase1's 32 features) was substring-matching
    # English text to recover facts that existed as floats one layer up. These carry the floats through.
    room_atr: float = float("nan")     # distance to the next opposing structural level, in ATR (at a birth)
    n_near: int = 0                    # count of structural levels within ~2 ATR beyond the break
    energy: str = ""                   # lifecycle: EXPANDING / ...
    fuel: str = ""
    loc: str = ""                      # NEAR_SUP / MID / NEAR_RES
    leg_ext_atr: float = float("nan")  # current leg extension (ATR)
    leg_bars: float = float("nan")
    in_run: float = float("nan")
    rsi: float = float("nan")
    d_res_atr: float = float("nan")    # numeric room up / down — the floats behind the coarse ROOM_* tag
    d_sup_atr: float = float("nan")
    prev_high: float = float("nan")    # prior-day levels (multi-day context)
    prev_low: float = float("nan")
    daily_ctx: str = ""                # prior-day / rolling-vs-trending read — SKIPPED entirely in lite before
    vwap_dist: float = float("nan")    # (close - vwap) / ATR
    rvol_now: float = float("nan")
    bbw_now: float = float("nan")      # Bollinger width pct (compression)


def _overhead_facts(c: pd.DataFrame, feat: pd.DataFrame, i: int, price: float, a: float,
                    direction: str) -> tuple[float, list[tuple[str, float]]]:
    """Surface the structure a fresh break is running into, as FACTS (the reasoner judges what it
    MEANS — hurdle vs clear air is NOT decided here). Returns (nearest_room_atr, near_levels) where
    near_levels = the structural opposing levels within ~2 ATR beyond the break, nearest-first, as
    (name, dist_atr). Only STRUCTURAL levels (prior-day H/L, opening range, multi-day swings) — round
    numbers and VWAP are everywhere and aren't a wall. Point-in-time: pass slices ending at i."""
    from intelligence.eye_levels import _gather_levels
    try:
        levels = _gather_levels(c, feat, i, price, a)
    except Exception:
        return 99.0, []
    structural = [z for z in levels if not (z.name.startswith("round") or z.name == "VWAP")]
    opp = [z for z in structural if (z.side == "above") == (direction == "up")]
    if not opp:
        return 99.0, []
    opp.sort(key=lambda z: abs(z.dist_atr))
    room = abs(opp[0].dist_atr)
    near = [(z.name, round(abs(z.dist_atr), 1)) for z in opp if abs(z.dist_atr) <= 2.0]
    return room, near


def _structure_now(c: pd.DataFrame, symbol: str, interval: str) -> tuple[str, str]:
    """The eye's prevailing-structure read: (htf trend, intraday swing structure) from lifecycle —
    e.g. ("bull", "HH_HL"). SESSION-scoped memory; refreshed slowly (it evolves over the session)."""
    try:
        from intelligence.lifecycle import lifecycle_frame
        lf = lifecycle_frame(c, symbol, interval)
        if lf.empty:
            return "", ""
        r = lf.iloc[-1]
        return str(r.get("htf", "")), str(r.get("structure", ""))
    except Exception:
        return "", ""


def _move_sequence(h, l, cl, vwap, atr, b: int, i: int, direction: str, ref: float, atr0: float) -> str:
    """The move's BEHAVIORAL STORY since birth (bar b → now i) as ONE structured fact for the reasoner —
    the sequence the mechanism reads off, not the birth snapshot. Captures: the extension ARC
    (advancing / stalling / rolling-over = bars since a new extreme), how the broken level has been
    RETESTED (acceptance vs being defended), the VWAP relationship held over the move, and the deepest
    pullback against it. Point-in-time (slice b..i only)."""
    if not (atr0 and atr0 > 0) or ref is None or i - b < 2:
        return ""
    down = direction == "down"
    H = h[b:i + 1]; L = l[b:i + 1]; C = cl[b:i + 1]; V = vwap[b:i + 1]
    n = len(C)
    extreme = float(L.min()) if down else float(H.max())
    ext_idx = int(np.argmin(L)) if down else int(np.argmax(H))
    peak = abs(ref - extreme) / atr0
    cur = abs(ref - C[-1]) / atr0
    since = (n - 1) - ext_idx
    arc = ("advancing (new ground)" if since <= 2 else "stalling — no new ground for a bit"
           if since <= 5 else "rolling over — no new ground")
    parts = [f"peak +{peak:.1f} ATR, now +{cur:.1f} ({arc})"]
    # VWAP relationship over the move (favorable side for the direction)
    if np.isfinite(V[-1]):
        fav = (C < V) if down else (C > V)
        if bool(fav[-1]):
            k = 0
            for x in fav[::-1]:
                if x:
                    k += 1
                else:
                    break
            parts.append(f"held the move's side of VWAP {k} bars")
        else:
            parts.append("price has RECLAIMED VWAP (counter-signal)")
    # retests of the broken level (ref): for a down move ref is broken support→now resistance, etc.
    side = H if down else L
    near = np.abs(side - ref) < 0.35 * atr0
    retests = 0; inb = False
    for x in near[1:]:
        if x and not inb:
            retests += 1; inb = True
        elif not x:
            inb = False
    if retests:
        parts.append(f"broken level retested {retests}×")
    # deepest pullback against the move (retrace from the running extreme)
    run_ext = np.minimum.accumulate(L) if down else np.maximum.accumulate(H)
    retr = (C - run_ext) if down else (run_ext - C)
    maxpb = float(np.max(retr)) / atr0 if n else 0.0
    if maxpb > 0.4:
        parts.append(f"deepest pullback against it {maxpb:.1f} ATR")
    return "MOVE STORY: " + "; ".join(parts)


def _trend_relation(birth_dir: str, htf: str, struct: str) -> str:
    """Is a birth WITH or AGAINST the prevailing structure? The common-sense weight: a down break inside
    an intact HH_HL up-structure is a counter-trend pullback, not a regime birth. The intraday swing
    structure (HH_HL/LH_LL) is primary; the slower 1h trend is context (it can lag a fresh turn)."""
    sdir = "up" if struct == "HH_HL" else ("down" if struct == "LH_LL" else "")
    if not sdir:
        if htf not in ("bull", "bear"):
            return ""
        rel = "WITH" if ((birth_dir == "up") == (htf == "bull")) else "COUNTER-TREND —"
        return f"{rel} the 1h {htf} trend"
    rel = "WITH" if birth_dir == sdir else "COUNTER-TREND — against"
    base = f"{rel} the intraday {struct} ({sdir}) structure"
    if htf in ("bull", "bear"):
        htf_dir = "up" if htf == "bull" else "down"
        base += f" · 1h {htf} {'agrees' if htf_dir == sdir else '(higher TF lagging)'}"
    return base


def _candidate_state(bs, mv, move_active: bool, move_ext: float, birth_kind: str, birth_dir: str,
                     rvol_rising: bool, compressing: bool) -> tuple[str, str, list[str]]:
    """Map the organs' FACTS → a CANDIDATE state, with the BIRTH recognizer as the PRIMARY arbiter of
    whether we are in a move at all (it is sharper + less noisy than the balance-acceptance heuristic,
    which flip-flopped). A move begins on a BIRTH/REBIRTH (strong break + held), stays alive while the
    tracker holds it (structural release ends it), and move_read only refines ALIVE-vs-EXHAUSTED.
    balance_state details the BALANCE sub-states when we are NOT in a move. NOT the final verdict — the
    reasoner owns that at change points."""
    why: list[str] = []
    bpos = getattr(bs, "pos", "") or ""
    bnote = getattr(bs, "note", "") or ""
    energy = getattr(mv, "energy", "") or ""
    mstate = getattr(mv, "state", "") or ""
    ext = float(move_ext)            # extension-from-BIRTH (monotonic), not move_read's local leg

    # ---- BIRTH MOMENT → TRANSITION (the move is being born: a strong break that left balance). ----
    if birth_kind in ("BIRTH", "REBIRTH"):
        verb = "born" if birth_kind == "BIRTH" else "re-born"
        why.append(f"{birth_dir} move {verb} — strong break left balance (the acceptance)")
        return TRANSITION, f"birth — move {verb} ({birth_dir})", why

    # ---- IN A MOVE (the tracker holds it) → ALIVE vs EXHAUSTED. EXHAUSTED needs WEAKENING evidence,
    #      NOT just distance: a move far-extended but still firing into clear air is mature-but-ALIVE
    #      (whether deep extension = done is the reasoner's call). Extension is an ALIVE texture. ----
    if move_active:
        # exhaustion needs a move that has actually RUN (≥3 ATR from birth) AND is weakening — a fresh
        # move can't be "exhausted" at 1.9 ATR. Whether a mature move's weakening is true exhaustion vs
        # a healthy pullback (clear air vs into resistance) is the REASONER's call, not this flag.
        weakening = mstate in ("WEAKENING", "STALLING")
        if weakening and ext >= 3.0:
            why.append(f"move {ext:.1f} ATR from birth · weakening — losing control")
            return EXHAUSTED, "exhaustion appearing", why
        if ext >= 4.0:
            why.append(f"move {ext:.1f} ATR from birth — mature/extended" + (f" / {energy}" if energy else ""))
            return ALIVE, "move alive — mature/extended", why
        why.append(f"directional control — move {ext:.1f} ATR from birth" + (f" / {energy}" if energy else ""))
        sub = "move alive — orderly pullback" if mstate == "PULLING_BACK" else "move alive — still healthy"
        return ALIVE, sub, why

    # ---- BALANCE: no active move — inside the cage / unformed (poking = a test, not a move). ----
    sub = "balance"
    if bpos.startswith("poking"):
        sub = "testing the edge"; why.append(bnote or "poking the edge — not accepted (still balance)")
    elif compressing:
        sub = "balance tightening"; why.append("range narrowing / volatility contracting")
    if rvol_rising:
        sub = (sub + " · participation increasing") if sub != "balance" else "participation increasing"
        why.append("volume picking up — attention rising")
    if not why:
        why.append(bnote or "rotating inside the range — no transition")
    return BALANCE, sub, why


def _fill_rich(st, i, lf_full, room, n_near, prev_hi, prev_lo, daily_ctx, cl, vwap, atr, rvol, bbw) -> None:
    """Carry the NUMERIC state the eye already computed onto the StateStep (rich harvest only).

    Everything here was previously computed and then discarded: `lifecycle_frame` produces leg_ext_atr /
    leg_bars / rsi / d_res_atr / d_sup_atr / energy / fuel / loc per bar and only `structure` survived (as a
    tag); `_overhead_facts` produces a numeric room + level list and only a coarse ROOM_* tag survived;
    prior-day levels and `daily_context` never reached disk at all. Downstream studies then substring-matched
    English prose to approximate these. This is the unlock.
    """
    import numpy as _np
    st.room_atr = float(room) if room == room else float("nan")
    st.n_near = int(n_near or 0)
    st.prev_high, st.prev_low = prev_hi, prev_lo
    st.daily_ctx = daily_ctx or ""
    try:
        a = float(atr[i])
        if _np.isfinite(vwap[i]) and _np.isfinite(a) and a > 0:
            st.vwap_dist = round(float((cl[i] - vwap[i]) / a), 3)
    except Exception:
        pass
    for src, dst in ((rvol, "rvol_now"), (bbw, "bbw_now")):
        try:
            v = float(src[i])
            if v == v:
                setattr(st, dst, round(v, 3))
        except Exception:
            pass
    if lf_full is None or not (0 <= i < len(lf_full)):
        return
    r = lf_full.iloc[i]
    for col, attr, num in (("energy", "energy", False), ("fuel", "fuel", False), ("loc", "loc", False),
                           ("leg_ext_atr", "leg_ext_atr", True), ("leg_bars", "leg_bars", True),
                           ("in_run", "in_run", True), ("rsi", "rsi", True),
                           ("d_res_atr", "d_res_atr", True), ("d_sup_atr", "d_sup_atr", True)):
        if col not in lf_full.columns:
            continue
        v = r.get(col)
        if num:
            try:
                fv = float(v)
                if fv == fv:
                    setattr(st, attr, round(fv, 3))
            except Exception:
                pass
        else:
            setattr(st, attr, str(v) if v is not None else "")


def narrate_session(symbol: str, interval: str, date, step: int = 1,
                    candles: pd.DataFrame | None = None, with_recall: bool = True,
                    lite: bool = False, skip_structure: bool = False,
                    rich: bool = False) -> list[StateStep]:
    """Walk `date`'s session bar-by-bar (point-in-time) and return the evolving-state timeline.
    `with_recall` loads the episodic LIBRARY and recalls past analogues at each birth (set False when
    HARVESTING, so a session isn't recalled against a half-built library / against itself).
    `lite` = HARVEST fast-path: emit only what an EPISODE needs (birth + tags + extension), skipping the
    per-bar BALANCE/MOVE state organs + bias/memory rendering. Quality-NEUTRAL for harvesting — tags come
    from the birth tracker/lifecycle/room/effort and ext from the tracker, NONE of which depend on
    read_balance/read_move (those only flavor the substate TEXT, which episodes never read).
    `skip_structure` = CASCADE-live fast-path (compose with lite): skip the prevailing-structure organ
    (lifecycle_frame, the single biggest precompute) entirely. It only feeds the birth's trend SUFFIX and
    the episodic TAGS — the cascade reads neither (it extracts kind/dir/level/atr), so births are identical
    (verified byte-for-byte on the extracted signal). NEVER set this for harvest/DayStory — it changes tags."""
    from marketdata import features as F
    from intelligence.balance_state import read_balance
    from intelligence.move_read import read_move
    from intelligence.birth import BirthTracker

    day = pd.Timestamp(date).normalize()
    if candles is None:
        from marketdata.candle_store import get_candle_store
        live = day.date() == pd.Timestamp.now().date()   # today → pull the forming bars; historical → no-op
        candles = get_candle_store().get(symbol, interval, day - pd.Timedelta(days=10),
                                         day + pd.Timedelta(days=1), live=live)
    c = candles.sort_values("ts").reset_index(drop=True)
    ts = pd.to_datetime(c["ts"])
    feat = F.compute_features(c, symbol, interval)   # eye computes its OWN window (window-warmed, the
    # validated behavior). NOT the shared store: a floor-warmed store differs at warm-up-front bars and
    # shifts level-detection (see project_brain/08 — warm-up semantics). Unifying needs re-validation.
    try:                                          # HTF trend: computed ONCE for the session, threaded per bar
        from intelligence.story_context import htf_trend_series
        htf_series = htf_trend_series(c, symbol, interval)
    except Exception:
        htf_series = None
    # STREAMING (O(n^2)→O(n)): the run-detectors are the per-bar narration hotspot AND are CAUSAL
    # (detect(full)[i] == detect(prefix<=i)[last], verified) → compute ONCE on the full session here and
    # thread a per-bar slice into read_move. Bit-identical (same pattern as `feat`). None → read_move falls
    # back to its own per-slice detect (unchanged behavior).
    Sup_full = Sdn_full = csf_full = None
    if not lite:
        try:
            from intelligence.session_clock import add_session_features as _asf
            from intelligence.momentum_run import detect_up_runs as _dur, detect_down_runs as _ddr
            csf_full = _asf(c)                          # session-featured ONCE (skips per-bar re-add)
            _, _, Sup_full = _dur(csf_full, symbol, interval, feat=feat)
            _, _, Sdn_full = _ddr(csf_full, symbol, interval, feat=feat)
        except Exception:
            Sup_full = Sdn_full = csf_full = None
    # STREAMING (O(n^2)→O(n)): prevailing-structure via lifecycle_frame — compute ONCE on the full window and
    # index per bar (same causal pattern as the run detectors above). None → _refresh_structure recomputes per-bar.
    lf_full = None
    if _USE_LF_FULL and not skip_structure:
        try:
            from intelligence.lifecycle import lifecycle_frame as _lff
            lf_full = _lff(c, symbol, interval)
            if lf_full is None or len(lf_full) != len(c):   # length must line up for positional indexing
                lf_full = None
        except Exception:
            lf_full = None
    rvol = feat["rvol"].to_numpy(float) if "rvol" in feat else np.full(len(c), np.nan)
    bbw = feat["bb_width_pct"].to_numpy(float) if "bb_width_pct" in feat else np.full(len(c), np.nan)
    atr = feat["atr"].to_numpy(float)
    vwap = feat["vwap"].to_numpy(float) if "vwap" in feat else np.full(len(c), np.nan)
    h = c["high"].to_numpy(float); l = c["low"].to_numpy(float)
    o = c["open"].to_numpy(float); cl = c["close"].to_numpy(float); vol = c["volume"].to_numpy(float)
    sess = np.where(ts.dt.date.to_numpy() == day.date())[0]
    if len(sess) == 0:
        return []

    out: list[StateStep] = []
    step = max(1, step)
    tracker = BirthTracker()           # fed EVERY bar (release/state need every bar, not just steps)
    from intelligence.working_memory import WorkingMemory, BAR, MOVE, SESSION, BIAS, LIBRARY
    from intelligence.directional_bias import read_bias, daily_context, opening_context
    wm = WorkingMemory()
    library = []                       # episodic LIBRARY (cross-session analogues), minus this session
    if with_recall:
        try:
            from intelligence.episodic import EpisodeStore
            _d = str(pd.Timestamp(date).date())
            library = [e for e in EpisodeStore().load() if not (e.symbol == symbol and e.date == _d)]
        except Exception:
            library = []
    daily_ctx = ""                    # daily topping/bottoming context (set below)
    # LITE (harvest/cascade) never reads daily_ctx (only read_bias below, which lite skips) — so skip its 1d fetch.
    # That fetch is a broker round-trip PER narration; at live scale it floods Angel's historical-data rate limit.
    # `rich` (harvest, offline, lake-only) DOES want it: skipping it is why multi-day context has never once
    # been present in the harvested dataset — the single biggest hole in every expansion study to date.
    if (not lite) or rich:
        try:
            from marketdata.candle_store import get_candle_store as _gcs
            _cd = _gcs().get(symbol, "1d", day - pd.Timedelta(days=30), day + pd.Timedelta(days=1))
            daily_ctx = daily_context(_cd, day)
        except Exception:
            pass
    open_fact = ""                     # the OPEN-context string (for the episodic recognition signature)
    pending_birth = ""                 # annotation text since the last emitted step
    pending_kind = ""                  # BIRTH/REBIRTH/CONTINUATION since the last step (drives state)
    pending_dir = ""
    pending_tags: list[str] = []       # recognition signature for the pending birth (episodic LIBRARY)
    pending_room = float("nan")        # rich: the NUMERIC room the break is running into (ATR)
    pending_near = 0                   # rich: how many structural levels sit within ~2 ATR
    cur_htf = cur_struct = ""          # prevailing structure (SESSION-scoped), refreshed slowly
    # SESSION EVENTS — the consequential things that happened today; they STAY pinned all session and
    # ride into every later reasoning call (a failed breakdown at 09:20 is context for the 11:05 birth).
    ev_n = 0
    prev_active = False
    prev_struct = ""
    prev_htf = ""
    move_seed = None                   # (dir, level, born_ts) of the active move, for event logging
    move_peak_ext = 0.0
    major_logged = False
    move_born_counter = False          # was this move born AGAINST the prevailing trend? (for evolution)

    def _log_event(when, text):
        nonlocal ev_n
        ev_n += 1
        wm.set(f"sev{ev_n}", f"{when:%H:%M} {text}", SESSION)

    # SESSION memory: prior-day high/low (static all session)
    date_arr = ts.dt.date.to_numpy()
    days = list(dict.fromkeys(date_arr.tolist()))
    di = days.index(day.date())
    prev_hi = prev_lo = float("nan")          # prior-day levels as FLOATS (rich); the prose form goes to wm
    if di > 0:
        pm = date_arr == days[di - 1]
        if pm.any():
            prev_hi, prev_lo = float(h[pm].max()), float(l[pm].min())
            wm.set("prev_day", f"prior-day high {h[pm].max():.2f} / low {l[pm].min():.2f}", SESSION)
            # OPEN context — the overnight handoff (gap + how yesterday ENDED), observed not ruled
            oc = opening_context(o, h, l, cl, atr, sess, np.where(pm)[0])
            if oc:
                wm.set("open", "OPEN: " + oc, SESSION)
                open_fact = oc

    def _refresh_structure(i):
        nonlocal cur_htf, cur_struct
        if skip_structure:            # CASCADE fast-path: structure only flavors birth-suffix/tags the cascade ignores;
            return                    # leave cur_htf/cur_struct = "" and skip BOTH lifecycle_frame and _structure_now
        if lf_full is not None and 0 <= i < len(lf_full):     # precomputed once — index row i (causal, bit-identical)
            r = lf_full.iloc[i]
            cur_htf, cur_struct = str(r.get("htf", "")), str(r.get("structure", ""))
        else:
            cur_htf, cur_struct = _structure_now(c.iloc[: i + 1], symbol, interval)
        parts = [p for p in (f"1h {cur_htf}" if cur_htf else "", cur_struct) if p]
        if parts:
            wm.set("structure", "prevailing: " + ", ".join(parts), SESSION)

    def _step(k):     # one bar of the walk — the per-bar unit the resumable narrator will call (append_bar)
        nonlocal pending_birth, pending_kind, pending_dir, pending_tags, pending_room, pending_near
        nonlocal prev_active, prev_struct, prev_htf, move_seed, move_peak_ext, major_logged, move_born_counter
        i = int(sess[k])
        ev = tracker.update(i, ts.iloc[i], h, l, o, cl, vol, atr, rvol)
        if k % 6 == 0:                  # SESSION structure evolves slowly — refresh periodically
            _refresh_structure(i)
        # SESSION EVENT — trend flip: the day's regime changed (swing structure OR the 1h trend)
        if cur_struct in ("HH_HL", "LH_LL"):
            if prev_struct in ("HH_HL", "LH_LL") and cur_struct != prev_struct:
                _log_event(ts.iloc[i], f"TREND FLIP: structure {prev_struct} → {cur_struct}")
            prev_struct = cur_struct
        if cur_htf in ("bull", "bear"):
            if prev_htf in ("bull", "bear") and cur_htf != prev_htf:
                _log_event(ts.iloc[i], f"TREND FLIP: 1h regime {prev_htf} → {cur_htf} (a move flipped the trend)")
            prev_htf = cur_htf
        # SESSION EVENT — a move just RELEASED: failed break (small) vs a real move that ran its course
        if prev_active and not tracker.move_active and move_seed is not None:
            d0, lvl0, t0, _b0 = move_seed
            if move_peak_ext < 3.0:
                kind = "FAILED BREAKDOWN" if d0 == "down" else "FAILED BREAKOUT"
                _log_event(t0, f"{kind}: broke {lvl0} ({d0}) then reclaimed "
                               f"(only {move_peak_ext:.1f} ATR follow-through)")
            move_seed = None
        prev_active = tracker.move_active
        # MOVE memory (and the move's recalled analogues) expire when the move is released
        if not tracker.move_active:
            wm.drop_scope(MOVE)
            wm.drop_scope(LIBRARY)
        if ev is not None:
            room, near = _overhead_facts(c.iloc[: i + 1], feat.iloc[: i + 1], i,
                                         float(cl[i]), float(atr[i]), ev.direction)
            pending_room, pending_near = float(room), len(near)   # keep the FLOATS, not just the prose
            where = "overhead" if ev.direction == "up" else "underfoot"
            if near:
                lv = ", ".join(f"{nm} +{d}" for nm, d in near[:3])
                ahead = f"{where}: {len(near)} structural within 2 ATR ({lv})"
            elif room >= 10:
                ahead = f"{where} clear — open air"
            else:
                ahead = f"{where} clear (nearest structural +{room:.1f} ATR)"
            _refresh_structure(i)       # accurate structure at the moment of the break
            trend = _trend_relation(ev.direction, cur_htf, cur_struct)
            # a break in the opening window (~first 30 min) is at the OPEN — a FACT, not a verdict: gap +
            # opening-drive dynamics dominate, so the reasoner reads it against the OPEN context (SESSION)
            at_open = " · at the OPEN (opening drive — read against the OPEN context)" if k <= 5 else ""
            pending_birth = f"{ev.kind} {ev.direction} broke {ev.level} · {ahead}" + (f" · {trend}" if trend else "") + at_open
            pending_kind = ev.kind; pending_dir = ev.direction
            if ev.kind in ("BIRTH", "REBIRTH"):       # the recognition SIGNATURE for the episodic library
                from intelligence.episodic import signature_from_birth
                vwap_side = ("above" if cl[i] >= vwap[i] else "below") if (np.isfinite(vwap[i]) and np.isfinite(atr[i]) and atr[i] > 0) else ""
                # effort = the break's conviction: volume surge (stocks) OR range-expansion (indices/all)
                rng_exp = (h[i] - l[i]) / atr[i] if atr[i] > 0 else 0.0
                effort = ("strong" if ((np.isfinite(rvol[i]) and rvol[i] >= 1.5) or rng_exp >= 1.5)
                          else "weak" if (np.isfinite(rvol[i]) and rvol[i] < 1.0 and rng_exp < 1.0) else "")
                pending_tags = signature_from_birth(ev.direction, trend, room, cur_struct,
                                                    cur_htf, open_fact, k <= 5, vwap_side, f"{ts.iloc[i]:%H:%M}", effort)
                if library:                           # recall past situations this birth RHYMES with
                    from intelligence.episodic import recall, render_recall
                    rec = render_recall(recall(library, ev.direction, pending_tags))
                    if rec:
                        wm.set("analogues", rec, LIBRARY)
            if ev.kind in ("BIRTH", "REBIRTH"):     # a fresh move (re)sets MOVE-scoped memory
                wm.drop_scope(MOVE)
                wm.set("move_dir", f"{ev.direction} move, born {ev.kind.lower()} at {ev.level}", MOVE)
                if trend:
                    wm.set("move_trend", trend, MOVE)
                wm.set("move_room", f"born into: {ahead}", MOVE)
                move_seed = (ev.direction, ev.level, ts.iloc[i], i); move_peak_ext = 0.0; major_logged = False
                move_born_counter = bool(trend and trend.startswith("COUNTER"))
        # track the active move's peak extension; log a SESSION EVENT once it becomes a major move
        if tracker.move_active and move_seed is not None:
            move_peak_ext = max(move_peak_ext, tracker.ext_atr())
            if not major_logged and move_peak_ext >= 5.0:
                _log_event(move_seed[2], f"MAJOR {move_seed[0]} MOVE: accepted beyond {move_seed[1]}, ran +{move_peak_ext:.0f} ATR")
                major_logged = True
            # the move's with/counter standing EVOLVES as the regime shifts (born counter, trend flips → it IS the trend)
            rel_now = _trend_relation(tracker.mdir, cur_htf, cur_struct)
            if rel_now:
                if move_born_counter and not rel_now.startswith("COUNTER"):
                    wm.set("move_trend", f"born counter-trend, but the regime has SINCE FLIPPED — now {rel_now} "
                                         f"(this move became the trend)", MOVE)
                else:
                    wm.set("move_trend", rel_now, MOVE)
        if k % step != 0:
            return
        as_of = ts.iloc[i]
        if lite:                                   # HARVEST fast-path: episode needs only birth/tags/ext
            st = StateStep(ts=as_of, state="", birth=pending_birth, tags=pending_tags,
                           move_ext=round(float(tracker.ext_atr()), 2) if tracker.move_active else float("nan"),
                           price=round(float(cl[i]), 2), atr=round(float(atr[i]), 2))
            if rich:
                _fill_rich(st, i, lf_full, pending_room, pending_near, prev_hi, prev_lo,
                           daily_ctx, cl, vwap, atr, rvol, bbw)
            out.append(st)
            pending_birth = pending_kind = pending_dir = ""
            pending_tags = []
            pending_room, pending_near = float("nan"), 0
            return
        c_slice = c.iloc[: i + 1]
        feat_slice = feat.iloc[: i + 1]    # features already computed once for the whole session — reuse
        try:
            bs = read_balance(symbol, interval, as_of, candles=c_slice, feat=feat_slice)
            runs = None; mv_candles = c_slice; prepared = False
            if Sup_full is not None and Sdn_full is not None and csf_full is not None:
                runs = ({k: v[: i + 1] for k, v in Sup_full.items()},      # causal per-bar slice
                        {k: v[: i + 1] for k, v in Sdn_full.items()})
                mv_candles = csf_full.iloc[: i + 1]                        # session-featured, clean slice
                prepared = True
            mv = read_move(symbol, interval, as_of, candles=mv_candles, feat=feat_slice,
                           htf_1h=(str(htf_series[i]) if htf_series is not None else None),
                           runs=runs, prepared=prepared)
        except Exception:
            pending_birth = pending_kind = pending_dir = ""
            return
        rvol_rising = bool(np.isfinite(rvol[i]) and i >= 6
                           and np.nanmean(rvol[max(sess[0], i - 2):i + 1]) >= 1.4)
        lo = max(0, i - 40)
        compressing = bool(np.isfinite(bbw[i]) and i - lo > 10
                           and bbw[i] <= np.nanpercentile(bbw[lo:i + 1], 30))
        state, sub, why = _candidate_state(bs, mv, tracker.move_active, tracker.ext_atr(),
                                           pending_kind, pending_dir, rvol_rising, compressing)
        # ── BAR memory: refreshed every bar (price/VWAP/participation expire on their own) ──
        wm.drop_scope(BAR)
        wm.set("price", f"price {cl[i]:.2f}, ATR {atr[i]:.2f}", BAR)
        if np.isfinite(vwap[i]) and np.isfinite(atr[i]) and atr[i] > 0:
            d = (cl[i] - vwap[i]) / atr[i]
            wm.set("vwap", f"{'above' if d >= 0 else 'below'} VWAP {abs(d):.1f} ATR", BAR)
        if rvol_rising:
            wm.set("participation", "participation rising (volume picking up)", BAR)
        # ── DIRECTIONAL BIAS: a LAYERED multi-TF read (separate clock from the regime; reasoner weighs) ──
        try:
            bias = read_bias(c_slice, feat.iloc[: i + 1], i, day.date(), daily_ctx, cur_htf, cur_struct)
            wm.drop_scope(BIAS)                          # current read — refresh each bar
            for tf, val in bias.layers():
                wm.set(f"bias_{tf}", f"{tf}: {val}", BIAS)
        except Exception:
            pass
        # ── MOVE memory: extension grows; 'already exhausted once' persists for the move's life ──
        if tracker.move_active:
            wm.set("move_ext", f"{tracker.ext_atr():.1f} ATR from birth", MOVE)
            if move_seed is not None:                    # the move's behavioral STORY (sequence) for the reasoner
                seq = _move_sequence(h, l, cl, vwap, atr, move_seed[3], i,
                                     tracker.mdir, tracker.morigin, tracker.mbirth_atr)
                if seq:
                    wm.set("move_seq", seq, MOVE)
            if state == EXHAUSTED:
                wm.setdefault("exhausted_once", f"already showed exhaustion once (by {as_of:%H:%M})", MOVE)
        out.append(StateStep(ts=as_of, state=state, substate=sub, why=why, birth=pending_birth,
                             price=round(float(cl[i]), 2), atr=round(float(atr[i]), 2),
                             tags=pending_tags,
                             move_ext=round(float(tracker.ext_atr()), 2) if tracker.move_active else float("nan"),
                             memory=wm.snapshot()))
        pending_birth = pending_kind = pending_dir = ""
        pending_tags = []

    for k in range(len(sess)):
        _step(k)
    _debounce(out)
    return out


def _debounce(steps: list[StateStep]) -> None:
    """Suppress single-step top-level-state blips (state[i] differs from BOTH neighbours → revert to
    the neighbour's state) and recompute `changed` on the smoothed top-level state only — so cosmetic
    sub-state churn (tightening↔balance) doesn't read as a transition."""
    for i in range(1, len(steps) - 1):
        if steps[i].substate.startswith("birth"):
            continue                                     # a birth/rebirth is an event, never a blip
        if steps[i].state != steps[i - 1].state and steps[i - 1].state == steps[i + 1].state:
            steps[i].state = steps[i - 1].state          # lone blip → revert to surrounding state
    prev = None
    for s in steps:
        if prev is not None and s.state != prev:
            s.changed = f"{prev} → {s.state}"
        prev = s.state


def narrate(steps: list[StateStep], only_changes: bool = False) -> str:
    if not steps:
        return "(no session bars)"
    L = [f"STATE NARRATION — {steps[0].ts:%Y-%m-%d} ({len(steps)} steps)"]
    for s in steps:
        if only_changes and not s.changed and not s.birth and s is not steps[0]:
            continue
        mark = "  ►" if s.changed else "   "
        L.append(f"{mark} {s.ts:%H:%M}  {s.state:<10} {s.substate:<34} {('| ' + s.changed) if s.changed else ''}")
        if s.birth:
            L.append(f"            ◆ {s.birth}")
        for w in s.why[:2]:
            L.append(f"            · {w}")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "HAL"
    interval = sys.argv[2] if len(sys.argv) > 2 else "15m"
    for d in (sys.argv[3:] or ["2023-07-27", "2023-07-28"]):
        steps = narrate_session(sym, interval, d)
        print("\n" + "=" * 100)
        print(narrate(steps))
