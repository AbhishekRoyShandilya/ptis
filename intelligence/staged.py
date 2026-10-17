"""
intelligence.staged — the TRADER staging over the eye's facts: Candidate → (validating) → Money Move
/ Failed Money Move. The fix for the "two senses of birth" confusion (project_candidate_to_money_move
_staging): a birth is only a CANDIDATE ("a strong break appeared"), never the claim that a move started.

Two principles (user, 2026-06-19):
  1. The unit is the money-move SETUP, not the candle. A studied setup (e.g. tight compression → clean
     break into clear air → big move) that appears and DOESN'T follow through is a FAILED MONEY MOVE —
     a failed instance of a setup — not a "failed birth".
  2. The CALL belongs at the CONFIDENT point, not candle 1. Potential and failed look identical at the
     first candle, so we do NOT call there. We call when the candidate has PROVED it is moving — held
     beyond its break level AND extended into room — late enough to be sure, early enough to be useful.
     "Sure, not late"; occasionally textbook setups confirm early, but there is no pressure to be early.

This is a READOUT over existing facts (birth/hold/ext from BirthTracker, room, participation) — NOT new
detection. The point-in-time recognition is CAUSAL (uses only info up to the call bar); the realised
peak is attached afterward only to SCORE whether the call matched reality.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# the staged vocabulary
CANDIDATE, MONEY_MOVE, FAILED_MOVE = "CANDIDATE", "MONEY_MOVE", "FAILED_MOVE"
# the OBSERVE-AGAIN stage (the handoff to the move-observer after recognition) + the exit
OBSERVE, STORY_CHANGED = "OBSERVE", "STORY_CHANGED"

# the confident-call knob: a candidate is RECOGNISED as a move once it has HELD and extended this far
# from birth (it has proved it is moving, not just a strong candle). Higher = surer but later; lower =
# earlier but more coin-flips. VALIDATED on a 177-episode/4-symbol library (2026-06-19): 2.5 halves the
# false-money-move calls vs 2.0 (24%->13% that die before +3 ATR) while still catching ALL real runs
# (the model has 100% recall of money-moves; the knob only trades precision). "Sure, not late."
# HONEST LIMIT: even at the call, magnitude is unpredictable — ~half of real-move calls stay modest
# (+3-5), ~half become big runs (+5). The call recognises "a real move is underway", never how far.
CONFIRM_EXT = 2.5
EARLY_BARS = 2          # confirmed within ~2 bars of the break = caught early (textbook)
LATE_EXT = 4.0          # already this extended at the moment of confidence = we are LATE to it


# a candidate's WORTH-ATTENTION read — the trader's higher bar (user, 2026-06-19): a candidate is not
# "strong candle detected", it is "worth attention" only if room/effort/timing support COMMITMENT. Graded,
# NOT a filter (every strong break still fires — we never miss a clear-air reversal). Built from the
# MECHANICAL, non-confounded signals only (room, effort, phase) — NOT trend-direction (a known confound:
# the big moves are reversals = counter-trend at birth, so trend-direction can't be trusted as a marker).
def candidate_quality(tags) -> tuple[str, str, list[str]]:
    """Grade a candidate's worth-attention from its tags. Returns (label, headline, reasons). ROOM_WALL
    (into immediate resistance, <0.5 ATR) is a near-VETO — a strong candle into a wall has nowhere to go,
    so it caps at LOW no matter the bonuses (the user's eye: SBIN 09:20 opened straight into resistance =
    not worthy, regardless of effort/open). Built from MECHANICAL signals only, NOT trend-direction."""
    t = set(tags)
    pos, neg = [], []
    wall = "ROOM_WALL" in t
    if "ROOM_CLEAR" in t:
        pos.append("clear air ahead")
    elif "ROOM_SOME" in t:
        pos.append("room ahead")
    elif "ROOM_TIGHT" in t:
        neg.append("limited room overhead")
    elif "ROOM_WALL" in t:
        neg.append("into a wall (resistance right here)")
    if "STRONG_EFFORT" in t:
        pos.append("strong effort (volume/expansion)")
    if "WEAK_EFFORT" in t:
        neg.append("weak effort (no participation)")
    if "PHASE_OPEN" in t or "AT_OPEN" in t:
        pos.append("opening drive")
    if "PHASE_AFTERNOON" in t:
        neg.append("late-day (afternoon)")
    # NOTE (2026-06-19, 453-episode validation): ROOM_WALL is NOT a validated failure marker — it FLIPS
    # sign across halves (a strong break can blow THROUGH a wall, e.g. SBIN 13:05 +18.8). So it is a
    # noted CAUTION (a single negative in `neg`), NOT the decisive veto it was. At the candle, "into a
    # wall" is genuinely ambiguous; only CONFIRMATION resolves stall-vs-breakthrough. The robustly
    # validated drivers are TIMING (open vs late/afternoon/flat) + ROOM_CLEAR.
    _ = wall
    score = len(pos) - len(neg)
    label = "HIGH" if score >= 2 else "LOW" if score <= -1 else "MED"
    head = "worth attention" if score >= 2 else "low-worth" if score <= -1 else "marginal"
    return label, head, pos + neg


@dataclass
class Stage:
    ts: object
    kind: str                      # CANDIDATE / MONEY_MOVE / FAILED_MOVE
    direction: str = ""
    quality: str = ""              # worth-attention read (CANDIDATE stage)
    ext: float = float("nan")      # extension-from-birth at this stage event
    timing: str = ""               # early / on-time / late (for MONEY_MOVE)
    peak_ext: float = float("nan") # realised peak of the move (scoring, not used for the causal call)
    tags: list[str] = field(default_factory=list)
    note: str = ""
    recall_note: str = ""          # memory-informed expectation at the CANDIDATE stage (analogy, not forecast)


def _recall_oneline(rr) -> str:
    """One-line memory expectation for a CANDIDATE: how candidates that RHYME with this one resolved.
    Analogy, not a forecast — it informs the watch, it does not predict this one."""
    if not rr.any():
        return ""
    ex = rr.analogues[0].episode if rr.analogues else None
    extra = (f" (closest: {ex.symbol} {ex.date} → {ex.outcome} +{ex.peak_ext:.0f} ATR)"
             if ex and ex.peak_ext == ex.peak_ext else "")
    pct = round(100 * rr.n_ran / rr.n_close) if rr.n_close else 0
    return (f"memory: {rr.n_close} candidates rhymed → {rr.n_ran} became money moves ({pct}%), "
            f"{rr.n_modest} modest, {rr.n_failed} failed{extra}")


def _direction(step) -> str:
    parts = (step.birth or "").split()
    return parts[1] if len(parts) > 1 and parts[1] in ("up", "down") else ""


# the move's health is read from the ONE thing we trust — how long since it last made new ground
# (monotonic extension). The noisy WEAKENING/EXHAUSTED flag is deliberately NOT used (it churns on every
# brief pause). A trend with short pauses is RUNNING; it only stalls/exhausts when it genuinely stops.
STALL_BARS = 4          # no new ground for this long = stalled (a real pause, not a 1-2 bar breather)


def _observer_read(since_high: int) -> tuple[str, str]:
    """The move-observer's real-time read of a LIVE move, from bars-since-last-new-extension-high — NO
    prediction, NO reliance on the churny weakening flag. Still making ground (incl. brief pauses) =
    RUNNING; stopped for a stretch = STALLING; stopped for a long stretch = EXHAUSTING (rolling over).
    The fine exhaustion-vs-healthy-pause nuance is ultimately the REASONER's call; this is the cheap,
    stable track."""
    if since_high <= STALL_BARS:
        return "RUNNING", "advancing — still making ground"
    if since_high <= 2 * STALL_BARS:
        return "STALLING", "paused — no new ground for a while, watching"
    return "EXHAUSTING", "stalled out — rolling over, story turning"


def stage_session(steps, confirm_ext: float = CONFIRM_EXT, library=None,
                  symbol: str = "", date=None) -> list[Stage]:
    """Walk a narrated session and emit the staged story for each candidate: a CANDIDATE at the birth
    (with a memory-informed expectation from the episodic LIBRARY, if given), then either a MONEY_MOVE
    (at the first bar it has held + extended >= confirm_ext — the confident call) or a FAILED_MOVE (it
    died before proving it was moving). Point-in-time: the call bar uses only extension up to that bar;
    the peak is recorded only to score the call. `library` recall EXCLUDES this session (symbol+date)."""
    libx = None
    if library is not None:
        from intelligence.episodic import recall as _recall
        _d = str(date) if date is not None else ""
        libx = [e for e in library if not (e.symbol == symbol and e.date == _d)]
    births = [i for i, s in enumerate(steps) if s.tags]
    out: list[Stage] = []
    for idx, j in enumerate(births):
        end = births[idx + 1] if idx + 1 < len(births) else len(steps)
        d = _direction(steps[j])
        cand = Stage(ts=steps[j].ts, kind=CANDIDATE, direction=d, tags=list(steps[j].tags),
                     note="a strong break appeared — watching, not a move yet")
        _lab, _head, _reasons = candidate_quality(steps[j].tags)
        if _reasons:
            cand.quality = f"{_head} [{_lab}]: {', '.join(_reasons)}"
        if libx is not None and d:
            cand.recall_note = _recall_oneline(_recall(libx, d, steps[j].tags))
        out.append(cand)
        # walk the candidate's life: first bar it is still active AND ext >= confirm_ext = the call
        call_k = None
        peak = float("nan")
        for k in range(j, end):
            me = steps[k].move_ext
            if me != me:                       # NaN → the move has been released (died)
                break
            peak = me if peak != peak else max(peak, me)
            if call_k is None and me >= confirm_ext:
                call_k = k
        # peak across the whole life (even past the break point, for scoring)
        for k in range(j, end):
            me = steps[k].move_ext
            if me == me:
                peak = me if peak != peak else max(peak, me)
        if call_k is not None:
            ext_at_call = steps[call_k].move_ext
            bars = call_k - j
            timing = ("early" if bars <= EARLY_BARS else "late" if ext_at_call >= LATE_EXT else "on-time")
            note = {"early": "caught near the break (textbook — confirmed early)",
                    "on-time": "confirmed at the sure point — held and travelling, still useful",
                    "late": "already extended by the time it confirmed — much of the move is done"}[timing]
            out.append(Stage(ts=steps[call_k].ts, kind=MONEY_MOVE, direction=d, ext=round(float(ext_at_call), 1),
                             timing=timing, peak_ext=round(float(peak), 1) if peak == peak else float("nan"),
                             tags=list(steps[j].tags), note=note))
            # ── OBSERVE-AGAIN: hand off to the move-observer. Watch the live move, comment on CHANGES,
            #    and exit when the story changes (structural release = swing broke / structure against). ──
            prev_read = None
            last_ext = float(ext_at_call)
            since_high = 0                                # bars since the last NEW extension high
            release_ts = None
            for k in range(call_k + 1, end):
                me = steps[k].move_ext
                if me != me:                              # released → the story has changed (move over)
                    release_ts = steps[k].ts
                    break
                if me > last_ext + 0.1:
                    last_ext = me; since_high = 0
                else:
                    since_high += 1
                tag, phrase = _observer_read(since_high)
                if tag != prev_read:                      # comment only when the read changes
                    out.append(Stage(ts=steps[k].ts, kind=OBSERVE, direction=d, ext=round(float(me), 1), note=phrase))
                    prev_read = tag
            out.append(Stage(ts=release_ts or steps[end - 1].ts, kind=STORY_CHANGED, direction=d,
                             peak_ext=round(float(peak), 1) if peak == peak else float("nan"),
                             note=("story changed — swing/structure broke against the move; out"
                                   if release_ts else "session/move ended")))
        else:
            out.append(Stage(ts=steps[j].ts, kind=FAILED_MOVE, direction=d,
                             peak_ext=round(float(peak), 1) if peak == peak else float("nan"),
                             tags=list(steps[j].tags),
                             note=f"never proved it was moving (peaked +{peak:.1f} ATR) — a failed money move"))
    return out


def render_stages(stages: list[Stage]) -> str:
    """Trader-facing staged readout."""
    if not stages:
        return "(no candidates)"
    L = []
    for s in stages:
        t = f"{s.ts:%H:%M}"
        if s.kind == CANDIDATE:
            L.append(f"  {t}  CANDIDATE ({s.direction})  · {s.quality or s.note}")
            if s.recall_note:
                L.append(f"            {s.recall_note}")
        elif s.kind == MONEY_MOVE:
            L.append(f"  {t}  → MONEY MOVE ({s.direction}) [{s.timing}] at +{s.ext} ATR  · {s.note}")
        elif s.kind == OBSERVE:
            L.append(f"  {t}      · observing (+{s.ext} ATR): {s.note}")
        elif s.kind == STORY_CHANGED:
            pk = f"  [move peaked +{s.peak_ext} ATR]" if s.peak_ext == s.peak_ext else ""
            L.append(f"  {t}  ◇ STORY CHANGED ({s.direction})  · {s.note}{pk}")
        else:
            L.append(f"  {t}  ✗ FAILED MOVE ({s.direction})  · {s.note}")
    return "\n".join(L)
