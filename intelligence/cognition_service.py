"""
intelligence.cognition_service — THE single public entry to the brain (the one seam every face calls).

Every face — the /intelligence route, the co-planner, the COACH, AVT — asks the SAME question through
this service and gets the SAME structured read, so no surface re-implements cognition. It wraps the
whole pipeline:

    eye (state_narrator + birth)  →  scoped working-memory + directional bias + session events
                                  →  MOVE STORY (the move's behavioral sequence)
                                  →  episodic recall (dated analogues)
                                  →  staged Candidate → Money Move
                                  →  reasoner VERDICT (mechanism · consequence · INVALIDATION, on Llama)

    read(symbol, as_of) -> CognitionRead     # point-in-time: narrates up to as_of, judges the "now".

This is how "the brain answers most of the questions": faces stop carrying their own eval logic and
just ask here. (Latency note: today it narrates the day up to as_of per call — correct but not cheap;
a streaming/incremental eye is a separate later optimization. The CONTRACT here won't change.)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import pandas as pd


@dataclass
class CognitionRead:
    """One structured read of what the brain sees for a symbol at a moment — the shared currency of
    every face. The COACH speaks the mechanism/bias/invalidation; /evaluate returns the verdict; AVT
    uses stage + invalidation; the co-planner uses the read + the supporting facts."""
    symbol: str
    interval: str
    as_of: str
    ok: bool = True
    note: str = ""
    # ── the eye ──
    state: str = ""               # BALANCE / TRANSITION / ALIVE / EXHAUSTED
    substate: str = ""
    direction: str = ""           # up / down (the active move's direction, if any)
    price: float = float("nan")
    atr: float = float("nan")
    # ── the reasoner verdict (the cognition) ──
    conviction: str = ""          # high / medium / low / none
    bias: str = ""                # go-with / wait-for-acceptance / stand-aside / already-late
    mechanism: str = ""           # what the market is doing & why (footprint-grounded)
    consequence: str = ""         # forward implication IF it holds (recognition, no price)
    invalidation: str = ""        # the observable that breaks the read = the EXIT
    read: str = ""                # analyst one-liner
    unsure: bool = False
    reasoner_mode: str = ""        # llm / proxy / pending
    # ── staged money-move framing ──
    stage: str = ""               # CANDIDATE / MONEY_MOVE / FAILED_MOVE / OBSERVE / STORY_CHANGED
    stage_note: str = ""
    tags: list = field(default_factory=list)   # the active move's recognition signature (for strategy match)
    # ── supporting facts (for faces that want detail) ──
    bias_layers: list = field(default_factory=list)
    move_story: str = ""
    recall: list = field(default_factory=list)
    session_events: list = field(default_factory=list)
    facts: str = ""               # the exact facts block the reasoner judged (for COACH / debugging)
    avoid: list = field(default_factory=list)   # VALIDATED failure-patterns this birth matches (failure_miner)

    def headline(self) -> str:
        if not self.ok:
            return f"{self.symbol}: {self.note}"
        bits = [f"{self.symbol} {self.as_of}", self.state]
        if self.conviction:
            bits.append(f"{self.conviction} conviction")
        if self.bias:
            bits.append(self.bias)
        return " · ".join(bits) + (f" — {self.read}" if self.read else "")

    def to_dict(self) -> dict:
        return asdict(self)


# ── Layer-2 verdict cache ──────────────────────────────────────────────────────────────────────
# The reasoner LLM call is the eye's rate-limit / latency ceiling. We reuse the last verdict while the
# material situation (state/substate/direction/stage) is unchanged, so re-reading a quiet symbol (the
# proactive loop every few minutes) costs ZERO LLM calls. Keyed by (symbol, interval, DATE) so a
# historical read (trade review at an old as_of) never collides with a live read; TTL bounds staleness.
_VERDICT_CACHE: dict = {}          # (symbol, interval, date) -> (signature, verdict, monotonic_ts)
_VERDICT_TTL_S = 900.0             # re-judge at least every 15 min even if the signature is unchanged
_VERDICT_CACHE_MAX = 800


def _cached_judge(symbol: str, interval: str, day, sig: tuple, cur, arc, avoid=None):
    """Reasoner verdict with material-change caching (see the note above)."""
    import time as _t
    from intelligence.reasoner import StateReasoner
    key = (symbol, interval, day)
    hit = _VERDICT_CACHE.get(key)
    if hit is not None and hit[0] == sig and (_t.monotonic() - hit[2]) < _VERDICT_TTL_S:
        return hit[1]
    v = StateReasoner().judge(cur, arc, avoid=avoid)
    if len(_VERDICT_CACHE) >= _VERDICT_CACHE_MAX:
        _VERDICT_CACHE.clear()      # simple bound; it re-warms in one pass
    _VERDICT_CACHE[key] = (sig, v, _t.monotonic())
    return v


def read(symbol: str, interval: str = "5m", as_of=None, candles=None,
         with_recall: bool = True) -> CognitionRead:
    """Point-in-time brain read for `symbol` at `as_of` (default: now). Narrates the session up to
    as_of, then returns the eye state + staged framing + episodic recall + reasoner verdict as ONE
    CognitionRead. Degrades gracefully: no LLM key → verdict fields empty (reasoner_mode='pending'),
    but the eye/staged/recall facts are still returned."""
    from intelligence.state_narrator import narrate_session
    from intelligence.reasoner import StateReasoner, facts_block
    from intelligence.staged import stage_session
    from intelligence.working_memory import MOVE, BIAS, SESSION, LIBRARY

    # Default as_of to the LATEST AVAILABLE bar, not literally "now" — so off-hours / pre-market it reads
    # the last real session instead of an empty "today" (the "no data" UX bug). During market hours the
    # latest bar IS today's forming bar.
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        # Broker/trade timestamps are tz-aware (+05:30); the lake is tz-naive IST wall-clock. Coerce to
        # tz-naive IST so the `s.ts <= as_of_ts` comparison below doesn't crash ("Cannot compare tz-naive
        # and tz-aware timestamps") — the bug that made the eye unable to read recent (today's) trades.
        if as_of_ts.tz is not None:
            as_of_ts = as_of_ts.tz_convert("Asia/Kolkata").tz_localize(None)
    else:
        try:
            from marketdata.candle_store import get_candle_store
            _now = pd.Timestamp.now()
            _recent = get_candle_store().get(symbol, interval, _now.normalize() - pd.Timedelta(days=12),
                                             _now + pd.Timedelta(days=1), live=True)
            as_of_ts = pd.to_datetime(_recent["ts"]).max() if _recent is not None and not _recent.empty else _now
        except Exception:
            as_of_ts = pd.Timestamp.now()
    date = as_of_ts.normalize()
    try:
        steps = narrate_session(symbol, interval, date, candles=candles, with_recall=with_recall)
    except Exception as e:
        return CognitionRead(symbol, interval, str(as_of_ts), ok=False, note=f"narration failed: {e}")
    steps = [s for s in steps if s.ts <= as_of_ts] or steps
    if not steps:
        return CognitionRead(symbol, interval, str(as_of_ts), ok=False, note="no session data")

    cur = steps[-1]
    arc = steps[max(0, len(steps) - 7):len(steps) - 1]

    # Facts needed BOTH for the verdict-cache signature and the return payload — compute before judging.
    mem = cur.memory or {}
    move_facts = mem.get(MOVE, [])
    direction = ("down" if any(f.lower().startswith("down move") for f in move_facts)
                 else "up" if any(f.lower().startswith("up move") for f in move_facts) else "")

    stage = stage_note = ""
    try:
        from intelligence.staged import FAILED_MOVE, OBSERVE, STALL_BARS
        stages = stage_session(steps)
        if stages:
            last = stages[-1]
            bars_ago = sum(1 for s in steps if s.ts > last.ts)   # how stale is the staged event?
            timing = f" [{last.timing}]" if getattr(last, "timing", "") else ""
            peak = f" · peak +{last.peak_ext} ATR" if last.peak_ext == last.peak_ext else ""
            if last.kind == FAILED_MOVE and bars_ago > STALL_BARS:
                # the failed break was a while ago — we're observing now, not in that move (was being
                # reported as the LIVE stage, e.g. an EOD read still saying "FAILED_MOVE" from noon).
                stage = OBSERVE
                stage_note = f"no active money move — watching (earlier {last.ts:%H:%M}: failed break{peak})"
            else:
                stage = last.kind
                stage_note = f"{last.note}{timing}{peak}".strip()
    except Exception:
        pass

    active_tags = next((list(s.tags) for s in reversed(steps) if s.tags), [])   # the live move's signature
    # VALIDATED failure-patterns this birth matches — the eye's learned "avoid this shape" (grounded, not a veto)
    try:
        from intelligence.failure_miner import check as _avoid_check
        avoid = _avoid_check(active_tags, direction) if active_tags else []
    except Exception:
        avoid = []

    # Layer-2 verdict cache: the reasoner LLM is the rate-limit/latency ceiling. Re-judge ONLY when the
    # material situation changed (state/substate/direction/stage/avoid-match); else reuse the last verdict —
    # so an UNCHANGED symbol costs ZERO LLM calls and the proactive loop scales to a full watchlist.
    avoid_sig = tuple(sorted(" ".join(a["combo"]) for a in avoid))    # re-judge when the avoid-match changes
    v = _cached_judge(symbol, interval, as_of_ts.date(),
                      (cur.state, cur.substate, direction, stage, avoid_sig), cur, arc, avoid)

    move_story = next((f for f in move_facts if f.startswith("MOVE STORY")), "")
    session_events = [f for f in mem.get(SESSION, []) if f[:1].isdigit()]   # events render as "HH:MM …"
    return CognitionRead(
        symbol=symbol, interval=interval, as_of=f"{cur.ts:%Y-%m-%d %H:%M}",
        state=cur.state, substate=cur.substate, direction=direction, price=cur.price, atr=cur.atr,
        conviction=v.conviction, bias=v.bias, mechanism=v.mechanism, consequence=v.consequence,
        invalidation=v.invalidation, read=v.read, unsure=v.unsure, reasoner_mode=v.mode,
        stage=stage, stage_note=stage_note, tags=active_tags, avoid=avoid,
        bias_layers=mem.get(BIAS, []), move_story=move_story, recall=mem.get(LIBRARY, []),
        session_events=session_events, facts=facts_block(cur, arc, avoid),
    )
