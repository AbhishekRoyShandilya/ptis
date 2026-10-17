"""
agents.partner — THE UNIFIED PARTNER: Co-Planner + COACH merged over the brain.

ONE call assembles the whole partner for a situation:
  brain read (cognition_service)  →  COACH voice (compose_coach)  →  matched against the user's TAUGHT
  STRATEGIES + their VALIDATED backtest stats  →  a co-planner ACTION that depends on whether the user
  is FLAT or IN A POSITION.

It reuses ONE brain read (no double narration). Works on-demand today, and it is exactly what the
proactive monitor will call (per watchlist symbol / per open position) once the live feed lands.
INFORMS, never commands.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Position:
    direction: str                 # "up" (long) / "down" (short)
    entry: float = float("nan")


@dataclass
class PartnerContext:
    symbol: str
    position: Position | None = None
    trader: object | None = None          # coach.TraderContext (optional)
    strategies: list = field(default_factory=list)   # list[strategy_backtest.Strategy] the user has taught


@dataclass
class PartnerRead:
    symbol: str
    as_of: str
    ok: bool = True
    note: str = ""
    in_position: bool = False
    stance: str = ""
    strategy_match: str = ""       # validated strategy this matches (+ its stats), or a caveat
    action: str = ""               # the co-planner action: what to do NOW
    coach: dict = field(default_factory=dict)   # the full CoachRead (for the UI)
    speak: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _match_strategies(strategies, direction: str, tags) -> str:
    """Which of the user's taught strategies fits the current signature — preferring a VALIDATED edge."""
    from intelligence.strategy_backtest import library_stats
    t = set(tags or [])
    validated, unvalidated = [], []
    for s in strategies or []:
        if s.direction != "both" and direction and s.direction != direction:
            continue
        if not s.require.issubset(t) or (s.exclude & t):
            continue
        st = library_stats(s)
        (validated if (st.validated and st.lift > 0) else unvalidated).append((s, st))
    if validated:
        s, st = max(validated, key=lambda x: x[1].lift)
        per = f", strongest on {st.per_symbol[0][0]}" if st.per_symbol else ""
        return (f"matches your VALIDATED '{s.name}' — {st.ran_rate*100:.0f}% become money moves "
                f"(vs {st.base_ran_rate*100:.0f}% base, median +{st.median_run} ATR){per}")
    if unvalidated:
        s, _ = unvalidated[0]
        return f"matches your '{s.name}', but that signature is NOT a validated edge — don't lean on it"
    return ""


def _action(r, stance: str, position, has_validated: bool) -> str:
    alive = r.state in ("ALIVE", "TRANSITION")
    changing = r.stage == "STORY_CHANGED" or r.state == "EXHAUSTED"
    exitline = f" — out if: {r.invalidation}" if r.invalidation else ""
    if position is not None:
        same = position.direction == r.direction
        if r.direction and not same:                 # wrong side — most urgent, takes priority
            turn = " (its exhaustion could turn your way — but wait for the reversal, don't add)" if changing else ""
            return f"AGAINST YOU: the eye reads {r.direction} — your position is on the wrong side{turn}{exitline}"
        if changing:                                 # same direction, story ending
            return f"MANAGE: the story is changing — tighten the stop / take it off{exitline}"
        if same and alive:
            return f"HOLD: your move is alive ({r.move_story.replace('MOVE STORY: ', '') or r.state}); ride it to the invalidation{exitline}"
        return f"WATCH: {stance}{exitline}"
    # flat
    if has_validated:
        return f"WORTH IT: your validated edge + the read is {stance} — if you take it, risk to the invalidation{exitline}"
    if stance.startswith(("act", "go", "wait")):
        return f"{stance.upper().split('—')[0].strip()}: real read, but NOT one of your validated strategies — size small{exitline}"
    return f"STAND ASIDE: {stance}"


def partner_read(ctx: PartnerContext, interval: str = "5m", as_of=None) -> PartnerRead:
    """Fetch a brain read for ctx.symbol and assemble the unified partner view."""
    from intelligence.cognition_service import read as cognition_read
    r = cognition_read(ctx.symbol, interval=interval, as_of=as_of)
    if not r.ok:
        return PartnerRead(symbol=ctx.symbol, as_of=str(as_of), ok=False, note=r.note)
    return partner_from_read(r, ctx)


def partner_from_read(r, ctx: PartnerContext) -> PartnerRead:
    """Assemble the unified partner view from an ALREADY-FETCHED CognitionRead (lets the proactive
    monitor narrate once per symbol, then reuse the read for the partner message)."""
    from agents.coach import compose_coach
    if not r.ok:
        return PartnerRead(symbol=ctx.symbol, as_of=str(r.as_of), ok=False, note=r.note)
    cr = compose_coach(r, ctx.trader)
    match = _match_strategies(ctx.strategies, r.direction, r.tags)
    has_validated = match.startswith("matches your VALIDATED")
    action = _action(r, cr.stance, ctx.position, has_validated)

    lines = [cr.speak()]
    if match:
        lines.append(f"  STRATEGY: {match}")
    lines.append(f"  ▶ {action}")
    return PartnerRead(
        symbol=ctx.symbol, as_of=r.as_of, in_position=ctx.position is not None,
        stance=cr.stance, strategy_match=match, action=action,
        coach=cr.to_dict(), speak="\n".join(lines),
    )
