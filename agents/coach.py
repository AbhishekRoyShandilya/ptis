"""
agents.coach — THE COACH face: the human-facing VOICE over the brain (the trading partner).

NOT a new engine. It asks the Cognition Service ONE question — `read(symbol)` — and turns that already-
grounded read into a partner message: what's happening (mechanism), the STANCE (act / wait / stand-aside
/ late), WHY it's valid (grounded facts + recall), the EXIT (the reasoner's invalidation), and — when it
knows the trader's context — a DISCIPLINE nudge and CALIBRATED DISAGREEMENT when the trader's intent
conflicts with the read.

It COMPOSES DETERMINISTICALLY from the CognitionRead (no second LLM call → no new folklore; the cognition
was already produced + grounded by the reasoner). It INFORMS; it never commands, sizes, or auto-trades.
The honest COACH edge (architecture): recognition + timing/attention + discipline — not market prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class TraderContext:
    """What the coach knows about the human right now (all optional — the coach works without it)."""
    intended_direction: str = ""        # "long" / "short" / ""
    todays_trade_count: int = 0
    current_pnl: float = 0.0
    tendencies: list[str] = field(default_factory=list)   # e.g. ["enters late on momentum", "holds losers"]


@dataclass
class CoachRead:
    symbol: str
    as_of: str
    ok: bool = True
    note: str = ""
    headline: str = ""          # the situation in one plain line
    stance: str = ""            # act-with / wait-for-acceptance / stand-aside / already-late / no-read
    whats_happening: str = ""   # the mechanism, plain
    why: list[str] = field(default_factory=list)   # grounded reasons (conviction · recall · move story)
    exit: str = ""              # the invalidation, framed as the exit
    blind_spot: str = ""        # discipline / behavioral nudge (only if trader context)
    disagreement: str = ""      # calibrated disagreement if trader intent conflicts with the read
    early_flag: str = ""        # proactive: a fresh setup worth attention
    unsure: bool = False

    def speak(self) -> str:
        if not self.ok:
            return f"[{self.symbol}] {self.note}"
        L = [f"[{self.symbol} · {self.as_of}]  {self.headline}"]
        if self.early_flag:
            L.append(f"  👀 {self.early_flag}")
        L.append(f"  STANCE: {self.stance}")
        if self.whats_happening:
            L.append(f"  WHAT'S HAPPENING: {self.whats_happening}")
        for w in self.why:
            L.append(f"   · {w}")
        if self.exit:
            L.append(f"  YOUR EXIT (you're wrong if): {self.exit}")
        if self.disagreement:
            L.append(f"  ⚠ HEADS-UP: {self.disagreement}")
        if self.blind_spot:
            L.append(f"  DISCIPLINE: {self.blind_spot}")
        return "\n".join(L)

    def to_dict(self) -> dict:
        return asdict(self)


_STANCE_SAY = {
    "go-with": "act-with — recognized & aligned; structural risk sits at the invalidation",
    "wait-for-acceptance": "wait-for-acceptance — could be real but hasn't proven it; let it hold first",
    "stand-aside": "stand-aside — low conviction / not an edge here",
    "already-late": "already-late — a real move, but don't chase the entry",
}


def coach(symbol: str, interval: str = "5m", as_of=None, trader: TraderContext | None = None) -> CoachRead:
    """Partner read for `symbol`: asks the Cognition Service, then speaks it as a coach. `trader` (optional)
    adds the discipline nudge + calibrated disagreement."""
    from intelligence.cognition_service import read as cognition_read
    r = cognition_read(symbol, interval=interval, as_of=as_of)
    return compose_coach(r, trader)


def compose_coach(r, trader: TraderContext | None = None) -> CoachRead:
    """Speak an already-fetched CognitionRead as a coach (no extra narration). Lets the Unified Partner
    reuse ONE brain read for both the coach voice and the strategy/position logic."""
    symbol = r.symbol
    if not r.ok:
        return CoachRead(symbol=symbol, as_of=str(r.as_of), ok=False, note=r.note)

    no_read = bool(r.unsure or r.conviction in ("", "none"))
    stance = "no-read" if no_read else _STANCE_SAY.get(r.bias, r.bias or "no-read")
    headline = r.read or (f"{r.state.lower()} — {r.substate}" if r.substate else r.state.lower())

    why: list[str] = []
    if r.conviction:
        why.append(f"conviction: {r.conviction}")
    if r.move_story:
        why.append(r.move_story.replace("MOVE STORY: ", "the move so far: "))
    if r.recall:
        why.append(r.recall[0])                    # the "N analogues → X ran/failed" line (analogy, not forecast)

    # proactive: a fresh setup worth attention
    early = ""
    if r.stage == "CANDIDATE":
        early = f"fresh candidate forming — {r.stage_note}" if r.stage_note else "fresh candidate forming — watching"
    elif r.stage == "MONEY_MOVE":
        early = f"a money move was recognized — {r.stage_note}"

    # trader-aware: discipline + calibrated disagreement
    blind_spot = disagreement = ""
    if trader is not None:
        intended = (trader.intended_direction or "").lower()
        intended_dir = "up" if intended in ("long", "buy", "up") else "down" if intended in ("short", "sell", "down") else ""
        if intended_dir and r.direction and intended_dir != r.direction:
            disagreement = (f"you're looking {intended}, but the read is a {r.direction} move "
                            f"({r.conviction or 'unclear'} conviction) — the eye is against your idea here")
        bits = []
        if trader.todays_trade_count >= 3 and (no_read or r.bias in ("stand-aside", "already-late")):
            bits.append(f"{trader.todays_trade_count} trades already today and this one is {('a no-read' if no_read else r.bias)} — overtrading check")
        if trader.current_pnl < 0 and not no_read and r.bias != "go-with":
            bits.append("you're red today — don't force a marginal one to get it back")
        for t in trader.tendencies:
            if "late" in t.lower() and r.bias == "already-late":
                bits.append(f"your tendency ({t}) is the risk here — it's already late")
            if "hold" in t.lower() and "loser" in t.lower() and r.invalidation:
                bits.append(f"your tendency ({t}) — pre-commit the exit: {r.invalidation}")
        blind_spot = "; ".join(bits)

    return CoachRead(
        symbol=symbol, as_of=r.as_of, headline=headline, stance=stance,
        whats_happening=r.mechanism, why=why, exit=r.invalidation,
        blind_spot=blind_spot, disagreement=disagreement, early_flag=early, unsure=bool(r.unsure),
    )
