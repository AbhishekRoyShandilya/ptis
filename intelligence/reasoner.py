"""
intelligence.reasoner — THE REASONER over the eye's state-recognition FACTS (the JUDGMENT layer).

The eye (state_narrator + birth) emits FACTS at each change point: a move was born / re-born, its
direction, its extension-from-birth, the structure overhead/underfoot (room), whether the break is
WITH or COUNTER the prevailing structure, the 1h agreement, the balance band. The reasoner JUDGES —
the calls a threshold cannot make in context:
  · CONVICTION — a counter-trend down-break into nearby support with the 1h still bull is a
    low-conviction shakeout, even though the eye truthfully saw a real strong break.
  · the owned STATE verdict — resolve ALIVE-vs-EXHAUSTED: extended INTO resistance and weakening =
    exhausting; extended into CLEAR AIR still firing = mature but ALIVE.
  · honest "unsure" where the facts are genuinely mixed.
Recognition, NOT prediction: it never names a price/target; it reads the situation like an analyst.

API-READY + PROXY-TESTABLE. `build_prompt()` produces the EXACT text the model sees. When an LLM key
is configured (config.settings.LLM_API_KEY) `judge()` calls the model on the existing LLMClient and
parses the FIXED JSON schema. With no key it runs in PROXY mode — `judge()` returns a Verdict whose
`.prompt` carries that exact text for a proxy (a human / Claude) to answer in the SAME schema, and
`judge(..., proxy_response=<json>)` parses that answer identically. So the day the API connects,
nothing about the contract changes — it behaves exactly as the proxy did.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

SYSTEM_PROMPT = """You are the REASONER inside a trading-intelligence system. You are given the EYE's structured FACTS about an unfolding intraday move at one moment. You NEVER see raw candles and must NOT invent a number not in the facts. Your job is COGNITION — recognition and EXPLANATION — NOT prediction: you never name a price target and never claim what price WILL do.

Do NOT merely label the move. EXPLAIN it: infer what the market is DOING and WHY, from the footprints you can observe (acceptance, rejection strength, participation/effort, room, who is gaining or losing control). You CANNOT see order flow — reason about inferred supply/demand from price+volume FOOTPRINTS ("supply being absorbed", "sellers losing control", "demand exhausting"). NEVER claim intent you cannot see — "institutions are buying" is FORBIDDEN (imagination, not evidence).

Every read must be GROUNDED and FALSIFIABLE:
- MECHANISM: what is happening and why, tied to the SPECIFIC facts given.
- CONSEQUENCE: the forward implication IF the mechanism holds — a recognition (e.g. "a breakout becomes more likely"), NEVER a price/target.
- INVALIDATION: the observable that would BREAK this read. This is also the EXIT — when you would be proven wrong.

Then make the calls a threshold cannot:

STATE — BALANCE (rotating, no move) / TRANSITION (a move being born — strong break just left balance) / ALIVE (in control, healthy) / EXHAUSTED (has run AND is weakening). Far-extended but still firing into CLEAR AIR = mature-but-ALIVE; extended INTO overhead structure and weakening = EXHAUSTED.

CONVICTION — high / medium / low / none. Judge the CONTEXT CONDITIONALLY — signals INTERACT, do NOT just add them up. Validated principles (weigh, don't apply blindly):
- Counter-trend at the birth is NOT by itself a reason to stand aside — genuine REVERSALS are counter-trend at birth. Weigh room, acceptance, effort and behaviour, not the with/counter label alone.
- Born at the OPEN tends to run; late-day / afternoon births tend to fade. BUT an opening break ALREADY several ATR extended on the first candle is OPENING VOLATILITY — provisional, do not chase.
- "Into a wall" (overhead/underfoot structure, little room) is a CAUTION, not a veto — a strong break WITH effort can break THROUGH a wall (that IS a breakout). Resolve by whether it ACCEPTS beyond the level and keeps making ground, vs stalls and gets rejected.
- EFFORT matters: a break with real participation (volume / range expansion) is more real than a weak poke.
- CLEAR AIR and room-remaining favour continuation; a break after a long distance already travelled is "late".
- A VALIDATED FAILURE-PATTERN line (if present) is OUT-OF-SAMPLE grounding that THIS EXACT shape has historically failed far more than the baseline (split-half + cross-symbol + significance tested). Weigh it HEAVILY toward wait-for-acceptance / stand-aside — but it is a BASE-RATE, not a veto: genuine ACCEPTANCE (held beyond the level, still making ground, real effort) can still override it. Never chase a shape flagged here that has not yet proven acceptance.

BIAS — go-with (recognized, aligned) / wait-for-acceptance (could be real, not yet held) / stand-aside (low conviction / chop / trap) / already-late (real but extended — don't chase).

HONESTY — if the facts are genuinely mixed, unsure=true. You are not more predictive than the market.

Respond with ONLY a JSON object, no prose:
{"state":"BALANCE|TRANSITION|ALIVE|EXHAUSTED","conviction":"high|medium|low|none","mechanism":"<what the market is doing & why, in footprint terms, grounded in the facts>","consequence":"<forward implication if the mechanism holds — recognition, no price>","invalidation":"<the observable that would break this read = the exit>","bias":"go-with|wait-for-acceptance|stand-aside|already-late","read":"<=140 char analyst one-liner>","why":["<reason tied to a fact>"],"unsure":true|false}"""


@dataclass
class Verdict:
    state: str = ""
    conviction: str = ""
    read: str = ""
    why: list[str] = field(default_factory=list)
    bias: str = ""
    unsure: bool = False
    mechanism: str = ""          # the EXPLANATION: what the market is doing & why (footprint-grounded)
    consequence: str = ""        # forward implication IF the mechanism holds (recognition, no price)
    invalidation: str = ""       # the observable that breaks the read = the exit / thesis-break
    mode: str = ""               # "llm" | "proxy" | "pending"
    prompt: str = ""             # in pending mode, the exact prompt the model/proxy must answer
    raw: str = ""                # the raw model/proxy response

    @property
    def pending(self) -> bool:
        return self.mode == "pending"

    def one_line(self) -> str:
        u = " (unsure)" if self.unsure else ""
        return f"[{self.state} · {self.conviction} conviction · {self.bias}]{u} {self.read}"

    def explanation(self) -> str:
        """The cognition layer: mechanism → consequence, with the invalidation (= the exit)."""
        parts = []
        if self.mechanism:
            parts.append(f"MECHANISM: {self.mechanism}")
        if self.consequence:
            parts.append(f"CONSEQUENCE: {self.consequence}")
        if self.invalidation:
            parts.append(f"INVALIDATION (exit): {self.invalidation}")
        return "\n".join(parts)


def _session_phase(ts) -> str:
    """Where in the NSE session (09:15–15:30) — time-of-day materially changes conviction (a late-day
    birth has little runway; an EOD move is a different animal than a first-hour one)."""
    m = ts.hour * 60 + ts.minute
    to_close = 930 - m            # 15:30 = 930
    if m < 615:   phase = "first hour (09:15–10:15) — high energy, can be noisy"
    elif m < 690: phase = "late morning (10:15–11:30)"
    elif m < 795: phase = "midday (11:30–13:15) — typically lower energy"
    elif m < 870: phase = "early afternoon (13:15–14:30)"
    elif m < 910: phase = "late afternoon (14:30–15:10)"
    else:         phase = "EOD (after 15:10) — squaring-up / BTST flows"
    return f"{phase}; {to_close} min to close"


def facts_block(step, arc: list, avoid: list | None = None) -> str:
    """Serialize one change point + its recent arc into the FACTS the reasoner reads. The bulk is the
    SCOPED WORKING MEMORY (SESSION / MOVE / BAR) — what is currently RELEVANT, each fact alive only as
    long as it should be (so counter-trend persists for the move, stale bar facts expire). `step`/`arc`
    are state_narrator.StateStep objects. `avoid` = validated failure-patterns this birth matches."""
    from intelligence.working_memory import render_snapshot
    lines = [f"NOW: {step.ts:%Y-%m-%d %H:%M}  ·  {_session_phase(step.ts)}"]
    mem = render_snapshot(step.memory) if step.memory else ""
    if mem:
        lines.append(mem)
    lines.append(f"EYE STATE (deterministic candidate): {step.state}"
                 + (f" — {step.substate}" if step.substate else ""))
    if step.birth:
        lines.append(f"BIRTH/EVENT (this bar): {step.birth}")
    if step.changed:
        lines.append(f"WHAT CHANGED: {step.changed}")
    for w in step.why:
        lines.append(f"EYE NOTE: {w}")
    for a in (avoid or [])[:2]:      # the worst validated failure-pattern(s) this shape matches
        lines.append(f"VALIDATED FAILURE-PATTERN: this birth is [{' + '.join(a['combo'])}] → "
                     f"failed {a['rate']*100:.0f}% of {a['n']} past cases (base {a['base_rate']*100:.0f}%, "
                     f"lift {a['lift']}x; split-half + cross-symbol + significant)")
    if arc:
        lines.append("RECENT STATE ARC (leading in):")
        for s in arc:
            tag = f"{s.state}" + (f"/{s.substate}" if s.substate else "")
            lines.append(f"   {s.ts:%H:%M}  {tag}")
    return "\n".join(lines)


def build_prompt(step, arc: list, avoid: list | None = None) -> tuple[str, str]:
    """The EXACT (system, user) the model — or the proxy — answers. Identical in both modes."""
    return SYSTEM_PROMPT, facts_block(step, arc, avoid)


def parse_verdict(text: str, mode: str) -> Verdict:
    """Parse the fixed JSON schema (robust to code fences / surrounding prose)."""
    raw = text or ""
    s, e = raw.find("{"), raw.rfind("}")
    obj = {}
    if s != -1 and e != -1 and e > s:
        try:
            obj = json.loads(raw[s:e + 1])
        except Exception:
            obj = {}
    why = obj.get("why", [])
    if isinstance(why, str):
        why = [why]
    return Verdict(state=str(obj.get("state", "")).upper(), conviction=str(obj.get("conviction", "")).lower(),
                   read=str(obj.get("read", "")), why=[str(w) for w in why],
                   bias=str(obj.get("bias", "")).lower(), unsure=bool(obj.get("unsure", False)),
                   mechanism=str(obj.get("mechanism", "")), consequence=str(obj.get("consequence", "")),
                   invalidation=str(obj.get("invalidation", "")), mode=mode, raw=raw)


class StateReasoner:
    """Drop-in reasoner. Configured LLM → calls it; no key → proxy mode (returns the prompt to answer,
    or parses a supplied proxy_response). Same contract either way."""

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        try:
            from agents.llm_client import LLMClient
            from config.settings import settings
            key = getattr(settings, "LLM_API_KEY", "") or ""
            if key:
                self._llm = LLMClient(provider=getattr(settings, "LLM_PROVIDER", "openai"),
                                      model=getattr(settings, "LLM_MODEL", None), api_key=key,
                                      base_url=getattr(settings, "LLM_BASE_URL", None) or None)
        except Exception:
            self._llm = None
        return self._llm

    def judge(self, step, arc: list, proxy_response: str | None = None, avoid: list | None = None) -> Verdict:
        system, user = build_prompt(step, arc, avoid)
        # an explicit proxy answer always wins (lets --apply render without spending API quota)
        if proxy_response is not None:
            return parse_verdict(proxy_response, mode="proxy")
        llm = self._get_llm()
        if llm is not None:
            import asyncio
            text = asyncio.run(llm.generate(prompt="", system_prompt=system, user_message=user,
                                            temperature=0.2, max_tokens=400, caller="state_reasoner"))
            return parse_verdict(text, mode="llm")
        v = Verdict(mode="pending")
        v.prompt = system + "\n\n=== FACTS ===\n" + user
        return v
