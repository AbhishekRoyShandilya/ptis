"""
intelligence.teaching_service — the model-pluggable STRATEGY-TEACHING orchestration (the thing that was
"Claude in chat"). The DETERMINISTIC SPINE — organs reason, the LLM only routes/phrases (project_brain/11):

- `ground_example`  — what the EYE saw at a moment the user POINTS to (symbol+time; charts-as-pointers,
   no pixels). Real data, not vision.
- `challenge_from_examples` — grounded gap-finding from the user's OWN examples (a tag in most TAKES not
   required → "you forgot ROOM"; a tag in a FAIL the takes lack → candidate EXCLUDE).
- `propose_refinements` — grounded conditional-edge search on the library (the engine that found
   +ROOM_CLEAR = +3→+15). PROPOSE only; the user confirms.
- `validate_with_examples` — `library_stats` PLUS the matched episodes to VERIFY against ("is this your
   setup?"), not just a number.

Reuses cognition_service (eye) + strategy_backtest/taught_strategies (validator) + episodic (library) —
UNIFIES the strategy paths, adds no 4th. The LLM front (NL → an initial signature) plugs on top of this.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict


import json
import re


def _get_llm():
    from config.settings import settings
    from agents.llm_client import LLMClient
    return LLMClient(provider=getattr(settings, "LLM_PROVIDER", "openai"),
                     model=getattr(settings, "LLM_MODEL", None),
                     api_key=getattr(settings, "LLM_API_KEY", None) or None)


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# Each tag → phrases in the user's text that JUSTIFY it. A tag with no textual evidence is NOT faithful
# and is dropped from the signature (the LLM may hallucinate; the eye/library SUGGESTS additions later).
_EVIDENCE = {
    "STRONG_EFFORT": ("breakout", "break out", "break above", "momentum", "strong", "volume", "body", "spike", "fast"),
    "WEAK_EFFORT": ("weak", "no momentum", "no body", "no volume", "limp", "poke"),
    "ROOM_CLEAR": ("clear room", "clear air", "open air", "space above", "room above", "no resistance", "no wall"),
    "ROOM_WALL": ("into resistance", "overhead resistance", "into a wall", "wall above", "near resistance"),
    "WITH_TREND": ("uptrend", "above 50", "above the 50", "supertrend green", "with trend", "with the trend", "above ema", "above dma", "green supertrend"),
    "COUNTER_TREND": ("counter trend", "counter-trend", "against trend", "against the trend"),
    "REGIME_BULL": ("bull regime", "bullish regime", "1h bull"),
    "REGIME_BEAR": ("bear regime", "bearish regime", "1h bear"),
    "STRUCT_RANGE": ("consolidat", "range", "tight", "base", "sideways", "coil"),
    "STRUCT_HH_HL": ("higher high", "hh hl", "higher highs"),
    "STRUCT_LH_LL": ("lower high", "lh ll", "lower lows"),
    "AT_OPEN": ("at open", "at the open", "opening", "first hour", "9:15", "9.15"),
    "PHASE_OPEN": ("at open", "at the open", "opening", "first hour"),
    "PHASE_AFTERNOON": ("afternoon",),
    "GAP_UP": ("gap up", "gapped up"), "GAP_DOWN": ("gap down", "gapped down"), "FLAT_OPEN": ("flat open",),
    "ABOVE_VWAP": ("above vwap", "over vwap"), "BELOW_VWAP": ("below vwap", "under vwap"),
}
# CONSERVATIVE require-set: only unambiguous, hard conditions get auto-added (catches an LLM miss like
# WITH_TREND). Soft preferences ("best at the open") are NOT auto-required — they surface as refinements.
_RULE_REQUIRE = ("STRONG_EFFORT", "ROOM_CLEAR", "WITH_TREND", "STRUCT_RANGE")


def _evidenced(description: str) -> set[str]:
    d = (description or "").lower()
    return {tag for tag, kws in _EVIDENCE.items() if any(k in d for k in kws)}


def _rule_based_signature(description: str) -> dict:
    """Deterministic evidence-based mapper: the fallback when the LLM is down AND the faithfulness guard
    on the LLM's output. Maps only tags with textual evidence; require = the conservative hard subset."""
    ev = _evidenced(description)
    d = (description or "").lower()
    req = [t for t in _RULE_REQUIRE if t in ev]
    exc = ["PHASE_AFTERNOON"] if any(w in d for w in ("avoid afternoon", "not afternoon", "skip afternoon")) else []
    direction = "down" if any(w in d for w in ("short", "sell side", "sell-side", "downside")) else "up"
    gaps = []
    if any(w in d for w in ("50 dma", "50dma", "50 ema", "moving average", "supertrend")):
        gaps.append("50 DMA / Supertrend gates (no tag yet — vocab extension ABOVE_EMA50, ST_GREEN)")
    return {"direction": direction, "require": req, "exclude": exc, "execution_notes": [], "gaps": gaps, "source": "rules"}


def propose_signature_from_text(description: str) -> dict:
    """Model-pluggable FRONT (Llama/Claude via config): map an NL strategy description (any language) →
    an initial signature over the eye's vocabulary. Returns {direction, require, exclude, execution_notes,
    gaps, source}. The SPINE then grounds/challenges/validates whatever this proposes — so a thin or wrong
    LLM proposal gets corrected by the organs, never trusted. Falls back to rules if the LLM is down."""
    from intelligence.taught_strategies import TAG_VOCAB, ALL_TAGS
    vocab = "; ".join(f"{g}=[{', '.join(ts)}]" for g, ts in TAG_VOCAB.items())
    system = (
        "Convert a trader's strategy description into a SIGNATURE over a FIXED tag vocabulary. "
        "Output STRICT JSON only: {\"direction\":\"up|down|both\",\"require\":[tags],\"exclude\":[tags],"
        "\"execution_notes\":[short strings],\"gaps\":[short strings]}. "
        "Use ONLY tags from this vocabulary: " + vocab + ". "
        "require=conditions that MUST hold; exclude=conditions to AVOID; "
        "execution_notes=entry/stop/exit rules that are NOT recognition conditions; "
        "gaps=anything the trader needs that NO tag can express (specific MAs, volume %, multi-day patterns).\n"
        "CRITICAL — be FAITHFUL: map ONLY conditions the trader EXPLICITLY states. NEVER add or infer a "
        "condition they did not mention (e.g. do NOT add VWAP, gap, structure, regime, or any tag they "
        "didn't say). When unsure whether a tag applies, LEAVE IT OUT — under-mapping is correct; "
        "improvements are suggested separately, never baked in. Output only tags clearly implied by their words."
    )
    try:
        import asyncio
        text = asyncio.run(_get_llm().generate(prompt="", system_prompt=system, user_message=description,
                                               temperature=0.1, max_tokens=500, caller="teach_signature"))
        d = _extract_json(text)
        if d:
            # FAITHFULNESS GUARD: keep only LLM tags that are real AND have textual evidence (drop
            # hallucinations like ABOVE_VWAP); union with the conservative rule require (catches an LLM
            # miss like WITH_TREND). The eye/library SUGGESTS additions later — never the LLM's hunch.
            ev = _evidenced(description)
            rule = _rule_based_signature(description)
            req = sorted(set(t for t in d.get("require", []) if t in ALL_TAGS and t in ev) | set(rule["require"]))
            exc = sorted(set(t for t in d.get("exclude", []) if t in ALL_TAGS and t in ev) | set(rule["exclude"]))
            unknown = [t for t in (d.get("require", []) + d.get("exclude", [])) if t not in ALL_TAGS]
            gaps = list(d.get("gaps", [])) + rule["gaps"] + [f"(model proposed unknown tag: {b})" for b in unknown]
            seen = set(); gaps = [g for g in gaps if not (g in seen or seen.add(g))]
            return {"direction": d.get("direction", rule["direction"]), "require": req, "exclude": exc,
                    "execution_notes": d.get("execution_notes", []), "gaps": gaps, "source": "llm"}
    except Exception:
        pass
    return _rule_based_signature(description)


def _symbol_known(symbol: str) -> bool:
    try:
        import pandas as pd
        from marketdata.candle_store import get_candle_store
        now = pd.Timestamp.now()
        df = get_candle_store().get(symbol, "1d", now.normalize() - pd.Timedelta(days=500), now)
        return df is not None and not df.empty
    except Exception:
        return False


def _data_present(symbol: str, when, interval: str = "15m") -> bool:
    try:
        import pandas as pd
        from marketdata.candle_store import get_candle_store
        w = pd.Timestamp(when)
        df = get_candle_store().get(symbol, interval, w - pd.Timedelta(days=3), w + pd.Timedelta(days=1))
        if df is None or df.empty:
            return False
        ts = pd.to_datetime(df["ts"])
        return bool((ts <= w).any() and (ts - w).abs().min() <= pd.Timedelta(hours=8))
    except Exception:
        return False


def _looks_like_question(text: str) -> bool:
    """A QUESTION about the strategy/stats — answered, not captured. Deterministic so a question is never
    mistaken for a strategy description."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    starts = ("what", "how", "why", "explain", "can you", "could you", "would you", "does it", "do these",
              "is it", "are these", "is this", "tell me", "which", "what's", "whats", "i don't understand",
              "i dont understand")
    return t.startswith(starts)


def extract_message(message: str) -> dict:
    """Route a natural chat message: a QUESTION (about the strategy/stats), a strategy DESCRIPTION, a trade
    EXAMPLE (symbol+time), or an ASK when it's ambiguous. The LLM extracts; deterministic guards VALIDATE
    (symbol exists, time parses, data present) — on any gap it returns an `ask` instead of guessing."""
    import pandas as pd
    if _looks_like_question(message):
        return {"kind": "question"}
    kind, symbol, date, time, label = "description", None, None, None, "take"
    sysp = ("Classify a trader's chat message. Output STRICT JSON only: "
            "{\"kind\":\"example|description\",\"symbol\":str|null,\"date\":\"YYYY-MM-DD\"|null,"
            "\"time\":\"HH:MM\"|null,\"label\":\"take|skip|fail\"}. "
            "kind=example ONLY if it points to a SPECIFIC past setup (a stock AND when). Else description. "
            "label: take=a trade they took, skip=one they avoided, fail=one that lost. null if absent.")
    try:
        import asyncio
        txt = asyncio.run(_get_llm().generate(prompt="", system_prompt=sysp, user_message=message,
                                              temperature=0.0, max_tokens=150, caller="teach_extract"))
        d = _extract_json(txt) or {}
        kind = d.get("kind", "description")
        symbol = ((d.get("symbol") or "").upper().strip() or None)
        date, time = d.get("date"), d.get("time")
        label = (d.get("label") or "take")
    except Exception:
        pass

    if kind != "example":
        return {"kind": "description"}
    if not symbol:
        return {"kind": "ask", "ask": "Which stock is that example?"}
    if not _symbol_known(symbol):
        return {"kind": "ask", "ask": f"I don't recognise \"{symbol}\" — what's the symbol?"}
    if not date:
        return {"kind": "ask", "ask": f"Which date for the {symbol} example?"}
    if not time:
        return {"kind": "ask", "ask": f"What time on {date} (e.g. 11:15)?"}
    when = f"{date} {time}"
    try:
        pd.Timestamp(when)
    except Exception:
        return {"kind": "ask", "ask": f"I couldn't read \"{when}\" — try a date/time like 2026-04-22 11:15?"}
    if not _data_present(symbol, when):
        return {"kind": "ask", "ask": f"I don't have {symbol} bars around {when} — point me to a moment I have data for?"}
    return {"kind": "example", "symbol": symbol, "when": when,
            "label": label if label in ("take", "skip", "fail") else "take"}


@dataclass
class TeachingResult:
    description: str
    direction: str
    require: list
    exclude: list
    execution_notes: list
    gaps: list
    source: str
    grounded_examples: list = field(default_factory=list)
    challenge: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    refinements: list = field(default_factory=list)
    explanation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# Plain-English meaning of each tag, written as a CLAUSE that reads after "the setup …". This is the
# faithful spine of the explanation — deterministic, so the stored meaning can never drift from the tags.
TAG_MEANINGS = {
    "WITH_TREND": "is aligned with the prevailing trend", "COUNTER_TREND": "goes against the prevailing trend",
    "NO_TREND": "has no clear trend",
    "ROOM_CLEAR": "has clear room ahead with no overhead resistance", "ROOM_SOME": "has some room ahead",
    "ROOM_TIGHT": "has limited room ahead", "ROOM_WALL": "runs into overhead resistance (a wall)",
    "STRUCT_HH_HL": "sits in an up-structure (higher highs / higher lows)",
    "STRUCT_LH_LL": "sits in a down-structure (lower highs / lower lows)",
    "STRUCT_RANGE": "follows a tight consolidation / base",
    "REGIME_BULL": "is in a bullish regime", "REGIME_BEAR": "is in a bearish regime",
    "GAP_UP": "opens on a gap up", "GAP_DOWN": "opens on a gap down", "FLAT_OPEN": "opens flat",
    "AT_OPEN": "triggers right at the open", "ABOVE_VWAP": "is trading above VWAP", "BELOW_VWAP": "is trading below VWAP",
    "STRONG_EFFORT": "breaks out with decisive effort (volume, momentum, strong body)",
    "WEAK_EFFORT": "moves with weak, low-effort participation",
    "PHASE_OPEN": "is in the opening phase", "PHASE_MORNING": "is in the morning session",
    "PHASE_MIDDAY": "is around midday", "PHASE_AFTERNOON": "is in the afternoon", "PHASE_LATE": "is late in the session",
}


def _join_clauses(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]


def explain_signature(direction: str, require=None, exclude=None, description: str = "") -> dict:
    """A faithful, human-readable explanation of a stored strategy: the CONCEPT (one plain sentence), the
    user's OWN WORDS, the LOGIC (each condition's meaning), and the EVIDENCE (validation in words). Built
    deterministically from the tag glossary — the LLM is not in this path, so it cannot drift."""
    require, exclude = sorted(set(require or [])), sorted(set(exclude or []))
    verb = {"up": "Go long", "down": "Go short"}.get(direction, "Trade")
    req_clauses = [TAG_MEANINGS.get(t, t.lower().replace("_", " ")) for t in require]
    concept = (f"{verb} when the setup " + _join_clauses(req_clauses) + ".") if req_clauses else f"{verb} — no conditions set yet."
    if exclude:
        exc_clauses = [TAG_MEANINGS.get(t, t.lower().replace("_", " ")) for t in exclude]
        concept += " Skip it when it " + _join_clauses(exc_clauses) + "."
    logic = [{"tag": t, "means": TAG_MEANINGS.get(t, t.lower().replace("_", " "))} for t in require]
    exclude_logic = [{"tag": t, "means": TAG_MEANINGS.get(t, t.lower().replace("_", " "))} for t in exclude]
    evidence = ""
    try:
        from intelligence.strategy_backtest import Strategy, library_stats
        st = library_stats(Strategy("(x)", direction, set(require), set(exclude))).to_dict()
        if st.get("n"):
            base = st.get("base_ran_rate") or 0
            mult = (st["ran_rate"] / base) if base else 0
            evidence = (f"On {st['n']} matched setups this ran {round(100 * st['ran_rate'])}% of the time "
                        f"vs {round(100 * base)}% base"
                        + (f" ({mult:.1f}× the base rate)" if mult else "")
                        + (" — validated." if st.get("validated") else " — not yet validated."))
    except Exception:
        pass
    return {"concept": concept, "your_words": (description or "").strip(),
            "logic": logic, "exclude_logic": exclude_logic, "evidence": evidence}


def answer_question(question: str, direction: str = "up", require=None, exclude=None) -> str:
    """Answer a trader's question about the strategy/stats — GROUNDED in the real numbers and HONEST about
    what they mean: these are RECOGNITION stats (did the move run), NOT profit/loss. The LLM writes the
    prose but is fed only the true facts + the honest framing; never invents a P&L number."""
    exp = explain_signature(direction, require, exclude)
    st = {}
    try:
        from intelligence.strategy_backtest import Strategy, library_stats
        st = library_stats(Strategy("(x)", direction, set(require or []), set(exclude or []))).to_dict()
    except Exception:
        pass
    ran, base, n = (st.get("ran_rate") or 0), (st.get("base_ran_rate") or 0), st.get("n", "?")
    mult = (ran / base) if base else 0
    facts = (f"Strategy: {exp['concept']}\n"
             f"Of {n} matched setups, the move RAN (extended favourably) {round(100 * ran)}% of the time, "
             f"vs a {round(100 * base)}% base rate ({mult:.1f}x base; "
             f"{'validated' if st.get('validated') else 'not yet validated'}).")
    sysp = ("You explain a taught trading strategy's stats to the trader, plainly and HONESTLY. "
            "CRITICAL: these numbers measure RECOGNITION, NOT profit/loss. 'RAN' = the move extended "
            "favourably after the setup; 'FAILED' = it didn't. The percentage is how OFTEN a real move "
            "follows this setup vs the base rate — it is NOT a profit percentage and NOT net P&L, and there "
            "is no overall loss/gain figure here. Actual profit depends on entry, exit, and position size "
            "(execution), which these stats do NOT include — that is measured separately by the autonomous "
            "trader (AVT). NEVER invent a profit number. Use ONLY the facts given. 2-4 short sentences.")
    try:
        import asyncio
        out = asyncio.run(_get_llm().generate(prompt="", system_prompt=sysp,
                          user_message=f"FACTS:\n{facts}\n\nQUESTION: {question}",
                          temperature=0.2, max_tokens=220, caller="teach_ask"))
        if out and out.strip():
            return out.strip()
    except Exception:
        pass
    return (f"These are recognition stats, not profit. Of {n} times this setup appeared, the move ran "
            f"{round(100 * ran)}% of the time vs {round(100 * base)}% normally — so it's ~{mult:.1f}x better at "
            f"spotting a real move. It is NOT a profit %, and there's no overall loss figure here; actual "
            f"profit/loss depends on your entry, exit and size (that's what AVT measures).")


def review_signature(direction: str, require=None, exclude=None) -> TeachingResult:
    """Review a STORED signature (no NL) — what the system currently understands about this strategy:
    its validation + matched examples + refinements. Used on REOPEN so a strategy shows its understanding
    (the structured state), even one taught before conversations existed."""
    from intelligence.strategy_backtest import Strategy
    require, exclude = sorted(set(require or [])), sorted(set(exclude or []))
    sig = Strategy("(stored)", direction, set(require), set(exclude))
    res = TeachingResult(description="", direction=direction, require=require, exclude=exclude,
                         execution_notes=[], gaps=[], source="stored")
    res.explanation = explain_signature(direction, require, exclude)
    if sig.require or sig.exclude:
        res.validation = validate_with_examples(sig)
        res.refinements = propose_refinements(sig)
    return res


def bootstrap(description: str, examples: list | None = None, interval: str = "15m") -> TeachingResult:
    """End-to-end teaching pass: NL → propose signature (LLM) → ground pointed examples (eye) → grounded
    challenge → validate with matched examples → propose refinements. Nothing is stored — this is the
    'review' a user verifies before confirming."""
    from intelligence.strategy_backtest import Strategy
    prop = propose_signature_from_text(description)
    sig = Strategy("(draft)", prop["direction"], set(prop["require"]), set(prop["exclude"]))
    grounded = [ground_example(e["symbol"], e["when"], e.get("interval", interval), e.get("label", "take"))
                for e in (examples or [])]
    res = TeachingResult(description=description, direction=prop["direction"], require=prop["require"],
                         exclude=prop["exclude"], execution_notes=prop["execution_notes"],
                         gaps=prop["gaps"], source=prop["source"],
                         grounded_examples=[g.to_dict() for g in grounded],
                         challenge=challenge_from_examples(grounded, sig.require))
    res.explanation = explain_signature(prop["direction"], prop["require"], prop["exclude"], description)
    if sig.require or sig.exclude:
        res.validation = validate_with_examples(sig)
        res.refinements = propose_refinements(sig)
    return res


@dataclass
class GroundedExample:
    symbol: str
    when: str
    interval: str
    label: str                       # take / skip / fail
    ok: bool = True
    state: str = ""
    substate: str = ""
    direction: str = ""
    stage: str = ""
    conviction: str = ""
    tags: list = field(default_factory=list)
    mechanism: str = ""
    move_story: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def ground_example(symbol: str, when, interval: str = "15m", label: str = "take") -> GroundedExample:
    """Ground a pointed example on the EYE — what it saw at (symbol, time), on real data."""
    from intelligence.cognition_service import read
    g = GroundedExample(symbol=str(symbol).upper(), when=str(when), interval=interval, label=label)
    try:
        r = read(symbol, interval=interval, as_of=when)
    except Exception as e:
        g.ok = False; g.note = f"eye read failed: {e}"; return g
    if not r.ok:
        g.ok = False; g.note = r.note or "no read"; return g
    g.state, g.substate, g.direction = r.state, r.substate, r.direction
    g.stage, g.conviction, g.tags = r.stage, r.conviction, list(r.tags)
    g.mechanism, g.move_story = r.mechanism, r.move_story
    return g


def challenge_from_examples(examples: list[GroundedExample], require_tags) -> list[str]:
    """Surface gaps from the user's OWN grounded examples — never invents, only reads the eye's tags."""
    req = set(require_tags)
    # only examples where the eye actually saw a MOVE (ok + has tags + a direction); a moment the user
    # pointed to that reads as plain balance has no signature to learn from and would only add noise.
    usable = [e for e in examples if e.ok and e.tags and e.direction]
    skipped = [e for e in examples if e not in usable]
    takes = [e for e in usable if e.label == "take"]
    bad = [e for e in usable if e.label in ("fail", "skip")]
    take_tags = set(t for e in takes for t in e.tags)
    out, seen = [], set()

    def add(m):
        if m not in seen:
            seen.add(m); out.append(m)

    if takes:
        n = len(takes)
        for tag, c in Counter(t for e in takes for t in e.tags).most_common():
            if tag not in req and c >= max(2, round(n * 0.6)):
                add(f"ADD? {c}/{n} of your takes show {tag}, but your rules don't require it — part of the setup?")
    for e in bad:
        for tag in e.tags:
            if tag not in req and tag not in take_tags:
                add(f"EXCLUDE? your {e.label} ({e.symbol} {e.when}) had {tag} — your takes don't. Avoid it?")
    for e in skipped:
        add(f"NOTE: {e.symbol} {e.when} read as {e.state or 'no move'} — point me to the exact breakout candle so I can learn from it.")
    return out


def propose_refinements(signature, candidates=None, min_n: int = 20, top: int = 5) -> list[dict]:
    """Grounded conditional-edge search: for each candidate ADD-tag, measure the validated-lift delta on
    the library. PROPOSE only (sorted by improvement); the user confirms — never auto-applies."""
    from intelligence.strategy_backtest import Strategy, library_stats
    from intelligence.taught_strategies import ALL_TAGS
    base = library_stats(signature)
    cands = candidates or sorted(ALL_TAGS - set(signature.require) - set(signature.exclude))
    out = []
    for tag in cands:
        s2 = Strategy(f"{signature.name}+{tag}", signature.direction,
                      set(signature.require) | {tag}, set(signature.exclude))
        st = library_stats(s2)
        if st.n >= min_n:
            out.append({"add": tag, "n": st.n, "rate": round(st.ran_rate, 3),
                        "lift": round(st.lift, 3), "lift_delta": round(st.lift - base.lift, 3),
                        "validated": st.validated})
    return sorted(out, key=lambda x: -x["lift_delta"])[:top]


def validate_with_examples(signature, k: int = 10) -> dict:
    """`library_stats` + a sample of the MATCHED episodes to verify against ('is this your setup?')."""
    from intelligence.strategy_backtest import library_stats
    from intelligence.episodic import EpisodeStore
    st = library_stats(signature)
    eps = [e for e in EpisodeStore().load() if e.peak_ext == e.peak_ext and signature.matches(e)]
    # spread the verify set across symbols so it's a fair check, not 10 of one stock
    by_sym, samples = {}, []
    for e in eps:
        by_sym.setdefault(e.symbol, []).append(e)
    for sym in sorted(by_sym):
        e = by_sym[sym][0]
        samples.append({"symbol": e.symbol, "date": e.date, "time": e.ts, "dir": e.direction,
                        "peak_ext": e.peak_ext, "outcome": e.outcome})
        if len(samples) >= k:
            break
    return {"stats": st.to_dict(), "n_matched": len(eps), "verify_examples": samples}
