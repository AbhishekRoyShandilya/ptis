"""
intelligence.working_memory — SCOPED working memory (the cure for memory-scoping bugs).

A reasoner is only as good as what it REMEMBERS at the moment of judgment. The 09:20 failure was not
bad reasoning — it was a broken memory chain: the "this move was born counter-trend" fact was set at
the birth bar and lost on the next, so the model judged a counter-trend continuation as a healthy one.

The cure is NOT "pass everything forward" — that makes a PROMPT-ASSEMBLER, burying the signal under
stale facts (a volume spike two hours ago is noise now). The cure is a HIERARCHY OF MEMORY where each
fact lives exactly as long as it stays RELEVANT, and the prompt is a RENDER of what is currently live:

  BAR     — this candle only: price, ATR, participation, VWAP. Refreshed every bar.
  MOVE    — from a move's birth until it dies: direction, birth level, with/counter-trend, the room it
            was born into, extension-from-birth, "already exhausted once". Cleared on structural release.
  SESSION — all session: 1h trend, intraday structure, prior-day levels, the day's character.
  LIBRARY — cross-session process-arc recall ("have I seen this before?"). Wired later via episodic
            memory — SAME abstraction, wider scope.

Relevance is enforced by LIFETIME: counter-trend persists for the whole move (09:20 fixed
structurally, not hand-carried), while bar facts expire on their own.
"""
from __future__ import annotations

from dataclasses import dataclass

BAR, MOVE, SESSION, BIAS, LIBRARY = "BAR", "MOVE", "SESSION", "BIAS", "LIBRARY"
_ORDER = {BIAS: 0, SESSION: 1, MOVE: 2, BAR: 3, LIBRARY: 4}


@dataclass
class _Item:
    scope: str
    text: str
    order: int          # insertion order within a scope (stable rendering)


class WorkingMemory:
    """Keyed, scoped facts. `set` updates-in-place by key (so 'ext' refreshes, not duplicates);
    `drop_scope` expires a whole tier (BAR each bar, MOVE on release). `snapshot` renders what is
    currently live, grouped by scope — that render IS the reasoner's fact context."""

    def __init__(self):
        self._items: dict[str, _Item] = {}
        self._n = 0

    def set(self, key: str, text: str, scope: str) -> None:
        prev = self._items.get(key)
        order = prev.order if prev is not None else self._n
        if prev is None:
            self._n += 1
        self._items[key] = _Item(scope=scope, text=text, order=order)

    def setdefault(self, key: str, text: str, scope: str) -> None:
        """Set only if absent (for facts that should be recorded once and then persist, e.g. the bar a
        move first showed exhaustion)."""
        if key not in self._items:
            self.set(key, text, scope)

    def has(self, key: str) -> bool:
        return key in self._items

    def drop_scope(self, scope: str) -> None:
        self._items = {k: v for k, v in self._items.items() if v.scope != scope}

    def snapshot(self) -> dict[str, list[str]]:
        """{scope: [fact, ...]} of currently-live memory, scopes in SESSION→MOVE→BAR order."""
        out: dict[str, list[str]] = {}
        for it in sorted(self._items.values(), key=lambda x: (_ORDER.get(x.scope, 9), x.order)):
            out.setdefault(it.scope, []).append(it.text)
        return out


_SCOPE_LABEL = {
    BIAS:    "DIRECTIONAL BIAS (multi-TF read — daily/1h/15m/5m; weigh the hierarchy, do not just sum)",
    SESSION: "SESSION MEMORY (holds all session)",
    MOVE:    "MOVE MEMORY (holds until this move dies)",
    BAR:     "NOW (this bar only)",
    LIBRARY: "LIBRARY (past analogues)",
}


def render_snapshot(snap: dict[str, list[str]]) -> str:
    """Render a snapshot as the scoped memory block the reasoner reads."""
    lines: list[str] = []
    for scope in (BIAS, SESSION, MOVE, BAR, LIBRARY):
        facts = snap.get(scope)
        if facts:
            lines.append(f"{_SCOPE_LABEL[scope]}:")
            lines.extend(f"   · {f}" for f in facts)
    return "\n".join(lines)
