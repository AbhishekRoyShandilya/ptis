"""
intelligence.taught_strategies — persist the user's TAUGHT strategies and serve them with validated
stats.

A strategy the user "teaches" = a SIGNATURE over the eye's validated vocabulary: a direction plus the
tags that MUST be present (`require`) and that must be ABSENT (`exclude`). The Unified Partner consumes
these so it points only at setups matching an edge the user actually trades; `strategy_backtest` scores
each signature against the episodic library — instant recognition stats + both-halves validation, so a
taught idea is graded HONESTLY (validated edge vs noise) the moment it's added.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from intelligence.strategy_backtest import Strategy, library_stats

logger = logging.getLogger(__name__)

_STORE = Path(__file__).resolve().parent.parent / "config" / "taught_strategies.json"

# The eye's validated tag vocabulary — the building blocks a strategy can be made of (the UI offers
# these; add_strategy validates against them). Mirrors episodic.signature_from_birth.
TAG_VOCAB: dict[str, list[str]] = {
    "trend": ["WITH_TREND", "COUNTER_TREND", "NO_TREND"],
    "room": ["ROOM_CLEAR", "ROOM_SOME", "ROOM_TIGHT", "ROOM_WALL"],
    "structure": ["STRUCT_HH_HL", "STRUCT_LH_LL", "STRUCT_RANGE"],
    "regime": ["REGIME_BULL", "REGIME_BEAR"],
    "open": ["GAP_UP", "GAP_DOWN", "FLAT_OPEN", "AT_OPEN"],
    "vwap": ["ABOVE_VWAP", "BELOW_VWAP"],
    "effort": ["STRONG_EFFORT", "WEAK_EFFORT"],
    "phase": ["PHASE_OPEN", "PHASE_MORNING", "PHASE_MIDDAY", "PHASE_AFTERNOON", "PHASE_LATE"],
}
ALL_TAGS: set[str] = {t for group in TAG_VOCAB.values() for t in group}

# Inert EXAMPLE templates the UI may offer as one-click starting points. These are NOT taught and are
# NEVER auto-stored — strategy teaching is user-driven through the UI. The store starts EMPTY.
EXAMPLE_TEMPLATES = [
    {"name": "Opening drive", "direction": "up", "require": ["PHASE_OPEN"], "exclude": []},
    {"name": "Clear-air break (strong)", "direction": "both",
     "require": ["ROOM_CLEAR", "STRONG_EFFORT"], "exclude": []},
]


def _from_dict(d: dict) -> Strategy:
    return Strategy(name=d["name"], direction=d.get("direction", "both"),
                    require=set(d.get("require", [])), exclude=set(d.get("exclude", [])))


def _to_dict(s: Strategy) -> dict:
    return {"name": s.name, "direction": s.direction,
            "require": sorted(s.require), "exclude": sorted(s.exclude)}


def _load_raw() -> list[dict]:
    """The full stored entries — signature fields PLUS the teaching session (description/messages/
    examples/versions). Signature consumers read the signature keys; the session keys ride along."""
    if not _STORE.exists():
        return []
    try:
        return json.loads(_STORE.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("failed to load taught strategies")
        return []


def _save_raw(entries: list[dict]) -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    _STORE.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _upsert(name: str, **fields) -> dict:
    """Upsert an entry by name, PRESERVING existing extra fields (the session data)."""
    raw = _load_raw()
    idx = next((i for i, d in enumerate(raw) if d.get("name", "").lower() == name.lower()), None)
    if idx is None:
        entry = {"name": name, **fields}
        raw.append(entry)
    else:
        entry = {**raw[idx], **fields, "name": raw[idx].get("name", name)}
        raw[idx] = entry
    _save_raw(raw)
    return entry


def load_strategies() -> list[Strategy]:
    """All strategies the USER has taught. Empty until the user teaches one (nothing pre-seeded)."""
    return [_from_dict(d) for d in _load_raw()]


def add_strategy(name: str, direction: str = "both", require=None, exclude=None) -> Strategy:
    """Teach a strategy signature. Validates tags + direction, upserts by name (preserving session data)."""
    name = (name or "").strip()
    if not name:
        raise ValueError("name is required")
    require, exclude = set(require or []), set(exclude or [])
    bad = (require | exclude) - ALL_TAGS
    if bad:
        raise ValueError(f"unknown tags: {sorted(bad)}")
    if direction not in ("up", "down", "both"):
        raise ValueError("direction must be one of up/down/both")
    _upsert(name, direction=direction, require=sorted(require), exclude=sorted(exclude))
    return Strategy(name=name, direction=direction, require=require, exclude=exclude)


def remove_strategy(name: str) -> bool:
    raw = _load_raw()
    after = [d for d in raw if d.get("name", "").lower() != (name or "").lower()]
    _save_raw(after)
    return len(after) < len(raw)


def strategies_with_stats() -> list[dict]:
    """Every taught strategy + its validated recognition stats + session metadata — what the UI shows."""
    out = []
    for d in _load_raw():
        s = _from_dict(d)
        out.append({**_to_dict(s), "stats": library_stats(s).to_dict(),
                    "versions": len(d.get("versions", []) or []), "updated": d.get("updated", "")})
    return out


# ── teaching SESSION persistence (the chat-history-per-strategy workspace) ──

def _now() -> str:
    import datetime as _dt
    return _dt.datetime.now().isoformat(timespec="seconds")


def save_session(name: str, *, direction: str, require=None, exclude=None, description: str = "",
                 messages=None, examples=None, thread: str = "teaching") -> dict:
    """Persist a strategy's SESSION: the signature (so partner/library read it) PLUS the conversation, the
    grounded examples, and a VERSIONED history (new version appended when the signature changes). The
    conversation is split into two threads — `teaching` (refine recognition) and `analysis` (money/
    hypotheses) — so they never mix. `messages` writes the thread named by `thread`."""
    name = (name or "").strip()
    if not name:
        raise ValueError("name is required")
    require, exclude = sorted(set(require or [])), sorted(set(exclude or []))
    bad = (set(require) | set(exclude)) - ALL_TAGS
    if bad:
        raise ValueError(f"unknown tags: {sorted(bad)}")
    existing = load_session(name) or {}
    now = _now()
    versions = existing.get("versions", []) or []
    if existing.get("require") != require or existing.get("exclude") != exclude:
        versions = versions + [{"v": len(versions) + 1, "require": require, "exclude": exclude, "ts": now}]
    msg_key = "analysis_messages" if thread == "analysis" else "messages"
    fields = {"direction": direction, "require": require, "exclude": exclude,
              "description": description or existing.get("description", ""),
              "examples": examples if examples is not None else existing.get("examples", []),
              "versions": versions, "updated": now, "created": existing.get("created", now)}
    fields[msg_key] = messages if messages is not None else existing.get(msg_key, [])
    return _upsert(name, **fields)


def save_performance(name: str, ledger: dict) -> dict:
    """Store the latest P&L backtest ledger on the strategy (the Analysis side's money record)."""
    return _upsert((name or "").strip(), performance={**(ledger or {}), "updated": _now()})


def add_hypothesis(name: str, text: str, result: str = "", **meta) -> dict:
    """Append a tested hypothesis to the strategy's analysis log ('does adding VWAP help? → +0.3R').
    `meta` carries structured fields (op/tags/baseline/status) so an applied refinement can be evaluated."""
    existing = load_session(name) or {}
    hyps = (existing.get("hypotheses", []) or []) + [{"text": text, "result": result, "ts": _now(), **meta}]
    return _upsert((name or "").strip(), hypotheses=hyps)


def save_hypotheses(name: str, hyps: list) -> dict:
    """Replace the hypothesis log (used to resolve a pending refinement with its verdict)."""
    return _upsert((name or "").strip(), hypotheses=hyps)


def load_session(name: str) -> dict | None:
    return next((d for d in _load_raw() if d.get("name", "").lower() == (name or "").lower()), None)


def list_sessions() -> list[dict]:
    """The chat-history list (= My Strategies), with session metadata — for the sidebar."""
    return [{"name": d.get("name", ""), "direction": d.get("direction", "up"),
             "require": d.get("require", []), "exclude": d.get("exclude", []),
             "versions": len(d.get("versions", []) or []), "examples": len(d.get("examples", []) or []),
             "total_R": (d.get("performance") or {}).get("total_R"), "updated": d.get("updated", "")}
            for d in _load_raw()]
