"""
avt.template_store — per-user record of which SYSTEM templates a user has ACTIVATED. System templates
(`avt.templates`) are the system's own strategies — not taught, not in the taught-strategy DB (they have no
scoring_config), so their selection is stored here, keyed by trader_id. On AVT start/reload, the activated
templates' SKOs are injected alongside the user's taught SKOs and routed by the DispatchingSignalGenerator.
Simple JSON store (single-user test today; per-trader_id keeps it multi-user correct).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from avt.templates import TEMPLATES, template_skos

_STORE = Path("config/activated_templates.json")   # {trader_id: [template_id, ...]}


def _load() -> Dict[str, List[str]]:
    if not _STORE.exists():
        return {}
    try:
        return json.loads(_STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(d: Dict[str, List[str]]) -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    _STORE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def activated(trader_id: Any) -> List[str]:
    return list(_load().get(str(trader_id), []))


def activate(trader_id: Any, template_id: str) -> List[str]:
    if template_id not in TEMPLATES:
        raise KeyError(f"unknown system template: {template_id}")
    d = _load(); key = str(trader_id)
    cur = d.get(key, [])
    if template_id not in cur:
        cur.append(template_id)
    d[key] = cur; _save(d)
    return cur


def deactivate(trader_id: Any, template_id: str) -> List[str]:
    d = _load(); key = str(trader_id)
    d[key] = [t for t in d.get(key, []) if t != template_id]
    _save(d)
    return d[key]


def activated_skos(trader_id: Any, watchlist: List[str] | None = None) -> List[Dict[str, Any]]:
    """The SKOs (long+short per template) for this user's activated system templates — injected into their AVT."""
    out: List[Dict[str, Any]] = []
    for tid in activated(trader_id):
        try:
            out.extend(template_skos(tid, watchlist))
        except KeyError:
            continue
    return out
