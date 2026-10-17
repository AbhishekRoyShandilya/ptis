"""
avt.templates — the SYSTEM strategy TEMPLATES registry. Unlike taught strategies (tag-signatures the user teaches),
templates are NON-TEACHABLE: the user SELECTS a template from the library and it instantiates as per-user SKOs wired
into their AVT. A template is FROZEN (its SPEC); the per-user instances are what the Capital Operator later tunes
(exit/sizing/filters) — never the template.

Public build: the registry ships EMPTY. The validated templates and their signal generators are private plugins;
the engine, dispatch and UI below run unchanged with taught strategies only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from avt.intent import Instrument


@dataclass
class SystemTemplate:
    id: str
    name: str
    one_line: str
    assets: str                              # human-readable asset scope
    preferred_instrument: Instrument
    signal_gen: str = "cascade"              # which AVT signal generator drives it
    spec_path: str = ""
    teachable: bool = False                  # system's own — cannot be taught
    fixed_universe: Optional[List[str]] = None  # pin this template to its OWN symbols, ignoring the global watchlist
    sides: tuple = ("long", "short")         # which directions this template instantiates
    trail_atr: float = 2.0                   # trailing-stop ATR multiple written into each SKO's exit_rules
    stop_atr: Optional[float] = None         # initial-stop ATR multiple (None = engine/template default)

    def skos(self, watchlist: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """The per-user SKOs (one per side) this template instantiates; the engine takes the side from
        exit_rules.direction."""
        wl = list(self.fixed_universe) if self.fixed_universe is not None else list(watchlist or [])
        out: List[Dict[str, Any]] = []
        for direction in self.sides:
            exit_rules = {"direction": direction, "trail_atr": self.trail_atr, "close_based_stops": False}
            if self.stop_atr is not None:
                exit_rules["stop_atr"] = self.stop_atr
            out.append({
                "id": f"{self.id.upper()}-{direction}",
                "short_code": f"{self.id.upper().replace('_', '')}_{direction[:1].upper()}",
                "name": f"{self.name} ({direction})",
                "template": self.id,
                "signal_gen": self.signal_gen,                       # dispatch marker → the matching signal generator
                "preferred_instrument": self.preferred_instrument.value,
                "exit_rules": exit_rules,
                "risk_parameters": {},
                "conviction_map": {"SKIP": [0.0, 0.3]},
                "watchlist": list(wl),
            })
        return out


TEMPLATES: Dict[str, SystemTemplate] = {}


def list_templates() -> List[Dict[str, Any]]:
    """Library the UI offers for one-click selection."""
    return [{"id": t.id, "name": t.name, "one_line": t.one_line, "assets": t.assets,
             "instrument": t.preferred_instrument.value, "teachable": t.teachable, "spec": t.spec_path}
            for t in TEMPLATES.values()]


def template_skos(template_id: str, watchlist: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    t = TEMPLATES.get(template_id)
    if t is None:
        raise KeyError(f"unknown system template: {template_id}")
    return t.skos(watchlist)


def is_cascade_sko(sko: Dict[str, Any]) -> bool:
    """True if this SKO is driven by the multi-timeframe template generator — the engine's dispatch key."""
    return (sko.get("signal_gen") == "cascade") or str(sko.get("id", "")).upper().startswith("CASCADE")


def is_breakdown_sko(sko: Dict[str, Any]) -> bool:
    """True if this SKO is driven by the breakdown template generator — the engine's dispatch key."""
    return (sko.get("signal_gen") == "breakdown") or str(sko.get("id", "")).upper().startswith("BREAKDOWN")


def preferred_instrument(sko: Dict[str, Any]) -> Instrument:
    val = sko.get("preferred_instrument", "future")
    try:
        return Instrument(val)
    except ValueError:
        return Instrument.FUTURE
