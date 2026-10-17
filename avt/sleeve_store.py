"""
avt.sleeve_store — per-user CAPITAL SLEEVE configuration (B.1). A user decides how to split their account across
instruments (e.g. ₹2L futures / ₹50k options / rest equity) — their money, their call — and it is persisted here,
keyed by trader_id. On AVT start the engine reads this to build `AVTConfig(virtual_capital, sleeves)`, so each
strategy runs concurrently off its own pool (a full futures book never starves options).

Empty / unset → no sleeves → the engine's historical shared pool (single implicit "all" sleeve). Simple JSON store
(matches `avt.template_store`), per-trader_id for multi-user correctness.

Persisted shape per user: {"virtual_capital": 500000, "sleeves": [
    {"name": "futures", "capital": 200000, "instruments": ["future","equity"], "max_positions": 3},
    {"name": "options", "capital": 50000,  "instruments": ["option"],          "max_positions": 2}]}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_STORE = Path("config/capital_sleeves.json")   # {trader_id: {"virtual_capital": float, "sleeves": [...]}}

_VALID_INSTRUMENTS = {"future", "equity", "option", "crypto"}


def _load() -> Dict[str, Any]:
    if not _STORE.exists():
        return {}
    try:
        return json.loads(_STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(d: Dict[str, Any]) -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    _STORE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def get_config(trader_id: Any) -> Dict[str, Any]:
    """This user's sleeve config, or {'virtual_capital': None, 'sleeves': []} if unset (→ shared pool)."""
    rec = _load().get(str(trader_id))
    if not isinstance(rec, dict):
        return {"virtual_capital": None, "sleeves": []}
    return {"virtual_capital": rec.get("virtual_capital"), "sleeves": list(rec.get("sleeves", []))}


def _validate(virtual_capital: float, sleeves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise + sanity-check a sleeve list. Raises ValueError on malformed input. Over-allocation (sleeves sum >
    account) is ALLOWED (the user may keep the rest as margin/drawdown buffer) — the engine only warns."""
    if virtual_capital is not None and float(virtual_capital) <= 0:
        raise ValueError("virtual_capital must be > 0")
    out: List[Dict[str, Any]] = []
    seen = set()
    for s in sleeves:
        name = str(s.get("name", "")).strip()
        if not name:
            raise ValueError("each sleeve needs a name")
        if name in seen:
            raise ValueError(f"duplicate sleeve name: {name}")
        seen.add(name)
        cap = float(s.get("capital", 0))
        if cap <= 0:
            raise ValueError(f"sleeve '{name}' capital must be > 0")
        instruments = [str(i).lower() for i in s.get("instruments", [])]
        bad = [i for i in instruments if i not in _VALID_INSTRUMENTS and i != "*"]
        if bad:
            raise ValueError(f"sleeve '{name}' has unknown instruments: {bad}")
        if not instruments:
            raise ValueError(f"sleeve '{name}' must fund at least one instrument")
        mp = int(s.get("max_positions", 0) or 0)
        size_mode = str(s.get("size_mode", "risk_pct") or "risk_pct")
        if size_mode not in ("risk_pct", "lots", "amount"):
            raise ValueError(f"sleeve '{name}' size_mode must be risk_pct | lots | amount")
        size_value = float(s.get("size_value", 0) or 0)
        if size_mode == "lots" and size_value < 1:
            raise ValueError(f"sleeve '{name}' needs lots >= 1")
        if size_mode == "amount" and size_value <= 0:
            raise ValueError(f"sleeve '{name}' needs an amount > 0")
        out.append({"name": name, "capital": cap, "instruments": instruments,
                    "max_positions": mp, "size_mode": size_mode, "size_value": size_value})
    return out


def set_config(trader_id: Any, virtual_capital: Optional[float],
               sleeves: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Persist this user's split. Returns the stored config. Applies on the next AVT (re)start."""
    clean = _validate(virtual_capital, sleeves or [])
    d = _load()
    d[str(trader_id)] = {"virtual_capital": virtual_capital, "sleeves": clean}
    _save(d)
    return {"virtual_capital": virtual_capital, "sleeves": clean}


def clear_config(trader_id: Any) -> None:
    """Remove this user's split → revert to the shared pool."""
    d = _load()
    d.pop(str(trader_id), None)
    _save(d)
