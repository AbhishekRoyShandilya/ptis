"""
Profile → Registry-Template bridge.

A taught strategy is stored as a StrategyProfile (DB) and a DNA-format YAML
(``name:`` keyed). But the StrategyRegistry — which the intelligence scanner,
co-planner, behavioral matcher and setup scanner all read — only understands
``strategy:``-format templates with scanner-evaluable entry conditions.

This module closes that gap: it converts a learned StrategyProfile into a
``strategy:``-format template whose entry confirmations use the exact condition
types the readiness scanner can evaluate (see
``intelligence/readiness_scanner.py:_eval_condition_env``). The template is
written next to the DNA YAML as ``{short_code}.tmpl.yaml`` so the two formats
coexist (each loader content-filters the other out), and the registry picks it
up on the next ``reload()`` (triggered by the BOOTSTRAP_COMPLETED hot-reload).

Result: the moment a user teaches/approves a strategy, the whole intelligence
stack becomes aware of THAT strategy — and nothing else.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


# ── Condition vocabulary the readiness scanner can actually evaluate ──
# Maps a known pattern id → ordered list of (condition_type, params).
_PATTERN_CONDITIONS: Dict[str, List[Tuple[str, dict]]] = {
    "compression_breakout": [("bb_squeeze", {}), ("breakout_candle", {}), ("volume_surge", {"min_ratio": 1.5})],
    "ema_pullback":         [("ema_alignment", {}), ("pullback_to_ema", {})],
    "vwap_bounce":          [("vwap_support", {}), ("vwap_proximity", {"max_distance_pct": 0.5})],
    "vwap_reclaim":         [("close_above_vwap", {}), ("volume_above_avg", {})],
    "orb":                  [("orb_high_break", {}), ("volume_above_avg", {})],
    "opening_range_breakout": [("orb_high_break", {}), ("volume_above_avg", {})],
    "momentum_exhaustion":  [("exhaustion_candle", {}), ("rsi_extreme", {}), ("price_extension", {})],
    "momentum_continuation": [("ema_alignment", {}), ("volume_above_avg", {})],
    "gap_fill":             [("gap_detected", {})],
    "expiry_fade":          [("expiry_day", {}), ("overextension", {})],
}

# Keyword → condition_type, scanned across the strategy's learned text.
# Order matters: more specific phrases first so they win.
_KEYWORD_CONDITIONS: List[Tuple[Tuple[str, ...], Tuple[str, dict]]] = [
    (("compression", "squeeze", "tight range", "narrow range", "consolidat"), ("bb_squeeze", {})),
    (("volume surge", "volume spike", "surge in volume", "volume expansion"), ("volume_surge", {"min_ratio": 1.5})),
    (("above average volume", "high volume", "rising volume", "volume confirm", "volume support"), ("volume_above_avg", {})),
    (("pullback to", "pull back to", "retrace to", "retest of the", "pullback", "dip to"), ("pullback_to_ema", {})),
    (("vwap reclaim", "reclaim vwap", "close above vwap", "above vwap"), ("close_above_vwap", {})),
    (("vwap bounce", "vwap support", "bounce off vwap", "near vwap", "hold vwap"), ("vwap_support", {})),
    (("opening range", "first 15", "first 5", "first 30", "opening break", "orb"), ("orb_high_break", {})),
    (("gap up", "gap down", "gap fill", "opening gap", "gaps "), ("gap_detected", {})),
    (("exhaustion", "reversal candle", "long wick", "rejection wick", "pin bar", "climax", "blow off"), ("exhaustion_candle", {})),
    (("overbought", "oversold", "rsi", "extreme reading"), ("rsi_extreme", {})),
    (("over-extended", "overextended", "extension", "extended", "stretched", "far from"), ("price_extension", {})),
    (("expiry", "expiration"), ("expiry_day", {})),
    (("atr expansion", "range expansion", "volatility expansion", "expanding range"), ("atr_expansion", {})),
    (("breakout", "break out", "breaks above", "break of", "breaks the", "resistance break"), ("breakout_candle", {})),
    (("ema", "moving average", "20 ema", "50 ema", "trend align", "above ema", "trend continuation"), ("ema_alignment", {})),
]

_MAX_CONDITIONS = 4


def _sanitize_id(short_code: str, name: str) -> str:
    base = (short_code or name or "strategy").lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return base or "strategy"


def _collect_text(profile: Any) -> str:
    parts: List[str] = [getattr(profile, "name", "") or "", getattr(profile, "narrative", "") or ""]
    for c in (getattr(profile, "entry_criteria", None) or []):
        if isinstance(c, dict):
            parts.append(str(c.get("criterion", "")))
            parts.append(str(c.get("description", "")))
    for t in (getattr(profile, "tags", None) or []):
        parts.append(str(t))
    return " ".join(parts).lower()


def _derive_conditions(profile: Any) -> List[Dict[str, Any]]:
    """Map the learned strategy onto scanner-evaluable condition types."""
    ordered: List[Tuple[str, dict]] = []
    seen: set[str] = set()

    def _add(ctype: str, params: dict) -> None:
        if ctype not in seen:
            seen.add(ctype)
            ordered.append((ctype, params))

    # 1. Strongest signal: explicitly matched pattern templates.
    meta = getattr(profile, "profile_metadata", None) or {}
    for pid in (meta.get("matched_patterns") or []):
        for ctype, params in _PATTERN_CONDITIONS.get(str(pid), []):
            _add(ctype, params)

    # 2. Keyword scan of the learned text.
    text = _collect_text(profile)
    for keywords, (ctype, params) in _KEYWORD_CONDITIONS:
        if any(k in text for k in keywords):
            _add(ctype, params)

    # 3. Fallback so the strategy still scores off live participation + regime
    #    rather than collapsing to zero readiness.
    if not ordered:
        _add("volume_above_avg", {})

    conditions: List[Dict[str, Any]] = []
    for i, (ctype, params) in enumerate(ordered[:_MAX_CONDITIONS]):
        conditions.append({
            "type": ctype,
            "params": params,
            # The defining condition is required (templates must have ≥1); the
            # rest are supporting so readiness stays graded rather than
            # all-or-nothing and we avoid false "triggered" alerts.
            "required": i == 0,
            "weight": 1.0,
            "description": ctype.replace("_", " "),
        })
    return conditions


def _map_regimes(profile: Any) -> Dict[str, List[str]]:
    rp = getattr(profile, "regime_preferences", None) or {}

    def _names(items: Any) -> List[str]:
        out: List[str] = []
        for it in (items or []):
            if isinstance(it, dict) and it.get("regime"):
                out.append(str(it["regime"]))
            elif isinstance(it, str):
                out.append(it)
        return out

    return {"preferred": _names(rp.get("preferred")), "avoid": _names(rp.get("avoid"))}


def _map_exit(profile: Any) -> Dict[str, Any]:
    ex = getattr(profile, "exit_rules", None) or {}
    stop_method = ex.get("stop_method", "atr")
    atr_mult = ex.get("typical_stop_atr", 1.5)
    targets = []
    for t in (ex.get("targets") or []):
        if isinstance(t, dict):
            targets.append({
                "r_multiple": float(t.get("r_multiple", t.get("atr_multiple", 2.0)) or 2.0),
                "size_pct": float(t.get("size_pct", 100.0) or 100.0),
            })
    if not targets:
        targets = [{"r_multiple": 2.0, "size_pct": 100.0}]
    return {
        "stop_loss": {"type": stop_method, "params": {"atr_multiple": atr_mult} if stop_method == "atr" else {}},
        "targets": targets,
    }


def build_template_dict(
    profile: Any,
    direction: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a StrategyProfile into a ``strategy:``-format template dict."""
    meta = getattr(profile, "profile_metadata", None) or {}
    dirn = (direction or meta.get("direction") or "any").lower()
    if dirn == "both":
        dirn = "any"
    tf = timeframe or (meta.get("preferred_timeframes") or ["5m"])[0]

    exit_block = _map_exit(profile)
    min_r = exit_block["targets"][0]["r_multiple"] if exit_block["targets"] else 1.5

    return {
        "strategy": {
            "id": _sanitize_id(getattr(profile, "short_code", ""), getattr(profile, "name", "")),
            "name": getattr(profile, "name", "Strategy"),
            "version": 1,
            "description": (getattr(profile, "narrative", "") or "")[:500] or f"User-taught strategy: {getattr(profile, 'name', '')}",
            "author": "user",
            "active": True,
            # Intentionally NOT is_library → this WILL load into the registry.
            "market_regime": _map_regimes(profile),
            "entry": {
                "direction": dirn,
                "confirmations": _derive_conditions(profile),
            },
            "exit": exit_block,
            "risk": {"max_risk_pct": 1.0, "min_r_target": float(min_r)},
            "execution": {"preferred_timeframe": tf, "entry_type": "market"},
        }
    }


def save_template_yaml(
    profile: Any,
    direction: Optional[str] = None,
    timeframe: Optional[str] = None,
    output_dir: str = "config/strategies",
) -> str:
    """
    Write a registry template for a taught strategy. Returns the file path.

    Filename is ``{short_code}.tmpl.yaml`` so it never collides with the DNA
    YAML (``{short_code}.yaml``); the two loaders content-filter each other out.
    """
    data = build_template_dict(profile, direction, timeframe)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    short_code = getattr(profile, "short_code", None) or data["strategy"]["id"]
    path = out / f"{short_code}.tmpl.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info(
        "Generated registry template: %s (%d entry conditions)",
        path, len(data["strategy"]["entry"]["confirmations"]),
    )
    return str(path)
