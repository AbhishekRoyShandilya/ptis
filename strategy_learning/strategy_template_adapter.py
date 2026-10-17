# strategy_learning/strategy_template_adapter.py
"""
Profile-to-Template Adapter — Bridges learned profiles to the existing
override-detection system by wrapping a StrategyProfile as a StrategyTemplate.

This lets ALL 8 existing override detectors work unchanged with AI-learned
strategies.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from core.models import StrategyProfile
from strategy_learning.models import ProfileConfidence

logger = logging.getLogger(__name__)

# Late imports to avoid circular dependencies — these are only used at
# call-time, not at module import.
_template_imports_ready = False
_ConditionDef = None
_ExitRules = None
_ExecutionPrefs = None
_RiskRules = None
_StopLossRule = None
_StrategyTemplate = None
_TargetLevel = None
_TimeExit = None
_TimeWindow = None
_TrailingRule = None


def _ensure_imports() -> None:
    """Lazy-import the template classes to avoid circular deps."""
    global _template_imports_ready
    if _template_imports_ready:
        return

    global _ConditionDef, _ExitRules, _ExecutionPrefs, _RiskRules, \
        _StopLossRule, _StrategyTemplate, _TargetLevel, _TimeExit, \
        _TimeWindow, _TrailingRule

    from strategy_engine.templates.loader import (
        ConditionDef,
        ExitRules,
        ExecutionPrefs,
        RiskRules,
        StopLossRule,
        StrategyTemplate,
        TargetLevel,
        TimeExit,
        TimeWindow,
        TrailingRule,
    )

    _ConditionDef = ConditionDef
    _ExitRules = ExitRules
    _ExecutionPrefs = ExecutionPrefs
    _RiskRules = RiskRules
    _StopLossRule = StopLossRule
    _StrategyTemplate = StrategyTemplate
    _TargetLevel = TargetLevel
    _TimeExit = TimeExit
    _TimeWindow = TimeWindow
    _TrailingRule = TrailingRule
    _template_imports_ready = True


# ──────────────────────────────────────────────────────────────
#  Regime mapping
# ──────────────────────────────────────────────────────────────

def _get_regime_map() -> dict[str, Any]:
    """
    Build the regime-name → MarketRegime mapping dynamically from the
    enum so it never goes stale when new members are added.
    """
    from core.enums import MarketRegime

    mapping: dict[str, MarketRegime] = {}
    for member in MarketRegime:
        # Map both the enum name and its value
        mapping[member.name.lower()] = member
        mapping[str(member.value).lower()] = member

    # Add common aliases
    aliases = {
        "range": "RANGING",
        "trending": "TRENDING_UP",
        "low_vol": "LOW_VOLATILITY",
        "high_vol": "HIGH_VOLATILITY",
    }
    for alias, canonical in aliases.items():
        try:
            mapping[alias] = MarketRegime[canonical]
        except KeyError:
            pass

    return mapping


# ──────────────────────────────────────────────────────────────
#  Public adapter function
# ──────────────────────────────────────────────────────────────

def profile_to_template(profile: StrategyProfile) -> Any:
    """
    Convert a learned StrategyProfile → StrategyTemplate.

    The returned template is fully compatible with OverrideDetector
    and all 8 existing detection methods.
    """
    _ensure_imports()
    assert _StrategyTemplate is not None  # mypy

    regime_map = _get_regime_map()

    prefs = profile.regime_preferences or {}
    preferred_regimes = _parse_regimes(prefs.get("preferred", []), regime_map)
    avoid_regimes = _parse_regimes(prefs.get("avoid", []), regime_map)

    entry_confirmations = _build_confirmations(profile.entry_criteria or [])
    exit_rules = _build_exit_rules(profile.exit_rules or {})
    risk_rules = _build_risk_rules(profile.risk_parameters or {})
    time_window = _build_time_window(profile.time_preferences or {})

    # Infer direction bias from examples
    direction = _infer_direction(profile)

    conf = ProfileConfidence(**(profile.confidence_scores or {}))

    return _StrategyTemplate(
        name=profile.name,
        id=profile.short_code,
        version=1,
        description=profile.narrative or f"Learned strategy: {profile.name}",
        author="ai_learned",
        active=profile.status == "active" and conf.is_tradeable,
        preferred_regimes=preferred_regimes,
        avoid_regimes=avoid_regimes,
        neutral_regimes=[],
        direction=direction,
        entry_confirmations=entry_confirmations,
        avoid_conditions=[],
        time_window=time_window,
        exit_rules=exit_rules,
        risk_rules=risk_rules,
        invalidation_conditions=[],
        execution=_ExecutionPrefs(preferred_timeframe="15m"),
    )


# ──────────────────────────────────────────────────────────────
#  Internal builders
# ──────────────────────────────────────────────────────────────

def _parse_regimes(
    regime_list: list[dict[str, Any]] | list[str],
    regime_map: dict[str, Any],
) -> list[Any]:
    regimes = []
    for entry in regime_list:
        name = (
            entry.get("regime", entry) if isinstance(entry, dict) else str(entry)
        )
        regime = regime_map.get(name.strip().lower())
        if regime:
            regimes.append(regime)
        else:
            logger.debug("Unknown regime '%s' — skipping", name)
    return regimes


def _build_confirmations(criteria: list[dict[str, Any]]) -> list[Any]:
    _ensure_imports()
    assert _ConditionDef is not None

    weight_map = {
        "critical": 1.5,
        "important": 1.0,
        "preferred": 0.7,
        "nice_to_have": 0.3,
    }

    confirmations = []
    for c in criteria:
        importance = c.get("importance", "important")
        required = importance in ("critical", "important")
        weight = weight_map.get(importance, 1.0)

        # Scale weight by criterion confidence
        crit_confidence = c.get("confidence", 0.5)
        effective_weight = round(weight * (0.5 + 0.5 * crit_confidence), 2)

        confirmations.append(_ConditionDef(
            type="learned_criterion",
            params={
                "criterion": c.get("criterion", ""),
                "description": c.get("description", ""),
                "confidence": crit_confidence,
            },
            required=required,
            weight=effective_weight,
            description=c.get("description", c.get("criterion", "")),
        ))
    return confirmations


def _build_exit_rules(exit_data: dict[str, Any]) -> Any:
    _ensure_imports()
    assert _ExitRules is not None and _StopLossRule is not None

    # Stop loss
    stop_type = exit_data.get("stop_method", "atr") or "atr"
    stop_params: dict[str, Any] = {}
    if exit_data.get("typical_stop_atr"):
        stop_params["multiplier"] = exit_data["typical_stop_atr"]
        stop_type = "atr"

    stop_loss = _StopLossRule(type=stop_type, params=stop_params)

    # Targets
    targets = []
    target_patterns = exit_data.get("target_patterns", [])
    remaining_pct = 100.0

    for tp in target_patterns:
        if not isinstance(tp, dict):
            continue
        r_str = str(tp.get("target", "1R")).replace("R", "").replace("r", "")
        try:
            r_val = float(r_str)
        except ValueError:
            r_val = 1.0

        # Heuristic: split position across targets
        action = tp.get("action", "")
        pct = 50.0  # default
        if "25%" in action or "25 %" in action:
            pct = 25.0
        elif "33%" in action or "third" in action.lower():
            pct = 33.0
        elif "75%" in action or "75 %" in action:
            pct = 75.0

        pct = min(pct, remaining_pct)
        remaining_pct -= pct

        targets.append(_TargetLevel(r_multiple=r_val, size_pct=pct))

    if not targets:
        targets = [_TargetLevel(r_multiple=2.0, size_pct=100.0)]

    # Trailing
    trailing_text = exit_data.get("trailing_rules", "")
    trailing = _TrailingRule(
        enabled=bool(trailing_text),
        type="candle" if "candle" in trailing_text.lower() else (
            "atr" if "atr" in trailing_text.lower() else ""
        ),
    )

    # Time exit
    time_exit_str = exit_data.get("time_exit", "")
    time_exit = _TimeExit(
        enabled=bool(time_exit_str),
        close_before=time_exit_str or None,
    )

    return _ExitRules(
        stop_loss=stop_loss,
        targets=targets,
        trailing=trailing,
        time_exit=time_exit,
    )


def _build_risk_rules(risk_data: dict[str, Any]) -> Any:
    _ensure_imports()
    assert _RiskRules is not None

    return _RiskRules(
        max_risk_pct=risk_data.get("typical_risk_pct", 1.0),
        min_r_target=risk_data.get("typical_rr", 1.5),
        max_daily_trades=risk_data.get("max_daily_trades"),
        max_daily_loss_pct=risk_data.get("max_daily_loss_pct"),
    )


def _build_time_window(
    time_data: dict[str, Any],
) -> Optional[Any]:
    _ensure_imports()
    assert _TimeWindow is not None

    peak_hours = time_data.get("peak_hours", [])
    if peak_hours:
        return _TimeWindow(
            start=f"{peak_hours[0]:02d}:00",
            end=f"{peak_hours[-1] + 1:02d}:00",
            timezone="Asia/Kolkata",
        )

    dist = time_data.get("time_distribution", {})
    if not dist:
        return None

    # Fall back to IST market hours
    return _TimeWindow(
        start="09:15",
        end="15:30",
        timezone="Asia/Kolkata",
    )


def _infer_direction(profile: StrategyProfile) -> str:
    """
    Infer whether the strategy is long-biased, short-biased, or any
    by examining the learned narrative and entry criteria.
    """
    text = (profile.narrative or "").lower()
    criteria_text = " ".join(
        c.get("description", "") for c in (profile.entry_criteria or [])
    ).lower()
    combined = text + " " + criteria_text

    long_signals = sum(
        1 for kw in ["long", "buy", "bullish", "support bounce", "breakout above"]
        if kw in combined
    )
    short_signals = sum(
        1 for kw in ["short", "sell", "bearish", "resistance rejection", "breakdown"]
        if kw in combined
    )

    if long_signals > short_signals * 2:
        return "long"
    if short_signals > long_signals * 2:
        return "short"
    return "any"