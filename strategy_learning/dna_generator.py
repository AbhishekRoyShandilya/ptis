# strategy_learning/dna_generator.py
"""
DNA Generator — Auto-generates StrategyDNA YAML from learned profiles.

This bridge ensures the 16 existing coaching agents (which consume
StrategyDNA YAML) continue working unchanged.  The DNA is now *generated*
from AI-learned profiles instead of being hand-authored.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from core.models import StrategyProfile

logger = logging.getLogger(__name__)


def _safe_str(val: Any, max_len: int = 500) -> str:
    """Convert any value to a truncated string, never raising."""
    s = str(val) if val else ""
    return s[:max_len]


def _infer_category(profile: StrategyProfile) -> str:
    """
    Infer strategy category from the narrative and entry criteria
    instead of always defaulting to 'momentum'.
    """
    text = (profile.narrative or "").lower()
    criteria_text = " ".join(
        c.get("criterion", "") + " " + c.get("description", "")
        for c in (profile.entry_criteria or [])
    ).lower()
    combined = text + " " + criteria_text

    category_keywords = {
        "momentum": ["momentum", "breakout", "strong candle", "follow-through"],
        "mean_reversion": ["mean revert", "oversold", "overbought", "fade", "pullback"],
        "scalping": ["scalp", "quick", "1-minute", "5-minute", "fast"],
        "swing": ["swing", "multi-day", "daily chart", "overnight"],
        "trend_following": ["trend follow", "moving average", "ride the trend"],
        "range": ["range", "support to resistance", "bouncing between"],
        "breakout": ["breakout", "break above", "break below", "squeeze"],
    }

    best_cat = "momentum"
    best_score = 0
    for cat, keywords in category_keywords.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > best_score:
            best_score = score
            best_cat = cat

    return best_cat


def _infer_timeframes(profile: StrategyProfile) -> tuple[str, str, str]:
    """
    Infer primary timeframe, confirmation timeframe, and typical hold
    from time preferences and annotations.
    """
    time_prefs = profile.time_preferences or {}

    # Check if any examples mention timeframes
    criteria_text = " ".join(
        c.get("description", "")
        for c in (profile.entry_criteria or [])
    ).lower()
    narrative = (profile.narrative or "").lower()
    combined = criteria_text + " " + narrative

    tf_map = {
        "1m": ("1m", "5m", "5m-30m"),
        "3m": ("3m", "15m", "15m-1h"),
        "5m": ("5m", "15m", "30m-2h"),
        "15m": ("15m", "5m", "30m-2h"),
        "1h": ("1h", "15m", "2h-8h"),
        "4h": ("4h", "1h", "1d-3d"),
        "daily": ("1D", "4h", "3d-2w"),
    }

    for label, values in tf_map.items():
        if label in combined:
            return values

    return ("15m", "5m", "30m-2h")  # sensible default


def generate_dna_yaml(profile: StrategyProfile) -> str:
    """
    Convert a learned StrategyProfile into StrategyDNA YAML.

    Compatible with strategy_engine/strategy_dna/schema.py and
    StrategyDNALoader.
    """
    entry_criteria = profile.entry_criteria or []
    exit_rules = profile.exit_rules or {}
    risk_params = profile.risk_parameters or {}
    regime_prefs = profile.regime_preferences or {}
    confidence = profile.confidence_scores or {}
    time_prefs = profile.time_preferences or {}

    category = _infer_category(profile)
    tf_primary, tf_confirm, typical_hold = _infer_timeframes(profile)

    # ── setup conditions from learned criteria ──
    setup_conditions: list[dict[str, Any]] = []
    for c in entry_criteria:
        setup_conditions.append({
            "name": c.get("criterion", "unknown"),
            "description": c.get("description", ""),
            "importance": c.get("importance", "important"),
            "what_good_looks_like": c.get("description", ""),
            "what_bad_looks_like": f"Missing or weak: {c.get('criterion', '')}",
            "confidence": c.get("confidence", 0.5),
        })

    if not setup_conditions:
        setup_conditions = [
            {
                "name": "learned_pattern",
                "description": "AI-learned entry pattern",
                "importance": "critical",
            },
            {
                "name": "market_context",
                "description": "Appropriate market context per learned profile",
                "importance": "important",
            },
        ]

    # ── A+ conditions: criteria with confidence ≥ 0.85 ──
    a_plus = [
        {
            "name": c.get("criterion", ""),
            "description": f"A+ signal: {c.get('description', '')}",
        }
        for c in entry_criteria
        if c.get("confidence", 0) >= 0.85
    ]

    # ── skip conditions from avoid regimes ──
    skip_conditions = [
        {
            "name": f"avoid_{entry.get('regime', 'unknown')}",
            "description": f"Strategy performs poorly in {entry.get('regime', '')} regime",
        }
        for entry in regime_prefs.get("avoid", [])
        if isinstance(entry, dict) and entry.get("regime")
    ]

    # ── regime sizing ──
    regime_sizing: dict[str, str] = {}
    for entry in regime_prefs.get("preferred", []):
        regime = entry.get("regime", "") if isinstance(entry, dict) else str(entry)
        if regime:
            regime_sizing[regime] = "standard_size"
    for entry in regime_prefs.get("neutral", []):
        regime = entry.get("regime", "") if isinstance(entry, dict) else str(entry)
        if regime:
            regime_sizing[regime] = "reduced_size"
    for entry in regime_prefs.get("avoid", []):
        regime = entry.get("regime", "") if isinstance(entry, dict) else str(entry)
        if regime:
            regime_sizing[regime] = "skip"

    # ── entry trigger (BUGFIX: was using exit_rules.stop_method) ──
    entry_trigger = "Pattern match confirmation"
    if entry_criteria:
        critical = [c for c in entry_criteria if c.get("importance") == "critical"]
        if critical:
            entry_trigger = f"When {critical[0].get('criterion', 'primary signal')} confirms"

    # ── stop loss ──
    stop_method = exit_rules.get("stop_method", "structure")
    stop_desc = exit_rules.get(
        "stop_description",
        "Per learned stop-placement pattern",
    )

    # ── position sizing ──
    risk_pct = risk_params.get("typical_risk_pct", 1.0)
    typical_rr = risk_params.get("typical_rr", 2.0)

    # ── targets ──
    target_patterns = exit_rules.get("target_patterns", [])
    if target_patterns:
        primary_target = target_patterns[0]
        target_str = str(primary_target.get("target", f"{typical_rr}R"))
    else:
        target_str = f"{typical_rr}R"

    # ── time window ──
    peak_hours = time_prefs.get("peak_hours", [])
    time_start = f"{peak_hours[0]:02d}:00" if peak_hours else "09:15"
    time_end = f"{peak_hours[-1] + 1:02d}:00" if peak_hours else "15:30"

    # ── Build the full DNA structure ──
    dna: dict[str, Any] = {
        "identity": {
            "name": profile.short_code,
            "display_name": profile.name,
            "short_code": profile.short_code.upper(),
            "version": 1,
            "category": category,
            "timeframe_primary": tf_primary,
            "timeframe_confirmation": tf_confirm,
            "typical_hold": typical_hold,
            "description": (
                profile.narrative[:300] if profile.narrative
                else f"AI-learned strategy: {profile.name}"
            ),
            "source": "ai_learned",
        },
        "entry_logic": {
            "narrative": profile.narrative or "",
            "setup_conditions": setup_conditions,
            "a_plus_conditions": a_plus,
            "skip_conditions": skip_conditions,
        },
        "execution": {
            "entry_trigger": entry_trigger,
            "entry_timing": "As per learned pattern confirmation",
            "common_mistakes": [],
            "ideal_entry": (profile.narrative or "")[:200],
        },
        "risk": {
            "stop_loss": {
                "method": stop_method,
                "description": stop_desc,
                "typical_atr_multiple": exit_rules.get("typical_stop_atr"),
            },
            "position_sizing": {
                "method": "fixed_fractional",
                "risk_per_trade": risk_pct,
                "max_risk": round(risk_pct * 1.5, 2),
            },
            "targets": {
                "primary": target_str,
                "method": "r_multiple",
                "typical_rr": typical_rr,
                "all_targets": target_patterns,
            },
        },
        "exit_rules": {
            "profit_taking": exit_rules.get("partial_exit_rules", []),
            "time_based": (
                [exit_rules["time_exit"]] if exit_rules.get("time_exit") else []
            ),
            "trailing": exit_rules.get("trailing_rules", ""),
            "common_exit_mistakes": [],
        },
        "market_conditions": {
            "works_best": [
                {
                    "regime": e.get("regime", ""),
                    "positive_ratio": e.get("positive_ratio", 0),
                    "notes": f"Learned from {e.get('example_count', 0)} examples",
                }
                for e in regime_prefs.get("preferred", [])
                if isinstance(e, dict)
            ],
            "works_poorly": [
                {
                    "regime": e.get("regime", ""),
                    "notes": "Learned to avoid",
                }
                for e in regime_prefs.get("avoid", [])
                if isinstance(e, dict)
            ],
            "regime_sizing": regime_sizing,
        },
        "time_window": {
            "start": time_start,
            "end": time_end,
            "peak_hours": peak_hours,
            "timezone": "Asia/Kolkata",
        },
        "psychology": {
            "emotional_traps": [],
            "ideal_mental_state": [
                "Calm and focused",
                "Following the learned pattern — not improvising",
            ],
            "do_not_trade_when": [
                "Profile confidence below 50%",
                "Outside learned time window",
                "In an avoided market regime",
            ],
        },
        "relationships": {
            "fallback_strategies": [],
            "conflicts_with": [],
            "pairs_well_with": [],
        },
        "personal_insights": {
            "strengths": [],
            "weaknesses": [],
            "edge_conditions": [],
            "learned_statistics": {
                "source": "ai_learned",
                "examples_count": profile.examples_count or 0,
                "confidence_scores": confidence,
                "category_inferred": category,
                "last_rebuilt": (
                    profile.last_rebuilt_at.isoformat()
                    if profile.last_rebuilt_at else None
                ),
            },
        },
        "checklist": (
            [f"✓ Verify: {c.get('criterion', '')}" for c in entry_criteria[:8]]
            or ["✓ Check pattern match", "✓ Verify market regime"]
        ),
        "tags": (profile.tags or []) + [f"category:{category}", "ai_learned"],
    }

    return yaml.dump(
        dna,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )


def save_dna_yaml(
    profile: StrategyProfile,
    output_dir: str = "config/strategies",
) -> str:
    """Generate and save the DNA YAML file.  Returns the file path."""
    yaml_content = generate_dna_yaml(profile)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{profile.short_code}.yaml"
    file_path.write_text(yaml_content, encoding="utf-8")
    logger.info("Generated StrategyDNA YAML: %s", file_path)
    return str(file_path)


def generate_and_validate(profile: StrategyProfile) -> tuple[str, list[str]]:
    """
    Generate YAML and return (yaml_string, list_of_warnings).

    Useful for previewing before saving.
    """
    warnings: list[str] = []
    conf = profile.confidence_scores or {}

    if conf.get("overall", 0) < 0.3:
        warnings.append(
            f"Low overall confidence ({conf.get('overall', 0):.0%}). "
            "DNA may be incomplete."
        )

    if not profile.entry_criteria:
        warnings.append("No entry criteria learned — DNA will use placeholders.")

    if not profile.exit_rules:
        warnings.append("No exit rules learned — DNA will use defaults.")

    examples = profile.examples_count or 0
    if examples < 5:
        warnings.append(
            f"Only {examples} examples. Recommend ≥5 for reliable DNA."
        )

    yaml_str = generate_dna_yaml(profile)
    return yaml_str, warnings