# bootstrap/pattern_library.py
"""
Pattern Library
================

Pre-built library of common trading patterns with:
- Detection rules (hard + soft)
- Feature signatures (what the pattern looks like in data)
- Known variants
- Historical characteristics (typical win rate, R:R, regime sensitivity)
- Common edges and failure modes
- Typical uncertainty factors

This is NOT a strategy backtester. It's a KNOWLEDGE BASE
that gives the system a head start.

When user says "compression breakout", the library provides:
- Initial detection rules
- Initial feature centroids
- Initial soft scoring functions
- Known failure modes
- Expected performance envelope

The LLM Strategy Interpreter maps user's description to
library entries and customizes them.

Library is MANUALLY CURATED by trading domain experts.
It's the system's "prior knowledge" about trading patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PatternVariant:
    """A known variant of a pattern."""
    name: str
    description: str
    distinguishing_features: Dict[str, str] = field(
        default_factory=dict
    )
    # How this variant differs from the base
    typical_win_rate: float = 0.0
    typical_rr: float = 0.0
    best_regime: str = ""
    worst_regime: str = ""
    notes: str = ""


@dataclass
class FailureMode:
    """Known way this pattern fails."""
    name: str
    description: str
    detection_hint: str  # how to detect this failure
    frequency: str       # "common" | "occasional" | "rare"
    severity: str        # "major" | "moderate" | "minor"
    mitigation: str      # what to do about it


@dataclass
class PatternTemplate:
    """
    Complete template for a trading pattern.
    
    Contains everything needed to bootstrap a strategy
    understanding from Day 1.
    """
    # Identity
    pattern_id: str
    name: str
    category: str          # "breakout" | "reversal" | "momentum" | "mean_reversion"
    description: str
    aliases: List[str] = field(default_factory=list)
    # e.g. ["squeeze breakout", "volatility contraction", "VCP"]

    # ── Hard rules (non-negotiable for this pattern) ──
    hard_rules: List[Dict[str, Any]] = field(default_factory=list)
    # Example: [{"name": "range_contracting", "field": "range_contraction_pct",
    #            "operator": "gte", "value": 0.2}]

    # ── Soft scoring functions ──
    soft_scorers: List[Dict[str, Any]] = field(default_factory=list)
    # Example: [{"name": "compression_cleanness", "weight": 0.3,
    #            "description": "How orderly is the contraction"}]

    # ── Context factors ──
    context_factors: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # ── Feature signature ──
    # What this pattern LOOKS LIKE in feature space
    # Used as initial centroids before real data arrives
    feature_keys: List[str] = field(default_factory=list)
    initial_centroids: List[Dict[str, float]] = field(
        default_factory=list
    )
    # Typical ranges for key features
    feature_ranges: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    # e.g. {"rsi_at_entry": {"min": 40, "max": 60, "typical": 50}}

    # ── Known variants ──
    variants: List[PatternVariant] = field(default_factory=list)

    # ── Historical characteristics ──
    typical_win_rate: float = 0.0
    typical_win_rate_range: tuple = (0.0, 0.0)
    typical_avg_winner_r: float = 0.0
    typical_avg_loser_r: float = 0.0
    typical_profit_factor: float = 0.0
    typical_max_drawdown_pct: float = 0.0

    # ── Regime sensitivity ──
    regime_performance: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    # e.g. {"trending": {"win_rate": 0.65, "avg_r": 0.8},
    #        "ranging": {"win_rate": 0.45, "avg_r": 0.2}}
    best_regimes: List[str] = field(default_factory=list)
    worst_regimes: List[str] = field(default_factory=list)

    # ── Known failure modes ──
    failure_modes: List[FailureMode] = field(default_factory=list)

    # ── Uncertainty factors ──
    uncertainty_factors: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # ── Edge characteristics ──
    edge_source: str = ""     # where does the edge come from
    edge_fragility: str = ""  # "robust" | "moderate" | "fragile"
    edge_notes: str = ""

    # ── Exit logic templates ──
    exit_templates: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # ── Teaching questions ──
    # Questions to ask user to customize this template
    teaching_questions: List[Dict[str, str]] = field(
        default_factory=list
    )


# ══════════════════════════════════════════════════════════════════
#  THE LIBRARY
# ══════════════════════════════════════════════════════════════════


class PatternLibrary:
    """
    The master library of pattern templates.
    
    This is the system's PRIOR KNOWLEDGE about trading.
    Manually curated. Updated periodically.
    """

    def __init__(self):
        self._templates: Dict[str, PatternTemplate] = {}
        self._alias_index: Dict[str, str] = {}
        self._load_builtin_patterns()

    def get_template(
        self, pattern_id: str
    ) -> Optional[PatternTemplate]:
        return self._templates.get(pattern_id)

    def search(self, query: str) -> List[PatternTemplate]:
        """
        Search library by name, alias, or description.
        Used by LLM interpreter to find matching templates.
        """
        query_lower = query.lower()
        results = []

        # Direct ID match
        if query_lower in self._templates:
            results.append(self._templates[query_lower])

        # Alias match
        if query_lower in self._alias_index:
            tid = self._alias_index[query_lower]
            template = self._templates.get(tid)
            if template and template not in results:
                results.append(template)

        # Fuzzy name/description match
        for template in self._templates.values():
            if template in results:
                continue

            name_match = query_lower in template.name.lower()
            desc_match = query_lower in template.description.lower()
            alias_match = any(
                query_lower in a.lower() for a in template.aliases
            )
            category_match = query_lower in template.category.lower()

            if name_match or alias_match:
                results.append(template)
            elif desc_match or category_match:
                results.append(template)

        return results

    def get_all_ids(self) -> List[str]:
        return list(self._templates.keys())

    def get_by_category(self, category: str) -> List[PatternTemplate]:
        return [
            t for t in self._templates.values()
            if t.category == category
        ]

    # ── Built-in patterns ─────────────────────────────────────────

    def _load_builtin_patterns(self) -> None:
        """Load all built-in pattern templates."""
        patterns = [
            self._compression_breakout(),
            self._pullback_to_ema(),
            self._vwap_reclaim(),
            self._opening_range_breakout(),
            self._supply_demand_zone(),
            self._momentum_continuation(),
            self._mean_reversion_extreme(),
            self._flag_pattern(),
        ]

        for p in patterns:
            self._templates[p.pattern_id] = p
            for alias in p.aliases:
                self._alias_index[alias.lower()] = p.pattern_id

    def _compression_breakout(self) -> PatternTemplate:
        return PatternTemplate(
            pattern_id="compression_breakout",
            name="Compression Breakout",
            category="breakout",
            description=(
                "Price consolidates in a narrowing range "
                "(compression/squeeze), then breaks out with "
                "volume expansion. Edge comes from stored "
                "energy releasing directionally."
            ),
            aliases=[
                "squeeze breakout", "volatility contraction",
                "vcp", "volatility contraction pattern",
                "tight range breakout", "coil breakout",
                "consolidation breakout", "range contraction",
            ],

            hard_rules=[
                {
                    "name": "range_contracting",
                    "field": "range_contraction_pct",
                    "operator": "gte",
                    "value": 0.20,
                    "description": "Range must contract by at least 20%",
                },
                {
                    "name": "min_compression_candles",
                    "field": "compression_candle_count",
                    "operator": "gte",
                    "value": 3,
                    "description": "At least 3 candles of compression",
                },
                {
                    "name": "breakout_candle_exists",
                    "field": "has_breakout_candle",
                    "operator": "eq",
                    "value": 1,
                    "description": "Must have a clear breakout candle",
                },
            ],

            soft_scorers=[
                {
                    "name": "compression_cleanness",
                    "weight": 0.25,
                    "description": (
                        "How orderly is the contraction? "
                        "Small bodies, minimal wicks, "
                        "progressively narrowing range."
                    ),
                    "scoring_hint": (
                        "Score based on: wick_to_body_ratio "
                        "(lower=cleaner), range_monotonicity "
                        "(higher=cleaner), body_consistency "
                        "(lower variance=cleaner)"
                    ),
                },
                {
                    "name": "volume_pattern",
                    "weight": 0.20,
                    "description": (
                        "Volume should contract during compression "
                        "and expand on breakout. Dry compression = "
                        "sellers exhausted."
                    ),
                    "scoring_hint": (
                        "Score: volume_contraction_during_squeeze * 0.5 "
                        "+ volume_expansion_on_breakout * 0.5"
                    ),
                },
                {
                    "name": "breakout_strength",
                    "weight": 0.25,
                    "description": (
                        "How strong is the breakout candle? "
                        "Full body, closes near high, "
                        "volume spike."
                    ),
                    "scoring_hint": (
                        "Score: body_pct_of_range * 0.4 + "
                        "close_position * 0.3 + "
                        "volume_vs_avg * 0.3"
                    ),
                },
                {
                    "name": "structure_quality",
                    "weight": 0.15,
                    "description": (
                        "Is the compression at a meaningful "
                        "structural location? Near support, "
                        "after a strong move, at VWAP?"
                    ),
                    "scoring_hint": (
                        "Score: near_support * 0.3 + "
                        "after_impulse * 0.3 + "
                        "vwap_proximity * 0.2 + "
                        "round_number * 0.2"
                    ),
                },
                {
                    "name": "higher_tf_alignment",
                    "weight": 0.15,
                    "description": (
                        "Does the higher timeframe support "
                        "the breakout direction?"
                    ),
                    "scoring_hint": (
                        "Score: htf_trend_aligned * 0.5 + "
                        "htf_ema_slope * 0.3 + "
                        "sector_momentum * 0.2"
                    ),
                },
            ],

            context_factors=[
                {
                    "name": "market_regime",
                    "weight": 0.30,
                    "description": "Breakouts work best in trending markets",
                },
                {
                    "name": "sector_momentum",
                    "weight": 0.20,
                    "description": "Stock's sector should show momentum",
                },
                {
                    "name": "relative_volume",
                    "weight": 0.25,
                    "description": "Higher RVOL = more institutional interest",
                },
                {
                    "name": "time_of_day",
                    "weight": 0.15,
                    "description": (
                        "Best in first 2 hours and last hour. "
                        "Avoid midday chop."
                    ),
                },
                {
                    "name": "atr_percentile",
                    "weight": 0.10,
                    "description": "Moderate ATR — not too dead, not too wild",
                },
            ],

            feature_keys=[
                "range_contraction_pct",
                "compression_candle_count",
                "volume_contraction_ratio",
                "breakout_candle_body_pct",
                "breakout_volume_ratio",
                "rsi_at_entry",
                "distance_from_vwap_atr",
                "higher_tf_trend_score",
                "atr_percentile",
                "time_bucket",
            ],

            initial_centroids=[
                # Variant 1: "Classic tight compression"
                {
                    "range_contraction_pct": 0.35,
                    "compression_candle_count": 6,
                    "volume_contraction_ratio": 0.6,
                    "breakout_candle_body_pct": 0.75,
                    "breakout_volume_ratio": 2.0,
                    "rsi_at_entry": 52,
                    "distance_from_vwap_atr": 0.3,
                    "higher_tf_trend_score": 0.7,
                    "atr_percentile": 50,
                    "time_bucket": 1,
                },
                # Variant 2: "Extended squeeze"
                {
                    "range_contraction_pct": 0.50,
                    "compression_candle_count": 12,
                    "volume_contraction_ratio": 0.4,
                    "breakout_candle_body_pct": 0.80,
                    "breakout_volume_ratio": 3.0,
                    "rsi_at_entry": 48,
                    "distance_from_vwap_atr": 0.1,
                    "higher_tf_trend_score": 0.6,
                    "atr_percentile": 35,
                    "time_bucket": 1,
                },
                # Variant 3: "Quick coil"
                {
                    "range_contraction_pct": 0.25,
                    "compression_candle_count": 4,
                    "volume_contraction_ratio": 0.7,
                    "breakout_candle_body_pct": 0.70,
                    "breakout_volume_ratio": 1.8,
                    "rsi_at_entry": 55,
                    "distance_from_vwap_atr": 0.5,
                    "higher_tf_trend_score": 0.8,
                    "atr_percentile": 55,
                    "time_bucket": 0,
                },
            ],

            feature_ranges={
                "range_contraction_pct": {
                    "min": 0.15, "max": 0.70, "typical": 0.35,
                },
                "compression_candle_count": {
                    "min": 3, "max": 20, "typical": 7,
                },
                "volume_contraction_ratio": {
                    "min": 0.3, "max": 0.9, "typical": 0.6,
                },
                "breakout_candle_body_pct": {
                    "min": 0.5, "max": 1.0, "typical": 0.75,
                },
                "breakout_volume_ratio": {
                    "min": 1.2, "max": 5.0, "typical": 2.0,
                },
                "rsi_at_entry": {
                    "min": 35, "max": 65, "typical": 50,
                },
            },

            variants=[
                PatternVariant(
                    name="Classic Tight",
                    description="5-8 candle compression with clean range contraction",
                    typical_win_rate=0.58,
                    typical_rr=2.5,
                    best_regime="trending_up",
                    worst_regime="ranging_choppy",
                ),
                PatternVariant(
                    name="Extended Squeeze",
                    description="10+ candle deep compression, very low volume",
                    typical_win_rate=0.52,
                    typical_rr=3.5,
                    best_regime="trending_up",
                    worst_regime="high_volatility",
                    notes="Lower win rate but bigger moves when it works",
                ),
                PatternVariant(
                    name="Quick Coil",
                    description="3-5 candle compression after strong impulse",
                    typical_win_rate=0.62,
                    typical_rr=1.8,
                    best_regime="strong_momentum",
                    worst_regime="ranging_choppy",
                    notes="Higher win rate, smaller moves. Best as continuation.",
                ),
            ],

            typical_win_rate=0.55,
            typical_win_rate_range=(0.45, 0.65),
            typical_avg_winner_r=2.2,
            typical_avg_loser_r=-0.9,
            typical_profit_factor=1.4,
            typical_max_drawdown_pct=15.0,

            regime_performance={
                "trending_up": {"win_rate": 0.63, "avg_r": 0.85},
                "trending_down": {"win_rate": 0.48, "avg_r": 0.15},
                "ranging": {"win_rate": 0.42, "avg_r": -0.10},
                "high_volatility": {"win_rate": 0.50, "avg_r": 0.30},
            },
            best_regimes=["trending_up", "strong_momentum"],
            worst_regimes=["ranging", "choppy"],

            failure_modes=[
                FailureMode(
                    name="False Breakout",
                    description="Breaks out then immediately reverses",
                    detection_hint=(
                        "Low volume on breakout, no higher TF support, "
                        "breakout into resistance"
                    ),
                    frequency="common",
                    severity="major",
                    mitigation=(
                        "Wait for breakout candle close. "
                        "Require volume confirmation. "
                        "Check for overhead resistance."
                    ),
                ),
                FailureMode(
                    name="Fakeout Shakeout",
                    description=(
                        "Breaks one direction, reverses hard the other way"
                    ),
                    detection_hint=(
                        "Happens near key levels, often before "
                        "news/events"
                    ),
                    frequency="occasional",
                    severity="major",
                    mitigation=(
                        "Don't trade around events. "
                        "Use wider stops. "
                        "Wait for retest of breakout level."
                    ),
                ),
                FailureMode(
                    name="Stall After Breakout",
                    description=(
                        "Breaks out cleanly but then stalls, "
                        "doesn't follow through"
                    ),
                    detection_hint=(
                        "Low RVOL, ranging market, "
                        "midday time period"
                    ),
                    frequency="common",
                    severity="moderate",
                    mitigation=(
                        "Require RVOL > 1.5. "
                        "Avoid midday entries. "
                        "Have time-based exit."
                    ),
                ),
            ],

            uncertainty_factors=[
                {
                    "name": "compression_quality_judgment",
                    "description": (
                        "User's subjective feel for compression "
                        "'energy' or 'cleanness' that goes beyond "
                        "measurable metrics"
                    ),
                    "impact": "high",
                    "penalty": 0.15,
                    "can_learn": True,
                },
                {
                    "name": "breakout_authenticity",
                    "description": (
                        "Whether the breakout is 'real' institutional "
                        "buying vs 'fake' retail chase"
                    ),
                    "impact": "medium",
                    "penalty": 0.10,
                    "can_learn": True,
                },
            ],

            exit_templates=[
                {
                    "name": "atr_trailing",
                    "description": "Trail stop at 1.5-2x ATR below",
                    "params": {"atr_multiple": 1.5},
                },
                {
                    "name": "structure_based",
                    "description": "Stop below compression low",
                    "params": {"reference": "compression_low"},
                },
                {
                    "name": "time_exit",
                    "description": "Exit if no follow-through in N candles",
                    "params": {"max_candles": 15},
                },
            ],

            teaching_questions=[
                {
                    "id": "compression_length",
                    "question": (
                        "How many candles of compression do you "
                        "typically look for? (e.g., 5-7, 10+)"
                    ),
                    "maps_to": "compression_candle_count",
                    "type": "range",
                },
                {
                    "id": "volume_requirement",
                    "question": (
                        "Do you require volume expansion on the "
                        "breakout? How much above average?"
                    ),
                    "maps_to": "breakout_volume_ratio",
                    "type": "threshold",
                },
                {
                    "id": "quality_filter",
                    "question": (
                        "What makes you SKIP a compression that "
                        "technically meets your criteria? What "
                        "makes it 'ugly'?"
                    ),
                    "maps_to": "compression_cleanness",
                    "type": "judgment",
                },
                {
                    "id": "regime_preference",
                    "question": (
                        "Do you trade this in all market conditions, "
                        "or only when the market is trending?"
                    ),
                    "maps_to": "regime_filter",
                    "type": "regime",
                },
                {
                    "id": "exit_method",
                    "question": (
                        "How do you decide when to exit? "
                        "Trailing stop? Target? Time-based? "
                        "Story broken?"
                    ),
                    "maps_to": "exit_logic",
                    "type": "exit",
                },
                {
                    "id": "timeframe",
                    "question": "What timeframe do you trade this on?",
                    "maps_to": "timeframe",
                    "type": "selection",
                },
                {
                    "id": "stop_placement",
                    "question": (
                        "Where do you place your stop loss? "
                        "Below compression low? ATR-based? Fixed %?"
                    ),
                    "maps_to": "stop_logic",
                    "type": "exit",
                },
                {
                    "id": "position_sizing",
                    "question": (
                        "How do you size your position? "
                        "Fixed risk %? Conviction-based?"
                    ),
                    "maps_to": "sizing_logic",
                    "type": "sizing",
                },
                {
                    "id": "stock_selection",
                    "question": (
                        "How do you choose WHICH stocks to "
                        "watch for this pattern? Any sector, "
                        "market cap, or liquidity preferences?"
                    ),
                    "maps_to": "stock_universe",
                    "type": "universe",
                },
                {
                    "id": "failure_experience",
                    "question": (
                        "When does this strategy FAIL for you? "
                        "What conditions lead to losses?"
                    ),
                    "maps_to": "failure_conditions",
                    "type": "judgment",
                },
            ],

            edge_source=(
                "Volatility contraction precedes expansion. "
                "Compression = equilibrium. Breakout = new "
                "directional decision by larger participants. "
                "Edge is in catching the early part of the "
                "expansion before the crowd."
            ),
            edge_fragility="moderate",
            edge_notes=(
                "Edge degrades in low-volatility/ranging markets "
                "where compressions resolve sideways. Works best "
                "when broader market has direction."
            ),
        )

    def _pullback_to_ema(self) -> PatternTemplate:
        return PatternTemplate(
            pattern_id="pullback_to_ema",
            name="Pullback to Moving Average",
            category="momentum",
            description=(
                "Stock in a trend pulls back to a key moving "
                "average (20 EMA, 50 SMA) and bounces. "
                "Edge: buying the dip in a trend at a "
                "statistically significant level."
            ),
            aliases=[
                "ema pullback", "moving average bounce",
                "buy the dip", "trend pullback",
                "20 ema pullback", "50 sma bounce",
            ],
            hard_rules=[
                {
                    "name": "trend_exists",
                    "field": "trend_score",
                    "operator": "gte",
                    "value": 0.5,
                    "description": "Must be in a defined trend",
                },
                {
                    "name": "near_ema",
                    "field": "distance_from_ema_atr",
                    "operator": "lte",
                    "value": 0.5,
                    "description": "Price must be within 0.5 ATR of EMA",
                },
            ],
            soft_scorers=[
                {
                    "name": "pullback_quality",
                    "weight": 0.30,
                    "description": "Orderly pullback vs chaotic",
                },
                {
                    "name": "trend_strength",
                    "weight": 0.25,
                    "description": "How strong is the underlying trend",
                },
                {
                    "name": "volume_dry_up",
                    "weight": 0.20,
                    "description": "Volume should decrease on pullback",
                },
                {
                    "name": "bounce_signal",
                    "weight": 0.25,
                    "description": "Early signs of bounce (hammer, engulfing)",
                },
            ],
            feature_keys=[
                "trend_score", "distance_from_ema_atr",
                "pullback_candle_count", "pullback_volume_ratio",
                "rsi_at_entry", "bounce_candle_strength",
                "higher_tf_trend_score", "atr_percentile",
            ],
            initial_centroids=[
                {
                    "trend_score": 0.7,
                    "distance_from_ema_atr": 0.2,
                    "pullback_candle_count": 3,
                    "pullback_volume_ratio": 0.6,
                    "rsi_at_entry": 45,
                    "bounce_candle_strength": 0.7,
                    "higher_tf_trend_score": 0.7,
                    "atr_percentile": 50,
                },
            ],
            typical_win_rate=0.58,
            typical_win_rate_range=(0.50, 0.65),
            typical_avg_winner_r=1.8,
            typical_avg_loser_r=-0.85,
            regime_performance={
                "trending_up": {"win_rate": 0.65, "avg_r": 0.70},
                "ranging": {"win_rate": 0.40, "avg_r": -0.15},
            },
            best_regimes=["trending_up", "strong_momentum"],
            worst_regimes=["ranging", "trending_down"],
            teaching_questions=[
                {
                    "id": "which_ema",
                    "question": "Which moving average? 9/20/50 EMA or SMA?",
                    "maps_to": "ema_period",
                    "type": "selection",
                },
                {
                    "id": "bounce_confirmation",
                    "question": "Do you wait for a bounce candle or enter at the EMA touch?",
                    "maps_to": "entry_timing",
                    "type": "judgment",
                },
            ],
            edge_source="Trend continuation at mean",
            edge_fragility="moderate",
        )

    def _vwap_reclaim(self) -> PatternTemplate:
        return PatternTemplate(
            pattern_id="vwap_reclaim",
            name="VWAP Reclaim",
            category="mean_reversion",
            description=(
                "Price drops below VWAP then reclaims it with "
                "conviction. Edge: institutional re-accumulation."
            ),
            aliases=[
                "vwap bounce", "vwap reclaim long",
                "vwap cross", "reclaim vwap",
            ],
            hard_rules=[
                {
                    "name": "was_below_vwap",
                    "field": "was_below_vwap",
                    "operator": "eq",
                    "value": 1,
                    "description": "Price must have been below VWAP",
                },
                {
                    "name": "now_above_vwap",
                    "field": "price_vs_vwap",
                    "operator": "gt",
                    "value": 0,
                    "description": "Price now reclaimed above VWAP",
                },
            ],
            soft_scorers=[
                {
                    "name": "reclaim_strength",
                    "weight": 0.35,
                    "description": "How convincingly did price reclaim",
                },
                {
                    "name": "volume_on_reclaim",
                    "weight": 0.30,
                    "description": "Volume should pick up on reclaim",
                },
                {
                    "name": "time_below",
                    "weight": 0.15,
                    "description": "Brief dip below is better than extended",
                },
                {
                    "name": "premarket_context",
                    "weight": 0.20,
                    "description": "Stock gapped up or has positive catalyst",
                },
            ],
            feature_keys=[
                "price_vs_vwap", "time_below_vwap_minutes",
                "reclaim_candle_strength", "reclaim_volume_ratio",
                "rsi_at_entry", "gap_pct", "atr_percentile",
            ],
            initial_centroids=[
                {
                    "price_vs_vwap": 0.1,
                    "time_below_vwap_minutes": 15,
                    "reclaim_candle_strength": 0.7,
                    "reclaim_volume_ratio": 1.8,
                    "rsi_at_entry": 48,
                    "gap_pct": 2.0,
                    "atr_percentile": 60,
                },
            ],
            typical_win_rate=0.55,
            best_regimes=["strong_momentum", "trending_up"],
            worst_regimes=["trending_down", "weak_market"],
            teaching_questions=[
                {
                    "id": "time_window",
                    "question": "How quickly must price reclaim VWAP? Minutes?",
                    "maps_to": "time_below_vwap_minutes",
                    "type": "threshold",
                },
            ],
            edge_source="Institutional VWAP execution creates support",
            edge_fragility="moderate",
        )

    def _opening_range_breakout(self) -> PatternTemplate:
        return PatternTemplate(
            pattern_id="opening_range_breakout",
            name="Opening Range Breakout",
            category="breakout",
            description=(
                "Trade the breakout of the first N minutes "
                "range (typically 15-30 min). "
                "Edge: initial balance sets the tone."
            ),
            aliases=[
                "orb", "opening range", "first 15 min breakout",
                "first 30 min breakout", "initial balance breakout",
            ],
            hard_rules=[
                {
                    "name": "within_opening_window",
                    "field": "minutes_since_open",
                    "operator": "between",
                    "value": [15, 120],
                    "description": "Must be after opening range forms but within 2 hours",
                },
            ],
            typical_win_rate=0.52,
            best_regimes=["trending_up", "gap_and_go"],
            worst_regimes=["ranging", "low_volatility"],
            edge_source="Opening range establishes institutional direction",
            edge_fragility="moderate",
        )

    def _supply_demand_zone(self) -> PatternTemplate:
        return PatternTemplate(
            pattern_id="supply_demand_zone",
            name="Supply/Demand Zone",
            category="reversal",
            description=(
                "Price returns to a zone where strong "
                "buying/selling previously occurred. "
                "Edge: unfilled institutional orders."
            ),
            aliases=[
                "supply zone", "demand zone", "order block",
                "institutional zone", "fresh level",
            ],
            typical_win_rate=0.55,
            best_regimes=["ranging", "trending_up"],
            edge_source="Unfilled institutional order flow",
            edge_fragility="moderate",
        )

    def _momentum_continuation(self) -> PatternTemplate:
        return PatternTemplate(
            pattern_id="momentum_continuation",
            name="Momentum Continuation",
            category="momentum",
            description=(
                "Strong momentum stock pauses briefly "
                "then continues. Trade the continuation."
            ),
            aliases=[
                "momentum play", "follow through",
                "continuation", "strong stock",
            ],
            typical_win_rate=0.53,
            best_regimes=["strong_momentum", "trending_up"],
            edge_source="Momentum persistence",
            edge_fragility="fragile",
        )

    def _mean_reversion_extreme(self) -> PatternTemplate:
        return PatternTemplate(
            pattern_id="mean_reversion_extreme",
            name="Mean Reversion from Extreme",
            category="mean_reversion",
            description=(
                "Price reaches extreme extension from mean. "
                "Fade the extreme expecting reversion."
            ),
            aliases=[
                "oversold bounce", "overbought short",
                "rubber band", "extreme extension",
                "rsi extreme",
            ],
            typical_win_rate=0.60,
            best_regimes=["ranging", "high_volatility"],
            worst_regimes=["strong_momentum"],
            edge_source="Statistical mean reversion",
            edge_fragility="moderate",
        )

    def _flag_pattern(self) -> PatternTemplate:
        return PatternTemplate(
            pattern_id="flag_pattern",
            name="Bull/Bear Flag",
            category="momentum",
            description=(
                "After a strong impulse move (pole), "
                "price consolidates in a flag/pennant. "
                "Trade the breakout from the flag."
            ),
            aliases=[
                "bull flag", "bear flag", "pennant",
                "flag breakout", "pole and flag",
            ],
            typical_win_rate=0.55,
            best_regimes=["trending_up", "strong_momentum"],
            edge_source="Continuation after institutional impulse",
            edge_fragility="moderate",
        )
    