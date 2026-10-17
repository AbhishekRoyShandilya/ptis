# strategy_engine/strategy_dna/schema.py
"""
Strategy DNA data structures.
Represents the COMPLETE knowledge of a personal trading strategy.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from core.enums import MarketRegime, SessionType


@dataclass
class SetupCondition:
    name: str
    description: str
    importance: str               # critical, important, preferred, nice_to_have
    what_good_looks_like: str
    what_bad_looks_like: str


@dataclass
class EmotionalTrap:
    trap: str
    description: str
    solution: str
    frequency: str                # happens_often, sometimes, rarely


@dataclass
class RegimeGuidance:
    regime: str
    notes: str
    expected_win_rate: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class MarketConditionProfile:
    """Complete profile of when this strategy works/fails."""
    works_best_regimes: List[RegimeGuidance]
    works_poorly_regimes: List[RegimeGuidance]
    works_best_sessions: List[Dict]
    works_poorly_sessions: List[Dict]
    works_best_additional: List[str]
    works_poorly_additional: List[Dict]
    regime_sizing: Dict[str, str]


@dataclass
class StrategyRelationships:
    fallback_strategies: List[Dict]
    conflicts_with: List[Dict]
    pairs_well_with: List[Dict]
    regime_alternatives: Dict[str, str]


@dataclass
class PersonalInsights:
    strengths: List[str]
    weaknesses: List[str]
    edge_conditions: List[str]
    learned_statistics: Dict[str, Any]


@dataclass
class StrategyDNA:
    """
    Complete DNA of a personal trading strategy.
    This is the knowledge that AI agents use to:
    - Evaluate trade quality
    - Detect behavioral deviations
    - Recommend regime-appropriate strategies
    - Coach on strategy-specific improvements
    - Score discipline per strategy
    """

    # Identity
    name: str
    display_name: str
    short_code: str
    version: int
    category: str
    timeframe_primary: str
    timeframe_confirmation: Optional[str]
    typical_hold: str
    description: str

    # Entry logic
    entry_narrative: str
    setup_conditions: List[SetupCondition]
    a_plus_conditions: List[str]
    skip_conditions: List[str]

    # Execution
    entry_trigger: str
    entry_timing: str
    common_entry_mistakes: List[str]
    ideal_entry_description: str

    # Risk management
    stop_method: str
    stop_description: str
    stop_common_mistakes: List[str]
    sizing_method: str
    risk_per_trade: str
    max_risk: str
    size_adjustments: Dict[str, str]
    target_primary: str
    target_secondary: str
    target_method: str

    # Exit rules
    profit_taking_rules: List[Dict]
    time_exit_rules: List[Dict]
    common_exit_mistakes: List[str]

    # Market conditions
    market_conditions: MarketConditionProfile

    # Psychology
    emotional_traps: List[EmotionalTrap]
    ideal_mental_state: List[str]
    do_not_trade_when: List[str]

    # Relationships
    relationships: StrategyRelationships

    # Personal insights
    personal_insights: PersonalInsights

    # Checklist
    checklist: List[str]

    # Tags
    tags: List[str]

    # ─── Derived Methods ───

    @property
    def critical_conditions(self) -> List[SetupCondition]:
        return [c for c in self.setup_conditions if c.importance == "critical"]

    @property
    def best_regimes(self) -> List[str]:
        return [r.regime for r in self.market_conditions.works_best_regimes]

    @property
    def worst_regimes(self) -> List[str]:
        return [r.regime for r in self.market_conditions.works_poorly_regimes]

    def get_regime_sizing(self, regime: str) -> str:
        return self.market_conditions.regime_sizing.get(regime, "standard_size")

    def get_regime_guidance(self, regime: str) -> Optional[RegimeGuidance]:
        """Get specific guidance for a regime."""
        for r in self.market_conditions.works_best_regimes:
            if r.regime == regime:
                return r
        for r in self.market_conditions.works_poorly_regimes:
            if r.regime == regime:
                return r
        return None

    def should_skip_in_regime(self, regime: str) -> bool:
        sizing = self.get_regime_sizing(regime)
        return sizing in ("skip", "skip_or_quarter_size")

    def get_fallback_for_regime(self, regime: str) -> Optional[str]:
        return self.relationships.regime_alternatives.get(regime)

    def to_agent_context(self) -> str:
        """
        Generate a text summary for LLM agent consumption.
        This is what gets injected into agent prompts.
        """
        ctx = f"""
STRATEGY: {self.display_name} ({self.short_code})
Category: {self.category} | Timeframe: {self.timeframe_primary} | Hold: {self.typical_hold}

DESCRIPTION: {self.description}

ENTRY LOGIC: {self.entry_narrative}

CRITICAL CONDITIONS:
{chr(10).join(f'  - {c.name}: {c.description}' for c in self.critical_conditions)}

A+ SETUP CONDITIONS:
{chr(10).join(f'  - {c}' for c in self.a_plus_conditions)}

SKIP WHEN:
{chr(10).join(f'  - {c}' for c in self.skip_conditions)}

COMMON ENTRY MISTAKES:
{chr(10).join(f'  - {m}' for m in self.common_entry_mistakes)}

STOP: {self.stop_method} — {self.stop_description}
TARGET: {self.target_primary} / {self.target_secondary}
RISK: {self.risk_per_trade} (max {self.max_risk})

WORKS BEST: {', '.join(self.best_regimes)}
WORKS POORLY: {', '.join(self.worst_regimes)}

PSYCHOLOGICAL TRAPS:
{chr(10).join(f'  - {t.trap}: {t.solution}' for t in self.emotional_traps)}

KNOWN WEAKNESSES:
{chr(10).join(f'  - {w}' for w in self.personal_insights.weaknesses)}

DO NOT TRADE WHEN:
{chr(10).join(f'  - {d}' for d in self.do_not_trade_when)}
"""
        return ctx.strip()

    def to_regime_summary(self, current_regime: str) -> str:
        """Generate regime-specific guidance for agents."""
        guidance = self.get_regime_guidance(current_regime)
        sizing = self.get_regime_sizing(current_regime)
        fallback = self.get_fallback_for_regime(current_regime)

        summary = f"Strategy: {self.display_name} | Current Regime: {current_regime}\n"

        if guidance:
            summary += f"Guidance: {guidance.notes}\n"
            if guidance.expected_win_rate:
                summary += f"Expected Win Rate: {guidance.expected_win_rate}\n"
            if guidance.recommendation:
                summary += f"Recommendation: {guidance.recommendation}\n"

        summary += f"Sizing: {sizing}\n"

        if self.should_skip_in_regime(current_regime):
            summary += f"⚠️ RECOMMENDATION: SKIP this strategy in {current_regime}\n"
            if fallback:
                summary += f"Consider instead: {fallback}\n"

        return summary