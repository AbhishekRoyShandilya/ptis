# strategy_engine/strategy_matcher.py
"""
Tier 2: Auto-match a trade against all loaded strategy templates.
Evaluates conditions, produces scored matches, ranks candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from core.enums import (
    AttributionTier,
    ConditionStatus,
    MarketRegime,
    TradeDirection,
)
from strategy_engine.conditions.base_condition import (
    ConditionResult,
    MarketSnapshot,
    TradeContext,
)
from strategy_engine.conditions.registry import evaluate_condition
from strategy_engine.strategy_registry import StrategyRegistry
from strategy_engine.templates.loader import ConditionDef, StrategyTemplate

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  MATCH RESULT STRUCTURES
# ══════════════════════════════════════════════

@dataclass
class ConditionEvalResult:
    """Full evaluation of one condition for one strategy."""
    condition_def: ConditionDef
    result: ConditionResult
    category: str = "confirmation"  # "confirmation", "avoid", "invalidation"


@dataclass
class StrategyMatchResult:
    """Result of matching a single strategy to a trade."""
    strategy_id: str
    strategy_name: str

    # Scoring
    match_score: float = 0.0               # 0.0 - 1.0
    required_met: int = 0
    required_total: int = 0
    supporting_met: int = 0
    supporting_total: int = 0
    weighted_score: float = 0.0
    max_weighted_score: float = 0.0

    # Avoid / Invalidation
    avoid_triggered: int = 0
    invalidation_triggered: int = 0

    # Flags
    all_required_met: bool = False
    regime_compatible: bool = True
    regime_preferred: bool = False
    time_window_ok: bool = True
    direction_compatible: bool = True

    # Disqualification
    disqualified: bool = False
    disqualification_reason: str = ""

    # Detailed results
    condition_results: list[ConditionEvalResult] = field(default_factory=list)

    @property
    def is_valid_match(self) -> bool:
        return (
            not self.disqualified
            and self.all_required_met
            and self.avoid_triggered == 0
            and self.direction_compatible
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "score": round(self.match_score, 4),
            "required_met": self.required_met,
            "required_total": self.required_total,
            "supporting_met": self.supporting_met,
            "supporting_total": self.supporting_total,
            "avoid_triggered": self.avoid_triggered,
            "invalidation_triggered": self.invalidation_triggered,
            "all_required_met": self.all_required_met,
            "regime_compatible": self.regime_compatible,
            "regime_preferred": self.regime_preferred,
            "direction_compatible": self.direction_compatible,
            "disqualified": self.disqualified,
            "disqualification_reason": self.disqualification_reason,
            "is_valid_match": self.is_valid_match,
            "conditions": [
                {
                    "type": cr.result.condition_type,
                    "category": cr.category,
                    "status": cr.result.status.value,
                    "required": cr.condition_def.required,
                    "weight": cr.condition_def.weight,
                    "details": cr.result.details,
                    "actual": cr.result.actual_value,
                    "expected": cr.result.expected_value,
                }
                for cr in self.condition_results
            ],
        }


@dataclass
class MatcherOutput:
    """Full output of the strategy matcher for a trade."""
    trade_id: Optional[str] = None
    matches: list[StrategyMatchResult] = field(default_factory=list)
    best_match: Optional[StrategyMatchResult] = None
    best_match_id: Optional[str] = None
    best_match_score: float = 0.0
    strategies_evaluated: int = 0
    valid_matches: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "strategies_evaluated": self.strategies_evaluated,
            "valid_matches": self.valid_matches,
            "best_match_id": self.best_match_id,
            "best_match_score": round(self.best_match_score, 4),
            "matches": [m.to_dict() for m in sorted(
                self.matches, key=lambda x: x.match_score, reverse=True
            )],
        }


# ══════════════════════════════════════════════
#  STRATEGY MATCHER
# ══════════════════════════════════════════════

class StrategyMatcher:
    """
    Evaluates a trade against all strategy templates.
    Produces ranked match results with condition-level detail.
    """

    def __init__(self, registry: StrategyRegistry):
        self._registry = registry

    def match(
        self,
        trade: TradeContext,
        market: MarketSnapshot,
        candidates: list[StrategyTemplate] | None = None,
    ) -> MatcherOutput:
        """
        Match a trade against candidate strategies.
        If candidates is None, uses all active strategies from registry.
        """
        if candidates is None:
            candidates = self._registry.get_candidates(
                direction=trade.direction,
                regime=trade.market_regime,
            )

        output = MatcherOutput(trade_id=str(getattr(trade, "trade_id", None)))
        output.strategies_evaluated = len(candidates)

        for template in candidates:
            result = self._evaluate_strategy(template, trade, market)
            output.matches.append(result)

        # Sort by score descending
        output.matches.sort(key=lambda r: r.match_score, reverse=True)

        # Find best valid match
        valid = [m for m in output.matches if m.is_valid_match]
        output.valid_matches = len(valid)

        if valid:
            output.best_match = valid[0]
            output.best_match_id = valid[0].strategy_id
            output.best_match_score = valid[0].match_score

        logger.info(
            "Strategy matching complete: %d evaluated, %d valid, best=%s (%.2f)",
            output.strategies_evaluated,
            output.valid_matches,
            output.best_match_id or "none",
            output.best_match_score,
        )

        return output

    def match_single(
        self,
        strategy_id: str,
        trade: TradeContext,
        market: MarketSnapshot,
    ) -> StrategyMatchResult | None:
        """Evaluate a trade against a specific strategy."""
        template = self._registry.get(strategy_id)
        if template is None:
            logger.warning("Strategy '%s' not found in registry", strategy_id)
            return None
        return self._evaluate_strategy(template, trade, market)

    def _evaluate_strategy(
        self,
        template: StrategyTemplate,
        trade: TradeContext,
        market: MarketSnapshot,
    ) -> StrategyMatchResult:
        """Evaluate all conditions for a single strategy against a trade."""
        result = StrategyMatchResult(
            strategy_id=template.id,
            strategy_name=template.name,
        )

        # ── Pre-checks ──
        if not template.direction_compatible(trade.direction):
            result.direction_compatible = False
            result.disqualified = True
            result.disqualification_reason = "Direction incompatible"
            return result

        if trade.market_regime:
            result.regime_compatible = template.is_regime_compatible(trade.market_regime)
            result.regime_preferred = template.is_regime_preferred(trade.market_regime)

            if not result.regime_compatible:
                result.disqualified = True
                result.disqualification_reason = f"Regime '{trade.market_regime.value}' in avoid list"
                return result

        # ── Time window check ──
        if template.time_window:
            tw_result = evaluate_condition(
                "time_inside_window",
                {
                    "start": template.time_window.start,
                    "end": template.time_window.end,
                    "timezone": template.time_window.timezone,
                },
                trade, market,
            )
            result.time_window_ok = tw_result.is_met
            result.condition_results.append(ConditionEvalResult(
                condition_def=ConditionDef(
                    type="time_inside_window",
                    params={},
                    required=True,
                    description="Trade within strategy time window",
                ),
                result=tw_result,
                category="time_window",
            ))

        # ── Entry Confirmations ──
        required_total = 0
        required_met = 0
        supporting_total = 0
        supporting_met = 0
        weighted_score = 0.0
        max_weighted = 0.0

        for cond_def in template.entry_confirmations:
            cond_result = evaluate_condition(
                cond_def.type,
                cond_def.params,
                trade,
                market,
                required=cond_def.required,
                weight=cond_def.weight,
            )

            result.condition_results.append(ConditionEvalResult(
                condition_def=cond_def,
                result=cond_result,
                category="confirmation",
            ))

            max_weighted += cond_def.weight

            if cond_def.required:
                required_total += 1
                if cond_result.is_met:
                    required_met += 1
                    weighted_score += cond_result.weighted_score
            else:
                supporting_total += 1
                if cond_result.is_met:
                    supporting_met += 1
                    weighted_score += cond_result.weighted_score

        result.required_met = required_met
        result.required_total = required_total
        result.supporting_met = supporting_met
        result.supporting_total = supporting_total
        result.all_required_met = (required_met == required_total)
        result.weighted_score = weighted_score
        result.max_weighted_score = max_weighted

        # ── Avoid Conditions ──
        for cond_def in template.avoid_conditions:
            cond_result = evaluate_condition(
                cond_def.type,
                cond_def.params,
                trade,
                market,
                required=False,
                weight=cond_def.weight,
            )
            # strategy_engine/strategy_matcher.py (continued from _evaluate_strategy avoid conditions)

            result.condition_results.append(ConditionEvalResult(
                condition_def=cond_def,
                result=cond_result,
                category="avoid",
            ))

            # For avoid conditions: MET = condition IS triggered = BAD
            if cond_result.is_met:
                result.avoid_triggered += 1

        # ── Invalidation Conditions ──
        for cond_def in template.invalidation_conditions:
            cond_result = evaluate_condition(
                cond_def.type,
                cond_def.params,
                trade,
                market,
                required=False,
                weight=cond_def.weight,
            )
            result.condition_results.append(ConditionEvalResult(
                condition_def=cond_def,
                result=cond_result,
                category="invalidation",
            ))

            if cond_result.is_met:
                result.invalidation_triggered += 1

        # ── Compute Final Match Score ──
        result.match_score = self._compute_match_score(result, template)

        # ── Disqualification check ──
        if not result.all_required_met:
            result.disqualified = True
            result.disqualification_reason = (
                f"Missing required confirmations: {required_met}/{required_total}"
            )

        if result.avoid_triggered > 0:
            result.disqualified = True
            triggered_avoids = [
                cr.condition_def.description or cr.condition_def.type
                for cr in result.condition_results
                if cr.category == "avoid" and cr.result.is_met
            ]
            result.disqualification_reason = (
                f"Avoid conditions triggered: {', '.join(triggered_avoids)}"
            )

        return result

    def _compute_match_score(
        self,
        result: StrategyMatchResult,
        template: StrategyTemplate,
    ) -> float:
        """
        Compute composite match score (0.0 - 1.0).

        Scoring formula:
        - Base score = weighted_met / max_weighted (40%)
        - Required coverage = required_met / required_total (30%)
        - Supporting bonus = supporting_met / supporting_total (15%)
        - Regime bonus = 10% if preferred, 5% if compatible
        - Penalty: -0.15 per avoid triggered, -0.10 per invalidation
        """
        if result.max_weighted_score == 0:
            return 0.0

        # Base weighted score (40%)
        base_score = (result.weighted_score / result.max_weighted_score) * 0.40

        # Required coverage (30%)
        required_score = 0.0
        if result.required_total > 0:
            required_score = (result.required_met / result.required_total) * 0.30

        # Supporting bonus (15%)
        supporting_score = 0.0
        if result.supporting_total > 0:
            supporting_score = (result.supporting_met / result.supporting_total) * 0.15

        # Regime bonus (10%)
        regime_score = 0.0
        if result.regime_preferred:
            regime_score = 0.10
        elif result.regime_compatible:
            regime_score = 0.05

        # Time window bonus (5%)
        time_score = 0.05 if result.time_window_ok else 0.0

        # Penalties
        avoid_penalty = result.avoid_triggered * 0.15
        invalidation_penalty = result.invalidation_triggered * 0.10

        total = (
            base_score
            + required_score
            + supporting_score
            + regime_score
            + time_score
            - avoid_penalty
            - invalidation_penalty
        )

        return max(0.0, min(1.0, total))