# attribution.py
# strategy_engine/attribution.py
"""
Unified 3-Tier Attribution Orchestrator.

Tier 1 — User-declared strategy
Tier 2 — Auto pattern matching (StrategyMatcher)
Tier 3 — Override detection (OverrideDetector)

Resolves final strategy attribution and produces complete analysis.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from core.enums import (
    AttributionTier,
    ConditionStatus,
    OverrideSeverity,
)
from strategy_engine.conditions.base_condition import MarketSnapshot, TradeContext
from strategy_engine.override_detector import (
    OverrideDetector,
    OverrideReport,
    SessionContext,
)
from strategy_engine.strategy_matcher import (
    MatcherOutput,
    StrategyMatchResult,
    StrategyMatcher,
)
from strategy_engine.strategy_registry import StrategyRegistry

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  ATTRIBUTION RESULT
# ══════════════════════════════════════════════

@dataclass
class AttributionResult:
    """Complete 3-tier attribution output for a trade."""
    trade_id: Optional[str] = None

    # ── Tier 1 ──
    tier1_declared: bool = False
    tier1_strategy_id: Optional[str] = None

    # ── Tier 2 ──
    tier2_output: Optional[MatcherOutput] = None
    tier2_best_match_id: Optional[str] = None
    tier2_best_match_score: float = 0.0

    # ── Tier 3 ──
    tier3_override_report: Optional[OverrideReport] = None

    # ── Resolved ──
    resolved_strategy_id: Optional[str] = None
    resolved_tier: Optional[AttributionTier] = None
    confidence_score: float = 0.0              # 0.0 - 1.0
    discipline_score: float = 100.0            # 0.0 - 100.0

    # ── Insights ──
    insights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "tier1": {
                "declared": self.tier1_declared,
                "strategy_id": self.tier1_strategy_id,
            },
            "tier2": {
                "best_match_id": self.tier2_best_match_id,
                "best_match_score": round(self.tier2_best_match_score, 4),
                "matches_count": len(self.tier2_output.matches) if self.tier2_output else 0,
                "valid_matches": self.tier2_output.valid_matches if self.tier2_output else 0,
            },
            "tier3": self.tier3_override_report.summary() if self.tier3_override_report else {},
            "resolved": {
                "strategy_id": self.resolved_strategy_id,
                "tier": self.resolved_tier.value if self.resolved_tier else None,
                "confidence": round(self.confidence_score, 4),
                "discipline_score": round(self.discipline_score, 1),
            },
            "insights": self.insights,
        }


# ══════════════════════════════════════════════
#  ATTRIBUTION ENGINE
# ══════════════════════════════════════════════

class AttributionEngine:
    """
    Orchestrates the 3-tier attribution pipeline.

    Flow:
    1. Check Tier 1 (user-declared strategy)
    2. Run Tier 2 (auto-match against all strategies)
    3. Resolve strategy (declared vs auto-matched)
    4. Run Tier 3 (override detection against resolved strategy)
    5. Compute final confidence & discipline scores
    6. Generate insights
    """

    # Minimum Tier 2 score to consider a valid auto-match
    MIN_MATCH_SCORE = 0.35
    # Score above which auto-match is considered high confidence
    HIGH_CONFIDENCE_THRESHOLD = 0.70
    # Tier 1 + Tier 2 agreement bonus
    AGREEMENT_BONUS = 0.15

    def __init__(self, registry: StrategyRegistry):
        self._registry = registry
        self._matcher = StrategyMatcher(registry)
        self._override_detector = OverrideDetector()

    def evaluate(
        self,
        trade: TradeContext,
        market: MarketSnapshot,
        session: SessionContext | None = None,
    ) -> AttributionResult:
        """Run full 3-tier attribution pipeline for a trade."""

        result = AttributionResult(
            trade_id=str(getattr(trade, "trade_id", None)),
        )

        # ══════════════════════════════════
        # TIER 1 — User Declared
        # ══════════════════════════════════
        if trade.declared_strategy_id:
            result.tier1_declared = True
            result.tier1_strategy_id = trade.declared_strategy_id
            template = self._registry.get(trade.declared_strategy_id)
            if template is None:
                result.insights.append(
                    f"⚠️ Declared strategy '{trade.declared_strategy_id}' not found in registry"
                )

        # ══════════════════════════════════
        # TIER 2 — Auto Match
        # ══════════════════════════════════
        tier2_output = self._matcher.match(trade, market)
        result.tier2_output = tier2_output
        result.tier2_best_match_id = tier2_output.best_match_id
        result.tier2_best_match_score = tier2_output.best_match_score

        # ══════════════════════════════════
        # RESOLVE STRATEGY
        # ══════════════════════════════════
        resolved_id, resolved_tier, confidence = self._resolve_strategy(result)
        result.resolved_strategy_id = resolved_id
        result.resolved_tier = resolved_tier
        result.confidence_score = confidence

        # ══════════════════════════════════
        # TIER 3 — Override Detection
        # ══════════════════════════════════
        if resolved_id:
            template = self._registry.get(resolved_id)
            if template:
                # Get the match result for the resolved strategy
                match_for_resolved = None
                if tier2_output:
                    for m in tier2_output.matches:
                        if m.strategy_id == resolved_id:
                            match_for_resolved = m
                            break

                override_report = self._override_detector.detect(
                    trade=trade,
                    template=template,
                    market=market,
                    session=session,
                    match_result=match_for_resolved,
                )
                result.tier3_override_report = override_report
                result.discipline_score = override_report.discipline_score

                # Adjust confidence based on overrides
                if override_report.has_overrides:
                    penalty = override_report.composite_deviation_score * 0.3
                    result.confidence_score = max(0.0, result.confidence_score - penalty)

        # ══════════════════════════════════
        # GENERATE INSIGHTS
        # ══════════════════════════════════
        self._generate_insights(result)

        logger.info(
            "Attribution complete for trade %s: resolved=%s (tier=%s, confidence=%.2f, discipline=%.1f)",
            result.trade_id,
            result.resolved_strategy_id,
            result.resolved_tier.value if result.resolved_tier else "none",
            result.confidence_score,
            result.discipline_score,
        )

        return result

    def _resolve_strategy(
        self,
        result: AttributionResult,
    ) -> tuple[Optional[str], Optional[AttributionTier], float]:
        """
        Resolve which strategy the trade belongs to.

        Priority logic:
        1. If declared AND auto-match agrees → high confidence
        2. If declared AND auto-match disagrees → use declared, lower confidence
        3. If no declaration, strong auto-match → use auto-match
        4. If no declaration, weak auto-match → Tier 3 override analysis
        5. If nothing matches → unattributed
        """
        tier1_id = result.tier1_strategy_id
        tier2_id = result.tier2_best_match_id
        tier2_score = result.tier2_best_match_score

        # Case 1: Declared + auto-match agree
        if tier1_id and tier2_id and tier1_id == tier2_id:
            confidence = min(1.0, tier2_score + self.AGREEMENT_BONUS)
            return tier1_id, AttributionTier.TIER1_DECLARED, confidence

        # Case 2: Declared, but auto-match disagrees or doesn't match
        if tier1_id:
            if tier2_id and tier2_id != tier1_id:
                # Check how the declared strategy actually scored
                declared_score = 0.0
                if result.tier2_output:
                    for m in result.tier2_output.matches:
                        if m.strategy_id == tier1_id:
                            declared_score = m.match_score
                            break

                # If declared scored reasonably, trust it
                if declared_score >= self.MIN_MATCH_SCORE:
                    confidence = declared_score * 0.8
                    result.insights.append(
                        f"🔄 Declared '{tier1_id}' scored {declared_score:.2f}, "
                        f"but auto-match prefers '{tier2_id}' ({tier2_score:.2f})"
                    )
                    return tier1_id, AttributionTier.TIER1_DECLARED, confidence
                else:
                    # Declared scored poorly — flag as potential override
                    result.insights.append(
                        f"⚠️ Declared '{tier1_id}' scored poorly ({declared_score:.2f}). "
                        f"Auto-match suggests '{tier2_id}' ({tier2_score:.2f}). "
                        f"Trade may not align with declared strategy."
                    )
                    return tier1_id, AttributionTier.TIER3_OVERRIDE, max(0.2, declared_score)
            else:
                # No auto-match at all, trust declaration
                return tier1_id, AttributionTier.TIER1_DECLARED, 0.6

        # Case 3: No declaration, strong auto-match
        if tier2_id and tier2_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return tier2_id, AttributionTier.TIER2_AUTO_MATCHED, tier2_score

        # Case 4: No declaration, weak auto-match
        if tier2_id and tier2_score >= self.MIN_MATCH_SCORE:
            result.insights.append(
                f"🔍 Auto-matched to '{tier2_id}' with moderate confidence ({tier2_score:.2f})"
            )
            return tier2_id, AttributionTier.TIER2_AUTO_MATCHED, tier2_score

        # Case 5: Nothing matches
        result.insights.append(
            "❌ No strategy matched. This trade doesn't align with any defined strategy."
        )
        return None, AttributionTier.TIER3_OVERRIDE, 0.0

    def _generate_insights(self, result: AttributionResult) -> None:
        """Generate human-readable coaching insights from attribution analysis."""
        override_report = result.tier3_override_report

        if not override_report or not override_report.has_overrides:
            if result.resolved_strategy_id:
                result.insights.append(
                    f"✅ Clean execution against '{result.resolved_strategy_id}' — "
                    f"all rules followed"
                )
            return

        # Severity-based insights
        if override_report.max_severity == OverrideSeverity.CRITICAL:
            result.insights.append(
                "🚨 CRITICAL rule violations detected. "
                "Review this trade carefully for emotional or impulsive behavior."
            )

        # Specific override insights
        by_type = override_report.by_type
        if "rapid_re_entry" in by_type:
            result.insights.append(
                "🔁 Rapid re-entry after loss detected. "
                "Consider implementing a mandatory cooldown rule."
            )
        if "no_stop_set" in by_type:
            result.insights.append(
                "🛑 No stop loss set. This is the highest-risk rule violation."
            )
        if "oversizing" in by_type:
            result.insights.append(
                "📏 Position sizing violation. Risk per trade exceeds strategy limits."
            )
        if "missing_confirmation" in by_type:
            count = by_type["missing_confirmation"]
            result.insights.append(
                f"📋 {count} required confirmation(s) not met. "
                f"Entry may have been premature."
            )
        if "avoid_condition_violated" in by_type:
            result.insights.append(
                "⚠️ Trade taken despite avoid conditions being active. "
                "Review setup selectivity."
            )

        # Discipline summary
        if result.discipline_score >= 80:
            result.insights.append(
                f"📊 Discipline score: {result.discipline_score:.0f}/100 — Minor deviations"
            )
        elif result.discipline_score >= 50:
            result.insights.append(
                f"📊 Discipline score: {result.discipline_score:.0f}/100 — "
                f"Significant deviations from strategy rules"
            )
        else:
            result.insights.append(
                f"📊 Discipline score: {result.discipline_score:.0f}/100 — "
                f"Major discipline breakdown. Focus on process over outcome."
            )