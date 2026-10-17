# behavioral/deviation_attribution.py
"""
Determines WHETHER a trade failure was caused by:
  A) Emotional / behavioral deviation (trader's fault)
  B) Market regime / condition change (not trader's fault)
  C) Strategy edge degradation (strategy needs review)
  D) Random variance (expected loss within edge)

This is the KEY intelligence that separates PTIS from any other system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from core.enums import (
    BehavioralEventType,
    MarketRegime,
    OverrideSeverity,
    OverrideType,
    TradeStatus,
)
from core.models import BehavioralEvent, OverrideEvent, StrategyAttribution, Trade
from strategy_engine.conditions.base_condition import MarketSnapshot

logger = logging.getLogger(__name__)


class DeviationCause:
    EMOTIONAL = "emotional"              # trader broke rules due to emotion
    MARKET_CONDITION = "market_condition" # market changed, not trader's fault
    STRATEGY_EDGE_LOSS = "strategy_edge" # strategy itself is losing edge
    RANDOM_VARIANCE = "random_variance"  # normal loss within expected distribution
    EXECUTION_ERROR = "execution_error"  # mechanical mistake, not emotional
    MIXED = "mixed"                      # combination of factors


@dataclass
class DeviationAnalysis:
    """Complete attribution of why a trade deviated from expected outcome."""
    trade_id: str
    primary_cause: str = DeviationCause.RANDOM_VARIANCE
    cause_confidence: float = 0.0       # 0-1

    # Breakdown
    emotional_contribution: float = 0.0   # 0-1
    market_contribution: float = 0.0      # 0-1
    strategy_contribution: float = 0.0    # 0-1
    variance_contribution: float = 0.0    # 0-1
    execution_contribution: float = 0.0   # 0-1

    # Evidence
    emotional_evidence: list[str] = field(default_factory=list)
    market_evidence: list[str] = field(default_factory=list)
    strategy_evidence: list[str] = field(default_factory=list)

    # Regime context
    expected_regime: Optional[str] = None
    actual_regime: Optional[str] = None
    regime_changed: bool = False

    # Risk context
    risk_was_appropriate: bool = True
    risk_adaptation_needed: bool = False
    suggested_risk_adjustment: Optional[str] = None

    # Verdict
    is_traders_fault: bool = False
    is_market_fault: bool = False
    is_strategy_fault: bool = False
    is_normal_loss: bool = True

    # Action items
    action_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "primary_cause": self.primary_cause,
            "cause_confidence": round(self.cause_confidence, 2),
            "breakdown": {
                "emotional": round(self.emotional_contribution, 3),
                "market": round(self.market_contribution, 3),
                "strategy": round(self.strategy_contribution, 3),
                "variance": round(self.variance_contribution, 3),
                "execution": round(self.execution_contribution, 3),
            },
            "emotional_evidence": self.emotional_evidence,
            "market_evidence": self.market_evidence,
            "strategy_evidence": self.strategy_evidence,
            "regime": {
                "expected": self.expected_regime,
                "actual": self.actual_regime,
                "changed": self.regime_changed,
            },
            "risk_context": {
                "was_appropriate": self.risk_was_appropriate,
                "adaptation_needed": self.risk_adaptation_needed,
                "suggested_adjustment": self.suggested_risk_adjustment,
            },
            "verdict": {
                "traders_fault": self.is_traders_fault,
                "market_fault": self.is_market_fault,
                "strategy_fault": self.is_strategy_fault,
                "normal_loss": self.is_normal_loss,
            },
            "action_items": self.action_items,
        }


class DeviationAttributionEngine:
    """
    Analyzes a losing/underperforming trade and determines the root cause.

    Input:
    - Trade data
    - Attribution data (overrides, condition evals)
    - Market snapshot at entry AND at exit
    - Strategy rules
    - Recent strategy performance

    Output:
    - DeviationAnalysis with cause breakdown
    """

    def analyze(
        self,
        trade: Trade,
        attribution: Optional[StrategyAttribution],
        overrides: list[OverrideEvent],
        behavioral_events: list[BehavioralEvent],
        entry_market: Optional[MarketSnapshot],
        exit_market: Optional[MarketSnapshot],
        recent_strategy_performance: Optional[dict] = None,
    ) -> DeviationAnalysis:
        """Run full deviation analysis on a trade."""
        analysis = DeviationAnalysis(trade_id=str(trade.id))

        # Only analyze losing or underperforming trades
        if trade.pnl and trade.pnl > 0 and (trade.r_multiple or 0) >= 1.0:
            analysis.primary_cause = DeviationCause.RANDOM_VARIANCE
            analysis.is_normal_loss = False  # it's a win
            analysis.variance_contribution = 1.0
            analysis.cause_confidence = 1.0
            return analysis

        # ══════════════════════════
        # 1. EMOTIONAL ANALYSIS
        # ══════════════════════════
        emotional_score = self._assess_emotional_contribution(
            trade, attribution, overrides, behavioral_events
        )
        analysis.emotional_contribution = emotional_score

        # ══════════════════════════
        # 2. MARKET CONDITION ANALYSIS
        # ══════════════════════════
        market_score = self._assess_market_contribution(
            trade, entry_market, exit_market
        )
        analysis.market_contribution = market_score

        # ══════════════════════════
        # 3. STRATEGY EDGE ANALYSIS
        # ══════════════════════════
        strategy_score = self._assess_strategy_contribution(
            trade, recent_strategy_performance
        )
        analysis.strategy_contribution = strategy_score

        # ══════════════════════════
        # 4. EXECUTION ERROR ANALYSIS
        # ══════════════════════════
        execution_score = self._assess_execution_contribution(
            trade, attribution, overrides
        )
        analysis.execution_contribution = execution_score

        # ══════════════════════════
        # 5. RANDOM VARIANCE
        # ══════════════════════════
        total_attributed = (
            emotional_score + market_score + strategy_score + execution_score
        )
        analysis.variance_contribution = max(0, 1.0 - total_attributed)

        # ══════════════════════════
        # DETERMINE PRIMARY CAUSE
        # ══════════════════════════
        scores = {
            DeviationCause.EMOTIONAL: emotional_score,
            DeviationCause.MARKET_CONDITION: market_score,
            DeviationCause.STRATEGY_EDGE_LOSS: strategy_score,
            DeviationCause.EXECUTION_ERROR: execution_score,
            DeviationCause.RANDOM_VARIANCE: analysis.variance_contribution,
        }

        primary = max(scores, key=scores.get)
        analysis.primary_cause = primary
        analysis.cause_confidence = scores[primary]

        # Check if mixed
        significant = [k for k, v in scores.items() if v > 0.25]
        if len(significant) >= 2:
            analysis.primary_cause = DeviationCause.MIXED

        # ══════════════════════════
        # SET VERDICTS
        # ══════════════════════════
        analysis.is_traders_fault = emotional_score > 0.4 or execution_score > 0.3
        analysis.is_market_fault = market_score > 0.4
        analysis.is_strategy_fault = strategy_score > 0.4
        analysis.is_normal_loss = analysis.variance_contribution > 0.5

        # ══════════════════════════
        # RISK CONTEXT
        # ══════════════════════════
        self._assess_risk_context(analysis, trade, entry_market, exit_market)

        # ══════════════════════════
        # ACTION ITEMS
        # ══════════════════════════
        self._generate_action_items(analysis)

        logger.info(
            "Deviation analysis for trade %s: primary=%s (%.0f%%), "
            "emotional=%.0f%%, market=%.0f%%, strategy=%.0f%%, variance=%.0f%%",
            trade.id, analysis.primary_cause,
            analysis.cause_confidence * 100,
            analysis.emotional_contribution * 100,
            analysis.market_contribution * 100,
            analysis.strategy_contribution * 100,
            analysis.variance_contribution * 100,
        )

        return analysis

    def _assess_emotional_contribution(
        self,
        trade: Trade,
        attribution: Optional[StrategyAttribution],
        overrides: list[OverrideEvent],
        behavioral: list[BehavioralEvent],
    ) -> float:
        score = 0.0
        evidence = []

        # Check overrides
        emotional_overrides = [
            o for o in overrides
            if o.override_type in (
                OverrideType.RAPID_RE_ENTRY,
                OverrideType.OVERSIZING,
                OverrideType.NO_STOP_SET,
                OverrideType.STOP_MOVED,
                OverrideType.MISSING_CONFIRMATION,
            )
        ]

        if emotional_overrides:
            severity_total = sum(o.severity_score for o in emotional_overrides)
            score += min(0.5, severity_total * 0.2)
            for o in emotional_overrides:
                evidence.append(f"Override: {o.description}")

        # Check behavioral events
        neg_behavioral = [b for b in behavioral if not b.is_positive]
        if neg_behavioral:
            score += min(0.3, len(neg_behavioral) * 0.15)
            for b in neg_behavioral:
                evidence.append(f"Behavioral: {b.event_type.value} — {b.description}")

        # Emotional state check
        high_risk_emotions = {
            EmotionalState.REVENGE, EmotionalState.FOMO,
            EmotionalState.FRUSTRATED, EmotionalState.GREEDY,
            EmotionalState.EUPHORIC,
        }
        if trade.emotion_pre and trade.emotion_pre in high_risk_emotions:
            score += 0.2
            evidence.append(f"Pre-trade emotion: {trade.emotion_pre.value}")

        # Discipline score
        if attribution and attribution.discipline_score is not None:
            if attribution.discipline_score < 50:
                score += 0.2
                evidence.append(
                    f"Low discipline score: {attribution.discipline_score:.0f}/100"
                )
            elif attribution.discipline_score < 70:
                score += 0.1

        # Cap and store evidence
        score = min(1.0, score)
        # Store evidence on the analysis object (will be set by caller)
        self._last_emotional_evidence = evidence
        return score

    def _assess_market_contribution(
        self,
        trade: Trade,
        entry_market: Optional[MarketSnapshot],
        exit_market: Optional[MarketSnapshot],
    ) -> float:
        score = 0.0
        evidence = []

        if not entry_market:
            return 0.0

        # Regime mismatch — now checked via learned profiles in strategy_agent
        # Hard-coded strategy rules removed in favor of AI-learned profiles

        # Regime changed during trade
        if entry_market and exit_market:
            if entry_market.regime != exit_market.regime:
                score += 0.3
                evidence.append(
                    f"Regime changed: {entry_market.regime.value} → "
                    f"{exit_market.regime.value if exit_market.regime else 'unknown'}"
                )

        # Volatility spike
        entry_atr = entry_market.atr
        if exit_market and entry_atr:
            exit_atr = exit_market.atr
            if exit_atr and exit_atr > entry_atr * 1.5:
                score += 0.2
                evidence.append(
                    f"Volatility spiked: ATR {entry_atr:.2f} → {exit_atr:.2f}"
                )

        # News / abnormal volume
        if exit_market and exit_market.relative_volume and exit_market.relative_volume > 3.0:
            score += 0.15
            evidence.append(
                f"Abnormal volume at exit: {exit_market.relative_volume:.1f}x"
            )

        score = min(1.0, score)
        self._last_market_evidence = evidence
        return score

    def _assess_strategy_contribution(
        self,
        trade: Trade,
        recent_performance: Optional[dict],
    ) -> float:
        score = 0.0
        evidence = []

        if not recent_performance:
            return 0.0

        # Strategy win rate declining
        recent_wr = recent_performance.get("recent_win_rate", 0.5)
        historical_wr = recent_performance.get("historical_win_rate", 0.5)

        if recent_wr < historical_wr * 0.7 and recent_performance.get("recent_trade_count", 0) >= 10:
            score += 0.3
            evidence.append(
                f"Strategy win rate declining: {recent_wr*100:.0f}% "
                f"(was {historical_wr*100:.0f}%)"
            )

        # Strategy expectancy turning negative
        recent_exp = recent_performance.get("recent_expectancy", 0)
        if recent_exp < 0 and recent_performance.get("recent_trade_count", 0) >= 15:
            score += 0.3
            evidence.append(
                f"Strategy expectancy negative: {recent_exp:.2f}R "
                f"over last {recent_performance.get('recent_trade_count')} trades"
            )

        # Losing streak for this strategy
        consecutive = recent_performance.get("current_streak", 0)
        if consecutive <= -5:
            score += 0.2
            evidence.append(f"Strategy on {abs(consecutive)}-trade losing streak")

        score = min(1.0, score)
        self._last_strategy_evidence = evidence
        return score

    def _assess_execution_contribution(
        self,
        trade: Trade,
        attribution: Optional[StrategyAttribution],
        overrides: list[OverrideEvent],
    ) -> float:
        score = 0.0

        # Non-emotional execution issues
        mechanical_overrides = [
            o for o in overrides
            if o.override_type in (
                OverrideType.EARLY_ENTRY,
                OverrideType.LATE_ENTRY,
                OverrideType.STOP_TOO_WIDE,
                OverrideType.STOP_TOO_TIGHT,
                OverrideType.UNDERSIZING,
                OverrideType.TARGET_SKIPPED,
                OverrideType.TIME_WINDOW_VIOLATION,
            )
        ]

        if mechanical_overrides:
            score += min(0.4, len(mechanical_overrides) * 0.1)

        return min(1.0, score)

    def _assess_risk_context(
        self,
        analysis: DeviationAnalysis,
        trade: Trade,
        entry_market: Optional[MarketSnapshot],
        exit_market: Optional[MarketSnapshot],
    ) -> None:
        """Determine if risk sizing was appropriate for conditions."""
        if not trade.risk_pct:
            return

        # Hard-coded strategy rules removed in favor of AI-learned profiles

        # Volatility-based adjustment
        if entry_market and entry_market.atr:
            atr = entry_market.atr
            stop_dist = abs(trade.entry_price - trade.stop_loss) if trade.stop_loss else 0
            if stop_dist > 0 and atr > 0:
                atr_ratio = stop_dist / atr
                if atr_ratio > 2.5:
                    analysis.risk_adaptation_needed = True
                    analysis.suggested_risk_adjustment = (
                        f"Stop at {atr_ratio:.1f}x ATR is wide — "
                        f"reduce size or tighten stop"
                    )

    def _generate_action_items(self, analysis: DeviationAnalysis) -> None:
        """Generate specific, actionable recommendations."""
        items = []

        if analysis.is_traders_fault:
            if analysis.emotional_contribution > 0.5:
                items.append(
                    "🧠 EMOTIONAL: Before next trade, pause 5 minutes and "
                    "write down the setup rationale. Only trade if it matches strategy rules."
                )
            if analysis.execution_contribution > 0.3:
                items.append(
                    "📋 EXECUTION: Use the strategy checklist before entry. "
                    "Don't enter until all required conditions show green."
                )

        if analysis.is_market_fault:
            if analysis.regime_changed:
                items.append(
                    "📊 REGIME: Market conditions changed during the trade. "
                    "Consider adding regime monitoring to your exit process."
                )
            else:
                items.append(
                    "📊 MARKET: Setup was valid but market didn't cooperate. "
                    "This is a normal loss — no behavior change needed."
                )

        if analysis.is_strategy_fault:
            items.append(
                "🔧 STRATEGY: This strategy may be losing edge in current conditions. "
                "Review last 20 trades for this strategy. Consider pausing if "
                "expectancy is negative over 15+ trades."
            )

        if analysis.is_normal_loss:
            items.append(
                "✅ NORMAL: This loss is within expected variance. "
                "No changes needed. Trust the process."
            )

        if analysis.risk_adaptation_needed:
            items.append(
                f"⚖️ RISK: {analysis.suggested_risk_adjustment}"
            )

        analysis.action_items = items

        # Populate evidence from last analysis
        if hasattr(self, '_last_emotional_evidence'):
            analysis.emotional_evidence = self._last_emotional_evidence
        if hasattr(self, '_last_market_evidence'):
            analysis.market_evidence = self._last_market_evidence
        if hasattr(self, '_last_strategy_evidence'):
            analysis.strategy_evidence = self._last_strategy_evidence