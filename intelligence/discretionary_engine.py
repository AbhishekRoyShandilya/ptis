"""
Discretionary Query Engine

Handles natural language queries about specific trade ideas.
When a trader asks "What do you think about this NVDA breakout?",
the engine:

1. Parses the query intent
2. Fetches real-time market environment for the symbol
3. Matches against user's strategy templates
4. Applies the trader's profile (risk style, biases, timing)
5. Estimates R:R with confidence
6. Generates a personalized, evidence-based assessment
7. Includes behavioral guardrails
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import pandas as pd

from intelligence.market_environment import (
    MarketEnvironment,
    MarketEnvironmentAnalyzer,
    OHLCVBar,
)
from intelligence.readiness_scanner import SetupScanner, SetupReadiness
from intelligence.r_estimator import RMultipleEstimator, REstimate
from intelligence.trader_profile_intelligence import (
    TraderIntelligenceProfile,
    DisciplinePattern,
)
from intelligence.breakout_detector import BreakoutDetector, BreakoutSignal

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """What the trader is asking about."""
    SETUP_EVALUATION = "setup_evaluation"   # "Is this setup valid?"
    ENTRY_TIMING = "entry_timing"           # "Should I enter now?"
    RISK_ASSESSMENT = "risk_assessment"     # "What's the risk here?"
    DIRECTION_BIAS = "direction_bias"       # "Is this bullish or bearish?"
    STRATEGY_MATCH = "strategy_match"       # "Which strategy fits?"
    EXIT_PLANNING = "exit_planning"         # "Where should I take profit?"
    GENERAL_ANALYSIS = "general_analysis"   # Broad market view


@dataclass
class BehavioralGuardrail:
    """A behavioral warning to include in the response."""
    severity: str  # "critical", "warning", "info"
    message: str
    recommendation: str


@dataclass
class DiscretionaryAssessment:
    """
    Complete assessment for a discretionary trade query.

    This is what gets sent to the coaching agent for
    final response generation.
    """
    symbol: str
    query: str
    intent: QueryIntent
    timestamp: datetime = field(default_factory=lambda: __import__('datetime').datetime.now(
        __import__('datetime').timezone(__import__('datetime').timedelta(hours=5, minutes=30))
    ))

    # Environment
    environment: Optional[MarketEnvironment] = None

    # Setup analysis
    matching_strategies: list[SetupReadiness] = field(default_factory=list)
    best_match: Optional[SetupReadiness] = None
    setup_valid: bool = False
    setup_confidence: float = 0.0

    # R:R estimation
    r_estimate: Optional[REstimate] = None
    risk_reward_assessment: str = ""  # "favorable", "neutral", "unfavorable"

    # Environment suitability
    environment_suitable: bool = True
    environment_score: float = 0.0
    regime_aligned: bool = False
    volatility_suitable: bool = True

    # Trader-specific
    matches_trading_style: bool = True
    timing_appropriate: bool = True
    within_risk_limits: bool = True
    within_daily_trade_limit: bool = True

    # Breakout / breakdown detection
    breakout_signal: Optional[BreakoutSignal] = None

    # Regime mismatch context (why strategy fails + what to use instead)
    regime_mismatch: Optional[dict[str, Any]] = None

    # Behavioral guardrails
    guardrails: list[BehavioralGuardrail] = field(default_factory=list)

    # Composite recommendation
    recommendation: str = ""  # "strong_consider", "consider", "caution", "avoid"
    recommendation_reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0

    # For the coaching agent to use
    context_for_coaching: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "query": self.query,
            "intent": self.intent.value,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M IST"),
            "environment": self.environment.to_dict() if self.environment else None,
            "best_match": self.best_match.to_dict() if self.best_match else None,
            "matching_strategy_count": len(self.matching_strategies),
            "setup_valid": self.setup_valid,
            "setup_confidence": round(self.setup_confidence, 3),
            "r_estimate": self.r_estimate.to_dict() if self.r_estimate else None,
            "risk_reward_assessment": self.risk_reward_assessment,
            "environment_suitable": self.environment_suitable,
            "environment_score": round(self.environment_score, 3),
            "regime_aligned": self.regime_aligned,
            "matches_trading_style": self.matches_trading_style,
            "timing_appropriate": self.timing_appropriate,
            "within_risk_limits": self.within_risk_limits,
            "guardrails": [
                {"severity": g.severity, "message": g.message,
                 "recommendation": g.recommendation}
                for g in self.guardrails
            ],
            "recommendation": self.recommendation,
            "recommendation_reasons": self.recommendation_reasons,
            "confidence": round(self.confidence, 3),
            "breakout_signal": self.breakout_signal.to_dict() if self.breakout_signal else None,
            "regime_mismatch": self.regime_mismatch,
        }


class DiscretionaryQueryEngine:
    """
    Processes discretionary trade queries with full context.

    This is NOT a trade recommendation engine. It provides structured
    analysis that helps the trader make their own informed decision.
    """

    def __init__(
        self,
        env_analyzer: MarketEnvironmentAnalyzer,
        setup_scanner: SetupScanner,
        r_estimator: Optional[RMultipleEstimator] = None,
    ):
        self.env_analyzer = env_analyzer
        self.scanner = setup_scanner
        self.r_estimator = r_estimator
        self._breakout_detector = BreakoutDetector()

    async def evaluate(
        self,
        query: str,
        symbol: str,
        df: pd.DataFrame,
        trader_profile: TraderIntelligenceProfile,
        direction: Optional[int] = None,
        todays_trade_count: int = 0,
        current_pnl: float = 0.0,
        timeframe: str = "5m",
    ) -> DiscretionaryAssessment:
        """
        Evaluate a discretionary trade query.

        Args:
            query: The trader's question
            symbol: Ticker symbol
            df: OHLCV data for the symbol
            trader_profile: The trader's intelligence profile
            direction: Intended direction (1=long, -1=short, None=unknown)
            todays_trade_count: How many trades already taken today
            current_pnl: Today's running P&L
            timeframe: Candle timeframe

        Returns:
            DiscretionaryAssessment with full analysis
        """
        assessment = DiscretionaryAssessment(
            symbol=symbol,
            query=query,
            intent=self._parse_intent(query),
        )

        # 1. Analyze market environment
        try:
            environment = self.env_analyzer.analyze(df, symbol, timeframe)
            assessment.environment = environment
            assessment.environment_score = environment.environment_score
        except Exception as e:
            logger.error(f"Environment analysis failed for {symbol}: {e}")
            assessment.guardrails.append(BehavioralGuardrail(
                severity="warning",
                message="Could not fully analyze market conditions.",
                recommendation="Proceed with extra caution if entering.",
            ))
            assessment.confidence *= 0.5
            return self._finalize(assessment, trader_profile, todays_trade_count, current_pnl)

        # 1b. Run breakout/breakdown detection
        try:
            # Convert df back to OHLCVBar sequence for the detector
            _bars = []
            for _, row in df.iterrows():
                try:
                    from datetime import datetime as _dt
                    _bars.append(OHLCVBar(
                        timestamp=_dt.utcnow(),
                        open=float(row.get("open", row.get("Open", 0))),
                        high=float(row.get("high", row.get("High", 0))),
                        low=float(row.get("low", row.get("Low", 0))),
                        close=float(row.get("close", row.get("Close", 0))),
                        volume=float(row.get("volume", row.get("Volume", 0))),
                    ))
                except Exception:
                    continue
            if _bars:
                _regime = environment.regime.value if environment else "unknown"
                _atr = environment.atr if environment else None
                _vwap = environment.vwap if environment else None
                assessment.breakout_signal = self._breakout_detector.detect(
                    bars=_bars, regime=_regime, atr=_atr, timeframe=timeframe,
                    vwap=_vwap,
                )
        except Exception as e:
            logger.warning("Breakout detection failed: %s", e)

        # 2. Scan for matching setups
        scan_result = self.scanner.scan(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            trader_id=trader_profile.trader_id,
        )
        assessment.matching_strategies = [
            s for s in scan_result.setups if s.readiness_score > 0.40
        ]
        assessment.best_match = scan_result.top_setup

        # 3. Evaluate setup validity
        if assessment.best_match:
            assessment.setup_valid = assessment.best_match.readiness_score >= 0.65
            assessment.setup_confidence = assessment.best_match.readiness_score
            assessment.regime_aligned = assessment.best_match.regime_aligned
            assessment.volatility_suitable = assessment.best_match.volatility_suitable
        else:
            assessment.setup_valid = False
            assessment.setup_confidence = 0.0

        # 4. Estimate R:R — wrapped in savepoint so a failed DB query doesn't poison
        #    the outer transaction (asyncpg marks the whole tx as aborted on any error)
        if self.r_estimator and assessment.best_match:
            try:
                async with self.db.begin_nested():
                    r_est = await self.r_estimator.estimate(
                        strategy_name=assessment.best_match.strategy_name,
                        environment=environment,
                        trader_id=trader_profile.trader_id,
                        direction=direction,
                    )
                assessment.r_estimate = r_est
                assessment.risk_reward_assessment = self._assess_rr(r_est)
            except Exception as e:
                logger.debug(f"R estimation skipped: {e}")

        # 5. Apply trader profile checks
        assessment = self._apply_profile_checks(
            assessment, trader_profile, environment,
            todays_trade_count, current_pnl
        )

        # 6. Generate behavioral guardrails
        guardrails = self._generate_guardrails(
            assessment, trader_profile,
            todays_trade_count, current_pnl
        )
        assessment.guardrails.extend(guardrails)

        # 7. Finalize recommendation
        return self._finalize(
            assessment, trader_profile,
            todays_trade_count, current_pnl
        )

    def _parse_intent(self, query: str) -> QueryIntent:
        """Parse the trader's query intent."""
        q = query.lower()

        if any(kw in q for kw in ["enter", "get in", "should i buy", "should i short", "take this"]):
            return QueryIntent.ENTRY_TIMING
        if any(kw in q for kw in ["risk", "stop", "how much to risk", "position size"]):
            return QueryIntent.RISK_ASSESSMENT
        if any(kw in q for kw in ["target", "exit", "take profit", "where to sell"]):
            return QueryIntent.EXIT_PLANNING
        if any(kw in q for kw in ["bullish", "bearish", "direction", "which way"]):
            return QueryIntent.DIRECTION_BIAS
        if any(kw in q for kw in ["which strategy", "what setup", "what pattern"]):
            return QueryIntent.STRATEGY_MATCH
        if any(kw in q for kw in ["breakout", "setup", "valid", "good trade", "what do you think"]):
            return QueryIntent.SETUP_EVALUATION

        return QueryIntent.GENERAL_ANALYSIS

    def _assess_rr(self, r_est: REstimate) -> str:
        """Assess risk/reward favorability."""
        if r_est.expected_rr_ratio >= 3.0 and r_est.win_probability >= 0.50:
            return "favorable"
        elif r_est.expected_rr_ratio >= 2.0 and r_est.win_probability >= 0.45:
            return "favorable"
        elif r_est.expected_rr_ratio >= 1.5:
            return "neutral"
        elif r_est.expected_rr_ratio >= 1.0:
            return "marginal"
        else:
            return "unfavorable"

    def _apply_profile_checks(
        self,
        assessment: DiscretionaryAssessment,
        profile: TraderIntelligenceProfile,
        environment: MarketEnvironment,
        todays_trade_count: int,
        current_pnl: float,
    ) -> DiscretionaryAssessment:
        """Apply trader-profile-specific checks."""

        # Timing check
        current_hour = datetime.utcnow().hour
        if current_hour in profile.timing.worst_hours:
            assessment.timing_appropriate = False

        # Trade count check
        if profile.timing.quality_decay_after_n_trades:
            if todays_trade_count >= profile.timing.quality_decay_after_n_trades:
                assessment.within_daily_trade_limit = False

        # Trading style alignment
        if profile.entry_style == "confirmation" and assessment.best_match:
            if assessment.best_match.conditions_met_ratio < 0.80:
                assessment.matches_trading_style = False

        # Risk limit check
        if assessment.r_estimate:
            # Check if the implied risk fits trader's profile
            if (assessment.r_estimate.expected_rr_ratio < 1.5 and
                    profile.risk.personality == "conservative"):
                assessment.within_risk_limits = False

        # Environment suitability
        env_context = f"{environment.regime.value}"
        assessment.environment_suitable = (
            env_context not in profile.worst_conditions
        )

        return assessment

    def _generate_guardrails(
        self,
        assessment: DiscretionaryAssessment,
        profile: TraderIntelligenceProfile,
        todays_trade_count: int,
        current_pnl: float,
    ) -> list[BehavioralGuardrail]:
        """Generate behavioral guardrails based on current state."""
        guardrails = []

        # Critical: Discipline is breaking down
        if profile.current_discipline_score < 0.50:
            guardrails.append(BehavioralGuardrail(
                severity="critical",
                message=(
                    "Your discipline score is critically low today "
                    f"({profile.current_discipline_score:.0%}). "
                    "Trading in this state significantly increases risk."
                ),
                recommendation=(
                    "Consider taking a break and resetting before entering "
                    "any new positions."
                ),
            ))

        # Critical: Overtrading
        if not assessment.within_daily_trade_limit:
            guardrails.append(BehavioralGuardrail(
                severity="critical",
                message=(
                    f"You've already taken {todays_trade_count} trades today. "
                    f"Your quality typically drops after "
                    f"{profile.timing.quality_decay_after_n_trades} trades."
                ),
                recommendation=(
                    "Your data shows decision quality declines at this point. "
                    "If you enter, use a smaller position size."
                ),
            ))

        # Warning: Trading during worst hours
        if not assessment.timing_appropriate:
            guardrails.append(BehavioralGuardrail(
                severity="warning",
                message=(
                    "You're trading during one of your historically "
                    "worst-performing hours."
                ),
                recommendation=(
                    "Your data shows lower R-multiples during this time. "
                    "Consider waiting for a better window."
                ),
            ))

        # Warning: Post-loss revenge pattern
        if (profile.emotional.post_loss_behavior == "revenge_trades" and
                current_pnl < 0):
            guardrails.append(BehavioralGuardrail(
                severity="warning",
                message=(
                    "You're down on the day and your pattern shows a tendency "
                    "toward revenge trading after losses."
                ),
                recommendation=(
                    "Ask yourself: Am I entering this because of the setup, "
                    "or because I want to make back today's loss?"
                ),
            ))

        # Warning: Setup doesn't match style
        if not assessment.matches_trading_style:
            guardrails.append(BehavioralGuardrail(
                severity="warning",
                message=(
                    f"This setup may not match your '{profile.entry_style}' "
                    f"entry style — not all conditions are confirmed yet."
                ),
                recommendation=(
                    "Based on your profile, you perform better when waiting "
                    "for full confirmation. Consider waiting."
                ),
            ))

        # Info: Environment in trader's worst conditions
        if not assessment.environment_suitable:
            guardrails.append(BehavioralGuardrail(
                severity="info",
                message=(
                    "Current conditions match patterns where you've historically "
                    "underperformed."
                ),
                recommendation=(
                    "Extra vigilance on risk management. "
                    "Consider reducing position size by 50%."
                ),
            ))

        # Info: Regime not aligned
        if not assessment.regime_aligned and assessment.best_match:
            guardrails.append(BehavioralGuardrail(
                severity="info",
                message=(
                    f"The current regime ({assessment.environment.regime.value if assessment.environment else 'unknown'}) "
                    f"is not the preferred regime for "
                    f"'{assessment.best_match.strategy_name}'."
                ),
                recommendation=(
                    "This strategy has lower historical win rates in this regime. "
                    "Tighten your stop or reduce size."
                ),
            ))

        return guardrails

    def _finalize(
        self,
        assessment: DiscretionaryAssessment,
        profile: TraderIntelligenceProfile,
        todays_trade_count: int,
        current_pnl: float,
    ) -> DiscretionaryAssessment:
        """Compute final recommendation and confidence."""
        reasons = []
        score = 0.5  # Neutral starting point

        # Setup quality
        if assessment.setup_valid:
            score += 0.15
            reasons.append(
                f"Setup matches '{assessment.best_match.strategy_name}' "
                f"({assessment.setup_confidence:.0%} readiness)"
            )
        elif assessment.best_match and assessment.best_match.is_forming:
            score += 0.05
            reasons.append(
                f"Setup forming for '{assessment.best_match.strategy_name}' "
                f"but not yet triggered ({assessment.best_match.conditions_met}/"
                f"{assessment.best_match.conditions_total} conditions)"
            )
        else:
            score -= 0.15
            reasons.append("No matching strategy template found for this setup")

        # Breakout signal contribution
        bs = assessment.breakout_signal
        if bs and bs.detected:
            if bs.signal_type == "breakout":
                if bs.quality_score >= 0.6:
                    score += 0.12
                    reasons.append(
                        f"Active breakout detected — quality {bs.quality_score:.0%}, "
                        f"~{int(bs.regime_success_rate*100)}% historical success in current regime"
                    )
                    if bs.room_to_next_level_pct >= 1.5:
                        reasons.append(
                            f"Room to next resistance: {bs.room_to_next_level_pct:.1f}% "
                            f"(₹{bs.next_resistance:.1f})" if bs.next_resistance else
                            f"Room to next resistance: {bs.room_to_next_level_pct:.1f}%"
                        )
                    elif bs.room_to_next_level_pct < 0.5 and bs.next_resistance:
                        score -= 0.05
                        reasons.append(
                            f"⚠ Next resistance only {bs.room_to_next_level_pct:.1f}% away "
                            f"(₹{bs.next_resistance:.1f}) — limited upside"
                        )
                else:
                    reasons.append(
                        f"Weak breakout detected ({bs.quality_score:.0%} quality) — "
                        "watch for fakeout"
                    )
            elif bs.signal_type == "breakdown":
                if bs.quality_score >= 0.6:
                    score -= 0.10   # breakdowns are bearish context
                    reasons.append(
                        f"Active breakdown — quality {bs.quality_score:.0%}, "
                        f"~{int(bs.regime_success_rate*100)}% follow-through chance"
                    )
        elif bs and not bs.detected and bs.next_resistance:
            gap = bs.room_to_next_level_pct
            if gap < 0.8:
                score -= 0.05
                reasons.append(
                    f"Price near resistance ₹{bs.next_resistance:.1f} "
                    f"({gap:.1f}% away) — potential hurdle"
                )

        # R:R assessment
        if assessment.r_estimate:
            rr = assessment.risk_reward_assessment
            if rr == "favorable":
                score += 0.15
                reasons.append(
                    f"Favorable R:R estimate "
                    f"({assessment.r_estimate.expected_rr_ratio:.1f}:1, "
                    f"win prob: {assessment.r_estimate.win_probability:.0%})"
                )
            elif rr == "neutral":
                score += 0.05
                reasons.append(
                    f"Neutral R:R ({assessment.r_estimate.expected_rr_ratio:.1f}:1)"
                )
            elif rr in {"marginal", "unfavorable"}:
                score -= 0.10
                reasons.append(
                    f"Unfavorable R:R estimate "
                    f"({assessment.r_estimate.expected_rr_ratio:.1f}:1)"
                )

        # Environment
        if assessment.environment_suitable and assessment.regime_aligned:
            score += 0.10
            reasons.append("Environment and regime aligned with strategy")
        elif not assessment.regime_aligned:
            score -= 0.05
            # Build regime mismatch context: WHY it fails + what to use instead
            if assessment.best_match and assessment.environment:
                current_regime = assessment.environment.regime.value
                try:
                    dna = self.scanner.strategies.get(assessment.best_match.strategy_name)
                    if dna:
                        guidance = dna.get_regime_guidance(current_regime)
                        alternative_raw = dna.get_regime_alternative(current_regime)
                        # Parse alternatives: may be "vwap_bounce or range_fade"
                        alt_names = [
                            a.strip().replace("_", " ").title()
                            for a in (alternative_raw or "").replace(" or ", ",").split(",")
                            if a.strip()
                        ]
                        assessment.regime_mismatch = {
                            "strategy": assessment.best_match.strategy_name,
                            "strategy_display": assessment.best_match.strategy_name.replace("_", " ").title(),
                            "current_regime": current_regime.replace("_", " ").title(),
                            "why_fails": guidance.notes if guidance else f"{assessment.best_match.strategy_name} does not perform well in {current_regime} conditions",
                            "recommendation": guidance.recommendation if guidance else "Consider waiting for a better regime",
                            "alternatives": alt_names,
                        }
                        reasons.append(
                            f"❌ {assessment.best_match.strategy_name.upper()} not suitable in {current_regime.replace('_',' ')} — "
                            f"{guidance.notes if guidance else 'poor regime fit'}"
                        )
                    else:
                        reasons.append("Current regime does not favor this strategy")
                except Exception:
                    reasons.append("Current regime does not favor this strategy")
            else:
                reasons.append("Current regime does not favor this strategy")
        if not assessment.environment_suitable:
            score -= 0.05
            reasons.append("Environment matches your historically weak conditions")

        # Trader state checks
        if not assessment.timing_appropriate:
            score -= 0.10
            reasons.append("Trading during your historically worst hours")

        if not assessment.within_daily_trade_limit:
            score -= 0.15
            reasons.append(
                f"Already {todays_trade_count} trades today — "
                f"quality decays after {profile.timing.quality_decay_after_n_trades}"
            )

        if not assessment.matches_trading_style:
            score -= 0.05
            reasons.append("Setup doesn't match your confirmation style")

        if not assessment.within_risk_limits:
            score -= 0.05
            reasons.append("R:R doesn't meet your risk profile standards")

        # Behavioral guardrails impact
        critical_count = sum(
            1 for g in assessment.guardrails if g.severity == "critical"
        )
        warning_count = sum(
            1 for g in assessment.guardrails if g.severity == "warning"
        )
        score -= critical_count * 0.15
        score -= warning_count * 0.05

        if critical_count > 0:
            reasons.append(
                f"{critical_count} critical behavioral guardrail(s) active"
            )

        # Clamp score
        score = max(0.0, min(1.0, score))

        # Map score to recommendation
        if score >= 0.75:
            recommendation = "strong_consider"
        elif score >= 0.60:
            recommendation = "consider"
        elif score >= 0.45:
            recommendation = "caution"
        else:
            recommendation = "avoid"

        # Override to "avoid" if any critical guardrails
        if critical_count >= 2:
            recommendation = "avoid"
            reasons.append("Multiple critical guardrails override recommendation")

        # Confidence is based on data quality
        confidence = self._compute_confidence(assessment, profile)

        assessment.recommendation = recommendation
        assessment.recommendation_reasons = reasons
        assessment.confidence = confidence

        # Build coaching context for the LLM
        assessment.context_for_coaching = self._build_coaching_context(
            assessment, profile, todays_trade_count, current_pnl
        )

        logger.info(
            f"Discretionary assessment for {assessment.symbol}: "
            f"recommendation={recommendation} (score={score:.2f}), "
            f"confidence={confidence:.2f}, "
            f"guardrails={critical_count}c/{warning_count}w"
        )

        return assessment

    def _compute_confidence(
        self,
        assessment: DiscretionaryAssessment,
        profile: TraderIntelligenceProfile,
    ) -> float:
        """Compute confidence in the assessment."""
        confidence = 0.5

        # More data = more confidence
        if profile.sample_size >= 100:
            confidence += 0.15
        elif profile.sample_size >= 50:
            confidence += 0.10
        elif profile.sample_size >= 20:
            confidence += 0.05

        # R estimate confidence
        if assessment.r_estimate:
            confidence += assessment.r_estimate.confidence * 0.20

        # Environment analysis available; penalise if it failed
        if assessment.environment:
            confidence += 0.10
        else:
            confidence -= 0.20  # significant penalty — key context missing

        # Setup match confidence
        if assessment.best_match:
            confidence += assessment.setup_confidence * 0.15

        return min(0.95, max(0.10, confidence))

    def _build_coaching_context(
        self,
        assessment: DiscretionaryAssessment,
        profile: TraderIntelligenceProfile,
        todays_trade_count: int,
        current_pnl: float,
    ) -> dict[str, Any]:
        """
        Build context dict that gets injected into the coaching agent's prompt.

        This is the bridge between the quantitative assessment and the
        natural language coaching response.
        """
        context = {
            "query_type": assessment.intent.value,
            "symbol": assessment.symbol,
            "recommendation": assessment.recommendation,
            "recommendation_reasons": assessment.recommendation_reasons,
            "confidence": assessment.confidence,
        }

        # Environment summary
        if assessment.environment:
            env = assessment.environment
            context["environment"] = {
                "regime": env.regime.value,
                "trend_direction": "bullish" if env.trend.direction > 0 else (
                    "bearish" if env.trend.direction < 0 else "neutral"
                ),
                "trend_strength": env.trend.strength.value,
                "volatility": env.volatility.state.value,
                "environment_score": env.environment_score,
                "price": env.current_price,
                "vwap": env.vwap,
                "nearest_support": env.structure.nearest_support,
                "nearest_resistance": env.structure.nearest_resistance,
                "structure_break": env.structure.has_structure_break,
                "liquidity_sweep": env.structure.has_liquidity_sweep,
            }

        # Setup summary
        if assessment.best_match:
            match = assessment.best_match
            context["setup"] = {
                "strategy": match.strategy_name,
                "readiness": match.readiness_score,
                "conditions_met": match.conditions_met,
                "conditions_total": match.conditions_total,
                "is_triggered": match.is_triggered,
                "is_forming": match.is_forming,
                "missing_conditions": [
                    c.name for c in match.conditions if not c.is_met
                ],
                "close_conditions": [
                    {"name": c.name, "proximity": c.proximity}
                    for c in match.conditions
                    if not c.is_met and c.proximity > 0.60
                ],
            }

        # R:R summary
        if assessment.r_estimate:
            r = assessment.r_estimate
            context["r_estimate"] = {
                "expected_r": r.expected_r,
                "expected_rr": r.expected_rr_ratio,
                "win_probability": r.win_probability,
                "confidence": r.confidence,
                "upside_r": r.upside_r,
                "downside_r": r.downside_r,
                "assessment": assessment.risk_reward_assessment,
            }

        # Trader state
        context["trader_state"] = {
            "trading_style": profile.trading_style,
            "entry_style": profile.entry_style,
            "risk_personality": profile.risk.personality,
            "discipline_score": profile.current_discipline_score,
            "discipline_trend": profile.discipline_trend,
            "todays_trade_count": todays_trade_count,
            "todays_pnl": current_pnl,
            "timing_appropriate": assessment.timing_appropriate,
            "within_trade_limit": assessment.within_daily_trade_limit,
            "post_loss_tendency": profile.emotional.post_loss_behavior,
            "current_biases": profile.emotional.primary_triggers[:3],
        }

        # Guardrails
        context["guardrails"] = [
            {
                "severity": g.severity,
                "message": g.message,
                "recommendation": g.recommendation,
            }
            for g in assessment.guardrails
        ]

        # Historical context for this strategy+regime
        if assessment.best_match and profile.strategy_preferences:
            for sp in profile.strategy_preferences:
                if sp.strategy_name == assessment.best_match.strategy_name:
                    context["strategy_history"] = {
                        "win_rate": sp.win_rate,
                        "avg_r": sp.avg_r,
                        "frequency": sp.frequency,
                        "best_regime": sp.best_regime,
                        "worst_regime": sp.worst_regime,
                        "is_core_strategy": sp.is_core,
                    }
                    break

        return context