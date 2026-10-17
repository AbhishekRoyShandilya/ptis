# strategy_agent.py
# agents/strategy_agent.py
"""
Strategy Intelligence Agent.
Validates rule adherence against Strategy DNA.
"""

from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_

from agents.base_agent import BaseAgent, AgentInsight
from core.models import Trade, JournalEntry
from core.enums import TradeStatus
from strategy_engine.strategy_dna.query_engine import StrategyQueryEngine
import logging

logger = logging.getLogger(__name__)


class StrategyAgent(BaseAgent):
    """
    1️⃣ Strategy Intelligence Agent

    Validates every trade against its strategy DNA:
    - Did you follow entry rules?
    - Was execution disciplined?
    - Were conditions right for this strategy?
    - Are you making repeated mistakes?
    """

    def __init__(self, strategy_query: StrategyQueryEngine, db: Session):
        super().__init__("strategy_agent", strategy_query, db)

    def analyze(self, context: Dict) -> List[AgentInsight]:
        """
        Context should include:
        - trade_id: str (optional, for single trade analysis)
        - lookback_trades: int (for pattern analysis)
        - current_regime: str
        """
        insights = []

        trade_id = context.get("trade_id")
        if trade_id:
            insights.extend(self._analyze_single_trade(trade_id, context))

        lookback = context.get("lookback_trades", 20)
        insights.extend(self._analyze_patterns(lookback, context))

        return insights

    def _analyze_single_trade(self, trade_id: str, context: Dict) -> List[AgentInsight]:
        """Analyze a single trade against its strategy DNA or learned profile."""
        insights = []

        trade = self.db.query(Trade).filter(Trade.id == trade_id).first()
        if not trade or not trade.strategy_name:
            return insights

        # Check for learned profile first — provides richer analysis
        learned_profile = self._get_learned_profile(trade.strategy_name)
        if learned_profile and learned_profile.narrative:
            insights.extend(self._analyze_against_learned_profile(trade, learned_profile))

        dna = self.strategy_query.get_strategy(trade.strategy_name)
        if not dna:
            return insights

        # ── Check regime fitness ──
        if trade.market_regime:
            regime = trade.market_regime.value if hasattr(trade.market_regime, 'value') else str(trade.market_regime)
            should_trade, sizing, reason = self.strategy_query.should_trade_strategy_in_regime(
                trade.strategy_name, regime
            )

            if not should_trade:
                insights.append(AgentInsight(
                    agent_name=self.name,
                    insight_type="execution",
                    severity="warning",
                    title=f"Strategy-Regime Mismatch",
                    message=(
                        f"You traded {dna.display_name} in {regime} regime. "
                        f"{reason}"
                    ),
                    trade_ids=[trade_id],
                    strategy_name=trade.strategy_name,
                    action_suggestion=f"Consider {dna.get_fallback_for_regime(regime)} in this regime",
                ))

            elif sizing in ("half_size", "quarter_size"):
                insights.append(AgentInsight(
                    agent_name=self.name,
                    insight_type="execution",
                    severity="info",
                    title="Regime Suggests Reduced Size",
                    message=(
                        f"{dna.display_name} in {regime}: recommended sizing is {sizing}. "
                        f"{reason}"
                    ),
                    trade_ids=[trade_id],
                    strategy_name=trade.strategy_name,
                ))

        # ── Check time window ──
        if trade.entry_time and dna.checklist:
            # Check against known common mistakes
            for mistake in dna.common_entry_mistakes:
                # Simple keyword matching for common patterns
                insights.extend(
                    self._check_known_mistake(trade, dna, mistake)
                )

        # ── Check checklist score ──
        if trade.tags and "checklist_score" in trade.tags:
            cs = trade.tags["checklist_score"]
            if cs < 50:
                insights.append(AgentInsight(
                    agent_name=self.name,
                    insight_type="execution",
                    severity="warning",
                    title="Low Checklist Score",
                    message=(
                        f"Checklist score was {cs:.0f}% for this {dna.display_name} trade. "
                        f"Multiple conditions weren't met. "
                        f"Your data shows low-checklist trades underperform by "
                        f"a significant margin."
                    ),
                    trade_ids=[trade_id],
                    strategy_name=trade.strategy_name,
                    action_suggestion="Only take trades where checklist score >= 70%",
                ))

        return insights

    def _check_known_mistake(
        self, trade: Trade, dna, mistake: str
    ) -> List[AgentInsight]:
        """Check if a known mistake pattern is present."""
        # This is pattern-matched heuristically
        # Full implementation would be more sophisticated
        insights = []

        mistake_lower = mistake.lower()

        # Check for "entering before candle closes" pattern
        if "before" in mistake_lower and "close" in mistake_lower:
            # Would check entry timing vs candle close timing
            # Requires market data — flag for later implementation
            pass

        # Check for "too many" pattern
        if "too many" in mistake_lower:
            # Count trades for this strategy today
            from datetime import datetime, timedelta
            today_start = datetime.combine(trade.entry_time.date(), datetime.min.time())
            count = self.db.query(Trade).filter(
                and_(
                    Trade.strategy_name == trade.strategy_name,
                    Trade.entry_time >= today_start,
                    Trade.entry_time < today_start + timedelta(days=1),
                )
            ).count()

            if count >= 3:
                insights.append(AgentInsight(
                    agent_name=self.name,
                    insight_type="execution",
                    severity="warning",
                    title="Over-Trading This Strategy",
                    message=(
                        f"This is {dna.display_name} trade #{count} today. "
                        f"Your known weakness: \"{mistake}\""
                    ),
                    trade_ids=[trade.id],
                    strategy_name=trade.strategy_name,
                    action_suggestion="Stick to max 2 trades per strategy per day",
                ))

        return insights

    def _analyze_patterns(self, lookback: int, context: Dict) -> List[AgentInsight]:
        """Analyze patterns across recent trades."""
        insights = []

        recent_trades = self.db.query(Trade).filter(
            Trade.status == TradeStatus.CLOSED,
        ).order_by(Trade.entry_time.desc()).limit(lookback).all()

        if not recent_trades:
            return insights

        # ── Strategy adherence rate ──
        strategy_counts = {}
        for trade in recent_trades:
            name = trade.strategy_name or "untagged"
            if name not in strategy_counts:
                strategy_counts[name] = {"total": 0, "overrides": 0, "wins": 0}
            strategy_counts[name]["total"] += 1
            if trade.tags and trade.tags.get("overrides"):
                strategy_counts[name]["overrides"] += 1
            if trade.pnl and trade.pnl > 0:
                strategy_counts[name]["wins"] += 1

        for name, data in strategy_counts.items():
            if data["total"] >= 5:
                override_pct = data["overrides"] / data["total"] * 100
                if override_pct > 40:
                    dna = self.strategy_query.get_strategy(name)
                    display = dna.display_name if dna else name
                    insights.append(AgentInsight(
                        agent_name=self.name,
                        insight_type="behavioral",
                        severity="warning",
                        title=f"High Override Rate: {display}",
                        message=(
                            f"{override_pct:.0f}% of your {display} trades "
                            f"had deviations from rules in the last {data['total']} trades. "
                            f"Either update your rules to match your actual trading, "
                            f"or recommit to discipline."
                        ),
                        strategy_name=name,
                        action_suggestion=(
                            "Review whether your rules need updating or "
                            "your discipline needs strengthening"
                        ),
                    ))

        # ── Wrong regime detection ──
        current_regime = context.get("current_regime")
        if current_regime:
            poor_fitness_trades = 0
            for trade in recent_trades[:10]:
                if trade.strategy_name:
                    should, _, _ = self.strategy_query.should_trade_strategy_in_regime(
                        trade.strategy_name, current_regime
                    )
                    if not should:
                        poor_fitness_trades += 1

            if poor_fitness_trades >= 3:
                recs = self.strategy_query.get_strategies_for_regime(current_regime)
                best = recs[0] if recs else None
                insights.append(AgentInsight(
                    agent_name=self.name,
                    insight_type="performance",
                    severity="critical",
                    title="Trading Wrong Strategies for Current Regime",
                    message=(
                        f"{poor_fitness_trades} of your last 10 trades used strategies "
                        f"that perform poorly in {current_regime}. "
                        f"Best strategy for this regime: "
                        f"{best.display_name if best else 'unknown'}"
                    ),
                    action_suggestion=(
                        f"Switch to {best.display_name}" if best else
                        "Review regime-strategy alignment"
                    ),
                ))

        return insights

    # ── Learned Profile Integration ──

    def _get_learned_profile(self, strategy_name: str) -> Optional["StrategyProfile"]:
        """Check if a learned StrategyProfile exists for this strategy."""
        try:
            from core.models import StrategyProfile
            profile = self.db.query(StrategyProfile).filter(
                StrategyProfile.short_code == strategy_name,
                StrategyProfile.status == "active",
            ).first()
            return profile
        except Exception:
            return None

    def _analyze_against_learned_profile(
        self, trade: Trade, profile: "StrategyProfile"
    ) -> List[AgentInsight]:
        """Generate insights by comparing trade against learned profile."""
        insights = []

        # Check regime against learned preferences
        if trade.market_regime and profile.regime_preferences:
            regime_str = trade.market_regime.value if hasattr(trade.market_regime, 'value') else str(trade.market_regime)
            avoid_regimes = [
                r.get("regime", "") if isinstance(r, dict) else str(r)
                for r in profile.regime_preferences.get("avoid", [])
            ]
            if regime_str.lower() in [r.lower() for r in avoid_regimes]:
                insights.append(AgentInsight(
                    agent_name=self.name,
                    insight_type="execution",
                    severity="warning",
                    title="Learned Profile: Regime Mismatch",
                    message=(
                        f"Your learned {profile.name} profile shows this strategy "
                        f"performs poorly in {regime_str} regime based on your "
                        f"training examples."
                    ),
                    trade_ids=[str(trade.id)],
                    strategy_name=trade.strategy_name,
                    action_suggestion="Consider skipping or reducing size in this regime",
                ))

        # Check confidence level
        confidence = profile.confidence_scores or {}
        overall_conf = confidence.get("overall", 0)
        if overall_conf < 0.5:
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="coaching",
                severity="info",
                title="Low Profile Confidence",
                message=(
                    f"Your {profile.name} profile confidence is {overall_conf:.0%}. "
                    f"The system needs more training examples to provide "
                    f"reliable analysis. Weak areas: "
                    + ", ".join(
                        k for k, v in confidence.items()
                        if isinstance(v, (int, float)) and v < 0.5 and k != "overall"
                    )
                ),
                trade_ids=[str(trade.id)],
                strategy_name=trade.strategy_name,
                action_suggestion="Upload more chart examples via the Strategy Learning page",
            ))