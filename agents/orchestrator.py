# orchestrator.py
# agents/orchestrator.py
"""
Agent Orchestrator — coordinates all agents into a unified pipeline.
This is the central nervous system of the intelligence layer.
"""

from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from agents.base_agent import AgentInsight
from agents.strategy_agent import StrategyAgent
from agents.behavior_agent import BehaviorAgent
from agents.coaching_agent import CoachingAgent
from agents.llm_client import LLMClient
from strategy_engine.strategy_dna.query_engine import StrategyQueryEngine
from strategy_engine.strategy_dna.loader import StrategyDNALoader
from core.models import Trade
import logging

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Coordinates all AI agents to produce unified intelligence output.

    Usage:
        orchestrator = AgentOrchestrator(db, dna_loader)

        # On trade entry
        insights = orchestrator.on_trade_entered(trade_id, regime)

        # On trade closed
        review = orchestrator.on_trade_closed(trade_id, regime)

        # Chat
        response = orchestrator.chat("What mistakes am I repeating?")

        # Daily summary
        summary = orchestrator.daily_intelligence_summary(regime)
    """

    def __init__(
        self,
        db: Session,
        dna_loader: StrategyDNALoader,
        llm_client: Optional[LLMClient] = None,
    ):
        self.db = db
        self.dna_loader = dna_loader
        self.query_engine = StrategyQueryEngine(dna_loader)
        self.llm_client = llm_client

        # Initialize agents
        self.strategy_agent = StrategyAgent(self.query_engine, db)
        self.behavior_agent = BehaviorAgent(self.query_engine, db)
        self.coaching_agent = CoachingAgent(llm_client=llm_client)

        # Agent registry for extensibility
        self._agents = {
            "strategy": self.strategy_agent,
            "behavior": self.behavior_agent,
            "coaching": self.coaching_agent,
        }

        logger.info(f"Agent Orchestrator initialized with {len(self._agents)} agents")

    def register_agent(self, name: str, agent):
        """Register additional agents."""
        self._agents[name] = agent
        logger.info(f"Registered agent: {name}")

    # ──────────────────────────────────────
    # TRADE LIFECYCLE HOOKS
    # ──────────────────────────────────────

    def on_trade_entered(
        self,
        trade_id: str,
        current_regime: Optional[str] = None,
    ) -> Dict:
        """
        Called when a new trade is entered.
        Runs all agents and returns unified intelligence.
        """
        context = {
            "trade_id": trade_id,
            "current_regime": current_regime,
            "event": "trade_entered",
        }

        # Collect insights from all agents
        all_insights = []
        for agent_name, agent in self._agents.items():
            try:
                insights = agent.analyze(context)
                all_insights.extend(insights)
            except Exception as e:
                logger.error(f"Agent {agent_name} error on trade entry: {e}", exc_info=True)

        # Sort by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
        all_insights.sort(key=lambda i: severity_order.get(i.severity, 99))

        # Get pre-trade briefing
        trade = self.db.query(Trade).filter(Trade.id == trade_id).first()
        briefing = None
        if trade and trade.strategy_name and current_regime:
            emotion = None
            if trade.journal_entry:
                emotion = trade.journal_entry.pre_trade_emotion.value if trade.journal_entry.pre_trade_emotion else None

            briefing = self.coaching_agent.get_pre_trade_briefing(
                strategy_name=trade.strategy_name,
                current_regime=current_regime,
                trader_emotion=emotion,
            )

        return {
            "trade_id": trade_id,
            "insights": [self._insight_to_dict(i) for i in all_insights],
            "critical_alerts": [
                self._insight_to_dict(i) for i in all_insights
                if i.severity == "critical"
            ],
            "warnings": [
                self._insight_to_dict(i) for i in all_insights
                if i.severity == "warning"
            ],
            "briefing": briefing,
            "total_insights": len(all_insights),
        }

    def on_trade_closed(
        self,
        trade_id: str,
        current_regime: Optional[str] = None,
    ) -> Dict:
        """
        Called when a trade is closed.
        Generates post-trade review and updated insights.
        """
        context = {
            "trade_id": trade_id,
            "current_regime": current_regime,
            "event": "trade_closed",
        }

        # Post-trade review from coaching agent
        review = self.coaching_agent.get_post_trade_review(trade_id)

        # Collect insights from all agents
        all_insights = []
        for agent_name, agent in self._agents.items():
            try:
                insights = agent.analyze(context)
                all_insights.extend(insights)
            except Exception as e:
                logger.error(f"Agent {agent_name} error on trade close: {e}", exc_info=True)

        severity_order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
        all_insights.sort(key=lambda i: severity_order.get(i.severity, 99))

        return {
            "trade_id": trade_id,
            "review": review,
            "insights": [self._insight_to_dict(i) for i in all_insights],
            "total_insights": len(all_insights),
        }

    # ──────────────────────────────────────
    # CHAT INTERFACE
    # ──────────────────────────────────────

    def chat(
        self,
        message: str,
        current_regime: Optional[str] = None,
    ) -> Dict:
        """
        Handle natural language chat.
        Routes to coaching agent with full context.
        """
        context = {"current_regime": current_regime or "unknown"}

        response = self.coaching_agent.chat(message, context)

        return {
            "user_message": message,
            "response": response,
            "timestamp": datetime.now().isoformat(),
        }

    # ──────────────────────────────────────
    # DAILY INTELLIGENCE
    # ──────────────────────────────────────

    def daily_intelligence_summary(
        self,
        current_regime: Optional[str] = None,
    ) -> Dict:
        """
        Generate a daily intelligence summary.
        Can be shown on dashboard or sent as notification.
        """
        context = {
            "current_regime": current_regime,
            "event": "daily_summary",
            "lookback_trades": 50,
        }

        all_insights = []
        for agent_name, agent in self._agents.items():
            try:
                insights = agent.analyze(context)
                all_insights.extend(insights)
            except Exception as e:
                logger.error(f"Agent {agent_name} error: {e}", exc_info=True)

        # Regime recommendations
        regime_recs = None
        if current_regime:
            recs = self.query_engine.get_strategies_for_regime(current_regime)
            regime_recs = [
                {
                    "strategy": r.strategy_name,
                    "display_name": r.display_name,
                    "fitness": r.fitness_score,
                    "sizing": r.sizing_recommendation,
                }
                for r in recs[:5]
            ]

        severity_order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
        all_insights.sort(key=lambda i: severity_order.get(i.severity, 99))

        return {
            "date": datetime.now().date().isoformat(),
            "regime": current_regime,
            "strategy_recommendations": regime_recs,
            "insights": [self._insight_to_dict(i) for i in all_insights],
            "critical_count": sum(1 for i in all_insights if i.severity == "critical"),
            "warning_count": sum(1 for i in all_insights if i.severity == "warning"),
        }

    # ──────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────

    def _insight_to_dict(self, insight: AgentInsight) -> Dict:
        return {
            "agent": insight.agent_name,
            "type": insight.insight_type,
            "severity": insight.severity,
            "title": insight.title,
            "message": insight.message,
            "strategy": insight.strategy_name,
            "action": insight.action_suggestion,
            "trade_ids": insight.trade_ids,
            "timestamp": insight.timestamp.isoformat(),
        }