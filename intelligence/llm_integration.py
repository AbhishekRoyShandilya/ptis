"""
PTIS LLM Integration Service
=============================

RAG-based system that combines world-class LLM trading knowledge
with personalized trader data for hyper-personalized coaching.

Key Features:
- RAG (Retrieval-Augmented Generation) with trader-specific data
- LLM integration (OpenAI GPT, Claude, etc.)
- Personalized context injection
- Strategy adaptation based on individual behavior
- Real-time coaching responses
- Knowledge base of trading strategies

Design
------
- Minimal ML training required
- Leverage existing LLM trading knowledge
- Personalize through context and behavioral data
- Fast response times
- Explainable AI responses
"""

from __future__ import annotations

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

# Use unified LLM client from agents module
try:
    from agents.llm_client import LLMClient, LLMMessage
    _has_llm_client = True
except ImportError:
    _has_llm_client = False

from core.events import Event, EventType, get_event_bus
from core.models import Trade, BehavioralEvent, DecisionQualityRecord
from intelligence.profile_synthesizer import UnifiedTraderProfile
from intelligence.market_context import MarketContext
from behavioral.pattern_service import get_pattern_service
from intelligence.insight_memory import get_insight_service

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class QueryType(str, Enum):
    """Types of coaching queries."""
    STRATEGY_ADVICE = "strategy_advice"
    RISK_ASSESSMENT = "risk_assessment"
    ENTRY_DECISION = "entry_decision"
    EXIT_DECISION = "exit_decision"
    PSYCHOLOGY_COACHING = "psychology_coaching"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    REGIME_ADAPTATION = "regime_adaptation"


@dataclass
class TraderContext:
    """Comprehensive trader context for LLM personalization."""
    
    trader_id: str
    
    # Behavioral Profile
    risk_style: str  # conservative, moderate, aggressive
    discipline_score: float
    consistency_score: float
    patience_score: float
    impulsiveness_score: float
    
    # Trading Style
    preferred_timeframes: List[str]
    favorite_strategies: List[str]
    average_holding_period: str
    position_sizing_style: str
    
    # Performance Metrics
    win_rate: float
    average_r_multiple: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    
    # Recent Behavior
    recent_trades: List[Dict[str, Any]]
    recent_mistakes: List[str]
    recent_successes: List[str]
    emotional_state: str
    
    # Market Context
    current_regime: str
    volatility_level: str
    session_type: str
    market_sentiment: str
    
    # Strategy Performance
    strategy_success_rates: Dict[str, float]
    setup_success_rates: Dict[str, float]
    time_of_day_performance: Dict[str, float]
    
    # Psychological Patterns
    common_biases: List[str]
    trigger_situations: List[str]
    emotional_triggers: List[str]
    
    def to_llm_context(self) -> str:
        """Convert to LLM-friendly context string."""
        context_parts = [
            f"Trader Profile for {self.trader_id}:",
            f"Risk Style: {self.risk_style}",
            f"Discipline Score: {self.discipline_score}/100",
            f"Consistency: {self.consistency_score}/100",
            f"Patience: {self.patience_score}/100",
            f"Impulsiveness: {self.impulsiveness_score}/100",
            "",
            "Trading Style:",
            f"Preferred Timeframes: {', '.join(self.preferred_timeframes)}",
            f"Favorite Strategies: {', '.join(self.favorite_strategies)}",
            f"Average Holding Period: {self.average_holding_period}",
            f"Position Sizing: {self.position_sizing_style}",
            "",
            "Performance Metrics:",
            f"Win Rate: {self.win_rate:.1%}",
            f"Average R-Multiple: {self.average_r_multiple:.2f}",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Max Drawdown: {self.max_drawdown:.1%}",
            "",
            "Recent Behavior:",
            f"Recent Mistakes: {', '.join(self.recent_mistakes[:3])}",
            f"Recent Successes: {', '.join(self.recent_successes[:3])}",
            f"Current Emotional State: {self.emotional_state}",
            "",
            "Market Context:",
            f"Current Regime: {self.current_regime}",
            f"Volatility: {self.volatility_level}",
            f"Session: {self.session_type}",
            "",
            "Strategy Performance:",
        ]
        
        for strategy, success_rate in self.strategy_success_rates.items():
            context_parts.append(f"- {strategy}: {success_rate:.1%} win rate")
        
        context_parts.extend([
            "",
            "Psychological Patterns:",
            f"Common Biases: {', '.join(self.common_biases)}",
            f"Trigger Situations: {', '.join(self.trigger_situations)}",
            f"Emotional Triggers: {', '.join(self.emotional_triggers)}"
        ])
        
        return "\n".join(context_parts)


@dataclass
class CoachingQuery:
    """Coaching query from trader."""
    
    query_id: str
    trader_id: str
    query_type: QueryType
    question: str
    
    # Context data
    current_trade: Optional[Dict[str, Any]] = None
    chart_context: Optional[Dict[str, Any]] = None
    market_data: Optional[Dict[str, Any]] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    urgency: str = "normal"  # low, normal, high, urgent


@dataclass
class LLMResponse:
    """Response from LLM with personalization."""
    
    query_id: str
    trader_id: str
    response: str
    
    # Personalization metadata
    personalization_factors: List[str]
    confidence_score: float
    relevant_patterns: List[str]
    
    # Action items
    recommended_actions: List[str]
    risk_warnings: List[str]
    
    # Follow-up
    follow_up_questions: List[str]
    
    # Metadata
    llm_provider: LLMProvider
    model_used: str
    response_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class LLMIntegrationService:
    """
    Main service for LLM integration with personalized trading coaching.
    
    Combines world-class LLM knowledge with individual trader data
    for hyper-personalized coaching responses.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider = LLMProvider.OPENAI,
        model_name: str = "gpt-4-turbo-preview"
    ):
        self.llm_provider = llm_provider
        self.model_name = model_name
        
        # Initialize LLM client
        self._init_llm_client()
        
        # Services for data retrieval
        self.pattern_service = get_pattern_service()
        self.insight_service = get_insight_service()
        
        # Context cache
        self.context_cache: Dict[str, Tuple[TraderContext, datetime]] = {}
        self.cache_ttl_minutes = 15
        
        # Knowledge base prompts
        self._init_knowledge_base()
        
        logger.info(f"LLMIntegrationService initialized with {llm_provider}")
    
    def _init_llm_client(self) -> None:
        """Initialize LLM client using shared agents.llm_client."""
        if _has_llm_client:
            provider = "anthropic" if self.llm_provider == LLMProvider.ANTHROPIC else "openai"
            self._llm = LLMClient(provider=provider, model=self.model_name)
        else:
            self._llm = None
            logger.warning("LLM client unavailable — agents.llm_client not importable")
    
    def _init_knowledge_base(self) -> None:
        """Initialize knowledge base prompts."""
        self.knowledge_base = {
            "trading_strategies": {
                "breakout": "Breakout strategies involve trading price movements beyond established support/resistance levels. Key factors include volume confirmation, false breakout patterns, and momentum indicators.",
                "trend_following": "Trend following strategies identify and trade in the direction of established market trends. Focus on trend strength, pullback entries, and trailing stops.",
                "mean_reversion": "Mean reversion strategies bet on prices returning to historical averages. Key elements include overbought/oversold conditions and support/resistance levels.",
                "momentum": "Momentum strategies trade strong price movements with acceleration. Focus on rate of change, volume spikes, and continuation patterns."
            },
            "risk_management": {
                "position_sizing": "Position sizing should account for account size, risk tolerance, and trade setup quality. Common methods include fixed percentage, volatility-based sizing, and Kelly criterion.",
                "stop_loss": "Stop losses should be placed at logical price levels where the trade setup is invalidated. Consider volatility, recent price action, and support/resistance.",
                "risk_reward": "Minimum risk-reward ratios should be 2:1 for most strategies, adjusted for win rate and market conditions."
            },
            "psychology": {
                "fear": "Fear in trading often leads to premature exits or missed opportunities. Address through systematic planning and risk management.",
                "greed": "Greed causes overtrading and excessive risk. Combat with position size limits and predefined profit targets.",
                "revenge": "Revenge trading after losses typically leads to further losses. Implement cooling-off periods and stick to your plan."
            }
        }
    
    async def process_coaching_query(self, query: CoachingQuery) -> LLMResponse:
        """
        Process a coaching query with personalized LLM response.
        
        Parameters
        ----------
        query : CoachingQuery
            Trader's coaching query
        
        Returns
        -------
        LLMResponse
            Personalized response with recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            # Get personalized trader context
            trader_context = await self._get_trader_context(query.trader_id)
            
            # Build personalized prompt
            personalized_prompt = self._build_personalized_prompt(
                query, trader_context
            )
            
            # Get LLM response
            llm_response = await self._call_llm(personalized_prompt)
            
            # Parse and structure response
            structured_response = self._parse_llm_response(
                llm_response, query, trader_context
            )
            
            # Calculate response time
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create response object
            response = LLMResponse(
                query_id=query.query_id,
                trader_id=query.trader_id,
                response=structured_response["response"],
                personalization_factors=structured_response["personalization_factors"],
                confidence_score=structured_response["confidence_score"],
                relevant_patterns=structured_response["relevant_patterns"],
                recommended_actions=structured_response["recommended_actions"],
                risk_warnings=structured_response["risk_warnings"],
                follow_up_questions=structured_response["follow_up_questions"],
                llm_provider=self.llm_provider,
                model_used=self.model_name,
                response_time_ms=response_time
            )
            
            logger.info(f"Processed coaching query for {query.trader_id} in {response_time:.0f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process coaching query: {e}")
            return self._create_error_response(query, str(e))
    
    async def _get_trader_context(self, trader_id: str) -> TraderContext:
        """Get comprehensive trader context from REAL DB data."""
        # Check cache first
        cached_context, cached_at = self.context_cache.get(trader_id, (None, None))

        if cached_context and (datetime.utcnow() - cached_at) < timedelta(minutes=self.cache_ttl_minutes):
            return cached_context

        # Pull real data from DB
        from core.database import SessionLocal
        from core.enums import TradeStatus
        from sqlalchemy import func

        db = SessionLocal()
        try:
            cutoff_90d = datetime.utcnow() - timedelta(days=90)

            # Recent closed trades
            trades = (
                db.query(Trade)
                .filter(
                    Trade.trader_id == trader_id,
                    Trade.status == TradeStatus.CLOSED,
                    Trade.entry_time >= cutoff_90d,
                )
                .order_by(Trade.entry_time.desc())
                .limit(100)
                .all()
            )

            n = len(trades)
            winners = [t for t in trades if (t.pnl or 0) > 0]
            r_vals = [t.r_multiple for t in trades if t.r_multiple is not None]

            win_rate = len(winners) / n if n else 0
            avg_r = sum(r_vals) / len(r_vals) if r_vals else 0
            gross_profit = sum(t.pnl for t in winners) if winners else 0
            losers = [t for t in trades if (t.pnl or 0) < 0]
            gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
            pf = (gross_profit / gross_loss) if gross_loss > 0 else 0

            # Discipline from attributions
            from core.models import StrategyAttribution
            trade_ids = [t.id for t in trades]
            disc_scores = []
            if trade_ids:
                rows = (
                    db.query(StrategyAttribution.discipline_score)
                    .filter(StrategyAttribution.trade_id.in_(trade_ids[:50]))
                    .all()
                )
                disc_scores = [s[0] for s in rows if s[0] is not None]

            discipline = sum(disc_scores) / len(disc_scores) if disc_scores else 0

            # Strategy breakdown
            strat_map: Dict[str, list] = defaultdict(list)
            for t in trades:
                sid = t.declared_strategy_id or "unknown"
                strat_map[sid].append(1 if (t.pnl or 0) > 0 else 0)
            strat_rates = {k: (sum(v) / len(v)) for k, v in strat_map.items() if len(v) >= 3}

            # Recent behavioral events
            recent_events = (
                db.query(BehavioralEvent)
                .filter(BehavioralEvent.trader_id == trader_id)
                .order_by(BehavioralEvent.created_at.desc())
                .limit(20)
                .all()
            )
            mistakes = list({e.event_type.value for e in recent_events if "deviation" in (e.event_type.value or "").lower() or "override" in (e.event_type.value or "").lower()})[:5]
            successes = list({e.event_type.value for e in recent_events if "clean" in (e.event_type.value or "").lower()})[:5]

            # Build context from REAL data
            context = TraderContext(
                trader_id=trader_id,
                risk_style="unknown" if n < 10 else ("conservative" if avg_r < 0.8 else "moderate" if avg_r < 1.5 else "aggressive"),
                discipline_score=round(discipline, 1),
                consistency_score=0.0,  # requires deeper analysis
                patience_score=0.0,
                impulsiveness_score=0.0,
                preferred_timeframes=[],
                favorite_strategies=list(strat_rates.keys())[:3],
                average_holding_period="unknown",
                position_sizing_style="unknown",
                win_rate=round(win_rate, 3),
                average_r_multiple=round(avg_r, 2),
                profit_factor=round(pf, 2),
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                recent_trades=[
                    {"symbol": t.symbol, "pnl": t.pnl, "r": t.r_multiple, "strategy": t.declared_strategy_id}
                    for t in trades[:5]
                ],
                recent_mistakes=mistakes if mistakes else ["insufficient data"],
                recent_successes=successes if successes else ["insufficient data"],
                emotional_state="unknown",
                current_regime="unknown",
                volatility_level="unknown",
                session_type="regular",
                market_sentiment="unknown",
                strategy_success_rates=strat_rates,
                setup_success_rates={},
                time_of_day_performance={},
                common_biases=[],
                trigger_situations=[],
                emotional_triggers=[],
            )
        finally:
            db.close()

        # Cache context
        self.context_cache[trader_id] = (context, datetime.utcnow())
        return context
    
    def _build_personalized_prompt(
        self,
        query: CoachingQuery,
        context: TraderContext,
    ) -> str:
        """Build personalized user prompt for LLM (system prompt is separate)."""
        parts = [
            f"Trader Context (from REAL data — {context.win_rate:.0%} win rate over recent trades):",
            context.to_llm_context(),
            "",
            f"Query Type: {query.query_type.value}",
            f"Question: {query.question}",
        ]

        if query.query_type == QueryType.ENTRY_DECISION and query.current_trade:
            parts.append(f"\nCurrent Trade Setup:\n{json.dumps(query.current_trade, indent=2)}")

        if query.chart_context:
            parts.append(f"\nChart Context:\n{json.dumps(query.chart_context, indent=2)}")

        if query.market_data:
            parts.append(f"\nMarket Data:\n{json.dumps(query.market_data, indent=2)}")

        parts.append(
            "\nRespond with:\n"
            "1. **Personalized Analysis** — how this applies to THIS trader's data\n"
            "2. **Recommendations** — specific, actionable advice grounded in the numbers\n"
            "3. **Risk Considerations** — warnings based on their actual profile\n"
            "4. **Follow-up** — questions to deepen understanding"
        )

        return "\n".join(parts)

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM via unified agents.llm_client with proper system prompt."""
        if self._llm is None:
            raise RuntimeError("LLM client not available — install openai or anthropic")

        from agents.prompts import COACHING_INTELLIGENCE

        messages = [
            LLMMessage(role="system", content=COACHING_INTELLIGENCE),
            LLMMessage(role="user", content=prompt),
        ]
        response = await self._llm.chat(messages, max_tokens=1000, temperature=0.5, caller="intelligence:coaching")
        return response.content
    
    def _parse_llm_response(
        self, 
        llm_response: str, 
        query: CoachingQuery,
        context: TraderContext
    ) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        
        # Extract personalization factors mentioned
        personalization_factors = self._extract_personalization_factors(
            llm_response, context
        )
        
        # Extract relevant patterns
        relevant_patterns = self._extract_relevant_patterns(
            llm_response, context
        )
        
        # Extract recommended actions
        recommended_actions = self._extract_actions(llm_response)
        
        # Extract risk warnings
        risk_warnings = self._extract_risk_warnings(llm_response)
        
        # Extract follow-up questions
        follow_up_questions = self._extract_follow_up_questions(llm_response)
        
        # Calculate confidence based on specificity and personalization
        confidence_score = self._calculate_confidence_score(
            llm_response, personalization_factors, context
        )
        
        return {
            "response": llm_response,
            "personalization_factors": personalization_factors,
            "relevant_patterns": relevant_patterns,
            "recommended_actions": recommended_actions,
            "risk_warnings": risk_warnings,
            "follow_up_questions": follow_up_questions,
            "confidence_score": confidence_score
        }
    
    def _extract_personalization_factors(
        self, 
        response: str, 
        context: TraderContext
    ) -> List[str]:
        """Extract personalization factors mentioned in response."""
        factors = []
        
        # Check for mentions of specific trader characteristics
        personalization_keywords = {
            "risk_style": ["risk", "conservative", "aggressive", "moderate"],
            "discipline": ["discipline", "consistency", "rules", "plan"],
            "patience": ["patient", "wait", "timing", "premature"],
            "impulsiveness": ["impulsive", "rush", "quick", "immediate"],
            "performance": ["win rate", "profit factor", "drawdown", "r-multiple"]
        }
        
        response_lower = response.lower()
        
        for factor, keywords in personalization_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                factors.append(factor)
        
        return factors
    
    def _extract_relevant_patterns(
        self, 
        response: str, 
        context: TraderContext
    ) -> List[str]:
        """Extract relevant behavioral patterns mentioned."""
        patterns = []
        
        # Check for mentions of specific patterns
        pattern_keywords = {
            "early_entry": ["early", "anticipate", "premature"],
            "late_exit": ["late", "hold", "missed"],
            "revenge_trading": ["revenge", "get back", "recover"],
            "fomo": ["fear of missing", "fomo", "chasing"],
            "overtrading": ["overtrade", "too many", "excessive"]
        }
        
        response_lower = response.lower()
        
        for pattern, keywords in pattern_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                patterns.append(pattern)
        
        return patterns
    
    def _extract_actions(self, response: str) -> List[str]:
        """Extract recommended actions from response."""
        actions = []
        
        # Look for action-oriented phrases
        action_indicators = [
            "consider", "try", "implement", "use", "set", "place",
            "wait for", "look for", "focus on", "practice", "review"
        ]
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in action_indicators):
                # Clean up the action
                action = line.strip().lstrip('- ').lstrip('* ').strip()
                if action and len(action) > 10:  # Filter out short fragments
                    actions.append(action)
        
        return actions[:5]  # Limit to top 5 actions
    
    def _extract_risk_warnings(self, response: str) -> List[str]:
        """Extract risk warnings from response."""
        warnings = []
        
        # Look for risk-related phrases
        risk_indicators = [
            "risk", "warning", "caution", "be careful", "avoid",
            "dangerous", "risky", "potential loss"
        ]
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in risk_indicators):
                warning = line.strip().lstrip('- ').lstrip('* ').strip()
                if warning and len(warning) > 10:
                    warnings.append(warning)
        
        return warnings[:3]  # Limit to top 3 warnings
    
    def _extract_follow_up_questions(self, response: str) -> List[str]:
        """Extract follow-up questions from response."""
        questions = []
        
        # Look for question marks and question phrases
        lines = response.split('\n')
        for line in lines:
            if '?' in line:
                question = line.strip().lstrip('- ').lstrip('* ').strip()
                if question and len(question) > 10:
                    questions.append(question)
        
        return questions[:3]  # Limit to top 3 questions
    
    def _calculate_confidence_score(
        self, 
        response: str, 
        personalization_factors: List[str],
        context: TraderContext
    ) -> float:
        """Calculate confidence score based on personalization and specificity."""
        
        # Base confidence
        confidence = 0.5
        
        # Boost for personalization
        personalization_boost = min(len(personalization_factors) * 0.1, 0.3)
        confidence += personalization_boost
        
        # Boost for specific actions
        action_count = len(self._extract_actions(response))
        action_boost = min(action_count * 0.05, 0.2)
        confidence += action_boost
        
        # Boost for risk warnings
        if self._extract_risk_warnings(response):
            confidence += 0.1
        
        # Cap at 0.95
        return min(confidence, 0.95)
    
    def _create_error_response(self, query: CoachingQuery, error: str) -> LLMResponse:
        """Create error response when LLM call fails."""
        return LLMResponse(
            query_id=query.query_id,
            trader_id=query.trader_id,
            response=f"I apologize, but I'm unable to process your request at the moment. Error: {error}",
            personalization_factors=[],
            confidence_score=0.0,
            relevant_patterns=[],
            recommended_actions=["Please try again later"],
            risk_warnings=["Technical difficulties detected"],
            follow_up_questions=[],
            llm_provider=self.llm_provider,
            model_used=self.model_name,
            response_time_ms=0.0
        )
    
    async def get_strategy_recommendations(
        self, 
        trader_id: str, 
        market_regime: str,
        setup_type: str
    ) -> Dict[str, Any]:
        """Get personalized strategy recommendations."""
        
        context = await self._get_trader_context(trader_id)
        
        prompt = f"""Based on this trader's profile and current market conditions, recommend the most suitable strategies:

Trader Context:
{context.to_llm_context()}

Current Market Regime: {market_regime}
Setup Type: {setup_type}

Provide:
1. Best matching strategies for this trader in current conditions
2. Specific entry/exit considerations
3. Risk management adjustments
4. Psychological preparation tips"""

        llm_response = await self._call_llm(prompt)
        
        return {
            "recommendations": llm_response,
            "trader_context": context.to_llm_context(),
            "market_regime": market_regime,
            "setup_type": setup_type
        }
    
    def clear_context_cache(self, trader_id: Optional[str] = None) -> None:
        """Clear cached trader contexts."""
        if trader_id:
            self.context_cache.pop(trader_id, None)
        else:
            self.context_cache.clear()
        logger.info(f"Cleared context cache for {trader_id or 'all traders'}")


# Global instance
_llm_service: Optional[LLMIntegrationService] = None

def get_llm_service() -> LLMIntegrationService:
    """Get global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMIntegrationService()
    return _llm_service
