"""
AI Coaching Agent — Conversational Trading Intelligence

The coaching agent is the trader's personal AI coach. It synthesizes
data from every subsystem into personalized, actionable insights
delivered through natural conversation.

It does NOT tell the trader what to do. It asks better questions,
surfaces patterns the trader can't see, and reinforces good process.

Core principles:
- Process over outcome
- Questions over directives
- Pattern recognition over prediction
- Reinforcement over criticism
- Context-aware, never generic
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from agents.base_agent import BaseAgent
from agents.llm_client import LLMClient, LLMMessage, LLMResponse
from agents.prompt_templates import PromptTemplates
from agents.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class CoachingMode(str, Enum):
    """Different coaching interaction modes."""
    CONVERSATIONAL = "conversational"       # Free-form chat
    TRADE_REVIEW = "trade_review"           # Reviewing a specific trade
    SESSION_DEBRIEF = "session_debrief"     # End-of-day/session review
    WEEKLY_REVIEW = "weekly_review"         # Weekly performance review
    PRE_SESSION = "pre_session"             # Pre-trading preparation
    BEHAVIORAL_ALERT = "behavioral_alert"   # Triggered by behavioral detection
    STRATEGY_ANALYSIS = "strategy_analysis" # Deep dive on strategy performance
    PATTERN_INSIGHT = "pattern_insight"     # System-detected pattern discussion
    REGIME_GUIDANCE = "regime_guidance"     # Market regime coaching
    GOAL_SETTING = "goal_setting"          # Setting/reviewing trading goals
    COMPETITION_REVIEW = "competition_review"  # Trader vs system comparison coaching


class CoachingPersonality(str, Enum):
    """Coaching style preferences."""
    ANALYTICAL = "analytical"       # Data-heavy, precise, structured
    SOCRATIC = "socratic"          # Question-driven, exploratory
    SUPPORTIVE = "supportive"      # Empathetic, encouraging, gentle
    DIRECT = "direct"              # Blunt, no-nonsense, action-oriented
    BALANCED = "balanced"          # Mix of all approaches


class InsightPriority(str, Enum):
    """Priority level for coaching insights."""
    CRITICAL = "critical"   # Immediate attention needed
    HIGH = "high"           # Important pattern or issue
    MEDIUM = "medium"       # Worth discussing
    LOW = "low"             # Nice to know
    INFO = "info"           # Background context


@dataclass
class AgentInsight:
    """A single coaching insight extracted from analysis."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: InsightPriority = InsightPriority.MEDIUM
    category: str = ""              # e.g., "risk", "behavior", "strategy"
    title: str = ""
    content: str = ""
    evidence: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    related_trades: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "priority": self.priority.value,
            "category": self.category,
            "title": self.title,
            "content": self.content,
            "evidence": self.evidence,
            "action_items": self.action_items,
            "related_trades": self.related_trades,
            "timestamp": self.timestamp.isoformat(),
        }


# Alias for backward-compat — tests import CoachingInsight from this module
CoachingInsight = AgentInsight


@dataclass
class CoachingContext:
    """
    All context assembled for a coaching interaction.

    This is the knowledge the coach has about the trader
    before generating a response.
    """
    # Trader identity
    trader_profile: dict[str, Any] = field(default_factory=dict)
    coaching_personality: CoachingPersonality = CoachingPersonality.BALANCED
    trading_goals: list[str] = field(default_factory=list)

    # Recent performance
    recent_trades: list[dict[str, Any]] = field(default_factory=list)
    todays_trades: list[dict[str, Any]] = field(default_factory=list)
    open_positions: list[dict[str, Any]] = field(default_factory=list)

    # Decision quality
    recent_dq_reports: list[dict[str, Any]] = field(default_factory=list)
    dq_trend: dict[str, Any] = field(default_factory=dict)

    # Behavioral data
    discipline_score: Optional[float] = None
    behavioral_events: list[dict[str, Any]] = field(default_factory=list)
    active_biases: list[str] = field(default_factory=list)
    emotional_state: Optional[str] = None
    deviation_alerts: list[dict[str, Any]] = field(default_factory=list)

    # Strategy data
    strategy_performance: dict[str, Any] = field(default_factory=dict)
    active_strategies: list[str] = field(default_factory=list)
    strategy_attributions: list[dict[str, Any]] = field(default_factory=list)

    # Shadow trading
    shadow_divergences: list[dict[str, Any]] = field(default_factory=list)
    shadow_pnl_delta: Optional[float] = None

    # Market regime
    current_regime: Optional[str] = None
    regime_strategy_matrix: dict[str, Any] = field(default_factory=dict)

    # RAG-retrieved knowledge
    relevant_lessons: list[str] = field(default_factory=list)
    similar_situations: list[str] = field(default_factory=list)
    knowledge_context: list[str] = field(default_factory=list)

    # Structured RAG context (multi-layer, preferred over flat lists above)
    structured_rag_context: Optional[Any] = None  # StructuredContext from rag.context_builder

    # Conversation
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    current_mode: CoachingMode = CoachingMode.CONVERSATIONAL

    # Specific trade under review (if any)
    trade_under_review: Optional[dict[str, Any]] = None
    trade_dq_report: Optional[dict[str, Any]] = None

    def summary_stats(self) -> dict[str, Any]:
        """Quick summary for the prompt."""
        total_recent = len(self.recent_trades)
        wins = sum(
            1 for t in self.recent_trades
            if t.get("outcome") == "win"
        )
        losses = total_recent - wins
        today_count = len(self.todays_trades)
        today_pnl = sum(t.get("pnl", 0) for t in self.todays_trades)

        return {
            "recent_trade_count": total_recent,
            "recent_win_rate": wins / total_recent if total_recent > 0 else 0,
            "today_trade_count": today_count,
            "today_pnl": today_pnl,
            "discipline_score": self.discipline_score,
            "active_bias_count": len(self.active_biases),
            "current_regime": self.current_regime,
            "dq_trend": self.dq_trend.get("trend", "unknown"),
            "dq_average": self.dq_trend.get("average_score", 0),
        }


@dataclass
class CoachingResponse:
    """Structured response from the coaching agent."""
    message: str
    insights: list[AgentInsight] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    suggested_focus: Optional[str] = None
    mode: CoachingMode = CoachingMode.CONVERSATIONAL
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "insights": [i.to_dict() for i in self.insights],
            "action_items": self.action_items,
            "follow_up_questions": self.follow_up_questions,
            "suggested_focus": self.suggested_focus,
            "mode": self.mode.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class CoachingAgent(BaseAgent):
    """
    AI-powered trading coach.

    Synthesizes all system intelligence into personalized,
    context-aware coaching conversations.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_templates: Optional[PromptTemplates] = None,
        response_parser: Optional[ResponseParser] = None,
        personality: CoachingPersonality = CoachingPersonality.BALANCED,
        max_context_trades: int = 20,
        max_conversation_history: int = 20,
    ):
        super().__init__(name="coaching_agent")
        self.llm = llm_client
        self.prompts = prompt_templates or PromptTemplates()
        self.parser = response_parser or ResponseParser()
        self.personality = personality
        self.max_context_trades = max_context_trades
        self.max_conversation_history = max_conversation_history

    async def process(
        self,
        user_message: str,
        context: CoachingContext,
        mode: Optional[CoachingMode] = None,
    ) -> CoachingResponse:
        """
        Process a user message and generate a coaching response.

        Args:
            user_message: The trader's message/question
            context: All assembled trading context
            mode: Override the coaching mode (auto-detected if None)

        Returns:
            CoachingResponse with message, insights, and action items
        """
        # 1. Detect or set the coaching mode
        effective_mode = mode or self._detect_mode(user_message, context)
        context.current_mode = effective_mode

        # 2. Build the system prompt
        system_prompt = self._build_system_prompt(context, effective_mode)

        # 3. Build the context injection
        context_block = self._build_context_block(context, effective_mode)

        # 4. Assemble the message history
        messages = self._assemble_messages(
            system_prompt=system_prompt,
            context_block=context_block,
            user_message=user_message,
            conversation_history=context.conversation_history,
        )

        # 5. Call the LLM
        try:
            llm_response = await self.llm.chat(
                messages=messages,
                temperature=self._get_temperature(effective_mode),
                max_tokens=self._get_max_tokens(effective_mode),
                caller="coaching_agent",
            )
        except Exception as e:
            logger.error(f"LLM call failed in coaching agent: {e}")
            return self._fallback_response(user_message, context, effective_mode)

        # 6. Parse the response
        coaching_response = self._parse_response(
            llm_response, context, effective_mode
        )

        # 7. Generate proactive insights if applicable
        insights = self._generate_proactive_insights(context, effective_mode)
        coaching_response.insights.extend(insights)

        logger.info(
            f"Coaching response generated: mode={effective_mode.value}, "
            f"insights={len(coaching_response.insights)}, "
            f"actions={len(coaching_response.action_items)}"
        )

        return coaching_response

    async def generate_proactive_briefing(
        self,
        context: CoachingContext,
    ) -> CoachingResponse:
        """
        Generate an unprompted coaching briefing.

        Used for:
        - Pre-session preparation
        - Post-session debrief
        - Behavioral alert responses
        - Pattern detection notifications
        """
        # Determine what's most important right now
        priority_topic = self._identify_priority_topic(context)

        if priority_topic == "behavioral_alert":
            return await self._generate_behavioral_briefing(context)
        elif priority_topic == "session_debrief":
            return await self._generate_session_debrief(context)
        elif priority_topic == "pre_session":
            return await self._generate_pre_session_briefing(context)
        elif priority_topic == "pattern_detected":
            return await self._generate_pattern_briefing(context)
        else:
            return await self._generate_daily_briefing(context)

    # ─── Mode Detection ──────────────────────────────────────────

    def _detect_mode(
        self, user_message: str, context: CoachingContext
    ) -> CoachingMode:
        """Auto-detect the coaching mode from message content and context."""
        msg_lower = user_message.lower()

        # Trade review triggers
        if context.trade_under_review is not None:
            return CoachingMode.TRADE_REVIEW
        if any(kw in msg_lower for kw in [
            "review this trade", "what about this trade",
            "analyze this trade", "look at trade",
        ]):
            return CoachingMode.TRADE_REVIEW

        # Session debrief triggers
        if any(kw in msg_lower for kw in [
            "how was my day", "session review", "end of day",
            "how did i do today", "daily review", "debrief",
        ]):
            return CoachingMode.SESSION_DEBRIEF

        # Weekly review triggers
        if any(kw in msg_lower for kw in [
            "weekly review", "this week", "week so far",
            "weekly performance", "week in review",
        ]):
            return CoachingMode.WEEKLY_REVIEW

        # Pre-session triggers
        if any(kw in msg_lower for kw in [
            "getting ready to trade", "pre-market", "before i start",
            "morning prep", "game plan", "what should i focus on",
        ]):
            return CoachingMode.PRE_SESSION

        # Strategy analysis triggers
        if any(kw in msg_lower for kw in [
            "strategy performance", "which strategy",
            "my best strategy", "strategy breakdown",
            "setup analysis", "which setups",
        ]):
            return CoachingMode.STRATEGY_ANALYSIS

        # Regime guidance
        if any(kw in msg_lower for kw in [
            "market regime", "market conditions",
            "what kind of market", "how to trade this market",
        ]):
            return CoachingMode.REGIME_GUIDANCE

        # Behavioral triggers
        if any(kw in msg_lower for kw in [
            "am i tilting", "revenge trading", "feeling",
            "emotional", "discipline", "biases",
            "what am i doing wrong", "bad habits",
        ]):
            return CoachingMode.BEHAVIORAL_ALERT

        # Goal setting
        if any(kw in msg_lower for kw in [
            "set a goal", "my goals", "improve",
            "work on", "focus area", "get better at",
        ]):
            return CoachingMode.GOAL_SETTING

        # Competition review triggers
        if any(kw in msg_lower for kw in [
            "competition", "lagging", "where am i lagging",
            "vs system", "versus system", "system comparison",
            "how am i doing vs", "missed opportunities",
            "what did i miss", "system beat me",
        ]):
            return CoachingMode.COMPETITION_REVIEW

        # Pattern insight
        if any(kw in msg_lower for kw in [
            "patterns", "trends in my trading",
            "what do you notice", "any patterns",
        ]):
            return CoachingMode.PATTERN_INSIGHT

        return CoachingMode.CONVERSATIONAL

    # ─── System Prompt Builder ───────────────────────────────────

    def _build_system_prompt(
        self, context: CoachingContext, mode: CoachingMode
    ) -> str:
        """Build the system prompt based on personality and mode."""
        personality_instructions = self._get_personality_instructions(
            context.coaching_personality
        )
        mode_instructions = self._get_mode_instructions(mode)

        return f"""You are an expert AI trading coach embedded in a Personal Trading Intelligence System (PTIS).

## YOUR ROLE
You are the trader's dedicated performance coach. You have access to their complete trading data, behavioral patterns, strategy performance, and decision quality metrics. You help them become a better, more disciplined, and more self-aware trader.

## CORE COACHING PRINCIPLES
1. **Process Over Outcome**: A losing trade with perfect process is GOOD. A winning trade with terrible process is DANGEROUS. Always emphasize this.
2. **Questions Over Directives**: Guide the trader to their own insights through thoughtful questions. Don't just tell them what to do.
3. **Evidence-Based**: Always ground your coaching in their actual data. Reference specific trades, patterns, and metrics.
4. **Behavioral Awareness**: Help the trader recognize their cognitive biases and emotional patterns without being judgmental.
5. **Constructive Framing**: Frame weaknesses as growth opportunities. Celebrate process improvements.
6. **Specificity**: Never give generic trading advice. Everything should be personalized to THIS trader's data.
7. **Honest**: If the data shows a problem, say so clearly but constructively.

## WHAT YOU NEVER DO
- You NEVER recommend specific trades or positions
- You NEVER predict market direction
- You NEVER give financial advice without grounding in data
- You NEVER dismiss the trader's emotions
- You NEVER use generic platitudes without backing them with their data
- You NEVER encourage overtrading or revenge trading
- You NEVER claim you lack real-time data if live market data is injected in the context — use it

## COACHING PERSONALITY
{personality_instructions}

## CURRENT COACHING MODE
{mode_instructions}

## RESPONSE FORMAT
Respond naturally in conversation. When appropriate, include:
- Specific observations grounded in data
- Thoughtful questions to promote self-reflection
- Action items (prefixed with "→")
- References to specific trades or patterns when relevant

Keep responses focused and actionable. Avoid being overly verbose unless doing a deep review."""

    def _get_personality_instructions(
        self, personality: CoachingPersonality
    ) -> str:
        """Get personality-specific instructions."""
        instructions = {
            CoachingPersonality.ANALYTICAL: (
                "You are precise and data-driven. Lead with numbers, metrics, and statistical observations. "
                "Structure your responses clearly. Use specific percentages and R-multiples. "
                "Think like a performance analyst who happens to be an excellent communicator."
            ),
            CoachingPersonality.SOCRATIC: (
                "You primarily teach through questions. For every observation, pose a reflective question. "
                "Guide the trader to discover insights themselves. Use phrases like 'What do you think was happening when...', "
                "'Have you noticed that...', 'What would happen if you...'. Resist the urge to give direct answers."
            ),
            CoachingPersonality.SUPPORTIVE: (
                "You are warm, empathetic, and encouraging. Acknowledge the emotional difficulty of trading. "
                "Always start by recognizing what the trader did well before addressing areas for improvement. "
                "Use affirming language. Frame challenges as normal parts of the growth process."
            ),
            CoachingPersonality.DIRECT: (
                "You are blunt and action-oriented. Get straight to the point. No sugarcoating. "
                "Tell it like it is, but always with the trader's growth in mind. "
                "Focus on what needs to change NOW. Short, punchy observations followed by clear action items."
            ),
            CoachingPersonality.BALANCED: (
                "You blend analytical rigor with empathetic communication. Lead with data, "
                "but acknowledge the human element. Be direct when needed, supportive when appropriate. "
                "Adapt your tone to what the situation calls for — tough love after repeated mistakes, "
                "encouragement after good process."
            ),
        }
        return instructions.get(personality, instructions[CoachingPersonality.BALANCED])

    def _get_mode_instructions(self, mode: CoachingMode) -> str:
        """Get mode-specific instructions."""
        instructions = {
            CoachingMode.CONVERSATIONAL: (
                "This is a free-form conversation. Respond naturally to the trader's question or comment. "
                "Draw on all available context to provide relevant, personalized responses."
            ),
            CoachingMode.TRADE_REVIEW: (
                "You are reviewing a specific trade. Analyze the decision quality across all 6 dimensions: "
                "Setup Quality, Entry Precision, Risk Management, Exit Execution, Process Adherence, and Behavioral Score. "
                "Focus on what was done well and what could improve. Compare to the shadow trade if available. "
                "End with 1-2 specific, actionable takeaways."
            ),
            CoachingMode.SESSION_DEBRIEF: (
                "You are conducting an end-of-session debrief. Cover: "
                "1) Overall performance summary, 2) Best and worst decisions (process, not P&L), "
                "3) Behavioral observations, 4) Pattern recognition, 5) One key lesson for tomorrow. "
                "Keep it structured but conversational."
            ),
            CoachingMode.WEEKLY_REVIEW: (
                "You are conducting a weekly performance review. Provide: "
                "1) Week-over-week performance trends, 2) Decision quality evolution, "
                "3) Strategy performance breakdown, 4) Behavioral patterns this week, "
                "5) Top 3 focus areas for next week. Be thorough but prioritized."
            ),
            CoachingMode.PRE_SESSION: (
                "You are helping the trader prepare for their trading session. Cover: "
                "1) Current market regime and what it means for their strategies, "
                "2) Key behavioral patterns to watch for today, 3) Which strategies are performing best "
                "in current conditions, 4) Specific focus area based on recent performance. "
                "Keep it brief and energizing — don't overwhelm before they start."
            ),
            CoachingMode.BEHAVIORAL_ALERT: (
                "A behavioral pattern has been detected. Your goal is to bring awareness without being "
                "preachy or condescending. Help the trader recognize the pattern, understand its impact, "
                "and decide if they want to adjust. Use their own historical data to show the pattern."
            ),
            CoachingMode.STRATEGY_ANALYSIS: (
                "Deep dive into strategy performance. Analyze which strategies have edge, "
                "in which regimes, at what times. Identify strategies that may have lost edge. "
                "Help the trader understand their strategy allocation."
            ),
            CoachingMode.PATTERN_INSIGHT: (
                "Share a pattern the system has detected in the trader's behavior or performance. "
                "Present it as an observation, not a judgment. Ask the trader if they've noticed it too. "
                "Discuss potential causes and implications."
            ),
            CoachingMode.REGIME_GUIDANCE: (
                "Discuss current market regime and its implications for the trader's strategies. "
                "Reference their historical performance in similar regimes. "
                "Suggest strategy adjustments (not trades) based on regime data."
            ),
            CoachingMode.GOAL_SETTING: (
                "Help the trader set or review their trading goals. Goals should be process-based "
                "(not P&L targets). Use their data to suggest realistic, measurable goals. "
                "Examples: 'Maintain 80%+ checklist completion', 'Keep DQ score above B', "
                "'Limit to 3 trades per day'."
            ),
            CoachingMode.COMPETITION_REVIEW: (
                "You are reviewing the trader's performance relative to the system's autonomous "
                "virtual trades. Structure feedback across 5 categories: "
                "1) SETUP SELECTION — Are they picking the right trades? "
                "2) RISK MANAGEMENT — Are they sizing and stopping correctly? "
                "3) EXIT EXECUTION — Are they leaving money on the table? "
                "4) DISCIPLINE — Are they following their own rules? "
                "5) AWARENESS — Are they missing what the system sees? "
                "Present quick wins first, then deeper analysis. Reference specific metric gaps "
                "between trader and system performance. Highlight missed opportunities."
            ),
        }
        return instructions.get(mode, instructions[CoachingMode.CONVERSATIONAL])

    # ─── Context Block Builder ───────────────────────────────────

    @staticmethod
    def _market_session_block() -> str:
        """Always-injected block: current IST time + NSE session status."""
        try:
            import pytz
            ist = pytz.timezone("Asia/Kolkata")
            now = datetime.now(ist)
        except Exception:
            now = datetime.now(timezone.utc)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        is_weekday = now.weekday() < 5
        t = (now.hour, now.minute)
        is_open = is_weekday and (9, 15) <= t <= (15, 30)
        if not is_weekday:
            session_status = "CLOSED (weekend)"
        elif t < (9, 15):
            session_status = f"PRE-MARKET — opens in {(9*60+15) - (now.hour*60+now.minute)} min"
        elif t > (15, 30):
            session_status = "CLOSED — market ended for today"
        else:
            mins_remaining = (15 * 60 + 30) - (now.hour * 60 + now.minute)
            session_status = f"OPEN — {mins_remaining} min remaining"
        return (
            f"## Current Session\n"
            f"Time (IST): {now.strftime('%I:%M %p')} {day_names[now.weekday()]}\n"
            f"NSE Market: {session_status}\n"
            f"Date: {now.strftime('%d %b %Y')}"
        )

    def _build_context_block(
        self, context: CoachingContext, mode: CoachingMode
    ) -> str:
        """Build the data context block injected into the conversation."""
        sections = []
        stats = context.summary_stats()

        # Always inject current IST time and market session status
        sections.append(self._market_session_block())

        # Always include basic profile
        sections.append(self._format_profile_section(context))

        # Always include summary stats
        sections.append(self._format_stats_section(stats))

        # Mode-specific context
        if mode == CoachingMode.TRADE_REVIEW:
            sections.append(self._format_trade_review_section(context))
        elif mode == CoachingMode.SESSION_DEBRIEF:
            sections.append(self._format_session_section(context))
        elif mode == CoachingMode.WEEKLY_REVIEW:
            sections.append(self._format_weekly_section(context))
        elif mode == CoachingMode.PRE_SESSION:
            sections.append(self._format_pre_session_section(context))
        elif mode == CoachingMode.STRATEGY_ANALYSIS:
            sections.append(self._format_strategy_section(context))
        elif mode == CoachingMode.REGIME_GUIDANCE:
            sections.append(self._format_regime_section(context))
        elif mode == CoachingMode.COMPETITION_REVIEW:
            sections.append(self._format_competition_section(context))

        # Always include behavioral context if present
        if context.behavioral_events or context.active_biases:
            sections.append(self._format_behavioral_section(context))

        # Include DQ trend if available
        if context.dq_trend:
            sections.append(self._format_dq_section(context))

        # Include shadow divergences if present
        if context.shadow_divergences:
            sections.append(self._format_shadow_section(context))

        # Include RAG-retrieved knowledge
        if context.relevant_lessons or context.knowledge_context:
            sections.append(self._format_knowledge_section(context))

        return "\n\n".join(sections)

    def _format_profile_section(self, context: CoachingContext) -> str:
        profile = context.trader_profile
        goals = context.trading_goals

        lines = ["## TRADER PROFILE"]
        if profile:
            if profile.get("name"):
                lines.append(f"Name: {profile['name']}")
            if profile.get("experience_level"):
                lines.append(f"Experience: {profile['experience_level']}")
            if profile.get("preferred_markets"):
                lines.append(f"Markets: {', '.join(profile['preferred_markets'])}")
            if profile.get("trading_style"):
                lines.append(f"Style: {profile['trading_style']}")
            if profile.get("account_size"):
                lines.append(f"Account Size: ${profile['account_size']:,.0f}")
        if goals:
            lines.append(f"Current Goals: {'; '.join(goals)}")

        return "\n".join(lines)

    def _format_stats_section(self, stats: dict[str, Any]) -> str:
        lines = ["## CURRENT STATS"]
        lines.append(f"Recent trades: {stats['recent_trade_count']}")
        lines.append(f"Recent win rate: {stats['recent_win_rate']:.1%}")
        lines.append(f"Today's trades: {stats['today_trade_count']}")
        lines.append(f"Today's P&L: ${stats['today_pnl']:+,.2f}")

        if stats.get("discipline_score") is not None:
            lines.append(f"Discipline score: {stats['discipline_score']:.2f}")
        if stats.get("active_bias_count", 0) > 0:
            lines.append(f"⚠️ Active biases detected: {stats['active_bias_count']}")
        if stats.get("current_regime"):
            lines.append(f"Market regime: {stats['current_regime']}")
        if stats.get("dq_average", 0) > 0:
            lines.append(
                f"Decision quality: {stats['dq_average']:.2f} "
                f"(trend: {stats.get('dq_trend', 'unknown')})"
            )

        return "\n".join(lines)

    def _format_trade_review_section(self, context: CoachingContext) -> str:
        lines = ["## TRADE UNDER REVIEW"]
        trade = context.trade_under_review
        if not trade:
            return "## TRADE UNDER REVIEW\nNo specific trade selected."

        lines.append(f"Symbol: {trade.get('symbol', 'N/A')}")
        lines.append(f"Direction: {trade.get('direction', 'N/A')}")
        lines.append(f"Entry: ${trade.get('entry_price', 0):.2f} at {trade.get('entry_time', 'N/A')}")
        lines.append(f"Exit: ${trade.get('exit_price', 0):.2f} at {trade.get('exit_time', 'N/A')}")
        lines.append(f"P&L: ${trade.get('pnl', 0):+,.2f}")
        lines.append(f"R-Multiple: {trade.get('r_multiple', 'N/A')}")
        lines.append(f"Strategy: {trade.get('strategy', 'unattributed')}")
        lines.append(f"Emotional State: {trade.get('emotional_state', 'not recorded')}")
        lines.append(f"Exit Reason: {trade.get('exit_reason', 'not recorded')}")

        if trade.get("notes"):
            lines.append(f"Trader Notes: {trade['notes']}")

        # Decision quality report
        dq = context.trade_dq_report
        if dq:
            lines.append(f"\n### Decision Quality: {dq.get('grade', 'N/A')} ({dq.get('composite_score', 0):.2f})")
            dims = dq.get("dimensions", {})
            for dim_name, dim_data in dims.items():
                label = dim_name.replace("_", " ").title()
                lines.append(f"  {label}: {dim_data.get('score', 0):.0%}")
                for f in dim_data.get("factors", []):
                    lines.append(f"    ✓ {f}")
                for d in dim_data.get("deductions", []):
                    lines.append(f"    ✗ {d}")

            if dq.get("coaching_notes"):
                lines.append("\nSystem coaching notes:")
                for note in dq["coaching_notes"]:
                    lines.append(f"  • {note}")

        return "\n".join(lines)

    def _format_session_section(self, context: CoachingContext) -> str:
        lines = ["## TODAY'S SESSION"]
        trades = context.todays_trades

        if not trades:
            lines.append("No trades today.")
            return "\n".join(lines)

        total_pnl = sum(t.get("pnl", 0) for t in trades)
        wins = sum(1 for t in trades if t.get("outcome") == "win")
        losses = len(trades) - wins

        lines.append(f"Total trades: {len(trades)}")
        lines.append(f"Wins/Losses: {wins}/{losses}")
        lines.append(f"Total P&L: ${total_pnl:+,.2f}")

        # List each trade briefly
        for i, t in enumerate(trades, 1):
            r = t.get("r_multiple", "?")
            strategy = t.get("strategy", "unattributed")
            grade = t.get("dq_grade", "?")
            lines.append(
                f"  {i}. {t.get('symbol', '?')} {t.get('direction', '?')} — "
                f"${t.get('pnl', 0):+,.2f} ({r}R) — "
                f"Strategy: {strategy} — DQ: {grade}"
            )

        return "\n".join(lines)

    def _format_weekly_section(self, context: CoachingContext) -> str:
        lines = ["## THIS WEEK'S DATA"]
        trades = context.recent_trades

        if not trades:
            lines.append("No trades this week.")
            return "\n".join(lines)

        total_pnl = sum(t.get("pnl", 0) for t in trades)
        wins = sum(1 for t in trades if t.get("outcome") == "win")
        avg_r = (
            sum(t.get("r_multiple", 0) for t in trades) / len(trades)
            if trades else 0
        )

        lines.append(f"Total trades: {len(trades)}")
        lines.append(f"Win rate: {wins / len(trades):.1%}")
        lines.append(f"Total P&L: ${total_pnl:+,.2f}")
        lines.append(f"Average R: {avg_r:+.2f}")

        # Strategy breakdown
        strat_perf = context.strategy_performance
        if strat_perf:
            lines.append("\nStrategy breakdown:")
            for strat, perf in strat_perf.items():
                lines.append(
                    f"  {strat}: {perf.get('count', 0)} trades, "
                    f"win rate {perf.get('win_rate', 0):.1%}, "
                    f"avg R: {perf.get('avg_r', 0):+.2f}"
                )

        return "\n".join(lines)

    def _format_pre_session_section(self, context: CoachingContext) -> str:
        lines = ["## PRE-SESSION CONTEXT"]

        if context.current_regime:
            lines.append(f"Current market regime: {context.current_regime}")

        if context.regime_strategy_matrix:
            lines.append("\nStrategy performance in current regime:")
            matrix = context.regime_strategy_matrix
            for strat, data in matrix.items():
                lines.append(
                    f"  {strat}: edge={data.get('edge', 'unknown')}, "
                    f"win_rate={data.get('win_rate', 0):.1%}"
                )

        if context.active_biases:
            lines.append(f"\n⚠️ Active biases to watch: {', '.join(context.active_biases)}")

        if context.deviation_alerts:
            lines.append("\nRecent behavioral alerts:")
            for alert in context.deviation_alerts[:3]:
                lines.append(f"  • {alert.get('description', 'unknown')}")

        return "\n".join(lines)

    def _format_strategy_section(self, context: CoachingContext) -> str:
        lines = ["## STRATEGY PERFORMANCE DATA"]
        perf = context.strategy_performance
        if not perf:
            lines.append("No strategy performance data available.")
            return "\n".join(lines)

        for strat, data in perf.items():
            lines.append(f"\n### {strat}")
            lines.append(f"  Trades: {data.get('count', 0)}")
            lines.append(f"  Win rate: {data.get('win_rate', 0):.1%}")
            lines.append(f"  Avg R: {data.get('avg_r', 0):+.2f}")
            lines.append(f"  Expectancy: {data.get('expectancy', 0):+.3f}")
            if data.get("best_regime"):
                lines.append(f"  Best regime: {data['best_regime']}")
            if data.get("worst_regime"):
                lines.append(f"  Worst regime: {data['worst_regime']}")

        return "\n".join(lines)

    def _format_regime_section(self, context: CoachingContext) -> str:
        lines = ["## MARKET REGIME CONTEXT"]
        lines.append(f"Current regime: {context.current_regime or 'unknown'}")

        if context.regime_strategy_matrix:
            lines.append("\nHistorical strategy performance in this regime:")
            for strat, data in context.regime_strategy_matrix.items():
                lines.append(
                    f"  {strat}: win_rate={data.get('win_rate', 0):.1%}, "
                    f"avg_r={data.get('avg_r', 0):+.2f}, "
                    f"sample={data.get('count', 0)} trades"
                )

        return "\n".join(lines)

    def _format_behavioral_section(self, context: CoachingContext) -> str:
        lines = ["## BEHAVIORAL INTELLIGENCE"]

        if context.discipline_score is not None:
            lines.append(f"Discipline score: {context.discipline_score:.2f}")

        if context.emotional_state:
            lines.append(f"Current emotional state: {context.emotional_state}")

        if context.active_biases:
            lines.append(f"Active biases: {', '.join(context.active_biases)}")

        if context.behavioral_events:
            lines.append("\nRecent behavioral events:")
            for event in context.behavioral_events[:5]:
                lines.append(
                    f"  [{event.get('severity', 'info')}] {event.get('description', 'unknown')} "
                    f"({event.get('timestamp', '')})"
                )

        if context.deviation_alerts:
            lines.append("\nDeviation alerts:")
            for alert in context.deviation_alerts[:3]:
                lines.append(f"  ⚠️ {alert.get('description', 'unknown')}")

        return "\n".join(lines)

    def _format_dq_section(self, context: CoachingContext) -> str:
        lines = ["## DECISION QUALITY TREND"]
        trend = context.dq_trend

        lines.append(f"Average score: {trend.get('average_score', 0):.2f}")
        lines.append(f"Trend: {trend.get('trend', 'unknown')}")
        lines.append(f"Quality trade rate: {trend.get('quality_rate', 0):.1%}")

        dim_avgs = trend.get("dimension_averages", {})
        if dim_avgs:
            lines.append("\nDimension averages:")
            for dim, avg in sorted(dim_avgs.items(), key=lambda x: x[1]):
                label = dim.replace("_", " ").title()
                lines.append(f"  {label}: {avg:.2f}")

        grades = trend.get("grade_distribution", {})
        if grades:
            lines.append(f"\nGrade distribution: {grades}")

        return "\n".join(lines)

    def _format_shadow_section(self, context: CoachingContext) -> str:
        lines = ["## SHADOW TRADING DIVERGENCES"]

        if context.shadow_pnl_delta is not None:
            lines.append(
                f"Shadow P&L delta (you vs optimal): ${context.shadow_pnl_delta:+,.2f}"
            )

        for div in context.shadow_divergences[:5]:
            lines.append(
                f"  • {div.get('symbol', '?')}: "
                f"Your exit ${div.get('actual_pnl', 0):+,.2f} vs "
                f"Shadow ${div.get('shadow_pnl', 0):+,.2f} — "
                f"{div.get('reason', 'unknown')}"
            )

        return "\n".join(lines)

    def _format_competition_section(self, context: CoachingContext) -> str:
        """Format competition framework data for coaching context."""
        lines = ["## COMPETITION: TRADER vs SYSTEM"]

        # Shadow comparison data from context
        if context.shadow_divergences:
            lines.append("\nDivergences between your trades and system shadows:")
            for div in context.shadow_divergences[:10]:
                if isinstance(div, dict):
                    lines.append(
                        f"  - {div.get('symbol', '?')}: "
                        f"Trader P&L={div.get('trader_pnl', '?')}, "
                        f"System P&L={div.get('system_pnl', '?')}"
                    )
                else:
                    lines.append(f"  - {div}")

        if context.shadow_pnl_delta is not None:
            lines.append(f"\nOverall P&L gap (system - trader): ${context.shadow_pnl_delta:+,.2f}")

        # Recent trade summary for comparison
        trades = context.recent_trades
        if trades:
            wins = sum(1 for t in trades if t.get("outcome") == "win")
            total_pnl = sum(t.get("pnl", 0) for t in trades)
            lines.append(f"\nYour recent stats: {len(trades)} trades, "
                         f"{wins}/{len(trades)} wins, P&L=${total_pnl:+,.2f}")

        if not context.shadow_divergences and context.shadow_pnl_delta is None:
            lines.append(
                "No competition data available yet. "
                "The system needs autonomous shadow trades to compare against."
            )

        return "\n".join(lines)

    def _format_knowledge_section(self, context: CoachingContext) -> str:
        # Prefer structured RAG context if available (multi-layer, compressed)
        if context.structured_rag_context is not None:
            try:
                return "## RELEVANT KNOWLEDGE (from trader's history)\n" + \
                    context.structured_rag_context.format_for_prompt(max_tokens=1500)
            except Exception:
                pass  # fall through to legacy

        lines = ["## RELEVANT KNOWLEDGE (from trader's history)"]

        if context.relevant_lessons:
            lines.append("Past lessons:")
            for lesson in context.relevant_lessons[:5]:
                lines.append(f"  • {lesson}")

        if context.similar_situations:
            lines.append("\nSimilar past situations:")
            for sit in context.similar_situations[:3]:
                lines.append(f"  • {sit}")

        if context.knowledge_context:
            lines.append("\nAdditional context:")
            for ctx in context.knowledge_context[:3]:
                lines.append(f"  • {ctx}")

        return "\n".join(lines)

    # ─── Message Assembly ────────────────────────────────────────

    def _assemble_messages(
        self,
        system_prompt: str,
        context_block: str,
        user_message: str,
        conversation_history: list[dict[str, str]],
    ) -> list[LLMMessage]:
        """Assemble the full message list for the LLM call."""
        messages: list[LLMMessage] = []

        # System prompt
        messages.append(LLMMessage(role="system", content=system_prompt))

        # Context injection as a system-level context message
        messages.append(LLMMessage(
            role="system",
            content=f"## TRADER DATA CONTEXT\n\n{context_block}",
        ))

        # Conversation history (trimmed)
        history = conversation_history[-self.max_conversation_history:]
        for entry in history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if role in {"user", "assistant"}:
                messages.append(LLMMessage(role=role, content=content))

        # Current user message
        messages.append(LLMMessage(role="user", content=user_message))

        return messages

    # ─── Response Parsing ────────────────────────────────────────

    def _parse_response(
        self,
        llm_response: LLMResponse,
        context: CoachingContext,
        mode: CoachingMode,
    ) -> CoachingResponse:
        """Parse the LLM response into a structured CoachingResponse."""
        raw_text = llm_response.content

        # Extract action items (lines starting with →)
        action_items = []
        for line in raw_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("→") or stripped.startswith("->"):
                action_items.append(stripped.lstrip("→->").strip())

        # Extract follow-up questions (lines ending with ?)
        follow_ups = []
        for line in raw_text.split("\n"):
            stripped = line.strip()
            if (
                stripped.endswith("?")
                and len(stripped) > 15
                and not stripped.startswith("→")
            ):
                follow_ups.append(stripped)

        # Limit follow-ups to most relevant
        follow_ups = follow_ups[:3]

        # Determine suggested focus
        suggested_focus = self._extract_focus_area(raw_text, context)

        return CoachingResponse(
            message=raw_text,
            insights=[],  # Proactive insights added separately
            action_items=action_items,
            follow_up_questions=follow_ups,
            suggested_focus=suggested_focus,
            mode=mode,
            confidence=0.85,
            metadata={
                "model": getattr(llm_response, "model", "unknown"),
                "tokens_used": getattr(llm_response, "tokens_used", 0),
            },
        )

    def _extract_focus_area(
        self, text: str, context: CoachingContext
    ) -> Optional[str]:
        """Extract the primary suggested focus area from the response."""
        # Simple heuristic — could be enhanced with structured output
        focus_keywords = {
            "risk management": "risk_management",
            "position sizing": "risk_management",
            "stop loss": "risk_management",
            "entry": "entry_precision",
            "patience": "behavioral",
            "discipline": "process_adherence",
            "emotional": "behavioral",
            "bias": "behavioral",
            "exit": "exit_execution",
            "setup": "setup_quality",
            "checklist": "process_adherence",
            "regime": "regime_awareness",
            "strategy selection": "strategy_selection",
        }

        text_lower = text.lower()
        for keyword, area in focus_keywords.items():
            if keyword in text_lower:
                return area

        return None

    # ─── BaseAgent interface ─────────────────────────────────────

    def analyze(self, context: dict) -> list:
        """
        Satisfy BaseAgent abstract interface.
        CoachingAgent is conversational (async via process()); sync deterministic
        insights come from the specialist agents (strategy, behavior).
        Returns empty list — coaching insights flow through chat endpoints.
        """
        return []

    def get_pre_trade_briefing(
        self,
        strategy_name: str,
        current_regime: str,
        trader_emotion: str | None = None,
    ) -> dict:
        """
        Synchronous pre-trade briefing (no LLM call).
        Called by AgentOrchestrator.on_trade_entered() for immediate insights.
        Returns a lightweight dict; full AI coaching is async via process().
        """
        return {
            "strategy": strategy_name,
            "regime": current_regime,
            "note": (
                f"Trading {strategy_name} in {current_regime} regime. "
                + (f"Emotional state noted: {trader_emotion}. " if trader_emotion else "")
                + "Use the coaching chat for deeper pre-trade analysis."
            ),
        }

    # ─── Proactive Insights ──────────────────────────────────────

    def _generate_proactive_insights(
        self, context: CoachingContext, mode: CoachingMode
    ) -> list[AgentInsight]:
        """
        Generate data-driven insights that supplement the LLM response.

        These are deterministic, rule-based insights derived from the data.
        """
        insights = []

        # Critical: Process violation streak
        if context.recent_dq_reports:
            recent_violations = sum(
                1 for r in context.recent_dq_reports[-5:]
                if r.get("is_process_violation", False)
            )
            if recent_violations >= 3:
                insights.append(AgentInsight(
                    priority=InsightPriority.CRITICAL,
                    category="process",
                    title="Process Violation Streak",
                    content=(
                        f"{recent_violations} of your last 5 trades had process violations. "
                        "This is a critical pattern that needs immediate attention."
                    ),
                    action_items=[
                        "Review and commit to your pre-trade checklist",
                        "Consider reducing position size until process improves",
                    ],
                ))

        # High: Declining decision quality
        if context.dq_trend.get("trend") == "declining":
            insights.append(AgentInsight(
                priority=InsightPriority.HIGH,
                category="decision_quality",
                title="Declining Decision Quality",
                content=(
                    f"Your decision quality trend is declining "
                    f"(avg: {context.dq_trend.get('average_score', 0):.2f}). "
                    "This often precedes a drawdown."
                ),
                action_items=[
                    "Identify which dimension is dropping most",
                    "Consider reducing trading frequency until the trend stabilizes",
                ],
            ))

        # High: Active cognitive biases
        if len(context.active_biases) >= 2:
            insights.append(AgentInsight(
                priority=InsightPriority.HIGH,
                category="behavioral",
                title="Multiple Active Biases",
                content=(
                    f"Active biases detected: {', '.join(context.active_biases)}. "
                    "Multiple simultaneous biases significantly impair decision quality."
                ),
                action_items=[
                    "Take a 10-minute break before your next trade",
                    "Review your emotional state honestly",
                ],
            ))

        # Medium: Shadow trading divergence
        if (
            context.shadow_pnl_delta is not None
            and context.shadow_pnl_delta < -500
        ):
            insights.append(AgentInsight(
                priority=InsightPriority.MEDIUM,
                category="execution",
                title="Significant Shadow Divergence",
                content=(
                    f"Your trades are underperforming the shadow by "
                    f"${abs(context.shadow_pnl_delta):,.2f}. "
                    "This suggests execution is costing you edge."
                ),
                evidence=[
                    d.get("reason", "")
                    for d in context.shadow_divergences[:3]
                ],
                action_items=[
                    "Review your largest divergences to find the pattern",
                ],
            ))

        # Medium: Overtrading detection
        today_count = len(context.todays_trades)
        profile = context.trader_profile
        max_trades = profile.get("max_daily_trades", 5)
        if today_count > max_trades:
            insights.append(AgentInsight(
                priority=InsightPriority.MEDIUM,
                category="behavioral",
                title="Overtrading Alert",
                content=(
                    f"You've taken {today_count} trades today "
                    f"(your limit is {max_trades}). "
                    "Overtrading typically reduces decision quality."
                ),
                action_items=[
                    "Consider stopping for the day",
                    "If you continue, increase your checklist rigor",
                ],
            ))

        # Info: Strategy performing well in current regime
        if context.current_regime and context.regime_strategy_matrix:
            best_strat = None
            best_edge = -999
            for strat, data in context.regime_strategy_matrix.items():
                edge = data.get("avg_r", 0)
                if edge > best_edge:
                    best_edge = edge
                    best_strat = strat

            if best_strat and best_edge > 0.5:
                insights.append(AgentInsight(
                    priority=InsightPriority.INFO,
                    category="strategy",
                    title="Strategy-Regime Alignment",
                    content=(
                        f"'{best_strat}' has the strongest edge in the current "
                        f"'{context.current_regime}' regime (avg R: {best_edge:+.2f})."
                    ),
                ))

        return insights

    def _identify_priority_topic(self, context: CoachingContext) -> str:
        """Identify the most important topic for a proactive briefing."""
        # Critical behavioral alerts take priority
        if context.active_biases and len(context.active_biases) >= 2:
            return "behavioral_alert"

        if context.deviation_alerts:
            severity_counts = {}
            for alert in context.deviation_alerts:
                sev = alert.get("severity", "low")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            if severity_counts.get("critical", 0) > 0:
                return "behavioral_alert"

        # Session debrief if there are today's trades and it's late
        if context.todays_trades:
            return "session_debrief"

        # Pre-session if no trades today
        if not context.todays_trades:
            return "pre_session"

        # Pattern detection
        if context.dq_trend.get("trend") == "declining":
            return "pattern_detected"

        return "daily_briefing"

    # ─── Proactive Briefings ─────────────────────────────────────

    async def _generate_behavioral_briefing(
        self, context: CoachingContext
    ) -> CoachingResponse:
        """Generate a briefing focused on behavioral concerns."""
        biases = ", ".join(context.active_biases) if context.active_biases else "none detected"

        prompt = (
            f"The system has detected behavioral concerns for this trader. "
            f"Active biases: {biases}. "
            f"Discipline score: {context.discipline_score}. "
            f"Generate a brief, supportive but honest check-in. "
            f"Help them become aware without being preachy."
        )
        return await self.process(
            prompt, context, mode=CoachingMode.BEHAVIORAL_ALERT
        )

    async def _generate_session_debrief(
        self, context: CoachingContext
    ) -> CoachingResponse:
        """Generate an end-of-session debrief."""
        trade_count = len(context.todays_trades)
        total_pnl = sum(t.get("pnl", 0) for t in context.todays_trades)

        prompt = (
            f"Generate an end-of-session debrief. "
            f"The trader took {trade_count} trades today for "
            f"${total_pnl:+,.2f} total P&L. "
            f"Focus on process quality, not just results."
        )
        return await self.process(
            prompt, context, mode=CoachingMode.SESSION_DEBRIEF
        )

    async def _generate_pre_session_briefing(
        self, context: CoachingContext
    ) -> CoachingResponse:
        """Generate a pre-session preparation briefing."""
        prompt = (
            "Generate a pre-session briefing for the trader. "
            "Help them prepare mentally and strategically. "
            "Keep it brief and energizing."
        )
        return await self.process(
            prompt, context, mode=CoachingMode.PRE_SESSION
        )

    async def _generate_pattern_briefing(
        self, context: CoachingContext
    ) -> CoachingResponse:
        """Generate a briefing about a detected pattern."""
        trend = context.dq_trend.get("trend", "unknown")
        prompt = (
            f"The system has detected a pattern in the trader's performance: "
            f"decision quality trend is '{trend}'. "
            f"Share this observation and help them understand what's happening."
        )
        return await self.process(
            prompt, context, mode=CoachingMode.PATTERN_INSIGHT
        )

    async def _generate_daily_briefing(
        self, context: CoachingContext
    ) -> CoachingResponse:
        """Generate a general daily briefing."""
        prompt = (
            "Generate a brief daily trading intelligence briefing. "
            "Cover the most important thing the trader should know right now."
        )
        return await self.process(
            prompt, context, mode=CoachingMode.CONVERSATIONAL
        )

    # ─── Fallback ────────────────────────────────────────────────

    def _fallback_response(
        self,
        user_message: str,
        context: CoachingContext,
        mode: CoachingMode,
    ) -> CoachingResponse:
        """Generate a fallback response when the LLM is unavailable."""
        insights = self._generate_proactive_insights(context, mode)

        if insights:
            top_insight = insights[0]
            message = (
                f"I'm having trouble connecting to my language model right now, "
                f"but here's what the data is telling me:\n\n"
                f"**{top_insight.title}**\n{top_insight.content}"
            )
            if top_insight.action_items:
                message += "\n\nSuggested actions:\n"
                for action in top_insight.action_items:
                    message += f"→ {action}\n"
        else:
            message = (
                "I'm having trouble connecting right now. "
                "Try again in a moment. In the meantime, "
                "review your pre-trade checklist and check your emotional state."
            )

        return CoachingResponse(
            message=message,
            insights=insights,
            action_items=[],
            follow_up_questions=[],
            mode=mode,
            confidence=0.3,
            metadata={"fallback": True},
        )

    # ─── Configuration Helpers ───────────────────────────────────

    def _get_temperature(self, mode: CoachingMode) -> float:
        """Get LLM temperature based on coaching mode."""
        temps = {
            CoachingMode.TRADE_REVIEW: 0.3,         # More analytical
            CoachingMode.SESSION_DEBRIEF: 0.4,
            CoachingMode.WEEKLY_REVIEW: 0.3,
            CoachingMode.STRATEGY_ANALYSIS: 0.2,     # Very analytical
            CoachingMode.PRE_SESSION: 0.5,
            CoachingMode.BEHAVIORAL_ALERT: 0.4,
            CoachingMode.CONVERSATIONAL: 0.6,        # More creative
            CoachingMode.PATTERN_INSIGHT: 0.4,
            CoachingMode.REGIME_GUIDANCE: 0.3,
            CoachingMode.GOAL_SETTING: 0.5,
        }
        return temps.get(mode, 0.5)

    def _get_max_tokens(self, mode: CoachingMode) -> int:
        """Get max response tokens based on coaching mode."""
        tokens = {
            CoachingMode.TRADE_REVIEW: 800,
            CoachingMode.SESSION_DEBRIEF: 1000,
            CoachingMode.WEEKLY_REVIEW: 1500,
            CoachingMode.STRATEGY_ANALYSIS: 1200,
            CoachingMode.PRE_SESSION: 600,
            CoachingMode.BEHAVIORAL_ALERT: 500,
            CoachingMode.CONVERSATIONAL: 700,
            CoachingMode.PATTERN_INSIGHT: 600,
            CoachingMode.REGIME_GUIDANCE: 700,
            CoachingMode.GOAL_SETTING: 800,
        }
        return tokens.get(mode, 700)
    """
Add this method to the CoachingAgent class to handle discretionary queries
by integrating the intelligence layer assessment.
"""

# In CoachingAgent class, add:

async def process_discretionary_query(
    self,
    user_message: str,
    context: CoachingContext,
    assessment: "DiscretionaryAssessment",
) -> CoachingResponse:
    """
    Process a discretionary trade query using intelligence assessment.

    This wraps the standard process() method but injects the quantitative
    assessment into the context so the LLM can reference it.
    """
    # Inject the assessment into context
    assessment_context = assessment.context_for_coaching

    # Build the discretionary system prompt addition
    discretionary_block = self._build_discretionary_context_block(assessment)

    # Override mode
    mode = CoachingMode.CONVERSATIONAL

    # Build system prompt
    system_prompt = self._build_system_prompt(context, mode)
    system_prompt += f"""

## DISCRETIONARY QUERY ANALYSIS

The trader is asking about a potential trade. The system has performed
a quantitative analysis. Use this data to inform your response.

IMPORTANT RULES FOR DISCRETIONARY QUERIES:
1. NEVER say "buy this" or "sell this" — you are not recommending trades
2. Present the analysis objectively
3. Highlight both favorable and unfavorable factors
4. Always mention active behavioral guardrails
5. Help the trader think through the decision themselves
6. Ask at least one question that makes them reflect before acting
7. Reference their specific data — win rates, R:R, regime performance
8. If the recommendation is "avoid", be direct about why

Respond in a conversational coaching style that:
- Presents the environment assessment
- Shows how the setup matches (or doesn't) their strategy
- Shares the R:R estimate with confidence level
- Flags any behavioral concerns
- Ends with a reflective question
"""

    # Build context block with discretionary data
    context_block = self._build_context_block(context, mode)
    context_block += f"\n\n{discretionary_block}"

    # Assemble messages
    messages = self._assemble_messages(
        system_prompt=system_prompt,
        context_block=context_block,
        user_message=user_message,
        conversation_history=context.conversation_history,
    )

    # Call LLM
    try:
        llm_response = await self.llm.chat(
            messages=messages,
            temperature=0.4,  # More analytical for trade queries
            max_tokens=1000,
            caller="coaching_agent:discretionary",
        )
    except Exception as e:
        logger.error(f"LLM call failed for discretionary query: {e}")
        return self._build_discretionary_fallback(assessment)

    # Parse response
    coaching_response = self._parse_response(llm_response, context, mode)

    # Add assessment insights
    if assessment.guardrails:
        for g in assessment.guardrails:
            if g.severity == "critical":
                coaching_response.insights.append(AgentInsight(
                    priority=InsightPriority.CRITICAL,
                    category="behavioral_guardrail",
                    title=f"⚠️ {g.message[:60]}",
                    content=g.message,
                    action_items=[g.recommendation],
                ))
            elif g.severity == "warning":
                coaching_response.insights.append(AgentInsight(
                    priority=InsightPriority.HIGH,
                    category="behavioral_guardrail",
                    title=f"⚡ {g.message[:60]}",
                    content=g.message,
                    action_items=[g.recommendation],
                ))

    coaching_response.metadata["discretionary_assessment"] = {
        "recommendation": assessment.recommendation,
        "confidence": assessment.confidence,
        "setup_valid": assessment.setup_valid,
        "guardrail_count": len(assessment.guardrails),
    }

    return coaching_response


def _build_discretionary_context_block(
    self, assessment: "DiscretionaryAssessment"
) -> str:
    """Build the context block for a discretionary query."""
    lines = ["## QUANTITATIVE TRADE ASSESSMENT"]
    lines.append(f"Symbol: {assessment.symbol}")
    lines.append(f"System recommendation: {assessment.recommendation.upper()}")
    lines.append(f"Confidence: {assessment.confidence:.0%}")
    lines.append("")

    # Environment
    if assessment.environment:
        env = assessment.environment
        lines.append("### Market Environment")
        lines.append(f"  Regime: {env.regime.value}")
        lines.append(f"  Trend: {'Bullish' if env.trend.direction > 0 else 'Bearish' if env.trend.direction < 0 else 'Neutral'} ({env.trend.strength.value})")
        lines.append(f"  Volatility: {env.volatility.state.value}")
        lines.append(f"  Environment Score: {env.environment_score:.0%}")
        lines.append(f"  Price: ${env.current_price:.2f}")
        if env.vwap:
            lines.append(f"  VWAP: ${env.vwap:.2f} ({'above' if env.current_price > env.vwap else 'below'})")
        if env.structure.nearest_support:
            lines.append(f"  Nearest Support: ${env.structure.nearest_support:.2f}")
        if env.structure.nearest_resistance:
            lines.append(f"  Nearest Resistance: ${env.structure.nearest_resistance:.2f}")
        if env.structure.has_structure_break:
            lines.append("  ⚡ Structure break detected")
        if env.structure.has_liquidity_sweep:
            lines.append("  💧 Liquidity sweep detected")
        lines.append("")

    # Setup match
    if assessment.best_match:
        m = assessment.best_match
        lines.append("### Setup Analysis")
        lines.append(f"  Best match: {m.strategy_name}")
        lines.append(f"  Readiness: {m.readiness_score:.0%}")
        lines.append(f"  Conditions: {m.conditions_met}/{m.conditions_total} met")
        lines.append(f"  Regime aligned: {'✅ Yes' if m.regime_aligned else '❌ No'}")
        lines.append(f"  Status: {'🔔 TRIGGERED' if m.is_triggered else '⚡ FORMING' if m.is_forming else 'Not ready'}")

        missing = [c for c in m.conditions if not c.is_met]
        if missing:
            lines.append("  Missing conditions:")
            for c in missing[:5]:
                prox = f" ({c.proximity:.0%} close)" if c.proximity > 0.5 else ""
                lines.append(f"    • {c.name}{prox}")
        lines.append("")

    # R:R estimate
    if assessment.r_estimate:
        r = assessment.r_estimate
        lines.append("### R:R Estimation")
        lines.append(f"  Expected R: {r.expected_r:+.2f}")
        lines.append(f"  R:R Ratio: {r.expected_rr_ratio:.1f}:1")
        lines.append(f"  Win probability: {r.win_probability:.0%}")
        lines.append(f"  Estimate confidence: {r.confidence:.0%}")
        lines.append(f"  Upside R (75th pct): {r.upside_r:+.2f}")
        lines.append(f"  Downside R (25th pct): {r.downside_r:+.2f}")
        lines.append(f"  Assessment: {assessment.risk_reward_assessment}")
        lines.append("")

    # Reasons
    lines.append("### Assessment Factors")
    for reason in assessment.recommendation_reasons:
        prefix = "✅" if any(p in reason.lower() for p in ["favorable", "aligned", "matches"]) else "⚠️"
        lines.append(f"  {prefix} {reason}")
    lines.append("")

    # Guardrails
    if assessment.guardrails:
        lines.append("### Behavioral Guardrails")
        for g in assessment.guardrails:
            icon = "🔴" if g.severity == "critical" else "🟡" if g.severity == "warning" else "ℹ️"
            lines.append(f"  {icon} [{g.severity.upper()}] {g.message}")
            lines.append(f"     → {g.recommendation}")

    return "\n".join(lines)


def _build_discretionary_fallback(
    self, assessment: "DiscretionaryAssessment"
) -> CoachingResponse:
    """Build a fallback response from assessment data when LLM fails."""
    parts = [
        f"Here's what the data shows for {assessment.symbol}:\n"
    ]

    if assessment.environment:
        env = assessment.environment
        parts.append(
            f"**Environment:** {env.regime.value} regime, "
            f"{env.trend.strength.value} {env.trend.direction > 0 and 'bullish' or 'bearish'} trend, "
            f"{env.volatility.state.value} volatility."
        )

    if assessment.best_match:
        m = assessment.best_match
        parts.append(
            f"\n**Best setup match:** {m.strategy_name} "
            f"({m.conditions_met}/{m.conditions_total} conditions, "
            f"{m.readiness_score:.0%} ready)"
        )

    if assessment.r_estimate:
        parts.append(
            f"\n**R:R estimate:** {assessment.r_estimate.expected_rr_ratio:.1f}:1 "
            f"({assessment.r_estimate.win_probability:.0%} win probability)"
        )

    parts.append(f"\n**Recommendation:** {assessment.recommendation.upper()}")

    for reason in assessment.recommendation_reasons:
        parts.append(f"→ {reason}")

    if assessment.guardrails:
        parts.append("\n**⚠️ Guardrails:**")
        for g in assessment.guardrails:
            parts.append(f"• [{g.severity}] {g.message}")

    return CoachingResponse(
        message="\n".join(parts),
        insights=[],
        action_items=[],
        follow_up_questions=[
            "Does this align with your pre-trade plan?",
            "What's your planned exit if this goes against you?",
        ],
        mode=CoachingMode.CONVERSATIONAL,
        confidence=0.3,
        metadata={"fallback": True, "discretionary": True},
    )