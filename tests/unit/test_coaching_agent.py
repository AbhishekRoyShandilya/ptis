"""Tests for the AI Coaching Agent."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from agents.coaching_agent import (
    CoachingAgent,
    CoachingContext,
    CoachingMode,
    CoachingPersonality,
    CoachingResponse,
    CoachingInsight,
    InsightPriority,
)
from agents.llm_client import LLMClient, LLMMessage, LLMResponse


# ─── Fixtures ────────────────────────────────────────────────────


def _make_llm_client(response_text: str = "Great trading session!") -> LLMClient:
    """Create a mock LLM client."""
    client = AsyncMock(spec=LLMClient)
    client.chat = AsyncMock(return_value=LLMResponse(
        content=response_text,
        model="gpt-4-test",
        tokens_used=150,
        prompt_tokens=100,
        completion_tokens=50,
    ))
    return client


def _make_context(**overrides) -> CoachingContext:
    """Create a test coaching context."""
    ctx = CoachingContext(
        trader_profile=overrides.get("trader_profile", {
            "name": "TestTrader",
            "experience_level": "intermediate",
            "preferred_markets": ["US_EQUITIES"],
            "trading_style": "day_trading",
            "account_size": 50000,
            "max_daily_trades": 5,
        }),
        coaching_personality=overrides.get(
            "personality", CoachingPersonality.BALANCED
        ),
        trading_goals=overrides.get("trading_goals", [
            "Maintain 70%+ checklist completion",
            "Keep average DQ score above B",
        ]),
        recent_trades=overrides.get("recent_trades", [
            {
                "id": "t1", "symbol": "AAPL", "direction": "long",
                "pnl": 250, "r_multiple": 1.5, "outcome": "win",
                "strategy": "opening_breakout",
            },
            {
                "id": "t2", "symbol": "TSLA", "direction": "long",
                "pnl": -150, "r_multiple": -1.0, "outcome": "loss",
                "strategy": "pullback_to_ema",
            },
        ]),
        todays_trades=overrides.get("todays_trades", [
            {
                "id": "t3", "symbol": "NVDA", "direction": "long",
                "pnl": 180, "r_multiple": 1.2, "outcome": "win",
                "strategy": "vwap_bounce", "dq_grade": "B+",
            },
        ]),
        recent_dq_reports=overrides.get("recent_dq_reports", [
            {"composite_score": 0.78, "grade": "B", "is_quality_trade": True,
             "is_process_violation": False},
            {"composite_score": 0.65, "grade": "C+", "is_quality_trade": False,
             "is_process_violation": False},
        ]),
        dq_trend=overrides.get("dq_trend", {
            "average_score": 0.72,
            "trend": "stable",
            "quality_rate": 0.65,
        }),
        discipline_score=overrides.get("discipline_score", 0.78),
        behavioral_events=overrides.get("behavioral_events", []),
        active_biases=overrides.get("active_biases", []),
        current_regime=overrides.get("current_regime", "trending_bullish"),
        conversation_history=overrides.get("conversation_history", []),
    )
    return ctx


# ─── Tests: Mode Detection ──────────────────────────────────────


class TestModeDetection:

    def setup_method(self):
        self.agent = CoachingAgent(llm_client=_make_llm_client())

    def test_detect_trade_review_from_context(self):
        ctx = _make_context()
        ctx.trade_under_review = {"symbol": "AAPL"}
        mode = self.agent._detect_mode("Tell me about this", ctx)
        assert mode == CoachingMode.TRADE_REVIEW

    def test_detect_trade_review_from_message(self):
        ctx = _make_context()
        mode = self.agent._detect_mode("Review this trade for me", ctx)
        assert mode == CoachingMode.TRADE_REVIEW

    def test_detect_session_debrief(self):
        ctx = _make_context()
        mode = self.agent._detect_mode("How was my day?", ctx)
        assert mode == CoachingMode.SESSION_DEBRIEF

    def test_detect_weekly_review(self):
        ctx = _make_context()
        mode = self.agent._detect_mode("Weekly review please", ctx)
        assert mode == CoachingMode.WEEKLY_REVIEW

    def test_detect_pre_session(self):
        ctx = _make_context()
        mode = self.agent._detect_mode("Getting ready to trade, what should I focus on?", ctx)
        assert mode == CoachingMode.PRE_SESSION

    def test_detect_behavioral(self):
        ctx = _make_context()
        mode = self.agent._detect_mode("Am I tilting right now?", ctx)
        assert mode == CoachingMode.BEHAVIORAL_ALERT

    def test_detect_strategy_analysis(self):
        ctx = _make_context()
        mode = self.agent._detect_mode("Which strategy is my best?", ctx)
        assert mode == CoachingMode.STRATEGY_ANALYSIS

    def test_detect_regime_guidance(self):
        ctx = _make_context()
        mode = self.agent._detect_mode("What kind of market are we in?", ctx)
        assert mode == CoachingMode.REGIME_GUIDANCE

    def test_detect_goal_setting(self):
        ctx = _make_context()
        mode = self.agent._detect_mode("I want to set a goal to improve my exits", ctx)
        assert mode == CoachingMode.GOAL_SETTING

    def test_detect_pattern_insight(self):
        ctx = _make_context()
        mode = self.agent._detect_mode("Do you notice any patterns in my trading?", ctx)
        assert mode == CoachingMode.PATTERN_INSIGHT

    def test_default_conversational(self):
        ctx = _make_context()
        mode = self.agent._detect_mode("Hello, how are you?", ctx)
        assert mode == CoachingMode.CONVERSATIONAL


# ─── Tests: System Prompt Building ───────────────────────────────


class TestSystemPrompt:

    def setup_method(self):
        self.agent = CoachingAgent(llm_client=_make_llm_client())

    def test_system_prompt_contains_core_principles(self):
        ctx = _make_context()
        prompt = self.agent._build_system_prompt(ctx, CoachingMode.CONVERSATIONAL)
        assert "Process Over Outcome" in prompt
        assert "Questions Over Directives" in prompt
        assert "Evidence-Based" in prompt
        assert "NEVER recommend specific trades" in prompt

    def test_personality_in_prompt(self):
        ctx = _make_context(personality=CoachingPersonality.SOCRATIC)
        prompt = self.agent._build_system_prompt(ctx, CoachingMode.CONVERSATIONAL)
        assert "question" in prompt.lower()

    def test_direct_personality(self):
        ctx = _make_context(personality=CoachingPersonality.DIRECT)
        prompt = self.agent._build_system_prompt(ctx, CoachingMode.CONVERSATIONAL)
        assert "blunt" in prompt.lower()

    def test_mode_instructions_in_prompt(self):
        ctx = _make_context()
        prompt = self.agent._build_system_prompt(ctx, CoachingMode.TRADE_REVIEW)
        assert "6 dimensions" in prompt or "Setup Quality" in prompt

    def test_session_debrief_mode_instructions(self):
        ctx = _make_context()
        prompt = self.agent._build_system_prompt(ctx, CoachingMode.SESSION_DEBRIEF)
        assert "debrief" in prompt.lower()


# ─── Tests: Context Block Building ───────────────────────────────


class TestContextBlock:

    def setup_method(self):
        self.agent = CoachingAgent(llm_client=_make_llm_client())

    def test_context_includes_profile(self):
        ctx = _make_context()
        block = self.agent._build_context_block(ctx, CoachingMode.CONVERSATIONAL)
        assert "TRADER PROFILE" in block
        assert "TestTrader" in block

    def test_context_includes_stats(self):
        ctx = _make_context()
        block = self.agent._build_context_block(ctx, CoachingMode.CONVERSATIONAL)
        assert "CURRENT STATS" in block

    def test_context_includes_behavioral_when_present(self):
        ctx = _make_context(
            active_biases=["recency_bias", "loss_aversion"],
            behavioral_events=[
                {"event_type": "cognitive_bias", "description": "recency_bias",
                 "severity": "medium", "timestamp": "2024-01-15T10:00:00"},
            ],
        )
        block = self.agent._build_context_block(ctx, CoachingMode.CONVERSATIONAL)
        assert "BEHAVIORAL INTELLIGENCE" in block
        assert "recency_bias" in block

    def test_trade_review_context_includes_trade(self):
        ctx = _make_context()
        ctx.trade_under_review = {
            "symbol": "AAPL", "direction": "long",
            "entry_price": 150.0, "exit_price": 153.0,
            "pnl": 300, "r_multiple": 2.0,
            "strategy": "opening_breakout",
            "emotional_state": "focused",
            "exit_reason": "target_hit",
            "entry_time": "2024-01-15T09:35:00",
            "exit_time": "2024-01-15T10:15:00",
        }
        ctx.trade_dq_report = {
            "grade": "A",
            "composite_score": 0.88,
            "dimensions": {
                "setup_quality": {"score": 0.90, "factors": ["Good setup"], "deductions": []},
            },
            "coaching_notes": ["Well executed trade."],
        }

        block = self.agent._build_context_block(ctx, CoachingMode.TRADE_REVIEW)
        assert "TRADE UNDER REVIEW" in block
        assert "AAPL" in block
        assert "Decision Quality: A" in block

    def test_shadow_section_included(self):
        ctx = _make_context()
        ctx.shadow_divergences = [
            {
                "symbol": "AAPL",
                "actual_pnl": 200,
                "shadow_pnl": 350,
                "reason": "early_exit",
            },
        ]
        ctx.shadow_pnl_delta = -150

        block = self.agent._build_context_block(ctx, CoachingMode.CONVERSATIONAL)
        assert "SHADOW TRADING" in block

    def test_regime_section_for_regime_mode(self):
        ctx = _make_context(current_regime="trending_bullish")
        ctx.regime_strategy_matrix = {
            "opening_breakout": {"win_rate": 0.72, "avg_r": 1.5, "count": 30},
        }
        block = self.agent._build_context_block(ctx, CoachingMode.REGIME_GUIDANCE)
        assert "MARKET REGIME" in block
        assert "trending_bullish" in block


# ─── Tests: Full Process Flow ────────────────────────────────────


class TestCoachingProcess:

    @pytest.mark.asyncio
    async def test_process_returns_coaching_response(self):
        llm = _make_llm_client(
            "Great session today! You showed good discipline.\n"
            "→ Keep following your checklist\n"
            "→ Review your TSLA trade entry\n"
            "What made you choose to enter TSLA at that level?"
        )
        agent = CoachingAgent(llm_client=llm)
        ctx = _make_context()

        response = await agent.process("How was my day?", ctx)

        assert isinstance(response, CoachingResponse)
        assert len(response.message) > 0
        assert response.mode == CoachingMode.SESSION_DEBRIEF

    @pytest.mark.asyncio
    async def test_action_items_extracted(self):
        llm = _make_llm_client(
            "Here's my analysis:\n"
            "→ Review your stop placement on losing trades\n"
            "→ Consider reducing position size in choppy markets\n"
            "Overall good process."
        )
        agent = CoachingAgent(llm_client=llm)
        ctx = _make_context()

        response = await agent.process("Any advice?", ctx)

        assert len(response.action_items) == 2
        assert "stop placement" in response.action_items[0].lower()

    @pytest.mark.asyncio
    async def test_follow_up_questions_extracted(self):
        llm = _make_llm_client(
            "I notice a pattern in your exits.\n"
            "What was your emotional state when you exited TSLA early?\n"
            "Have you noticed this happening on other pullback trades?"
        )
        agent = CoachingAgent(llm_client=llm)
        ctx = _make_context()

        response = await agent.process("Review my trading", ctx)

        assert len(response.follow_up_questions) >= 1

    @pytest.mark.asyncio
    async def test_mode_override_works(self):
        llm = _make_llm_client("Here's your weekly review...")
        agent = CoachingAgent(llm_client=llm)
        ctx = _make_context()

        response = await agent.process(
            "Hello",
            ctx,
            mode=CoachingMode.WEEKLY_REVIEW,
        )

        assert response.mode == CoachingMode.WEEKLY_REVIEW

    @pytest.mark.asyncio
    async def test_llm_called_with_correct_messages(self):
        llm = _make_llm_client("Response")
        agent = CoachingAgent(llm_client=llm)
        ctx = _make_context()

        await agent.process("Hello coach", ctx)

        llm.chat.assert_called_once()
        call_kwargs = llm.chat.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]

        # Should have system prompt, context, and user message
        assert len(messages) >= 3
        assert messages[0].role == "system"
        assert messages[-1].role == "user"
        assert messages[-1].content == "Hello coach"

    @pytest.mark.asyncio
    async def test_conversation_history_included(self):
        llm = _make_llm_client("Following up on your question...")
        agent = CoachingAgent(llm_client=llm)
        ctx = _make_context(conversation_history=[
            {"role": "user", "content": "How's my risk management?"},
            {"role": "assistant", "content": "Your risk management has been solid."},
        ])

        await agent.process("Can you elaborate?", ctx)

        call_kwargs = llm.chat.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]

        # Should include history + new message
        user_messages = [m for m in messages if m.role == "user"]
        assert len(user_messages) >= 2  # history + current


# ─── Tests: Proactive Insights ───────────────────────────────────


class TestProactiveInsights:

    def setup_method(self):
        self.agent = CoachingAgent(llm_client=_make_llm_client())

    def test_process_violation_streak_insight(self):
        ctx = _make_context(recent_dq_reports=[
            {"is_process_violation": True, "composite_score": 0.35, "grade": "F"},
            {"is_process_violation": True, "composite_score": 0.38, "grade": "F"},
            {"is_process_violation": True, "composite_score": 0.40, "grade": "D"},
            {"is_process_violation": False, "composite_score": 0.75, "grade": "B"},
            {"is_process_violation": True, "composite_score": 0.30, "grade": "F"},
        ])

        insights = self.agent._generate_proactive_insights(
            ctx, CoachingMode.CONVERSATIONAL
        )

        critical = [i for i in insights if i.priority == InsightPriority.CRITICAL]
        assert len(critical) >= 1
        assert "violation" in critical[0].title.lower()

    def test_declining_dq_trend_insight(self):
        ctx = _make_context(dq_trend={
            "average_score": 0.55,
            "trend": "declining",
            "quality_rate": 0.40,
        })

        insights = self.agent._generate_proactive_insights(
            ctx, CoachingMode.CONVERSATIONAL
        )

        high = [i for i in insights if i.priority == InsightPriority.HIGH]
        assert any("declining" in i.title.lower() for i in high)

    def test_multiple_biases_insight(self):
        ctx = _make_context(
            active_biases=["recency_bias", "loss_aversion", "confirmation_bias"]
        )

        insights = self.agent._generate_proactive_insights(
            ctx, CoachingMode.CONVERSATIONAL
        )

        high = [i for i in insights if i.priority == InsightPriority.HIGH]
        assert any("bias" in i.title.lower() for i in high)

    def test_overtrading_insight(self):
        ctx = _make_context(
            todays_trades=[
                {"id": f"t{i}", "symbol": "SPY", "pnl": 10}
                for i in range(8)  # 8 trades vs default max of 5
            ],
            trader_profile={
                "name": "TestTrader",
                "max_daily_trades": 5,
            },
        )

        insights = self.agent._generate_proactive_insights(
            ctx, CoachingMode.CONVERSATIONAL
        )

        assert any("overtrad" in i.title.lower() for i in insights)

    def test_shadow_divergence_insight(self):
        ctx = _make_context()
        ctx.shadow_pnl_delta = -800
        ctx.shadow_divergences = [
            {"reason": "early_exit", "symbol": "AAPL",
             "actual_pnl": 100, "shadow_pnl": 400},
        ]

        insights = self.agent._generate_proactive_insights(
            ctx, CoachingMode.CONVERSATIONAL
        )

        assert any("shadow" in i.title.lower() or "divergence" in i.title.lower()
                    for i in insights)

    def test_regime_alignment_insight(self):
        ctx = _make_context(current_regime="trending_bullish")
        ctx.regime_strategy_matrix = {
            "opening_breakout": {"win_rate": 0.75, "avg_r": 1.8, "count": 25},
            "range_fade": {"win_rate": 0.30, "avg_r": -0.5, "count": 10},
        }

        insights = self.agent._generate_proactive_insights(
            ctx, CoachingMode.CONVERSATIONAL
        )

        info = [i for i in insights if i.priority == InsightPriority.INFO]
        assert any("regime" in i.title.lower() or "alignment" in i.title.lower()
                    for i in info)

    def test_no_insights_when_everything_is_fine(self):
        ctx = _make_context(
            recent_dq_reports=[
                {"is_process_violation": False, "composite_score": 0.85,
                 "grade": "A"},
            ],
            dq_trend={"trend": "stable", "average_score": 0.82,
                       "quality_rate": 0.90},
            active_biases=[],
        )

        insights = self.agent._generate_proactive_insights(
            ctx, CoachingMode.CONVERSATIONAL
        )

        # Should have no critical or high priority insights
        critical_high = [
            i for i in insights
            if i.priority in {InsightPriority.CRITICAL, InsightPriority.HIGH}
        ]
        assert len(critical_high) == 0


# ─── Tests: Fallback Response ────────────────────────────────────


class TestFallback:

    def setup_method(self):
        self.agent = CoachingAgent(llm_client=_make_llm_client())

    def test_fallback_with_insights(self):
        ctx = _make_context(
            active_biases=["recency_bias", "loss_aversion"],
            dq_trend={"trend": "declining", "average_score": 0.50},
        )

        response = self.agent._fallback_response(
            "How am I doing?", ctx, CoachingMode.CONVERSATIONAL
        )

        assert len(response.message) > 0
        assert response.confidence < 0.5
        assert response.metadata.get("fallback") is True

    def test_fallback_without_insights(self):
        ctx = _make_context()

        response = self.agent._fallback_response(
            "Hello", ctx, CoachingMode.CONVERSATIONAL
        )

        assert "try again" in response.message.lower() or "trouble" in response.message.lower()

    @pytest.mark.asyncio
    async def test_process_falls_back_on_llm_error(self):
        llm = AsyncMock(spec=LLMClient)
        llm.chat = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        agent = CoachingAgent(llm_client=llm)
        ctx = _make_context(
            active_biases=["recency_bias", "loss_aversion"],
        )

        response = await agent.process("Hello", ctx)

        assert response.metadata.get("fallback") is True
        assert response.confidence < 0.5


# ─── Tests: Proactive Briefings ──────────────────────────────────


class TestProactiveBriefings:

    @pytest.mark.asyncio
    async def test_generate_proactive_briefing(self):
        llm = _make_llm_client("Good morning! Here's your pre-session prep...")
        agent = CoachingAgent(llm_client=llm)
        ctx = _make_context(todays_trades=[])  # No trades = pre-session

        response = await agent.generate_proactive_briefing(ctx)

        assert isinstance(response, CoachingResponse)
        assert len(response.message) > 0

    @pytest.mark.asyncio
    async def test_behavioral_alert_briefing(self):
        llm = _make_llm_client("I've noticed some concerning patterns...")
        agent = CoachingAgent(llm_client=llm)
        ctx = _make_context(
            active_biases=["revenge_trading", "loss_aversion"],
        )

        # Should detect behavioral alert as priority
        topic = agent._identify_priority_topic(ctx)
        assert topic == "behavioral_alert"

    @pytest.mark.asyncio
    async def test_session_debrief_when_trades_exist(self):
        llm = _make_llm_client("End of day review...")
        agent = CoachingAgent(llm_client=llm)
        ctx = _make_context(
            todays_trades=[
                {"id": "t1", "symbol": "AAPL", "pnl": 200},
            ],
            active_biases=[],
        )

        topic = agent._identify_priority_topic(ctx)
        assert topic == "session_debrief"

    @pytest.mark.asyncio
    async def test_pre_session_when_no_trades(self):
        llm = _make_llm_client("Pre-session prep...")
        agent = CoachingAgent(llm_client=llm)
        ctx = _make_context(
            todays_trades=[],
            active_biases=[],
        )

        topic = agent._identify_priority_topic(ctx)
        assert topic == "pre_session"


# ─── Tests: Temperature & Token Config ───────────────────────────


# ... continuing from TestConfiguration class ...

class TestConfiguration:

    def setup_method(self):
        self.agent = CoachingAgent(llm_client=_make_llm_client())

    def test_analytical_modes_have_lower_temperature(self):
        assert self.agent._get_temperature(CoachingMode.STRATEGY_ANALYSIS) < \
               self.agent._get_temperature(CoachingMode.CONVERSATIONAL)

    def test_weekly_review_has_more_tokens(self):
        assert self.agent._get_max_tokens(CoachingMode.WEEKLY_REVIEW) > \
               self.agent._get_max_tokens(CoachingMode.PRE_SESSION)

    def test_trade_review_moderate_temperature(self):
        temp = self.agent._get_temperature(CoachingMode.TRADE_REVIEW)
        assert 0.2 <= temp <= 0.5

    def test_behavioral_alert_moderate_tokens(self):
        tokens = self.agent._get_max_tokens(CoachingMode.BEHAVIORAL_ALERT)
        assert 400 <= tokens <= 700

    def test_all_modes_have_temperature(self):
        for mode in CoachingMode:
            temp = self.agent._get_temperature(mode)
            assert 0.0 <= temp <= 1.0

    def test_all_modes_have_max_tokens(self):
        for mode in CoachingMode:
            tokens = self.agent._get_max_tokens(mode)
            assert tokens >= 400


# ─── Tests: Message Assembly ─────────────────────────────────────


class TestMessageAssembly:

    def setup_method(self):
        self.agent = CoachingAgent(llm_client=_make_llm_client())

    def test_message_structure(self):
        messages = self.agent._assemble_messages(
            system_prompt="You are a trading coach.",
            context_block="## DATA\nSome data here.",
            user_message="Hello",
            conversation_history=[],
        )

        assert messages[0].role == "system"
        assert "trading coach" in messages[0].content
        assert messages[1].role == "system"
        assert "DATA" in messages[1].content
        assert messages[-1].role == "user"
        assert messages[-1].content == "Hello"

    def test_history_included_in_messages(self):
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]

        messages = self.agent._assemble_messages(
            system_prompt="System",
            context_block="Context",
            user_message="Third question",
            conversation_history=history,
        )

        user_msgs = [m for m in messages if m.role == "user"]
        assistant_msgs = [m for m in messages if m.role == "assistant"]

        assert len(user_msgs) == 3  # 2 from history + 1 current
        assert len(assistant_msgs) == 2

    def test_history_truncated_to_max(self):
        agent = CoachingAgent(
            llm_client=_make_llm_client(),
            max_conversation_history=4,
        )

        history = [
            {"role": "user", "content": f"Message {i}"}
            if i % 2 == 0 else
            {"role": "assistant", "content": f"Response {i}"}
            for i in range(20)
        ]

        messages = agent._assemble_messages(
            system_prompt="System",
            context_block="Context",
            user_message="Current",
            conversation_history=history,
        )

        # 2 system + 4 history + 1 current = 7
        assert len(messages) == 7

    def test_invalid_roles_filtered(self):
        history = [
            {"role": "user", "content": "Valid"},
            {"role": "system", "content": "Should be skipped"},
            {"role": "assistant", "content": "Also valid"},
        ]

        messages = self.agent._assemble_messages(
            system_prompt="System",
            context_block="Context",
            user_message="Current",
            conversation_history=history,
        )

        # system messages from history should be excluded
        non_system_history = [
            m for m in messages
            if m.role != "system"
        ]
        # user from history + assistant from history + current user
        assert len(non_system_history) == 3


# ─── Tests: Response Parsing ────────────────────────────────────


class TestResponseParsing:

    def setup_method(self):
        self.agent = CoachingAgent(llm_client=_make_llm_client())

    def test_action_items_parsed_from_arrow(self):
        llm_response = LLMResponse(
            content=(
                "Here's what I noticed:\n"
                "→ Tighten your stop losses on momentum trades\n"
                "→ Review your entry timing on VWAP bounces\n"
                "→ Consider journaling before each session\n"
                "Overall solid day."
            ),
            model="test",
            tokens_used=100,
        )
        ctx = _make_context()
        result = self.agent._parse_response(
            llm_response, ctx, CoachingMode.CONVERSATIONAL
        )
        assert len(result.action_items) == 3
        assert "stop losses" in result.action_items[0].lower()

    def test_action_items_parsed_from_dash_arrow(self):
        llm_response = LLMResponse(
            content=(
                "Suggestions:\n"
                "-> Reduce position size\n"
                "-> Take a break after 3 losses\n"
            ),
            model="test",
            tokens_used=50,
        )
        ctx = _make_context()
        result = self.agent._parse_response(
            llm_response, ctx, CoachingMode.CONVERSATIONAL
        )
        assert len(result.action_items) == 2

    def test_follow_up_questions_extracted(self):
        llm_response = LLMResponse(
            content=(
                "I see an interesting pattern.\n"
                "What was going through your mind when you moved your stop?\n"
                "Have you noticed that your win rate drops after 3pm?\n"
                "Short question?"
            ),
            model="test",
            tokens_used=80,
        )
        ctx = _make_context()
        result = self.agent._parse_response(
            llm_response, ctx, CoachingMode.CONVERSATIONAL
        )
        # "Short question?" should be excluded (< 15 chars)
        assert len(result.follow_up_questions) == 2

    def test_follow_ups_limited_to_three(self):
        questions = "\n".join([
            f"Have you considered approach number {i} for your trading plan?"
            for i in range(10)
        ])
        llm_response = LLMResponse(
            content=questions,
            model="test",
            tokens_used=200,
        )
        ctx = _make_context()
        result = self.agent._parse_response(
            llm_response, ctx, CoachingMode.CONVERSATIONAL
        )
        assert len(result.follow_up_questions) <= 3

    def test_focus_area_extracted(self):
        llm_response = LLMResponse(
            content="Your risk management needs significant improvement. Stop losses are inconsistent.",
            model="test",
            tokens_used=30,
        )
        ctx = _make_context()
        result = self.agent._parse_response(
            llm_response, ctx, CoachingMode.CONVERSATIONAL
        )
        assert result.suggested_focus == "risk_management"

    def test_focus_area_behavioral(self):
        llm_response = LLMResponse(
            content="I'm seeing emotional and bias patterns in your trading behavior.",
            model="test",
            tokens_used=30,
        )
        ctx = _make_context()
        result = self.agent._parse_response(
            llm_response, ctx, CoachingMode.CONVERSATIONAL
        )
        assert result.suggested_focus == "behavioral"


# ─── Tests: Coaching Insight Dataclass ───────────────────────────


class TestCoachingInsight:

    def test_insight_to_dict(self):
        insight = CoachingInsight(
            priority=InsightPriority.HIGH,
            category="behavioral",
            title="Tilt Detected",
            content="You appear to be revenge trading.",
            evidence=["3 rapid-fire trades after a loss"],
            action_items=["Take a 15-minute break"],
            related_trades=["t1", "t2", "t3"],
        )

        d = insight.to_dict()
        assert d["priority"] == "high"
        assert d["category"] == "behavioral"
        assert d["title"] == "Tilt Detected"
        assert len(d["evidence"]) == 1
        assert len(d["action_items"]) == 1
        assert "timestamp" in d

    def test_insight_default_values(self):
        insight = CoachingInsight()
        assert insight.priority == InsightPriority.MEDIUM
        assert insight.category == ""
        assert len(insight.id) > 0


# ─── Tests: Coaching Context ────────────────────────────────────


class TestCoachingContext:

    def test_summary_stats(self):
        ctx = _make_context()
        stats = ctx.summary_stats()

        assert stats["recent_trade_count"] == 2
        assert stats["recent_win_rate"] == 0.5
        assert stats["today_trade_count"] == 1
        assert stats["today_pnl"] == 180
        assert stats["discipline_score"] == 0.78

    def test_summary_stats_empty(self):
        ctx = CoachingContext()
        stats = ctx.summary_stats()

        assert stats["recent_trade_count"] == 0
        assert stats["recent_win_rate"] == 0
        assert stats["today_trade_count"] == 0
        assert stats["today_pnl"] == 0

    def test_summary_stats_regime(self):
        ctx = _make_context(current_regime="ranging")
        stats = ctx.summary_stats()
        assert stats["current_regime"] == "ranging"