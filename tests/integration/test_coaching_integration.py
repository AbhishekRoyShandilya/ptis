"""
Integration tests for the AI Coaching Agent pipeline.

Tests the full flow: context assembly → coaching agent → response parsing.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agents.coaching_agent import (
    CoachingAgent,
    CoachingContext,
    CoachingMode,
    CoachingPersonality,
    CoachingResponse,
)
from agents.context_assembler import ContextAssembler
from agents.conversation_manager import ConversationManager
from agents.llm_client import LLMClient, LLMResponse


# ─── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    """Create a mock async database session."""
    db = AsyncMock()
    # Mock execute to return empty results by default
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_result.scalars.return_value.all.return_value = []
    mock_result.scalar_one.return_value = 0
    db.execute = AsyncMock(return_value=mock_result)
    db.commit = AsyncMock()
    db.add = MagicMock()
    return db


@pytest.fixture
def mock_llm():
    """Create a mock LLM client with realistic responses."""
    client = AsyncMock(spec=LLMClient)

    async def smart_response(messages, temperature=0.5, max_tokens=700, **kwargs):
        # Generate contextually aware mock responses
        user_msg = ""
        for m in reversed(messages):
            if m.role == "user":
                user_msg = m.content.lower()
                break

        if "how was my day" in user_msg or "session" in user_msg:
            text = (
                "Let me review your session.\n\n"
                "You took 3 trades today with a net positive P&L. "
                "Your best trade was the NVDA long — clean entry, "
                "good risk management, and you followed your exit plan.\n\n"
                "The TSLA trade was the one that needs attention. "
                "The entry was solid, but you moved your stop wider "
                "which violates your risk rules.\n\n"
                "→ Review your stop-loss discipline on momentum trades\n"
                "→ Journal about what triggered the stop move\n\n"
                "What was going through your mind when you moved the stop on TSLA?"
            )
        elif "review" in user_msg and "trade" in user_msg:
            text = (
                "Looking at this trade in detail:\n\n"
                "**Setup Quality: B+**\n"
                "The setup matched your opening breakout template well. "
                "8 of 10 conditions were met.\n\n"
                "**Entry Precision: A**\n"
                "Excellent entry within 0.1% of the signal price.\n\n"
                "**Risk Management: C**\n"
                "This is where it breaks down. Your stop was too wide "
                "for the setup, giving you only a 1.5:1 R:R instead of "
                "your target 2.5:1.\n\n"
                "→ Recalculate your standard stop distance for breakouts\n"
                "→ Consider using ATR-based stops\n\n"
                "Why did you set the stop wider than your template suggests?"
            )
        elif "bias" in user_msg or "tilt" in user_msg or "emotional" in user_msg:
            text = (
                "I want to be honest with you — the data suggests "
                "some emotional interference today.\n\n"
                "After your TSLA loss, you immediately entered two more "
                "trades within 5 minutes. This is a classic revenge "
                "trading pattern.\n\n"
                "Your discipline score dropped from 0.85 to 0.62 in "
                "the second half of the session.\n\n"
                "→ Implement a mandatory 10-minute cool-down after any loss\n"
                "→ Set a hard rule: no new trades within 15 minutes of a loss\n\n"
                "Do you recall what you were feeling right after the TSLA stop hit?"
            )
        else:
            text = (
                "Happy to help. Based on your recent data, "
                "your trading has been solid overall.\n\n"
                "Your decision quality trend is stable at a B average. "
                "Your best-performing strategy right now is the opening "
                "breakout in this trending regime.\n\n"
                "→ Keep focusing on your top 2 strategies\n\n"
                "Is there anything specific you'd like to dig into?"
            )

        return LLMResponse(
            content=text,
            model="gpt-4-test",
            tokens_used=200,
            prompt_tokens=150,
            completion_tokens=50,
        )

    client.chat = AsyncMock(side_effect=smart_response)
    return client


# ─── Integration Tests ───────────────────────────────────────────


class TestCoachingPipeline:
    """Test the full coaching pipeline from message to response."""

    @pytest.mark.asyncio
    async def test_session_debrief_flow(self, mock_llm):
        """Test a complete session debrief interaction."""
        agent = CoachingAgent(
            llm_client=mock_llm,
            personality=CoachingPersonality.BALANCED,
        )

        context = CoachingContext(
            trader_profile={
                "name": "Alex",
                "experience_level": "intermediate",
                "account_size": 50000,
                "max_daily_trades": 5,
            },
            todays_trades=[
                {"id": "t1", "symbol": "NVDA", "pnl": 280, "r_multiple": 1.8,
                 "outcome": "win", "strategy": "opening_breakout", "dq_grade": "A"},
                {"id": "t2", "symbol": "TSLA", "pnl": -150, "r_multiple": -1.0,
                 "outcome": "loss", "strategy": "pullback_to_ema", "dq_grade": "C"},
                {"id": "t3", "symbol": "AAPL", "pnl": 95, "r_multiple": 0.6,
                 "outcome": "win", "strategy": "vwap_bounce", "dq_grade": "B"},
            ],
            recent_trades=[],
            discipline_score=0.75,
            dq_trend={"average_score": 0.72, "trend": "stable", "quality_rate": 0.65},
            current_regime="trending_bullish",
        )

        response = await agent.process(
            "How was my day?", context
        )

        assert isinstance(response, CoachingResponse)
        assert response.mode == CoachingMode.SESSION_DEBRIEF
        assert len(response.message) > 100
        assert len(response.action_items) >= 1
        assert len(response.follow_up_questions) >= 1

    @pytest.mark.asyncio
    async def test_trade_review_flow(self, mock_llm):
        """Test a complete trade review interaction."""
        agent = CoachingAgent(
            llm_client=mock_llm,
            personality=CoachingPersonality.ANALYTICAL,
        )

        context = CoachingContext(
            trader_profile={"name": "Alex"},
            trade_under_review={
                "symbol": "AAPL",
                "direction": "long",
                "entry_price": 185.50,
                "exit_price": 187.20,
                "pnl": 170,
                "r_multiple": 1.1,
                "strategy": "opening_breakout",
                "emotional_state": "focused",
                "exit_reason": "target_hit",
                "entry_time": "2024-01-15T09:35:00",
                "exit_time": "2024-01-15T10:15:00",
            },
            trade_dq_report={
                "composite_score": 0.78,
                "grade": "B",
                "dimensions": {
                    "setup_quality": {"score": 0.85, "factors": ["Good match"], "deductions": []},
                    "entry_precision": {"score": 0.90, "factors": ["Tight entry"], "deductions": []},
                    "risk_management": {"score": 0.55, "factors": [], "deductions": ["Stop too wide"]},
                    "exit_execution": {"score": 0.80, "factors": ["Clean exit"], "deductions": []},
                    "process_adherence": {"score": 0.75, "factors": ["Checklist done"], "deductions": []},
                    "behavioral_score": {"score": 0.82, "factors": ["Calm state"], "deductions": []},
                },
                "coaching_notes": ["Risk management needs attention."],
            },
        )

        response = await agent.process(
            "Review this trade for me", context
        )

        assert response.mode == CoachingMode.TRADE_REVIEW
        assert len(response.message) > 100

    @pytest.mark.asyncio
    async def test_behavioral_alert_flow(self, mock_llm):
        """Test behavioral alert coaching."""
        agent = CoachingAgent(
            llm_client=mock_llm,
            personality=CoachingPersonality.SUPPORTIVE,
        )

        context = CoachingContext(
            trader_profile={"name": "Alex"},
            active_biases=["revenge_trading", "loss_aversion"],
            behavioral_events=[
                {"event_type": "cognitive_bias", "description": "revenge_trading",
                 "severity": "high", "timestamp": "2024-01-15T14:30:00"},
                {"event_type": "deviation", "description": "rapid_fire_entries",
                 "severity": "medium", "timestamp": "2024-01-15T14:32:00"},
            ],
            discipline_score=0.52,
            emotional_state="frustrated",
        )

        response = await agent.process(
            "Am I tilting right now?", context
        )

        assert response.mode == CoachingMode.BEHAVIORAL_ALERT
        assert len(response.message) > 50

        # Should generate proactive insights about biases
        high_priority = [
            i for i in response.insights
            if i.priority.value in {"critical", "high"}
        ]
        assert len(high_priority) >= 1

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, mock_llm):
        """Test multi-turn conversation maintains context."""
        agent = CoachingAgent(llm_client=mock_llm)

        context = CoachingContext(
            trader_profile={"name": "Alex"},
            conversation_history=[
                {"role": "user", "content": "How was my day?"},
                {"role": "assistant", "content": "You had a mixed day with 2 wins and 1 loss."},
                {"role": "user", "content": "Tell me more about the loss."},
                {"role": "assistant", "content": "The TSLA trade had issues with stop placement."},
            ],
            todays_trades=[
                {"id": "t1", "symbol": "NVDA", "pnl": 280, "outcome": "win"},
                {"id": "t2", "symbol": "TSLA", "pnl": -150, "outcome": "loss"},
            ],
        )

        response = await agent.process(
            "What should I focus on tomorrow based on today?", context
        )

        # Verify the LLM received the full history
        call_args = mock_llm.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]

        user_messages = [m for m in messages if m.role == "user"]
        # Should have 3 user messages from history + 1 current
        assert len(user_messages) >= 3

    @pytest.mark.asyncio
    async def test_proactive_briefing_pre_session(self, mock_llm):
        """Test proactive pre-session briefing."""
        agent = CoachingAgent(llm_client=mock_llm)

        context = CoachingContext(
            trader_profile={"name": "Alex", "max_daily_trades": 5},
            todays_trades=[],  # No trades yet = pre-session
            current_regime="trending_bullish",
            regime_strategy_matrix={
                "opening_breakout": {"win_rate": 0.75, "avg_r": 1.8, "count": 25},
                "range_fade": {"win_rate": 0.30, "avg_r": -0.5, "count": 10},
            },
            discipline_score=0.85,
            dq_trend={"average_score": 0.78, "trend": "improving", "quality_rate": 0.70},
        )

        response = await agent.generate_proactive_briefing(context)

        assert isinstance(response, CoachingResponse)
        assert len(response.message) > 50

    @pytest.mark.asyncio
    async def test_different_personalities_produce_different_prompts(self, mock_llm):
        """Verify personality setting affects the system prompt."""
        captured_messages = {"analytical": None, "socratic": None}

        async def capture_messages(messages, **kwargs):
            return LLMResponse(content="Response", model="test", tokens_used=10)

        mock_llm.chat = AsyncMock(side_effect=capture_messages)

        # personality is chosen per request, on the context (the chat API sets it from the request)
        agent = CoachingAgent(llm_client=mock_llm)

        # Analytical
        await agent.process("Hello", CoachingContext(trader_profile={"name": "Alex"},
                                                     coaching_personality=CoachingPersonality.ANALYTICAL))
        call1 = mock_llm.chat.call_args_list[-1]
        msgs1 = call1.kwargs.get("messages") or call1.args[0]
        system1 = msgs1[0].content

        # Socratic
        await agent.process("Hello", CoachingContext(trader_profile={"name": "Alex"},
                                                     coaching_personality=CoachingPersonality.SOCRATIC))
        call2 = mock_llm.chat.call_args_list[-1]
        msgs2 = call2.kwargs.get("messages") or call2.args[0]
        system2 = msgs2[0].content

        # System prompts should differ due to personality
        assert system1 != system2
        assert "precise" in system1.lower() or "data" in system1.lower()
        assert "question" in system2.lower()


class TestContextAssemblerIntegration:
    """The assembler fetches each section independently: a failing source never aborts assembly."""

    @staticmethod
    def _session_factory(execute):
        session = AsyncMock()
        session.execute = execute
        factory = MagicMock()
        factory.return_value.__aenter__ = AsyncMock(return_value=session)
        factory.return_value.__aexit__ = AsyncMock(return_value=False)
        return factory

    @pytest.mark.asyncio
    async def test_missing_trade_gives_empty_context(self):
        result = MagicMock()
        result.fetchone.return_value = None
        result.fetchall.return_value = []
        assembler = ContextAssembler(db_session_factory=self._session_factory(AsyncMock(return_value=result)))
        trade_id = uuid4()
        ctx = await assembler.assemble(trade_id)
        assert ctx.trade_id == trade_id
        assert ctx.symbol is None

    @pytest.mark.asyncio
    async def test_assembly_survives_database_errors(self):
        broken = self._session_factory(AsyncMock(side_effect=RuntimeError("db down")))
        trade_id = uuid4()
        ctx = await ContextAssembler(db_session_factory=broken).assemble(trade_id)
        assert ctx.trade_id == trade_id
        assert ctx.symbol is None
