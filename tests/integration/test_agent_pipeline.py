# tests/integration/test_agent_pipeline.py
"""Tests for agents/coaching_agent.py — mode detection and pipeline logic."""

import pytest
from unittest.mock import MagicMock

from agents.coaching_agent import CoachingAgent, CoachingContext, CoachingMode


class TestCoachingMode:
    def test_all_modes_exist(self):
        expected = [
            "conversational", "trade_review", "session_debrief", "weekly_review",
            "pre_session", "behavioral_alert", "strategy_analysis", "pattern_insight",
            "regime_guidance", "goal_setting", "competition_review",
        ]
        values = [m.value for m in CoachingMode]
        for v in expected:
            assert v in values, f"Missing CoachingMode: {v}"

    def test_competition_review_mode(self):
        assert hasattr(CoachingMode, "COMPETITION_REVIEW")
        assert CoachingMode.COMPETITION_REVIEW.value == "competition_review"

    def test_mode_is_str(self):
        assert isinstance(CoachingMode.CONVERSATIONAL, str)
        assert CoachingMode.TRADE_REVIEW == "trade_review"


@pytest.fixture
def agent():
    # the LLM client is injected; mode detection never calls it
    return CoachingAgent(llm_client=MagicMock())


class TestCoachingAgentInit:
    def test_instantiation(self, agent):
        assert agent.name == "coaching_agent"

    @pytest.mark.parametrize("message, mode", [
        ("Can you review this trade for me?", CoachingMode.TRADE_REVIEW),
        ("How did I do today?", CoachingMode.SESSION_DEBRIEF),
        ("Weekly review please", CoachingMode.WEEKLY_REVIEW),
        ("Morning prep - what should I focus on?", CoachingMode.PRE_SESSION),
        ("Which strategy is my best strategy?", CoachingMode.STRATEGY_ANALYSIS),
        ("What kind of market is this?", CoachingMode.REGIME_GUIDANCE),
        ("Am I tilting? I think I'm revenge trading", CoachingMode.BEHAVIORAL_ALERT),
        ("Where am I lagging vs the system?", CoachingMode.COMPETITION_REVIEW),
        ("Any patterns in how I trade?", CoachingMode.PATTERN_INSIGHT),
        ("Hello, how are you?", CoachingMode.CONVERSATIONAL),
    ])
    def test_detect_mode(self, agent, message, mode):
        assert agent._detect_mode(message, CoachingContext()) == mode

    def test_trade_under_review_wins(self, agent):
        ctx = CoachingContext(trade_under_review={"id": "t1"})
        assert agent._detect_mode("Hello", ctx) == CoachingMode.TRADE_REVIEW
