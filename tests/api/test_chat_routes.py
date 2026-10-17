# test_chat_routes.py
"""Tests for chat API routes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient

from agents.coaching_agent import CoachingResponse, CoachingMode, CoachingInsight, InsightPriority
from agents.llm_client import LLMResponse


@pytest.fixture
def mock_coaching_response():
    """Create a mock coaching response."""
    return CoachingResponse(
        message="Great session! Your risk management improved significantly.",
        insights=[
            CoachingInsight(
                priority=InsightPriority.MEDIUM,
                category="risk",
                title="Improving Risk Management",
                content="Your stop-loss adherence went from 60% to 85%.",
                action_items=["Keep using bracket orders"],
            ),
        ],
        action_items=["Review your TSLA exit", "Journal about the morning session"],
        follow_up_questions=["What was your emotional state during the TSLA trade?"],
        suggested_focus="risk_management",
        mode=CoachingMode.SESSION_DEBRIEF,
        confidence=0.85,
    )


class TestChatMessageEndpoint:

    @pytest.mark.asyncio
    async def test_send_message_success(
        self, async_client: AsyncClient, mock_coaching_response
    ):
        """Test sending a chat message."""
        with patch(
            "api.routes.chat.CoachingAgent"
        ) as MockAgent, patch(
            "api.routes.chat.ConversationManager"
        ) as MockConvMgr:
            # Setup mocks
            agent_instance = AsyncMock()
            agent_instance.process = AsyncMock(
                return_value=mock_coaching_response
            )
            MockAgent.return_value = agent_instance


            conv_instance = AsyncMock()
            conv_instance.get_or_create_session = AsyncMock(
                return_value=MagicMock(session_id="sess-123")
            )
            conv_instance.get_conversation_history = AsyncMock(return_value=[])
            conv_instance.add_user_message = AsyncMock()
            conv_instance.add_assistant_message = AsyncMock()
            MockConvMgr.return_value = conv_instance

            response = await async_client.post(
                "/api/v1/chat/message",
                json={
                    "message": "How was my trading day?",
                    "personality": "balanced",
                },
                headers={"X-Trader-ID": "test-trader"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "session_id" in data
        assert data["session_id"] == "sess-123"
        assert data["mode"] == "session_debrief"
        assert len(data["action_items"]) >= 1
        assert len(data["insights"]) >= 1

    @pytest.mark.asyncio
    async def test_send_message_empty_rejected(
        self, async_client: AsyncClient
    ):
        """Test that empty messages are rejected."""
        response = await async_client.post(
            "/api/v1/chat/message",
            json={"message": ""},
            headers={"X-Trader-ID": "test-trader"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_send_message_with_trade_id(
        self, async_client: AsyncClient, mock_coaching_response
    ):
        """Test sending a message with a specific trade reference."""
        mock_coaching_response.mode = CoachingMode.TRADE_REVIEW

        with patch(
            "api.routes.chat.CoachingAgent"
        ) as MockAgent, patch(
            "api.routes.chat.ConversationManager"
        ) as MockConvMgr:
            agent_instance = AsyncMock()
            agent_instance.process = AsyncMock(
                return_value=mock_coaching_response
            )
            MockAgent.return_value = agent_instance


            conv_instance = AsyncMock()
            conv_instance.get_or_create_session = AsyncMock(
                return_value=MagicMock(session_id="sess-456")
            )
            conv_instance.get_conversation_history = AsyncMock(return_value=[])
            conv_instance.add_user_message = AsyncMock()
            conv_instance.add_assistant_message = AsyncMock()
            MockConvMgr.return_value = conv_instance

            response = await async_client.post(
                "/api/v1/chat/message",
                json={
                    "message": "Review this trade",
                    "trade_id": "trade-abc-123",
                    "personality": "analytical",
                },
                headers={"X-Trader-ID": "test-trader"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "trade_review"

    @pytest.mark.asyncio
    async def test_send_message_with_mode_override(
        self, async_client: AsyncClient, mock_coaching_response
    ):
        """Test sending a message with explicit mode override."""
        mock_coaching_response.mode = CoachingMode.WEEKLY_REVIEW

        with patch(
            "api.routes.chat.CoachingAgent"
        ) as MockAgent, patch(
            "api.routes.chat.ConversationManager"
        ) as MockConvMgr:
            agent_instance = AsyncMock()
            agent_instance.process = AsyncMock(
                return_value=mock_coaching_response
            )
            MockAgent.return_value = agent_instance


            conv_instance = AsyncMock()
            conv_instance.get_or_create_session = AsyncMock(
                return_value=MagicMock(session_id="sess-789")
            )
            conv_instance.get_conversation_history = AsyncMock(return_value=[])
            conv_instance.add_user_message = AsyncMock()
            conv_instance.add_assistant_message = AsyncMock()
            MockConvMgr.return_value = conv_instance

            response = await async_client.post(
                "/api/v1/chat/message",
                json={
                    "message": "Give me a report",
                    "mode": "weekly_review",
                },
                headers={"X-Trader-ID": "test-trader"},
            )

        assert response.status_code == 200


class TestTradeReviewEndpoint:

    @pytest.mark.asyncio
    async def test_trade_review_success(
        self, async_client: AsyncClient, mock_coaching_response
    ):
        """Test trade review endpoint."""
        mock_coaching_response.mode = CoachingMode.TRADE_REVIEW

        with patch(
            "api.routes.chat.CoachingAgent"
        ) as MockAgent, patch(
            "api.routes.chat.ConversationManager"
        ) as MockConvMgr:
            agent_instance = AsyncMock()
            agent_instance.process = AsyncMock(
                return_value=mock_coaching_response
            )
            MockAgent.return_value = agent_instance

            conv_instance = AsyncMock()
            conv_instance.get_or_create_session = AsyncMock(
                return_value=MagicMock(session_id="sess-review")
            )
            conv_instance.add_user_message = AsyncMock()
            conv_instance.add_assistant_message = AsyncMock()
            MockConvMgr.return_value = conv_instance

            response = await async_client.post(
                "/api/v1/chat/trade-review",
                json={
                    "trade_id": "trade-abc",
                    "focus_areas": ["risk", "exit"],
                },
                headers={"X-Trader-ID": "test-trader"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["trade_id"] == "trade-abc"
        assert "message" in data
        assert "trade_summary" in data
        assert "decision_quality" in data


class TestBriefingEndpoint:

    @pytest.mark.asyncio
    async def test_proactive_briefing(
        self, async_client: AsyncClient, mock_coaching_response
    ):
        """Test proactive briefing endpoint."""
        with patch(
            "api.routes.chat.CoachingAgent"
        ) as MockAgent:
            agent_instance = AsyncMock()
            agent_instance.generate_proactive_briefing = AsyncMock(
                return_value=mock_coaching_response
            )
            MockAgent.return_value = agent_instance


            response = await async_client.post(
                "/api/v1/chat/briefing",
                json={"personality": "direct"},
                headers={"X-Trader-ID": "test-trader"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "insights" in data


class TestSessionEndpoints:

    @pytest.mark.asyncio
    async def test_list_sessions(self, async_client: AsyncClient):
        """Test listing conversation sessions."""
        with patch(
            "api.routes.chat.ConversationManager"
        ) as MockConvMgr:
            conv_instance = AsyncMock()
            conv_instance.get_recent_sessions = AsyncMock(return_value=[
                {
                    "session_id": "sess-1",
                    "mode": "session_debrief",
                    "preview": "How was my day?",
                    "turn_count": 6,
                    "created_at": "2024-01-15T16:00:00",
                    "updated_at": "2024-01-15T16:30:00",
                },
                {
                    "session_id": "sess-2",
                    "mode": "trade_review",
                    "preview": "Review my AAPL trade",
                    "turn_count": 4,
                    "created_at": "2024-01-15T14:00:00",
                    "updated_at": "2024-01-15T14:20:00",
                },
            ])
            MockConvMgr.return_value = conv_instance

            response = await async_client.get(
                "/api/v1/chat/sessions?limit=10",
                headers={"X-Trader-ID": "test-trader"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 2
        assert data["sessions"][0]["session_id"] == "sess-1"

    @pytest.mark.asyncio
    async def test_get_session_history(self, async_client: AsyncClient):
        """Test retrieving session history."""
        with patch(
            "api.routes.chat.ConversationManager"
        ) as MockConvMgr:
            conv_instance = AsyncMock()
            conv_instance.get_conversation_history = AsyncMock(return_value=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "Review my day"},
                {"role": "assistant", "content": "Here's your session review..."},
            ])
            MockConvMgr.return_value = conv_instance

            response = await async_client.get(
                "/api/v1/chat/sessions/sess-1/history",
                headers={"X-Trader-ID": "test-trader"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess-1"
        assert data["total_turns"] == 4
        assert len(data["messages"]) == 4

    @pytest.mark.asyncio
    async def test_end_session(self, async_client: AsyncClient):
        """Test ending a session."""
        with patch(
            "api.routes.chat.ConversationManager"
        ) as MockConvMgr:
            conv_instance = AsyncMock()
            conv_instance.end_session = AsyncMock()
            MockConvMgr.return_value = conv_instance

            response = await async_client.post(
                "/api/v1/chat/sessions/sess-1/end",
                headers={"X-Trader-ID": "test-trader"},
            )

        assert response.status_code == 200
        assert response.json()["status"] == "ok"