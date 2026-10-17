"""
Conversation Manager — Handles conversation persistence, session management,
and conversation-level operations for the coaching agent.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import select, desc, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.models import ConversationSession, ConversationMessage

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """A conversation session with history."""
    session_id: str
    trader_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    mode: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def history(self) -> list[dict[str, str]]:
        """Get conversation history in LLM-compatible format."""
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self.turns
        ]

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def last_user_message(self) -> Optional[str]:
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return None


class ConversationManager:
    """
    Manages coaching conversations.

    Responsibilities:
    - Create and resume conversation sessions
    - Persist conversation turns
    - Manage conversation context window
    - Provide conversation summaries for long sessions
    """

    def __init__(
        self,
        db: AsyncSession,
        max_turns_in_context: int = 20,
        session_timeout_hours: int = 4,
    ):
        self.db = db
        self.max_turns_in_context = max_turns_in_context
        self.session_timeout_hours = session_timeout_hours

    async def get_or_create_session(
        self,
        trader_id: str,
        session_id: Optional[str] = None,
    ) -> Conversation:
        """
        Get an existing session or create a new one.

        If no session_id is provided, tries to resume the most recent
        active session. Creates new if none found or timed out.
        """
        if session_id:
            conversation = await self._load_session(session_id)
            if conversation:
                return conversation

        # Try to resume the most recent active session
        conversation = await self._find_active_session(trader_id)
        if conversation:
            return conversation

        # Create new session
        return await self._create_session(trader_id)

    async def add_user_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Record a user message in the conversation."""
        turn = ConversationTurn(
            role="user",
            content=content,
            metadata=metadata or {},
        )
        await self._persist_turn(session_id, turn)
        return turn

    async def add_assistant_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Record an assistant message in the conversation."""
        turn = ConversationTurn(
            role="assistant",
            content=content,
            metadata=metadata or {},
        )
        await self._persist_turn(session_id, turn)
        return turn

    async def get_conversation_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None,
    ) -> list[dict[str, str]]:
        """
        Get conversation history for context injection.

        Returns the most recent turns up to max_turns.
        """
        limit = max_turns or self.max_turns_in_context
        session_uuid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
        stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_uuid)
            .order_by(desc(ConversationMessage.created_at))
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        messages = result.scalars().all()

        # Reverse to get chronological order
        messages = list(reversed(messages))

        return [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

    async def get_recent_sessions(
        self,
        trader_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent conversation sessions for the coaching history view."""
        stmt = (
            select(ConversationSession)
            .where(ConversationSession.trader_id == trader_id)
            .order_by(desc(ConversationSession.updated_at))
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        sessions = result.scalars().all()

        session_list = []
        for s in sessions:
            # Get first user message as preview
            preview_stmt = (
                select(ConversationMessage.content)
                .where(
                    and_(
                        ConversationMessage.session_id == s.id,
                        ConversationMessage.role == "user",
                    )
                )
                .order_by(ConversationMessage.created_at)
                .limit(1)
            )
            preview_result = await self.db.execute(preview_stmt)
            preview = preview_result.scalar_one_or_none()

            # Get turn count
            count_stmt = (
                select(func.count())
                .select_from(ConversationMessage)
                .where(ConversationMessage.session_id == s.id)
            )
            count_result = await self.db.execute(count_stmt)
            turn_count = count_result.scalar_one()

            session_list.append({
                "session_id": str(s.id),
                "mode": s.mode,
                "preview": preview[:100] + "..." if preview and len(preview) > 100 else preview,
                "turn_count": turn_count,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            })

        return session_list

    async def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        session_uuid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
        stmt = (
            select(ConversationSession)
            .where(ConversationSession.id == session_uuid)
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()
        if session:
            session.is_active = False
            session.ended_at = datetime.utcnow()
            await self.db.commit()

    # ─── Internal Methods ────────────────────────────────────────

    async def _load_session(self, session_id: str) -> Optional[Conversation]:
        """Load a specific session by ID."""
        session_uuid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
        stmt = select(ConversationSession).where(
            ConversationSession.id == session_uuid
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            return None

        history = await self.get_conversation_history(session_id)
        turns = [
            ConversationTurn(role=h["role"], content=h["content"])
            for h in history
        ]

        return Conversation(
            session_id=str(session.id),
            trader_id=session.trader_id,
            turns=turns,
            mode=session.mode,
            created_at=session.created_at,
            updated_at=session.updated_at,
            metadata=session.metadata or {},
        )

    async def _find_active_session(
        self, trader_id: str
    ) -> Optional[Conversation]:
        """Find the most recent active session that hasn't timed out."""
        timeout = datetime.utcnow() - timedelta(hours=self.session_timeout_hours)
        stmt = (
            select(ConversationSession)
            .where(
                and_(
                    ConversationSession.trader_id == trader_id,
                    ConversationSession.is_active == True,
                    ConversationSession.updated_at >= timeout,
                )
            )
            .order_by(desc(ConversationSession.updated_at))
            .limit(1)
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            return None

        return await self._load_session(str(session.id))

    async def _create_session(self, trader_id: str) -> Conversation:
        """Create a new conversation session."""
        session_uuid = uuid.uuid4()
        now = datetime.utcnow()

        session = ConversationSession(
            id=session_uuid,
            trader_id=trader_id,
            is_active=True,
            created_at=now,
            updated_at=now,
            extra_data={},
        )
        self.db.add(session)
        await self.db.commit()

        return Conversation(
            session_id=str(session_uuid),
            trader_id=trader_id,
            turns=[],
            created_at=now,
            updated_at=now,
        )

    async def _persist_turn(
        self, session_id: str, turn: ConversationTurn
    ) -> None:
        """Persist a conversation turn to the database."""
        session_uuid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
        message = ConversationMessage(
            id=uuid.uuid4(),
            session_id=session_uuid,
            role=turn.role,
            content=turn.content,
            extra_data=turn.metadata,
            created_at=turn.timestamp,
        )
        self.db.add(message)

        # Update session timestamp
        stmt = select(ConversationSession).where(
            ConversationSession.id == session_uuid
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()
        if session:
            session.updated_at = turn.timestamp
            if turn.role == "user" and not session.mode:
                # Set mode on first interaction if not yet set
                pass

        await self.db.commit()