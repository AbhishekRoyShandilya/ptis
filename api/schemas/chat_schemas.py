# chat_schemas.py
"""
Chat API Schemas — Request/Response models for the coaching agent.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Shared Models ───────────────────────────────────────────────


class CoachingInsightResponse(BaseModel):
    """A single coaching insight."""
    id: str
    priority: str = ""     # legacy field — kept for backward compat
    severity: str = "info" # frontend uses this: "critical"|"warning"|"info"|"positive"
    category: str = ""
    title: str
    content: str = ""      # legacy field — kept for backward compat
    message: str = ""      # frontend uses this (ins.message in ChatPanel)
    evidence: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)


# ─── Chat Message ────────────────────────────────────────────────


class ChatMessageRequest(BaseModel):
    """Request to send a chat message."""
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = None
    trade_id: Optional[str] = None
    mode: Optional[str] = None  # Override coaching mode
    personality: Optional[str] = Field(
        default="balanced",
        description="Coaching personality: analytical, socratic, supportive, direct, balanced",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "How was my trading day today?",
                "personality": "balanced",
            }
        }


class ChatMessageResponse(BaseModel):
    """Response from the coaching agent."""
    session_id: str
    message: str
    mode: str
    insights: list[CoachingInsightResponse] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    suggested_focus: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


# ─── Trade Review ────────────────────────────────────────────────


class TradeReviewRequest(BaseModel):
    """Request to review a specific trade."""
    trade_id: str
    message: Optional[str] = None
    focus_areas: Optional[list[str]] = Field(
        default=None,
        description="Specific areas to focus on: risk, entry, exit, process, behavior",
    )
    session_id: Optional[str] = None
    personality: Optional[str] = "balanced"

    class Config:
        json_schema_extra = {
            "example": {
                "trade_id": "abc123",
                "message": "Why did I exit too early on this trade?",
                "focus_areas": ["exit", "behavior"],
            }
        }


class TradeReviewResponse(BaseModel):
    """Detailed trade review response."""
    session_id: str
    trade_id: str
    message: str
    trade_summary: dict[str, Any] = Field(default_factory=dict)
    decision_quality: dict[str, Any] = Field(default_factory=dict)
    insights: list[CoachingInsightResponse] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


# ─── Proactive Briefing ─────────────────────────────────────────


class ProactiveBriefingRequest(BaseModel):
    """Request a proactive coaching briefing."""
    personality: Optional[str] = "balanced"
    briefing_type: Optional[str] = Field(
        default=None,
        description="Type: pre_session, session_debrief, weekly_review",
    )


class ProactiveBriefingResponse(BaseModel):
    """Proactive briefing response."""
    message: str
    mode: str
    insights: list[CoachingInsightResponse] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    suggested_focus: Optional[str] = None


# ─── Session Management ─────────────────────────────────────────


class ChatSessionResponse(BaseModel):
    """Summary of a conversation session."""
    session_id: str
    mode: Optional[str] = None
    preview: Optional[str] = None
    turn_count: int
    created_at: str
    updated_at: str


class ChatSessionListResponse(BaseModel):
    """List of conversation sessions."""
    sessions: list[ChatSessionResponse]


class ChatHistoryResponse(BaseModel):
    """Full conversation history for a session."""
    session_id: str
    messages: list[dict[str, str]]
    total_turns: int