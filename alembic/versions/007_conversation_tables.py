"""Add conversation tables for coaching agent.

Revision ID: 007
Revises: 006
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "007_conversation_tables"
down_revision = "006_decision_quality_tables"


def upgrade() -> None:
    # Conversation sessions
    op.create_table(
        "conversation_sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trader_id", sa.String(255), nullable=False, index=True),
        sa.Column("mode", sa.String(50), nullable=True),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column("metadata", JSONB, default={}),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("ended_at", sa.DateTime, nullable=True),
    )

    op.create_index(
        "ix_conv_sessions_active",
        "conversation_sessions",
        ["trader_id", "is_active", "updated_at"],
    )

    # Conversation messages
    op.create_table(
        "conversation_messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("role", sa.String(20), nullable=False),  # user, assistant
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("metadata", JSONB, default={}),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_index(
        "ix_conv_messages_session_time",
        "conversation_messages",
        ["session_id", "created_at"],
    )

    # Coaching insights (persisted for analytics)
    op.create_table(
        "coaching_insights",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trader_id", sa.String(255), nullable=False, index=True),
        sa.Column("session_id", UUID(as_uuid=True), nullable=True),
        sa.Column("trade_id", UUID(as_uuid=True), nullable=True),
        sa.Column("priority", sa.String(20), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("evidence", JSONB, default=[]),
        sa.Column("action_items", JSONB, default=[]),
        sa.Column("was_acknowledged", sa.Boolean, default=False),
        sa.Column("was_acted_upon", sa.Boolean, default=False),
        sa.Column("metadata", JSONB, default={}),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_index(
        "ix_insights_trader_priority",
        "coaching_insights",
        ["trader_id", "priority", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("coaching_insights")
    op.drop_table("conversation_messages")
    op.drop_table("conversation_sessions")