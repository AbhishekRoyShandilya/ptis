"""Add strategy_profiles, learning_sessions, and learning_examples tables.

Supports the AI-learned strategy understanding system.

Revision ID: 011
Revises: 010
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "011"
down_revision = "010"


def upgrade() -> None:
    # ------------------------------------------------------------------
    # strategy_profiles — AI-learned strategy understanding
    # ------------------------------------------------------------------
    op.create_table(
        "strategy_profiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("short_code", sa.String(20), nullable=False, unique=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="draft"),
        # Learned understanding
        sa.Column("narrative", sa.Text, nullable=True),
        sa.Column("entry_criteria", JSONB, server_default="[]"),
        sa.Column("exit_rules", JSONB, server_default="{}"),
        sa.Column("risk_parameters", JSONB, server_default="{}"),
        sa.Column("regime_preferences", JSONB, server_default="{}"),
        sa.Column("time_preferences", JSONB, server_default="{}"),
        sa.Column("confidence_scores", JSONB, server_default="{}"),
        # Counters
        sa.Column("examples_count", sa.Integer, server_default="0"),
        sa.Column("sessions_count", sa.Integer, server_default="0"),
        # Timestamps
        sa.Column("last_rebuilt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        # Flexible metadata
        sa.Column("tags", JSONB, server_default="[]"),
        sa.Column("metadata", JSONB, server_default="{}"),
    )
    op.create_index("ix_strategy_profile_status", "strategy_profiles", ["status"])
    op.create_index("ix_strategy_profile_short_code", "strategy_profiles", ["short_code"])

    # ------------------------------------------------------------------
    # learning_sessions — teaching sessions grouping examples
    # ------------------------------------------------------------------
    op.create_table(
        "learning_sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "strategy_profile_id",
            UUID(as_uuid=True),
            sa.ForeignKey("strategy_profiles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("session_type", sa.String(30), nullable=False, server_default="initial_teaching"),
        # Session data
        sa.Column("examples_count", sa.Integer, server_default="0"),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("ai_summary", sa.Text, nullable=True),
        # Timestamps
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_learning_session_profile", "learning_sessions", ["strategy_profile_id"])
    op.create_index("ix_learning_session_status", "learning_sessions", ["status"])

    # ------------------------------------------------------------------
    # learning_examples — individual screenshot + annotation pairs
    # ------------------------------------------------------------------
    op.create_table(
        "learning_examples",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("learning_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "strategy_profile_id",
            UUID(as_uuid=True),
            sa.ForeignKey("strategy_profiles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Screenshot
        sa.Column("screenshot_path", sa.String(500), nullable=False),
        sa.Column("screenshot_hash", sa.String(64), nullable=True),
        # AI analysis
        sa.Column("vision_analysis", JSONB, server_default="{}"),
        sa.Column("ai_interpretation", sa.Text, nullable=True),
        # User input
        sa.Column("user_annotations", JSONB, server_default="{}"),
        sa.Column("is_positive_example", sa.Boolean, server_default="true"),
        # Context
        sa.Column("market_context", JSONB, server_default="{}"),
        sa.Column("embedding_id", sa.String(100), nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_learning_example_session", "learning_examples", ["session_id"])
    op.create_index("ix_learning_example_profile", "learning_examples", ["strategy_profile_id"])


def downgrade() -> None:
    op.drop_table("learning_examples")
    op.drop_table("learning_sessions")
    op.drop_table("strategy_profiles")
