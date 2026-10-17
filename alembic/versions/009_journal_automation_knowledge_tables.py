"""Add journal_entries, automation_rules, and knowledge_entries tables.

Revision ID: 009
Revises: 008
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "009"
down_revision = "008_intelligence_tables"


def upgrade() -> None:
    # ------------------------------------------------------------------
    # journal_entries — one-to-one with trades (pre/post trade journal)
    # ------------------------------------------------------------------
    op.create_table(
        "journal_entries",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "trade_id",
            UUID(as_uuid=True),
            sa.ForeignKey("trades.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        # Pre-trade
        sa.Column("plan", sa.Text, nullable=True),
        sa.Column("was_planned", sa.Boolean, default=True),
        sa.Column("pre_trade_confidence", sa.Integer, nullable=True),
        sa.Column("pre_trade_emotion", sa.String(30), nullable=True),
        sa.Column("sleep_quality", sa.Integer, nullable=True),
        sa.Column("energy_level", sa.Integer, nullable=True),
        sa.Column("stress_level", sa.Integer, nullable=True),
        # Post-trade
        sa.Column("post_trade_emotion", sa.String(30), nullable=True),
        sa.Column("followed_plan", sa.Boolean, nullable=True),
        sa.Column("would_take_again", sa.Boolean, nullable=True),
        sa.Column("what_went_well", sa.Text, nullable=True),
        sa.Column("what_went_wrong", sa.Text, nullable=True),
        sa.Column("lesson", sa.Text, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_journal_entries_trade_id", "journal_entries", ["trade_id"])

    # ------------------------------------------------------------------
    # automation_rules — guardrail and circuit-breaker configuration
    # ------------------------------------------------------------------
    op.create_table(
        "automation_rules",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("rule_type", sa.String(50), nullable=False),
        sa.Column("trigger_value", sa.Float, nullable=False, default=0.0),
        sa.Column("action", sa.String(30), nullable=False, default="alert"),
        sa.Column("enabled", sa.Boolean, default=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("triggered_count", sa.Integer, default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_automation_rules_rule_type", "automation_rules", ["rule_type"])
    op.create_index("ix_automation_rules_enabled", "automation_rules", ["enabled"])
    op.create_index(
        "ix_automation_enabled_type", "automation_rules", ["enabled", "rule_type"]
    )

    # ------------------------------------------------------------------
    # knowledge_entries — personal trading knowledge for RAG retrieval
    # ------------------------------------------------------------------
    op.create_table(
        "knowledge_entries",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("category", sa.String(30), nullable=False),
        # "strategy" | "lesson" | "pattern" | "insight" | "rule" | "review"
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("source", sa.String(50), default="manual"),
        sa.Column("tags", JSONB, default=[]),
        sa.Column("relevance_count", sa.Integer, default=0),
        sa.Column("last_referenced", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_knowledge_entries_category", "knowledge_entries", ["category"])
    op.create_index("ix_knowledge_entries_created_at", "knowledge_entries", ["created_at"])


    # ------------------------------------------------------------------
    # market_regime_logs — timestamped regime change log
    # ------------------------------------------------------------------
    op.create_table(
        "market_regime_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("regime", sa.String(30), nullable=False),
        sa.Column("confidence", sa.Float, default=0.8),
        sa.Column("indicators", JSONB, default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_market_regime_logs_timestamp", "market_regime_logs", ["timestamp"])


def downgrade() -> None:
    op.drop_table("market_regime_logs")
    op.drop_table("knowledge_entries")
    op.drop_table("automation_rules")
    op.drop_table("journal_entries")
