"""Add daily_performance_summaries + automation_readiness_scores tables

Revision ID: 020
Revises: 019
Create Date: 2026-03-23
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Daily performance summaries
    op.create_table(
        "daily_performance_summaries",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.Column("total_trades", sa.Integer, server_default="0"),
        sa.Column("winners", sa.Integer, server_default="0"),
        sa.Column("losers", sa.Integer, server_default="0"),
        sa.Column("total_pnl", sa.Float, server_default="0"),
        sa.Column("total_r", sa.Float, server_default="0"),
        sa.Column("win_rate", sa.Float, server_default="0"),
        sa.Column("avg_r", sa.Float, server_default="0"),
        sa.Column("rule_adherence_rate", sa.Float, server_default="100"),
        sa.Column("avg_decision_quality", sa.Float, nullable=True),
        sa.Column("deviations_count", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_daily_summary_date", "daily_performance_summaries", ["date"])

    # Automation readiness scores
    op.create_table(
        "automation_readiness_scores",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("strategy_name", sa.String(100), nullable=False),
        sa.Column("rule_clarity", sa.Float, server_default="0"),
        sa.Column("rule_consistency", sa.Float, server_default="0"),
        sa.Column("performance_stability", sa.Float, server_default="0"),
        sa.Column("edge_confidence", sa.Float, server_default="0"),
        sa.Column("behavioral_discipline", sa.Float, server_default="0"),
        sa.Column("overall_readiness", sa.Float, server_default="0"),
        sa.Column("current_phase", sa.String(20), server_default="observation"),
        sa.Column("recommended_phase", sa.String(20), server_default="observation"),
        sa.Column("blockers", JSONB, server_default="[]"),
        sa.Column("recommendations", JSONB, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_readiness_strategy", "automation_readiness_scores", ["strategy_name"])
    op.create_index(
        "ix_readiness_strategy_created",
        "automation_readiness_scores",
        ["strategy_name", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("automation_readiness_scores")
    op.drop_table("daily_performance_summaries")
