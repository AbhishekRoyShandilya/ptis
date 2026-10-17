# 004_behavioral_events_tables.py
# alembic/versions/004_behavioral_events_tables.py
"""
Behavioral events table for behavioral dataset accumulation.

Revision ID: 004
Revises: 003
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "behavioral_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trade_id", UUID(as_uuid=True), sa.ForeignKey("trades.id", ondelete="SET NULL"), nullable=True),
        sa.Column("session_date", sa.DateTime, nullable=False),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("severity_score", sa.Float, nullable=False),
        sa.Column("is_positive", sa.Boolean, default=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("trigger_conditions", JSONB, default={}),
        sa.Column("emotional_state", sa.String(20), nullable=True),
        sa.Column("market_regime", sa.String(20), nullable=True),
        sa.Column("session_type", sa.String(20), nullable=True),
        sa.Column("pnl_at_detection", sa.Float, nullable=True),
        sa.Column("trades_today_at_detection", sa.Integer, nullable=True),
        sa.Column("outcome_pnl", sa.Float, nullable=True),
        sa.Column("outcome_r", sa.Float, nullable=True),
        sa.Column("outcome_validated", sa.Boolean, default=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_behav_session_type", "behavioral_events", ["session_date", "event_type"])
    op.create_index("ix_behav_detected", "behavioral_events", ["detected_at"])
    op.create_index("ix_behav_event_type", "behavioral_events", ["event_type"])
    op.create_index("ix_behav_trade", "behavioral_events", ["trade_id"])


def downgrade() -> None:
    op.drop_table("behavioral_events")