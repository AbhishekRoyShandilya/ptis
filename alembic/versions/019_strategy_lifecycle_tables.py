"""AVT — strategy lifecycle + calibration signal tables

Revision ID: 019
Revises: 018
Create Date: 2026-03-22
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── avt_strategy_lifecycle ───────────────────────────────────
    op.create_table(
        "avt_strategy_lifecycle",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("strategy_id", UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("state", sa.String(20), server_default="ingested", nullable=False),
        sa.Column("grade", sa.String(20), nullable=True),
        # Health gate
        sa.Column("health_score", sa.Numeric(5, 2), server_default="0"),
        sa.Column("structural_score", sa.Numeric(5, 2), server_default="0"),
        sa.Column("coherence_score", sa.Numeric(5, 2), server_default="0"),
        sa.Column("health_issues", JSONB, server_default="[]"),
        # Calibration
        sa.Column("calibration_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("calibration_alignment", sa.Numeric(5, 3), nullable=True),
        sa.Column("calibration_comparisons", sa.Integer, server_default="0"),
        sa.Column("calibration_report", JSONB, server_default="{}"),
        # Baseline
        sa.Column("baseline_source", sa.String(30), nullable=True),
        sa.Column("baseline_metrics", JSONB, server_default="{}"),
        # Edge monitoring
        sa.Column("degradation_level", sa.String(20), server_default="healthy"),
        sa.Column("monitor_action", sa.String(20), server_default="continue"),
        sa.Column("rolling_expectancy", sa.Numeric(8, 4), nullable=True),
        sa.Column("rolling_win_rate", sa.Numeric(5, 4), nullable=True),
        sa.Column("translation_drift", sa.Numeric(5, 3), nullable=True),
        # Probation
        sa.Column("probation_trade_count", sa.Integer, server_default="0"),
        sa.Column("allocation_multiplier", sa.Numeric(4, 2), server_default="0"),
        # Timestamps
        sa.Column("health_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("graduated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_lifecycle_state", "avt_strategy_lifecycle", ["state"])
    op.create_index("idx_lifecycle_strategy", "avt_strategy_lifecycle", ["strategy_id"], unique=True)

    # ── avt_calibration_signals ──────────────────────────────────
    op.create_table(
        "avt_calibration_signals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("strategy_id", UUID(as_uuid=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("direction", sa.String(5), nullable=False),
        sa.Column("signal_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("conviction", sa.Numeric(4, 2), nullable=True),
        sa.Column("signal_details", JSONB, server_default="{}"),
        # Matching
        sa.Column("matched_trade_id", UUID(as_uuid=True), nullable=True),
        sa.Column("match_type", sa.String(20), nullable=True),
        sa.Column("user_annotation", sa.Text, nullable=True),
        # Timestamp
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_calibration_signals_strategy", "avt_calibration_signals", ["strategy_id"])
    op.create_index("idx_calibration_signals_time", "avt_calibration_signals", ["signal_time"])


def downgrade() -> None:
    op.drop_index("idx_calibration_signals_time", table_name="avt_calibration_signals")
    op.drop_index("idx_calibration_signals_strategy", table_name="avt_calibration_signals")
    op.drop_table("avt_calibration_signals")
    op.drop_index("idx_lifecycle_strategy", table_name="avt_strategy_lifecycle")
    op.drop_index("idx_lifecycle_state", table_name="avt_strategy_lifecycle")
    op.drop_table("avt_strategy_lifecycle")
