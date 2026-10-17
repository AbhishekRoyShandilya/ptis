"""AVT — autonomous virtual trader tables

Revision ID: 018
Revises: 017
Create Date: 2026-03-22
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── avt_virtual_trades ──────────────────────────────────────
    op.create_table(
        "avt_virtual_trades",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "trader_id", UUID(as_uuid=True),
            sa.ForeignKey("users.id"), nullable=True,
        ),
        sa.Column("strategy_id", UUID(as_uuid=True), nullable=False),
        sa.Column("strategy_version", sa.String(20), nullable=False),
        sa.Column("variant_id", UUID(as_uuid=True), nullable=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("direction", sa.String(5), nullable=False),
        # Entry
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entry_price", sa.Numeric(12, 2), nullable=False),
        sa.Column("quantity", sa.Integer, nullable=False),
        sa.Column("stop_price", sa.Numeric(12, 2), nullable=False),
        sa.Column("conviction_at_entry", sa.Numeric(4, 2), nullable=True),
        sa.Column("feature_vector", JSONB, server_default="[]"),
        sa.Column("regime_at_entry", JSONB, server_default="{}"),
        sa.Column("ranking_data", JSONB, server_default="{}"),
        # During-trade
        sa.Column("max_favorable_r", sa.Numeric(6, 3), server_default="0"),
        sa.Column("max_adverse_r", sa.Numeric(6, 3), server_default="0"),
        sa.Column("features_at_peak", JSONB, nullable=True),
        # Exit
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_price", sa.Numeric(12, 2), nullable=True),
        sa.Column("exit_type", sa.String(30), nullable=True),
        sa.Column("exit_reason", sa.Text, nullable=True),
        sa.Column("pnl_r", sa.Numeric(6, 3), nullable=True),
        sa.Column("pnl_pct", sa.Numeric(8, 4), nullable=True),
        sa.Column("pnl_absolute", sa.Numeric(12, 2), nullable=True),
        # Improvement tracking
        sa.Column("active_improvements", ARRAY(UUID(as_uuid=True)), server_default="{}"),
        sa.Column("improvement_exits", JSONB, server_default="[]"),
        # Status
        sa.Column("status", sa.String(10), server_default="open", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_avt_trades_strategy", "avt_virtual_trades", ["strategy_id", "strategy_version"])
    op.create_index("idx_avt_trades_date", "avt_virtual_trades", ["entry_time"])
    op.create_index("idx_avt_trades_symbol", "avt_virtual_trades", ["symbol"])
    op.create_index("idx_avt_trades_status", "avt_virtual_trades", ["status"])

    # ── avt_strategy_versions ───────────────────────────────────
    op.create_table(
        "avt_strategy_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("strategy_id", UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("version", sa.String(20), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("active_improvements", ARRAY(UUID(as_uuid=True)), server_default="{}"),
        sa.Column("trade_count", sa.Integer, server_default="0"),
        sa.Column("performance_stats", JSONB, server_default="{}"),
        sa.Column("notes", sa.Text, nullable=True),
    )
    op.create_index(
        "idx_avt_versions_strategy_version",
        "avt_strategy_versions",
        ["strategy_id", "version"],
        unique=True,
    )

    # ── avt_improvements ────────────────────────────────────────
    op.create_table(
        "avt_improvements",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("strategy_id", UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("hypothesis_type", sa.String(50), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("params", JSONB, server_default="{}"),
        sa.Column("status", sa.String(20), server_default="active", nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("applied_at_version", sa.String(20), nullable=True),
        sa.Column("validation_result", JSONB, server_default="{}"),
        sa.Column("performance_before", JSONB, nullable=True),
        sa.Column("performance_after", JSONB, nullable=True),
        sa.Column("evidence_strength", sa.Numeric(4, 2), server_default="0"),
        sa.Column("reverted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revert_reason", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "idx_avt_improvements_strategy_status",
        "avt_improvements",
        ["strategy_id", "status"],
    )

    # ── avt_scan_logs ───────────────────────────────────────────
    op.create_table(
        "avt_scan_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("strategy_id", UUID(as_uuid=True), nullable=True),
        sa.Column("cycle_time", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("candidates_found", sa.Integer, server_default="0"),
        sa.Column("trades_opened", sa.Integer, server_default="0"),
        sa.Column("trades_closed", sa.Integer, server_default="0"),
        sa.Column("skipped", JSONB, server_default="[]"),
        sa.Column("top_ranked", JSONB, server_default="[]"),
        sa.Column("duration_ms", sa.Integer, server_default="0"),
    )
    op.create_index("idx_avt_scan_logs_time", "avt_scan_logs", ["cycle_time"])

    # ── avt_competition_snapshots ───────────────────────────────
    op.create_table(
        "avt_competition_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("strategy_id", UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("period_days", sa.Integer, nullable=False),
        sa.Column("snapshot_date", sa.Date, nullable=False),
        sa.Column("user_stats", JSONB, server_default="{}"),
        sa.Column("avt_stats", JSONB, server_default="{}"),
        sa.Column("performance_gap", sa.Numeric(6, 3), nullable=True),
        sa.Column("differences", JSONB, server_default="[]"),
        sa.Column("proven_improvements", JSONB, server_default="[]"),
    )
    op.create_index(
        "idx_avt_competition_strategy_date",
        "avt_competition_snapshots",
        ["strategy_id", "snapshot_date"],
    )


def downgrade() -> None:
    op.drop_table("avt_competition_snapshots")
    op.drop_table("avt_scan_logs")
    op.drop_index("idx_avt_improvements_strategy_status", table_name="avt_improvements")
    op.drop_table("avt_improvements")
    op.drop_index("idx_avt_versions_strategy_version", table_name="avt_strategy_versions")
    op.drop_table("avt_strategy_versions")
    op.drop_index("idx_avt_trades_status", table_name="avt_virtual_trades")
    op.drop_index("idx_avt_trades_symbol", table_name="avt_virtual_trades")
    op.drop_index("idx_avt_trades_date", table_name="avt_virtual_trades")
    op.drop_index("idx_avt_trades_strategy", table_name="avt_virtual_trades")
    op.drop_table("avt_virtual_trades")
