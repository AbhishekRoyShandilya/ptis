# 002_shadow_trade_tables.py
# alembic/versions/002_shadow_trade_tables.py
"""
Shadow trading tables: sessions, trades, snapshots, divergences.

Revision ID: 002
Revises: 001
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Shadow Sessions ──
    op.create_table(
        "shadow_sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("real_trade_id", UUID(as_uuid=True),
                  sa.ForeignKey("trades.id", ondelete="CASCADE"), nullable=False),
        sa.Column("strategy_id", sa.String(50), nullable=True),
        sa.Column("status", sa.String(20), default="active"),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("summary", JSONB, default={}),
    )
    op.create_index("ix_shadow_session_trade", "shadow_sessions", ["real_trade_id"])
    op.create_index("ix_shadow_session_strategy", "shadow_sessions", ["strategy_id"])

    # ── Shadow Trades ──
    op.create_table(
        "shadow_trades",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True),
                  sa.ForeignKey("shadow_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("real_trade_id", UUID(as_uuid=True),
                  sa.ForeignKey("trades.id", ondelete="CASCADE"), nullable=False),
        sa.Column("shadow_type", sa.String(30), nullable=False),
        sa.Column("label", sa.String(100), nullable=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("direction", sa.String(10), nullable=False),
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("position_size", sa.Float, nullable=False),
        sa.Column("stop_loss", sa.Float, nullable=True),
        sa.Column("original_stop", sa.Float, nullable=True),
        sa.Column("take_profit", sa.Float, nullable=True),
        sa.Column("trailing_stop", sa.Float, nullable=True),
        sa.Column("trailing_config", JSONB, default={}),
        sa.Column("status", sa.String(20), default="open"),
        sa.Column("current_price", sa.Float, nullable=True),
        sa.Column("unrealized_pnl", sa.Float, default=0.0),
        sa.Column("max_favorable", sa.Float, default=0.0),
        sa.Column("max_adverse", sa.Float, default=0.0),
        sa.Column("high_since_entry", sa.Float, nullable=True),
        sa.Column("low_since_entry", sa.Float, nullable=True),
        sa.Column("exit_price", sa.Float, nullable=True),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_reason", sa.String(50), nullable=True),
        sa.Column("realized_pnl", sa.Float, nullable=True),
        sa.Column("r_multiple", sa.Float, nullable=True),
        sa.Column("risk_amount", sa.Float, nullable=True),
        sa.Column("partial_exits", JSONB, default=[]),
        sa.Column("remaining_size", sa.Float, nullable=True),
        sa.Column("strategy_id", sa.String(50), nullable=True),
        sa.Column("config", JSONB, default={}),
        sa.Column("price_updates", sa.Integer, default=0),
        sa.Column("last_update", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_shadow_trade_real", "shadow_trades", ["real_trade_id"])
    op.create_index("ix_shadow_trade_type", "shadow_trades", ["shadow_type"])
    op.create_index("ix_shadow_trade_status", "shadow_trades", ["status"])

    # ── Price Snapshots ──
    op.create_table(
        "shadow_price_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("shadow_trade_id", UUID(as_uuid=True),
                  sa.ForeignKey("shadow_trades.id", ondelete="CASCADE"), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("price", sa.Float, nullable=False),
        sa.Column("unrealized_pnl", sa.Float, nullable=False),
        sa.Column("r_multiple", sa.Float, nullable=True),
    )
    op.create_index("ix_snapshot_shadow_time", "shadow_price_snapshots",
                    ["shadow_trade_id", "timestamp"])

    # ── Trade Divergences ──
    op.create_table(
        "trade_divergences",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("shadow_trade_id", UUID(as_uuid=True),
                  sa.ForeignKey("shadow_trades.id", ondelete="CASCADE"), nullable=False),
        sa.Column("real_trade_id", UUID(as_uuid=True),
                  sa.ForeignKey("trades.id", ondelete="CASCADE"), nullable=False),
        sa.Column("divergence_type", sa.String(50), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("real_value", sa.String(200), nullable=True),
        sa.Column("shadow_value", sa.String(200), nullable=True),
        sa.Column("impact_amount", sa.Float, nullable=True),
        sa.Column("impact_r", sa.Float, nullable=True),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_divergence_real", "trade_divergences", ["real_trade_id"])
    op.create_index("ix_divergence_shadow", "trade_divergences", ["shadow_trade_id"])


def downgrade() -> None:
    op.drop_table("trade_divergences")
    op.drop_table("shadow_price_snapshots")
    op.drop_table("shadow_trades")
    op.drop_table("shadow_sessions")