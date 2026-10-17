"""
001 — Initial schema: trades table.

This is the foundation migration. Every other migration chains off this one
(002 creates shadow tables with FK to trades.id, 003 adds attribution tables, etc.).

Revision ID: 001
Revises: None (initial migration)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "trades",
        # ── Identity ──
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        # Enum columns stored as VARCHAR — ORM layer enforces valid values
        sa.Column("direction", sa.String(10), nullable=False),   # long | short
        sa.Column("status", sa.String(20), nullable=False, server_default="open"),

        # ── Prices & Times ──
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("exit_price", sa.Float, nullable=True),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),

        # ── Sizing ──
        sa.Column("position_size", sa.Float, nullable=False),
        sa.Column("stop_loss", sa.Float, nullable=True),
        sa.Column("take_profit", sa.Float, nullable=True),

        # ── P&L ──
        sa.Column("pnl", sa.Float, nullable=True),
        sa.Column("pnl_pct", sa.Float, nullable=True),
        sa.Column("r_multiple", sa.Float, nullable=True),
        sa.Column("commission", sa.Float, nullable=True, server_default="0"),

        # ── Risk ──
        sa.Column("risk_amount", sa.Float, nullable=True),
        sa.Column("risk_pct", sa.Float, nullable=True),
        sa.Column("account_balance_at_entry", sa.Float, nullable=True),

        # ── Context ──
        sa.Column("session_type", sa.String(20), nullable=True),
        sa.Column("market_regime", sa.String(30), nullable=True),

        # ── Psychology ──
        sa.Column("emotion_pre", sa.String(20), nullable=True),
        sa.Column("emotion_post", sa.String(20), nullable=True),
        sa.Column("conviction_score", sa.Float, nullable=True),  # 1–10

        # ── Notes ──
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("setup_screenshot", sa.String(500), nullable=True),

        # ── Source ──
        sa.Column("source", sa.String(50), nullable=True, server_default="manual"),
        sa.Column("broker", sa.String(50), nullable=True),
        sa.Column("external_id", sa.String(100), nullable=True),

        # ── Timestamps ──
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Single-column indexes (mirrors Column(index=True) in the ORM model)
    op.create_index("ix_trades_symbol", "trades", ["symbol"])
    op.create_index("ix_trades_entry_time", "trades", ["entry_time"])

    # Composite indexes (mirrors __table_args__ in the ORM model)
    op.create_index("ix_trades_symbol_entry", "trades", ["symbol", "entry_time"])


def downgrade() -> None:
    op.drop_index("ix_trades_symbol_entry", table_name="trades")
    op.drop_index("ix_trades_entry_time", table_name="trades")
    op.drop_index("ix_trades_symbol", table_name="trades")
    op.drop_table("trades")
