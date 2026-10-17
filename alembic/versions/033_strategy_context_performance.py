"""Add strategy_context_performance — the edge-decay matrix (Phase 2)

Per (trader, strategy, context_key) performance aggregated EOD from the CLEAN
AVT strategy-research record (marketdata/strategy_research) — not raw user
trades. context_key examples: 'regime:breakout', 'vol:high', 'session:morning',
'dir:long'. Powers edge-decay detection ("ORB edge falling in ranging, VWAP edge rising").

Revision ID: 033
Revises: 032
Create Date: 2026-06-07
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "strategy_context_performance",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trader_id", sa.String(50), nullable=False),
        sa.Column("strategy_code", sa.String(64), nullable=False),
        sa.Column("context_key", sa.String(64), nullable=False),
        sa.Column("trade_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("win_rate", sa.Float, nullable=True),
        sa.Column("avg_r", sa.Float, nullable=True),
        sa.Column("expectancy", sa.Float, nullable=True),
        sa.Column("total_r", sa.Float, nullable=True),
        sa.Column("last_n", sa.Integer, nullable=True),
        sa.Column("last_n_win_rate", sa.Float, nullable=True),
        sa.Column("last_n_avg_r", sa.Float, nullable=True),
        sa.Column("edge_trend", sa.String(16), nullable=True),
        sa.Column("sample_window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sample_window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False,
        ),
        sa.UniqueConstraint("trader_id", "strategy_code", "context_key", name="uq_scp_trader_strat_ctx"),
    )
    op.create_index("ix_scp_trader", "strategy_context_performance", ["trader_id"])
    op.create_index("ix_scp_trader_strategy", "strategy_context_performance", ["trader_id", "strategy_code"])


def downgrade() -> None:
    op.drop_index("ix_scp_trader_strategy", table_name="strategy_context_performance")
    op.drop_index("ix_scp_trader", table_name="strategy_context_performance")
    op.drop_table("strategy_context_performance")
