"""Add strategy_symbol_performance — per-(strategy, stock[, context]) ranking baseline

Initial-ranking dataset so a freshly-taught strategy isn't ranked blind. Aggregated
from the CLEAN AVT strategy-research record (marketdata/strategy_research) —
BENCHMARK REPLAY ONLY, never user trades. Includes a confidence layer
(effective_sample_size, confidence_score) so thin cells don't outrank deep ones,
and nullable regime/market_state/strategy_version columns to future-proof for
market-state + per-version intelligence.

Revision ID: 034
Revises: 033
Create Date: 2026-06-08
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "strategy_symbol_performance",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trader_id", sa.String(50), nullable=False),
        sa.Column("strategy_code", sa.String(64), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        # context (nullable / sentinel)
        sa.Column("regime", sa.String(32), nullable=True),
        sa.Column("market_state", sa.String(32), nullable=True),
        sa.Column("strategy_version", sa.String(16), nullable=True),
        # full-window stats
        sa.Column("trade_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("win_rate", sa.Float, nullable=True),
        sa.Column("avg_r", sa.Float, nullable=True),
        sa.Column("expectancy", sa.Float, nullable=True),
        sa.Column("total_r", sa.Float, nullable=True),
        # recent window
        sa.Column("last_n", sa.Integer, nullable=True),
        sa.Column("last_n_avg_r", sa.Float, nullable=True),
        # confidence
        sa.Column("effective_sample_size", sa.Float, nullable=True),
        sa.Column("confidence_score", sa.Float, nullable=True),
        sa.Column("sample_window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sample_window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False,
        ),
        sa.UniqueConstraint(
            "trader_id", "strategy_code", "symbol", "regime", "market_state",
            "strategy_version", name="uq_ssp_trader_strat_symbol_ctx",
        ),
    )
    op.create_index("ix_ssp_trader", "strategy_symbol_performance", ["trader_id"])
    op.create_index("ix_ssp_trader_strategy", "strategy_symbol_performance", ["trader_id", "strategy_code"])
    op.create_index(
        "ix_ssp_trader_strategy_symbol", "strategy_symbol_performance",
        ["trader_id", "strategy_code", "symbol"],
    )


def downgrade() -> None:
    op.drop_index("ix_ssp_trader_strategy_symbol", table_name="strategy_symbol_performance")
    op.drop_index("ix_ssp_trader_strategy", table_name="strategy_symbol_performance")
    op.drop_index("ix_ssp_trader", table_name="strategy_symbol_performance")
    op.drop_table("strategy_symbol_performance")
