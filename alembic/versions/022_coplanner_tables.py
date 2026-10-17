"""Add co-planner intelligence tables

Revision ID: 022_coplanner_tables
Revises: 021
Create Date: 2026-03-24
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "022_coplanner_tables"
down_revision = "021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Strategy × Regime Matrix cache ──
    op.create_table(
        "coplanner_regime_matrix",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("strategy_name", sa.String(100), nullable=False),
        sa.Column("regime", sa.String(50), nullable=False),
        sa.Column("win_rate", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("avg_rr", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("total_trades", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_wins", sa.Integer, nullable=False, server_default="0"),
        sa.Column("avg_pnl", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("max_drawdown", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("data", JSONB, nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.UniqueConstraint("strategy_name", "regime", name="uq_regime_matrix_strat_regime"),
    )

    # ── Pattern Cluster statistics cache ──
    op.create_table(
        "coplanner_cluster_stats",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("cluster_id", sa.String(100), nullable=False, unique=True),
        sa.Column("win_rate", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("total_trades", sa.Integer, nullable=False, server_default="0"),
        sa.Column("avg_rr", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("avg_pnl", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("data", JSONB, nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )

    # ── Decision log (session-level audit trail) ──
    op.create_table(
        "coplanner_decisions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(64), nullable=False, index=True),
        sa.Column("trader_id", sa.String(64), nullable=True),
        sa.Column("decision_data", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Indexes for common queries
    op.create_index(
        "ix_regime_matrix_strategy",
        "coplanner_regime_matrix",
        ["strategy_name"],
    )
    op.create_index(
        "ix_regime_matrix_regime",
        "coplanner_regime_matrix",
        ["regime"],
    )
    op.create_index(
        "ix_decisions_created",
        "coplanner_decisions",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_table("coplanner_decisions")
    op.drop_table("coplanner_cluster_stats")
    op.drop_table("coplanner_regime_matrix")
