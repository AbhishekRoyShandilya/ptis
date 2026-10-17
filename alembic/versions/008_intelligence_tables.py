"""Add intelligence layer tables.

Revision ID: 008
Revises: 007
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "008_intelligence_tables"
down_revision = "007_conversation_tables"


def upgrade() -> None:
    # Trader intelligence profiles (cached, rebuilt daily)
    op.create_table(
        "trader_intelligence_profiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trader_id", sa.String(255), nullable=False, index=True),
        sa.Column("trading_style", sa.String(50)),
        sa.Column("entry_style", sa.String(50)),
        sa.Column("cognitive_type", sa.String(50)),
        sa.Column("risk_personality", sa.String(50)),
        sa.Column("discipline_pattern", sa.String(50)),
        sa.Column("current_discipline_score", sa.Float),
        sa.Column("avg_dq_score", sa.Float),
        sa.Column("weakest_dimension", sa.String(100)),
        sa.Column("strongest_dimension", sa.String(100)),
        sa.Column("core_strategies", JSONB, default=[]),
        sa.Column("best_conditions", JSONB, default=[]),
        sa.Column("worst_conditions", JSONB, default=[]),
        sa.Column("risk_profile", JSONB, default={}),
        sa.Column("timing_profile", JSONB, default={}),
        sa.Column("emotional_profile", JSONB, default={}),
        sa.Column("strategy_preferences", JSONB, default=[]),
        sa.Column("sample_size", sa.Integer, default=0),
        sa.Column("confidence", sa.Float, default=0.0),
        sa.Column("full_profile", JSONB, default={}),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_index(
        "ix_intel_profiles_trader_updated",
        "trader_intelligence_profiles",
        ["trader_id", "updated_at"],
    )

    # Setup alerts (persisted for analytics)
    op.create_table(
        "setup_alerts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trader_id", sa.String(255), nullable=False, index=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("strategy_name", sa.String(100)),
        sa.Column("alert_type", sa.String(50), nullable=False),
        sa.Column("priority", sa.String(20), nullable=False),
        sa.Column("message", sa.Text),
        sa.Column("readiness_score", sa.Float),
        sa.Column("is_triggered", sa.Boolean, default=False),
        sa.Column("regime", sa.String(50)),
        sa.Column("environment_score", sa.Float),
        sa.Column("estimated_r", sa.Float),
        sa.Column("was_acted_upon", sa.Boolean, default=False),
        sa.Column("outcome_r", sa.Float, nullable=True),
        sa.Column("metadata", JSONB, default={}),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_index(
        "ix_setup_alerts_trader_created",
        "setup_alerts",
        ["trader_id", "created_at"],
    )
    op.create_index(
        "ix_setup_alerts_symbol",
        "setup_alerts",
        ["symbol", "created_at"],
    )

    # Discretionary evaluations (track accuracy over time)
    op.create_table(
        "discretionary_evaluations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trader_id", sa.String(255), nullable=False, index=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("query", sa.Text),
        sa.Column("intent", sa.String(50)),
        sa.Column("recommendation", sa.String(50)),
        sa.Column("confidence", sa.Float),
        sa.Column("setup_valid", sa.Boolean),
        sa.Column("regime", sa.String(50)),
        sa.Column("environment_score", sa.Float),
        sa.Column("estimated_r", sa.Float),
        sa.Column("estimated_rr", sa.Float),
        sa.Column("guardrail_count", sa.Integer, default=0),
        sa.Column("trader_action", sa.String(50), nullable=True),
        sa.Column("actual_r", sa.Float, nullable=True),
        sa.Column("assessment_data", JSONB, default={}),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_index(
        "ix_disc_eval_trader_created",
        "discretionary_evaluations",
        ["trader_id", "created_at"],
    )

    # Market environment snapshots (for historical analysis)
    op.create_table(
        "market_environment_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False, index=True),
        sa.Column("timeframe", sa.String(10)),
        sa.Column("regime", sa.String(50)),
        sa.Column("volatility_state", sa.String(50)),
        sa.Column("trend_direction", sa.Integer),
        sa.Column("trend_strength", sa.String(20)),
        sa.Column("environment_score", sa.Float),
        sa.Column("atr", sa.Float),
        sa.Column("vwap", sa.Float),
        sa.Column("current_price", sa.Float),
        sa.Column("structure_data", JSONB, default={}),
        sa.Column("full_snapshot", JSONB, default={}),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_index(
        "ix_env_snapshots_symbol_time",
        "market_environment_snapshots",
        ["symbol", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("market_environment_snapshots")
    op.drop_table("discretionary_evaluations")
    op.drop_table("setup_alerts")
    op.drop_table("trader_intelligence_profiles")