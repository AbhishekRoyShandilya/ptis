# 003_strategy_attribution_tables.py
# alembic/versions/003_strategy_attribution_tables.py
"""
Strategy attribution, condition evaluations, and override events tables.

Revision ID: 003
Revises: 002
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Strategy Attributions ──
    op.create_table(
        "strategy_attributions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trade_id", UUID(as_uuid=True), sa.ForeignKey("trades.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("tier1_strategy_id", sa.String(50), nullable=True),
        sa.Column("tier1_declared", sa.Boolean, default=False),
        sa.Column("tier2_matches", JSONB, default=[]),
        sa.Column("tier2_best_match_id", sa.String(50), nullable=True),
        sa.Column("tier2_best_match_score", sa.Float, nullable=True),
        sa.Column("tier3_override_count", sa.Integer, default=0),
        sa.Column("tier3_max_severity", sa.String(20), nullable=True),
        sa.Column("tier3_summary", JSONB, default={}),
        sa.Column("resolved_strategy_id", sa.String(50), nullable=True),
        sa.Column("resolved_tier", sa.String(30), nullable=True),
        sa.Column("confidence_score", sa.Float, nullable=True),
        sa.Column("discipline_score", sa.Float, nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("market_regime_at_entry", sa.String(20), nullable=True),
        sa.Column("evaluation_context", JSONB, default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_attr_resolved_strategy", "strategy_attributions", ["resolved_strategy_id"])
    op.create_index("ix_attr_discipline", "strategy_attributions", ["discipline_score"])

    # ── Condition Evaluations ──
    op.create_table(
        "condition_evaluations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trade_id", UUID(as_uuid=True), sa.ForeignKey("trades.id", ondelete="CASCADE"), nullable=False),
        sa.Column("strategy_id", sa.String(50), nullable=False),
        sa.Column("condition_type", sa.String(50), nullable=False),
        sa.Column("condition_params", JSONB, default={}),
        sa.Column("required", sa.Boolean, default=True),
        sa.Column("weight", sa.Float, default=1.0),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("actual_value", sa.String(200), nullable=True),
        sa.Column("expected_value", sa.String(200), nullable=True),
        sa.Column("details", sa.Text, nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_cond_eval_trade_strategy", "condition_evaluations", ["trade_id", "strategy_id"])
    op.create_index("ix_cond_eval_strategy", "condition_evaluations", ["strategy_id"])

    # ── Override Events ──
    op.create_table(
        "override_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trade_id", UUID(as_uuid=True), sa.ForeignKey("trades.id", ondelete="CASCADE"), nullable=False),
        sa.Column("attribution_id", UUID(as_uuid=True), sa.ForeignKey("strategy_attributions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("override_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("severity_score", sa.Float, nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("rule_expected", sa.Text, nullable=True),
        sa.Column("actual_value", sa.String(200), nullable=True),
        sa.Column("expected_value", sa.String(200), nullable=True),
        sa.Column("strategy_id", sa.String(50), nullable=True),
        sa.Column("context", JSONB, default={}),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_override_trade_type", "override_events", ["trade_id", "override_type"])
    op.create_index("ix_override_severity", "override_events", ["severity"])
    op.create_index("ix_override_detected", "override_events", ["detected_at"])

    # Add declared_strategy_id to trades if not present
    op.add_column("trades", sa.Column("declared_strategy_id", sa.String(50), nullable=True))
    op.create_index("ix_trades_declared_strategy", "trades", ["declared_strategy_id"])


def downgrade() -> None:
    op.drop_index("ix_trades_declared_strategy", table_name="trades")
    op.drop_column("trades", "declared_strategy_id")
    op.drop_table("override_events")
    op.drop_table("condition_evaluations")
    op.drop_table("strategy_attributions")