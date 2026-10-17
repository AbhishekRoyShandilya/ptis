"""improvement lab tables

Revision ID: 017
Revises: 016
Create Date: 2026-03-21
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # improvement_records — core hypothesis tracking
    op.create_table(
        "improvement_records",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("strategy_id", sa.String(50), nullable=False, index=True),
        sa.Column("improvement_type", sa.String(30), nullable=False),
        sa.Column("analysis_layer", sa.String(20), nullable=False, server_default="pattern"),
        sa.Column("pattern_type", sa.String(20), nullable=False, server_default="rule_based"),
        sa.Column("detection_method", sa.String(30), nullable=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("expected_impact_r", sa.Float, server_default="0.0"),
        sa.Column("confidence", sa.Float, server_default="0.0"),
        sa.Column("evidence", JSONB, server_default="{}"),
        sa.Column("suggested_action", sa.Text, nullable=True),
        sa.Column("proposed_rule", JSONB, nullable=True),
        sa.Column("shap_explanation", JSONB, nullable=True),
        # Variant
        sa.Column("variant_id", sa.String(50), nullable=True, index=True),
        sa.Column("applies_to", sa.String(20), server_default="all_variants"),
        sa.Column("variant_concentration", JSONB, nullable=True),
        # Behavioral
        sa.Column("behavioral_classification", JSONB, nullable=True),
        # Tradeoff
        sa.Column("tradeoff", JSONB, nullable=True),
        # Lifecycle
        sa.Column("status", sa.String(25), nullable=False, server_default="detected", index=True),
        sa.Column("trade_ids", JSONB, server_default="[]"),
        sa.Column("shadow_validation_id", UUID(as_uuid=True), nullable=True),
        sa.Column("validation_results", JSONB, nullable=True),
        # Suggestion tracking
        sa.Column("times_suggested", sa.Integer, server_default="0"),
        sa.Column("times_accepted", sa.Integer, server_default="0"),
        sa.Column("times_ignored", sa.Integer, server_default="0"),
        sa.Column("r_when_accepted", sa.Float, server_default="0.0"),
        sa.Column("r_when_ignored", sa.Float, server_default="0.0"),
        # Trader response
        sa.Column("trader_response", sa.String(20), nullable=True),
        sa.Column("trader_rejection_reason", sa.Text, nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("insight_presented_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("graduated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_improvement_strategy_status",
        "improvement_records",
        ["strategy_id", "status"],
    )
    op.create_index(
        "ix_improvement_variant",
        "improvement_records",
        ["strategy_id", "variant_id"],
    )

    # shadow_validations — A/B test tracking
    op.create_table(
        "shadow_validations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "improvement_id", UUID(as_uuid=True),
            sa.ForeignKey("improvement_records.id"),
            nullable=False, index=True,
        ),
        sa.Column("strategy_id", sa.String(50), nullable=False),
        sa.Column("improvement_rules", JSONB, nullable=False),
        sa.Column("min_trades", sa.Integer, server_default="50"),
        sa.Column("min_weeks", sa.Integer, server_default="4"),
        sa.Column("trades_tested", sa.Integer, server_default="0"),
        sa.Column("trades_helped", sa.Integer, server_default="0"),
        sa.Column("trades_hurt", sa.Integer, server_default="0"),
        sa.Column("total_r_saved", sa.Float, server_default="0.0"),
        sa.Column("total_r_lost", sa.Float, server_default="0.0"),
        sa.Column("trade_results", JSONB, server_default="[]"),
        sa.Column("shadow_r_with", sa.Float, server_default="0.0"),
        sa.Column("shadow_r_without", sa.Float, server_default="0.0"),
        sa.Column("p_value", sa.Float, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("recommendation", sa.String(20), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # graduated_rules — continuous monitoring of graduated improvements
    op.create_table(
        "graduated_rules",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "improvement_id", UUID(as_uuid=True),
            sa.ForeignKey("improvement_records.id"),
            nullable=False, unique=True,
        ),
        sa.Column("strategy_id", sa.String(50), nullable=False, index=True),
        sa.Column("variant_id", sa.String(50), nullable=True),
        sa.Column("rule_description", sa.Text, nullable=False),
        sa.Column("rule_config", JSONB, nullable=False),
        sa.Column("status", sa.String(20), server_default="monitoring"),
        sa.Column("recent_r_contribution", sa.Float, server_default="0.0"),
        sa.Column("consecutive_neutral_checks", sa.Integer, server_default="0"),
        sa.Column("total_trades_applied", sa.Integer, server_default="0"),
        sa.Column("total_r_contributed", sa.Float, server_default="0.0"),
        sa.Column("graduated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("degrading_since", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
    )

    # suggestion_logs — real-time suggestion tracking
    op.create_table(
        "suggestion_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "improvement_id", UUID(as_uuid=True),
            sa.ForeignKey("improvement_records.id"),
            nullable=False, index=True,
        ),
        sa.Column("trade_id", UUID(as_uuid=True), nullable=True),
        sa.Column("strategy_id", sa.String(50), nullable=False),
        sa.Column("suggestion_text", sa.Text, nullable=False),
        sa.Column("suggested_action", sa.String(30), nullable=False),
        sa.Column("probability", sa.Float, nullable=True),
        sa.Column("trader_response", sa.String(20), nullable=True),
        sa.Column("outcome_r", sa.Float, nullable=True),
        sa.Column("would_have_helped", sa.Boolean, nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("suggestion_logs")
    op.drop_table("graduated_rules")
    op.drop_table("shadow_validations")
    op.drop_index("ix_improvement_variant", table_name="improvement_records")
    op.drop_index("ix_improvement_strategy_status", table_name="improvement_records")
    op.drop_table("improvement_records")
