# alembic/versions/013_ske_tables.py
"""
Create Strategy Knowledge Engine tables:
- strategy_knowledge_objects
- sko_variants
- sko_signals
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── strategy_knowledge_objects ──
    op.create_table(
        "strategy_knowledge_objects",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("short_code", sa.String(30), nullable=False, unique=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="hypothesis"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("narrative", sa.Text, nullable=True),
        sa.Column("tier", sa.Integer, server_default="0"),
        sa.Column("total_trades", sa.Integer, server_default="0"),
        sa.Column("win_rate", sa.Float, nullable=True),
        sa.Column("expectancy", sa.Float, nullable=True),
        sa.Column("profit_factor", sa.Float, nullable=True),
        sa.Column("scoring_config", JSONB, server_default="{}"),
        sa.Column("conviction_map", JSONB, server_default="{}"),
        sa.Column("entry_zones", JSONB, server_default="[]"),
        sa.Column("exit_rules", JSONB, server_default="{}"),
        sa.Column("risk_parameters", JSONB, server_default="{}"),
        sa.Column("regime_fingerprint", JSONB, server_default="{}"),
        sa.Column("time_preferences", JSONB, server_default="{}"),
        sa.Column("dqal_score", sa.Float, nullable=True),
        sa.Column("dqal_grade", sa.String(1), nullable=True),
        sa.Column("tags", JSONB, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_signal_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_sko_status_tier", "strategy_knowledge_objects", ["status", "tier"])
    op.create_index("ix_sko_short_code", "strategy_knowledge_objects", ["short_code"])

    # ── sko_variants ──
    op.create_table(
        "sko_variants",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("sko_id", UUID(as_uuid=True),
                  sa.ForeignKey("strategy_knowledge_objects.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("variant_type", sa.String(20), nullable=False, server_default="variant"),
        sa.Column("label", sa.String(100), nullable=False),
        sa.Column("centroid", JSONB, server_default="{}"),
        sa.Column("trade_count", sa.Integer, server_default="0"),
        sa.Column("win_rate", sa.Float, nullable=True),
        sa.Column("avg_r", sa.Float, nullable=True),
        sa.Column("expectancy", sa.Float, nullable=True),
        sa.Column("distinguishing_features", JSONB, server_default="[]"),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_sko_variant_sko", "sko_variants", ["sko_id"])
    op.create_index("ix_sko_variant_type", "sko_variants", ["variant_type"])

    # ── sko_signals ──
    op.create_table(
        "sko_signals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("sko_id", UUID(as_uuid=True),
                  sa.ForeignKey("strategy_knowledge_objects.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("variant_id", UUID(as_uuid=True),
                  sa.ForeignKey("sko_variants.id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("symbol", sa.String(30), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_vector", JSONB, server_default="{}"),
        sa.Column("individual_scores", JSONB, server_default="{}"),
        sa.Column("conviction_score", sa.Float, nullable=False),
        sa.Column("conviction_band", sa.String(20), nullable=False),
        sa.Column("sizing_recommendation", sa.String(20), nullable=True),
        sa.Column("size_multiplier", sa.Float, nullable=True),
        sa.Column("regime_fingerprint", JSONB, server_default="{}"),
        sa.Column("regime_match_score", sa.Float, nullable=True),
        sa.Column("entry_zone", JSONB, server_default="{}"),
        sa.Column("stop_level", sa.Float, nullable=True),
        sa.Column("target_levels", JSONB, server_default="[]"),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("outcome_r", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_sko_signal_sko", "sko_signals", ["sko_id"])
    op.create_index("ix_sko_signal_symbol", "sko_signals", ["symbol"])
    op.create_index("ix_sko_signal_status", "sko_signals", ["status"])
    op.create_index("ix_sko_signal_created", "sko_signals", ["created_at"])


def downgrade() -> None:
    op.drop_table("sko_signals")
    op.drop_table("sko_variants")
    op.drop_table("strategy_knowledge_objects")
