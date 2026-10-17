# alembic/versions/014_quality_gate_divergence.py
"""
Add quality gate and divergence tracking:
- New columns on strategy_knowledge_objects (divergence snapshot + quality gate config)
- New table: strategy_divergence_history
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Add divergence snapshot columns to strategy_knowledge_objects ──
    op.add_column("strategy_knowledge_objects",
                  sa.Column("filtered_win_rate", sa.Float(), nullable=True))
    op.add_column("strategy_knowledge_objects",
                  sa.Column("unfiltered_win_rate", sa.Float(), nullable=True))
    op.add_column("strategy_knowledge_objects",
                  sa.Column("filtered_expectancy_r", sa.Float(), nullable=True))
    op.add_column("strategy_knowledge_objects",
                  sa.Column("unfiltered_expectancy_r", sa.Float(), nullable=True))
    op.add_column("strategy_knowledge_objects",
                  sa.Column("behavioral_cost_per_trade_r", sa.Float(), nullable=True))
    op.add_column("strategy_knowledge_objects",
                  sa.Column("centroid_divergence_cosine", sa.Float(), nullable=True))
    op.add_column("strategy_knowledge_objects",
                  sa.Column("divergence_interpretation", sa.String(50), nullable=True))
    op.add_column("strategy_knowledge_objects",
                  sa.Column("divergence_computed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("strategy_knowledge_objects",
                  sa.Column("effective_sample_size", sa.Float(), nullable=True))
    op.add_column("strategy_knowledge_objects",
                  sa.Column("quality_gate_config", JSONB(), server_default="{}"))
    op.add_column("strategy_knowledge_objects",
                  sa.Column("temporal_decay_half_life", sa.Float(), nullable=True, server_default="120.0"))

    # ── strategy_divergence_history ──
    op.create_table(
        "strategy_divergence_history",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("sko_id", UUID(as_uuid=True),
                  sa.ForeignKey("strategy_knowledge_objects.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),

        # Raw counts
        sa.Column("raw_trade_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("filtered_trade_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("excluded_trade_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("effective_sample_size", sa.Float(), nullable=True),

        # Unfiltered model stats
        sa.Column("unfiltered_win_rate", sa.Float(), nullable=True),
        sa.Column("unfiltered_expectancy_r", sa.Float(), nullable=True),
        sa.Column("unfiltered_profit_factor", sa.Float(), nullable=True),
        sa.Column("unfiltered_avg_winner_r", sa.Float(), nullable=True),
        sa.Column("unfiltered_avg_loser_r", sa.Float(), nullable=True),
        sa.Column("unfiltered_sharpe", sa.Float(), nullable=True),

        # Filtered model stats
        sa.Column("filtered_win_rate", sa.Float(), nullable=True),
        sa.Column("filtered_expectancy_r", sa.Float(), nullable=True),
        sa.Column("filtered_profit_factor", sa.Float(), nullable=True),
        sa.Column("filtered_avg_winner_r", sa.Float(), nullable=True),
        sa.Column("filtered_avg_loser_r", sa.Float(), nullable=True),
        sa.Column("filtered_sharpe", sa.Float(), nullable=True),

        # Divergence metrics
        sa.Column("win_rate_delta", sa.Float(), nullable=True),
        sa.Column("expectancy_delta_r", sa.Float(), nullable=True),
        sa.Column("profit_factor_delta", sa.Float(), nullable=True),
        sa.Column("behavioral_cost_per_trade_r", sa.Float(), nullable=True),
        sa.Column("centroid_divergence_cosine", sa.Float(), nullable=True),

        # Interpretation
        sa.Column("divergence_interpretation", sa.String(50), nullable=True),

        # JSONB detail fields
        sa.Column("feature_divergence", JSONB(), server_default="{}"),
        sa.Column("exclusion_breakdown", JSONB(), server_default="{}"),
    )

    op.create_index("ix_divergence_sko_computed",
                    "strategy_divergence_history", ["sko_id", "computed_at"])
    op.create_index("ix_divergence_computed",
                    "strategy_divergence_history", ["computed_at"])


def downgrade() -> None:
    op.drop_table("strategy_divergence_history")

    cols = [
        "filtered_win_rate", "unfiltered_win_rate",
        "filtered_expectancy_r", "unfiltered_expectancy_r",
        "behavioral_cost_per_trade_r", "centroid_divergence_cosine",
        "divergence_interpretation", "divergence_computed_at",
        "effective_sample_size", "quality_gate_config",
        "temporal_decay_half_life",
    ]
    for col in cols:
        op.drop_column("strategy_knowledge_objects", col)
