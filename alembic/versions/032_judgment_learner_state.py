"""Add judgment_learner_state table — persist the user's learned judgment models

Stores the raw accept/reject decision history (features + labels) per
(strategy_id, judgment_name). The Bayesian tables and GBM are rebuilt from
these decisions on load, so the user's reverse-engineered judgment survives
server restarts AND is reconstructable if the model implementation changes.

Mirrors avt_learner_state (migration 028) but is JSONB-only: we replay the
raw decisions rather than load a pickled estimator.

Revision ID: 032
Revises: 031
Create Date: 2026-06-01
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "judgment_learner_state",
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("judgment_name", sa.String(64), nullable=False),
        # The micro-feature vocabulary this learner was trained on
        sa.Column("micro_features", JSONB, nullable=True),
        # Raw decision history: {"X": [ {feature: value}, ... ], "y": [0|1, ...]}
        sa.Column("decisions", JSONB, nullable=True),
        sa.Column("total_samples", sa.Integer, nullable=True),
        sa.Column("model_score", sa.Float, nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("strategy_id", "judgment_name"),
    )
    op.create_index(
        "ix_judgment_learner_state_strategy",
        "judgment_learner_state",
        ["strategy_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_judgment_learner_state_strategy",
        table_name="judgment_learner_state",
    )
    op.drop_table("judgment_learner_state")
