"""Add avt_learner_state table for persisting deep ML model blobs across restarts

avt_learner_state: stores serialized GBM+RF model blobs (joblib) and Bayesian state
per (strategy_id, detector_type) pair — survives server restarts.

Revision ID: 028
Revises: 027
Create Date: 2026-04-20
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "avt_learner_state",
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("detector_type", sa.String(32), nullable=False),
        # Bayesian tables + SGD metadata (JSON-serializable, fast to load)
        sa.Column("bayesian_state", JSONB, nullable=True),
        # Joblib-serialized GBM + RF model bytes (can be several MB)
        sa.Column("model_blob", sa.LargeBinary, nullable=True),
        sa.Column("trade_count_at_train", sa.Integer, nullable=True),
        sa.Column("model_score", sa.Float, nullable=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("strategy_id", "detector_type"),
    )
    op.create_index(
        "ix_avt_learner_state_updated",
        "avt_learner_state",
        ["updated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_avt_learner_state_updated", table_name="avt_learner_state")
    op.drop_table("avt_learner_state")
