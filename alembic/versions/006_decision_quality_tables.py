"""
006 — Decision quality evaluation tables.

Revision ID: 006
Revises: 005
"""

from alembic import op
import sqlalchemy as sa

revision = "006_decision_quality_tables"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Main evaluation table ───────────────────────────────────
    op.create_table(
        "decision_quality_evaluations",
        sa.Column(
            "evaluation_id",
            sa.String(36),
            primary_key=True,
        ),
        sa.Column("trade_id", sa.String(36), nullable=True, index=True),
        sa.Column(
            "evaluated_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("final_score", sa.Float, nullable=False),
        sa.Column("grade", sa.String(20), nullable=False),
        sa.Column("data_completeness", sa.Float, nullable=False, default=0.0),
        sa.Column(
            "outcome_independent",
            sa.Boolean,
            nullable=False,
            default=True,
        ),
        sa.Column("raw_result", sa.Text, nullable=True),
    )

    op.create_index(
        "ix_dqe_evaluated_at",
        "decision_quality_evaluations",
        ["evaluated_at"],
    )
    op.create_index(
        "ix_dqe_grade",
        "decision_quality_evaluations",
        ["grade"],
    )

    # ── Dimension scores ────────────────────────────────────────
    op.create_table(
        "decision_quality_dimensions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "evaluation_id",
            sa.String(36),
            sa.ForeignKey(
                "decision_quality_evaluations.evaluation_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("dimension_name", sa.String(30), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("weight", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("components", sa.Text, nullable=True),
        sa.Column("reasoning", sa.Text, nullable=True),
        sa.UniqueConstraint(
            "evaluation_id", "dimension_name",
            name="uq_eval_dimension",
        ),
    )

    op.create_index(
        "ix_dqd_evaluation_id",
        "decision_quality_dimensions",
        ["evaluation_id"],
    )

    # ── Insights ────────────────────────────────────────────────
    op.create_table(
        "decision_quality_insights",
        sa.Column("insight_id", sa.String(36), primary_key=True),
        sa.Column(
            "evaluation_id",
            sa.String(36),
            sa.ForeignKey(
                "decision_quality_evaluations.evaluation_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("dimension_name", sa.String(30), nullable=False),
        sa.Column("severity", sa.String(15), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("suggestion", sa.Text, nullable=False),
        sa.Column("score_impact", sa.Float, nullable=False, default=0.0),
    )

    op.create_index(
        "ix_dqi_evaluation_id",
        "decision_quality_insights",
        ["evaluation_id"],
    )
    op.create_index(
        "ix_dqi_severity",
        "decision_quality_insights",
        ["severity"],
    )


def downgrade() -> None:
    op.drop_table("decision_quality_insights")
    op.drop_table("decision_quality_dimensions")
    op.drop_table("decision_quality_evaluations")