"""Add research_findings table for autonomous research agent (Phase 4 + 5)

research_findings: stores findings emitted by the research agent —
decay alerts, emergence alerts, regime target shifts, reversal cliffs,
and meta-findings (cross-correlation across multiple strategies).

Real-money safety: NO finding is ever auto-applied to live strategies.
Trader reviews and accepts/rejects via the /research page.

Revision ID: 029
Revises: 028
Create Date: 2026-04-26
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "029"
down_revision = "028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_findings",
        sa.Column("id", sa.String(64), primary_key=True),

        # Categorization
        sa.Column("type", sa.String(40), nullable=False, index=True),
        # type values: "decay_detected", "edge_emerging", "regime_target_shift",
        #              "reversal_cliff", "new_pattern", "regime_shift" (meta),
        #              "time_slot_risk" (meta)

        # Context
        sa.Column("strategy_id", sa.String(64), nullable=True, index=True),
        sa.Column("symbol", sa.String(32), nullable=True),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False, index=True),

        # Statistics
        sa.Column("confidence", sa.String(16), nullable=False),  # high|moderate|low
        sa.Column("p_value", sa.Float, nullable=True),
        sa.Column("sample_size", sa.Integer, nullable=True),

        # Human-readable
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("rationale", sa.Text, nullable=True),
        sa.Column("suggested_action", sa.Text, nullable=True),

        # Full finding payload (JSON of detector-specific fields)
        sa.Column("payload", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),

        # Workflow status
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'pending_review'"), index=True),
        # status values: "pending_review", "accepted", "rejected", "in_proof_cycle", "proven", "expired"

        sa.Column("status_history", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        # appended on every status change: [{"from": "...", "to": "...", "at": "...", "by": "...", "note": "..."}]

        # Cross-correlation linkage (for meta-findings)
        sa.Column("related_finding_ids", JSONB, nullable=True),

        # Audit
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Composite index for the most common query: pending findings for a strategy
    op.create_index(
        "ix_research_findings_strategy_status",
        "research_findings",
        ["strategy_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_research_findings_strategy_status", table_name="research_findings")
    op.drop_table("research_findings")
