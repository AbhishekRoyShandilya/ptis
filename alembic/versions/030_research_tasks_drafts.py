"""Add research_tasks + discovery_drafts tables for user-initiated continuous research

research_tasks: tracks user-initiated continuous discovery sessions
  - Each task is a long-running background job that enumerates candidate
    (symbol, pattern_type) combos and runs the discovery pipeline on each.
  - Status: pending | running | stopped | completed | failed
  - User can start/stop tasks via API.

discovery_drafts: validated strategies discovered by tasks, awaiting user decision
  - Each draft is a strategy that passed discovery validation (sample, win_rate, p-value)
  - Status: pending_review | sent_to_avt | discarded
  - User reviews each draft and clicks "Send to AVT" to graduate it to a YAML file
  - That triggers existing accept-flow → AVT picks up via hot-reload

Revision ID: 030
Revises: 029
Create Date: 2026-04-28
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ─── research_tasks ──────────────────────────────────────────────
    op.create_table(
        "research_tasks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("trader_id", sa.String(64), nullable=True, index=True),

        # Task config
        sa.Column("theme", sa.String(255), nullable=True),       # user's hint, e.g. "momentum on FNO"
        sa.Column("scope", sa.String(64), nullable=False),       # fno | nifty50 | comma-separated tickers
        sa.Column("symbols", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("pattern_types", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("lookback_years", sa.Integer, nullable=False, server_default=sa.text("2")),
        sa.Column("similarity_threshold", sa.Float, nullable=False, server_default=sa.text("0.85")),

        # Validation thresholds for "good enough" drafts
        sa.Column("min_sample_size", sa.Integer, nullable=False, server_default=sa.text("30")),
        sa.Column("min_win_rate", sa.Float, nullable=False, server_default=sa.text("0.55")),
        sa.Column("min_avg_r", sa.Float, nullable=False, server_default=sa.text("0.5")),

        # Progress
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'pending'"), index=True),
        # pending | running | stopped | completed | failed
        sa.Column("candidates_planned", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("candidates_completed", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("candidates_validated", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("current_candidate", sa.String(255), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),

        # Audit
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # ─── discovery_drafts ────────────────────────────────────────────
    op.create_table(
        "discovery_drafts",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("task_id", sa.String(64), nullable=True, index=True),
        sa.Column("trader_id", sa.String(64), nullable=True, index=True),

        # Source
        sa.Column("discovery_id", sa.String(64), nullable=True),  # links to existing discovery report
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("pattern_type", sa.String(64), nullable=False),

        # Backtest stats (for quick listing without loading full discovery report)
        sa.Column("sample_size", sa.Integer, nullable=False),
        sa.Column("win_rate", sa.Float, nullable=False),
        sa.Column("avg_r", sa.Float, nullable=False),
        sa.Column("profit_factor", sa.Float, nullable=True),
        sa.Column("best_regime", sa.String(64), nullable=True),
        sa.Column("best_time_slot", sa.String(64), nullable=True),
        sa.Column("htf_alignment", sa.String(64), nullable=True),    # from Phase 1
        sa.Column("optimal_target_r", sa.Float, nullable=True),       # from Phase 2
        sa.Column("reversal_cliff_r", sa.Float, nullable=True),       # from Phase 2

        # Strategy YAML (the deliverable)
        sa.Column("strategy_name", sa.String(255), nullable=False),
        sa.Column("strategy_yaml", sa.Text, nullable=False),
        sa.Column("strategy_dna", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),

        # Workflow
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'pending_review'"), index=True),
        # pending_review | sent_to_avt | discarded
        sa.Column("sent_to_avt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("avt_strategy_id", sa.String(64), nullable=True),  # set when sent to AVT (matches strategy YAML id)
        sa.Column("yaml_path", sa.String(512), nullable=True),       # written file location
        sa.Column("user_note", sa.Text, nullable=True),

        # Audit
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    op.create_index("ix_discovery_drafts_task_status", "discovery_drafts", ["task_id", "status"])
    op.create_index("ix_discovery_drafts_trader_status", "discovery_drafts", ["trader_id", "status"])


def downgrade() -> None:
    op.drop_index("ix_discovery_drafts_trader_status", table_name="discovery_drafts")
    op.drop_index("ix_discovery_drafts_task_status", table_name="discovery_drafts")
    op.drop_table("discovery_drafts")
    op.drop_table("research_tasks")
