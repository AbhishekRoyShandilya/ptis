"""Add trade_replays + behavior_profiles tables for Behavioral Agent

trade_replays: stores per-trade replay output (one row per trade_id)
  - Reproducible reconstruction of market state at entry/exit
  - JSONB payload includes snapshots, observations, excursion path
  - Tier (full|standard|sample) drives how rich the data is

behavior_profiles: per-trader rolling behavioral synthesis
  - Top 1-3 patterns (strengths + weaknesses) — never lecture
  - Best strategy / worst pattern
  - Discipline trend
  - Last analyzed timestamp for incremental processing

Revision ID: 031
Revises: 030
Create Date: 2026-05-02
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ─── trade_replays ──────────────────────────────────────────────
    op.create_table(
        "trade_replays",
        sa.Column("id", sa.String(64), primary_key=True),  # str(trade_id) for portability
        sa.Column("trade_id", UUID(as_uuid=True), sa.ForeignKey("trades.id", ondelete="CASCADE"),
                  nullable=False, unique=True, index=True),

        # Tier of replay (drives how much data is in payload)
        sa.Column("tier", sa.String(16), nullable=False, server_default=sa.text("'standard'")),
        # full | standard | sample

        # Quick-access fields (denormalized from payload for fast queries)
        sa.Column("symbol", sa.String(32), nullable=False, index=True),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),

        sa.Column("regime_at_entry", sa.String(40), nullable=True),
        sa.Column("regime_at_exit", sa.String(40), nullable=True),
        sa.Column("trend_alignment", sa.String(20), nullable=True),  # with_trend | against_trend | no_trend

        sa.Column("peak_r", sa.Float, nullable=True),
        sa.Column("final_r", sa.Float, nullable=True),
        sa.Column("early_exit_severity", sa.Float, nullable=True),
        sa.Column("move_pct_complete_at_entry", sa.Float, nullable=True),
        sa.Column("rsi_at_entry", sa.Float, nullable=True),

        # Full replay payload (snapshots, excursion, observations) as JSONB
        sa.Column("payload", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),

        sa.Column("candles_fetched", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("fetch_errors", JSONB, nullable=True),

        sa.Column("replayed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )
    op.create_index("ix_trade_replays_symbol_entry", "trade_replays", ["symbol", "entry_time"])

    # ─── behavior_profiles ──────────────────────────────────────────
    op.create_table(
        "behavior_profiles",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("trader_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"),
                  nullable=False, unique=True, index=True),

        # Synthesis output (capped to 1-3 items each per "no lecture" rule)
        sa.Column("strengths", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("weaknesses", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("biggest_issue", JSONB, nullable=True),
        # {pattern, frequency, impact_text, sample_size, ...}

        sa.Column("best_strategy", sa.String(64), nullable=True),
        sa.Column("worst_pattern", sa.String(64), nullable=True),

        sa.Column("discipline_score", sa.Float, nullable=True),
        sa.Column("causal_summary", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        # Top 3 cause-tag patterns with frequencies

        # Cluster archetype from ml/behavior_clustering.py (if computed)
        sa.Column("archetype", sa.String(64), nullable=True),

        # Counts + processing state
        sa.Column("trades_analyzed_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("first_trade_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_trade_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_analyzed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_full_recompute_at", sa.DateTime(timezone=True), nullable=True),

        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_index("ix_trade_replays_symbol_entry", table_name="trade_replays")
    op.drop_table("trade_replays")
    op.drop_table("behavior_profiles")
