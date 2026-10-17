# alembic/versions/012_watchlist_feedback_tables.py
"""Create watchlists, watchlist_items, and trade_feedback tables."""

revision = "012"
down_revision = "011"

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


def upgrade():
    # ── Watchlists ──
    op.create_table(
        "watchlists",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("watchlist_type", sa.String(20), nullable=False, server_default="master"),
        sa.Column("strategy_profile_id", UUID(as_uuid=True), sa.ForeignKey("strategy_profiles.id"), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("auto_scan", sa.Boolean, server_default="true"),
        sa.Column("last_scanned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_watchlist_type", "watchlists", ["watchlist_type"])
    op.create_index("ix_watchlist_strategy", "watchlists", ["strategy_profile_id"])

    # ── Watchlist Items ──
    op.create_table(
        "watchlist_items",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("watchlist_id", UUID(as_uuid=True), sa.ForeignKey("watchlists.id", ondelete="CASCADE"), nullable=False),
        sa.Column("symbol", sa.String(30), nullable=False),
        sa.Column("added_by", sa.String(20), server_default="user"),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("match_score", sa.Float, nullable=True),
        sa.Column("match_reason", sa.Text, nullable=True),
        sa.Column("last_analysis", JSONB, server_default="{}"),
        sa.Column("regime_at_add", sa.String(30), nullable=True),
        sa.Column("source_watchlist_id", UUID(as_uuid=True), nullable=True),
        sa.Column("source_item_id", UUID(as_uuid=True), nullable=True),
        sa.Column("added_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("watchlist_id", "symbol", name="uq_watchlist_symbol"),
    )
    op.create_index("ix_watchlist_item_symbol", "watchlist_items", ["symbol"])
    op.create_index("ix_watchlist_item_status", "watchlist_items", ["status"])

    # ── Trade Feedback ──
    op.create_table(
        "trade_feedback",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trade_id", UUID(as_uuid=True), sa.ForeignKey("trades.id"), nullable=False),
        sa.Column("strategy_profile_id", UUID(as_uuid=True), sa.ForeignKey("strategy_profiles.id"), nullable=False),
        sa.Column("adherence_score", sa.Float, nullable=True),
        sa.Column("deviations", JSONB, server_default="[]"),
        sa.Column("strengths", JSONB, server_default="[]"),
        sa.Column("lessons_learned", JSONB, server_default="[]"),
        sa.Column("outcome_alignment", sa.String(20), nullable=True),
        sa.Column("pnl", sa.Float, nullable=True),
        sa.Column("r_multiple", sa.Float, nullable=True),
        sa.Column("was_profitable", sa.Boolean, nullable=True),
        sa.Column("profile_updated", sa.Boolean, server_default="false"),
        sa.Column("confidence_delta", JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_trade_feedback_trade", "trade_feedback", ["trade_id"])
    op.create_index("ix_trade_feedback_profile", "trade_feedback", ["strategy_profile_id"])


def downgrade():
    op.drop_table("trade_feedback")
    op.drop_table("watchlist_items")
    op.drop_table("watchlists")
