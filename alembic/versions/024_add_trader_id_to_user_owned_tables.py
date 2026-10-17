"""Add trader_id FK to all user-owned tables for multi-user data isolation.

Tables modified:
  - strategy_profiles
  - watchlists
  - automation_rules
  - knowledge_entries
  - strategy_knowledge_objects
  - daily_performance_summaries (also removes unique constraint on date)
  - automation_readiness_scores

Also fixes Trade.trader_id to add ondelete="SET NULL".

Revision ID: 024_trader_id_isolation
Revises: 023_rag_ingestion
Create Date: 2026-04-04
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "024_trader_id_isolation"
down_revision = "023_rag_ingestion"
branch_labels = None
depends_on = None

# Tables that need trader_id added
TABLES = [
    "strategy_profiles",
    "watchlists",
    "automation_rules",
    "knowledge_entries",
    "strategy_knowledge_objects",
    "daily_performance_summaries",
    "automation_readiness_scores",
]


def upgrade() -> None:
    for table in TABLES:
        op.add_column(
            table,
            sa.Column("trader_id", UUID(as_uuid=True), nullable=True),
        )
        op.create_foreign_key(
            f"fk_{table}_trader_id",
            table,
            "users",
            ["trader_id"],
            ["id"],
            ondelete="CASCADE",
        )
        op.create_index(f"ix_{table}_trader_id", table, ["trader_id"])

    # daily_performance_summaries: drop unique on date (now scoped per user)
    try:
        op.drop_constraint(
            "daily_performance_summaries_date_key",
            "daily_performance_summaries",
            type_="unique",
        )
    except Exception:
        pass  # constraint may not exist

    # Fix Trade.trader_id: add ondelete=SET NULL (drop + re-create FK)
    try:
        op.drop_constraint("trades_trader_id_fkey", "trades", type_="foreignkey")
    except Exception:
        pass
    op.create_foreign_key(
        "trades_trader_id_fkey",
        "trades",
        "users",
        ["trader_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    # Revert Trade FK
    try:
        op.drop_constraint("trades_trader_id_fkey", "trades", type_="foreignkey")
    except Exception:
        pass
    op.create_foreign_key(
        "trades_trader_id_fkey",
        "trades",
        "users",
        ["trader_id"],
        ["id"],
    )

    for table in reversed(TABLES):
        op.drop_index(f"ix_{table}_trader_id", table_name=table)
        op.drop_constraint(f"fk_{table}_trader_id", table, type_="foreignkey")
        op.drop_column(table, "trader_id")
