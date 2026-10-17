"""AVT — change strategy_id from UUID to VARCHAR(100) on all AVT tables

AVT uses strategy template names (e.g. "compression_breakout") as strategy
identifiers, not UUIDs.  The original migration incorrectly used UUID for
these columns, causing psycopg2.errors.InvalidTextRepresentation on startup.

Revision ID: 026
Revises: 025
Create Date: 2026-04-17
"""
from alembic import op
import sqlalchemy as sa

revision = "026"
down_revision = "025_token_blacklist"
branch_labels = None
depends_on = None

_TABLES = [
    # (table_name, index_names_to_drop_and_recreate, unique)
    ("avt_virtual_trades",        ["idx_avt_trades_strategy"],                False),
    ("avt_strategy_versions",     ["idx_avt_versions_strategy_version"],       True),
    ("avt_improvements",          ["idx_avt_improvements_strategy_status"],    False),
    ("avt_scan_logs",             [],                                           False),
    ("avt_competition_snapshots", [],                                           False),
    ("avt_strategy_lifecycle",    ["idx_lifecycle_strategy"],                  True),
    ("avt_calibration_signals",   ["idx_calibration_signals_strategy"],        False),
]


def upgrade() -> None:
    # Drop indexes that include strategy_id first (can't alter column with active index)
    op.drop_index("idx_avt_trades_strategy",            table_name="avt_virtual_trades")
    op.drop_index("idx_avt_versions_strategy_version",  table_name="avt_strategy_versions")
    op.drop_index("idx_avt_improvements_strategy_status", table_name="avt_improvements")
    op.drop_index("idx_lifecycle_strategy",             table_name="avt_strategy_lifecycle")
    op.drop_index("idx_calibration_signals_strategy",   table_name="avt_calibration_signals")

    # Alter each table
    for table, _, _ in _TABLES:
        op.alter_column(
            table, "strategy_id",
            type_=sa.String(100),
            postgresql_using="strategy_id::text",
        )

    # Recreate indexes
    op.create_index("idx_avt_trades_strategy",
                    "avt_virtual_trades", ["strategy_id", "strategy_version"])
    op.create_index("idx_avt_versions_strategy_version",
                    "avt_strategy_versions", ["strategy_id", "version"], unique=True)
    op.create_index("idx_avt_improvements_strategy_status",
                    "avt_improvements", ["strategy_id", "status"])
    op.create_index("idx_lifecycle_strategy",
                    "avt_strategy_lifecycle", ["strategy_id"], unique=True)
    op.create_index("idx_calibration_signals_strategy",
                    "avt_calibration_signals", ["strategy_id"])


def downgrade() -> None:
    # Reverse: drop indexes, alter back to UUID (data must be valid UUIDs), recreate
    op.drop_index("idx_avt_trades_strategy",            table_name="avt_virtual_trades")
    op.drop_index("idx_avt_versions_strategy_version",  table_name="avt_strategy_versions")
    op.drop_index("idx_avt_improvements_strategy_status", table_name="avt_improvements")
    op.drop_index("idx_lifecycle_strategy",             table_name="avt_strategy_lifecycle")
    op.drop_index("idx_calibration_signals_strategy",   table_name="avt_calibration_signals")

    from sqlalchemy.dialects.postgresql import UUID
    for table, _, _ in _TABLES:
        op.alter_column(
            table, "strategy_id",
            type_=UUID(as_uuid=True),
            postgresql_using="strategy_id::uuid",
        )

    op.create_index("idx_avt_trades_strategy",
                    "avt_virtual_trades", ["strategy_id", "strategy_version"])
    op.create_index("idx_avt_versions_strategy_version",
                    "avt_strategy_versions", ["strategy_id", "version"], unique=True)
    op.create_index("idx_avt_improvements_strategy_status",
                    "avt_improvements", ["strategy_id", "status"])
    op.create_index("idx_lifecycle_strategy",
                    "avt_strategy_lifecycle", ["strategy_id"], unique=True)
    op.create_index("idx_calibration_signals_strategy",
                    "avt_calibration_signals", ["strategy_id"])
