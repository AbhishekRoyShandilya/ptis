"""Create users table and add trader_id to trades.

Adds the users table for DB-backed authentication, and adds a
trader_id foreign key column to the trades table so each trade
can be associated with a specific user.

Revision ID: 016
Revises: 015
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers
revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create users table
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("username", sa.String(50), unique=True, nullable=False, index=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(100), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # 2. Add trader_id column to trades table
    op.add_column(
        "trades",
        sa.Column("trader_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
    )

    # 3. Add index on trades.trader_id
    op.create_index("ix_trades_trader_id", "trades", ["trader_id"])


def downgrade() -> None:
    op.drop_index("ix_trades_trader_id", table_name="trades")
    op.drop_column("trades", "trader_id")
    op.drop_table("users")
