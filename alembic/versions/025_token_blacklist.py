"""Add token_blacklist table for JWT revocation.

Revision ID: 025_token_blacklist
Revises: 024_trader_id_isolation
Create Date: 2026-04-05
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "025_token_blacklist"
down_revision = "024_trader_id_isolation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add tokens_invalidated_at to users (for logout-all)
    op.add_column("users", sa.Column("tokens_invalidated_at", sa.DateTime(timezone=True), nullable=True))

    op.create_table(
        "token_blacklist",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("jti", sa.String(64), nullable=False, unique=True, index=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_type", sa.String(10), nullable=False, server_default="access"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    # Index for cleanup job (delete expired blacklist entries)
    op.create_index("ix_token_blacklist_expires_at", "token_blacklist", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_token_blacklist_expires_at", table_name="token_blacklist")
    op.drop_table("token_blacklist")
    op.drop_column("users", "tokens_invalidated_at")
