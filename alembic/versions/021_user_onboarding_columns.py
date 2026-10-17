"""Add onboarding_completed + onboarding_data columns to users table

Revision ID: 021
Revises: 020
Create Date: 2026-03-23
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("onboarding_completed", sa.Boolean, server_default="false"))
    op.add_column("users", sa.Column("onboarding_data", JSONB, server_default="{}"))


def downgrade() -> None:
    op.drop_column("users", "onboarding_data")
    op.drop_column("users", "onboarding_completed")
