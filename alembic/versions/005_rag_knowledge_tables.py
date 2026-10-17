"""
005 — RAG knowledge tables (placeholder).

The knowledge_entries table is created in migration 009 alongside the other
tables added during the Completion Sprint. This migration exists solely to
maintain the revision chain (006 chains off 005 via down_revision = "005").

Revision ID: 005
Revises: 004
"""

from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # knowledge_entries is created in 009_journal_automation_knowledge_tables
    pass


def downgrade() -> None:
    pass
