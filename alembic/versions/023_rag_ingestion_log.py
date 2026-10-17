"""RAG ingestion log table for dual storage pattern.

Revision ID: 023_rag_ingestion
Revises: 022_coplanner_tables
Create Date: 2026-04-03
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "023_rag_ingestion"
down_revision = "022_coplanner_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rag_ingestion_log",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("layer", sa.String(20), nullable=False, index=True),
        sa.Column("source", sa.String(20), nullable=False, index=True),
        sa.Column("source_id", sa.String(64), nullable=False, index=True),
        sa.Column("structured_data", JSONB, nullable=True),
        sa.Column("quality_score", sa.Float, nullable=True),
        sa.Column("chunk_count", sa.Integer, server_default="1"),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("rag_ingestion_log")
