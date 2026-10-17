"""Add enrichment columns to sko_variants table.

Adds 8 columns to match VariantInfo Pydantic model in ske/models.py:
narrative, regime_affinity, scoring_overrides, provenance,
taught_examples_count, tags, preferred_timeframes, confidence_level.

Revision ID: 015
Revises: 014
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("sko_variants", sa.Column("narrative", sa.Text(), nullable=True))
    op.add_column("sko_variants", sa.Column("regime_affinity", JSONB(), nullable=True))
    op.add_column("sko_variants", sa.Column("scoring_overrides", JSONB(), nullable=True))
    op.add_column(
        "sko_variants",
        sa.Column("provenance", sa.String(20), nullable=False, server_default="system"),
    )
    op.add_column(
        "sko_variants",
        sa.Column("taught_examples_count", sa.Integer(), server_default="0"),
    )
    op.add_column("sko_variants", sa.Column("tags", JSONB(), nullable=True))
    op.add_column("sko_variants", sa.Column("preferred_timeframes", JSONB(), nullable=True))
    op.add_column(
        "sko_variants",
        sa.Column("confidence_level", sa.Float(), server_default="0.0"),
    )
    op.create_index("ix_sko_variant_provenance", "sko_variants", ["provenance"])


def downgrade() -> None:
    op.drop_index("ix_sko_variant_provenance", table_name="sko_variants")
    op.drop_column("sko_variants", "confidence_level")
    op.drop_column("sko_variants", "preferred_timeframes")
    op.drop_column("sko_variants", "tags")
    op.drop_column("sko_variants", "taught_examples_count")
    op.drop_column("sko_variants", "provenance")
    op.drop_column("sko_variants", "scoring_overrides")
    op.drop_column("sko_variants", "regime_affinity")
    op.drop_column("sko_variants", "narrative")
