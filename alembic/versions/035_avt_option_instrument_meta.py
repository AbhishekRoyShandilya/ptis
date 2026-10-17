"""Add instrument_meta JSONB to avt_virtual_trades — options native in the engine (M3b)

One additive, nullable column carrying option-trade metadata {instrument, strike, expiry,
right, lots, lot_size, underlying_stop, entry_premium} so OPTION positions booked by the
AVT engine are first-class in the dashboard without a wide schema change. NULL for the
existing underlying (future/equity) trades — behaviour unchanged.

Revision ID: 035
Revises: 034
Create Date: 2026-07-11
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("avt_virtual_trades", sa.Column("instrument_meta", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("avt_virtual_trades", "instrument_meta")
