"""Add order_intents and system_flags tables for execution durability and kill switch

order_intents: every broker order is journaled here BEFORE submission (idempotency, audit trail)
system_flags: operator key/value store — kill_switch=active blocks all live order submission

Revision ID: 027
Revises: 026
Create Date: 2026-04-18
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "027"
down_revision = "026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── order_intents ────────────────────────────────────────────
    op.create_table(
        "order_intents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        # Order snapshot
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("direction", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Integer, nullable=False),
        sa.Column("order_type", sa.String(20), nullable=False),
        sa.Column("price", sa.Float, nullable=True),
        sa.Column("product", sa.String(20), nullable=True),
        sa.Column("tag", sa.String(100), nullable=True),
        sa.Column("trade_id", sa.String(100), nullable=True),
        # Execution context
        sa.Column("broker_name", sa.String(50), nullable=False, server_default="paper"),
        sa.Column("is_live", sa.Boolean, server_default="false"),
        sa.Column("request_payload", JSONB, server_default="{}"),
        # Lifecycle
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("broker_order_id", sa.String(100), nullable=True),
        sa.Column("response_payload", JSONB, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_unique_constraint("uq_order_intents_idempotency_key", "order_intents", ["idempotency_key"])
    op.create_index("ix_order_intents_symbol_created", "order_intents", ["symbol", "created_at"])
    op.create_index("ix_order_intents_status", "order_intents", ["status"])
    op.create_index("ix_order_intents_trade_id", "order_intents", ["trade_id"])

    # ── system_flags ─────────────────────────────────────────────
    op.create_table(
        "system_flags",
        sa.Column("key", sa.String(100), primary_key=True),
        sa.Column("value", sa.Text, nullable=False, server_default=""),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("system_flags")
    op.drop_index("ix_order_intents_trade_id", table_name="order_intents")
    op.drop_index("ix_order_intents_status", table_name="order_intents")
    op.drop_index("ix_order_intents_symbol_created", table_name="order_intents")
    op.drop_constraint("uq_order_intents_idempotency_key", "order_intents", type_="unique")
    op.drop_table("order_intents")
