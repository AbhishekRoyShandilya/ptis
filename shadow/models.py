# shadow/models.py
"""
SQLAlchemy ORM models for the Shadow Trading Engine.
Tracks mirror trades, strategy shadows, variants, price snapshots,
and divergences between real and shadow executions.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, Text, DateTime,
    ForeignKey, Index, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.enums import ShadowTradeType, TradeDirection, TradeStatus
from core.models import Base


class ShadowSession(Base):
    """Groups shadow trades for a single real trade."""
    __tablename__ = "shadow_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    real_trade_id = Column(
        UUID(as_uuid=True),
        ForeignKey("trades.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    strategy_id = Column(String(50), nullable=True, index=True)
    status = Column(String(20), default="active")  # active, closed, cancelled
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    closed_at = Column(DateTime(timezone=True), nullable=True)
    summary = Column(JSONB, default=dict)

    shadow_trades = relationship(
        "ShadowTrade", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_shadow_session_trade", "real_trade_id"),
    )


class ShadowTrade(Base):
    """
    A single shadow trade — can be mirror, strategy, or variant.
    Updated in real-time via price feed.
    """
    __tablename__ = "shadow_trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("shadow_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    real_trade_id = Column(
        UUID(as_uuid=True),
        ForeignKey("trades.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Type
    shadow_type = Column(String(30), nullable=False)  # ShadowTradeType value
    label = Column(String(100), nullable=True)  # human readable label

    # Trade parameters
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime(timezone=True), nullable=False)
    position_size = Column(Float, nullable=False)

    # Risk management
    stop_loss = Column(Float, nullable=True)
    original_stop = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    trailing_stop = Column(Float, nullable=True)
    trailing_config = Column(JSONB, default=dict)

    # Current state
    status = Column(String(20), default="open")  # open, closed, stopped_out, target_hit, time_exit
    current_price = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, default=0.0)
    max_favorable = Column(Float, default=0.0)   # max favorable excursion
    max_adverse = Column(Float, default=0.0)      # max adverse excursion
    high_since_entry = Column(Float, nullable=True)
    low_since_entry = Column(Float, nullable=True)

    # Exit
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime(timezone=True), nullable=True)
    exit_reason = Column(String(50), nullable=True)  # stop_hit, target_hit, trailing, time_exit, manual

    # P&L
    realized_pnl = Column(Float, nullable=True)
    r_multiple = Column(Float, nullable=True)
    risk_amount = Column(Float, nullable=True)

    # Partial exits
    partial_exits = Column(JSONB, default=list)
    # [{price, size, r_multiple, timestamp, target_level}]
    remaining_size = Column(Float, nullable=True)

    # Metadata
    strategy_id = Column(String(50), nullable=True)
    config = Column(JSONB, default=dict)  # full config used to create this shadow
    price_updates = Column(Integer, default=0)
    last_update = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("ShadowSession", back_populates="shadow_trades")
    price_snapshots = relationship(
        "ShadowPriceSnapshot", back_populates="shadow_trade",
        cascade="all, delete-orphan",
    )
    divergences = relationship(
        "TradeDivergence", back_populates="shadow_trade",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_shadow_trade_real", "real_trade_id"),
        Index("ix_shadow_trade_type", "shadow_type"),
        Index("ix_shadow_trade_status", "status"),
    )


class ShadowPriceSnapshot(Base):
    """Periodic price snapshots for shadow trade P&L curve."""
    __tablename__ = "shadow_price_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    shadow_trade_id = Column(
        UUID(as_uuid=True),
        ForeignKey("shadow_trades.id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp = Column(DateTime(timezone=True), nullable=False)
    price = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    r_multiple = Column(Float, nullable=True)

    shadow_trade = relationship("ShadowTrade", back_populates="price_snapshots")

    __table_args__ = (
        Index("ix_snapshot_shadow_time", "shadow_trade_id", "timestamp"),
    )


class TradeDivergence(Base):
    """Records divergences between real trade and shadow trade."""
    __tablename__ = "trade_divergences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    shadow_trade_id = Column(
        UUID(as_uuid=True),
        ForeignKey("shadow_trades.id", ondelete="CASCADE"),
        nullable=False,
    )
    real_trade_id = Column(
        UUID(as_uuid=True),
        ForeignKey("trades.id", ondelete="CASCADE"),
        nullable=False,
    )

    divergence_type = Column(String(50), nullable=False)
    # entry_timing, entry_price, exit_timing, exit_price,
    # stop_placement, position_size, pnl_difference, exit_reason

    description = Column(Text, nullable=False)
    real_value = Column(String(200), nullable=True)
    shadow_value = Column(String(200), nullable=True)
    impact_amount = Column(Float, nullable=True)  # dollar impact
    impact_r = Column(Float, nullable=True)        # R-multiple impact

    detected_at = Column(DateTime(timezone=True), server_default=func.now())

    shadow_trade = relationship("ShadowTrade", back_populates="divergences")

    __table_args__ = (
        Index("ix_divergence_real", "real_trade_id"),
        Index("ix_divergence_shadow", "shadow_trade_id"),
    )