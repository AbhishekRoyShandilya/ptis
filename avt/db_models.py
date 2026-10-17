# avt/db_models.py
"""
SQLAlchemy ORM models for AVT tables.

Maps the 5 AVT database tables to Python classes:
  - AVTVirtualTradeRow   → avt_virtual_trades
  - AVTStrategyVersionRow → avt_strategy_versions
  - AVTImprovementRow    → avt_improvements
  - AVTScanLogRow        → avt_scan_logs
  - AVTCompetitionSnapshotRow → avt_competition_snapshots
"""

import uuid

from sqlalchemy import (
    Column, String, Float, Integer, Text, DateTime, Date,
    ForeignKey, Index, Numeric,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func

from core.models import Base


# ══════════════════════════════════════════════════════════════════
#  AVT VIRTUAL TRADES
# ══════════════════════════════════════════════════════════════════

class AVTVirtualTradeRow(Base):
    """A virtual trade executed by the autonomous virtual trader."""
    __tablename__ = "avt_virtual_trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trader_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    strategy_id = Column(String(100), nullable=False)
    strategy_version = Column(String(20), nullable=False)
    variant_id = Column(UUID(as_uuid=True), nullable=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(5), nullable=False)  # long | short

    # Entry
    entry_time = Column(DateTime(timezone=True), nullable=False)
    entry_price = Column(Numeric(12, 2), nullable=False)
    quantity = Column(Integer, nullable=False)
    stop_price = Column(Numeric(12, 2), nullable=False)
    conviction_at_entry = Column(Numeric(4, 2), nullable=True)
    feature_vector = Column(JSONB, server_default="[]")
    regime_at_entry = Column(JSONB, server_default="{}")
    ranking_data = Column(JSONB, server_default="{}")

    # During-trade tracking
    max_favorable_r = Column(Numeric(6, 3), server_default="0")
    max_adverse_r = Column(Numeric(6, 3), server_default="0")
    features_at_peak = Column(JSONB, nullable=True)

    # Exit
    exit_time = Column(DateTime(timezone=True), nullable=True)
    exit_price = Column(Numeric(12, 2), nullable=True)
    exit_type = Column(String(30), nullable=True)
    exit_reason = Column(Text, nullable=True)
    pnl_r = Column(Numeric(6, 3), nullable=True)
    pnl_pct = Column(Numeric(8, 4), nullable=True)
    pnl_absolute = Column(Numeric(12, 2), nullable=True)

    # Improvement tracking
    active_improvements = Column(ARRAY(UUID(as_uuid=True)), server_default="{}")
    improvement_exits = Column(JSONB, server_default="[]")

    status = Column(String(10), server_default="open", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Options (M3b): {instrument, strike, expiry, right, lots, lot_size, underlying_stop, entry_premium};
    # NULL for underlying (future/equity) trades.
    instrument_meta = Column(JSONB, nullable=True)

    __table_args__ = (
        Index("idx_avt_trades_strategy", "strategy_id", "strategy_version"),
        Index("idx_avt_trades_date", "entry_time"),
        Index("idx_avt_trades_symbol", "symbol"),
        Index("idx_avt_trades_status", "status"),
    )


# ══════════════════════════════════════════════════════════════════
#  AVT STRATEGY VERSIONS
# ══════════════════════════════════════════════════════════════════

class AVTStrategyVersionRow(Base):
    """Tracks each version of a strategy as AVT evolves it."""
    __tablename__ = "avt_strategy_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(String(100), nullable=False, index=True)
    version = Column(String(20), nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)  # null = current
    active_improvements = Column(ARRAY(UUID(as_uuid=True)), server_default="{}")
    trade_count = Column(Integer, server_default="0")
    performance_stats = Column(JSONB, server_default="{}")
    notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_avt_versions_strategy_version", "strategy_id", "version", unique=True),
    )


# ══════════════════════════════════════════════════════════════════
#  AVT IMPROVEMENTS
# ══════════════════════════════════════════════════════════════════

class AVTImprovementRow(Base):
    """An improvement discovered, applied, and tracked by AVT."""
    __tablename__ = "avt_improvements"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(String(100), nullable=False, index=True)
    hypothesis_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    params = Column(JSONB, server_default="{}")
    status = Column(String(20), server_default="active", nullable=False)
    # active | proven | reverted | monitoring

    applied_at = Column(DateTime(timezone=True), nullable=True)
    applied_at_version = Column(String(20), nullable=True)
    validation_result = Column(JSONB, server_default="{}")

    performance_before = Column(JSONB, nullable=True)
    performance_after = Column(JSONB, nullable=True)
    evidence_strength = Column(Numeric(4, 2), server_default="0")

    reverted_at = Column(DateTime(timezone=True), nullable=True)
    revert_reason = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_avt_improvements_strategy_status", "strategy_id", "status"),
    )


# ══════════════════════════════════════════════════════════════════
#  AVT SCAN LOGS
# ══════════════════════════════════════════════════════════════════

class AVTScanLogRow(Base):
    """Record of one AVT scan cycle — transparency into AVT decisions."""
    __tablename__ = "avt_scan_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(String(100), nullable=True)
    cycle_time = Column(DateTime(timezone=True), server_default=func.now())
    candidates_found = Column(Integer, server_default="0")
    trades_opened = Column(Integer, server_default="0")
    trades_closed = Column(Integer, server_default="0")
    skipped = Column(JSONB, server_default="[]")       # [{symbol, reason}]
    top_ranked = Column(JSONB, server_default="[]")     # top 10 with scores
    duration_ms = Column(Integer, server_default="0")

    __table_args__ = (
        Index("idx_avt_scan_logs_time", "cycle_time"),
    )


# ══════════════════════════════════════════════════════════════════
#  AVT COMPETITION SNAPSHOTS
# ══════════════════════════════════════════════════════════════════

class AVTCompetitionSnapshotRow(Base):
    """Periodic snapshot of AVT vs user performance comparison."""
    __tablename__ = "avt_competition_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(String(100), nullable=False, index=True)
    period_days = Column(Integer, nullable=False)
    snapshot_date = Column(Date, nullable=False)
    user_stats = Column(JSONB, server_default="{}")
    avt_stats = Column(JSONB, server_default="{}")
    performance_gap = Column(Numeric(6, 3), nullable=True)
    differences = Column(JSONB, server_default="[]")
    proven_improvements = Column(JSONB, server_default="[]")

    __table_args__ = (
        Index("idx_avt_competition_strategy_date", "strategy_id", "snapshot_date"),
    )


# ══════════════════════════════════════════════════════════════════
#  AVT STRATEGY LIFECYCLE
# ══════════════════════════════════════════════════════════════════

class AVTStrategyLifecycleRow(Base):
    """Tracks lifecycle state for each strategy managed by AVT."""
    __tablename__ = "avt_strategy_lifecycle"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(String(100), nullable=False, unique=True)
    state = Column(String(20), server_default="ingested", nullable=False)
    grade = Column(String(20), nullable=True)  # ready | probation | rejected

    # Health gate results
    health_score = Column(Numeric(5, 2), server_default="0")
    structural_score = Column(Numeric(5, 2), server_default="0")
    coherence_score = Column(Numeric(5, 2), server_default="0")
    health_issues = Column(JSONB, server_default="[]")

    # Calibration
    calibration_started_at = Column(DateTime(timezone=True), nullable=True)
    calibration_alignment = Column(Numeric(5, 3), nullable=True)
    calibration_comparisons = Column(Integer, server_default="0")
    calibration_report = Column(JSONB, server_default="{}")

    # Baseline metrics (established from history, calibration, or probation)
    baseline_source = Column(String(30), nullable=True)
    baseline_metrics = Column(JSONB, server_default="{}")

    # Edge monitoring
    degradation_level = Column(String(20), server_default="healthy")
    monitor_action = Column(String(20), server_default="continue")
    rolling_expectancy = Column(Numeric(8, 4), nullable=True)
    rolling_win_rate = Column(Numeric(5, 4), nullable=True)
    translation_drift = Column(Numeric(5, 3), nullable=True)

    # Probation tracking
    probation_trade_count = Column(Integer, server_default="0")
    allocation_multiplier = Column(Numeric(4, 2), server_default="0")

    # Timestamps
    health_checked_at = Column(DateTime(timezone=True), nullable=True)
    graduated_at = Column(DateTime(timezone=True), nullable=True)
    paused_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_lifecycle_state", "state"),
        Index("idx_lifecycle_strategy", "strategy_id", unique=True),
    )


# ══════════════════════════════════════════════════════════════════
#  AVT CALIBRATION SIGNALS
# ══════════════════════════════════════════════════════════════════

class AVTCalibrationSignalRow(Base):
    """AVT signals recorded during calibration for alignment scoring."""
    __tablename__ = "avt_calibration_signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(5), nullable=False)
    signal_time = Column(DateTime(timezone=True), nullable=False)
    conviction = Column(Numeric(4, 2), nullable=True)
    signal_details = Column(JSONB, server_default="{}")

    # Matching
    matched_trade_id = Column(UUID(as_uuid=True), nullable=True)
    match_type = Column(String(20), nullable=True)  # true_positive | false_positive | false_negative
    user_annotation = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_calibration_signals_strategy", "strategy_id"),
        Index("idx_calibration_signals_time", "signal_time"),
    )
