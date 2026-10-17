# api/schemas/avt_schemas.py
"""Pydantic schemas for AVT API endpoints."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════
#  STATUS / CONTROL
# ══════════════════════════════════════════════════════════════════

class AVTStatusResponse(BaseModel):
    status: str                     # idle | scanning | running | stopped | error
    uptime_seconds: float = 0.0
    current_cycle: int = 0
    open_positions: int = 0
    total_trades: int = 0
    portfolio_equity: float = 0.0
    portfolio_drawdown_pct: float = 0.0
    active_strategies: List[str] = []
    last_scan_at: Optional[datetime] = None
    errors: List[str] = []

    # Capital sleeves (B.1): per-pool capital / equity / free / open slots (futures / options / equity run off
    # separate capital). One implicit "all" pool when the account isn't partitioned.
    capital_sleeves: List[Dict[str, Any]] = []
    partitioned: bool = False


class AVTStartRequest(BaseModel):
    strategies: List[str] = Field(
        default=[],
        description="Strategy IDs to trade. Empty = all loaded strategies.",
    )
    virtual_capital: Optional[float] = Field(
        None, gt=0,
        description="Override default virtual capital.",
    )


class AVTStartResponse(BaseModel):
    status: str
    message: str
    strategies_loaded: int = 0


class AVTStopResponse(BaseModel):
    status: str
    message: str
    open_positions_at_stop: int = 0


# ══════════════════════════════════════════════════════════════════
#  TRADES
# ══════════════════════════════════════════════════════════════════

class VirtualTradeResponse(BaseModel):
    id: UUID
    strategy_id: str            # system templates use a string id (e.g. "CASCADE-OPTIONS-V1-long"); taught = UUID str
    strategy_version: str
    variant_id: Optional[UUID] = None
    symbol: str
    direction: str

    entry_time: datetime
    entry_price: float
    quantity: int
    stop_price: float
    conviction_at_entry: float = 0.0
    regime_at_entry: Dict[str, Any] = {}

    ranking_position: int = 0
    ranking_score: float = 0.0

    max_favorable_r: float = 0.0

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_type: Optional[str] = None
    exit_reason: Optional[str] = None

    pnl_r: Optional[float] = None
    pnl_pct: Optional[float] = None
    pnl_absolute: Optional[float] = None

    active_improvements: List[UUID] = []
    improvement_exits: List[Dict[str, Any]] = []
    status: str = "open"

    # Options (M3b): first-class option trade fields (null for underlying trades)
    instrument: str = "future"
    tradeable: Optional[str] = None
    option_strike: Optional[float] = None
    option_expiry: Optional[str] = None
    option_right: Optional[str] = None
    option_lots: Optional[int] = None
    option_lot_size: Optional[int] = None
    underlying_stop: Optional[float] = None


class VirtualTradeListResponse(BaseModel):
    trades: List[VirtualTradeResponse]
    total: int


# ══════════════════════════════════════════════════════════════════
#  IMPROVEMENTS
# ══════════════════════════════════════════════════════════════════

class ImprovementResponse(BaseModel):
    id: UUID
    strategy_id: UUID
    type: str
    description: str
    params: Dict[str, Any] = {}
    status: str

    applied_at: Optional[datetime] = None
    applied_at_version: Optional[str] = None
    evidence_strength: float = 0.0

    performance_before: Optional[Dict[str, Any]] = None
    performance_after: Optional[Dict[str, Any]] = None
    validation_result: Dict[str, Any] = {}

    reverted_at: Optional[datetime] = None
    revert_reason: Optional[str] = None


class ImprovementListResponse(BaseModel):
    improvements: List[ImprovementResponse]
    total: int


class AdoptImprovementRequest(BaseModel):
    notes: str = ""


class AdoptImprovementResponse(BaseModel):
    improvement_id: UUID
    status: str
    message: str


# ══════════════════════════════════════════════════════════════════
#  COMPETITION
# ══════════════════════════════════════════════════════════════════

class CompetitionReportResponse(BaseModel):
    strategy_name: str
    period_days: int
    user_stats: Dict[str, Any]
    avt_stats: Dict[str, Any]
    performance_gap: float
    differences: List[Dict[str, Any]] = []
    proven_improvements: List[Dict[str, Any]] = []
    overall_narrative: str = ""


class CompetitionHistoryEntry(BaseModel):
    snapshot_date: date
    period_days: int
    performance_gap: Optional[float] = None
    user_stats: Dict[str, Any] = {}
    avt_stats: Dict[str, Any] = {}


class CompetitionHistoryResponse(BaseModel):
    strategy_id: UUID
    snapshots: List[CompetitionHistoryEntry]


# ══════════════════════════════════════════════════════════════════
#  VERSIONS
# ══════════════════════════════════════════════════════════════════

class StrategyVersionResponse(BaseModel):
    id: UUID
    version: str
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    active_improvements: List[UUID] = []
    trade_count: int = 0
    performance_stats: Dict[str, Any] = {}
    notes: Optional[str] = None


class VersionHistoryResponse(BaseModel):
    strategy_id: UUID
    versions: List[StrategyVersionResponse]


# ══════════════════════════════════════════════════════════════════
#  SCAN LOGS
# ══════════════════════════════════════════════════════════════════

class ScanLogEntry(BaseModel):
    id: UUID
    strategy_id: Optional[UUID] = None
    cycle_time: datetime
    stocks_scanned: int = 0
    candidates_found: int = 0
    ranked_count: int = 0
    filtered_count: int = 0
    trades_opened: int = 0
    trades_closed: int = 0
    skipped: List[Dict[str, Any]] = []
    top_ranked: List[Dict[str, Any]] = []
    duration_ms: int = 0


class ScanLogListResponse(BaseModel):
    logs: List[ScanLogEntry]
    total: int


# ══════════════════════════════════════════════════════════════════
#  PRESENTATION
# ══════════════════════════════════════════════════════════════════

class ImprovementPresentationResponse(BaseModel):
    title: str
    narrative: str
    summary_stats: Dict[str, Any] = {}
    examples: List[Dict[str, Any]] = []
    user_action_options: List[str] = []
    avt_confidence: float = 0.0
