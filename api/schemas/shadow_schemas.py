# shadow_schemas.py
# api/schemas/shadow_schemas.py
"""Pydantic schemas for shadow trading endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ShadowTradeResponse(BaseModel):
    id: str
    shadow_type: str
    label: Optional[str] = None
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    status: str
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: Optional[float] = None
    r_multiple: Optional[float] = None
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    partial_exits: list[dict] = []
    remaining_size: Optional[float] = None
    strategy_id: Optional[str] = None
    price_updates: int = 0

    model_config = {"from_attributes": True}


class ShadowSessionResponse(BaseModel):
    session_id: str
    real_trade_id: str
    strategy_id: Optional[str] = None
    status: str
    started_at: datetime
    closed_at: Optional[datetime] = None
    shadow_trades: list[ShadowTradeResponse] = []
    summary: dict = {}


class PriceUpdateRequest(BaseModel):
    price: float = Field(..., gt=0)
    timestamp: Optional[datetime] = None


class PriceUpdateResponse(BaseModel):
    results: list[dict[str, Any]] = []


class ComparisonResponse(BaseModel):
    real_trade_id: str
    real_pnl: float
    real_r: Optional[float] = None
    comparisons: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    best_shadow: dict = {}
    opportunity_cost: float = 0.0
    opportunity_cost_r: Optional[float] = None
    execution_quality: float = 0.0
    insights: list[str] = []


class ShadowSummaryResponse(BaseModel):
    total_sessions: int = 0
    active_sessions: int = 0
    total_shadow_trades: int = 0
    avg_execution_quality: Optional[float] = None
    total_opportunity_cost: float = 0.0
    strategy_shadow_better_pct: float = 0.0
    variant_insights: list[str] = []