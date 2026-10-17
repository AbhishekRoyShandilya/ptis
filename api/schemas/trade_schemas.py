# trade_schemas.py
# api/schemas/trade_schemas.py
"""Pydantic schemas for trade endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class TradeEntryRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    direction: str = Field(..., pattern="^(long|short)$")
    entry_price: float = Field(..., gt=0)
    position_size: float = Field(..., gt=0)

    entry_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    declared_strategy_id: Optional[str] = None
    emotion_pre: Optional[str] = None
    conviction_score: Optional[float] = Field(None, ge=1, le=10)

    notes: Optional[str] = None
    account_balance: Optional[float] = Field(None, gt=0)

    source: str = "manual"
    broker: Optional[str] = None
    external_id: Optional[str] = None

    model_config = {"json_schema_extra": {
        "example": {
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 178.50,
            "position_size": 100,
            "stop_loss": 176.80,
            "take_profit": 182.00,
            "declared_strategy_id": "opening_breakout",
            "emotion_pre": "confident",
            "conviction_score": 7.5,
            "account_balance": 50000.0,
        }
    }}


class TradeCloseRequest(BaseModel):
    exit_price: float = Field(..., gt=0)
    exit_time: Optional[datetime] = None
    emotion_post: Optional[str] = None
    exit_notes: Optional[str] = None


class AttributionResponse(BaseModel):
    tier1: dict[str, Any] = {}
    tier2: dict[str, Any] = {}
    tier3: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    insights: list[str] = []


class TradeEntryResponse(BaseModel):
    success: bool
    trade_id: Optional[str] = None
    attribution: Optional[AttributionResponse] = None
    shadow_trade_ids: list[str] = []
    insights: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []


class TradeCloseResponse(BaseModel):
    success: bool
    trade_id: Optional[str] = None
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None
    pnl_pct: Optional[float] = None
    error: Optional[str] = None


class TradeDetailResponse(BaseModel):
    id: str
    symbol: str
    direction: str
    status: str
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime
    exit_time: Optional[datetime] = None
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: Optional[float] = None            # NET P&L (after charges)
    pnl_pct: Optional[float] = None
    r_multiple: Optional[float] = None
    commission: Optional[float] = None     # brokerage + taxes (estimated)
    risk_amount: Optional[float] = None
    risk_pct: Optional[float] = None
    session_type: Optional[str] = None
    market_regime: Optional[str] = None
    declared_strategy_id: Optional[str] = None
    emotion_pre: Optional[str] = None
    emotion_post: Optional[str] = None
    conviction_score: Optional[float] = None
    notes: Optional[str] = None
    source: str = "manual"

    # Attribution summary
    resolved_strategy: Optional[str] = None
    discipline_score: Optional[float] = None
    confidence_score: Optional[float] = None
    override_count: int = 0

    model_config = {"from_attributes": True}