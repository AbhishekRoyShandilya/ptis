# api/schemas/dashboard_schemas.py
"""
Dashboard Pydantic schemas for API request/response validation.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DailySummary(BaseModel):
    date: date
    trades_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    max_drawdown: float = 0.0
    discipline_score: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0


class WeeklyStats(BaseModel):
    week_start: date
    week_end: date
    total_trades: int = 0
    net_pnl: float = 0.0
    win_rate: float = 0.0
    avg_r: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    discipline_avg: float = 0.0
    best_day: Optional[DailySummary] = None
    worst_day: Optional[DailySummary] = None


class StrategyBreakdown(BaseModel):
    strategy_id: str
    strategy_name: str
    trades: int = 0
    net_pnl: float = 0.0
    win_rate: float = 0.0
    avg_r: float = 0.0
    contribution_pct: float = 0.0


class EquityCurvePoint(BaseModel):
    timestamp: datetime
    equity: float
    drawdown: float = 0.0
    drawdown_pct: float = 0.0


class DashboardResponse(BaseModel):
    """Main dashboard data."""
    as_of: datetime
    daily: DailySummary
    weekly: WeeklyStats
    monthly_pnl: float = 0.0
    equity_curve: List[EquityCurvePoint] = []
    strategies: List[StrategyBreakdown] = []
    open_trades: int = 0
    active_signals: int = 0
    regime: str = "unknown"
    discipline_trend: str = "stable"  # improving | stable | declining
    coaching_alerts: List[str] = []


class TradeHeatmapPoint(BaseModel):
    """For time-of-day / day-of-week heatmap."""
    day: str
    hour: int
    trades: int = 0
    net_r: float = 0.0
    win_rate: float = 0.0
