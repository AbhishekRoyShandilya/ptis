# base_condition.py
# strategy_engine/conditions/base_condition.py
"""
Abstract condition evaluator and shared data structures for strategy matching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from core.enums import (
    ConditionStatus,
    MarketRegime,
    SessionType,
    TradeDirection,
)


@dataclass
class MarketSnapshot:
    """Point-in-time market data available for condition evaluation."""
    symbol: str
    timestamp: datetime

    # OHLCV (current bar)
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None

    # Indicators (pre-computed, keyed by name)
    indicators: dict[str, Any] = field(default_factory=dict)
    # e.g., {"ema_9": 150.5, "ema_20": 149.8, "rsi_14": 62, "vwap": 150.1, "atr_14": 2.3}

    # Structure
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)

    # Candle history (most recent first)
    recent_candles: list[dict[str, float]] = field(default_factory=list)
    # Each: {"open", "high", "low", "close", "volume", "timestamp"}

    # Derived
    opening_range_high: Optional[float] = None
    opening_range_low: Optional[float] = None
    prev_day_high: Optional[float] = None
    prev_day_low: Optional[float] = None
    prev_day_close: Optional[float] = None

    # Volume context
    avg_volume: Optional[float] = None
    relative_volume: Optional[float] = None

    # Regime
    regime: MarketRegime = MarketRegime.UNKNOWN
    session_type: Optional[SessionType] = None

    def get_indicator(self, name: str) -> Optional[float]:
        return self.indicators.get(name)

    @property
    def atr(self) -> Optional[float]:
        return self.indicators.get("atr_14") or self.indicators.get("atr")


@dataclass
class TradeContext:
    """All data about the trade being evaluated."""
    symbol: str
    direction: TradeDirection
    entry_price: float
    entry_time: datetime
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_amount: Optional[float] = None
    risk_pct: Optional[float] = None
    account_balance: Optional[float] = None

    # Exit (filled if trade closed)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None

    # Context
    declared_strategy_id: Optional[str] = None
    session_type: Optional[SessionType] = None
    market_regime: Optional[MarketRegime] = None

    # Session state
    trades_today: int = 0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    time_since_last_trade_seconds: Optional[float] = None
    last_trade_pnl: Optional[float] = None


@dataclass
class ConditionResult:
    """Result of evaluating a single condition."""
    condition_type: str
    status: ConditionStatus
    confidence: float = 1.0          # 0.0 - 1.0, how certain
    actual_value: Optional[str] = None
    expected_value: Optional[str] = None
    details: str = ""
    weight: float = 1.0
    required: bool = True
    error: Optional[str] = None

    @property
    def is_met(self) -> bool:
        return self.status == ConditionStatus.MET

    @property
    def is_failed(self) -> bool:
        return self.status == ConditionStatus.NOT_MET

    @property
    def weighted_score(self) -> float:
        """Weighted contribution: 1.0 if met, 0.5 if partial, 0.0 otherwise."""
        if self.status == ConditionStatus.MET:
            return self.weight * self.confidence
        elif self.status == ConditionStatus.PARTIALLY_MET:
            return self.weight * 0.5 * self.confidence
        return 0.0


class BaseCondition(ABC):
    """Abstract base for all condition evaluators."""

    @property
    @abstractmethod
    def condition_type(self) -> str:
        """Unique identifier matching YAML 'type' field."""
        ...

    @abstractmethod
    def evaluate(
        self,
        params: dict[str, Any],
        trade: TradeContext,
        market: MarketSnapshot,
    ) -> ConditionResult:
        """Evaluate this condition against trade + market data."""
        ...

    def _insufficient_data(self, field_name: str) -> ConditionResult:
        return ConditionResult(
            condition_type=self.condition_type,
            status=ConditionStatus.INCONCLUSIVE,
            details=f"Insufficient data: '{field_name}' not available",
            confidence=0.0,
        )