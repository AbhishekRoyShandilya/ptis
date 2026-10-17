# tests/factories.py
"""Test data factories for PTIS test suite."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd

from core.enums import (
    TradeDirection, TradeStatus, MarketRegime, VolatilityRegime,
    SessionType, EmotionalState,
)


# ── Trade dict factory ────────────────────────────────────────────────────────

def make_trade_dict(
    symbol: str = "RELIANCE",
    direction: str = "long",
    entry_price: float = 2500.0,
    exit_price: Optional[float] = None,
    stop_loss: float = 2450.0,
    take_profit: float = 2600.0,
    position_size: float = 10.0,
    pnl: Optional[float] = None,
    r_multiple: Optional[float] = None,
    status: str = "open",
    **overrides,
) -> dict:
    """Create a trade dict suitable for API payloads and pipeline input."""
    d = {
        "id": str(uuid.uuid4()),
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_size": position_size,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "exit_time": None,
        "pnl": pnl,
        "r_multiple": r_multiple,
        "status": status,
        "risk_amount": abs(entry_price - stop_loss) * position_size,
        "risk_pct": 1.0,
        "account_balance_at_entry": 100_000.0,
        "session_type": "morning",
        "market_regime": "trending_up",
    }
    d.update(overrides)
    return d


def make_closed_trade(**overrides) -> dict:
    """Convenience: a closed profitable trade."""
    defaults = {
        "status": "closed",
        "exit_price": 2580.0,
        "pnl": 800.0,
        "r_multiple": 1.6,
        "exit_time": datetime.now(timezone.utc).isoformat(),
    }
    defaults.update(overrides)
    return make_trade_dict(**defaults)


def make_losing_trade(**overrides) -> dict:
    """Convenience: a closed losing trade."""
    defaults = {
        "status": "closed",
        "exit_price": 2440.0,
        "pnl": -600.0,
        "r_multiple": -1.2,
        "exit_time": datetime.now(timezone.utc).isoformat(),
    }
    defaults.update(overrides)
    return make_trade_dict(**defaults)


# ── OHLCV DataFrame factory ──────────────────────────────────────────────────

def make_ohlcv_df(
    n_bars: int = 100,
    start_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0005,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data.

    Args:
        n_bars: Number of bars.
        start_price: Starting close price.
        volatility: Per-bar volatility (std of returns).
        trend: Per-bar drift (positive = uptrend).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: open, high, low, close, volume.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, volatility, n_bars)
    close = start_price * np.cumprod(1 + returns)

    # Build OHLC from close
    open_ = np.roll(close, 1)
    open_[0] = start_price
    high = np.maximum(open_, close) * (1 + rng.uniform(0, volatility * 0.5, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, volatility * 0.5, n_bars))
    volume = rng.integers(100_000, 1_000_000, n_bars).astype(float)

    dates = pd.date_range("2025-01-01", periods=n_bars, freq="1h")

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


def make_trending_up_df(n_bars: int = 100, **kwargs) -> pd.DataFrame:
    """OHLCV with strong uptrend."""
    return make_ohlcv_df(n_bars=n_bars, trend=0.002, volatility=0.01, **kwargs)


def make_trending_down_df(n_bars: int = 100, **kwargs) -> pd.DataFrame:
    """OHLCV with strong downtrend."""
    return make_ohlcv_df(n_bars=n_bars, trend=-0.002, volatility=0.01, **kwargs)


def make_ranging_df(n_bars: int = 100, **kwargs) -> pd.DataFrame:
    """OHLCV with low-volatility sideways action."""
    return make_ohlcv_df(n_bars=n_bars, trend=0.0, volatility=0.005, **kwargs)


def make_volatile_df(n_bars: int = 100, **kwargs) -> pd.DataFrame:
    """OHLCV with high volatility."""
    return make_ohlcv_df(n_bars=n_bars, trend=0.0, volatility=0.05, **kwargs)


# ── Candle dict factory ───────────────────────────────────────────────────────

def make_candle(
    open_: float = 100.0,
    high: float = 102.0,
    low: float = 99.0,
    close: float = 101.0,
    volume: float = 500_000.0,
) -> dict:
    """Single candle dict for candle_patterns tests."""
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def make_doji_candle(mid: float = 100.0, range_: float = 2.0) -> dict:
    """Doji: open ≈ close, wicks on both sides."""
    return make_candle(
        open_=mid,
        high=mid + range_ / 2,
        low=mid - range_ / 2,
        close=mid + 0.01,
    )


def make_hammer_candle(close: float = 100.0) -> dict:
    """Hammer: small body at top, long lower wick."""
    return make_candle(
        open_=close - 1.0,
        high=close + 0.2,
        low=close - 5.0,
        close=close,
    )


def make_shooting_star_candle(close: float = 100.0) -> dict:
    """Shooting star: small body at bottom, long upper wick."""
    return make_candle(
        open_=close + 1.0,
        high=close + 5.0,
        low=close - 0.2,
        close=close,
    )


def make_marubozu_candle(open_: float = 98.0, close: float = 102.0) -> dict:
    """Marubozu: full body, no wicks."""
    return make_candle(
        open_=open_,
        high=max(open_, close) + 0.01,
        low=min(open_, close) - 0.01,
        close=close,
    )
