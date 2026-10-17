# utils.py
# core/utils.py
"""
Shared utility functions used across the system.
"""

from datetime import datetime, time, timedelta, date
from typing import Optional, Tuple, List
from core.enums import SessionType, TradeDirection
import math


def calculate_r_multiple(
    entry_price: float,
    exit_price: float,
    stop_loss: float,
    direction: TradeDirection,
) -> Optional[float]:
    """Calculate R-multiple for a trade."""
    if stop_loss == entry_price:
        return None

    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit == 0:
        return None

    if direction == TradeDirection.LONG:
        return (exit_price - entry_price) / risk_per_unit
    else:
        return (entry_price - exit_price) / risk_per_unit


def calculate_position_size(
    account_size: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
) -> int:
    """Calculate position size based on risk."""
    risk_amount = account_size * (risk_percent / 100)
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share == 0:
        return 0
    shares = math.floor(risk_amount / risk_per_share)
    return max(1, shares)


def classify_session(dt: datetime) -> SessionType:
    """Classify a datetime into a trading session."""
    hour = dt.hour
    minute = dt.minute
    t = hour + minute / 60

    if t < 9.5:
        return SessionType.PREMARKET
    elif t < 10.5:
        return SessionType.OPEN
    elif t < 14.0:
        return SessionType.MIDDAY
    elif t < 16.0:
        return SessionType.POWER_HOUR
    else:
        return SessionType.AFTER_HOURS


def calculate_hold_duration(
    entry_time: datetime,
    exit_time: datetime,
) -> float:
    """Calculate hold duration in minutes."""
    return (exit_time - entry_time).total_seconds() / 60


def format_currency(amount: float) -> str:
    """Format number as currency."""
    if amount >= 0:
        return f"${amount:,.2f}"
    return f"-${abs(amount):,.2f}"


def format_r_multiple(r: float) -> str:
    """Format R-multiple with color hint."""
    if r >= 2:
        return f"+{r:.2f}R 🟢"
    elif r >= 0:
        return f"+{r:.2f}R 🟡"
    elif r >= -1:
        return f"{r:.2f}R 🔴"
    else:
        return f"{r:.2f}R 🔴🔴"


def format_win_rate(rate: float) -> str:
    """Format win rate with context."""
    if rate >= 60:
        return f"{rate:.1f}% ✅"
    elif rate >= 45:
        return f"{rate:.1f}%"
    else:
        return f"{rate:.1f}% ⚠️"


def is_market_hours(dt: datetime = None) -> bool:
    """Check if current time is during market hours (IST)."""
    dt = dt or datetime.now()
    t = dt.hour + dt.minute / 60
    return 9.25 <= t <= 15.5  # NSE: 9:15 AM to 3:30 PM IST


def get_trading_day_range(target_date: date = None) -> Tuple[datetime, datetime]:
    """Get start and end of trading day."""
    target_date = target_date or date.today()
    start = datetime.combine(target_date, time(9, 0))    # pre-market buffer
    end = datetime.combine(target_date, time(16, 0))      # post-close buffer
    return start, end


def streak_counter(results: List[bool]) -> Tuple[int, str]:
    """
    Count current win/loss streak.
    Returns (streak_length, 'win' or 'loss').
    """
    if not results:
        return 0, "none"

    current = results[0]
    count = 0

    for result in results:
        if result == current:
            count += 1
        else:
            break

    return count, "win" if current else "loss"