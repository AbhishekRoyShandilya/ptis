# core/session_manager.py
"""
Manages intraday trading session state.
Tracks trades, P&L, consecutive losses, timing — everything needed
for behavioral detection and override analysis during a live session.
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import func, and_
from sqlalchemy.orm import Session as DBSession

from core.enums import EmotionalState, MarketRegime, SessionType, TradeStatus
from core.models import Trade
from strategy_engine.override_detector import SessionContext

logger = logging.getLogger(__name__)


class TradingSessionManager:
    """
    Maintains live session state. Rebuilt from DB on startup,
    updated incrementally as trades flow in.
    """

    def __init__(self):
        self._session_date: Optional[date] = None
        self._trades_today: int = 0
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._last_trade_time: Optional[datetime] = None
        self._last_trade_pnl: Optional[float] = None
        self._last_trade_symbol: Optional[str] = None
        self._account_balance: Optional[float] = None
        self._max_daily_pnl: float = 0.0
        self._min_daily_pnl: float = 0.0
        self._trade_ids_today: list[UUID] = []
        self._current_regime: MarketRegime = MarketRegime.UNKNOWN

    # ── Properties ──

    @property
    def session_date(self) -> Optional[date]:
        return self._session_date

    @property
    def trades_today(self) -> int:
        return self._trades_today

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def is_active(self) -> bool:
        return self._session_date == date.today()

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    # ── Session Context Export ──

    def to_session_context(self) -> SessionContext:
        """Export current state as SessionContext for override detection."""
        return SessionContext(
            trades_today=self._trades_today,
            daily_pnl=self._daily_pnl,
            consecutive_losses=self._consecutive_losses,
            last_trade_time=self._last_trade_time,
            last_trade_pnl=self._last_trade_pnl,
            last_trade_symbol=self._last_trade_symbol,
            account_balance=self._account_balance,
        )

    # ── State Updates ──

    def on_trade_entered(
        self,
        trade_id: UUID,
        symbol: str,
        entry_time: datetime,
        risk_amount: Optional[float] = None,
    ) -> None:
        """Called when a new trade is entered."""
        self._ensure_session()
        self._trades_today += 1
        self._trade_ids_today.append(trade_id)
        logger.info(
            "Session trade #%d entered: %s at %s",
            self._trades_today, symbol, entry_time.strftime("%H:%M:%S"),
        )

    def on_trade_closed(
        self,
        trade_id: UUID,
        symbol: str,
        pnl: float,
        exit_time: datetime,
    ) -> None:
        """Called when a trade is closed."""
        self._ensure_session()
        self._daily_pnl += pnl
        self._last_trade_time = exit_time
        self._last_trade_pnl = pnl
        self._last_trade_symbol = symbol

        if pnl >= 0:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        # Track daily extremes
        self._max_daily_pnl = max(self._max_daily_pnl, self._daily_pnl)
        self._min_daily_pnl = min(self._min_daily_pnl, self._daily_pnl)

        logger.info(
            "Trade closed: %s P&L=$%.2f | Day: $%.2f | Consecutive losses: %d",
            symbol, pnl, self._daily_pnl, self._consecutive_losses,
        )

    def set_account_balance(self, balance: float) -> None:
        self._account_balance = balance

    def set_regime(self, regime: MarketRegime) -> None:
        self._current_regime = regime

    # ── Session Lifecycle ──

    def _ensure_session(self) -> None:
        """Auto-reset if day has changed."""
        today = date.today()
        if self._session_date != today:
            self.reset(today)

    def reset(self, session_date: Optional[date] = None) -> None:
        """Reset all session state for a new day."""
        self._session_date = session_date or date.today()
        self._trades_today = 0
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._last_trade_time = None
        self._last_trade_pnl = None
        self._last_trade_symbol = None
        self._max_daily_pnl = 0.0
        self._min_daily_pnl = 0.0
        self._trade_ids_today.clear()
        logger.info("Trading session reset for %s", self._session_date)

    def rebuild_from_db(self, db: DBSession) -> None:
        """
        Rebuild session state from today's trades in DB.
        Called on app startup to restore session.
        """
        today = date.today()
        self.reset(today)

        today_start = datetime.combine(today, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )

        trades = (
            db.query(Trade)
            .filter(
                Trade.entry_time >= today_start,
                Trade.status.in_([TradeStatus.OPEN, TradeStatus.CLOSED]),
            )
            .order_by(Trade.entry_time.asc())
            .all()
        )

        for trade in trades:
            self._trades_today += 1
            self._trade_ids_today.append(trade.id)

            if trade.status == TradeStatus.CLOSED and trade.pnl is not None:
                self._daily_pnl += trade.pnl
                self._last_trade_time = trade.exit_time
                self._last_trade_pnl = trade.pnl
                self._last_trade_symbol = trade.symbol

                if trade.pnl >= 0:
                    self._consecutive_losses = 0
                else:
                    self._consecutive_losses += 1

        self._max_daily_pnl = max(0.0, self._daily_pnl)
        self._min_daily_pnl = min(0.0, self._daily_pnl)

        logger.info(
            "Session rebuilt from DB: %d trades, P&L=$%.2f, consecutive_losses=%d",
            self._trades_today, self._daily_pnl, self._consecutive_losses,
        )

    def summary(self) -> dict:
        return {
            "session_date": str(self._session_date),
            "trades_today": self._trades_today,
            "daily_pnl": round(self._daily_pnl, 2),
            "consecutive_losses": self._consecutive_losses,
            "max_daily_pnl": round(self._max_daily_pnl, 2),
            "min_daily_pnl": round(self._min_daily_pnl, 2),
            "last_trade_symbol": self._last_trade_symbol,
            "account_balance": self._account_balance,
            "current_regime": self._current_regime.value,
        }


# ── Module-level singleton ──
_session_mgr: TradingSessionManager | None = None


def get_session_manager() -> TradingSessionManager:
    global _session_mgr
    if _session_mgr is None:
        _session_mgr = TradingSessionManager()
    return _session_mgr