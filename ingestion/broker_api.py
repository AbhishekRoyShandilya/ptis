# ingestion/broker_api.py
"""
Broker API facade — unified interface for fetching trade data
from multiple broker adapters.

Each broker adapter implements the same read-only interface:
- get_historical_candles()
- get_positions()
- get_orders()
- get_account_info()
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BrokerAPIError(Exception):
    """Base exception for broker API errors."""
    pass


class BrokerRateLimitError(BrokerAPIError):
    """Rate limit exceeded."""
    pass


class BrokerAuthError(BrokerAPIError):
    """Authentication failed."""
    pass


class BrokerAdapter:
    """
    Abstract broker adapter interface.
    All broker-specific clients extend this.
    """

    name: str = "base"

    def get_historical_candles(
        self,
        symbol: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV candles.
        Returns: [{timestamp, open, high, low, close, volume}]
        """
        raise NotImplementedError

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Fetch current open positions.
        Returns: [{symbol, quantity, avg_price, pnl, side}]
        """
        raise NotImplementedError

    def get_orders(
        self,
        from_date: Optional[date] = None,
        status: str = "filled",
    ) -> List[Dict[str, Any]]:
        """
        Fetch order history.
        Returns: [{symbol, side, qty, price, filled_at, status}]
        """
        raise NotImplementedError

    def get_account_info(self) -> Dict[str, Any]:
        """
        Fetch account summary.
        Returns: {equity, buying_power, cash, margin_used}
        """
        raise NotImplementedError

    def get_market_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current market quote.
        Returns: {ltp, bid, ask, volume, open, high, low}
        """
        raise NotImplementedError


class BrokerAPIFacade:
    """
    Unified facade that routes requests to the appropriate broker adapter.

    Usage:
        facade = BrokerAPIFacade()
        facade.register("upstox", UpstoxClient(...))
        facade.register("alpaca", AlpacaClient(...))

        candles = facade.get_candles("upstox", "RELIANCE", "15m", ...)
    """

    def __init__(self):
        self._adapters: Dict[str, BrokerAdapter] = {}

    def register(self, name: str, adapter: BrokerAdapter) -> None:
        """Register a broker adapter."""
        self._adapters[name] = adapter
        logger.info("Registered broker adapter: %s", name)

    def get_adapter(self, name: str) -> BrokerAdapter:
        """Get a registered adapter by name."""
        adapter = self._adapters.get(name)
        if adapter is None:
            raise BrokerAPIError(f"No broker adapter registered: {name}")
        return adapter

    @property
    def available_brokers(self) -> List[str]:
        return list(self._adapters.keys())

    def get_candles(
        self,
        broker: str,
        symbol: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> List[Dict[str, Any]]:
        adapter = self.get_adapter(broker)
        return adapter.get_historical_candles(symbol, interval, from_date, to_date)

    def get_positions(self, broker: str) -> List[Dict[str, Any]]:
        return self.get_adapter(broker).get_positions()

    def get_orders(
        self,
        broker: str,
        from_date: Optional[date] = None,
        status: str = "filled",
    ) -> List[Dict[str, Any]]:
        return self.get_adapter(broker).get_orders(from_date, status)

    def get_quote(self, broker: str, symbol: str) -> Dict[str, Any]:
        return self.get_adapter(broker).get_market_quote(symbol)
