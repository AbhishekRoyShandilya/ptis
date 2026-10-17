# shadow/price_adapters/manual_injector.py
"""
Manual Price Injector — allows injecting prices from the frontend,
API calls, or test harnesses into the shadow price feed.

Primary adapter for paper-trading and manual backtesting where
real-time feeds are not available.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from shadow.price_feed import PriceFeedManager, PriceTick

logger = logging.getLogger(__name__)


class ManualPriceInjector:
    """
    Injects prices manually into PriceFeedManager.

    Usage:
        feed = PriceFeedManager()
        injector = ManualPriceInjector(feed)
        injector.inject("RELIANCE", 2450.50)
        injector.inject_batch({"RELIANCE": 2451.0, "TCS": 3820.5})
    """

    def __init__(self, feed: PriceFeedManager):
        self._feed = feed

    def inject(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> PriceTick:
        """Inject a single price tick."""
        ts = timestamp or datetime.now(timezone.utc)
        tick = PriceTick(
            symbol=symbol.upper(),
            price=price,
            timestamp=ts,
            bid=bid,
            ask=ask,
            volume=volume,
        )
        self._feed._prices[tick.symbol] = tick
        self._feed._dispatch(tick)
        logger.debug("Injected price: %s @ %.2f", symbol, price)
        return tick

    def inject_batch(
        self,
        prices: dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> list[PriceTick]:
        """Inject prices for multiple symbols at once."""
        ts = timestamp or datetime.now(timezone.utc)
        ticks = []
        for symbol, price in prices.items():
            tick = self.inject(symbol, price, ts)
            ticks.append(tick)
        logger.info("Batch-injected %d prices", len(ticks))
        return ticks

    def inject_ohlcv_bar(
        self,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0,
        timestamp: Optional[datetime] = None,
    ) -> list[PriceTick]:
        """
        Simulate an OHLCV bar as 4 sequential ticks.
        Useful for backtesting shadow trades bar-by-bar.
        Order: open -> low -> high -> close (conservative assumption).
        """
        ts = timestamp or datetime.now(timezone.utc)
        ticks = []
        for price in [open_, low, high, close]:
            tick = self.inject(symbol, price, ts, volume=volume)
            ticks.append(tick)
        return ticks
