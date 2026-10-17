# shadow/price_adapters/yahoo_polling.py
"""
Yahoo Finance polling adapter — fetches prices at configurable
intervals for subscribed symbols.

Best for: paper trading, development, low-frequency shadow updates.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from shadow.price_feed import PriceFeedManager, PriceTick

logger = logging.getLogger(__name__)


class YahooPollingAdapter:
    """
    Polls Yahoo Finance for prices at a configurable interval.
    Uses aiohttp for async HTTP requests.
    """

    CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    def __init__(
        self,
        feed: PriceFeedManager,
        interval_seconds: float = 5.0,
    ):
        self._feed = feed
        self._interval = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the polling loop."""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Yahoo polling adapter started (interval=%.1fs)", self._interval
        )

    async def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Yahoo polling adapter stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    async def _poll_loop(self) -> None:
        import aiohttp

        while self._running:
            symbols = list(self._feed._subscribed_symbols)
            if not symbols:
                await asyncio.sleep(self._interval)
                continue

            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    if not self._running:
                        break
                    try:
                        price = await self._fetch_price(session, symbol)
                        if price is not None:
                            tick = PriceTick(
                                symbol=symbol,
                                price=price,
                                timestamp=datetime.now(timezone.utc),
                            )
                            self._feed._prices[symbol] = tick
                            self._feed._dispatch(tick)
                    except Exception as exc:
                        logger.warning(
                            "Yahoo fetch failed for %s: %s", symbol, exc
                        )

            await asyncio.sleep(self._interval)

    async def _fetch_price(
        self,
        session: "aiohttp.ClientSession",
        symbol: str,
    ) -> Optional[float]:
        """Fetch LTP from Yahoo Finance chart API."""
        url = self.CHART_URL.format(symbol=symbol)
        params = {"interval": "1m", "range": "1d"}

        async with session.get(url, params=params, timeout=10) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            result = data.get("chart", {}).get("result")
            if not result:
                return None
            meta = result[0].get("meta", {})
            return meta.get("regularMarketPrice")

    async def fetch_once(self, symbol: str) -> Optional[float]:
        """One-shot price fetch (no polling loop needed)."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            return await self._fetch_price(session, symbol)
