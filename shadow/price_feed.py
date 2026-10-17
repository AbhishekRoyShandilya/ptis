# shadow/price_feed.py
"""
Real-Time Price Feed Manager
==============================
Central price feed that dispatches ticks to shadow engine, scanners, and alerts.

Backends (priority order):
1. NseKiteAdapter — Kite Connect WebSocket (production, requires API keys)
2. Manual injection — inject_price() for testing and frontend override
3. Yahoo Finance polling — fallback for development (no API key needed)

Usage:
    manager = PriceFeedManager()
    manager.on_price(my_callback)
    manager.subscribe("RELIANCE")

    # Production: connect Kite WebSocket
    manager.connect_kite(api_key="xxx", access_token="yyy")

    # Development: inject prices manually
    manager.inject_price("RELIANCE", 2450.50)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class PriceTick:
    """A single price update."""
    symbol: str
    price: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None


class PriceFeedManager:
    """
    Manages real-time price feeds for active shadow trades, scanners, and alerts.

    Integrates with NseKiteAdapter for production NSE feeds and provides
    manual injection + Yahoo polling as fallbacks.
    """

    def __init__(self):
        self._subscribed_symbols: Set[str] = set()
        self._callbacks: List[Callable[[PriceTick], None]] = []
        self._running = False
        self._prices: Dict[str, PriceTick] = {}
        self._kite_adapter = None  # NseKiteAdapter when connected

    # ── Subscription ──

    def subscribe(self, symbol: str) -> None:
        """Subscribe to price updates for a symbol."""
        sym = symbol.upper()
        self._subscribed_symbols.add(sym)
        # Forward to Kite adapter if connected
        if self._kite_adapter:
            self._kite_adapter.subscribe([sym])
        logger.info("Price feed: subscribed to %s", sym)

    def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from price updates."""
        sym = symbol.upper()
        self._subscribed_symbols.discard(sym)
        if self._kite_adapter:
            self._kite_adapter.unsubscribe([sym])
        logger.info("Price feed: unsubscribed from %s", sym)

    def on_price(self, callback: Callable[[PriceTick], None]) -> None:
        """Register a callback for price updates."""
        self._callbacks.append(callback)

    # ── Manual Injection (testing + frontend) ──

    def inject_price(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> None:
        """Manually inject a price update (for testing or manual entry)."""
        tick = PriceTick(
            symbol=symbol.upper(),
            price=price,
            timestamp=timestamp or datetime.now(timezone.utc),
            bid=bid,
            ask=ask,
            volume=volume,
        )
        self._prices[symbol.upper()] = tick
        self._dispatch(tick)

    # ── Price Queries ──

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last known price for a symbol."""
        tick = self._prices.get(symbol.upper())
        return tick.price if tick else None

    def get_last_tick(self, symbol: str) -> Optional[PriceTick]:
        """Get the full last tick for a symbol."""
        return self._prices.get(symbol.upper())

    @property
    def subscribed_symbols(self) -> Set[str]:
        return set(self._subscribed_symbols)

    @property
    def is_connected(self) -> bool:
        """Whether a live feed backend is active."""
        if self._kite_adapter:
            return getattr(self._kite_adapter, "is_connected", False)
        return self._running

    # ── Kite Connect Integration (Production) ──

    def connect_kite(
        self,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        mode: str = "ltp",
    ) -> None:
        """
        Connect to NSE via Kite Connect WebSocket.

        Args:
            api_key: Kite API key (falls back to settings.KITE_API_KEY)
            access_token: Kite access token (falls back to settings.KITE_ACCESS_TOKEN)
            mode: "ltp" (price only), "quote" (OHLC+vol), "full" (depth)
        """
        try:
            from shadow.price_adapters.nse_kite import NseKiteAdapter
        except ImportError:
            logger.error("NseKiteAdapter not available — kiteconnect not installed")
            return

        try:
            adapter = NseKiteAdapter(api_key=api_key, access_token=access_token)
        except ValueError as e:
            logger.warning("Cannot connect Kite: %s — falling back to manual/polling", e)
            return

        # Bridge Kite ticks into PriceFeedManager
        def _on_kite_tick(tick_data: dict) -> None:
            tick = PriceTick(
                symbol=tick_data.get("symbol", ""),
                price=tick_data.get("ltp", tick_data.get("last_price", 0.0)),
                timestamp=tick_data.get("timestamp", datetime.now(timezone.utc)),
                bid=tick_data.get("best_bid_price"),
                ask=tick_data.get("best_ask_price"),
                volume=tick_data.get("volume"),
                open=tick_data.get("ohlc", {}).get("open") if isinstance(tick_data.get("ohlc"), dict) else None,
                high=tick_data.get("ohlc", {}).get("high") if isinstance(tick_data.get("ohlc"), dict) else None,
                low=tick_data.get("ohlc", {}).get("low") if isinstance(tick_data.get("ohlc"), dict) else None,
                close=tick_data.get("ohlc", {}).get("close") if isinstance(tick_data.get("ohlc"), dict) else None,
            )
            if tick.symbol:
                self._prices[tick.symbol] = tick
                self._dispatch(tick)

        adapter.on_tick(_on_kite_tick)

        # Subscribe existing symbols
        if self._subscribed_symbols:
            adapter.subscribe(list(self._subscribed_symbols), mode=mode)

        # Connect in background thread
        adapter.connect(threaded=True)
        self._kite_adapter = adapter
        self._running = True
        logger.info(
            "Kite WebSocket connected — %d symbols subscribed",
            len(self._subscribed_symbols),
        )

    def disconnect_kite(self) -> None:
        """Disconnect from Kite WebSocket."""
        if self._kite_adapter:
            self._kite_adapter.disconnect()
            self._kite_adapter = None
            logger.info("Kite WebSocket disconnected")

    # ── Yahoo Finance Polling (Fallback) ──

    async def start_polling(self, interval_seconds: float = 5.0) -> None:
        """Start polling prices for subscribed symbols via Yahoo Finance."""
        self._running = True
        logger.info("Starting Yahoo Finance polling every %.1fs", interval_seconds)

        while self._running:
            for symbol in list(self._subscribed_symbols):
                try:
                    price = await self._fetch_price_yahoo(symbol)
                    if price:
                        self.inject_price(symbol, price)
                except Exception as e:
                    logger.error("Price fetch error for %s: %s", symbol, e)

            await asyncio.sleep(interval_seconds)

    def stop(self) -> None:
        """Stop all feeds (polling + WebSocket)."""
        self._running = False
        self.disconnect_kite()
        logger.info("Price feed manager stopped")

    async def _fetch_price_yahoo(self, symbol: str) -> Optional[float]:
        """Fetch current price from Yahoo Finance (development fallback)."""
        try:
            import aiohttp
        except ImportError:
            return None

        # NSE symbols need .NS suffix for Yahoo
        yahoo_symbol = f"{symbol}.NS" if not symbol.endswith((".NS", ".BO")) else symbol
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        params = {"interval": "1m", "range": "1d"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        meta = data["chart"]["result"][0]["meta"]
                        return meta.get("regularMarketPrice")
        except Exception as e:
            logger.debug("Yahoo fetch failed for %s: %s", symbol, e)
        return None

    # ── Internal ──

    def _dispatch(self, tick: PriceTick) -> None:
        """Dispatch price tick to all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(tick)
            except Exception as e:
                logger.error("Price callback error for %s: %s", tick.symbol, e)


# ── Module singleton ──
_manager: Optional[PriceFeedManager] = None


def get_price_feed_manager() -> PriceFeedManager:
    """Get or create the singleton PriceFeedManager."""
    global _manager
    if _manager is None:
        _manager = PriceFeedManager()
    return _manager
