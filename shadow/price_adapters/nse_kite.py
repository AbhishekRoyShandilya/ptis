# shadow/price_adapters/nse_kite.py
"""
NSE real-time price feed via Kite Connect WebSocket (KiteTicker).

Provides live price ticks for shadow trading engine, autonomous scanning,
and watchlist alerts.

Features:
- Subscribes to instrument tokens via KiteTicker
- Maps ticks to PriceTick dataclass for PriceFeedManager
- Auto-reconnection with exponential backoff
- Supports MODE_LTP (lightweight) and MODE_FULL (depth)

Requires: kiteconnect>=5.0.0 with KiteTicker
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time as time_mod
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from config.settings import settings

logger = logging.getLogger(__name__)


class NseKiteAdapter:
    """
    Real-time NSE price feed adapter using Kite Connect WebSocket.

    Usage:
        adapter = NseKiteAdapter(api_key="xxx", access_token="yyy")
        adapter.on_tick(my_callback)
        adapter.subscribe(["RELIANCE", "NIFTY 50"])
        adapter.connect()  # Blocking — run in thread
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
    ):
        self._api_key = api_key or settings.KITE_API_KEY
        self._access_token = access_token or settings.KITE_ACCESS_TOKEN

        if not self._api_key or not self._access_token:
            raise ValueError(
                "KITE_API_KEY and KITE_ACCESS_TOKEN are required for NSE price feed"
            )

        self._kws = None  # KiteTicker instance
        self._callbacks: List[Callable] = []
        self._subscribed_symbols: Set[str] = set()
        self._token_map: Dict[int, str] = {}  # instrument_token → symbol
        self._symbol_map: Dict[str, int] = {}  # symbol → instrument_token
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._last_prices: Dict[str, Dict[str, Any]] = {}

    def on_tick(self, callback: Callable) -> None:
        """Register a callback for price tick events."""
        self._callbacks.append(callback)

    def subscribe(self, symbols: List[str], mode: str = "full") -> None:
        """
        Subscribe to price updates for given symbols.

        Args:
            symbols: List of trading symbols (e.g., ["RELIANCE", "NIFTY 50"])
            mode: "ltp" (price only), "quote" (OHLC + volume), "full" (depth)
        """
        for sym in symbols:
            self._subscribed_symbols.add(sym.upper())

        # If already connected, resolve tokens and subscribe
        if self._kws and self._running:
            self._resolve_and_subscribe(mode)

    def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from price updates."""
        tokens_to_unsub = []
        for sym in symbols:
            sym = sym.upper()
            self._subscribed_symbols.discard(sym)
            token = self._symbol_map.get(sym)
            if token:
                tokens_to_unsub.append(token)
                del self._token_map[token]
                del self._symbol_map[sym]

        if self._kws and tokens_to_unsub:
            try:
                self._kws.unsubscribe(tokens_to_unsub)
            except Exception as e:
                logger.error("Failed to unsubscribe: %s", e)

    def connect(self, threaded: bool = True) -> None:
        """
        Start the WebSocket connection.

        Args:
            threaded: If True, runs in a background thread. If False, blocks.
        """
        try:
            from kiteconnect import KiteTicker
        except ImportError:
            raise ImportError(
                "kiteconnect package not installed. "
                "Install with: pip install kiteconnect>=5.0.0"
            )

        self._kws = KiteTicker(self._api_key, self._access_token)

        # Register KiteTicker callbacks
        self._kws.on_ticks = self._on_ticks
        self._kws.on_connect = self._on_connect
        self._kws.on_close = self._on_close
        self._kws.on_error = self._on_error
        self._kws.on_reconnect = self._on_reconnect
        self._kws.on_noreconnect = self._on_noreconnect

        self._running = True

        if threaded:
            self._kws.connect(threaded=True)
            logger.info("NSE KiteTicker started in background thread")
        else:
            logger.info("NSE KiteTicker starting (blocking)")
            self._kws.connect()

    def disconnect(self) -> None:
        """Stop the WebSocket connection."""
        self._running = False
        if self._kws:
            try:
                self._kws.close()
            except Exception:
                pass
        logger.info("NSE KiteTicker disconnected")

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get the last known price for a symbol."""
        data = self._last_prices.get(symbol.upper())
        return data.get("ltp") if data else None

    def get_last_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the full last tick data for a symbol."""
        return self._last_prices.get(symbol.upper())

    @property
    def is_connected(self) -> bool:
        return self._running and self._kws is not None

    @property
    def subscribed_count(self) -> int:
        return len(self._subscribed_symbols)

    # ══════════════════════════════════════════
    #  KITETICKER CALLBACKS
    # ══════════════════════════════════════════

    def _on_ticks(self, ws, ticks: List[Dict[str, Any]]) -> None:
        """Called when ticks are received from KiteTicker."""
        for tick in ticks:
            token = tick.get("instrument_token")
            symbol = self._token_map.get(token, f"UNKNOWN({token})")

            tick_data = {
                "symbol": symbol,
                "ltp": tick.get("last_price"),
                "timestamp": tick.get("last_trade_time") or datetime.now(timezone.utc),
                "open": tick.get("ohlc", {}).get("open"),
                "high": tick.get("ohlc", {}).get("high"),
                "low": tick.get("ohlc", {}).get("low"),
                "close": tick.get("ohlc", {}).get("close"),
                "volume": tick.get("volume_traded") or tick.get("volume"),
                "oi": tick.get("oi"),
                "bid": tick.get("depth", {}).get("buy", [{}])[0].get("price")
                       if tick.get("depth") else None,
                "ask": tick.get("depth", {}).get("sell", [{}])[0].get("price")
                       if tick.get("depth") else None,
                "change_pct": tick.get("change"),
            }

            self._last_prices[symbol] = tick_data

            # Dispatch to all registered callbacks
            for callback in self._callbacks:
                try:
                    callback(tick_data)
                except Exception as e:
                    logger.error("Tick callback error for %s: %s", symbol, e)

    def _on_connect(self, ws, response) -> None:
        """Called when WebSocket connects successfully."""
        logger.info("NSE KiteTicker connected")
        self._reconnect_attempts = 0
        self._resolve_and_subscribe()

    def _on_close(self, ws, code, reason) -> None:
        """Called when WebSocket closes."""
        logger.warning("NSE KiteTicker closed: code=%s reason=%s", code, reason)

    def _on_error(self, ws, code, reason) -> None:
        """Called when WebSocket encounters an error."""
        logger.error("NSE KiteTicker error: code=%s reason=%s", code, reason)

    def _on_reconnect(self, ws, attempts_count) -> None:
        """Called when WebSocket is attempting to reconnect."""
        self._reconnect_attempts = attempts_count
        logger.info("NSE KiteTicker reconnecting (attempt %d)", attempts_count)

    def _on_noreconnect(self, ws) -> None:
        """Called when max reconnection attempts are exhausted."""
        logger.error("NSE KiteTicker max reconnection attempts exhausted")
        self._running = False

    # ══════════════════════════════════════════
    #  TOKEN RESOLUTION
    # ══════════════════════════════════════════

    def _resolve_and_subscribe(self, mode: str = "full") -> None:
        """Resolve symbols to instrument tokens and subscribe via KiteTicker."""
        try:
            from kiteconnect import KiteConnect
        except ImportError:
            logger.error("Cannot resolve instruments — kiteconnect not installed")
            return

        if not self._subscribed_symbols:
            return

        # Use a temporary KiteConnect instance for instrument resolution
        kite = KiteConnect(api_key=self._api_key)
        kite.set_access_token(self._access_token)

        # Determine which symbols need token resolution
        unresolved = [s for s in self._subscribed_symbols if s not in self._symbol_map]

        if unresolved:
            try:
                # Fetch instruments for NSE and NFO
                for exchange in ("NSE", "NFO"):
                    instruments = kite.instruments(exchange)
                    for inst in instruments:
                        sym = inst["tradingsymbol"].upper()
                        if sym in unresolved or inst.get("name", "").upper() in unresolved:
                            self._symbol_map[sym] = inst["instrument_token"]
                            self._token_map[inst["instrument_token"]] = sym
            except Exception as e:
                logger.error("Failed to resolve instrument tokens: %s", e)
                return

        # Subscribe via KiteTicker
        tokens = list(self._token_map.keys())
        if tokens and self._kws:
            try:
                self._kws.subscribe(tokens)

                # Set mode
                try:
                    from kiteconnect import KiteTicker
                    mode_map = {
                        "ltp": KiteTicker.MODE_LTP,
                        "quote": KiteTicker.MODE_QUOTE,
                        "full": KiteTicker.MODE_FULL,
                    }
                    ws_mode = mode_map.get(mode, KiteTicker.MODE_FULL)
                    self._kws.set_mode(ws_mode, tokens)
                except Exception:
                    pass

                logger.info(
                    "Subscribed to %d instruments: %s",
                    len(tokens),
                    list(self._token_map.values())[:10],
                )
            except Exception as e:
                logger.error("Failed to subscribe to tokens: %s", e)


# ══════════════════════════════════════════
#  BRIDGE TO PriceFeedManager
# ══════════════════════════════════════════

def create_nse_price_bridge(
    price_feed_manager,
    api_key: Optional[str] = None,
    access_token: Optional[str] = None,
) -> NseKiteAdapter:
    """
    Create an NseKiteAdapter that bridges ticks into PriceFeedManager.

    Usage:
        from shadow.price_feed import PriceFeedManager
        pfm = PriceFeedManager()
        adapter = create_nse_price_bridge(pfm)
        adapter.subscribe(["RELIANCE", "NIFTY 50"])
        adapter.connect()

    Each tick from Kite is injected into PriceFeedManager.inject_price().
    """
    adapter = NseKiteAdapter(api_key=api_key, access_token=access_token)

    def bridge_tick(tick_data: Dict[str, Any]) -> None:
        symbol = tick_data.get("symbol", "")
        ltp = tick_data.get("ltp")
        ts = tick_data.get("timestamp")
        if symbol and ltp is not None:
            price_feed_manager.inject_price(symbol, ltp, ts)

    adapter.on_tick(bridge_tick)
    return adapter
