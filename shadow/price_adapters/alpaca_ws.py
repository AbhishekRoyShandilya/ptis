# shadow/price_adapters/alpaca_ws.py
"""
Alpaca Markets WebSocket adapter — real-time price feed via
Alpaca's streaming API (v2).

Requires: ALPACA_API_KEY, ALPACA_API_SECRET in settings.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from shadow.price_feed import PriceFeedManager, PriceTick

logger = logging.getLogger(__name__)

_IEX_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"
_SIP_WS_URL = "wss://stream.data.alpaca.markets/v2/sip"


class AlpacaWebSocketAdapter:
    """
    Connects to Alpaca's real-time WebSocket for trade/quote data.

    Usage:
        adapter = AlpacaWebSocketAdapter(feed, api_key, api_secret)
        await adapter.start()
        ...
        await adapter.stop()
    """

    def __init__(
        self,
        feed: PriceFeedManager,
        api_key: str,
        api_secret: str,
        use_sip: bool = False,
    ):
        self._feed = feed
        self._api_key = api_key
        self._api_secret = api_secret
        self._ws_url = _SIP_WS_URL if use_sip else _IEX_WS_URL
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._ws = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Alpaca WS adapter starting (%s)", self._ws_url)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Alpaca WS adapter stopped")

    async def _run(self) -> None:
        import aiohttp

        retry_delay = 1.0

        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self._ws_url) as ws:
                        self._ws = ws
                        retry_delay = 1.0

                        # Authenticate
                        await ws.send_json({
                            "action": "auth",
                            "key": self._api_key,
                            "secret": self._api_secret,
                        })

                        auth_msg = await ws.receive_json()
                        if not self._check_auth(auth_msg):
                            logger.error("Alpaca auth failed: %s", auth_msg)
                            return

                        # Subscribe to trades
                        symbols = list(self._feed._subscribed_symbols)
                        if symbols:
                            await ws.send_json({
                                "action": "subscribe",
                                "trades": symbols,
                            })
                            logger.info("Subscribed to %d symbols", len(symbols))

                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self._handle_message(msg.data)
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                break

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "Alpaca WS error (reconnecting in %.0fs): %s",
                    retry_delay, exc,
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)

    def _check_auth(self, msg) -> bool:
        if isinstance(msg, list):
            for item in msg:
                if item.get("T") == "success" and item.get("msg") == "authenticated":
                    return True
        return False

    def _handle_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        events = data if isinstance(data, list) else [data]
        for event in events:
            msg_type = event.get("T")
            if msg_type == "t":
                self._handle_trade(event)
            elif msg_type == "q":
                self._handle_quote(event)

    def _handle_trade(self, event: dict) -> None:
        symbol = event.get("S", "").upper()
        price = event.get("p")
        size = event.get("s")
        ts_str = event.get("t")

        if not symbol or price is None:
            return

        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.now(timezone.utc)

        tick = PriceTick(
            symbol=symbol,
            price=float(price),
            timestamp=ts,
            volume=float(size) if size else None,
        )
        self._feed._prices[symbol] = tick
        self._feed._dispatch(tick)

    def _handle_quote(self, event: dict) -> None:
        symbol = event.get("S", "").upper()
        bid = event.get("bp")
        ask = event.get("ap")

        if not symbol or (bid is None and ask is None):
            return

        mid = ((bid or ask) + (ask or bid)) / 2
        tick = PriceTick(
            symbol=symbol,
            price=mid,
            timestamp=datetime.now(timezone.utc),
            bid=float(bid) if bid else None,
            ask=float(ask) if ask else None,
        )
        self._feed._prices[symbol] = tick
        self._feed._dispatch(tick)
