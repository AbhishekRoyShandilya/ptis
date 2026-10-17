# shadow/price_adapters/polygon_ws.py
"""
Polygon.io WebSocket adapter — real-time trade-level price feed.

Requires: POLYGON_API_KEY in settings.
Supports: trades (T.*), minute aggregates (AM.*).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from shadow.price_feed import PriceFeedManager, PriceTick

logger = logging.getLogger(__name__)

_POLYGON_WS_URL = "wss://socket.polygon.io/stocks"


class PolygonWebSocketAdapter:
    """
    Connects to Polygon.io WebSocket for real-time trade data.

    Usage:
        adapter = PolygonWebSocketAdapter(feed, api_key="...")
        await adapter.start()
        ...
        await adapter.stop()
    """

    def __init__(
        self,
        feed: PriceFeedManager,
        api_key: str,
    ):
        self._feed = feed
        self._api_key = api_key
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._ws = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Polygon WS adapter starting")

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
        logger.info("Polygon WS adapter stopped")

    async def _run(self) -> None:
        import aiohttp

        retry_delay = 1.0

        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(_POLYGON_WS_URL) as ws:
                        self._ws = ws
                        retry_delay = 1.0

                        await ws.send_json({
                            "action": "auth",
                            "params": self._api_key,
                        })

                        symbols = list(self._feed._subscribed_symbols)
                        if symbols:
                            params = ",".join(f"T.{s}" for s in symbols)
                            await ws.send_json({
                                "action": "subscribe",
                                "params": params,
                            })
                            logger.info(
                                "Polygon: subscribed to %d symbols", len(symbols)
                            )

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
                    "Polygon WS error (reconnecting in %.0fs): %s",
                    retry_delay, exc,
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)

    def _handle_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        events = data if isinstance(data, list) else [data]
        for event in events:
            ev_type = event.get("ev")
            if ev_type == "T":
                self._handle_trade(event)
            elif ev_type == "AM":
                self._handle_aggregate(event)

    def _handle_trade(self, event: dict) -> None:
        symbol = event.get("sym", "").upper()
        price = event.get("p")
        size = event.get("s")
        ts_ms = event.get("t")

        if not symbol or price is None:
            return

        ts = (
            datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            if ts_ms
            else datetime.now(timezone.utc)
        )

        tick = PriceTick(
            symbol=symbol,
            price=float(price),
            timestamp=ts,
            volume=float(size) if size else None,
        )
        self._feed._prices[symbol] = tick
        self._feed._dispatch(tick)

    def _handle_aggregate(self, event: dict) -> None:
        symbol = event.get("sym", "").upper()
        close = event.get("c")
        volume = event.get("v")
        ts_ms = event.get("s")

        if not symbol or close is None:
            return

        ts = (
            datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            if ts_ms
            else datetime.now(timezone.utc)
        )

        tick = PriceTick(
            symbol=symbol,
            price=float(close),
            timestamp=ts,
            volume=float(volume) if volume else None,
        )
        self._feed._prices[symbol] = tick
        self._feed._dispatch(tick)

    async def subscribe_symbols(self, symbols: list[str]) -> None:
        if self._ws:
            params = ",".join(f"T.{s.upper()}" for s in symbols)
            await self._ws.send_json({"action": "subscribe", "params": params})
            for s in symbols:
                self._feed.subscribe(s)

    async def unsubscribe_symbols(self, symbols: list[str]) -> None:
        if self._ws:
            params = ",".join(f"T.{s.upper()}" for s in symbols)
            await self._ws.send_json({"action": "unsubscribe", "params": params})
            for s in symbols:
                self._feed.unsubscribe(s)
