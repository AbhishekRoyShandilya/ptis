# api/routes/websocket.py
"""
WebSocket endpoints for real-time updates.
- Live shadow trade price feed
- Behavioral alerts
- Trade events
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session as DBSession

from api.auth import decode_token
from core.database import SessionLocal
from core.events import Event, EventType, get_event_bus
from core.models import Trade, User
from shadow.engine import ShadowEngine
from shadow.models import ShadowTrade

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])


async def _authenticate_ws(websocket: WebSocket, token: str | None) -> User | None:
    """Validate JWT token from WebSocket query param and return User, or None."""
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return None
    try:
        payload = decode_token(token)
    except Exception:
        await websocket.close(code=4001, reason="Invalid token")
        return None

    if payload.get("type") != "access":
        await websocket.close(code=4001, reason="Invalid token type")
        return None

    user_id = payload.get("sub")
    if not user_id:
        await websocket.close(code=4001, reason="Invalid token payload")
        return None

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    finally:
        db.close()

    if not user:
        await websocket.close(code=4001, reason="User not found")
        return None

    return user


# ══════════════════════════════════════════════
#  CONNECTION MANAGER
# ══════════════════════════════════════════════

class ConnectionManager:
    """Manages WebSocket connections by channel."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        await websocket.accept()
        if channel not in self._connections:
            self._connections[channel] = []
        self._connections[channel].append(websocket)
        logger.info("WebSocket connected to channel '%s' (%d total)",
                     channel, len(self._connections[channel]))

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        conns = self._connections.get(channel, [])
        if websocket in conns:
            conns.remove(websocket)
        logger.info("WebSocket disconnected from channel '%s'", channel)

    async def broadcast(self, channel: str, data: dict) -> None:
        """Send data to all connections on a channel."""
        conns = self._connections.get(channel, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_json(data)
            except Exception:
                logger.debug("WS broadcast failed on channel '%s', marking dead", channel)
                dead.append(ws)

        for ws in dead:
            conns.remove(ws)

    async def send_personal(self, websocket: WebSocket, data: dict) -> None:
        try:
            await websocket.send_json(data)
        except Exception:
            logger.debug("WS personal send failed")

    @property
    def active_connections(self) -> int:
        return sum(len(conns) for conns in self._connections.values())


manager = ConnectionManager()


# ══════════════════════════════════════════════
#  SHADOW TRADE LIVE FEED
# ══════════════════════════════════════════════

@router.websocket("/ws/shadow/{real_trade_id}")
async def shadow_live_feed(
    websocket: WebSocket,
    real_trade_id: str,
    token: str | None = Query(None),
):
    """
    WebSocket for live shadow trade updates.
    Requires ?token=<jwt> query param for authentication.

    Client connects → receives real-time shadow P&L updates.

    Messages sent to client:
    - shadow_update: {shadow_id, type, pnl, r_multiple, status, ...}
    - shadow_closed: {shadow_id, exit_reason, final_pnl, ...}
    - comparison: {full comparison report when real trade closes}

    Messages from client:
    - price_update: {price, timestamp}
    """
    user = await _authenticate_ws(websocket, token)
    if not user:
        return

    # Verify user owns the trade
    db = SessionLocal()
    try:
        trade = db.query(Trade).filter(
            Trade.id == UUID(real_trade_id), Trade.trader_id == user.id
        ).first()
    finally:
        db.close()
    if not trade:
        await websocket.close(code=4003, reason="Trade not found")
        return

    channel = f"shadow_{real_trade_id}"
    await manager.connect(websocket, channel)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "price_update":
                price = data.get("price")
                if price:
                    db = SessionLocal()
                    try:
                        engine = ShadowEngine()
                        results = engine.update_session_prices(
                            db=db,
                            real_trade_id=UUID(real_trade_id),
                            price=float(price),
                            timestamp=datetime.now(timezone.utc),
                        )

                        # Broadcast updates to all connected clients
                        await manager.broadcast(channel, {
                            "type": "shadow_update",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "price": price,
                            "shadows": results,
                        })

                        # Check if any shadows closed
                        for r in results:
                            if r.get("status") == "closed":
                                await manager.broadcast(channel, {
                                    "type": "shadow_closed",
                                    **r,
                                })

                    finally:
                        db.close()

            elif msg_type == "ping":
                await manager.send_personal(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        manager.disconnect(websocket, channel)


# ══════════════════════════════════════════════
#  TRADING SESSION LIVE FEED
# ══════════════════════════════════════════════

@router.websocket("/ws/session")
async def session_live_feed(
    websocket: WebSocket,
    token: str | None = Query(None),
):
    """
    WebSocket for live trading session updates.
    Requires ?token=<jwt> query param for authentication.
    Receives all trade events, behavioral alerts, and session state changes.
    """
    user = await _authenticate_ws(websocket, token)
    if not user:
        return

    channel = f"session_{user.id}"
    await manager.connect(websocket, channel)

    # Wire event bus to broadcast
    bus = get_event_bus()

    async def _broadcast_event(event_data: dict):
        await manager.broadcast(channel, event_data)

    def on_event(event: Event):
        """Sync handler that queues async broadcast."""
        event_data = {
            "type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "trade_id": str(event.trade_id) if event.trade_id else None,
            "data": event.data,
        }
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_broadcast_event(event_data))
        except RuntimeError:
            pass

    # Subscribe to relevant events
    bus.subscribe(EventType.TRADE_ENTERED, on_event)
    bus.subscribe(EventType.TRADE_CLOSED, on_event)
    bus.subscribe(EventType.OVERRIDE_DETECTED, on_event)
    bus.subscribe(EventType.SHADOW_CREATED, on_event)
    bus.subscribe(EventType.SHADOW_DIVERGENCE, on_event)
    bus.subscribe(EventType.ATTRIBUTION_COMPLETED, on_event)
    # cognitive support alerts (stop widen, oversizing, etc.)
    bus.subscribe(EventType.COGNITIVE_ALERT, on_event)
    # setup scanner alerts (strategy matches found in market)
    bus.subscribe(EventType.SETUP_DETECTED, on_event)
    # AVT proven improvements ready for user review
    bus.subscribe(EventType.AVT_IMPROVEMENT_PROVEN, on_event)

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await manager.send_personal(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)
        # Unsubscribe
        bus.unsubscribe(EventType.TRADE_ENTERED, on_event)
        bus.unsubscribe(EventType.TRADE_CLOSED, on_event)
        bus.unsubscribe(EventType.OVERRIDE_DETECTED, on_event)
        bus.unsubscribe(EventType.SHADOW_CREATED, on_event)
        bus.unsubscribe(EventType.SHADOW_DIVERGENCE, on_event)
        bus.unsubscribe(EventType.ATTRIBUTION_COMPLETED, on_event)