"""
Execution Bridge
=================
Translates PTIS trade signals into broker-compatible order formats.
Acts as a thin adaptor layer so the rest of the system never needs to know
which broker is connected.

Architecture:
  - BrokerAdapter (abstract base)
  - PaperBrokerAdapter (default, no real execution — logs orders only)
  - Future: ZerodhaAdapter, AngelOneAdapter — drop-in replacements

Safety properties (live mode only):
  1. Durability  — order intent is written to DB *before* broker submission.
                   If the process crashes after send-but-before-ack, the
                   PENDING intent record proves the order was placed.
  2. Idempotency — callers provide an idempotency_key; duplicate submissions
                   with the same key return the cached result without hitting
                   the broker a second time.
  3. Kill switch — operator can block all live orders via DB flag
                   (key="kill_switch", value="active") or env var
                   PTIS_KILL_SWITCH=1. Checked before every live submit.

IMPORTANT: Real broker integrations require platform-specific API keys
configured in config/settings.py (BROKER_NAME, BROKER_API_KEY, BROKER_SECRET).
This module performs NO live execution by default.
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from core.events import Event, EventType, get_event_bus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order model
# ---------------------------------------------------------------------------

@dataclass
class OrderRequest:
    symbol: str
    direction: str           # "long" | "short"
    quantity: int
    order_type: str          # "market" | "limit" | "sl" | "sl-m"
    price: Optional[float]   # None for market orders
    trigger_price: Optional[float] = None   # For SL orders
    product: str = "MIS"     # MIS (intraday) | CNC (delivery) | NRML (F&O)
    exchange: str = "NSE"
    tag: Optional[str] = None        # strategy identifier
    trade_id: Optional[str] = None   # PTIS trade ID for reconciliation
    idempotency_key: Optional[str] = None  # Caller-provided; auto-generated if None


@dataclass
class OrderResult:
    order_id: str
    status: str              # "submitted" | "filled" | "rejected" | "paper" | "cached"
    symbol: str
    direction: str
    quantity: int
    fill_price: Optional[float]
    fill_time: Optional[datetime]
    message: str
    broker: str
    intent_id: Optional[str] = None   # DB order_intents.id for audit trail


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BrokerAdapter:
    broker_name: str = "unknown"

    def submit_order(self, order: OrderRequest) -> OrderResult:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def get_positions(self) -> list[dict]:
        raise NotImplementedError

    def is_connected(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Paper broker (safe default)
# ---------------------------------------------------------------------------

class PaperBrokerAdapter(BrokerAdapter):
    """
    Simulates order execution for paper trading / testing.
    All orders are immediately "filled" at requested price.
    No real money involved — no durability/idempotency checks applied.
    """
    broker_name = "paper"

    def __init__(self):
        self._orders: list[OrderResult] = []

    def submit_order(self, order: OrderRequest) -> OrderResult:
        fill_price = order.price or 0.0
        result = OrderResult(
            order_id=str(uuid.uuid4()),
            status="paper",
            symbol=order.symbol,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            fill_time=datetime.now(timezone.utc),
            message=f"[PAPER] {order.direction.upper()} {order.quantity} {order.symbol} @ {fill_price}",
            broker="paper",
        )
        self._orders.append(result)
        logger.info(result.message)
        return result

    def cancel_order(self, order_id: str) -> bool:
        logger.info(f"[PAPER] Cancel order {order_id}")
        return True

    def get_positions(self) -> list[dict]:
        return []

    def is_connected(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Execution Bridge
# ---------------------------------------------------------------------------

class ExecutionBridge:
    """
    Validates orders, forwards to the broker adapter, and publishes events.

    For live orders: writes intent to DB before submission, checks kill switch,
    enforces idempotency. Paper orders skip DB persistence (no money at stake).
    """

    def __init__(self, adapter: Optional[BrokerAdapter] = None):
        self._adapter = adapter or PaperBrokerAdapter()
        self._bus = get_event_bus()

    @property
    def broker(self) -> str:
        return self._adapter.broker_name

    def is_live(self) -> bool:
        return self._adapter.broker_name != "paper" and self._adapter.is_connected()

    # ── Public entry point ────────────────────────────────────────

    def submit(self, order: OrderRequest, db=None) -> OrderResult:
        """
        Validate + submit an order.

        For live orders db is required — used for kill switch check,
        idempotency lookup, and intent journaling.
        For paper orders db is optional (no persistence).
        """
        # Pre-flight validation
        errors = self._validate(order)
        if errors:
            return self._rejected(order, f"Order rejected: {'; '.join(errors)}")

        # Paper path — skip all DB safety checks
        if not self.is_live():
            result = self._adapter.submit_order(order)
            self._publish(result, order)
            return result

        # ── Live path — full safety checks ──────────────────────

        if db is None:
            return self._rejected(
                order,
                "Live order requires db session (kill switch + idempotency checks need DB)",
            )

        # 1. Kill switch — env var wins, then DB flag
        if self._is_kill_switch_active(db):
            return self._rejected(order, "Kill switch is active — all live orders blocked")

        # 2. Guardrails
        try:
            from automation.guardrails import GuardrailEngine
            ok, reason = GuardrailEngine().allow_trade(db)
            if not ok:
                return self._rejected(order, f"Guardrail blocked: {reason}")
        except Exception as e:
            logger.warning("Guardrail check failed: %s — proceeding with caution", e)

        # 3. Idempotency key — auto-generate if not provided
        ikey = order.idempotency_key or self._auto_ikey(order)

        # 4. Check if this key was already submitted
        from core.models import OrderIntent
        existing = db.query(OrderIntent).filter(
            OrderIntent.idempotency_key == ikey
        ).first()

        if existing and existing.status in ("SUBMITTED", "FILLED"):
            logger.info(
                "Duplicate order suppressed (idempotency_key=%s, existing_status=%s)",
                ikey, existing.status,
            )
            return OrderResult(
                order_id=existing.broker_order_id or existing.id,
                status="cached",
                symbol=existing.symbol,
                direction=existing.direction,
                quantity=existing.quantity,
                fill_price=None,
                fill_time=None,
                message=f"Duplicate suppressed — order already {existing.status}",
                broker=self.broker,
                intent_id=str(existing.id),
            )

        # 5. Write intent to DB BEFORE hitting the broker
        intent_id = uuid.uuid4()
        intent = OrderIntent(
            id=intent_id,
            idempotency_key=ikey,
            symbol=order.symbol,
            direction=order.direction,
            quantity=order.quantity,
            order_type=order.order_type,
            price=order.price,
            product=order.product,
            tag=order.tag,
            trade_id=order.trade_id,
            broker_name=self.broker,
            is_live=True,
            request_payload={
                "symbol": order.symbol, "direction": order.direction,
                "quantity": order.quantity, "order_type": order.order_type,
                "price": order.price, "product": order.product,
                "tag": order.tag, "trade_id": order.trade_id,
            },
            status="PENDING",
        )
        db.add(intent)
        db.commit()   # commit intent BEFORE broker call — crash-safe

        # 6. Submit to broker
        try:
            result = self._adapter.submit_order(order)
            intent.status = "SUBMITTED"
            intent.broker_order_id = result.order_id
            intent.submitted_at = datetime.now(timezone.utc)
            intent.response_payload = {
                "order_id": result.order_id, "status": result.status,
                "fill_price": result.fill_price, "message": result.message,
            }
        except Exception as e:
            intent.status = "ERROR"
            intent.error_message = str(e)
            db.commit()
            logger.error("Broker submission failed for intent %s: %s", intent_id, e)
            raise

        db.commit()
        result.intent_id = str(intent_id)

        # 7. Publish event
        self._publish(result, order)
        return result

    # ── Kill switch ───────────────────────────────────────────────

    def _is_kill_switch_active(self, db) -> bool:
        """Check env var (instant) then DB flag (persistent)."""
        if os.environ.get("PTIS_KILL_SWITCH", "0") == "1":
            logger.warning("Kill switch active via PTIS_KILL_SWITCH env var")
            return True
        try:
            from core.models import SystemFlag
            flag = db.query(SystemFlag).filter(SystemFlag.key == "kill_switch").first()
            if flag and flag.value == "active":
                logger.warning("Kill switch active (DB flag, reason: %s)", flag.reason or "none")
                return True
        except Exception as e:
            logger.warning("Kill switch DB check failed: %s — defaulting to ALLOW", e)
        return False

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _auto_ikey(order: OrderRequest) -> str:
        """Deterministic key from order fields — same logical order = same key."""
        raw = f"{order.trade_id}:{order.symbol}:{order.direction}:{order.quantity}:{order.order_type}"
        return hashlib.sha256(raw.encode()).hexdigest()[:64]

    def _rejected(self, order: OrderRequest, message: str) -> OrderResult:
        logger.warning(message)
        return OrderResult(
            order_id="", status="rejected", symbol=order.symbol,
            direction=order.direction, quantity=order.quantity,
            fill_price=None, fill_time=None, message=message, broker=self.broker,
        )

    def _publish(self, result: OrderResult, order: OrderRequest) -> None:
        self._bus.publish(Event(
            event_type=EventType.TRADE_EXECUTED,
            source="execution_bridge",
            data={
                "order_id": result.order_id, "status": result.status,
                "symbol": result.symbol, "direction": result.direction,
                "quantity": result.quantity, "fill_price": result.fill_price,
                "broker": result.broker, "trade_id": order.trade_id,
                "intent_id": result.intent_id,
            },
        ))

    @staticmethod
    def _validate(order: OrderRequest) -> list[str]:
        errors = []
        if not order.symbol:
            errors.append("symbol required")
        if order.direction not in ("long", "short"):
            errors.append(f"invalid direction: {order.direction}")
        if order.quantity <= 0:
            errors.append(f"quantity must be > 0, got {order.quantity}")
        if order.order_type in ("limit", "sl") and order.price is None:
            errors.append("price required for limit/sl orders")
        return errors


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_bridge: Optional[ExecutionBridge] = None


def get_execution_bridge() -> ExecutionBridge:
    """Return the module-level bridge singleton (lazy-init)."""
    global _bridge
    if _bridge is None:
        try:
            from config.settings import settings
            broker_name = getattr(settings, "BROKER_NAME", "paper").lower()
        except Exception:
            broker_name = "paper"

        if broker_name == "paper":
            adapter = PaperBrokerAdapter()
        else:
            # Future: load Angel One / Zerodha adapter from broker_name
            logger.warning(f"No live adapter for broker '{broker_name}' — using paper")
            adapter = PaperBrokerAdapter()

        _bridge = ExecutionBridge(adapter)
        logger.info(
            "ExecutionBridge initialised: broker=%s, live=%s",
            _bridge.broker, _bridge.is_live(),
        )
    return _bridge
