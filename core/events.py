# core/events.py
"""
Internal event bus for decoupled communication between PTIS subsystems.
Pub/sub pattern — components publish events, others subscribe to them.

Thread-safe, synchronous by default with async dispatch option.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  EVENT TYPES
# ══════════════════════════════════════════════

class EventType(str, Enum):
    # Trade lifecycle
    TRADE_ENTERED = "trade.entered"
    TRADE_UPDATED = "trade.updated"
    TRADE_CLOSED = "trade.closed"
    TRADE_CANCELLED = "trade.cancelled"

    # Attribution
    ATTRIBUTION_COMPLETED = "attribution.completed"
    OVERRIDE_DETECTED = "override.detected"
    BEHAVIORAL_EVENT = "behavioral.event"

    # Shadow
    SHADOW_CREATED = "shadow.created"
    SHADOW_UPDATED = "shadow.updated"
    SHADOW_CLOSED = "shadow.closed"
    SHADOW_DIVERGENCE = "shadow.divergence"

    # Session
    SESSION_STARTED = "session.started"
    SESSION_ENDED = "session.ended"
    DAILY_LIMIT_REACHED = "session.daily_limit"

    # Cognitive / real-time alerts
    COGNITIVE_ALERT = "cognitive.alert"
    STOP_WIDENED = "cognitive.stop_widened"
    POSITION_SIZE_VIOLATION = "cognitive.position_size_violation"

    # Setup detection
    SETUP_DETECTED = "setup.detected"
    EXPANSION_DETECTED = "expansion.detected"   # directional breakout/expansion at inception

    # Execution
    TRADE_EXECUTED = "execution.trade_executed"

    # Discovery
    STRATEGY_DISCOVERED = "discovery.strategy_discovered"
    DISCOVERY_PROGRESS = "discovery.progress"

    # Strategy Learning
    LEARNING_EXAMPLE_ADDED = "learning.example_added"
    STRATEGY_PROFILE_UPDATED = "learning.profile_updated"
    LEARNING_SESSION_COMPLETED = "learning.session_completed"

    # Continuous Learning
    TRADE_FEEDBACK_RECORDED = "learning.trade_feedback"
    TRADE_INSIGHTS_READY = "learning.insights_ready"  # Enough feedback for analysis (no profile mutation)

    # Watchlists
    WATCHLIST_SCANNED = "watchlist.scanned"
    WATCHLIST_STOCK_MATCHED = "watchlist.stock_matched"

    # Strategy Knowledge Engine (SKE)
    SKO_CREATED = "ske.sko_created"
    SKO_UPDATED = "ske.sko_updated"
    SKO_TIER_CHANGED = "ske.tier_changed"
    SKO_SIGNAL_GENERATED = "ske.signal_generated"
    SKO_VARIANT_DISCOVERED = "ske.variant_discovered"
    DQAL_COMPLETED = "ske.dqal_completed"

    # SKE — Behavioral Cost / Divergence
    BEHAVIORAL_COST_IMPROVED = "ske.behavioral_cost_improved"
    BEHAVIORAL_COST_REGRESSED = "ske.behavioral_cost_regressed"
    STRATEGY_DIVERGENCE_HIGH = "ske.strategy_divergence_high"
    DIVERGENCE_HISTORY_UPDATED = "ske.divergence_history_updated"

    # Improvement Lab
    IMPROVEMENT_DETECTED = "improvement.detected"
    IMPROVEMENT_VALIDATED = "improvement.validated"
    IMPROVEMENT_GRADUATED = "improvement.graduated"
    IMPROVEMENT_REJECTED = "improvement.rejected"

    # Autonomous Virtual Trader (AVT)
    AVT_STARTED = "avt.started"
    AVT_STOPPED = "avt.stopped"
    AVT_TRADE_OPENED = "avt.trade_opened"
    AVT_TRADE_CLOSED = "avt.trade_closed"
    AVT_SCAN_CYCLE = "avt.scan_cycle"
    AVT_IMPROVEMENT_APPLIED = "avt.improvement_applied"
    AVT_IMPROVEMENT_PROVEN = "avt.improvement_proven"
    AVT_IMPROVEMENT_REVERTED = "avt.improvement_reverted"
    AVT_VERSION_BUMPED = "avt.version_bumped"

    # AVT — Strategy Lifecycle
    STRATEGY_HEALTH_CHECKED = "avt.strategy_health_checked"
    STRATEGY_CALIBRATION_STARTED = "avt.calibration_started"
    STRATEGY_CALIBRATION_GRADUATED = "avt.calibration_graduated"
    STRATEGY_DEGRADATION_DETECTED = "avt.degradation_detected"
    STRATEGY_PAUSED = "avt.strategy_paused"
    STRATEGY_RESUMED = "avt.strategy_resumed"

    # Bootstrap Understanding System
    BOOTSTRAP_STARTED = "bootstrap.started"
    BOOTSTRAP_COMPLETED = "bootstrap.completed"
    BOOTSTRAP_CORRECTION = "bootstrap.correction"

    # System
    STRATEGY_RELOADED = "system.strategy_reloaded"
    REGIME_CHANGED = "system.regime_changed"


# ══════════════════════════════════════════════
#  EVENT DATA
# ══════════════════════════════════════════════

@dataclass
class Event:
    """Immutable event payload."""
    event_type: EventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    # Optional typed references
    trade_id: Optional[UUID] = None
    strategy_id: Optional[str] = None
    session_id: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"Event({self.event_type.value}, trade={self.trade_id}, "
            f"source={self.source}, keys={list(self.data.keys())})"
        )


# ══════════════════════════════════════════════
#  SUBSCRIBER TYPE
# ══════════════════════════════════════════════

EventHandler = Callable[[Event], None]


# ══════════════════════════════════════════════
#  EVENT BUS
# ══════════════════════════════════════════════

class EventBus:
    """
    Simple synchronous event bus.
    Handlers are called in registration order.
    Errors in one handler don't block others.
    """

    def __init__(self):
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._global_handlers: list[EventHandler] = []
        self._event_log: list[Event] = []
        self._max_log_size: int = 1000

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe a handler to a specific event type."""
        self._handlers[event_type].append(handler)
        logger.debug(
            "Subscribed %s to %s",
            getattr(handler, "__qualname__", str(handler)),
            event_type.value,
        )

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe a handler to ALL events."""
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a handler from event type."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event) -> int:
        """
        Publish an event. Returns number of handlers that ran successfully.
        """
        self._log_event(event)
        handled = 0

        # Type-specific handlers
        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
                handled += 1
            except Exception as e:
                logger.error(
                    "Handler %s failed for %s: %s",
                    getattr(handler, "__qualname__", str(handler)),
                    event.event_type.value,
                    e,
                    exc_info=True,
                )

        # Global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
                handled += 1
            except Exception as e:
                logger.error(
                    "Global handler %s failed for %s: %s",
                    getattr(handler, "__qualname__", str(handler)),
                    event.event_type.value,
                    e,
                    exc_info=True,
                )

        logger.debug("Published %s — %d handlers executed", event.event_type.value, handled)
        return handled

    def _log_event(self, event: Event) -> None:
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size:]

    @property
    def recent_events(self) -> list[Event]:
        return list(self._event_log[-50:])

    def handler_count(self, event_type: EventType) -> int:
        return len(self._handlers.get(event_type, [])) + len(self._global_handlers)

    def clear(self) -> None:
        self._handlers.clear()
        self._global_handlers.clear()
        self._event_log.clear()


# ── Module-level singleton ──
_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus