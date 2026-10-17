# avt/triage.py
"""
AVT Triage — Message Queue for User
=====================================

Simple layer between observer and conversation/UI.

ALL observations from the observer are behavioral/direct.
There is no hypothesis routing here — that's internal to AVT's
self_improver which gets hypotheses from improvement_lab and
proves them with its own trading.

This module:
  1. Receives observations from the observer
  2. Assigns priority
  3. Queues messages for the conversation layer / UI
  4. Tracks acknowledgement
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from avt.observer import AVTObserver, Observation, ObservationType

logger = logging.getLogger(__name__)


# Priority boosts for specific observation types
_PRIORITY_MAP = {
    ObservationType.REVENGE_TRADE: 10,
    ObservationType.POSITION_OVERSIZE: 8,
    ObservationType.SL_WIDENED: 7,
    ObservationType.OVERTRADING: 6,
    ObservationType.PLAN_DEVIATION: 6,
    ObservationType.DISCIPLINE_DROP: 5,
    ObservationType.EARLY_EXIT: 4,
    ObservationType.EMOTION_FLAG: 5,
    ObservationType.IMPROVEMENT_PROVEN: 9,      # high — user should see this
    ObservationType.IMPROVEMENT_STARTED: 3,     # low — informational
    ObservationType.IMPROVEMENT_REVERTED: 2,    # low — informational
    ObservationType.GOOD_DISCIPLINE: 2,
}

_SEVERITY_BASE = {
    "critical": 8,
    "warning": 5,
    "info": 3,
    "positive": 2,
}


class AVTTriage:
    """
    Routes observations to the user's message queue.

    No hypothesis routing. No AVT proving ground integration.
    Just: observation → priority → queue → UI.
    """

    def __init__(self, observer: AVTObserver):
        self._observer = observer
        self._message_queue: List[Dict[str, Any]] = []
        self._max_queue = 200
        self._on_message: Optional[Callable[[Dict[str, Any]], None]] = None

        # Wire observer
        self._observer.on_observation(self._on_observation)

    def set_message_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        """Called when a new message is queued (for WebSocket push)."""
        self._on_message = cb

    # ── Core ──────────────────────────────────────────────────

    def _on_observation(self, obs: Observation) -> None:
        """Convert observation to a user message and queue it."""
        priority = _PRIORITY_MAP.get(
            obs.type,
            _SEVERITY_BASE.get(obs.severity, 3),
        )

        msg = {
            "id": obs.id,
            "type": obs.type,
            "title": obs.title,
            "message": obs.description,
            "severity": obs.severity,
            "priority": priority,
            "strategy_id": obs.strategy_id,
            "data": obs.data,
            "timestamp": obs.timestamp.isoformat(),
            "acknowledged": False,
        }

        self._message_queue.append(msg)
        if len(self._message_queue) > self._max_queue:
            self._message_queue = self._message_queue[-self._max_queue:]

        # Push callback (for WebSocket)
        if self._on_message:
            try:
                self._on_message(msg)
            except Exception as e:
                logger.debug("Triage message callback error: %s", e)

    # ── Query ─────────────────────────────────────────────────

    def get_pending_messages(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get unacknowledged messages, highest priority first."""
        unacked = [m for m in self._message_queue if not m["acknowledged"]]
        unacked.sort(key=lambda m: m["priority"], reverse=True)
        return unacked[:limit]

    def acknowledge_message(self, message_id: str) -> bool:
        """Mark a message as acknowledged."""
        for msg in self._message_queue:
            if msg["id"] == message_id:
                msg["acknowledged"] = True
                return True
        return False

    def get_unread_count(self) -> int:
        """Count of unacknowledged messages."""
        return sum(1 for m in self._message_queue if not m["acknowledged"])

    def get_critical_count(self) -> int:
        """Count of unacked critical messages."""
        return sum(
            1 for m in self._message_queue
            if not m["acknowledged"] and m["severity"] == "critical"
        )
