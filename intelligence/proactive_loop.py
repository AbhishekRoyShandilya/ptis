"""
intelligence.proactive_loop — push the PROACTIVE partner alerts into the running app.

Reuses ALL existing plumbing: it publishes `COGNITIVE_ALERT` events on the event bus; `/ws/session`
already subscribes to that type and broadcasts to the browser (LiveAlertsPage / chat). So a money move
building on the watchlist, or an open position needing management, auto-arrives in the UI — the user
never has to ask. A background loop runs the monitor on a schedule (heavy narration runs off-thread so
it never blocks the event loop).
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


def publish_alerts(alerts, bus=None) -> int:
    """Publish each ProactiveAlert as a COGNITIVE_ALERT event (→ /ws/session → UI). Returns count."""
    from core.events import Event, EventType, get_event_bus
    bus = bus or get_event_bus()
    n = 0
    for a in alerts:
        bus.publish(Event(
            event_type=EventType.COGNITIVE_ALERT,
            source="proactive-partner",
            data={"source": "partner", "kind": a.kind, "symbol": a.symbol, "urgency": a.urgency,
                  "headline": a.headline, "message": a.message, "as_of": a.as_of, "alert_key": a.key},
        ))
        n += 1
    return n


def run_once(symbols=None, positions=None, as_of=None, seen: set | None = None, bus=None) -> list:
    """One proactive pass: scan the watchlist + monitor open positions → publish the alerts. `seen`
    (a de-dup set of alert keys) persists across passes so the same move isn't re-announced."""
    from intelligence.proactive_monitor import scan_watchlist, monitor_positions
    seen = seen if seen is not None else set()
    alerts = []
    if symbols:
        alerts += scan_watchlist(list(symbols), as_of=as_of, already_alerted=seen)
    if positions:
        alerts += monitor_positions(positions, as_of=as_of, already_alerted=seen)
    if alerts:
        publish_alerts(alerts, bus=bus)
    return alerts


class ProactiveLoop:
    """Background poller: every `interval_sec`, run the monitor over the live watchlist + positions and
    publish alerts. `watchlist_fn`/`positions_fn` are injected at app startup (same pattern as the setup
    scanner) so this stays decoupled. The blocking monitor runs in a thread."""

    def __init__(self, watchlist_fn=None, positions_fn=None, interval_sec: int = 300):
        self.watchlist_fn = watchlist_fn
        self.positions_fn = positions_fn
        self.interval_sec = interval_sec
        self._running = False
        self._seen: set = set()

    async def run(self) -> None:
        self._running = True
        logger.info("ProactiveLoop started (interval %ds)", self.interval_sec)
        while self._running:
            try:
                # BOTH of these are BLOCKING (a DB read and a Groww HTTP call). Calling them straight on the
                # event loop froze it for seconds at a time — and because this loop shares the AVT scan's 300s
                # cadence, the freeze landed exactly when the bar-close-aligned scan was due to fire, delaying
                # every scan by ~8s on top of its intended 3s epsilon (measured live 2026-07-20: scans starting
                # 11-12s after the close instead of 3s). Offload them like `run_once` already was.
                symbols = list(await asyncio.to_thread(self.watchlist_fn)) if self.watchlist_fn else []
                positions = await asyncio.to_thread(self.positions_fn) if self.positions_fn else {}
                if symbols or positions:
                    alerts = await asyncio.to_thread(run_once, symbols, positions, None, self._seen)
                    if alerts:
                        logger.info("ProactiveLoop: published %d alert(s)", len(alerts))
            except Exception:
                logger.exception("ProactiveLoop pass failed")
            await asyncio.sleep(self.interval_sec)

    def stop(self) -> None:
        self._running = False
