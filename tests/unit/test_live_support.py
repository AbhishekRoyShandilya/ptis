import pytest

from intelligence.live_support import LiveCognitiveSupport, get_live_support
from core.events import Event, EventType


def test_live_support_instantiation():
    # Should return a singleton and support basic interface
    support1 = get_live_support()
    support2 = get_live_support()
    assert support1 is support2
    assert isinstance(support1, LiveCognitiveSupport)


def test_publish_alert_event(bus=None):
    # verify that publishing on event bus calls handler
    from core.events import get_event_bus

    bus = get_event_bus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe(EventType.COGNITIVE_ALERT, handler)
    support = get_live_support()
    support._publish({"foo": "bar"})
    assert len(received) == 1
    assert received[0].data.get("cognitive_alert")
