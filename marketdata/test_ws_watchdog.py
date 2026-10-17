"""Offline test of the AngelWSAdapter watchdog (2026-10-05 leak fix): no network, a fake socket and a fake clock.
  PTIS_ENV=test PYTHONPATH=. python -m pytest marketdata/test_ws_watchdog.py -q
"""
import threading, time
from datetime import datetime
import marketdata.ws_feed as W


class FakeSock:
    def __init__(self): self.MAX_RETRY_ATTEMPT, self.closed = 1000, False
    def close_connection(self): self.closed = True


class SilentFeed:
    def seconds_since_tick(self): return 999.0          # the market is open but no ticks ever arrive


def make(monkeypatch):
    a = W.AngelWSAdapter.__new__(W.AngelWSAdapter)
    a._feed, a._restarts, a._wd_stop, a._sws, a.socks = SilentFeed(), 0, threading.Event(), FakeSock(), []
    def fake_start(_watchdog=True):
        a.socks.append(a._sws); a._sws = FakeSock()
    a.start = fake_start
    monkeypatch.setattr(W, "_now_ist", lambda: datetime(2026, 10, 5, 10, 0))
    return a


def test_backoff_cap_and_neutralise(monkeypatch):
    a = make(monkeypatch)
    clock = {"t": 0.0}
    monkeypatch.setattr(W.time, "time", lambda: clock["t"])
    waits = {"n": 0}
    def fake_wait(_):                                    # each watchdog tick advances the fake clock by 5 s
        waits["n"] += 1; clock["t"] += 5.0
        return waits["n"] > 2000                         # about 10,000 fake seconds (~2.8 h of total silence)
    a._wd_stop.wait = fake_wait
    W.AngelWSAdapter._start_watchdog(a, silence_s=20.0)
    a._wd.join(timeout=10)
    assert a._restarts == 8, a._restarts                 # capped, not one rebuild every 5 s (was ~2,000)
    assert all(s.MAX_RETRY_ATTEMPT == 0 and s.closed for s in a.socks), "abandoned sockets must be neutralised"


def test_no_rebuild_outside_market_hours(monkeypatch):
    a = make(monkeypatch)
    monkeypatch.setattr(W, "_now_ist", lambda: datetime(2026, 10, 5, 9, 10))   # pre-open: the 10-05 start time
    n = {"k": 0}
    def fake_wait(_):
        n["k"] += 1; return n["k"] > 50
    a._wd_stop.wait = fake_wait
    W.AngelWSAdapter._start_watchdog(a, silence_s=20.0)
    a._wd.join(timeout=10)
    assert a._restarts == 0
