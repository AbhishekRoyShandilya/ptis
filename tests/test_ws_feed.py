"""
Offline unit tests for marketdata.ws_feed — the live websocket tick feed.

Everything the socket doesn't touch is verified here (parser purity, tick→1m-bar→lake wiring, LTP cache +
staleness, feed-latency math, gap-fill trigger logic). The live Angel socket itself is validatable only against
market ticks (Monday) — these tests lock the deterministic core so that validation is a config flip, not a debug.
"""
from __future__ import annotations

import pandas as pd
import pytest

from marketdata.ws_feed import (
    LiveTick,
    LiveTickFeed,
    parse_angel_tick,
    _looks_like_index,
)


T2S = {"3045": "RELIANCE", "99926000": "NIFTY"}


def _ist(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# ── parser ─────────────────────────────────────────────────────────────────────

def test_parse_angel_tick_basic():
    now = _ist("2026-07-20 09:20:03")
    msg = {"token": "3045", "last_traded_price": 245050,  # paise → 2450.50
           "exchange_timestamp": 1_752_998_400_000, "volume_trade_for_the_day": 12345}
    tick = parse_angel_tick(msg, T2S, now=now)
    assert tick is not None
    assert tick.symbol == "RELIANCE"
    assert tick.price == pytest.approx(2450.50)
    assert tick.recv_ts == now
    assert tick.exch_ts is not None and tick.exch_ts.tzinfo is not None  # UTC
    assert tick.cum_volume == 12345.0


def test_parse_angel_tick_unknown_token_dropped():
    assert parse_angel_tick({"token": "999", "last_traded_price": 100}, T2S) is None


def test_parse_angel_tick_bad_price_dropped():
    now = _ist("2026-07-20 09:20:03")
    assert parse_angel_tick({"token": "3045", "last_traded_price": 0}, T2S, now=now) is None
    assert parse_angel_tick({"token": "3045", "last_traded_price": None}, T2S, now=now) is None
    assert parse_angel_tick({"token": "3045"}, T2S, now=now) is None


def test_parse_angel_tick_missing_exch_ts_uses_recv():
    now = _ist("2026-07-20 09:20:03")
    tick = parse_angel_tick({"token": "3045", "last_traded_price": 10000}, T2S, now=now)
    assert tick is not None and tick.exch_ts is None and tick.recv_ts == now


def test_parse_non_dict():
    assert parse_angel_tick(b"binary", T2S) is None
    assert parse_angel_tick(None, T2S) is None


# ── feed: tick → 1m bar → lake writer ────────────────────────────────────────────

def test_feed_builds_and_writes_1m_bar():
    written = []
    feed = LiveTickFeed(lake_writer=lambda sym, itype, row: written.append((sym, itype, row)))

    # three ticks in minute 09:20, then one in 09:21 → rolls the 09:20 bar
    feed.ingest(LiveTick("RELIANCE", 2450.0, _ist("2026-07-20 09:20:05"), cum_volume=100))
    feed.ingest(LiveTick("RELIANCE", 2455.0, _ist("2026-07-20 09:20:30"), cum_volume=250))
    feed.ingest(LiveTick("RELIANCE", 2448.0, _ist("2026-07-20 09:20:55"), cum_volume=400))
    assert written == []  # bar not closed yet
    feed.ingest(LiveTick("RELIANCE", 2452.0, _ist("2026-07-20 09:21:02"), cum_volume=420))

    assert len(written) == 1
    sym, itype, row = written[0]
    assert sym == "RELIANCE" and itype == "EQ"
    assert row["timestamp"] == _ist("2026-07-20 09:20:00")
    assert row["open"] == 2450.0 and row["high"] == 2455.0
    assert row["low"] == 2448.0 and row["close"] == 2448.0
    assert row["volume"] == 300.0  # cumulative 400 − 100 at bar start


def test_feed_index_inst_type():
    written = []
    feed = LiveTickFeed(lake_writer=lambda sym, itype, row: written.append((sym, itype, row)))
    feed.ingest(LiveTick("NIFTY", 24000.0, _ist("2026-07-20 09:20:05")))
    feed.ingest(LiveTick("NIFTY", 24010.0, _ist("2026-07-20 09:21:05")))
    assert written and written[0][1] == "INDEX"


def test_bad_bar_write_does_not_raise():
    def boom(sym, itype, row):
        raise ValueError("disk full")
    feed = LiveTickFeed(lake_writer=boom)
    feed.ingest(LiveTick("RELIANCE", 2450.0, _ist("2026-07-20 09:20:05")))
    feed.ingest(LiveTick("RELIANCE", 2451.0, _ist("2026-07-20 09:21:05")))  # rolls → boom, swallowed


# ── LTP cache + staleness ────────────────────────────────────────────────────────

def test_ltp_cache_latest_wins():
    feed = LiveTickFeed(lake_writer=lambda *a: None)
    feed.ingest(LiveTick("RELIANCE", 2450.0, _ist("2026-07-20 09:20:05")))
    feed.ingest(LiveTick("RELIANCE", 2460.0, _ist("2026-07-20 09:20:06")))
    assert feed.get_ltp("RELIANCE") == 2460.0
    assert feed.get_ltp("MISSING") is None


def test_ltp_age_positive():
    feed = LiveTickFeed(lake_writer=lambda *a: None)
    feed.ingest(LiveTick("RELIANCE", 2450.0, _ist("2000-01-01 09:20:05")))
    age = feed.ltp_age_s("RELIANCE")
    assert age is not None and age > 0
    assert feed.ltp_age_s("MISSING") is None


# ── latency stats ─────────────────────────────────────────────────────────────────

def test_latency_stats():
    feed = LiveTickFeed(lake_writer=lambda *a: None)
    # recv 500ms after exch: recv_ts IST-naive 09:20:00.5 == exch 03:50:00.0 UTC + 0.5s
    exch = pd.Timestamp("2026-07-20 03:50:00", tz="UTC")
    recv = pd.Timestamp("2026-07-20 09:20:00.5")  # IST-naive = 03:50:00.5 UTC
    feed.ingest(LiveTick("RELIANCE", 2450.0, recv, exch_ts=exch))
    st = feed.latency_stats()
    assert st["samples"] == 1
    assert st["p50"] == pytest.approx(0.5, abs=1e-6)


def test_latency_stats_empty():
    feed = LiveTickFeed(lake_writer=lambda *a: None)
    st = feed.latency_stats()
    assert st["samples"] == 0 and st["ticks"] == 0


# ── gap-fill trigger logic ─────────────────────────────────────────────────────────

def test_gap_fill_calls_filler_only_past_last_bar():
    calls = []
    feed = LiveTickFeed(lake_writer=lambda *a: None,
                        gap_filler=lambda sym, start, end: calls.append((sym, start, end)) or 3)
    # seed a last-bar timestamp well in the past so a gap exists
    feed._last_bar_ts["RELIANCE"] = pd.Timestamp("2000-01-01 09:20:00")
    n = feed.gap_fill(["RELIANCE", "NOBAR"])  # NOBAR has no last bar → skipped
    assert n == 3
    assert len(calls) == 1 and calls[0][0] == "RELIANCE"


def test_gap_fill_no_filler_is_noop():
    feed = LiveTickFeed(lake_writer=lambda *a: None)
    assert feed.gap_fill(["RELIANCE"]) == 0


def test_looks_like_index():
    assert _looks_like_index("NIFTY") and _looks_like_index("banknifty") and _looks_like_index("SENSEX")
    assert not _looks_like_index("RELIANCE")


# ── REAL Angel frames (captured live 2026-07-19) — these lock the wire format, not my guess of it ──

REAL_QUOTE = {  # mode 2 (QUOTE) — RELIANCE
    "subscription_mode": 2, "exchange_type": 1, "token": "2885", "sequence_number": 24582,
    "exchange_timestamp": 1784337906877, "last_traded_price": 132720, "subscription_mode_val": "QUOTE",
    "last_traded_quantity": 31, "average_traded_price": 132139, "volume_trade_for_the_day": 18302218,
    "total_buy_quantity": 2797.0, "total_sell_quantity": 0.0, "open_price_of_the_day": 130000,
    "high_price_of_the_day": 133030, "low_price_of_the_day": 129610, "closed_price": 129660,
}
REAL_LTP = {  # mode 1 (LTP) — NIFTY. NOTE: no volume field at all.
    "subscription_mode": 1, "exchange_type": 1, "token": "99926000", "sequence_number": 0,
    "exchange_timestamp": 1784284694000, "last_traded_price": 2433430, "subscription_mode_val": "LTP",
}
REAL_T2S = {"2885": "RELIANCE", "99926000": "NIFTY"}


def test_real_quote_frame_parses():
    """Parser against a REAL captured frame — paise/100 and cumulative volume verified on the wire."""
    t = parse_angel_tick(REAL_QUOTE, REAL_T2S, now=_ist("2026-07-20 09:20:03"))
    assert t is not None and t.symbol == "RELIANCE"
    assert t.price == pytest.approx(1327.20)          # 132720 paise
    assert t.cum_volume == 18302218.0


def test_real_ltp_frame_has_no_volume():
    """Mode 1 (LTP) carries NO volume -> tick bars would be volume=0 -> rvol/effort corrupted.
    This is WHY AngelWSAdapter must subscribe in mode 2 (QUOTE)."""
    t = parse_angel_tick(REAL_LTP, REAL_T2S, now=_ist("2026-07-20 09:20:03"))
    assert t is not None and t.price == pytest.approx(24334.30)
    assert t.cum_volume == 0.0
    from marketdata.ws_feed import AngelWSAdapter
    assert AngelWSAdapter.MODE_QUOTE == 2
    import inspect
    assert inspect.signature(AngelWSAdapter.__init__).parameters["mode"].default == 2


def test_adapter_callbacks_tolerate_extra_args():
    """The SDK calls on_close with 4 positional args; a strict signature raises and kills the
    close/reconnect (gap-fill) path."""
    import inspect
    from marketdata.ws_feed import AngelWSAdapter
    for name in ("_on_open", "_on_data", "_on_error", "_on_close"):
        params = inspect.signature(getattr(AngelWSAdapter, name)).parameters
        assert any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()), \
            f"{name} must accept *args (SDK varies the arg count)"


def test_bar_volume_carries_baseline_across_bars():
    """Bar volume = cum(this bar end) - cum(PREVIOUS bar end). Baselining off the first tick INSIDE the
    bar drops inter-bar volume and reports 0 for a single-tick bar."""
    written = []
    feed = LiveTickFeed(lake_writer=lambda s, i, r: written.append(r))
    feed.ingest(LiveTick("RELIANCE", 100.0, _ist("2026-07-20 09:20:05"), cum_volume=1000))
    feed.ingest(LiveTick("RELIANCE", 101.0, _ist("2026-07-20 09:20:55"), cum_volume=1500))
    feed.ingest(LiveTick("RELIANCE", 102.0, _ist("2026-07-20 09:21:05"), cum_volume=1800))  # rolls 09:20
    feed.ingest(LiveTick("RELIANCE", 103.0, _ist("2026-07-20 09:22:05"), cum_volume=2000))  # rolls 09:21
    assert written[0]["volume"] == 500.0    # 1500-1000
    # 09:21 had ONE tick (cum 1800); baseline must be the prior bar's end (1500) -> 300, not 0
    assert written[1]["volume"] == 300.0


def test_cumulative_volume_day_rollover_resets():
    """Broker resets cumulative volume each session; a DROP must re-baseline, never go negative."""
    written = []
    feed = LiveTickFeed(lake_writer=lambda s, i, r: written.append(r))
    feed.ingest(LiveTick("RELIANCE", 100.0, _ist("2026-07-20 15:29:05"), cum_volume=900000))
    feed.ingest(LiveTick("RELIANCE", 101.0, _ist("2026-07-21 09:15:05"), cum_volume=1200))  # new day
    feed.ingest(LiveTick("RELIANCE", 102.0, _ist("2026-07-21 09:16:05"), cum_volume=1500))
    for r in written:
        assert r["volume"] >= 0.0
