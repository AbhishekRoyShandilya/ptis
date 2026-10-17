"""
marketdata.ws_feed — the LIVE websocket tick feed (the <1s freshness track).

Pipeline (one code path — tick-built bars flow downstream exactly like REST-fetched ones):

    broker WS ─▶ parse_*_tick() ─▶ LiveTick ─▶ LiveTickFeed.ingest()
                                                    │
                          ┌─────────────────────────┼──────────────────────────┐
                          ▼                          ▼                          ▼
                    LTP cache (<1s reads)    TickBarAggregator ─▶ 1m bar ─▶ lake.write_candles("1m")
                    (stop-watch / spot)                                        │
                                                                              ▼
                                                   candle_store resample derives 5m/15m/1h (unchanged)

Design notes
------------
* **Broker-agnostic core.** `LiveTickFeed` knows nothing about Angel/Upstox — an adapter parses the broker's
  frames into `LiveTick` and calls `ingest()`. Today the concrete adapter is `AngelWSAdapter` (Angel's
  `SmartWebSocketV2` already decodes the binary frames for us, and Angel self-auths via TOTP so there's no daily
  token dance). An Upstox v3 adapter can drop in later behind the same `ingest()` contract.
* **Bars are bucketed by EXCHANGE time and closed by the CLOCK** (changed 2026-09-17 after S2 day 1 measured the
  cost of the old behaviour): receive-time bucketing misfiled 0.78% of ticks and changed 14.6% of 1-min closes,
  and closing a bar only on the symbol's next tick left the full 107-symbol set 4.7 s (95%) to 7.6 s (all) behind
  the candle close. A closer thread finalizes every bar 1.5 s after its minute ends (`WS_BAR_GRACE_S`); ticks of a
  closed minute are counted as late drops. The (receive − exchange) gap is still recorded as feed latency, and
  official REST 1m bars remain the source of truth for history.
* **Day high/low from QUOTE mode widen the bar exactly.** A snapshot feed misses trades between snapshots, so
  sampled highs/lows are biased inward (measured: tick high BELOW official on 100% of differing bars). The
  exchange-maintained running day high/low cannot miss a print, so a bar that makes a new day extreme gets it exactly.
* **Offline-testable.** `parse_angel_tick` is a pure function; `LiveTickFeed`'s only I/O (the lake write and the
  gap-fill REST fetch) is injected, so the whole feed is unit-testable off-market. The live socket lives only in
  the adapter, whose SDK import is deferred to `start()`.

The seam: swap this feed in behind `avt.engine.MarketDataInterface` (or just point the stop-watch/spot LTP reads
at `feed.get_ltp`) to make live scanning tick-fresh. That wiring is a Monday flip once ticks validate — this
module is the piece that has to exist and be tested first.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Callable, Deque, Dict, List, Optional, Tuple

import pandas as pd

from marketdata.tick_aggregator import TickBarAggregator

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> pd.Timestamp:
    """Current wall-clock in IST, tz-naive — matches the lake/aggregator convention."""
    return pd.Timestamp(datetime.now(_IST).replace(tzinfo=None))


# ══════════════════════════════════════════════════════════════════════════════
#  Normalized tick + pure parsers (offline-testable)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LiveTick:
    symbol: str
    price: float
    recv_ts: pd.Timestamp                 # local receive time, IST-naive — used to build bars
    exch_ts: Optional[pd.Timestamp] = None  # broker exchange timestamp (UTC) — bar bucketing + latency diagnostics
    cum_volume: float = 0.0               # cumulative day volume (Angel convention)
    day_high: Optional[float] = None      # exchange-maintained running day HIGH (QUOTE mode) — exact, no missed print
    day_low: Optional[float] = None       # exchange-maintained running day LOW


def parse_angel_tick(
    msg: dict,
    token_to_symbol: Dict[str, str],
    now: Optional[pd.Timestamp] = None,
) -> Optional[LiveTick]:
    """Parse one Angel `SmartWebSocketV2.on_data` message → LiveTick (or None to drop).

    The SDK hands us an already-decoded dict; the fields we use:
      token                        - instrument token (str/int) → reverse-mapped to our symbol
      last_traded_price            - LTP in PAISE (÷100 to rupees)
      exchange_timestamp           - epoch MILLISECONDS (bar bucketing + latency diagnostics)
      volume_trade_for_the_day     - cumulative day volume (0 for spot indices)
      high/low_price_of_the_day    - PAISE; exchange-exact running day extremes (QUOTE mode) → widen the bar

    Pure: pass `now` (IST-naive) to make it deterministic in tests.
    """
    if not isinstance(msg, dict):
        return None
    token = str(msg.get("token") or "").strip()
    symbol = token_to_symbol.get(token)
    if not symbol:
        return None

    ltp_paise = msg.get("last_traded_price")
    if ltp_paise is None:
        return None
    try:
        price = float(ltp_paise) / 100.0
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None

    exch_ts: Optional[pd.Timestamp] = None
    ts_ms = msg.get("exchange_timestamp")
    if ts_ms:
        try:
            exch_ts = pd.Timestamp(int(ts_ms), unit="ms", tz="UTC")
        except (TypeError, ValueError, OverflowError):
            exch_ts = None

    try:
        cum_vol = float(msg.get("volume_trade_for_the_day") or 0.0)
    except (TypeError, ValueError):
        cum_vol = 0.0

    def _paise(key: str) -> Optional[float]:
        v = msg.get(key)
        try:
            v = float(v) / 100.0
        except (TypeError, ValueError):
            return None
        return v if v > 0 else None

    return LiveTick(
        symbol=symbol,
        price=price,
        recv_ts=now if now is not None else _now_ist(),
        exch_ts=exch_ts,
        cum_volume=cum_vol,
        day_high=_paise("high_price_of_the_day"),
        day_low=_paise("low_price_of_the_day"),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Broker-agnostic feed orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def _default_lake_writer(symbol: str, inst_type: str, row: dict) -> None:
    """Write one finalized 1m bar to the lake (the same schema REST fetches use)."""
    from marketdata import lake
    df = pd.DataFrame([{
        "instrument_key": None,
        "symbol": symbol,
        "underlying_symbol": None,
        "ts": pd.Timestamp(row["timestamp"]),
        "interval": "1m",
        "open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"],
        "volume": row.get("volume", 0), "oi": row.get("oi", 0),
    }])
    lake.write_candles(df, inst_type, symbol, "1m")


class LiveTickFeed:
    """Broker-agnostic live tick sink. An adapter calls `ingest(tick)`; the feed maintains:

      * an LTP cache for sub-second reads (`get_ltp` / `ltp_age_s`) — the stop-watch & spot path;
      * a `TickBarAggregator` that emits finalized 1m bars → `lake_writer` (default: lake.write_candles);
      * feed-latency stats (exchange vs receive time) — the Monday <1s proof;
      * a gap-fill hook: on (re)connect after a WS drop, REST-backfill any 1m bars missed while disconnected.

    All I/O is injectable, so the whole feed is unit-testable off-market.
    """

    def __init__(
        self,
        inst_type_of: Optional[Callable[[str], str]] = None,
        lake_writer: Optional[Callable[[str, str, dict], None]] = None,
        gap_filler: Optional[Callable[[str, pd.Timestamp, pd.Timestamp], int]] = None,
    ):
        self._ltp: Dict[str, Tuple[float, pd.Timestamp]] = {}
        self._last_bar_ts: Dict[str, pd.Timestamp] = {}
        self._lock = threading.Lock()
        self._agg = TickBarAggregator(on_bar=self._on_bar, cumulative_volume=True)
        self._inst_type_of = inst_type_of or (lambda s: "INDEX" if _looks_like_index(s) else "EQ")
        self._lake_writer = lake_writer or _default_lake_writer
        self._gap_filler = gap_filler
        self._latencies: Deque[float] = deque(maxlen=2000)
        self._tick_count = 0
        # bars are closed by the CLOCK (a daemon thread), not by the next tick of each symbol — see
        # TickBarAggregator.close_due. The aggregator is touched from the WS thread and the closer thread → one lock.
        self._last_tick_ts: Optional[pd.Timestamp] = None
        self._agg_lock = threading.Lock()
        self._closer: Optional[threading.Thread] = None
        self._closer_stop = threading.Event()
        self._close_grace_s = float(os.environ.get("WS_BAR_GRACE_S", "1.5"))

    # ── bar closing on the clock ──
    def _ensure_closer(self) -> None:
        if self._closer is not None:
            return
        def _loop():
            while not self._closer_stop.wait(0.25):
                try:
                    with self._agg_lock:
                        self._agg.close_due(_now_ist(), self._close_grace_s)
                except Exception as e:                      # a closer failure must never kill the feed
                    logger.warning("ws_feed: bar closer: %s", e)
        self._closer = threading.Thread(target=_loop, name="ws-bar-closer", daemon=True)
        self._closer.start()

    def stop_closer(self) -> None:
        self._closer_stop.set()

    # ── ingest ──
    def ingest(self, tick: LiveTick) -> None:
        if tick is None or tick.price <= 0:
            return
        self._ensure_closer()
        with self._lock:
            self._ltp[tick.symbol] = (tick.price, tick.recv_ts)
            self._last_tick_ts = tick.recv_ts
            self._tick_count += 1
            if tick.exch_ts is not None:
                # receive − exchange, in seconds (both to UTC): the feed latency
                recv_utc = tick.recv_ts.tz_localize(_IST) if tick.recv_ts.tzinfo is None else tick.recv_ts
                self._latencies.append((recv_utc.tz_convert("UTC") - tick.exch_ts).total_seconds())
        # BUCKET BY EXCHANGE TIME (2026-09-15: receive-time bucketing misfiled 0.78% of ticks and changed 14.6% of
        # 1-min closes). Fall back to receive time when the broker sends no exchange timestamp.
        bts = tick.recv_ts
        if tick.exch_ts is not None:
            e = tick.exch_ts
            bts = e.tz_convert("Asia/Kolkata").tz_localize(None) if e.tzinfo is not None else e
        with self._agg_lock:
            self._agg.on_tick(tick.symbol, tick.price, bts, tick.cum_volume,
                              day_high=tick.day_high, day_low=tick.day_low)

    # ── sub-second reads ──
    def get_ltp(self, symbol: str) -> Optional[float]:
        with self._lock:
            v = self._ltp.get(symbol)
        return v[0] if v else None

    def ltp_age_s(self, symbol: str) -> Optional[float]:
        """Seconds since the last tick for `symbol` (staleness guard)."""
        with self._lock:
            v = self._ltp.get(symbol)
        if not v:
            return None
        return max(0.0, (_now_ist() - v[1]).total_seconds())

    # ── bar sink ──
    def _on_bar(self, symbol: str, row: dict) -> None:
        try:
            self._last_bar_ts[symbol] = pd.Timestamp(row["timestamp"])
            self._lake_writer(symbol, self._inst_type_of(symbol), row)
        except Exception as e:  # a bad bar must never kill the feed thread
            logger.warning("ws_feed: bar write failed for %s: %s", symbol, e)

    def flush(self) -> int:
        """Finalize forming bars (call at 15:30 close). Returns count emitted."""
        with self._agg_lock:
            return len(self._agg.flush())

    def seconds_since_tick(self) -> Optional[float]:
        """Silence detector for the watchdog: seconds since ANY symbol last ticked (None = nothing yet)."""
        with self._lock:
            t = self._last_tick_ts
        return None if t is None else max(0.0, (_now_ist() - t).total_seconds())

    def bar_stats(self) -> dict:
        """Aggregator counters: exchange-exact high/low widenings and ticks dropped as late."""
        return self._agg.stats()

    # ── gap-fill on reconnect ──
    def gap_fill(self, symbols: List[str], since: Optional[pd.Timestamp] = None) -> int:
        """REST-backfill 1m bars missed during a WS drop, per symbol, [last_bar .. now].
        No-op if no gap_filler was injected. Returns total rows backfilled."""
        if self._gap_filler is None:
            return 0
        now = _now_ist()
        total = 0
        for sym in symbols:
            start = since or self._last_bar_ts.get(sym)
            if start is None:
                continue
            if now - start < pd.Timedelta(minutes=1):
                continue  # nothing meaningful missed
            try:
                total += int(self._gap_filler(sym, start, now) or 0)
            except Exception as e:
                logger.warning("ws_feed: gap-fill failed for %s: %s", sym, e)
        if total:
            logger.info("ws_feed: gap-filled %d bars across %d symbols", total, len(symbols))
        return total

    # ── diagnostics ──
    def latency_stats(self) -> Dict[str, float]:
        """Feed-latency summary (seconds). The Monday <1s proof."""
        with self._lock:
            lats = list(self._latencies)
            ticks = self._tick_count
        if not lats:
            return {"ticks": ticks, "samples": 0}
        s = pd.Series(lats)
        return {
            "ticks": ticks,
            "samples": len(lats),
            "p50": float(s.median()),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
            "mean": float(s.mean()),
        }


_TOKEN_CACHE = Path("data/angel_ws_tokens.json")


def _load_token_cache() -> Dict[str, str]:
    """symbol→token map persisted to disk, so subscribing needs NO REST call (searchScrip is rate-limited)."""
    try:
        if _TOKEN_CACHE.exists():
            return json.loads(_TOKEN_CACHE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("ws_feed: token cache unreadable (%s) — will re-resolve", e)
    return {}


def _save_token_cache(cache: Dict[str, str]) -> None:
    try:
        _TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as e:
        logger.warning("ws_feed: could not persist token cache: %s", e)


def _looks_like_index(symbol: str) -> bool:
    return symbol.upper() in {
        "NIFTY", "NIFTY 50", "NIFTY50", "BANKNIFTY", "NIFTY BANK", "FINNIFTY",
        "MIDCPNIFTY", "SENSEX", "BANKEX", "INDIA VIX", "VIX",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Angel adapter (the live socket — SDK import deferred)
# ══════════════════════════════════════════════════════════════════════════════

class AngelWSAdapter:
    """Live Angel feed over `SmartWebSocketV2`. Resolves symbol↔token from an `AngelOneClient`, connects on a
    background thread, and routes every decoded frame → `parse_angel_tick` → `feed.ingest`.

    Reconnects: the SDK retries internally; our `on_close` fires a gap-fill so no minute is lost across a drop.
    Nothing here runs at import time — the SDK is imported inside `start()`, so this module imports fine off-market
    and in CI.
    """

    # Angel SmartWebSocketV2 exchangeType codes
    _EXCH_TYPE = {"NSE": 1, "NFO": 2, "BSE": 3, "BFO": 4, "MCX": 5, "CDS": 13}
    # subscription mode: 1=LTP, 2=Quote, 3=SnapQuote.
    # ★ MUST be QUOTE (2), not LTP (1). VERIFIED against live frames 2026-07-19: mode 1 carries ONLY
    # last_traded_price — NO `volume_trade_for_the_day` — so every tick-built 1m bar would be written with
    # volume=0, which corrupts rvol → the Eye's effort/birth detection → the cascade signal itself. Mode 2
    # adds volume_trade_for_the_day (cumulative day volume, what TickBarAggregator expects) + day OHLC.
    MODE_LTP = 1
    MODE_QUOTE = 2

    def __init__(self, feed: LiveTickFeed, angel_client, symbols: List[str], mode: int = MODE_QUOTE,
                 correlation_id: str = "ptis_ws"):
        self._feed = feed
        self._angel = angel_client
        self._symbols = list(symbols)
        self._mode = mode
        self._wd: Optional[threading.Thread] = None       # watchdog: rebuilds the socket when ticks stop
        self._wd_stop = threading.Event()
        self._restarts = 0
        self._corr = correlation_id
        self._sws = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._token_to_symbol: Dict[str, str] = {}
        self._sub_groups: Dict[int, List[str]] = {}  # exchangeType → [token,...]
        self.unsubscribed: List[str] = []            # symbols that could NOT be subscribed (produce NO data)

    # ── token maps (needs a live Angel session for equities; indices are hardcoded) ──
    def _build_token_maps(self) -> None:
        self._token_to_symbol.clear()
        self._sub_groups.clear()
        cache = _load_token_cache()
        failed: List[str] = []
        resolved_new = False
        for sym in self._symbols:
            exch = self._angel._detect_exchange(sym)
            ck = f"{exch}:{sym.upper()}"
            token = cache.get(ck)
            if not token:
                # `get_symbol_token` hits searchScrip — a REST call under the SAME Angel rate limit. On
                # 2026-07-20 two symbols (RELIANCE, TITAN) failed here at 09:17 while the backfill/AVT were
                # hammering Angel, were silently skipped, and the feed ran all morning subscribed to 6 of 8
                # with nothing to show it. Retry, then persist — so a rate-limited moment can never again
                # shrink the universe, and a restart needs no REST at all.
                for attempt in range(3):
                    try:
                        token = str(self._angel.get_symbol_token(sym, exch))
                        cache[ck] = token
                        resolved_new = True
                        break
                    except Exception as e:
                        if attempt == 2:
                            logger.error("ws_feed: token resolve FAILED for %s after 3 tries: %s", sym, e)
                        else:
                            time.sleep(1.0 + attempt)
            if not token:
                failed.append(sym)
                continue
            et = self._EXCH_TYPE.get(exch, 1)
            self._token_to_symbol[token] = sym
            self._sub_groups.setdefault(et, []).append(token)
        if resolved_new:
            _save_token_cache(cache)
        n, want = len(self._token_to_symbol), len(self._symbols)
        logger.info("ws_feed: resolved %d/%d tokens across %d exchange groups", n, want, len(self._sub_groups))
        if failed:
            # LOUD: a partial subscription means those symbols produce NO signal, silently, all session.
            logger.error("ws_feed: *** %d/%d SYMBOLS NOT SUBSCRIBED — they will produce NO data: %s ***",
                         len(failed), want, ", ".join(failed))
            self.unsubscribed = failed

    # ── lifecycle ──
    def start(self, _watchdog: bool = True) -> None:
        """Authenticate, build token maps, and connect on a background thread (connect() blocks)."""
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # deferred import (SDK)

        self._angel._ensure_session()
        self._build_token_maps()
        if not self._token_to_symbol:
            raise RuntimeError("ws_feed: no tokens resolved — cannot subscribe")

        # SDK defaults allow ONE reconnect (max_retry_attempt=1, 10 s delay) and the counter is never reset, so one
        # hiccup uses it up and the next one closes the feed for the day (S2 2026-09-15: a 10 s stall at 11:40, a 17 s
        # stall at 11:46 -> "Max retry attempt reached", no ticks after 11:46:32). Allow many quick retries.
        self._sws = SmartWebSocketV2(
            auth_token=self._angel._auth_token,
            api_key=self._angel._api_key,
            client_code=self._angel._client_id,
            feed_token=self._angel._feed_token,
            max_retry_attempt=1000, retry_strategy=0, retry_delay=3,
        )
        self._sws.on_open = self._on_open
        self._sws.on_data = self._on_data
        self._sws.on_error = self._on_error
        self._sws.on_close = self._on_close

        self._running = True
        self._thread = threading.Thread(target=self._sws.connect, name="angel-ws", daemon=True)
        self._thread.start()
        logger.info("ws_feed: Angel WS connecting (%d symbols, mode=%d)", len(self._symbols), self._mode)
        if _watchdog:
            self._start_watchdog()

    def _start_watchdog(self, silence_s: float = 20.0) -> None:
        """Rebuild the socket when ticks stop.

        On 2026-09-15 the SDK hit its (then) one-retry limit at 11:46 and closed for good; the collector kept running
        with NO data until the user noticed 70 minutes later. The retry settings are fixed, but a dead socket must
        also be detected locally: during market hours, silence beyond `silence_s` triggers a full rebuild
        (close -> new SmartWebSocketV2 -> resubscribe) and a LOUD log line.
        """
        # 2026-10-05 LEAK FIX: the watchdog rebuilt every 5 s while silent, and each abandoned SDK socket kept
        # reconnecting on its own (SmartWebSocketV2._on_error calls connect() recursively and never checks that we
        # closed it; max_retry_attempt=1000) -> 139k reconnect lines, 776 MB, stale quotes and the S2 socket knocked
        # into "maximum recursion depth exceeded". Now: (1) an abandoned socket is neutralised (retry limit 0) before
        # it is closed; (2) exponential backoff between rebuilds (silence_s, 2x, 4x ... capped at 300 s);
        # (3) at most MAX_REBUILDS per day, then one CRIT line and no more rebuilds.
        MAX_REBUILDS = 8
        state = {"next_ok": 0.0, "gave_up": False}

        def _loop():
            while not self._wd_stop.wait(5.0):
                try:
                    now = _now_ist()
                    if not (now.replace(hour=9, minute=15) <= now <= now.replace(hour=15, minute=30)):
                        continue
                    age = self._feed.seconds_since_tick()
                    if age is None or age < silence_s or time.time() < state["next_ok"]:
                        continue
                    if self._restarts >= MAX_REBUILDS:
                        if not state["gave_up"]:
                            state["gave_up"] = True
                            logger.critical("ws_feed: %d rebuilds and still no ticks - giving up rebuilding today", self._restarts)
                            print(f"  [CRIT] websocket: {self._restarts} rebuilds without ticks - no further rebuilds today", flush=True)
                        continue
                    self._restarts += 1
                    state["next_ok"] = time.time() + min(300.0, silence_s * (2 ** self._restarts))
                    logger.error("ws_feed: NO TICKS for %.0fs — rebuilding the Angel socket (restart #%d)", age, self._restarts)
                    print(f"  [ALARM] no ticks for {age:.0f}s — rebuilding the websocket (restart #{self._restarts})", flush=True)
                    try:
                        if self._sws is not None:
                            self._sws.MAX_RETRY_ATTEMPT = 0          # the abandoned socket must not reconnect by itself
                            self._sws.close_connection()
                    except Exception:
                        pass
                    try:
                        self.start(_watchdog=False)      # fresh socket + resubscribe; token maps are cached
                    except Exception as e:
                        logger.error("ws_feed: socket rebuild FAILED: %s", e)
                except Exception as e:
                    logger.warning("ws_feed: watchdog: %s", e)
        self._wd = threading.Thread(target=_loop, name="angel-ws-watchdog", daemon=True)
        self._wd.start()

    def stop(self) -> None:
        self._running = False
        self._wd_stop.set()
        try:
            self._feed.stop_closer()          # end the clock-close thread with the socket
        except Exception:
            pass
        try:
            if self._sws is not None:
                self._sws.MAX_RETRY_ATTEMPT = 0              # closing must not trigger the SDK's own reconnect loop
                self._sws.close_connection()
        except Exception:
            pass

    # ── SmartWebSocketV2 callbacks ──
    # NOTE: the SDK invokes these with a VARYING arg count (verified 2026-07-19: it called on_close with 4
    # positional args and logged "takes 2 positional arguments but 4 were given"). Accept *args on every
    # callback — a TypeError here kills the close/reconnect path, which is exactly where gap-fill lives.
    def _on_open(self, wsapp, *args) -> None:
        token_list = [{"exchangeType": et, "tokens": toks} for et, toks in self._sub_groups.items()]
        try:
            self._sws.subscribe(self._corr, self._mode, token_list)
            logger.info("ws_feed: subscribed %d tokens", len(self._token_to_symbol))
        except Exception as e:
            logger.error("ws_feed: subscribe failed: %s", e)

    def _on_data(self, wsapp, message, *args) -> None:
        tick = parse_angel_tick(message, self._token_to_symbol)
        if tick is not None:
            self._feed.ingest(tick)

    def _on_error(self, wsapp, error, *args) -> None:
        logger.warning("ws_feed: Angel WS error: %s", error)

    def _on_close(self, wsapp, *args) -> None:
        logger.info("ws_feed: Angel WS closed; gap-filling on reconnect")
        # SDK auto-reconnects; backfill anything missed while we were down.
        try:
            self._feed.gap_fill(self._symbols)
        except Exception as e:
            logger.warning("ws_feed: gap-fill on close failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  Wiring + Monday preflight
# ══════════════════════════════════════════════════════════════════════════════

def _rest_gap_filler(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Default gap-fill: pull today's 1m bars via the existing REST path (candle_store, which itself
    writes to the lake), so a WS drop is healed by the same feed the app already trusts. Returns rows
    covering (start, end]."""
    from marketdata.candle_store import get_candle_store
    day = pd.Timestamp(end).normalize()
    df = get_candle_store().get(symbol, "1m", day - pd.Timedelta(days=1), day + pd.Timedelta(days=1), live=True)
    if df is None or len(df) == 0:
        return 0
    mask = (df["ts"] > pd.Timestamp(start)) & (df["ts"] <= pd.Timestamp(end))
    return int(mask.sum())


def build_angel_live_feed(symbols: List[str], angel_client=None,
                          with_gap_fill: bool = True) -> Tuple[LiveTickFeed, AngelWSAdapter]:
    """Build a wired Angel live feed for `symbols`. Does NOT connect — call `adapter.start()` (Monday).
    `angel_client` defaults to the app's MarketDataService Angel client (built if the app isn't up)."""
    if angel_client is None:
        from ingestion.market_data_service import get_market_data_service, init_market_data_service
        svc = get_market_data_service() or init_market_data_service()
        angel_client = getattr(svc, "_angel", None) if svc else None
        if angel_client is None:
            raise RuntimeError("build_angel_live_feed: no Angel client configured (need ANGEL_* creds)")
    feed = LiveTickFeed(gap_filler=_rest_gap_filler if with_gap_fill else None)
    adapter = AngelWSAdapter(feed, angel_client, symbols)
    return feed, adapter


def _preflight(symbols: List[str], seconds: int = 60) -> None:
    """Monday validation entry point: connect, print live LTPs + feed-latency, then report.
    Run DURING MARKET HOURS:  PTIS_ENV=test python -m marketdata.ws_feed NIFTY BANKNIFTY RELIANCE
    """
    import time
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    feed, adapter = build_angel_live_feed(symbols)
    adapter.start()
    print(f"[ws_feed] connecting for {len(symbols)} symbols; sampling {seconds}s...")
    try:
        for i in range(seconds):
            time.sleep(1)
            if i % 5 == 4:
                snap = {s: feed.get_ltp(s) for s in symbols}
                ages = {s: feed.ltp_age_s(s) for s in symbols}
                print(f"[{i+1:>3}s] LTP={snap}")
                print(f"       age_s={ {k: (round(v,2) if v is not None else None) for k,v in ages.items()} }")
    finally:
        adapter.stop()
        n = feed.flush()
        print(f"[ws_feed] flushed {n} forming bars")
        print(f"[ws_feed] latency stats (recv−exch, s): {feed.latency_stats()}")


if __name__ == "__main__":
    import sys
    syms = sys.argv[1:] or ["NIFTY", "BANKNIFTY", "RELIANCE"]
    _preflight(syms)
