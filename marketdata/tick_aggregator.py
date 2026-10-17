"""
marketdata.tick_aggregator — build 1-minute OHLCV bars from a live TICK stream (websocket), so the lake stays
real-time-fresh without per-symbol REST polling. This is the CORE of the websocket feed track; it is pure and
deterministic (no I/O), so it is fully unit-testable off-market. The websocket ADAPTER (Angel/Upstox) feeds ticks
in via `on_tick`; completed 1m bars come out via the `on_bar` callback and are written to the lake exactly like a
REST fetch — the existing candle_store resample path then derives 5m/15m/1h, so tick-built and REST-fetched bars
flow through the SAME downstream pipeline (one code path, no cascade change).

FIDELITY: a tick-aggregated 1m bar should reconcile with the broker's official 1m bar once the minute closes; small
differences (a missed tick, volume convention) are possible, so the design keeps official REST bars as the source
of truth for HISTORY and uses tick bars for the live tail — with a reconciliation hook to overwrite the tick bar
with the official one when it arrives. OHLC is exact from ticks (first/max/min/last price); volume is best-effort.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import pandas as pd


@dataclass
class _Bar:
    minute: pd.Timestamp          # bar start (minute floor), tz-naive IST — matches the lake convention
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    _last_cum_vol: Optional[float] = None   # last CUMULATIVE day-volume seen (broker ticks carry cumulative)

    def as_row(self, symbol: str) -> dict:
        return {"timestamp": self.minute, "symbol": symbol, "open": self.open, "high": self.high,
                "low": self.low, "close": self.close, "volume": max(0.0, self.volume), "oi": 0}


class TickBarAggregator:
    """Per-symbol 1-minute OHLCV builder. Feed ticks with `on_tick`; when a tick crosses into a new minute the
    previous minute's bar is FINALIZED and handed to `on_bar(symbol, row_dict)`. Minute boundaries are the tick
    timestamp floored to the minute (tz-naive IST), so bars align exactly with the lake's 1m bars.

    Volume: broker ticks usually carry CUMULATIVE day volume → per-bar volume = (cum at bar end − cum at bar start).
    If ticks carry incremental volume instead, pass cumulative=False and volumes are summed. OHLC is always exact.
    """

    def __init__(self, on_bar: Optional[Callable[[str, dict], None]] = None, cumulative_volume: bool = True):
        self._on_bar = on_bar
        self._cumulative = cumulative_volume
        self._cur: Dict[str, _Bar] = {}          # symbol -> forming bar
        # Last CUMULATIVE day-volume seen per symbol, carried ACROSS bars. A bar's volume must be
        # (cum at this bar's end) − (cum at the PREVIOUS bar's end); baselining off the first tick INSIDE
        # the bar silently drops everything traded between the bars (and reports 0 for a single-tick bar).
        self._last_cum: Dict[str, float] = {}
        self._last_dh: Dict[str, float] = {}     # running day HIGH per symbol (exchange-exact, QUOTE mode)
        self._last_dl: Dict[str, float] = {}     # running day LOW
        self._exact_hi = self._exact_lo = self._late = 0

    def on_tick(self, symbol: str, price: float, ts, volume: float = 0.0,
                day_high: Optional[float] = None, day_low: Optional[float] = None) -> Optional[dict]:
        """Ingest one tick. Returns the FINALIZED prior-minute bar dict if this tick rolled the minute, else None.

        `day_high` / `day_low` (Angel QUOTE mode: high/low_price_of_the_day) are EXACT exchange-maintained extremes.
        A snapshot feed misses trades between snapshots, so a bar's high/low built from sampled prices is biased
        INWARD (measured 2026-09-15: tick high below official on 100% of differing bars, median −18 bps). Whenever the
        running day high RISES (or the day low FALLS) the extreme must have printed since the last tick, so the bar it
        lands in gets that exact value. Extremes that stay inside the day's range are still only sampled.
        """
        if price is None or price <= 0:
            return None
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("Asia/Kolkata").tz_localize(None)
        minute = t.floor("1min")
        cur = self._cur.get(symbol)
        finalized = None

        if cur is not None and minute > cur.minute:
            finalized = self._finalize(symbol)               # a new minute began → close the old bar
            cur = None
        elif cur is not None and minute < cur.minute:
            self._late += 1
            return None                                      # stale/out-of-order tick → ignore (never rewrite a closed bar)

        dh = float(day_high) if day_high else None
        dl = float(day_low) if day_low else None

        if cur is None:
            baseline = None
            if self._cumulative:
                prev = self._last_cum.get(symbol)
                # carry the previous bar's final cum; a DROP means the day rolled over (broker resets
                # cumulative volume each session) → re-baseline on this tick instead of going negative.
                baseline = prev if (prev is not None and volume >= prev) else volume
            nb = _Bar(minute=minute, open=price, high=price, low=price, close=price,
                      volume=0.0, _last_cum_vol=baseline)
            if self._cumulative:
                nb.volume = max(0.0, float(volume) - float(baseline))
                self._last_cum[symbol] = float(volume)
            elif volume:
                nb.volume = float(volume)
            self._apply_day_extremes(symbol, nb, dh, dl)
            self._cur[symbol] = nb
            return finalized

        # update the forming bar
        cur.high = max(cur.high, price)
        cur.low = min(cur.low, price)
        cur.close = price
        self._apply_day_extremes(symbol, cur, dh, dl)
        if self._cumulative:
            if cur._last_cum_vol is None:
                cur._last_cum_vol = volume
            cur.volume = max(0.0, float(volume) - float(cur._last_cum_vol))   # delta since the PREVIOUS bar's end
            self._last_cum[symbol] = float(volume)
        else:
            cur.volume += float(volume)
        return finalized

    def _apply_day_extremes(self, symbol: str, bar: "_Bar", day_high: Optional[float], day_low: Optional[float]) -> None:
        """Widen the forming bar with exchange-exact extremes when the day high/low just moved."""
        if day_high is not None and day_high > 0:
            prev = self._last_dh.get(symbol)
            if prev is None or day_high > prev:
                bar.high = max(bar.high, day_high)
                self._exact_hi += 1
            self._last_dh[symbol] = day_high if prev is None else max(prev, day_high)
        if day_low is not None and day_low > 0:
            prev = self._last_dl.get(symbol)
            if prev is None or day_low < prev:
                bar.low = min(bar.low, day_low)
                self._exact_lo += 1
            self._last_dl[symbol] = day_low if prev is None else min(prev, day_low)

    def close_due(self, now, grace_s: float = 1.5) -> list:
        """Finalize every bar whose minute ended at least `grace_s` ago — the CLOCK close.

        Without this a bar is emitted only when that symbol's NEXT tick arrives, so the complete set of symbols lags
        the candle close (measured 2026-09-15: 95% of 107 symbols at 4.7 s, all at 7.6 s, worst 48 s). The grace
        covers ticks of the closing minute that arrive late (p95 1.2 s).
        """
        t = pd.Timestamp(now)
        if t.tzinfo is not None:
            t = t.tz_convert("Asia/Kolkata").tz_localize(None)
        cutoff = t - pd.Timedelta(seconds=60 + float(grace_s))
        out = []
        for sym in [s for s, b in list(self._cur.items()) if b.minute <= cutoff]:
            row = self._finalize(sym)
            if row is not None:
                out.append(row)
        return out

    def stats(self) -> dict:
        return {"exact_high_updates": self._exact_hi, "exact_low_updates": self._exact_lo, "late_ticks_dropped": self._late}

    def flush(self, symbol: Optional[str] = None) -> list:
        """Finalize the forming bar(s) — call at session close (15:30) so the last minute is emitted."""
        out = []
        syms = [symbol] if symbol else list(self._cur.keys())
        for s in syms:
            row = self._finalize(s)
            if row is not None:
                out.append(row)
        return out

    def _finalize(self, symbol: str) -> Optional[dict]:
        bar = self._cur.pop(symbol, None)
        if bar is None:
            return None
        row = bar.as_row(symbol)
        if self._on_bar is not None:
            try:
                self._on_bar(symbol, row)
            except Exception:
                pass
        return row
