# discovery/event_extractor.py
"""
Stage 2 — Market Event Extractor

Given a PatternQuery (symbol, timeframe, time), this stage:
  1. Fetches OHLCV candle data for the specified symbol/timeframe
  2. Identifies the exact breakout bar (or the bar nearest to event_time)
  3. Computes a 12-dimensional PatternFingerprint that captures the *shape*
     of the event in a normalised, comparable vector.

The fingerprint is market-cap-agnostic, price-agnostic, and
timeframe-normalised so it can match across different stocks and dates.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from discovery.models import PatternFingerprint, PatternQuery
from strategy_engine.indicators import IndicatorEngine

logger = logging.getLogger(__name__)

# Type for a candle-fetching function the caller injects
# Signature: fetch(symbol, timeframe, start, end) -> pd.DataFrame
#   with columns: datetime, open, high, low, close, volume
CandleFetcher = Callable[[str, str, datetime, datetime], pd.DataFrame]


class EventExtractor:
    """
    Extracts a PatternFingerprint from raw market data.

    Usage:
        extractor = EventExtractor(fetcher=my_candle_fetcher)
        fingerprint = await extractor.extract(query)
    """

    # Minimum bars needed to compute all fingerprint dimensions
    MIN_BARS = 50

    def __init__(self, fetcher: Optional[CandleFetcher] = None):
        self._fetcher = fetcher

    def set_fetcher(self, fetcher: CandleFetcher) -> None:
        self._fetcher = fetcher

    async def extract(self, query: PatternQuery) -> PatternFingerprint:
        """Fetch candles and compute the fingerprint for the queried event."""
        if not self._fetcher:
            raise RuntimeError("No candle fetcher configured — call set_fetcher()")

        # Determine date range: we need enough bars before the event for indicators
        event_date = datetime.strptime(query.event_date, "%Y-%m-%d") if query.event_date else datetime.utcnow()
        # Fetch extra history for indicator warm-up
        start = event_date - timedelta(days=5)
        end = event_date + timedelta(days=1)

        df = self._fetcher(query.symbol, query.timeframe, start, end)
        if df is None or len(df) < self.MIN_BARS:
            raise ValueError(
                f"Insufficient data for {query.symbol}/{query.timeframe}: "
                f"got {len(df) if df is not None else 0} bars, need {self.MIN_BARS}+"
            )

        # Find the breakout bar
        bar_idx = self._find_event_bar(df, query)
        logger.info("Event bar for %s: index %d / %d", query.symbol, bar_idx, len(df))

        # Compute fingerprint
        fp = self.compute_fingerprint(df, bar_idx, query.symbol, query.timeframe)
        return fp

    async def extract_from_dataframe(
        self, df: pd.DataFrame, bar_index: int,
        symbol: str = "", timeframe: str = ""
    ) -> PatternFingerprint:
        """Compute fingerprint directly from a DataFrame (for batch scanning)."""
        return self.compute_fingerprint(df, bar_index, symbol, timeframe)

    # ── Core fingerprint computation ────────────────────────────────

    def compute_fingerprint(
        self, df: pd.DataFrame, bar_idx: int,
        symbol: str = "", timeframe: str = ""
    ) -> PatternFingerprint:
        """
        Build a 12-dimensional fingerprint vector from OHLCV data
        at the specified bar index.
        """
        bar = df.iloc[bar_idx]
        o, h, l, c, v = bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]
        candle_range = h - l if h != l else 1e-9  # avoid div/0

        # Pre-compute indicators over the full DataFrame
        atr_series = IndicatorEngine.atr(df, period=14)
        rsi_series = IndicatorEngine.rsi(df["close"], period=14)
        ema20 = IndicatorEngine.ema(df["close"], period=20)
        vol_sma20 = IndicatorEngine.sma(df["volume"], period=20)

        current_atr = atr_series.iloc[bar_idx] if bar_idx < len(atr_series) else candle_range
        atr_sma20 = atr_series.rolling(20).mean()

        # Compute VWAP (cumulative for the session — approximate with rolling)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical_price * df["volume"]).rolling(window=min(bar_idx + 1, 78)).sum()
        cum_vol = df["volume"].rolling(window=min(bar_idx + 1, 78)).sum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

        # ── Dimension 0: Body ratio ──
        body = abs(c - o)
        body_ratio = np.clip(body / candle_range, 0, 1)

        # ── Dimension 1: Upper wick ratio ──
        upper_wick = h - max(o, c)
        upper_wick_ratio = np.clip(upper_wick / candle_range, 0, 1)

        # ── Dimension 2: Lower wick ratio ──
        lower_wick = min(o, c) - l
        lower_wick_ratio = np.clip(lower_wick / candle_range, 0, 1)

        # ── Dimension 3: Volume surge ──
        avg_vol = vol_sma20.iloc[bar_idx] if bar_idx < len(vol_sma20) else v
        volume_surge = np.clip(v / max(avg_vol, 1), 0, 5) / 5  # normalise to 0-1

        # ── Dimension 4: ATR expansion ──
        avg_atr = atr_sma20.iloc[bar_idx] if bar_idx < len(atr_sma20) else current_atr
        atr_expansion = np.clip(current_atr / max(avg_atr, 1e-9), 0, 3) / 3

        # ── Dimension 5: Momentum score (ATR-normalised move) ──
        prev_close = df["close"].iloc[bar_idx - 1] if bar_idx > 0 else o
        move = (c - prev_close)
        momentum_score = np.clip(abs(move) / max(current_atr, 1e-9), 0, 4) / 4

        # ── Dimension 6: Consolidation bars ──
        consol_bars = self._count_consolidation_bars(df, bar_idx, current_atr)
        consolidation_norm = np.clip(consol_bars / 20, 0, 1)

        # ── Dimension 7: Range compression ──
        lookback_10 = df["high"].iloc[max(0, bar_idx-10):bar_idx].max() - \
                       df["low"].iloc[max(0, bar_idx-10):bar_idx].min()
        lookback_30 = df["high"].iloc[max(0, bar_idx-30):bar_idx].max() - \
                       df["low"].iloc[max(0, bar_idx-30):bar_idx].min()
        range_compression = np.clip(
            lookback_10 / max(lookback_30, 1e-9), 0, 1
        )

        # ── Dimension 8: Trend alignment ──
        ema_val = ema20.iloc[bar_idx] if bar_idx < len(ema20) else c
        ema_prev = ema20.iloc[bar_idx - 5] if bar_idx >= 5 else ema_val
        ema_slope = (ema_val - ema_prev) / max(current_atr, 1e-9)
        direction = 1 if c > o else -1
        # +1 if breakout direction aligns with EMA trend, -1 if counter, 0 if flat
        if abs(ema_slope) < 0.1:
            trend_alignment = 0.5  # flat
        elif (ema_slope > 0 and direction > 0) or (ema_slope < 0 and direction < 0):
            trend_alignment = 1.0  # aligned
        else:
            trend_alignment = 0.0  # counter-trend

        # ── Dimension 9: RSI at trigger ──
        rsi_val = rsi_series.iloc[bar_idx] if bar_idx < len(rsi_series) else 50
        rsi_normalised = np.clip(rsi_val / 100, 0, 1)

        # ── Dimension 10: VWAP distance ──
        vwap_val = vwap.iloc[bar_idx] if bar_idx < len(vwap) else c
        vwap_dist = (c - vwap_val) / max(current_atr, 1e-9)
        vwap_normalised = np.clip((vwap_dist + 3) / 6, 0, 1)  # map [-3, +3] ATRs to [0, 1]

        # ── Dimension 11: Relative volume ──
        rel_vol = np.clip(v / max(avg_vol, 1), 0, 5) / 5

        # ── Assemble vector ──
        vector = np.array([
            float(body_ratio),
            float(upper_wick_ratio),
            float(lower_wick_ratio),
            float(volume_surge),
            float(atr_expansion),
            float(momentum_score),
            float(consolidation_norm),
            float(range_compression),
            float(trend_alignment),
            float(rsi_normalised),
            float(vwap_normalised),
            float(rel_vol),
        ])

        fp = PatternFingerprint(
            vector=vector,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=pd.Timestamp(bar["datetime"]) if "datetime" in df.columns else None,
            bar_index=bar_idx,
            direction=direction,
            raw_ohlcv={
                "open": float(o), "high": float(h), "low": float(l),
                "close": float(c), "volume": float(v),
            },
        )
        return fp

    # ── Helpers ───────────────────────────────────────────────────────

    def _find_event_bar(self, df: pd.DataFrame, query: PatternQuery) -> int:
        """Find the bar in df that best matches the user's described event."""
        # If event_time is specified, find the closest bar
        if query.event_time and "datetime" in df.columns:
            target_time = query.event_time  # "HH:MM"
            hh, mm = map(int, target_time.split(":"))

            # Filter to event_date
            if query.event_date:
                event_dt = datetime.strptime(query.event_date, "%Y-%m-%d")
                mask = pd.to_datetime(df["datetime"]).dt.date == event_dt.date()
                day_df = df[mask]
                if len(day_df) > 0:
                    # Find bar with closest time
                    times = pd.to_datetime(day_df["datetime"])
                    target = times.iloc[0].replace(hour=hh, minute=mm, second=0)
                    diffs = (times - target).abs()
                    return int(day_df.index[diffs.argmin()])

        # Fallback: find the bar with the highest momentum (biggest ATR-normalised move)
        atr = IndicatorEngine.atr(df, period=14)
        moves = df["close"].diff().abs() / atr.replace(0, np.nan)
        # Only consider last trading day
        if query.event_date and "datetime" in df.columns:
            event_dt = datetime.strptime(query.event_date, "%Y-%m-%d")
            mask = pd.to_datetime(df["datetime"]).dt.date == event_dt.date()
            day_indices = df[mask].index
            if len(day_indices) > 0:
                day_moves = moves.loc[day_indices]
                return int(day_moves.idxmax())

        # Absolute fallback: last big move
        return int(moves.iloc[-50:].idxmax()) if len(moves) > 50 else len(df) - 1

    def _count_consolidation_bars(
        self, df: pd.DataFrame, bar_idx: int, current_atr: float
    ) -> int:
        """
        Count how many consecutive bars before bar_idx were in a tight range
        (i.e. the consolidation before the breakout).
        """
        threshold = current_atr * 0.5  # bars with range < 0.5 ATR = tight
        count = 0
        for i in range(bar_idx - 1, max(bar_idx - 30, 0), -1):
            bar_range = df["high"].iloc[i] - df["low"].iloc[i]
            if bar_range < threshold:
                count += 1
            else:
                break
        return count
