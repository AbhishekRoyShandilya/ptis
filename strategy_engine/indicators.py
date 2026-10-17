"""Technical indicator utilities used across strategy_engine.

Key fix (quant review):
  - ATR and ADX now use Wilder's smoothing (alpha = 1/period) instead of EWM
    span which gives alpha = 2/(period+1) — a significantly different decay rate
    that understates persistence of volatility.  Wilder's is the industry
    standard for ATR, ADX, RSI, and all indicators Wilder originally defined.
  - SuperTrend: first bar is now properly initialized so the loop does not
    compare against a NaN on i=1, which was silently corrupting direction on
    the first candle.
  - Added RSI — needed for mean-reversion regime detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's exponential smoothing: alpha = 1/period.

    This is the correct smoothing for ATR, ADX, and any Wilder-derived
    indicator.  pandas EWM with span=period uses alpha = 2/(period+1)
    which decays roughly twice as fast — wrong for these indicators.
    """
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


class IndicatorEngine:
    """Core technical indicator calculations — all using standard quant formulas."""

    # ── Smoothing primitives ───────────────────────────────────────────────

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Standard EMA (alpha = 2/(period+1)).  Correct for EMA-based signals."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    # ── Volatility ─────────────────────────────────────────────────────────

    @staticmethod
    def true_range(df: pd.DataFrame) -> pd.Series:
        """True Range = max(H-L, |H-Cprev|, |L-Cprev|)."""
        high = df["high"]
        low = df["low"]
        close_prev = df["close"].shift(1)
        return pd.concat(
            [high - low, (high - close_prev).abs(), (low - close_prev).abs()],
            axis=1,
        ).max(axis=1)

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range using Wilder's smoothing (industry standard).

        FIXED: was using ewm(span=period) which gives alpha=2/(period+1)≈0.133
               Wilder's uses alpha=1/period≈0.071 — a materially different
               decay that keeps ATR more stable and avoids over-reacting to
               single extreme candles.
        """
        tr = IndicatorEngine.true_range(df)
        return _wilder_smooth(tr, period)

    @staticmethod
    def atr_percentile(df: pd.DataFrame, period: int = 14, lookback: int = 252) -> float:
        """Current ATR expressed as a percentile of its own history.

        Uses a proper lookback (default 252 bars ≈ 1 trading year) so the
        percentile is statistically meaningful.  The previous code used only
        atr_lookback (14 bars) — far too short for percentile estimation.

        Returns a float 0-100.
        """
        atr_series = IndicatorEngine.atr(df, period)
        hist_window = min(lookback, len(atr_series))
        hist = atr_series.iloc[-hist_window:]
        current = atr_series.iloc[-1]
        return float((hist < current).mean() * 100)

    # ── Trend Strength ─────────────────────────────────────────────────────

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index using Wilder's smoothing throughout.

        FIXED:
          1. TR, +DM, -DM all smoothed with Wilder's (alpha=1/period).
          2. +DI / -DI computed from Wilder-smoothed components, not EWM span.
          3. DX then smoothed with Wilder's to produce ADX — matching the
             original Wilder (1978) definition and all major charting platforms.

        The previous version used ewm(span=period) for everything, producing
        ADX values that respond too fast and overstate trend strength in choppy
        conditions.
        """
        high = df["high"]
        low = df["low"]

        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # Wilder-smooth TR, +DM, -DM independently
        tr_smooth = _wilder_smooth(IndicatorEngine.true_range(df), period)
        plus_dm_smooth = _wilder_smooth(plus_dm, period)
        minus_dm_smooth = _wilder_smooth(minus_dm, period)

        # Directional Indicators
        plus_di = 100.0 * (plus_dm_smooth / tr_smooth.replace(0, np.nan))
        minus_di = 100.0 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))

        # DX and ADX
        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx_val = _wilder_smooth(dx.fillna(0), period)

        return adx_val

    @staticmethod
    def plus_minus_di(df: pd.DataFrame, period: int = 14):
        """Return (+DI, -DI) as a tuple of Series — used for ADX direction."""
        high = df["high"]
        low = df["low"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr_smooth = _wilder_smooth(IndicatorEngine.true_range(df), period)
        plus_di = 100.0 * (_wilder_smooth(plus_dm, period) / tr_smooth.replace(0, np.nan))
        minus_di = 100.0 * (_wilder_smooth(minus_dm, period) / tr_smooth.replace(0, np.nan))
        return plus_di, minus_di

    # ── Momentum / Mean-Reversion ──────────────────────────────────────────

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """RSI using Wilder's smoothing — needed for mean-reversion detection."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = _wilder_smooth(gain, period)
        avg_loss = _wilder_smooth(loss, period)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    # ── Trend Following ─────────────────────────────────────────────────────

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """SuperTrend indicator — returns direction +1 (bullish) / -1 (bearish).

        FIXED: First bar (index 0) is now explicitly initialized before the
               loop.  Previously supertrend.iloc[0] was NaN, so the comparison
               `prev_st == final_upper.iloc[i-1]` on i=1 always evaluated to
               False (NaN != anything), silently forcing the wrong branch and
               corrupting the first N directions.
        """
        # NOTE: the band-locking recurrence is inherently sequential, but it runs on NUMPY arrays (not
        # per-element pandas .iloc, which made this the system's #1 hotspot — ~1M pandas index ops per
        # call, and it's recomputed on every bar-slice by read_balance/read_move). Bit-identical to the
        # prior pandas loop: same branches, same NaN-comparison semantics (NaN < x and x > NaN are both
        # False under numpy too), so early-bar NaN bands propagate exactly as before.
        atr_val = IndicatorEngine.atr(df, period).to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        hl2 = (high + low) / 2.0

        upper_band = hl2 + (multiplier * atr_val)
        lower_band = hl2 - (multiplier * atr_val)
        n = len(df)

        final_upper = upper_band.copy()
        final_lower = lower_band.copy()
        supertrend = np.full(n, np.nan, dtype=float)
        direction = np.zeros(n, dtype=int)

        if n:
            # ── Initialize first bar: direction from close vs midpoint band ──
            if close[0] > hl2[0]:
                direction[0] = 1
                supertrend[0] = final_lower[0]
            else:
                direction[0] = -1
                supertrend[0] = final_upper[0]

            # ── Main loop (band-locking requires sequential processing) ──
            for i in range(1, n):
                # Lock upper band: only move lower; break if price closed above
                if not (upper_band[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]):
                    final_upper[i] = final_upper[i - 1]
                # Lock lower band: only move higher; break if price closed below
                if not (lower_band[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]):
                    final_lower[i] = final_lower[i - 1]

                if direction[i - 1] == -1:  # previously bearish
                    direction[i] = 1 if close[i] > final_upper[i] else -1
                else:                       # previously bullish
                    direction[i] = -1 if close[i] < final_lower[i] else 1

                supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

        result = df.copy()
        result["st_value"] = supertrend
        result["st_dir"] = direction
        result["st_upper"] = final_upper
        result["st_lower"] = final_lower
        return result

    # ── Candle Utilities ───────────────────────────────────────────────────

    @staticmethod
    def candle_body(df: pd.DataFrame) -> pd.Series:
        return (df["close"] - df["open"]).abs()

    @staticmethod
    def avg_body(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
        return IndicatorEngine.candle_body(df).rolling(lookback).mean()

    @staticmethod
    def avg_volume(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
        return df["volume"].rolling(lookback).mean()

    # ── Bollinger Bands ────────────────────────────────────────────────────

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
        """Return (upper, mid, lower, width_pct) — width_pct is normalized by mid."""
        mid = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        width_pct = (upper - lower) / mid.replace(0, np.nan) * 100
        return upper, mid, lower, width_pct
