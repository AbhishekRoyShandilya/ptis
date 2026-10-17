"""Market regime detector — fixed and hardened.

Quant review findings addressed:
  1. ATR in detect_volatility() used ewm(span=period) — wrong decay rate.
     Now uses Wilder's smoothing (alpha=1/period) for all ATR computations,
     consistent with indicators.py.
  2. ATR percentile lookback was min(atr_lookback, len) = min(14, n) bars —
     far too short for a meaningful percentile.  Now uses 252 bars (1 year).
  3. BB squeeze compared current_bb_width (scalar, normalised %) against
     raw bb_std.rolling(50).mean() (unnormalized std) — units mismatch bug.
     Fixed to maintain a bb_width series and compare in normalised units.
  4. RegimeDetector used MarketRegime.RANGING, HIGH_VOLATILITY, LOW_VOLATILITY,
     BREAKOUT, MEAN_REVERTING which did not exist in core/enums.py.
     Those values are now added to the enum so both systems are consistent.
  5. Added RSI for mean-reversion regime detection (was completely absent).
  6. Added volume-weighted trend confirmation to reduce false trending signals
     in low-volume consolidation periods.
  7. Regime scoring weights are documented so they can be tuned/backtested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.enums import MarketRegime, VolatilityRegime
from strategy_engine.indicators import IndicatorEngine, _wilder_smooth


logger = logging.getLogger(__name__)

# ── Volatility percentile lookback ───────────────────────────────────────────
# Use 252 bars (≈ 1 trading year of daily data).  For intraday bars scale
# proportionally: 252 * (1440 / bar_size_minutes) but cap at available length.
_PERCENTILE_LOOKBACK = 252


@dataclass
class RegimeState:
    """Multi-timeframe regime snapshot — used by strategy filters."""
    daily_trend_up: bool
    weekly_trend_up: bool
    h1_supertrend_up: bool
    h1_above_50dma: bool
    m15_supertrend_up: bool
    m15_above_50dma: bool
    regime: MarketRegime
    volatility: VolatilityRegime
    atr_value: float
    atr_percentile: float


@dataclass
class RegimeAnalysis:
    """Output of a single-timeframe regime detection call."""
    regime: MarketRegime
    confidence: float           # 0.0–1.0
    indicators: Dict
    description: str


class MarketRegimeDetector:
    """
    Deterministic multi-indicator market regime classifier.

    Classification order (highest-confidence wins):
      1. Trending Up / Down  (ADX + slope + MA alignment + volume)
      2. Breakout            (BB squeeze → expansion)
      3. Mean Reverting      (ADX low + RSI extreme + bounded range)
      4. Ranging             (ADX low + flat slope + tight range)
      5. High / Low Vol      (ATR percentile — fallback when trend is absent)
    """

    def __init__(
        self,
        trend_lookback: int = 20,
        volatility_lookback: int = 14,
        atr_lookback: int = 14,
        adx_lookback: int = 14,
        percentile_lookback: int = _PERCENTILE_LOOKBACK,
    ):
        self.trend_lookback = trend_lookback
        self.volatility_lookback = volatility_lookback
        self.atr_lookback = atr_lookback
        self.adx_lookback = adx_lookback
        self.percentile_lookback = percentile_lookback

    # ── Primary entry point ───────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> RegimeAnalysis:
        """
        Detect current market regime from OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume].
                Must have at least 50 rows.

        Returns:
            RegimeAnalysis with regime, confidence, and raw indicators.
        """
        if len(df) < 50:
            raise ValueError("Need at least 50 bars for regime detection")

        indicators = self._compute_indicators(df)
        regime, confidence = self._classify(indicators)

        descriptions = {
            MarketRegime.TRENDING_UP:    "Strong uptrend — momentum with directional volume",
            MarketRegime.TRENDING_DOWN:  "Strong downtrend — directional momentum to the downside",
            MarketRegime.RANGING:        "Sideways consolidation — no clear directional bias",
            MarketRegime.RANGE:          "Sideways consolidation — no clear directional bias",
            MarketRegime.HIGH_VOLATILITY: "Elevated volatility — wide ranges, caution on sizing",
            MarketRegime.LOW_VOLATILITY:  "Compressed volatility — narrow ranges, watch for breakout",
            MarketRegime.BREAKOUT:        "Volatility expansion from compression — potential trend start",
            MarketRegime.MEAN_REVERTING:  "Oscillating around mean — suited for fade/counter-trend",
            MarketRegime.UNKNOWN:         "Insufficient data for confident classification",
        }

        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            description=descriptions.get(regime, "Unknown regime"),
        )

    # ── Indicator computation ─────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> Dict:
        """Compute all regime detection indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df.get("volume", pd.Series(dtype=float))

        # ── Trend ─────────────────────────────────────────────────────────
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        trend_slope = (
            (sma_20.iloc[-1] - sma_20.iloc[-self.trend_lookback])
            / self.trend_lookback
            if len(sma_20) > self.trend_lookback
            else 0.0
        )
        price_vs_sma20 = (
            (close.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] * 100
            if pd.notna(sma_20.iloc[-1]) and sma_20.iloc[-1] != 0
            else 0.0
        )
        sma_20_above_50 = (
            sma_20.iloc[-1] > sma_50.iloc[-1]
            if pd.notna(sma_20.iloc[-1]) and pd.notna(sma_50.iloc[-1])
            else None
        )

        # ── Volume confirmation ────────────────────────────────────────────
        # Trending moves should have above-average volume; ranges have flat vol.
        # Breakouts need volume surge for confirmation.
        if len(volume) >= 20 and volume.sum() > 0:
            avg_vol_20 = float(volume.rolling(20).mean().iloc[-1])
            recent_vol = float(volume.tail(3).mean())
            vol_ratio = recent_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
        else:
            vol_ratio = 1.0  # neutral if volume data unavailable

        # ── Volatility ─────────────────────────────────────────────────────
        # ATR using Wilder's smoothing (FIXED from simple rolling)
        atr_series = IndicatorEngine.atr(df, self.atr_lookback)
        current_atr = float(atr_series.iloc[-1])
        avg_atr = float(atr_series.mean()) if len(atr_series) > 0 else current_atr
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

        # Bollinger Band Width series (FIXED: maintain as series for proper comparison)
        bb_upper, bb_mid, bb_lower, bb_width_series = IndicatorEngine.bollinger_bands(close, 20, 2.0)
        bb_width_current = float(bb_width_series.iloc[-1]) if pd.notna(bb_width_series.iloc[-1]) else 0.0
        bb_width_avg = float(bb_width_series.rolling(50).mean().iloc[-1]) if len(bb_width_series) > 50 else bb_width_current

        # BB squeeze: current width < 80% of 50-bar average (FIXED: same units)
        bb_squeeze = bb_width_current < 0.80 * bb_width_avg if bb_width_avg > 0 else False
        # BB expansion: width surging above average (breakout signal)
        bb_expanding = bb_width_current > 1.20 * bb_width_avg and not bb_squeeze

        # Historical volatility (annualised)
        returns = close.pct_change().dropna()
        hist_vol = float(
            returns.rolling(self.volatility_lookback).std().iloc[-1] * np.sqrt(252) * 100
        ) if len(returns) >= self.volatility_lookback else 0.0

        # ── ADX (Wilder's — FIXED) ─────────────────────────────────────────
        adx_series = IndicatorEngine.adx(df, self.adx_lookback)
        adx = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0

        # ── RSI — for mean-reversion detection (NEW) ──────────────────────
        rsi_series = IndicatorEngine.rsi(close, 14)
        rsi = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50.0
        rsi_extreme = rsi < 30 or rsi > 70   # oversold/overbought

        # ── Range measurement ──────────────────────────────────────────────
        recent_high = float(high.tail(self.trend_lookback).max())
        recent_low = float(low.tail(self.trend_lookback).min())
        range_pct = (
            (recent_high - recent_low) / recent_low * 100
            if recent_low > 0 else 0.0
        )

        return {
            "trend_slope": trend_slope,
            "price_vs_sma20": price_vs_sma20,
            "sma_20_above_50": sma_20_above_50,
            "atr": current_atr,
            "atr_ratio": atr_ratio,
            "bb_width": bb_width_current,
            "bb_width_avg": bb_width_avg,
            "bb_squeeze": bb_squeeze,
            "bb_expanding": bb_expanding,
            "hist_vol": hist_vol,
            "adx": adx,
            "rsi": rsi,
            "rsi_extreme": rsi_extreme,
            "range_pct": range_pct,
            "vol_ratio": vol_ratio,
        }

    # ── Regime classification ─────────────────────────────────────────────

    def _classify(self, ind: Dict) -> Tuple[MarketRegime, float]:
        """
        Rule-based regime scoring.  Each regime accumulates a score 0–1.
        The regime with the highest score wins.

        Scoring philosophy:
          - ADX > 25 = trending; ADX < 20 = not trending.
          - Volume confirmation required for trending (vol_ratio > 1.1).
          - BB squeeze → breakout candidate.
          - RSI extreme + low ADX → mean reverting.
          - Low range + low ADX + flat slope → ranging.
          - ATR ratio as a secondary tiebreaker for vol regimes.
        """
        adx = ind["adx"]
        slope = ind["trend_slope"]
        atr_ratio = ind["atr_ratio"]
        price_vs_sma = ind["price_vs_sma20"]
        sma_aligned = ind["sma_20_above_50"]
        bb_squeeze = ind["bb_squeeze"]
        bb_expanding = ind["bb_expanding"]
        rsi_extreme = ind["rsi_extreme"]
        vol_ratio = ind["vol_ratio"]

        scores: dict[MarketRegime, float] = {}

        # ── Trending Up ───────────────────────────────────────────────────
        up = 0.0
        if adx > 25:            up += 0.30
        if slope > 0:            up += 0.25
        if price_vs_sma > 0:    up += 0.20
        if sma_aligned is True: up += 0.15
        if vol_ratio > 1.1:     up += 0.10   # volume confirms the move
        scores[MarketRegime.TRENDING_UP] = up

        # ── Trending Down ──────────────────────────────────────────────────
        dn = 0.0
        if adx > 25:               dn += 0.30
        if slope < 0:              dn += 0.25
        if price_vs_sma < 0:       dn += 0.20
        if sma_aligned is False:   dn += 0.15
        if vol_ratio > 1.1:        dn += 0.10
        scores[MarketRegime.TRENDING_DOWN] = dn

        # ── Breakout ───────────────────────────────────────────────────────
        bo = 0.0
        if bb_squeeze:          bo += 0.40   # compression before breakout
        if bb_expanding:        bo += 0.30   # volatility already expanding
        if vol_ratio > 1.3:     bo += 0.20   # surge in volume on the break
        if 18 < adx < 35:       bo += 0.10   # ADX starting to rise
        scores[MarketRegime.BREAKOUT] = bo

        # ── Mean Reverting ─────────────────────────────────────────────────
        mr = 0.0
        if adx < 25:                        mr += 0.30
        if rsi_extreme:                     mr += 0.35   # RSI at extremes is key signal
        if abs(price_vs_sma) > 2:           mr += 0.20   # price stretched from mean
        if 3 < ind["range_pct"] < 8:        mr += 0.15
        scores[MarketRegime.MEAN_REVERTING] = mr

        # ── Ranging ───────────────────────────────────────────────────────
        rg = 0.0
        if adx < 20:                        rg += 0.40
        if abs(slope) < 0.01:               rg += 0.30
        if ind["range_pct"] < 5:            rg += 0.20
        if vol_ratio < 0.90:                rg += 0.10   # low volume in ranges
        scores[MarketRegime.RANGING] = rg

        # ── High Volatility ────────────────────────────────────────────────
        hv = 0.0
        if atr_ratio > 1.5:                 hv += 0.40
        if ind["bb_width"] > ind["bb_width_avg"] * 1.5: hv += 0.30
        if ind["hist_vol"] > 30:            hv += 0.30
        scores[MarketRegime.HIGH_VOLATILITY] = hv

        # ── Low Volatility ─────────────────────────────────────────────────
        lv = 0.0
        if atr_ratio < 0.70:                lv += 0.40
        if bb_squeeze:                      lv += 0.30
        if ind["hist_vol"] < 15:            lv += 0.30
        scores[MarketRegime.LOW_VOLATILITY] = lv

        best = max(scores, key=scores.get)
        confidence = min(scores[best], 1.0)

        # Low confidence → unknown
        if confidence < 0.30:
            return MarketRegime.UNKNOWN, confidence

        return best, confidence

    # ── Volatility classification ─────────────────────────────────────────

    def detect_volatility(
        self, df: pd.DataFrame, period: int = 14
    ) -> Tuple[VolatilityRegime, float, float]:
        """
        Classify volatility by ATR percentile over a proper lookback window.

        FIXED:
          - ATR now uses Wilder's smoothing (alpha=1/period) not ewm(span).
          - Lookback now uses self.percentile_lookback (default 252 bars)
            instead of atr_lookback (14 bars) — 14 bars is statistically
            meaningless for percentile estimation.
          - Percentile logic: (hist < current).mean() * 100 gives the fraction
            of historical values BELOW current → high percentile = high vol.
            This logic was CORRECT in the original; preserved here.

        Returns:
            (VolatilityRegime, current_atr_value, percentile_0_to_100)
        """
        atr_series = IndicatorEngine.atr(df, period)
        current_atr = float(atr_series.iloc[-1])

        # Use proper lookback for percentile (FIXED)
        lookback = min(self.percentile_lookback, len(atr_series))
        hist = atr_series.iloc[-lookback:]
        percentile = float((hist < current_atr).mean() * 100)

        if percentile < 25:
            regime = VolatilityRegime.LOW
        elif percentile < 60:
            regime = VolatilityRegime.AVERAGE
        elif percentile < 85:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.EXTREME

        return regime, current_atr, percentile

    # ── Multi-timeframe regime state ──────────────────────────────────────

    def get_regime_state(
        self,
        df_daily: pd.DataFrame,
        df_weekly: pd.DataFrame,
        df_h1: pd.DataFrame,
        df_m15: pd.DataFrame,
    ) -> RegimeState:
        """
        Full multi-timeframe regime assessment.

        Uses:
          - Daily/weekly: SMA slope for macro trend bias.
          - H1: regime + volatility (primary classification).
          - H1 + M15: SuperTrend + 50 DMA for entry-timeframe bias.
        """
        def slope_up(df: pd.DataFrame, window: int = 50) -> bool:
            if len(df) < window + 1:
                return False
            sma = df["close"].rolling(window).mean()
            return bool(sma.iloc[-1] > sma.iloc[-2])

        daily_trend_up = slope_up(df_daily, window=50)
        weekly_trend_up = slope_up(df_weekly, window=50)

        def supertrend_flag(df: pd.DataFrame, period: int = 10) -> bool:
            if len(df) < period + 2:
                return False
            result = IndicatorEngine.supertrend(df, period=period, multiplier=3.0)
            return bool(result["st_dir"].iloc[-1] == 1)

        h1_supertrend_up = supertrend_flag(df_h1)
        m15_supertrend_up = supertrend_flag(df_m15)

        h1_above_50dma = False
        m15_above_50dma = False
        if len(df_h1) >= 50:
            sma = df_h1["close"].rolling(50).mean()
            h1_above_50dma = bool(df_h1["close"].iloc[-1] > sma.iloc[-1])
        if len(df_m15) >= 50:
            sma = df_m15["close"].rolling(50).mean()
            m15_above_50dma = bool(df_m15["close"].iloc[-1] > sma.iloc[-1])

        # Regime and volatility from H1 (primary trading timeframe)
        regime_analysis = self.detect(df_h1)
        regime = regime_analysis.regime
        volatility, atr_val, atr_pct = self.detect_volatility(df_h1)

        return RegimeState(
            daily_trend_up=daily_trend_up,
            weekly_trend_up=weekly_trend_up,
            h1_supertrend_up=h1_supertrend_up,
            h1_above_50dma=h1_above_50dma,
            m15_supertrend_up=m15_supertrend_up,
            m15_above_50dma=m15_above_50dma,
            regime=regime,
            volatility=volatility,
            atr_value=atr_val,
            atr_percentile=atr_pct,
        )


# Alias for backward compatibility — __init__.py imports this name
RegimeDetector = MarketRegimeDetector
