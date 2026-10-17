# ske/feature_computer.py
"""
Feature Computer — transforms raw OHLCV candle data into a strategy-specific
feature vector (dict[str, float]) for fuzzy scoring and variant classification.

Uses existing indicator functions from strategy_engine/indicators.py and
candle pattern detectors from strategy_engine/candle_patterns.py.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeatureComputer:
    """
    Computes a feature vector from OHLCV candle data.

    Each candle is a dict with keys: open, high, low, close, volume, timestamp.
    The feature vector is a flat dict[str, float] — ready for fuzzy scoring.
    """

    # All built-in features this computer can produce
    AVAILABLE_FEATURES = [
        "compression_ratio",
        "hl_slope_5",
        "hl_slope_20",
        "distance_to_nearest_level",
        "rvol",
        "vol_slope",
        "atr_percentile",
        "rsi",
        "adx",
        "bb_width",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "volume_surge",
        "momentum_5",
        "momentum_20",
        "range_compression",
        "trend_alignment",
        "vwap_distance",
        "relative_volume",
    ]

    def __init__(self, feature_list: Optional[List[str]] = None):
        """
        Args:
            feature_list: Specific features to compute. If None, compute all.
        """
        self._feature_list = feature_list or self.AVAILABLE_FEATURES

    def compute_features(
        self,
        candles: List[Dict[str, Any]],
        symbol: str = "",
        indicators_cache: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute the feature vector for the most recent bar given candle history.

        Args:
            candles: List of OHLCV dicts (oldest first), minimum ~30 bars.
            symbol: Optional symbol identifier for logging.
            indicators_cache: Pre-computed indicator values to skip recomputation.

        Returns:
            Dict mapping feature_name → float value.
        """
        if len(candles) < 5:
            logger.warning("Too few candles (%d) for feature computation", len(candles))
            return {}

        cache = indicators_cache or {}
        features: Dict[str, float] = {}

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        opens = [c["open"] for c in candles]
        volumes = [c.get("volume", 0) for c in candles]

        last = candles[-1]
        last_close = closes[-1]

        for feat in self._feature_list:
            try:
                if feat == "compression_ratio":
                    features[feat] = self._compression_ratio(highs, lows)
                elif feat == "hl_slope_5":
                    features[feat] = self._hl_slope(closes, 5)
                elif feat == "hl_slope_20":
                    features[feat] = self._hl_slope(closes, 20)
                elif feat == "distance_to_nearest_level":
                    features[feat] = self._distance_to_level(closes, highs, lows)
                elif feat == "rvol":
                    features[feat] = self._rvol(volumes)
                elif feat == "vol_slope":
                    features[feat] = self._vol_slope(volumes)
                elif feat == "atr_percentile":
                    features[feat] = cache.get("atr_percentile") or self._atr_percentile(highs, lows, closes)
                elif feat == "rsi":
                    features[feat] = cache.get("rsi") or self._rsi(closes)
                elif feat == "adx":
                    features[feat] = cache.get("adx") or self._adx(highs, lows, closes)
                elif feat == "bb_width":
                    features[feat] = self._bb_width(closes)
                elif feat == "body_ratio":
                    features[feat] = self._body_ratio(last)
                elif feat == "upper_wick_ratio":
                    features[feat] = self._upper_wick_ratio(last)
                elif feat == "lower_wick_ratio":
                    features[feat] = self._lower_wick_ratio(last)
                elif feat == "volume_surge":
                    features[feat] = self._volume_surge(volumes)
                elif feat == "momentum_5":
                    features[feat] = self._momentum(closes, 5)
                elif feat == "momentum_20":
                    features[feat] = self._momentum(closes, 20)
                elif feat == "range_compression":
                    features[feat] = self._range_compression(highs, lows)
                elif feat == "trend_alignment":
                    features[feat] = self._trend_alignment(closes)
                elif feat == "vwap_distance":
                    features[feat] = self._vwap_distance(candles)
                elif feat == "relative_volume":
                    features[feat] = self._relative_volume(volumes)
            except Exception as e:
                logger.debug("Feature %s computation failed for %s: %s", feat, symbol, e)
                features[feat] = 0.0

        return features

    # ──────────────────────────────────────────
    #  Individual feature computations
    # ──────────────────────────────────────────

    @staticmethod
    def _compression_ratio(highs: List[float], lows: List[float], lookback: int = 20) -> float:
        """Ratio of recent range to lookback range. Low = compressed."""
        if len(highs) < lookback:
            lookback = len(highs)
        recent = 5
        if len(highs) < recent:
            return 1.0

        full_range = max(highs[-lookback:]) - min(lows[-lookback:])
        recent_range = max(highs[-recent:]) - min(lows[-recent:])

        if full_range <= 0:
            return 1.0
        return recent_range / full_range

    @staticmethod
    def _hl_slope(closes: List[float], period: int) -> float:
        """Linear slope of closes over period, normalized by price."""
        if len(closes) < period:
            return 0.0
        segment = closes[-period:]
        avg = sum(segment) / len(segment)
        if avg <= 0:
            return 0.0
        # Simple slope: (last - first) / (period * avg_price)
        return (segment[-1] - segment[0]) / (period * avg)

    @staticmethod
    def _distance_to_level(
        closes: List[float], highs: List[float], lows: List[float], lookback: int = 50,
    ) -> float:
        """Distance from current close to nearest swing high/low, as fraction of ATR."""
        if len(closes) < 10:
            return 0.5

        lb = min(lookback, len(closes))
        swing_highs = []
        swing_lows = []
        for i in range(-lb + 2, -2):
            if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
                swing_highs.append(highs[i])
            if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
                swing_lows.append(lows[i])

        levels = swing_highs + swing_lows
        if not levels:
            return 0.5

        current = closes[-1]
        min_dist = min(abs(current - level) for level in levels)

        # Normalize by average range
        avg_range = sum(h - l for h, l in zip(highs[-14:], lows[-14:])) / min(14, len(highs))
        if avg_range <= 0:
            return 0.5
        return min_dist / avg_range

    @staticmethod
    def _rvol(volumes: List[float], lookback: int = 20) -> float:
        """Relative volume: current volume / average volume."""
        if len(volumes) < 2:
            return 1.0
        lb = min(lookback, len(volumes) - 1)
        avg_vol = sum(volumes[-lb - 1:-1]) / lb if lb > 0 else 1.0
        if avg_vol <= 0:
            return 1.0
        return volumes[-1] / avg_vol

    @staticmethod
    def _vol_slope(volumes: List[float], period: int = 10) -> float:
        """Slope of volume over period, normalized by average volume."""
        if len(volumes) < period:
            return 0.0
        segment = volumes[-period:]
        avg = sum(segment) / len(segment)
        if avg <= 0:
            return 0.0
        return (segment[-1] - segment[0]) / (period * avg)

    @staticmethod
    def _atr_percentile(
        highs: List[float], lows: List[float], closes: List[float],
        period: int = 14, lookback: int = 252,
    ) -> float:
        """Current ATR as percentile of historical ATR values."""
        n = len(closes)
        if n < period + 1:
            return 0.5

        # Compute ATR series using Wilder smoothing
        trs = []
        for i in range(1, n):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)

        if len(trs) < period:
            return 0.5

        # Wilder ATR
        alpha = 1.0 / period
        atr = sum(trs[:period]) / period
        atr_values = [atr]
        for tr in trs[period:]:
            atr = atr * (1 - alpha) + tr * alpha
            atr_values.append(atr)

        if not atr_values:
            return 0.5

        current_atr = atr_values[-1]
        lb = min(lookback, len(atr_values))
        historical = sorted(atr_values[-lb:])
        rank = sum(1 for v in historical if v <= current_atr)
        return rank / len(historical)

    @staticmethod
    def _rsi(closes: List[float], period: int = 14) -> float:
        """RSI (Wilder-smoothed), returned as 0–100."""
        if len(closes) < period + 1:
            return 50.0

        gains = []
        losses = []
        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))

        if len(gains) < period:
            return 50.0

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        alpha = 1.0 / period
        for i in range(period, len(gains)):
            avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
            avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _adx(
        highs: List[float], lows: List[float], closes: List[float], period: int = 14,
    ) -> float:
        """ADX value (0–100)."""
        n = len(closes)
        if n < period * 2:
            return 25.0  # neutral default

        # +DM, -DM, TR
        plus_dm = []
        minus_dm = []
        trs = []
        for i in range(1, n):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
            minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))

        if len(trs) < period:
            return 25.0

        alpha = 1.0 / period

        # Wilder smoothing
        sm_plus = sum(plus_dm[:period]) / period
        sm_minus = sum(minus_dm[:period]) / period
        sm_tr = sum(trs[:period]) / period

        dx_values = []
        for i in range(period, len(trs)):
            sm_plus = sm_plus * (1 - alpha) + plus_dm[i] * alpha
            sm_minus = sm_minus * (1 - alpha) + minus_dm[i] * alpha
            sm_tr = sm_tr * (1 - alpha) + trs[i] * alpha

            if sm_tr > 0:
                plus_di = 100.0 * sm_plus / sm_tr
                minus_di = 100.0 * sm_minus / sm_tr
                di_sum = plus_di + minus_di
                if di_sum > 0:
                    dx_values.append(100.0 * abs(plus_di - minus_di) / di_sum)

        if len(dx_values) < period:
            return 25.0

        adx = sum(dx_values[:period]) / period
        for dx in dx_values[period:]:
            adx = adx * (1 - alpha) + dx * alpha

        return adx

    @staticmethod
    def _bb_width(closes: List[float], period: int = 20) -> float:
        """Bollinger Band width as fraction of middle band."""
        if len(closes) < period:
            return 0.0
        segment = closes[-period:]
        sma = sum(segment) / period
        if sma <= 0:
            return 0.0
        variance = sum((c - sma) ** 2 for c in segment) / period
        std = math.sqrt(variance)
        return (4.0 * std) / sma  # width = 2*std*2 / sma

    @staticmethod
    def _body_ratio(candle: Dict[str, Any]) -> float:
        """Body size as fraction of total range."""
        total = candle["high"] - candle["low"]
        if total <= 0:
            return 0.0
        body = abs(candle["close"] - candle["open"])
        return body / total

    @staticmethod
    def _upper_wick_ratio(candle: Dict[str, Any]) -> float:
        """Upper wick as fraction of total range."""
        total = candle["high"] - candle["low"]
        if total <= 0:
            return 0.0
        upper_body = max(candle["open"], candle["close"])
        return (candle["high"] - upper_body) / total

    @staticmethod
    def _lower_wick_ratio(candle: Dict[str, Any]) -> float:
        """Lower wick as fraction of total range."""
        total = candle["high"] - candle["low"]
        if total <= 0:
            return 0.0
        lower_body = min(candle["open"], candle["close"])
        return (lower_body - candle["low"]) / total

    @staticmethod
    def _volume_surge(volumes: List[float], lookback: int = 20) -> float:
        """Current volume / max volume in lookback."""
        if len(volumes) < 2:
            return 0.0
        lb = min(lookback, len(volumes) - 1)
        max_vol = max(volumes[-lb - 1:-1]) if lb > 0 else 1.0
        if max_vol <= 0:
            return 0.0
        return volumes[-1] / max_vol

    @staticmethod
    def _momentum(closes: List[float], period: int) -> float:
        """Rate of change over period: (close - close_n) / close_n."""
        if len(closes) <= period:
            return 0.0
        prev = closes[-period - 1]
        if prev <= 0:
            return 0.0
        return (closes[-1] - prev) / prev

    @staticmethod
    def _range_compression(highs: List[float], lows: List[float], period: int = 10) -> float:
        """Average bar range over last `period` bars / max bar range in lookback."""
        if len(highs) < period:
            return 1.0
        recent_ranges = [h - l for h, l in zip(highs[-period:], lows[-period:])]
        avg_range = sum(recent_ranges) / period

        lb = min(50, len(highs))
        all_ranges = [h - l for h, l in zip(highs[-lb:], lows[-lb:])]
        max_range = max(all_ranges) if all_ranges else 1.0
        if max_range <= 0:
            return 1.0
        return avg_range / max_range

    @staticmethod
    def _trend_alignment(closes: List[float]) -> float:
        """
        Alignment score: +1 if 5 SMA > 20 SMA > 50 SMA (bullish alignment),
        -1 for reverse, 0 for mixed. Returned as 0–1 (0.5 = neutral).
        """
        if len(closes) < 50:
            return 0.5
        sma5 = sum(closes[-5:]) / 5
        sma20 = sum(closes[-20:]) / 20
        sma50 = sum(closes[-50:]) / 50

        if sma5 > sma20 > sma50:
            return 1.0
        elif sma5 < sma20 < sma50:
            return 0.0
        else:
            return 0.5

    @staticmethod
    def _vwap_distance(candles: List[Dict[str, Any]], period: int = 50) -> float:
        """Distance of current close from VWAP, as fraction of price."""
        lb = min(period, len(candles))
        if lb < 1:
            return 0.0

        recent = candles[-lb:]
        cum_vol = 0.0
        cum_pv = 0.0
        for c in recent:
            vol = c.get("volume", 0)
            typical = (c["high"] + c["low"] + c["close"]) / 3.0
            cum_vol += vol
            cum_pv += typical * vol

        if cum_vol <= 0:
            return 0.0
        vwap = cum_pv / cum_vol
        current = candles[-1]["close"]
        if vwap <= 0:
            return 0.0
        return (current - vwap) / vwap

    @staticmethod
    def _relative_volume(volumes: List[float], period: int = 20) -> float:
        """Current bar volume relative to average. Same as rvol but explicit."""
        if len(volumes) < 2:
            return 1.0
        lb = min(period, len(volumes) - 1)
        avg = sum(volumes[-lb - 1:-1]) / lb if lb > 0 else 1.0
        if avg <= 0:
            return 1.0
        return volumes[-1] / avg
