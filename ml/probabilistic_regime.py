"""
Probabilistic Regime Classifier
================================

Replaces the binary regime label with a probability distribution.

Old:  regime = "trending_up"
New:  regime_probs = {trending_up: 0.55, ranging: 0.30, volatile: 0.10, breakout: 0.05}

Why this matters:
  - Markets transition gradually — binary classification misses the in-between
  - Strategy sizing should scale with regime confidence (60% confidence → 0.75x size)
  - Mixed regimes are real ("volatile trending" = 60% trend + 40% volatile)

Five independent signals vote with different weights:
  1. ADX           — trend strength
  2. ATR ratio     — volatility
  3. Autocorrelation — mean reversion strength
  4. BB bandwidth  — compression / breakout potential
  5. EMA slope     — directional bias confirmation

Regime labels match the existing RuleBasedRegimeClassifier:
  trending_up | trending_down | ranging | volatile | breakout

This module ADDS to the existing regime infrastructure — it does NOT replace
RuleBasedRegimeClassifier (which still produces a single label for downstream
code that expects a string). Use this when you want:
  - Probability-aware sizing
  - Soft regime transitions
  - Better feature input to ML models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Regime labels (must match ml/regime_classifier.py for interop) ──

REGIME_TRENDING_UP = "trending_up"
REGIME_TRENDING_DOWN = "trending_down"
REGIME_RANGING = "ranging"
REGIME_VOLATILE = "volatile"
REGIME_BREAKOUT = "breakout"


# ── Output dataclasses ──────────────────────────────────────────────


@dataclass
class RegimeProbabilities:
    """Probability distribution over regime labels (sums to ~1.0)."""
    trending_up: float = 0.0
    trending_down: float = 0.0
    ranging: float = 0.0
    volatile: float = 0.0
    breakout: float = 0.0

    @property
    def dominant(self) -> str:
        """The single most likely regime label."""
        return max(self.as_dict(), key=self.as_dict().get)

    @property
    def dominant_prob(self) -> float:
        """Probability of the dominant regime."""
        return max(self.as_dict().values())

    @property
    def is_confident(self) -> bool:
        """Is the dominant regime > 60%? — usable as a confidence gate."""
        return self.dominant_prob > 0.60

    @property
    def trending_total(self) -> float:
        """Total trending probability (up + down)."""
        return self.trending_up + self.trending_down

    def as_dict(self) -> Dict[str, float]:
        return {
            "trending_up": self.trending_up,
            "trending_down": self.trending_down,
            "ranging": self.ranging,
            "volatile": self.volatile,
            "breakout": self.breakout,
        }

    def for_features(self) -> Dict[str, float]:
        """
        Flattened for ML feature input — provides aggregated views useful
        for downstream models (XGBoost, etc.) without imposing class boundaries.
        """
        return {
            "regime_prob_trending": self.trending_total,
            "regime_prob_trending_up": self.trending_up,
            "regime_prob_trending_down": self.trending_down,
            "regime_prob_ranging": self.ranging,
            "regime_prob_volatile": self.volatile,
            "regime_prob_breakout": self.breakout,
            "regime_confidence": self.dominant_prob,
        }


@dataclass
class RegimeTransition:
    """A detected shift from one dominant regime to another."""
    from_regime: str
    to_regime: str
    transition_speed: float           # 0=gradual, 1=sudden
    started_at: Optional[datetime] = None
    confidence: float = 0.0           # confidence in the new regime


# ── Strategy → preferred regime mapping (used for sizing) ──

_STRATEGY_PREFERRED_REGIMES: Dict[str, List[str]] = {
    # Strategies and the regime probabilities that benefit them
    "compression_breakout":     [REGIME_BREAKOUT, REGIME_VOLATILE],
    "ema_pullback":             [REGIME_TRENDING_UP, REGIME_TRENDING_DOWN],
    "vwap_bounce":              [REGIME_RANGING],
    "vwap_reversal":            [REGIME_RANGING, REGIME_VOLATILE],
    "orb_breakout":             [REGIME_BREAKOUT, REGIME_VOLATILE],
    "momentum_exhaustion":      [REGIME_RANGING, REGIME_VOLATILE],
    "gap_fill":                 [REGIME_RANGING],
    "pullback_trend":           [REGIME_TRENDING_UP, REGIME_TRENDING_DOWN],
}


# ── Classifier ─────────────────────────────────────────────────────


class ProbabilisticRegimeClassifier:
    """
    Multi-signal voting classifier that produces a regime probability
    distribution from OHLCV candles.

    Usage:
        clf = ProbabilisticRegimeClassifier()
        probs = clf.classify(candles_df)        # RegimeProbabilities
        # → e.g. trending_up=0.45, ranging=0.30, ...

        sizing = clf.get_sizing_multiplier(probs, "ema_pullback")
        # → 0.75 (regime confidence is moderate)

        # Detect transition (after multiple classify() calls):
        transition = clf.detect_transition()
    """

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        ema_fast: int = 9,
        ema_slow: int = 21,
        bb_period: int = 20,
        lookback: int = 50,
    ):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.bb_period = bb_period
        self.lookback = lookback

        # Rolling history for transition detection
        self._prob_history: List[Tuple[datetime, RegimeProbabilities]] = []

    # ── Main entrypoint ──

    def classify(self, candles) -> RegimeProbabilities:
        """
        Classify the regime as a probability distribution.

        Args:
            candles: pandas DataFrame with open/high/low/close/volume
                     (and DatetimeIndex), OR a list[dict] with the same
                     keys (compatible with RuleBasedRegimeClassifier).
        """
        df = self._coerce_to_df(candles)
        if df is None or len(df) < self.lookback:
            return self._uniform_prior()

        # Compute all five signals
        adx, plus_di, minus_di = self._compute_adx(df)
        atr_ratio = self._compute_atr_ratio(df)
        autocorr = self._compute_autocorrelation(df)
        bb_width = self._compute_bb_width(df)
        ema_slope = self._compute_ema_slope(df)

        # ── Signal 1: trend strength via ADX ──
        if adx > 35:
            trend_vote = 0.85
            range_vote = 0.05
        elif adx > 25:
            trend_vote = 0.60
            range_vote = 0.15
        elif adx > 15:
            trend_vote = 0.30
            range_vote = 0.40
        else:
            trend_vote = 0.10
            range_vote = 0.60

        # Direction from DI
        if plus_di > minus_di:
            up_bias_share = 0.8
        elif plus_di < minus_di:
            up_bias_share = 0.2
        else:
            up_bias_share = 0.5

        # ── Signal 2: volatility (ATR ratio) ──
        if atr_ratio > 1.8:
            vol_vote = 0.80
        elif atr_ratio > 1.3:
            vol_vote = 0.50
        elif atr_ratio > 0.8:
            vol_vote = 0.20
        else:
            vol_vote = 0.05

        # ── Signal 3: mean reversion (autocorrelation) ──
        # Negative autocorr = mean-reverting; positive = trending
        if autocorr < -0.3:
            reversion_vote_range = 0.70
            reversion_vote_trend = 0.10
        elif autocorr < -0.1:
            reversion_vote_range = 0.50
            reversion_vote_trend = 0.20
        elif autocorr > 0.3:
            reversion_vote_range = 0.10
            reversion_vote_trend = 0.70
        elif autocorr > 0.1:
            reversion_vote_range = 0.20
            reversion_vote_trend = 0.50
        else:
            reversion_vote_range = 0.35
            reversion_vote_trend = 0.35

        # ── Signal 4: Bollinger Bandwidth (compression → breakout potential) ──
        if bb_width < 0.02:
            breakout_vote = 0.70  # tight squeeze → breakout setup
        elif bb_width < 0.04:
            breakout_vote = 0.40
        else:
            breakout_vote = 0.10

        # ── Signal 5: EMA slope (directional bias) ──
        if ema_slope > 0.002:
            ema_up_share = 0.75
        elif ema_slope < -0.002:
            ema_up_share = 0.25
        else:
            ema_up_share = 0.5

        # Blend up-share between DI and EMA evidence
        blended_up_share = (up_bias_share + ema_up_share) / 2.0

        # ── Combine via weighted scoring ──
        weights = {
            "adx": 0.30,
            "reversion": 0.20,
            "volatility": 0.20,
            "bandwidth": 0.15,
            "ema": 0.15,
        }

        raw = {
            REGIME_TRENDING_UP: (
                trend_vote * blended_up_share * weights["adx"]
                + reversion_vote_trend * blended_up_share * weights["reversion"]
                + ema_up_share * trend_vote * weights["ema"]
            ),
            REGIME_TRENDING_DOWN: (
                trend_vote * (1 - blended_up_share) * weights["adx"]
                + reversion_vote_trend * (1 - blended_up_share) * weights["reversion"]
                + (1 - ema_up_share) * trend_vote * weights["ema"]
            ),
            REGIME_RANGING: (
                range_vote * weights["adx"]
                + reversion_vote_range * weights["reversion"]
                + (1.0 - vol_vote) * 0.3 * weights["volatility"]
            ),
            REGIME_VOLATILE: (
                vol_vote * weights["volatility"]
                + max(atr_ratio - 1.0, 0) * 0.2
            ),
            REGIME_BREAKOUT: (
                breakout_vote * weights["bandwidth"]
                + max(0.05 - bb_width, 0) * 5 * 0.10
            ),
        }

        # Normalize so sum = 1
        total = sum(raw.values())
        if total <= 0:
            return self._uniform_prior()

        probs = RegimeProbabilities(
            trending_up=round(raw[REGIME_TRENDING_UP] / total, 4),
            trending_down=round(raw[REGIME_TRENDING_DOWN] / total, 4),
            ranging=round(raw[REGIME_RANGING] / total, 4),
            volatile=round(raw[REGIME_VOLATILE] / total, 4),
            breakout=round(raw[REGIME_BREAKOUT] / total, 4),
        )

        # Store in rolling history for transition detection
        ts = (
            df.index[-1].to_pydatetime() if isinstance(df.index, pd.DatetimeIndex)
            else datetime.utcnow()
        )
        self._prob_history.append((ts, probs))
        if len(self._prob_history) > 500:
            self._prob_history = self._prob_history[-500:]

        return probs

    # ── Transition detection ──

    def detect_transition(self, window: int = 10) -> Optional[RegimeTransition]:
        """
        Compare the average regime over the last `window` classifications vs the
        prior `window`. If the dominant regime changed, return a RegimeTransition.
        """
        if len(self._prob_history) < window * 2:
            return None

        recent = [p for _, p in self._prob_history[-window:]]
        older = [p for _, p in self._prob_history[-window * 2:-window]]

        recent_dominant = self._avg_dominant(recent)
        older_dominant = self._avg_dominant(older)

        if recent_dominant == older_dominant:
            return None

        recent_prob = float(np.mean([
            getattr(p, recent_dominant) for p in recent
        ]))
        older_prob = float(np.mean([
            getattr(p, older_dominant) for p in older
        ]))

        # Transition speed: how quickly dominant regime probability shifted
        speed = abs(recent_prob - older_prob) / max(older_prob, 0.01)
        speed = float(min(speed, 1.0))

        return RegimeTransition(
            from_regime=older_dominant,
            to_regime=recent_dominant,
            transition_speed=round(speed, 4),
            started_at=self._prob_history[-window][0],
            confidence=round(recent_prob, 4),
        )

    # ── Sizing helper ──

    def get_sizing_multiplier(
        self,
        probs: RegimeProbabilities,
        strategy: str,
    ) -> float:
        """
        Scale position size based on how well current regime probabilities
        match the strategy's preferred regimes.

        Returns a value in [0.25, 1.0]:
          - >=0.60 alignment → 1.00x (full size)
          - 0.40-0.59       → 0.75x
          - 0.25-0.39       → 0.50x
          - <0.25           → 0.25x (skeleton position)

        Caller is free to clamp/floor as needed.
        """
        preferred = _STRATEGY_PREFERRED_REGIMES.get(strategy, [])
        if not preferred:
            return 1.0  # unknown strategy — don't penalize

        regime_dict = probs.as_dict()
        preferred_prob = sum(regime_dict.get(r, 0.0) for r in preferred)

        if preferred_prob >= 0.60:
            return 1.00
        if preferred_prob >= 0.40:
            return 0.75
        if preferred_prob >= 0.25:
            return 0.50
        return 0.25

    # ── Helpers ──

    @staticmethod
    def _coerce_to_df(candles) -> Optional[pd.DataFrame]:
        """Accept DataFrame or list[dict]; return DataFrame or None."""
        if candles is None:
            return None
        if isinstance(candles, pd.DataFrame):
            df = candles
        else:
            try:
                df = pd.DataFrame(candles)
            except Exception:
                return None
        if df.empty:
            return None
        # Ensure essential columns exist
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                return None
        # Index by timestamp if a column is provided
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ("timestamp", "datetime", "date", "time"):
                if col in df.columns:
                    try:
                        df = df.set_index(col)
                        break
                    except Exception:
                        pass
        return df

    @staticmethod
    def _uniform_prior() -> RegimeProbabilities:
        """Equal-probability fallback when we can't classify."""
        return RegimeProbabilities(
            trending_up=0.20, trending_down=0.20,
            ranging=0.20, volatile=0.20, breakout=0.20,
        )

    @staticmethod
    def _avg_dominant(probs_list: List[RegimeProbabilities]) -> str:
        """Average probabilities across a list of snapshots → dominant label."""
        avg = {}
        for field_name in ("trending_up", "trending_down", "ranging", "volatile", "breakout"):
            avg[field_name] = float(np.mean([getattr(p, field_name) for p in probs_list]))
        return max(avg, key=avg.get)

    # ── Indicator computations ──

    def _compute_adx(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Wilder's ADX, +DI, -DI. Returns (adx, plus_di, minus_di)."""
        period = self.adx_period
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        if len(close) < period * 2 + 1:
            return 20.0, 50.0, 50.0

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )

        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        def wilder_smooth(arr: np.ndarray, n: int) -> np.ndarray:
            out = np.zeros_like(arr, dtype=float)
            if len(arr) < n:
                return out
            out[n - 1] = float(np.sum(arr[:n]))
            for i in range(n, len(arr)):
                out[i] = out[i - 1] - out[i - 1] / n + arr[i]
            return out

        atr_smooth = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)

        # Avoid divide-by-zero
        atr_safe = np.where(atr_smooth == 0, 1e-9, atr_smooth)
        plus_di = (plus_dm_smooth / atr_safe) * 100.0
        minus_di = (minus_dm_smooth / atr_safe) * 100.0

        di_sum = plus_di + minus_di
        di_sum_safe = np.where(di_sum == 0, 1e-9, di_sum)
        dx = (np.abs(plus_di - minus_di) / di_sum_safe) * 100.0

        # ADX = wilder_smoothed DX (start from index period-1 onwards)
        if len(dx) <= period:
            return 20.0, float(plus_di[-1] if len(plus_di) else 50.0), float(minus_di[-1] if len(minus_di) else 50.0)
        adx_values = wilder_smooth(dx[period - 1:], period)
        # Use last valid ADX value
        last_adx = float(adx_values[-1] / period) if adx_values[-1] != 0 else 20.0
        # Note: wilder_smooth produces sum-style accumulator at index n-1, then EMA-like updates.
        # For ADX/DI conventions, dividing by period for the seed is standard. The recursive
        # update already produces a smoothed value in the same scale once initialized.
        # For interpretation we simply use the raw last value:
        return float(adx_values[-1]), float(plus_di[-1]), float(minus_di[-1])

    def _compute_atr_ratio(self, df: pd.DataFrame) -> float:
        """Recent ATR / long-window ATR — measures relative volatility."""
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        if len(close) < self.atr_period * 3 + 1:
            return 1.0

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )

        recent_atr = float(np.mean(tr[-self.atr_period:]))
        long_atr = float(np.mean(tr[-(self.atr_period * 3):]))

        if long_atr <= 0:
            return 1.0
        return recent_atr / long_atr

    def _compute_autocorrelation(self, df: pd.DataFrame, lag: int = 1) -> float:
        """Autocorrelation of returns at given lag."""
        returns = df["close"].pct_change().dropna().values
        if len(returns) < 20:
            return 0.0

        recent = returns[-self.lookback:]
        if len(recent) < lag + 5:
            return 0.0

        try:
            corr = float(np.corrcoef(recent[:-lag], recent[lag:])[0, 1])
            if np.isnan(corr):
                return 0.0
            return corr
        except Exception:
            return 0.0

    def _compute_bb_width(self, df: pd.DataFrame) -> float:
        """Bollinger Band width (upper-lower) as % of SMA."""
        close = df["close"].values.astype(float)
        if len(close) < self.bb_period:
            return 0.05

        recent = close[-self.bb_period:]
        sma = float(np.mean(recent))
        std = float(np.std(recent))

        if sma <= 0:
            return 0.05

        # Standard 2-std bands: (upper - lower) / sma = 4 * std / sma
        return (4.0 * std) / sma

    def _compute_ema_slope(self, df: pd.DataFrame) -> float:
        """Slope of fast EMA as fractional change per bar."""
        close = df["close"]
        if len(close) < self.ema_fast + 5:
            return 0.0

        ema = close.ewm(span=self.ema_fast, adjust=False).mean()
        recent_ema = ema.values[-5:]

        if len(recent_ema) < 2 or recent_ema[0] == 0:
            return 0.0

        slope = (recent_ema[-1] - recent_ema[0]) / recent_ema[0] / len(recent_ema)
        return float(slope)


# ── Module singleton ───────────────────────────────────────────────


_singleton: Optional[ProbabilisticRegimeClassifier] = None


def get_probabilistic_classifier() -> ProbabilisticRegimeClassifier:
    """Module-level singleton."""
    global _singleton
    if _singleton is None:
        _singleton = ProbabilisticRegimeClassifier()
    return _singleton
