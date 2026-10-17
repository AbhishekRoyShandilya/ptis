"""
Regime Classifier
==================
Classifies market conditions into regimes (trending_up, trending_down, ranging, volatile)
using price action features. Supports rule-based classification (no external deps) and
an optional scikit-learn Random Forest for improved accuracy with enough data.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Regime labels
REGIME_TRENDING_UP   = "trending_up"
REGIME_TRENDING_DOWN = "trending_down"
REGIME_RANGING       = "ranging"
REGIME_VOLATILE      = "volatile"
REGIME_UNKNOWN       = "unknown"

ALL_REGIMES = [REGIME_TRENDING_UP, REGIME_TRENDING_DOWN,
               REGIME_RANGING, REGIME_VOLATILE, REGIME_UNKNOWN]


@dataclass
class RegimeClassification:
    regime: str
    confidence: float       # 0-1
    features: dict          # raw features used for classification
    method: str             # "rule_based" | "ml"


class RuleBasedRegimeClassifier:
    """
    Classifies regime from OHLCV candle data without any ML library.
    Works on a sliding window of candles (list of dicts with open/high/low/close/volume).
    """

    def classify(self, candles: list[dict], window: int = 20) -> RegimeClassification:
        """
        candles: list of {"open": float, "high": float, "low": float, "close": float, "volume": float}
        window: lookback period for feature calculation
        """
        if len(candles) < window:
            return RegimeClassification(REGIME_UNKNOWN, 0.0, {}, "rule_based")

        recent = candles[-window:]
        closes = [c["close"] for c in recent]
        highs  = [c["high"]  for c in recent]
        lows   = [c["low"]   for c in recent]

        # Feature 1: Linear trend slope (normalized by mean price)
        n = len(closes)
        mean_price = statistics.mean(closes) or 1.0
        xs = list(range(n))
        x_mean = statistics.mean(xs)
        c_mean = mean_price
        num = sum((x - x_mean) * (c - c_mean) for x, c in zip(xs, closes))
        den = sum((x - x_mean) ** 2 for x in xs) or 1.0
        slope = num / den / mean_price   # normalized slope per bar

        # Feature 2: Price range as % of mean price (volatility proxy)
        total_range = max(highs) - min(lows)
        range_pct = total_range / mean_price

        # Feature 3: Std deviation of closes (dispersion)
        std_pct = statistics.stdev(closes) / mean_price if len(closes) >= 2 else 0.0

        # Feature 4: Directional consistency (% of bars moving in dominant direction)
        up_bars = sum(1 for i in range(1, n) if closes[i] > closes[i - 1])
        dir_ratio = up_bars / (n - 1) if n > 1 else 0.5

        features = {
            "slope_pct": round(slope * 100, 4),
            "range_pct": round(range_pct * 100, 2),
            "std_pct": round(std_pct * 100, 2),
            "dir_ratio": round(dir_ratio, 3),
        }

        # Classification thresholds (tuned for daily/intraday candles)
        ABS_SLOPE_TREND = 0.0015    # >0.15% per bar = trending
        VOLATILE_RANGE  = 0.04      # >4% total range in window = volatile
        RANGING_STD     = 0.008     # <0.8% std = ranging

        if range_pct > VOLATILE_RANGE and std_pct > 0.012:
            regime = REGIME_VOLATILE
            confidence = min(range_pct / (VOLATILE_RANGE * 2), 1.0)
        elif abs(slope) > ABS_SLOPE_TREND:
            if slope > 0 and dir_ratio > 0.55:
                regime = REGIME_TRENDING_UP
            elif slope < 0 and dir_ratio < 0.45:
                regime = REGIME_TRENDING_DOWN
            else:
                regime = REGIME_VOLATILE
            confidence = min(abs(slope) / (ABS_SLOPE_TREND * 3), 1.0)
        elif std_pct < RANGING_STD:
            regime = REGIME_RANGING
            confidence = min((RANGING_STD - std_pct) / RANGING_STD, 1.0)
        else:
            regime = REGIME_RANGING
            confidence = 0.5

        return RegimeClassification(regime=regime, confidence=round(confidence, 3),
                                    features=features, method="rule_based")


class MLRegimeClassifier:
    """
    Random Forest regime classifier trained on labeled candle data.
    Requires: pip install scikit-learn
    Falls back gracefully if scikit-learn is not installed.
    """

    def __init__(self):
        self._model = None
        self._trained = False

    def _feature_vector(self, candles: list[dict], window: int = 20) -> Optional[list[float]]:
        if len(candles) < window:
            return None
        recent = candles[-window:]
        closes = [c["close"] for c in recent]
        highs  = [c["high"]  for c in recent]
        lows   = [c["low"]   for c in recent]
        mean_p = statistics.mean(closes) or 1.0

        n = len(closes)
        xs = list(range(n))
        xm = statistics.mean(xs)
        num = sum((x - xm) * (c - mean_p) for x, c in zip(xs, closes))
        den = sum((x - xm) ** 2 for x in xs) or 1.0
        slope = num / den / mean_p
        range_pct = (max(highs) - min(lows)) / mean_p
        std_pct = statistics.stdev(closes) / mean_p if n >= 2 else 0.0
        up_bars = sum(1 for i in range(1, n) if closes[i] > closes[i - 1])
        dir_ratio = up_bars / (n - 1) if n > 1 else 0.5
        # additional: momentum (last 5 vs first 5)
        momentum = (statistics.mean(closes[-5:]) - statistics.mean(closes[:5])) / mean_p
        return [slope, range_pct, std_pct, dir_ratio, momentum]

    def train(self, candle_windows: list[list[dict]], labels: list[str],
              window: int = 20) -> bool:
        """
        Train on historical labeled data.
        labels: list of regime strings matching ALL_REGIMES
        Returns True if training succeeded.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            logger.warning("scikit-learn not installed; ML regime classifier unavailable")
            return False

        X, y = [], []
        for candles, label in zip(candle_windows, labels):
            fv = self._feature_vector(candles, window)
            if fv:
                X.append(fv)
                y.append(label)

        if len(X) < 10:
            logger.warning("Not enough training samples for regime classifier")
            return False

        self._le = LabelEncoder().fit(y)
        self._model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._model.fit(X, self._le.transform(y))
        self._trained = True
        logger.info(f"Regime classifier trained on {len(X)} samples")
        return True

    def classify(self, candles: list[dict], window: int = 20) -> RegimeClassification:
        fv = self._feature_vector(candles, window)
        if fv is None or not self._trained or self._model is None:
            return RuleBasedRegimeClassifier().classify(candles, window)

        proba = self._model.predict_proba([fv])[0]
        idx = proba.argmax()
        regime = self._le.inverse_transform([idx])[0]
        confidence = float(proba[idx])
        return RegimeClassification(regime=regime, confidence=round(confidence, 3),
                                    features={"fv": fv}, method="ml")

    def save(self, path: str) -> None:
        if not self._trained:
            raise RuntimeError("Model not trained")
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "le": self._le}, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._le    = data["le"]
        self._trained = True


def get_regime_classifier(use_ml: bool = False) -> RuleBasedRegimeClassifier | MLRegimeClassifier:
    return MLRegimeClassifier() if use_ml else RuleBasedRegimeClassifier()
