"""
============================================================
ONBOARDING FEATURE EXTRACTOR
Wraps MicroFeatureComputer for onboarding use case
============================================================
File: soe/feature_extractor.py
============================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger("SOE.FeatureExtractor")


class OnboardingFeatureExtractor:
    """
    Extracts features from each training example.
    
    Wraps MicroFeatureComputer and adds:
    - Context features (regime, location, time)
    - Broader market features
    - Explanation-aligned features
    
    Produces a complete feature vector per example.
    """

    # All features this extractor can compute
    ALL_FEATURES = [
        # Candle Structure (from MicroFeatureComputer)
        "wick_ratio_mean", "wick_ratio_max",
        "body_consistency", "body_size_variance",
        "avg_body_size", "body_to_range_ratio",
        "upper_wick_dominance", "lower_wick_dominance",

        # Range / Compression
        "range_contraction_r2", "range_width_change",
        "range_width_normalized", "squeeze_score",
        "compression_candle_count", "range_volatility",
        "bb_width_percentile",

        # Volume
        "volume_trend_slope", "volume_ratio_last5",
        "avg_volume_ratio", "volume_spike_count",
        "max_volume_ratio", "volume_price_correlation",
        "volume_consistency",

        # Direction / Entropy
        "direction_entropy", "direction_consistency",
        "consecutive_same_direction_max",
        "body_direction_ratio",  # % of candles that are bullish

        # Momentum
        "rsi_at_setup", "momentum_score",
        "rate_of_change", "macd_histogram_trend",

        # Location
        "distance_to_nearest_support", "distance_to_nearest_resistance",
        "support_strength", "resistance_strength",
        "distance_to_vwap", "price_percentile_in_day_range",
        "distance_to_ema20", "distance_to_ema50",
        "atr_normalized_location",

        # Trend / Regime
        "adx", "trend_strength", "trend_direction_score",
        "price_vs_ema20", "price_vs_ema50",
        "higher_lows_count", "lower_highs_count",

        # Time
        "time_bucket",  # 0=premarket, 1=morning, 2=midday, 3=afternoon, 4=close
        "minutes_since_open",
        "is_first_hour", "is_last_hour",

        # Breakout characteristics (if applicable)
        "breakout_candle_size_vs_avg",
        "breakout_volume_vs_avg",
        "breakout_wick_ratio",
    ]

    def __init__(self, micro_feature_computer=None):
        """
        Args:
            micro_feature_computer: Existing MicroFeatureComputer instance
                                   If None, uses built-in computation
        """
        self.micro_feature_computer = micro_feature_computer
        self._feature_cache = {}

    def extract_features(
        self,
        example: 'TrainingExample',
        additional_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Extract complete feature vector from a training example.
        
        Returns dict of feature_name → value
        """
        features = {}

        if example.candle_data is not None and len(example.candle_data) > 10:
            # Extract from candle data
            df = example.candle_data

            # Candle structure features
            features.update(self._extract_candle_features(df))

            # Range / compression features
            features.update(self._extract_range_features(df))

            # Volume features
            features.update(self._extract_volume_features(df))

            # Direction / entropy features
            features.update(self._extract_direction_features(df))

            # Momentum features
            features.update(self._extract_momentum_features(df))

            # Location features
            features.update(self._extract_location_features(df, additional_data))

            # Trend features
            features.update(self._extract_trend_features(df))

            # Time features
            features.update(self._extract_time_features(df, example.entry_time))

        else:
            logger.warning(
                f"No candle data for {example.example_id}. "
                f"Using explanation-only features."
            )
            # Fallback: infer what we can from explanation tags
            features = self._infer_features_from_explanation(example)

        # Store in example
        example.features = features
        return features

    def _extract_candle_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract candle structure features from OHLCV data"""
        features = {}

        # Use last 20 candles as the "setup zone"
        setup = df.tail(20) if len(df) >= 20 else df

        opens = setup['open'].values
        highs = setup['high'].values
        lows = setup['low'].values
        closes = setup['close'].values

        # Body and wick calculations
        bodies = np.abs(closes - opens)
        ranges = highs - lows
        ranges = np.where(ranges == 0, 1e-10, ranges)  # avoid div/0

        upper_wicks = highs - np.maximum(opens, closes)
        lower_wicks = np.minimum(opens, closes) - lows

        wick_ratios = (upper_wicks + lower_wicks) / ranges

        features["wick_ratio_mean"] = float(np.mean(wick_ratios))
        features["wick_ratio_max"] = float(np.max(wick_ratios))

        # Body consistency (how similar are body sizes)
        if np.mean(bodies) > 0:
            features["body_size_variance"] = float(np.std(bodies) / np.mean(bodies))
        else:
            features["body_size_variance"] = 0.0

        features["body_consistency"] = max(0, 1.0 - features["body_size_variance"])
        features["avg_body_size"] = float(np.mean(bodies))
        features["body_to_range_ratio"] = float(np.mean(bodies / ranges))

        # Wick dominance
        features["upper_wick_dominance"] = float(
            np.mean(upper_wicks / ranges)
        )
        features["lower_wick_dominance"] = float(
            np.mean(lower_wicks / ranges)
        )

        return features

    def _extract_range_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract range/compression features"""
        features = {}

        setup = df.tail(30) if len(df) >= 30 else df
        highs = setup['high'].values
        lows = setup['low'].values

        # Rolling range width
        window = min(5, len(setup))
        range_widths = []
        for i in range(window, len(setup)):
            rng = max(highs[i-window:i]) - min(lows[i-window:i])
            range_widths.append(rng)

        if len(range_widths) >= 3:
            # Range contraction (R² of linear fit to range widths)
            x = np.arange(len(range_widths))
            if np.std(range_widths) > 0:
                correlation = np.corrcoef(x, range_widths)[0, 1]
                r2 = correlation ** 2
                features["range_contraction_r2"] = float(r2)

                # Negative slope = contracting
                slope = np.polyfit(x, range_widths, 1)[0]
                features["range_width_change"] = float(slope)
            else:
                features["range_contraction_r2"] = 1.0  # Perfect flat = perfect squeeze
                features["range_width_change"] = 0.0

            # Squeeze score (combination)
            features["squeeze_score"] = float(
                features["range_contraction_r2"] *
                (1.0 if features["range_width_change"] <= 0 else 0.5)
            )
        else:
            features["range_contraction_r2"] = 0.0
            features["range_width_change"] = 0.0
            features["squeeze_score"] = 0.0

        # Normalized range width
        atr = np.mean(highs - lows)
        if atr > 0:
            features["range_width_normalized"] = float(
                (max(highs[-10:]) - min(lows[-10:])) / atr
            )
        else:
            features["range_width_normalized"] = 0.0

        # How many candles in compression
        features["compression_candle_count"] = float(
            self._count_compression_candles(highs, lows)
        )

        # Range volatility (std of range widths)
        if range_widths:
            features["range_volatility"] = float(np.std(range_widths))
        else:
            features["range_volatility"] = 0.0

        return features

    def _count_compression_candles(
        self, highs: np.ndarray, lows: np.ndarray
    ) -> int:
        """Count how many consecutive candles show tightening range"""
        if len(highs) < 5:
            return 0

        ranges = highs - lows
        count = 0

        for i in range(len(ranges) - 1, 0, -1):
            if ranges[i] <= ranges[i-1] * 1.1:  # Allow 10% tolerance
                count += 1
            else:
                break

        return count

    def _extract_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract volume features"""
        features = {}

        if 'volume' not in df.columns:
            return {
                "volume_trend_slope": 0.0,
                "volume_ratio_last5": 1.0,
                "avg_volume_ratio": 1.0,
                "volume_spike_count": 0.0,
                "max_volume_ratio": 1.0,
                "volume_price_correlation": 0.0,
                "volume_consistency": 0.5
            }

        setup = df.tail(20) if len(df) >= 20 else df
        volumes = setup['volume'].values.astype(float)
        avg_vol = np.mean(volumes) if np.mean(volumes) > 0 else 1

        # Volume trend (slope of linear fit)
        x = np.arange(len(volumes))
        if len(volumes) >= 3 and np.std(volumes) > 0:
            slope = np.polyfit(x, volumes / avg_vol, 1)[0]
            features["volume_trend_slope"] = float(slope)
        else:
            features["volume_trend_slope"] = 0.0

        # Last 5 vs average
        last5_avg = np.mean(volumes[-5:]) if len(volumes) >= 5 else avg_vol
        features["volume_ratio_last5"] = float(last5_avg / avg_vol)

        # Average volume ratio (setup vs broader)
        if len(df) >= 30:
            broader_avg = np.mean(df['volume'].values[:20].astype(float))
            if broader_avg > 0:
                features["avg_volume_ratio"] = float(avg_vol / broader_avg)
            else:
                features["avg_volume_ratio"] = 1.0
        else:
            features["avg_volume_ratio"] = 1.0

        # Volume spikes (>2x average)
        features["volume_spike_count"] = float(
            np.sum(volumes > avg_vol * 2)
        )
        features["max_volume_ratio"] = float(np.max(volumes) / avg_vol)

        # Volume-price correlation
        closes = setup['close'].values
        if len(volumes) >= 5 and np.std(volumes) > 0 and np.std(closes) > 0:
            features["volume_price_correlation"] = float(
                np.corrcoef(volumes, np.abs(np.diff(np.append(closes[0], closes))))[0, 1]
            )
        else:
            features["volume_price_correlation"] = 0.0

        # Volume consistency
        if avg_vol > 0:
            features["volume_consistency"] = float(
                1.0 - min(1.0, np.std(volumes) / avg_vol)
            )
        else:
            features["volume_consistency"] = 0.5

        return features

    def _extract_direction_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract directional/entropy features"""
        features = {}

        setup = df.tail(20) if len(df) >= 20 else df
        opens = setup['open'].values
        closes = setup['close'].values

        # Direction: 1 = bullish, 0 = bearish
        directions = (closes > opens).astype(int)

        # Direction entropy (randomness of candle directions)
        if len(directions) > 1:
            p_bull = np.mean(directions)
            p_bear = 1 - p_bull
            if 0 < p_bull < 1:
                entropy = -(p_bull * np.log2(p_bull) + p_bear * np.log2(p_bear))
            else:
                entropy = 0.0
            features["direction_entropy"] = float(entropy)
        else:
            features["direction_entropy"] = 0.5

        # Direction consistency
        features["direction_consistency"] = float(
            1.0 - features["direction_entropy"]
        )

        # Maximum consecutive same direction
        max_consecutive = 1
        current = 1
        for i in range(1, len(directions)):
            if directions[i] == directions[i-1]:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 1
        features["consecutive_same_direction_max"] = float(max_consecutive)

        # Bullish ratio
        features["body_direction_ratio"] = float(np.mean(directions))

        return features

    def _extract_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract momentum features"""
        features = {}

        closes = df['close'].values

        # RSI (14-period)
        if len(closes) >= 15:
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features["rsi_at_setup"] = float(100 - (100 / (1 + rs)))
            else:
                features["rsi_at_setup"] = 100.0
        else:
            features["rsi_at_setup"] = 50.0

        # Rate of change
        if len(closes) >= 10:
            features["rate_of_change"] = float(
                (closes[-1] - closes[-10]) / closes[-10]
            )
        else:
            features["rate_of_change"] = 0.0

        # Simple momentum score
        if len(closes) >= 5:
            short_change = (closes[-1] - closes[-5]) / closes[-5]
            features["momentum_score"] = float(short_change)
        else:
            features["momentum_score"] = 0.0

        return features

    def _extract_location_features(
        self,
        df: pd.DataFrame,
        additional_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Extract location-based features"""
        features = {}

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        current_price = closes[-1]

        # ATR for normalization
        atr = np.mean(highs[-20:] - lows[-20:]) if len(df) >= 20 else np.mean(highs - lows)
        if atr == 0:
            atr = 1e-10

        # Simple support/resistance from price history
        # Support: recent lows that were tested
        # Resistance: recent highs that were tested
        if len(df) >= 20:
            recent_lows = self._find_swing_points(lows, direction="low")
            recent_highs = self._find_swing_points(highs, direction="high")

            if recent_lows:
                nearest_support = min(recent_lows, key=lambda x: abs(current_price - x))
                features["distance_to_nearest_support"] = float(
                    (current_price - nearest_support) / atr
                )
            else:
                features["distance_to_nearest_support"] = 5.0  # Far away

            if recent_highs:
                nearest_resistance = min(recent_highs, key=lambda x: abs(x - current_price))
                features["distance_to_nearest_resistance"] = float(
                    (nearest_resistance - current_price) / atr
                )
            else:
                features["distance_to_nearest_resistance"] = 5.0  # Far away

        return features

    def _extract_trend_features(self, df) -> Dict[str, float]:
        """Extract trend features"""
        features = {}
        closes = df["close"].values
        if len(closes) >= 20:
            ema20 = np.mean(closes[-20:])
            features["price_vs_ema20"] = float((closes[-1] - ema20) / ema20)
        else:
            features["price_vs_ema20"] = 0.0
        if len(closes) >= 50:
            ema50 = np.mean(closes[-50:])
            features["price_vs_ema50"] = float((closes[-1] - ema50) / ema50)
        else:
            features["price_vs_ema50"] = 0.0
        features["trend_direction_score"] = features.get("price_vs_ema20", 0.0)
        features["trend_strength"] = abs(features.get("price_vs_ema50", 0.0))
        features["adx"] = 25.0  # Placeholder, needs real ADX calculation
        return features

    def _extract_time_features(self, df, entry_time=None) -> Dict[str, float]:
        """Extract time-based features"""
        features = {}
        features["time_bucket"] = 1.0  # Default: morning
        features["minutes_since_open"] = 60.0
        features["is_first_hour"] = 1.0
        features["is_last_hour"] = 0.0
        if entry_time:
            try:
                parts = entry_time.split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                market_open_hour = 9
                mins = (hour - market_open_hour) * 60 + minute - 15
                features["minutes_since_open"] = float(max(0, mins))
                if hour < 10:
                    features["time_bucket"] = 1.0
                    features["is_first_hour"] = 1.0
                elif hour < 12:
                    features["time_bucket"] = 2.0
                    features["is_first_hour"] = 0.0
                elif hour < 14:
                    features["time_bucket"] = 3.0
                    features["is_first_hour"] = 0.0
                else:
                    features["time_bucket"] = 4.0
                    features["is_first_hour"] = 0.0
                    features["is_last_hour"] = 1.0 if hour >= 14 else 0.0
            except (ValueError, IndexError):
                pass
        return features

    def _infer_features_from_explanation(self, example) -> Dict[str, float]:
        """Infer basic features from explanation tags when no candle data"""
        features = {}
        tags = getattr(example, "explanation_tags", [])
        if "clean" in tags or "smooth" in tags:
            features["wick_ratio_mean"] = 0.15
            features["body_consistency"] = 0.8
        if "choppy" in tags or "messy" in tags:
            features["wick_ratio_mean"] = 0.45
            features["body_consistency"] = 0.3
        if "compression" in tags or "squeeze" in tags:
            features["range_contraction_r2"] = 0.7
            features["squeeze_score"] = 0.8
        if "volume_dying" in tags or "declining_volume" in tags:
            features["volume_trend_slope"] = -0.3
        return features

    def _find_swing_points(self, values, direction="high"):
        """Find swing highs or lows"""
        if len(values) < 5:
            return list(values)
        swing_points = []
        for i in range(2, len(values) - 2):
            if direction == "high":
                if (values[i] > values[i-1] and values[i] > values[i-2] and
                    values[i] > values[i+1] and values[i] > values[i+2]):
                    swing_points.append(float(values[i]))
            else:
                if (values[i] < values[i-1] and values[i] < values[i-2] and
                    values[i] < values[i+1] and values[i] < values[i+2]):
                    swing_points.append(float(values[i]))
        return swing_points if swing_points else [float(values[-1])]
