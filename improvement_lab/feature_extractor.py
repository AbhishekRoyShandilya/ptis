# improvement_lab/feature_extractor.py
"""
Feature Extractor — The 80% of the ML Work
=============================================

Extracts SPECIFIC, DEFINED features for each detector type:

1. RR Killer features — captured at the moment trade reaches X:R
   Price exhaustion, volume signals, market context, structural, time

2. Early Exit features — captured N candles before SL hit
   Engulfing patterns, volume spikes, level breaks, divergences

3. General trade context — shared across detectors
   Regime, session, emotion, streak, etc.

Every feature has a name, definition, and computation.
The MODEL is 20% of the work. THIS is the 80%.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from improvement_lab.models import TradeFeatureVector

logger = logging.getLogger(__name__)

# IST offset
IST = timezone(timedelta(hours=5, minutes=30))

# Categorical encodings
REGIME_VALUES = [
    "trending_up", "trending_down", "range", "ranging",
    "volatile", "high_volatility", "low_vol", "breakout",
    "mean_reverting", "unknown",
]
SESSION_VALUES = ["pre_market", "opening", "morning", "afternoon", "closing", "post_market"]
EMOTION_VALUES = ["calm", "confident", "anxious", "fearful", "excited", "frustrated", "neutral"]
VOLATILITY_VALUES = ["low", "average", "high", "extreme"]


def _one_hot(value: Optional[str], categories: List[str], prefix: str) -> Dict[str, float]:
    """One-hot encode a categorical value."""
    result = {}
    val = (value or "").lower().strip()
    for cat in categories:
        result[f"{prefix}_{cat}"] = 1.0 if val == cat else 0.0
    return result


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert to float, return default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _get_field(obj: Any, field: str, default: Any = None) -> Any:
    """Get field from ORM object or dict."""
    if isinstance(obj, dict):
        return obj.get(field, default)
    val = getattr(obj, field, default)
    return default if val is None else val


class FeatureExtractor:
    """
    Extracts ML feature vectors from Trade objects.

    Three extraction modes:
      1. extract_trade_context() — general features for any detector
      2. extract_rr_killer_features() — features at moment of peak R
      3. extract_early_exit_features() — features N candles before SL hit

    Usage:
        extractor = FeatureExtractor()
        vectors = extractor.extract_batch(trades)
    """

    # ── General trade context ─────────────────────────────────────

    def extract_trade_context(
        self,
        trade: Any,
        recent_trades: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Extract general trade context features.
        Shared across all detectors.
        """
        ctx = context or {}
        features: Dict[str, float] = {}

        # Market context
        regime = _get_field(trade, "market_regime", "unknown")
        if hasattr(regime, "value"):
            regime = regime.value
        features.update(_one_hot(str(regime), REGIME_VALUES, "regime"))

        vol_regime = ctx.get("volatility_regime", "average")
        features.update(_one_hot(str(vol_regime), VOLATILITY_VALUES, "vol"))

        features["atr_percentile"] = _safe_float(ctx.get("atr_percentile"), 0.5)

        # Momentum / trend
        features["rsi_at_entry"] = _safe_float(ctx.get("rsi_at_entry"), 50.0)
        features["trend_alignment"] = _safe_float(ctx.get("trend_alignment"), 0.0)

        # Volume
        features["volume_ratio"] = _safe_float(ctx.get("volume_ratio"), 1.0)

        # Time
        entry_time = _get_field(trade, "entry_time", None)
        if isinstance(entry_time, datetime):
            ist_time = entry_time.astimezone(IST) if entry_time.tzinfo else entry_time
            features["hour_ist"] = float(ist_time.hour) + ist_time.minute / 60.0
            features["day_of_week"] = float(ist_time.weekday())
        else:
            features["hour_ist"] = 10.0
            features["day_of_week"] = 2.0

        session = _get_field(trade, "session_type", "morning")
        if hasattr(session, "value"):
            session = session.value
        features.update(_one_hot(str(session), SESSION_VALUES, "session"))

        # Trader state
        streak = self._compute_streak(recent_trades) if recent_trades else 0
        features["streak_at_entry"] = float(streak)
        features["trades_today"] = _safe_float(ctx.get("trades_today"), 0.0)

        emotion = _get_field(trade, "emotion_pre", "neutral")
        if hasattr(emotion, "value"):
            emotion = emotion.value
        features.update(_one_hot(str(emotion), EMOTION_VALUES, "emotion"))

        features["conviction_score"] = _safe_float(
            _get_field(trade, "conviction_score", 5.0), 5.0
        )
        features["discipline_score"] = _safe_float(ctx.get("discipline_score"), 7.0)

        # Trade params
        direction = _get_field(trade, "direction", "long")
        if hasattr(direction, "value"):
            direction = direction.value
        features["direction"] = 1.0 if direction == "long" else -1.0
        features["risk_pct"] = _safe_float(_get_field(trade, "risk_pct", 1.0), 1.0)

        # Performance context
        features["drawdown_pct"] = _safe_float(ctx.get("drawdown_pct"), 0.0)
        features["recent_win_rate"] = self._compute_recent_wr(recent_trades)

        return features

    # ── R:R Killer features (at moment trade reaches X:R) ─────────

    def extract_rr_killer_features(
        self,
        trade: Any,
        candles: Optional[List[Dict[str, float]]] = None,
        peak_candle_idx: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Extract features at the moment a trade reaches peak R.

        Args:
            trade: Trade object
            candles: OHLCV candles during the trade [{open, high, low, close, volume}, ...]
            peak_candle_idx: index in candles where MFE occurred
            context: additional market context at peak
        """
        ctx = context or {}
        features: Dict[str, float] = {}

        # ── Price Exhaustion Signals ──

        features["distance_from_nearest_resistance_atr"] = _safe_float(
            ctx.get("distance_from_nearest_resistance_atr"), 2.0
        )

        features["rsi_at_peak"] = _safe_float(ctx.get("rsi_at_peak"), 50.0)

        features["extension_from_vwap_atr"] = _safe_float(
            ctx.get("extension_from_vwap_atr"), 0.0
        )

        # Candle exhaustion score — shrinking bodies, growing wicks
        features["candle_exhaustion_score"] = self._compute_candle_exhaustion(
            candles, peak_candle_idx
        )

        # Move acceleration — is the move speeding up or slowing down?
        features["move_acceleration"] = self._compute_move_acceleration(
            candles, peak_candle_idx
        )

        # ── Volume Signals ──

        features["volume_at_peak_vs_breakout"] = self._compute_volume_ratio_at_peak(
            candles, peak_candle_idx
        )

        features["volume_climax_detected"] = self._detect_volume_climax(
            candles, peak_candle_idx
        )

        features["cumulative_volume_profile"] = self._compute_volume_profile(
            candles, peak_candle_idx
        )

        # ── Market Context at Peak ──

        regime_at_peak = ctx.get("regime_at_peak", "unknown")
        features.update(_one_hot(str(regime_at_peak), REGIME_VALUES, "peak_regime"))

        features["nifty_momentum_alignment"] = _safe_float(
            ctx.get("nifty_momentum_alignment"), 0.0
        )

        features["sector_exhaustion"] = _safe_float(
            ctx.get("sector_exhaustion_percentile"), 0.5
        )

        # ── Structural Context ──

        features["next_level_distance_atr"] = _safe_float(
            ctx.get("next_level_distance_atr"), 3.0
        )

        features["gap_fill_proximity"] = 1.0 if ctx.get("gap_fill_proximity") else 0.0

        features["round_number_proximity"] = _safe_float(
            ctx.get("round_number_proximity_atr"), 1.0
        )

        # ── Time Context ──

        entry_time = _get_field(trade, "entry_time", None)
        peak_time = ctx.get("peak_time")
        if isinstance(entry_time, datetime) and isinstance(peak_time, datetime):
            features["minutes_since_entry"] = (peak_time - entry_time).total_seconds() / 60.0
        else:
            features["minutes_since_entry"] = _safe_float(ctx.get("minutes_since_entry"), 30.0)

        features["time_of_day_ist"] = _safe_float(ctx.get("time_of_day_ist"), 10.0)

        features["candles_since_last_pullback"] = _safe_float(
            ctx.get("candles_since_last_pullback"),
            self._compute_candles_since_pullback(candles, peak_candle_idx)
        )

        # ── Trade-Specific History ──

        variant_id = ctx.get("variant_id", "")
        features["variant_hash"] = hash(variant_id) % 1000 / 1000.0 if variant_id else 0.0

        features["entry_conviction_score"] = _safe_float(
            _get_field(trade, "conviction_score", 5.0), 5.0
        )

        features["historical_max_r_for_variant"] = _safe_float(
            ctx.get("historical_max_r_for_variant"), 3.0
        )

        return features

    # ── Early Exit features (N candles before SL hit) ─────────────

    def extract_early_exit_features(
        self,
        trade: Any,
        candles: Optional[List[Dict[str, float]]] = None,
        lookback_candles: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Extract warning signals N candles before SL was hit.

        Args:
            trade: Trade object
            candles: OHLCV candles during the trade
            lookback_candles: how many candles before SL to look
            context: additional context
        """
        ctx = context or {}
        features: Dict[str, float] = {}
        direction = _get_field(trade, "direction", "long")
        if hasattr(direction, "value"):
            direction = direction.value
        is_long = direction == "long"

        if candles and len(candles) >= lookback_candles:
            window = candles[-lookback_candles:]
            atr = self._compute_atr(candles) if len(candles) >= 14 else None

            features["engulfing_against_formed"] = float(
                self._detect_engulfing_against(window, is_long)
            )
            features["volume_spike_against"] = float(
                self._detect_volume_spike_against(candles, window, is_long)
            )
            features["key_level_broken_against"] = _safe_float(
                ctx.get("key_level_broken_against"), 0.0
            )
            features["momentum_divergence"] = float(
                self._detect_momentum_divergence(candles, is_long)
            )
            features["vwap_cross_against"] = _safe_float(
                ctx.get("vwap_cross_against"), 0.0
            )
            features["regime_shift_detected"] = _safe_float(
                ctx.get("regime_shift_detected"), 0.0
            )
            features["wide_range_bar_against"] = float(
                self._detect_wide_range_bar(window, is_long, atr)
            )
            features["consecutive_closes_against"] = float(
                self._count_consecutive_against(window, is_long)
            )
            features["spread_widening"] = _safe_float(
                ctx.get("spread_widening"), 0.0
            )
            features["related_stock_breaking_down"] = _safe_float(
                ctx.get("related_stock_breaking_down"), 0.0
            )
        else:
            # No candle data — fill from context or defaults
            for feat_name in [
                "engulfing_against_formed", "volume_spike_against",
                "key_level_broken_against", "momentum_divergence",
                "vwap_cross_against", "regime_shift_detected",
                "wide_range_bar_against", "consecutive_closes_against",
                "spread_widening", "related_stock_breaking_down",
            ]:
                features[feat_name] = _safe_float(ctx.get(feat_name), 0.0)

        return features

    # ── Full extraction ───────────────────────────────────────────

    def extract(
        self,
        trade: Any,
        recent_trades: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TradeFeatureVector:
        """Extract features from a single trade (general context only)."""
        ctx = context or {}
        features = self.extract_trade_context(trade, recent_trades, ctx)

        trade_id = str(_get_field(trade, "id", ""))
        strategy_id = str(
            _get_field(trade, "declared_strategy_id", "")
            or ctx.get("strategy_id", "unknown")
        )
        variant_id = ctx.get("variant_id")

        r_multiple = _safe_float(_get_field(trade, "r_multiple"), 0.0)
        pnl = _safe_float(_get_field(trade, "pnl"), 0.0)

        # Check for override events
        overrides = _get_field(trade, "overrides", []) or []
        override_types = []
        if overrides:
            for ov in overrides:
                ov_type = _get_field(ov, "override_type", "")
                if hasattr(ov_type, "value"):
                    ov_type = ov_type.value
                if ov_type:
                    override_types.append(str(ov_type))

        # Shadow data if available
        shadow_final_r = _safe_float(ctx.get("shadow_final_r"), None)
        shadow_max_r = _safe_float(ctx.get("shadow_max_favorable_r"), None)

        return TradeFeatureVector(
            trade_id=trade_id,
            strategy_id=strategy_id,
            variant_id=variant_id,
            features=features,
            hit_sl=(r_multiple <= -0.9) if r_multiple else None,
            was_winner=pnl > 0 if pnl else None,
            final_r=r_multiple if r_multiple else None,
            had_overrides=len(override_types) > 0,
            override_types=override_types,
            shadow_final_r=shadow_final_r,
            shadow_max_favorable_r=shadow_max_r,
        )

    def extract_batch(
        self,
        trades: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TradeFeatureVector]:
        """Extract feature vectors for a list of trades."""
        vectors = []
        for i, trade in enumerate(trades):
            recent = trades[max(0, i - 20):i] if i > 0 else []
            vec = self.extract(trade, recent_trades=recent, context=context)
            vectors.append(vec)
        return vectors

    def vectors_to_matrix(
        self, vectors: List[TradeFeatureVector],
    ) -> Tuple[np.ndarray, List[str]]:
        """Convert list of TradeFeatureVector to numpy matrix + feature names."""
        if not vectors:
            return np.empty((0, 0)), []

        feature_names = sorted(vectors[0].features.keys())
        X = np.zeros((len(vectors), len(feature_names)))

        for i, vec in enumerate(vectors):
            for j, fname in enumerate(feature_names):
                X[i, j] = vec.features.get(fname, 0.0)

        return X, feature_names

    # ── Candle computation helpers ────────────────────────────────

    @staticmethod
    def _compute_candle_exhaustion(
        candles: Optional[List[Dict]], peak_idx: Optional[int],
    ) -> float:
        """
        Candle exhaustion score: shrinking bodies + growing upper wicks.
        Returns 0-1 (1 = extreme exhaustion).
        """
        if not candles or peak_idx is None or peak_idx < 6:
            return 0.5

        # First 3 candles of the move vs last 3 before peak
        first_3 = candles[max(0, peak_idx - 6):max(0, peak_idx - 3)]
        last_3 = candles[max(0, peak_idx - 2):peak_idx + 1]

        if not first_3 or not last_3:
            return 0.5

        def avg_body(bars: List[Dict]) -> float:
            bodies = [abs(b.get("close", 0) - b.get("open", 0)) for b in bars]
            return np.mean(bodies) if bodies else 1.0

        def avg_upper_wick_ratio(bars: List[Dict]) -> float:
            ratios = []
            for b in bars:
                body = abs(b.get("close", 0) - b.get("open", 0))
                high = b.get("high", 0)
                top = max(b.get("open", 0), b.get("close", 0))
                upper_wick = high - top
                total = body + upper_wick + 1e-10
                ratios.append(upper_wick / total)
            return float(np.mean(ratios)) if ratios else 0.0

        body_ratio = avg_body(last_3) / max(avg_body(first_3), 1e-10)
        wick_change = avg_upper_wick_ratio(last_3) - avg_upper_wick_ratio(first_3)

        # Exhaustion = shrinking bodies (ratio < 1) + growing wicks
        score = (1.0 - min(body_ratio, 1.0)) * 0.6 + max(wick_change, 0) * 0.4
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _compute_move_acceleration(
        candles: Optional[List[Dict]], peak_idx: Optional[int],
    ) -> float:
        """
        Second derivative of price — is the move accelerating or decelerating?
        Returns < 1 = decelerating, > 1 = accelerating.
        """
        if not candles or peak_idx is None or peak_idx < 6:
            return 1.0

        half = peak_idx // 2
        first_half = candles[max(0, half - 3):half]
        last_3 = candles[max(0, peak_idx - 2):peak_idx + 1]

        if not first_half or not last_3:
            return 1.0

        def slope(bars: List[Dict]) -> float:
            if len(bars) < 2:
                return 0.0
            closes = [b.get("close", 0) for b in bars]
            return (closes[-1] - closes[0]) / max(len(bars), 1)

        s_first = abs(slope(first_half))
        s_last = abs(slope(last_3))
        return s_last / max(s_first, 1e-10)

    @staticmethod
    def _compute_volume_ratio_at_peak(
        candles: Optional[List[Dict]], peak_idx: Optional[int],
    ) -> float:
        """Volume at peak / volume at entry. < 1 = fading."""
        if not candles or peak_idx is None or len(candles) < 2:
            return 1.0

        entry_vol = candles[0].get("volume", 1)
        peak_vol = candles[min(peak_idx, len(candles) - 1)].get("volume", 1)
        return peak_vol / max(entry_vol, 1)

    @staticmethod
    def _detect_volume_climax(
        candles: Optional[List[Dict]], peak_idx: Optional[int],
    ) -> float:
        """Did any candle in last 3 have volume > 3x average?"""
        if not candles or peak_idx is None or len(candles) < 5:
            return 0.0

        volumes = [c.get("volume", 0) for c in candles if c.get("volume")]
        if not volumes:
            return 0.0
        avg_vol = np.mean(volumes)

        last_3 = candles[max(0, peak_idx - 2):peak_idx + 1]
        for c in last_3:
            if c.get("volume", 0) > 3 * avg_vol:
                return 1.0
        return 0.0

    @staticmethod
    def _compute_volume_profile(
        candles: Optional[List[Dict]], peak_idx: Optional[int],
    ) -> float:
        """
        Front-loaded (1.0) vs back-loaded (-1.0) volume distribution.
        Healthy moves have front-loaded volume.
        """
        if not candles or peak_idx is None or peak_idx < 4:
            return 0.0

        move_candles = candles[:peak_idx + 1]
        if len(move_candles) < 4:
            return 0.0

        mid = len(move_candles) // 2
        first_half_vol = sum(c.get("volume", 0) for c in move_candles[:mid])
        second_half_vol = sum(c.get("volume", 0) for c in move_candles[mid:])
        total = first_half_vol + second_half_vol

        if total < 1:
            return 0.0

        # Positive = front-loaded, negative = back-loaded
        return (first_half_vol - second_half_vol) / total

    @staticmethod
    def _compute_candles_since_pullback(
        candles: Optional[List[Dict]], peak_idx: Optional[int],
        pullback_threshold_frac: float = 0.3,
    ) -> float:
        """Count candles since price pulled back > threshold fraction of ATR."""
        if not candles or peak_idx is None or peak_idx < 2:
            return 0.0

        # Simple: count consecutive candles that didn't close below prior close
        count = 0
        for i in range(peak_idx, 0, -1):
            curr_close = candles[i].get("close", 0)
            prev_close = candles[i - 1].get("close", 0)
            if curr_close < prev_close:
                break
            count += 1
        return float(count)

    # ── Early exit candle helpers ─────────────────────────────────

    @staticmethod
    def _detect_engulfing_against(window: List[Dict], is_long: bool) -> bool:
        """Detect engulfing candle against the trade direction."""
        for i in range(1, len(window)):
            prev = window[i - 1]
            curr = window[i]
            prev_body = abs(prev.get("close", 0) - prev.get("open", 0))
            curr_body = abs(curr.get("close", 0) - curr.get("open", 0))

            if curr_body <= prev_body:
                continue

            curr_bearish = curr.get("close", 0) < curr.get("open", 0)
            curr_bullish = curr.get("close", 0) > curr.get("open", 0)

            if is_long and curr_bearish:
                # Bearish engulfing against long
                if (curr.get("open", 0) >= prev.get("close", 0) and
                        curr.get("close", 0) <= prev.get("open", 0)):
                    return True
            elif not is_long and curr_bullish:
                # Bullish engulfing against short
                if (curr.get("open", 0) <= prev.get("close", 0) and
                        curr.get("close", 0) >= prev.get("open", 0)):
                    return True
        return False

    @staticmethod
    def _detect_volume_spike_against(
        all_candles: List[Dict], window: List[Dict], is_long: bool,
    ) -> bool:
        """Volume > 2x average on a candle moving against position."""
        volumes = [c.get("volume", 0) for c in all_candles if c.get("volume")]
        avg_vol = np.mean(volumes) if volumes else 0

        for c in window:
            vol = c.get("volume", 0)
            if vol < 2 * avg_vol:
                continue
            # Is it moving against?
            if is_long and c.get("close", 0) < c.get("open", 0):
                return True
            if not is_long and c.get("close", 0) > c.get("open", 0):
                return True
        return False

    @staticmethod
    def _detect_momentum_divergence(candles: List[Dict], is_long: bool) -> bool:
        """
        Simplified RSI divergence detection.
        Price making new high but RSI making lower high (for longs).
        """
        if len(candles) < 14:
            return False

        closes = np.array([c.get("close", 0) for c in candles])
        # Simple RSI approximation
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        if len(gains) < 14:
            return False

        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        prev_avg_gain = np.mean(gains[-28:-14]) if len(gains) >= 28 else avg_gain
        prev_avg_loss = np.mean(losses[-28:-14]) if len(losses) >= 28 else avg_loss

        def rsi_from_avgs(ag: float, al: float) -> float:
            if al < 1e-10:
                return 100.0
            rs = ag / al
            return 100 - (100 / (1 + rs))

        rsi_now = rsi_from_avgs(avg_gain, avg_loss)
        rsi_prev = rsi_from_avgs(prev_avg_gain, prev_avg_loss)

        if is_long:
            # Price higher but RSI lower = bearish divergence
            price_higher = closes[-1] > np.max(closes[-28:-14]) if len(closes) >= 28 else False
            return price_higher and rsi_now < rsi_prev
        else:
            price_lower = closes[-1] < np.min(closes[-28:-14]) if len(closes) >= 28 else False
            return price_lower and rsi_now > rsi_prev

    @staticmethod
    def _detect_wide_range_bar(
        window: List[Dict], is_long: bool, atr: Optional[float],
    ) -> bool:
        """Single candle with range > 1.5x ATR moving against."""
        if atr is None or atr < 1e-10:
            return False

        for c in window:
            bar_range = c.get("high", 0) - c.get("low", 0)
            if bar_range < 1.5 * atr:
                continue
            if is_long and c.get("close", 0) < c.get("open", 0):
                return True
            if not is_long and c.get("close", 0) > c.get("open", 0):
                return True
        return False

    @staticmethod
    def _count_consecutive_against(window: List[Dict], is_long: bool) -> int:
        """Count consecutive candle closes moving against position."""
        count = 0
        for c in reversed(window):
            if is_long and c.get("close", 0) < c.get("open", 0):
                count += 1
            elif not is_long and c.get("close", 0) > c.get("open", 0):
                count += 1
            else:
                break
        return count

    @staticmethod
    def _compute_atr(candles: List[Dict], period: int = 14) -> Optional[float]:
        """Simple ATR computation."""
        if len(candles) < period + 1:
            return None

        trs = []
        for i in range(1, len(candles)):
            h = candles[i].get("high", 0)
            l = candles[i].get("low", 0)
            pc = candles[i - 1].get("close", 0)
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)

        if len(trs) < period:
            return None

        # Wilder smoothing
        atr = np.mean(trs[:period])
        alpha = 1.0 / period
        for tr in trs[period:]:
            atr = atr * (1 - alpha) + tr * alpha
        return float(atr)

    # ── Streak / win rate helpers ─────────────────────────────────

    @staticmethod
    def _compute_streak(recent_trades: List[Any]) -> int:
        """Compute current win/loss streak from recent trades."""
        if not recent_trades:
            return 0
        streak = 0
        for t in recent_trades:
            pnl = _safe_float(
                t.get("pnl") if isinstance(t, dict) else getattr(t, "pnl", 0), 0
            )
            if streak == 0:
                streak = 1 if pnl > 0 else -1 if pnl < 0 else 0
            elif pnl > 0 and streak > 0:
                streak += 1
            elif pnl < 0 and streak < 0:
                streak -= 1
            else:
                break
        return streak

    @staticmethod
    def _compute_recent_wr(recent_trades: Optional[List[Any]]) -> float:
        """Win rate of last 20 trades."""
        if not recent_trades:
            return 0.5
        wins = 0
        total = 0
        for t in recent_trades[-20:]:
            pnl = _safe_float(
                t.get("pnl") if isinstance(t, dict) else getattr(t, "pnl", 0), 0
            )
            if pnl != 0:
                total += 1
                if pnl > 0:
                    wins += 1
        return wins / total if total > 0 else 0.5
