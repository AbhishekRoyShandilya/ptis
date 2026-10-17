# bootstrap/judgment_learner.py
"""
Judgment Learner
=================

Learns the user's SUBJECTIVE judgment by observing their
accept/reject decisions on setups.

When user says "I skip ugly compressions", the system
doesn't just label this as uncertainty. It LEARNS what
"ugly" means by:

1. Showing user setups and recording accept/reject
2. Computing micro-features the user can't articulate
3. Training a model: features → user_would_trade (yes/no)
4. Interpreting the model to explain what "ugly" means
5. Converting learned judgment into a soft score
6. Reducing uncertainty as the model improves

THE KEY INSIGHT:
The user's judgment IS consistent. They just can't
describe the algorithm. We can reverse-engineer it
from their decisions.

This is supervised learning where the user IS the label.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score
    from sklearn.inspection import permutation_importance
    _SKLEARN_AVAILABLE = True
except ImportError:
    logger.info("sklearn not available — using fallback judgment learning")


# ══════════════════════════════════════════════════════════════════
#  MICRO-FEATURE DEFINITIONS
# ══════════════════════════════════════════════════════════════════

"""
These are features the user processes VISUALLY but can't
articulate. We compute them from raw candle data.

The user sees a chart and says "ugly" in 2 seconds.
They're actually processing ALL of these simultaneously.
"""

COMPRESSION_MICRO_FEATURES = {
    # ── Candle structure ──
    "avg_wick_ratio": (
        "Average (upper_wick + lower_wick) / body across "
        "compression candles. High = indecisive (ugly). "
        "Low = clean bodies."
    ),
    "wick_ratio_variance": (
        "How consistent are the wick ratios? "
        "High variance = erratic candles (ugly)."
    ),
    "body_size_cv": (
        "Coefficient of variation of candle body sizes. "
        "High = inconsistent sizing (ugly). "
        "Low = orderly, similar-sized bodies (clean)."
    ),
    "body_direction_entropy": (
        "How random is the alternation between green/red candles? "
        "High entropy = no directional bias (ugly). "
        "Some bias = subtle accumulation direction (clean)."
    ),

    # ── Range structure ──
    "range_smoothness": (
        "How smoothly does the range contract? "
        "Measured as R² of linear fit on range sizes. "
        "High R² = smooth contraction (clean). "
        "Low R² = jagged, random narrowing (ugly)."
    ),
    "range_contraction_monotonicity": (
        "What fraction of consecutive range pairs are decreasing? "
        "1.0 = perfectly monotonic contraction. "
        "0.5 = random. Lower than expected = ugly."
    ),
    "has_range_gap": (
        "Is there a candle where range suddenly expands then "
        "re-contracts? This is a 'hole' in the compression. "
        "Boolean → 1 = has gap (ugly), 0 = no gap (clean)."
    ),

    # ── Volume structure ──
    "volume_trend_during_compression": (
        "Slope of volume across compression. "
        "Negative = drying up (clean: sellers exhausted). "
        "Positive or flat = no dry-up (ugly: still contested)."
    ),
    "volume_spike_count": (
        "Number of volume spikes (> 2x avg) during compression. "
        "0 = quiet compression (clean). "
        "2+ = active trading during squeeze (ugly)."
    ),

    # ── Structural context ──
    "distance_to_resistance_atr": (
        "How close is the compression top to next resistance? "
        "Very close = will fail on breakout (ugly). "
        "1+ ATR away = room to run (clean)."
    ),
    "compression_position_in_range": (
        "Where in the stock's daily range is the compression? "
        "Near top of daily range = extended (ugly). "
        "Near VWAP or middle = balanced (clean)."
    ),

    # ── Visual harmony (what the eye actually sees) ──
    "candle_overlap_pct": (
        "What % of consecutive candles overlap in price? "
        "High overlap = tight, orderly (clean). "
        "Low overlap = gapping within compression (ugly)."
    ),
    "symmetry_score": (
        "How symmetric is the compression shape? "
        "Triangle/wedge = symmetric (clean). "
        "Lopsided = ugly."
    ),
}


# ══════════════════════════════════════════════════════════════════
#  MICRO-FEATURE COMPUTER
# ══════════════════════════════════════════════════════════════════


@dataclass
class MicroFeatureVector:
    """Computed micro-features for one setup."""
    features: Dict[str, float] = field(default_factory=dict)
    raw_candles_used: int = 0
    computation_confidence: float = 0.0


class MicroFeatureComputer:
    """
    Computes the micro-features that model human visual judgment.
    
    These features capture what the eye sees but the brain
    can't articulate. They're the bridge between
    "this looks ugly" and quantifiable measurement.
    """

    def compute(
        self,
        candles: List[Dict[str, float]],
        compression_start: int,
        compression_end: int,
        context_candles: Optional[List[Dict]] = None,
    ) -> MicroFeatureVector:
        """
        Compute all micro-features for a compression setup.
        
        Args:
            candles: all candles in the relevant window
            compression_start: index where compression begins
            compression_end: index where compression ends
            context_candles: broader context (for resistance, etc.)
        """
        comp_candles = candles[compression_start:compression_end + 1]
        
        if len(comp_candles) < 3:
            return MicroFeatureVector(
                computation_confidence=0.0
            )

        features = {}
        confidence_scores = []

        # ── Candle structure features ──
        try:
            wick_ratios = []
            body_sizes = []
            body_directions = []  # 1 = green, 0 = red

            for c in comp_candles:
                o = float(c.get("open", 0))
                h = float(c.get("high", 0))
                l = float(c.get("low", 0))
                cl = float(c.get("close", 0))

                body = abs(cl - o)
                upper_wick = h - max(o, cl)
                lower_wick = min(o, cl) - l
                total_wick = upper_wick + lower_wick

                if body > 0:
                    wick_ratios.append(total_wick / body)
                else:
                    wick_ratios.append(10.0)  # doji = high wick ratio

                body_sizes.append(body)
                body_directions.append(1 if cl >= o else 0)

            wick_arr = np.array(wick_ratios)
            body_arr = np.array(body_sizes)
            dir_arr = np.array(body_directions)

            features["avg_wick_ratio"] = round(
                float(np.mean(wick_arr)), 4
            )
            features["wick_ratio_variance"] = round(
                float(np.var(wick_arr)), 4
            )

            # Body size coefficient of variation
            body_mean = float(np.mean(body_arr))
            body_std = float(np.std(body_arr))
            features["body_size_cv"] = round(
                body_std / max(body_mean, 1e-8), 4
            )

            # Direction entropy
            # Shannon entropy of green/red distribution
            green_pct = float(np.mean(dir_arr))
            if 0 < green_pct < 1:
                entropy = -(
                    green_pct * np.log2(green_pct)
                    + (1 - green_pct) * np.log2(1 - green_pct)
                )
            else:
                entropy = 0.0  # all same direction = 0 entropy
            features["body_direction_entropy"] = round(entropy, 4)

            confidence_scores.append(0.9)

        except Exception as e:
            logger.debug("Candle structure computation failed: %s", e)
            confidence_scores.append(0.0)

        # ── Range structure features ──
        try:
            ranges = []
            for c in comp_candles:
                h = float(c.get("high", 0))
                l = float(c.get("low", 0))
                ranges.append(h - l)

            range_arr = np.array(ranges)

            # Range smoothness (R² of linear fit)
            x = np.arange(len(range_arr))
            if len(range_arr) >= 3:
                # Linear regression
                slope, intercept = np.polyfit(x, range_arr, 1)
                predicted = slope * x + intercept
                ss_res = np.sum((range_arr - predicted) ** 2)
                ss_tot = np.sum(
                    (range_arr - np.mean(range_arr)) ** 2
                )
                r_squared = 1 - (ss_res / max(ss_tot, 1e-8))
                features["range_smoothness"] = round(
                    max(0.0, float(r_squared)), 4
                )
            else:
                features["range_smoothness"] = 0.5

            # Monotonicity: fraction of consecutive decreasing ranges
            if len(range_arr) >= 2:
                decreasing = sum(
                    1 for i in range(1, len(range_arr))
                    if range_arr[i] <= range_arr[i - 1]
                )
                features["range_contraction_monotonicity"] = round(
                    decreasing / (len(range_arr) - 1), 4
                )
            else:
                features["range_contraction_monotonicity"] = 0.5

            # Gap detection: range suddenly expands > 1.5x then contracts
            has_gap = False
            for i in range(1, len(range_arr) - 1):
                if (
                    range_arr[i] > range_arr[i - 1] * 1.5
                    and range_arr[i] > range_arr[i + 1] * 1.3
                ):
                    has_gap = True
                    break
            features["has_range_gap"] = 1.0 if has_gap else 0.0

            confidence_scores.append(0.9)

        except Exception as e:
            logger.debug("Range structure computation failed: %s", e)
            confidence_scores.append(0.0)

        # ── Volume structure features ──
        try:
            volumes = [
                float(c.get("volume", 0)) for c in comp_candles
            ]
            vol_arr = np.array(volumes)

            if len(vol_arr) >= 3 and np.mean(vol_arr) > 0:
                # Volume trend (slope normalized by mean)
                x = np.arange(len(vol_arr))
                slope = np.polyfit(x, vol_arr, 1)[0]
                features["volume_trend_during_compression"] = round(
                    slope / max(float(np.mean(vol_arr)), 1), 4
                )

                # Volume spike count
                avg_vol = float(np.mean(vol_arr))
                spikes = int(np.sum(vol_arr > avg_vol * 2))
                features["volume_spike_count"] = float(spikes)
            else:
                features["volume_trend_during_compression"] = 0.0
                features["volume_spike_count"] = 0.0

            confidence_scores.append(0.8)

        except Exception as e:
            logger.debug("Volume computation failed: %s", e)
            confidence_scores.append(0.0)

        # ── Visual harmony features ──
        try:
            # Candle overlap percentage
            overlaps = 0
            for i in range(1, len(comp_candles)):
                prev_high = float(comp_candles[i - 1].get("high", 0))
                prev_low = float(comp_candles[i - 1].get("low", 0))
                curr_high = float(comp_candles[i].get("high", 0))
                curr_low = float(comp_candles[i].get("low", 0))

                overlap = min(prev_high, curr_high) - max(prev_low, curr_low)
                if overlap > 0:
                    overlaps += 1

            features["candle_overlap_pct"] = round(
                overlaps / max(1, len(comp_candles) - 1), 4
            )

            # Symmetry: compare first half range vs second half range
            mid = len(comp_candles) // 2
            first_half = comp_candles[:mid]
            second_half = comp_candles[mid:]

            if first_half and second_half:
                fh_range = max(
                    c.get("high", 0) for c in first_half
                ) - min(c.get("low", 0) for c in first_half)
                sh_range = max(
                    c.get("high", 0) for c in second_half
                ) - min(c.get("low", 0) for c in second_half)

                if fh_range > 0:
                    ratio = sh_range / fh_range
                    # Perfect contraction: ratio = 0.5-0.7
                    # Symmetric wedge: ratio ≈ 0.6
                    symmetry = 1.0 - abs(ratio - 0.6) * 2
                    features["symmetry_score"] = round(
                        max(0.0, min(1.0, symmetry)), 4
                    )
                else:
                    features["symmetry_score"] = 0.5
            else:
                features["symmetry_score"] = 0.5

            confidence_scores.append(0.7)

        except Exception as e:
            logger.debug("Visual harmony computation failed: %s", e)
            confidence_scores.append(0.0)

        # ── Structural context ──
        if context_candles:
            try:
                # Distance to resistance
                comp_high = max(
                    float(c.get("high", 0)) for c in comp_candles
                )
                # Simple resistance: highest high in prior 50 candles
                prior_highs = [
                    float(c.get("high", 0))
                    for c in context_candles[-50:]
                ]
                if prior_highs:
                    resistance = max(prior_highs)
                    atr = self._compute_atr(context_candles[-20:])
                    if atr > 0:
                        features["distance_to_resistance_atr"] = round(
                            (resistance - comp_high) / atr, 4
                        )

                # Position in daily range
                day_high = max(prior_highs) if prior_highs else comp_high
                day_low = min(
                    float(c.get("low", 0))
                    for c in context_candles[-50:]
                ) if context_candles else comp_high
                day_range = day_high - day_low

                comp_mid = (comp_high + min(
                    float(c.get("low", 0)) for c in comp_candles
                )) / 2

                if day_range > 0:
                    features["compression_position_in_range"] = round(
                        (comp_mid - day_low) / day_range, 4
                    )

                confidence_scores.append(0.7)

            except Exception as e:
                logger.debug("Context computation failed: %s", e)
                confidence_scores.append(0.0)

        overall_confidence = (
            float(np.mean(confidence_scores))
            if confidence_scores
            else 0.0
        )

        return MicroFeatureVector(
            features=features,
            raw_candles_used=len(comp_candles),
            computation_confidence=round(overall_confidence, 3),
        )

    @staticmethod
    def _compute_atr(
        candles: List[Dict], period: int = 14
    ) -> float:
        """Simple ATR computation."""
        if len(candles) < 2:
            return 0.0
        trs = []
        for i in range(1, len(candles)):
            h = float(candles[i].get("high", 0))
            l = float(candles[i].get("low", 0))
            pc = float(candles[i - 1].get("close", 0))
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        return float(np.mean(trs[-period:])) if trs else 0.0


# ══════════════════════════════════════════════════════════════════
#  JUDGMENT MODEL
# ══════════════════════════════════════════════════════════════════


@dataclass
class JudgmentModelState:
    """Current state of the judgment learner."""
    judgment_name: str = ""         # e.g., "compression_cleanness"
    total_samples: int = 0
    positive_samples: int = 0       # user accepted
    negative_samples: int = 0       # user rejected

    # Model performance
    model_accuracy: float = 0.0
    model_auc: float = 0.0
    model_ready: bool = False

    # Learned insights
    top_positive_features: List[Tuple[str, float]] = field(
        default_factory=list
    )
    top_negative_features: List[Tuple[str, float]] = field(
        default_factory=list
    )
    learned_threshold: Optional[float] = None

    # Impact on uncertainty
    uncertainty_reduction: float = 0.0


@dataclass
class JudgmentPrediction:
    """Model's prediction of user's judgment."""
    would_user_accept: bool = False
    acceptance_probability: float = 0.5
    confidence: float = 0.0
    top_reasons_for: List[str] = field(default_factory=list)
    top_reasons_against: List[str] = field(default_factory=list)
    model_used: str = ""        # "bayesian" | "gbm" | "ensemble"


class JudgmentLearner:
    """
    Learns ONE specific judgment from user's decisions.
    
    Example: "compression_cleanness"
    
    Training data:
      features: micro-features of the setup
      label: did user accept (1) or reject (0)?
    
    Uses two-speed learning:
    FAST: Bayesian conditional tables (update per decision)
    DEEP: GBM model (retrain every 20 decisions)
    
    Over time:
    - Uncertainty for this judgment factor DECREASES
    - Soft score for this factor becomes MODEL-DRIVEN
      instead of placeholder
    - System explains what "ugly" actually means
    """

    def __init__(
        self,
        judgment_name: str,
        micro_features: List[str],
        min_samples_for_model: int = 15,
        deep_retrain_interval: int = 20,
    ):
        self._name = judgment_name
        self._micro_features = micro_features
        self._min_samples = min_samples_for_model
        self._deep_interval = deep_retrain_interval

        # Training data
        self._X: List[Dict[str, float]] = []
        self._y: List[int] = []  # 1 = accepted, 0 = rejected

        # FAST layer: Bayesian tables
        # For each feature, track P(accept | feature_bucket)
        self._bayesian_tables: Dict[str, Dict[int, List[int]]] = (
            defaultdict(lambda: defaultdict(lambda: [0, 0]))
        )
        # [reject_count, accept_count] per bucket

        # DEEP layer: trained model
        self._deep_model = None
        self._deep_model_score = 0.0
        self._last_deep_train_count = 0

        # State
        self._state = JudgmentModelState(judgment_name=judgment_name)

    def record_decision(
        self,
        micro_features: Dict[str, float],
        user_accepted: bool,
    ) -> None:
        """
        Record a user's accept/reject decision.
        Updates fast layer immediately.
        Triggers deep retrain when enough data.
        """
        self._X.append(micro_features)
        self._y.append(1 if user_accepted else 0)

        self._state.total_samples = len(self._y)
        self._state.positive_samples = sum(self._y)
        self._state.negative_samples = len(self._y) - sum(self._y)

        # Fast layer update
        for feature_name in self._micro_features:
            val = micro_features.get(feature_name)
            if val is None:
                continue

            # Bucket the value (5 buckets)
            bucket = self._to_bucket(feature_name, float(val))

            table = self._bayesian_tables[feature_name]
            if user_accepted:
                table[bucket][1] += 1  # accept count
            else:
                table[bucket][0] += 1  # reject count

        # Check deep retrain
        samples_since_train = len(self._y) - self._last_deep_train_count
        if (
            len(self._y) >= self._min_samples
            and samples_since_train >= self._deep_interval
        ):
            self._retrain_deep()

    def predict(
        self, micro_features: Dict[str, float]
    ) -> JudgmentPrediction:
        """
        Predict whether the user would accept this setup.
        
        Uses ensemble of fast (Bayesian) + deep (GBM) layers.
        """
        prediction = JudgmentPrediction()

        # Fast layer prediction
        fast_prob = self._predict_bayesian(micro_features)

        # Deep layer prediction
        deep_prob = None
        if self._deep_model is not None and _SKLEARN_AVAILABLE:
            deep_prob = self._predict_deep(micro_features)

        # Ensemble
        if deep_prob is not None:
            # Weighted ensemble: deep gets more weight as it improves
            deep_weight = min(0.7, self._deep_model_score)
            fast_weight = 1.0 - deep_weight
            prob = fast_prob * fast_weight + deep_prob * deep_weight
            prediction.model_used = "ensemble"
        else:
            prob = fast_prob
            prediction.model_used = "bayesian"

        prediction.acceptance_probability = round(prob, 3)
        prediction.would_user_accept = prob >= 0.5

        # Confidence based on data and model quality
        data_conf = min(0.8, len(self._y) / 50)
        model_conf = self._deep_model_score if self._deep_model else 0.3
        prediction.confidence = round(
            data_conf * 0.4 + model_conf * 0.6, 3
        )

        # Explain prediction
        prediction.top_reasons_for = self._explain_positive(
            micro_features
        )
        prediction.top_reasons_against = self._explain_negative(
            micro_features
        )

        return prediction

    def get_soft_score(
        self, micro_features: Dict[str, float]
    ) -> float:
        """
        Convert judgment prediction into a soft score (0-1).
        This REPLACES the placeholder soft score as the
        model improves.
        
        The soft score IS the model's acceptance probability.
        """
        if self._state.total_samples < self._min_samples:
            return 0.5  # not enough data

        pred = self.predict(micro_features)
        return pred.acceptance_probability

    def get_uncertainty_reduction(self) -> float:
        """
        How much has this learner reduced uncertainty?
        Returns 0-1 where 1 = fully resolved.
        
        This feeds back into the UncertaintyLayer to
        reduce the penalty for this judgment factor.
        """
        if self._state.total_samples < self._min_samples:
            return 0.0

        # Based on model accuracy and data quantity
        data_factor = min(0.4, self._state.total_samples / 100)
        
        if self._deep_model is not None:
            # Deep model accuracy contribution
            model_factor = max(0, self._deep_model_score - 0.5) * 0.8
        else:
            # Bayesian-only: limited uncertainty reduction
            model_factor = 0.1

        reduction = min(0.8, data_factor + model_factor)
        self._state.uncertainty_reduction = round(reduction, 3)
        return reduction

    def explain_judgment(self) -> Dict[str, Any]:
        """
        Human-readable explanation of what the model learned.
        
        This is THE key output. It tells the user:
        "Here's what 'ugly' actually means based on
        your 47 accept/reject decisions."
        """
        if self._state.total_samples < self._min_samples:
            return {
                "status": "learning",
                "samples": self._state.total_samples,
                "needed": self._min_samples,
                "message": (
                    f"Need {self._min_samples - self._state.total_samples} "
                    f"more accept/reject decisions to learn your "
                    f"'{self._name}' judgment."
                ),
            }

        # Get feature importances
        importances = self._get_feature_importances()

        # Build explanation
        positive_factors = []
        negative_factors = []

        for feat_name, importance in importances:
            if importance < 0.02:
                continue

            # Determine direction: does high value = accept or reject?
            direction = self._get_feature_direction(feat_name)
            description = COMPRESSION_MICRO_FEATURES.get(
                feat_name, feat_name
            )

            if direction == "positive":
                positive_factors.append({
                    "feature": feat_name,
                    "importance": round(importance, 3),
                    "meaning": (
                        f"Higher {feat_name} → more likely to accept"
                    ),
                    "description": description,
                })
            else:
                negative_factors.append({
                    "feature": feat_name,
                    "importance": round(importance, 3),
                    "meaning": (
                        f"Higher {feat_name} → more likely to reject"
                    ),
                    "description": description,
                })

        return {
            "status": "learned",
            "judgment_name": self._name,
            "total_decisions": self._state.total_samples,
            "model_accuracy": round(self._deep_model_score, 3),
            "acceptance_rate": round(
                self._state.positive_samples
                / max(1, self._state.total_samples),
                3,
            ),
            "uncertainty_reduction": self.get_uncertainty_reduction(),
            "what_you_like": positive_factors[:5],
            "what_you_dislike": negative_factors[:5],
            "plain_english": self._generate_plain_explanation(
                positive_factors, negative_factors
            ),
        }

    # ── Persistence ────────────────────────────────────────────────

    @property
    def total_samples(self) -> int:
        """Number of recorded accept/reject decisions."""
        return len(self._y)

    def export_decisions(self) -> Dict[str, Any]:
        """
        Serialize this learner's full training history for durable storage.

        We persist the RAW decisions (features + labels) rather than the
        fitted GBM. The Bayesian tables and the GBM are both deterministically
        rebuilt from these on load — so the blob is small, portable, and
        immune to sklearn-version / feature-schema drift.
        """
        return {
            "micro_features": list(self._micro_features),
            "X": self._X,
            "y": self._y,
            "model_score": round(float(self._deep_model_score), 4),
        }

    def load_decisions(
        self,
        X: List[Dict[str, float]],
        y: List[int],
        retrain: bool = True,
    ) -> None:
        """
        Restore training history saved by export_decisions().

        Rebuilds the Bayesian conditional tables from scratch and (if enough
        samples) retrains the deep GBM, leaving the learner exactly as if every
        decision had been replayed.
        """
        self._X = [dict(x) for x in (X or [])]
        self._y = [int(v) for v in (y or [])]

        self._state.total_samples = len(self._y)
        self._state.positive_samples = sum(self._y)
        self._state.negative_samples = len(self._y) - sum(self._y)

        # Rebuild Bayesian tables from the full decision history
        self._bayesian_tables = defaultdict(
            lambda: defaultdict(lambda: [0, 0])
        )
        for feats, label in zip(self._X, self._y):
            for feature_name in self._micro_features:
                val = feats.get(feature_name)
                if val is None:
                    continue
                bucket = self._to_bucket(feature_name, float(val))
                self._bayesian_tables[feature_name][bucket][
                    1 if label else 0
                ] += 1

        # Rebuild the deep model from the restored data
        self._deep_model = None
        self._last_deep_train_count = 0
        if retrain and len(self._y) >= self._min_samples:
            self._retrain_deep()

    # ── Fast layer (Bayesian) ─────────────────────────────────────

    def _predict_bayesian(
        self, features: Dict[str, float]
    ) -> float:
        """
        Bayesian prediction from conditional tables.
        P(accept | features) ∝ ∏ P(accept | feature_bucket)
        """
        if not self._bayesian_tables:
            return 0.5

        # Prior: overall acceptance rate
        total = max(1, self._state.total_samples)
        prior = self._state.positive_samples / total

        log_likelihood_ratio = 0.0

        for feat_name in self._micro_features:
            val = features.get(feat_name)
            if val is None:
                continue

            bucket = self._to_bucket(feat_name, float(val))
            table = self._bayesian_tables.get(feat_name, {})
            cell = table.get(bucket, [0, 0])

            reject_count, accept_count = cell
            total_cell = reject_count + accept_count

            if total_cell < 2:
                continue  # not enough data for this bucket

            # Laplace-smoothed probability
            p_accept = (accept_count + 1) / (total_cell + 2)
            p_reject = (reject_count + 1) / (total_cell + 2)

            if p_reject > 0 and p_accept > 0:
                log_likelihood_ratio += np.log(p_accept / p_reject)

        # Convert log-odds to probability
        log_prior_odds = np.log(
            max(prior, 0.01) / max(1 - prior, 0.01)
        )
        log_posterior_odds = log_prior_odds + log_likelihood_ratio

        # Sigmoid
        prob = 1.0 / (1.0 + np.exp(-log_posterior_odds))
        return float(np.clip(prob, 0.01, 0.99))

    def _to_bucket(
        self, feature_name: str, value: float, n_buckets: int = 5
    ) -> int:
        """
        Bucket a feature value. Uses learned quantiles
        when available, falls back to fixed ranges.
        """
        # Get all values for this feature
        all_vals = [
            x.get(feature_name, 0.0) for x in self._X
            if feature_name in x
        ]

        if len(all_vals) < 10:
            # Not enough data — use simple bucketing
            # Normalize to 0-1 range
            if all_vals:
                min_v = min(all_vals)
                max_v = max(all_vals)
                rng = max_v - min_v
                if rng > 0:
                    normalized = (value - min_v) / rng
                    return min(
                        n_buckets - 1,
                        max(0, int(normalized * n_buckets)),
                    )
            return n_buckets // 2

        # Quantile-based bucketing
        sorted_vals = sorted(all_vals)
        for i in range(n_buckets):
            threshold_idx = int((i + 1) / n_buckets * len(sorted_vals)) - 1
            threshold_idx = max(0, min(len(sorted_vals) - 1, threshold_idx))
            if value <= sorted_vals[threshold_idx]:
                return i

        return n_buckets - 1

    # ── Deep layer (GBM) ──────────────────────────────────────────

    def _retrain_deep(self) -> None:
        """Retrain the deep model."""
        if not _SKLEARN_AVAILABLE:
            return
        if len(self._y) < self._min_samples:
            return

        # Build feature matrix
        feature_names = self._micro_features
        X = np.array([
            [x.get(f, 0.0) for f in feature_names]
            for x in self._X
        ])
        y = np.array(self._y)

        # Check class balance
        pos_ratio = np.mean(y)
        if pos_ratio < 0.1 or pos_ratio > 0.9:
            logger.info(
                "Judgment '%s': skipping deep train — "
                "class imbalance (%.0f%% positive)",
                self._name, pos_ratio * 100,
            )
            return

        try:
            # Train GBM
            model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=max(2, len(y) // 10),
                learning_rate=0.1,
                random_state=42,
            )

            # Cross-validation
            if len(y) >= 20:
                cv_scores = cross_val_score(
                    model, X, y, cv=min(5, len(y) // 5),
                    scoring="accuracy",
                )
                self._deep_model_score = float(np.mean(cv_scores))
            else:
                self._deep_model_score = 0.5

            # Train on all data
            model.fit(X, y)
            self._deep_model = model
            self._last_deep_train_count = len(y)

            # Extract feature importances
            importances = model.feature_importances_
            self._state.top_positive_features = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            self._state.model_accuracy = self._deep_model_score
            self._state.model_ready = self._deep_model_score > 0.55

            logger.info(
                "Judgment '%s' deep model trained: "
                "accuracy=%.3f, samples=%d",
                self._name,
                self._deep_model_score,
                len(y),
            )

        except Exception as e:
            logger.warning(
                "Deep model training failed for '%s': %s",
                self._name, e,
            )

    def _predict_deep(
        self, features: Dict[str, float]
    ) -> Optional[float]:
        """Predict using deep model."""
        if self._deep_model is None:
            return None

        try:
            x = np.array([[
                features.get(f, 0.0) for f in self._micro_features
            ]])
            proba = self._deep_model.predict_proba(x)[0]
            # proba[1] = P(accept)
            return float(proba[1]) if len(proba) > 1 else 0.5
        except Exception:
            return None

    # ── Feature analysis ──────────────────────────────────────────

    def _get_feature_importances(
        self,
    ) -> List[Tuple[str, float]]:
        """Get feature importances from best available model."""
        if (
            self._deep_model is not None
            and hasattr(self._deep_model, "feature_importances_")
        ):
            importances = self._deep_model.feature_importances_
            return sorted(
                zip(self._micro_features, importances),
                key=lambda x: x[1],
                reverse=True,
            )

        # Fallback: Bayesian table analysis
        importances = []
        for feat_name in self._micro_features:
            table = self._bayesian_tables.get(feat_name, {})
            if not table:
                importances.append((feat_name, 0.0))
                continue

            # Importance = variance of acceptance rates across buckets
            rates = []
            for bucket, (rej, acc) in table.items():
                total = rej + acc
                if total >= 2:
                    rates.append(acc / total)

            if len(rates) >= 2:
                importance = float(np.std(rates))
            else:
                importance = 0.0

            importances.append((feat_name, importance))

        return sorted(importances, key=lambda x: x[1], reverse=True)

    def _get_feature_direction(self, feat_name: str) -> str:
        """
        Determine if higher value = more likely to accept
        or reject.
        """
        if self._deep_model is not None and _SKLEARN_AVAILABLE:
            # Use partial dependence direction
            try:
                feat_idx = self._micro_features.index(feat_name)
                X = np.array([
                    [x.get(f, 0.0) for f in self._micro_features]
                    for x in self._X
                ])
                # Simple: compare prediction at low vs high value
                low_x = X.mean(axis=0).copy()
                high_x = X.mean(axis=0).copy()
                low_x[feat_idx] = np.percentile(X[:, feat_idx], 25)
                high_x[feat_idx] = np.percentile(X[:, feat_idx], 75)

                low_pred = self._deep_model.predict_proba(
                    low_x.reshape(1, -1)
                )[0][1]
                high_pred = self._deep_model.predict_proba(
                    high_x.reshape(1, -1)
                )[0][1]

                return "positive" if high_pred > low_pred else "negative"
            except Exception:
                pass

        # Fallback: Bayesian analysis
        table = self._bayesian_tables.get(feat_name, {})
        if not table:
            return "positive"

        low_buckets = [0, 1]
        high_buckets = [3, 4]

        low_accept_rate = 0.5
        high_accept_rate = 0.5

        low_data = [table.get(b, [0, 0]) for b in low_buckets]
        high_data = [table.get(b, [0, 0]) for b in high_buckets]

        low_total = sum(r + a for r, a in low_data)
        high_total = sum(r + a for r, a in high_data)

        if low_total >= 2:
            low_accept_rate = sum(a for _, a in low_data) / low_total
        if high_total >= 2:
            high_accept_rate = sum(a for _, a in high_data) / high_total

        return "positive" if high_accept_rate > low_accept_rate else "negative"

    def _explain_positive(
        self, features: Dict[str, float]
    ) -> List[str]:
        """Why this setup looks GOOD."""
        reasons = []
        importances = self._get_feature_importances()

        for feat_name, imp in importances[:3]:
            if imp < 0.02:
                continue
            val = features.get(feat_name)
            if val is None:
                continue
            direction = self._get_feature_direction(feat_name)
            description = COMPRESSION_MICRO_FEATURES.get(
                feat_name, feat_name
            )

            bucket = self._to_bucket(feat_name, float(val))
            table = self._bayesian_tables.get(feat_name, {})
            cell = table.get(bucket, [0, 0])
            total = cell[0] + cell[1]
            if total >= 2:
                accept_rate = cell[1] / total
                if accept_rate > 0.6:
                    reasons.append(
                        f"{feat_name}={val:.2f}: you accept "
                        f"{accept_rate:.0%} of setups in this range"
                    )

        return reasons

    def _explain_negative(
        self, features: Dict[str, float]
    ) -> List[str]:
        """Why this setup looks BAD."""
        reasons = []
        importances = self._get_feature_importances()

        for feat_name, imp in importances[:3]:
            if imp < 0.02:
                continue
            val = features.get(feat_name)
            if val is None:
                continue

            bucket = self._to_bucket(feat_name, float(val))
            table = self._bayesian_tables.get(feat_name, {})
            cell = table.get(bucket, [0, 0])
            total = cell[0] + cell[1]
            if total >= 2:
                reject_rate = cell[0] / total
                if reject_rate > 0.6:
                    reasons.append(
                        f"{feat_name}={val:.2f}: you reject "
                        f"{reject_rate:.0%} of setups in this range"
                    )

        return reasons

    def _generate_plain_explanation(
        self,
        positive: List[Dict],
        negative: List[Dict],
    ) -> str:
        """
        Generate a plain English explanation of what the
        user's judgment means.
        
        THIS IS THE MAGIC OUTPUT.
        """
        parts = [
            f"Based on your {self._state.total_samples} decisions, "
            f"here's what '{self._name}' means to you:\n"
        ]

        if positive:
            parts.append("You LIKE setups where:")
            for p in positive[:3]:
                parts.append(f"  • {p['meaning']}")

        if negative:
            parts.append("\nYou DISLIKE setups where:")
            for n in negative[:3]:
                parts.append(f"  • {n['meaning']}")

        accuracy = self._state.model_accuracy
        if accuracy > 0.7:
            parts.append(
                f"\nModel accuracy: {accuracy:.0%} — we understand "
                f"your judgment well."
            )
        elif accuracy > 0.55:
            parts.append(
                f"\nModel accuracy: {accuracy:.0%} — decent but "
                f"improving with more decisions."
            )
        else:
            parts.append(
                f"\nModel accuracy: {accuracy:.0%} — still learning. "
                f"More decisions will help."
            )

        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════
#  JUDGMENT MANAGER
# ══════════════════════════════════════════════════════════════════


class JudgmentManager:
    """
    Manages multiple judgment learners for a strategy.
    
    Each uncertainty factor that has can_learn=True
    gets its own JudgmentLearner.
    
    Integrates with:
    - UncertaintyLayer (reduces penalty as judgment is learned)
    - SoftScoringLayer (replaces placeholder scores with model)
    - CalibrationTracker (feeds decisions into calibration)
    """

    def __init__(
        self,
        strategy_id: str,
        uncertainty_factors: List[Dict[str, Any]],
        micro_feature_computer: MicroFeatureComputer,
    ):
        self._strategy_id = strategy_id
        self._computer = micro_feature_computer
        self._learners: Dict[str, JudgmentLearner] = {}

        # Create a learner for each learnable uncertainty
        for factor in uncertainty_factors:
            if factor.get("can_learn", False):
                name = factor["name"]
                # Determine which micro-features to use
                micro_feats = self._get_micro_features_for(name)
                self._learners[name] = JudgmentLearner(
                    judgment_name=name,
                    micro_features=micro_feats,
                )

    def record_decision(
        self,
        judgment_name: str,
        candles: List[Dict],
        compression_start: int,
        compression_end: int,
        user_accepted: bool,
        context_candles: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Record a user's accept/reject decision for a specific
        judgment factor.
        """
        learner = self._learners.get(judgment_name)
        if not learner:
            return {"status": "no_learner_for_" + judgment_name}

        # Compute micro-features
        micro = self._computer.compute(
            candles, compression_start, compression_end,
            context_candles,
        )

        if micro.computation_confidence < 0.3:
            return {"status": "low_confidence_features"}

        # Record
        learner.record_decision(micro.features, user_accepted)

        return {
            "status": "recorded",
            "judgment": judgment_name,
            "total_samples": learner._state.total_samples,
            "model_ready": learner._state.model_ready,
            "uncertainty_reduction": learner.get_uncertainty_reduction(),
        }

    def predict_judgment(
        self,
        judgment_name: str,
        candles: List[Dict],
        compression_start: int,
        compression_end: int,
        context_candles: Optional[List[Dict]] = None,
    ) -> Optional[JudgmentPrediction]:
        """Predict user's judgment for a setup."""
        learner = self._learners.get(judgment_name)
        if not learner or not learner._state.model_ready:
            return None

        micro = self._computer.compute(
            candles, compression_start, compression_end,
            context_candles,
        )

        if micro.computation_confidence < 0.3:
            return None

        return learner.predict(micro.features)

    def get_soft_score(
        self,
        judgment_name: str,
        micro_features: Dict[str, float],
    ) -> Optional[float]:
        """
        Get the learned soft score for this judgment.
        Returns None if model not ready yet.
        
        When this returns a value, it REPLACES the
        placeholder soft score in the SoftScoringLayer.
        """
        learner = self._learners.get(judgment_name)
        if not learner:
            return None

        if learner._state.total_samples < learner._min_samples:
            return None

        return learner.get_soft_score(micro_features)

    def get_all_uncertainty_reductions(self) -> Dict[str, float]:
        """
        Get uncertainty reduction for all judgment factors.
        Feed this into UncertaintyLayer.update_learning().
        """
        return {
            name: learner.get_uncertainty_reduction()
            for name, learner in self._learners.items()
        }

    def get_all_explanations(self) -> Dict[str, Any]:
        """Get explanations for all learned judgments."""
        return {
            name: learner.explain_judgment()
            for name, learner in self._learners.items()
        }

    def get_status(self) -> Dict[str, Any]:
        """Overall judgment learning status."""
        return {
            "strategy_id": self._strategy_id,
            "judgment_factors": {
                name: {
                    "samples": learner._state.total_samples,
                    "model_ready": learner._state.model_ready,
                    "accuracy": round(
                        learner._state.model_accuracy, 3
                    ),
                    "uncertainty_reduction": round(
                        learner.get_uncertainty_reduction(), 3
                    ),
                }
                for name, learner in self._learners.items()
            },
            "overall_ready": all(
                l._state.model_ready for l in self._learners.values()
            ),
            "overall_uncertainty_reduction": round(
                float(np.mean([
                    l.get_uncertainty_reduction()
                    for l in self._learners.values()
                ])) if self._learners else 0.0,
                3,
            ),
        }

    def total_decisions(self, judgment_name: str) -> int:
        """Decisions recorded for one judgment factor."""
        learner = self._learners.get(judgment_name)
        return learner.total_samples if learner else 0

    def total_decisions_all(self) -> int:
        """Total decisions recorded across all judgment factors."""
        return sum(l.total_samples for l in self._learners.values())

    # ── Persistence ────────────────────────────────────────────────

    def export_state(self) -> Dict[str, Dict[str, Any]]:
        """Serialize all learners' decision history for durable storage."""
        return {
            name: learner.export_decisions()
            for name, learner in self._learners.items()
            if learner.total_samples > 0
        }

    def load_state(self, state: Dict[str, Dict[str, Any]]) -> int:
        """
        Restore learners from export_state() output.
        Returns the number of judgment factors restored.
        """
        restored = 0
        for name, record in (state or {}).items():
            learner = self._learners.get(name)
            if learner is None:
                continue
            X = record.get("X") or []
            y = record.get("y") or []
            if not X or not y:
                continue
            learner.load_decisions(X, y)
            restored += 1
        return restored

    @staticmethod
    def _get_micro_features_for(judgment_name: str) -> List[str]:
        """Map judgment name to relevant micro-features."""
        # Default: use all compression micro-features
        all_features = list(COMPRESSION_MICRO_FEATURES.keys())

        # Customize per judgment type
        judgment_feature_map = {
            "compression_quality_judgment": all_features,
            "compression_cleanness": [
                "avg_wick_ratio", "wick_ratio_variance",
                "body_size_cv", "range_smoothness",
                "range_contraction_monotonicity", "has_range_gap",
                "candle_overlap_pct", "symmetry_score",
                "body_direction_entropy",
            ],
            "breakout_authenticity": [
                "volume_trend_during_compression",
                "volume_spike_count",
                "distance_to_resistance_atr",
                "compression_position_in_range",
            ],
        }

        return judgment_feature_map.get(judgment_name, all_features)