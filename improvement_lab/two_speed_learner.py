# improvement_lab/two_speed_learner.py
"""
Two-Speed Learning Engine
==========================

FAST LAYER (every trade, milliseconds):
  - Bayesian conditional probability tables
    P(reversal | regime=ranging, rsi_bucket=high)
    Updated with simple count increment
  - Online logistic regression (SGD with partial_fit)
  - Exponentially weighted running stats per condition slice

DEEP LAYER (periodic batch retrain):
  - Gradient Boosting (XGBoost/LightGBM/sklearn GBM)
  - Random Forest ensemble
  - Retrained when 20+ new trades since last training
  - SHAP for feature importance

Prediction: weighted ensemble of both layers.
Deep model gets more weight as it accumulates more training data.
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from improvement_lab.models import BayesianCell, TwoSpeedState

logger = logging.getLogger(__name__)

# Bucketize continuous features for Bayesian tables
BUCKET_DEFINITIONS = {
    "rsi_at_peak": [(0, 30, "oversold"), (30, 50, "low"), (50, 70, "mid"), (70, 100, "overbought")],
    "rsi_at_entry": [(0, 30, "oversold"), (30, 50, "low"), (50, 70, "mid"), (70, 100, "overbought")],
    "volume_at_peak_vs_breakout": [(0, 0.5, "fading"), (0.5, 1.0, "declining"), (1.0, 2.0, "steady"), (2.0, 999, "surging")],
    "extension_from_vwap_atr": [(-999, -2, "far_below"), (-2, -0.5, "below"), (-0.5, 0.5, "near"), (0.5, 2, "above"), (2, 999, "far_above")],
    "candle_exhaustion_score": [(0, 0.3, "low"), (0.3, 0.6, "moderate"), (0.6, 1.0, "high")],
    "hour_ist": [(9, 10.5, "opening"), (10.5, 12, "morning"), (12, 14, "lunch"), (14, 15.5, "afternoon")],
    "atr_percentile": [(0, 0.25, "low"), (0.25, 0.5, "below_avg"), (0.5, 0.75, "above_avg"), (0.75, 1.0, "high")],
    "conviction_score": [(1, 4, "low"), (4, 7, "medium"), (7, 10, "high")],
    "streak_at_entry": [(-99, -2, "losing_streak"), (-2, 0, "slight_loss"), (0, 2, "slight_win"), (2, 99, "winning_streak")],
}

# Categorical features used directly (no bucketing needed)
CATEGORICAL_FEATURES = [
    "regime_at_peak", "market_regime", "session_type",
    "emotion_pre", "direction",
]

# Features used for Bayesian tables (key conditioning axes)
BAYESIAN_AXES = [
    "regime_bucket", "rsi_bucket", "volume_bucket",
    "exhaustion_bucket", "time_bucket", "session_type",
    "direction", "conviction_bucket", "streak_bucket",
]


def _bucketize(value: float, buckets: List[Tuple[float, float, str]]) -> str:
    """Assign a continuous value to a named bucket."""
    for low, high, label in buckets:
        if low <= value < high:
            return label
    return "unknown"


def _feature_to_bucket(feature_name: str, value: float) -> str:
    """Convert a feature value to its bucket label."""
    if feature_name in BUCKET_DEFINITIONS:
        return _bucketize(value, BUCKET_DEFINITIONS[feature_name])
    return str(value)


class BayesianConditionalTables:
    """
    Fast Bayesian conditional probability tables.

    For each conditioning axis (regime, RSI bucket, volume bucket, etc.),
    maintains a table of P(outcome | axis=value).

    Update: O(1) per trade — just increment counts.
    Predict: Average across all axis predictions.
    """

    def __init__(self):
        # axis_name -> {bucket_value -> BayesianCell}
        self.tables: Dict[str, Dict[str, BayesianCell]] = defaultdict(
            lambda: defaultdict(BayesianCell)
        )
        self.total = BayesianCell()  # unconditional baseline

    def update(self, buckets: Dict[str, str], outcome: bool) -> None:
        """Update all tables with a new observation."""
        self.total.update(outcome)
        for axis, bucket in buckets.items():
            self.tables[axis][bucket].update(outcome)

    def predict(self, buckets: Dict[str, str]) -> float:
        """
        Predict probability from all available axes.
        Returns weighted average of axis-specific predictions.
        """
        predictions = []
        weights = []

        for axis, bucket in buckets.items():
            cell = self.tables.get(axis, {}).get(bucket)
            if cell and cell.count >= 3:  # minimum observations
                predictions.append(cell.rate)
                # Weight by count — more observations = more reliable
                weights.append(min(cell.count, 50))

        if not predictions:
            return self.total.rate if self.total.count > 0 else 0.5

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        return float(np.dot(predictions, weights))

    def get_strong_signals(self, threshold: float = 0.7, min_count: int = 5) -> List[Dict]:
        """Find conditions where outcome probability exceeds threshold."""
        signals = []
        for axis, buckets in self.tables.items():
            for bucket, cell in buckets.items():
                if cell.count >= min_count and cell.rate >= threshold:
                    signals.append({
                        "axis": axis,
                        "bucket": bucket,
                        "rate": round(cell.rate, 3),
                        "count": cell.count,
                    })
        signals.sort(key=lambda s: s["rate"], reverse=True)
        return signals

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            "total": {"count": self.total.count, "successes": self.total.successes},
            "tables": {
                axis: {
                    bucket: {"count": cell.count, "successes": cell.successes}
                    for bucket, cell in buckets.items()
                }
                for axis, buckets in self.tables.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BayesianConditionalTables":
        """Deserialize from persistence."""
        obj = cls()
        total = data.get("total", {})
        obj.total.count = total.get("count", 0)
        obj.total.successes = total.get("successes", 0)
        for axis, buckets in data.get("tables", {}).items():
            for bucket, cell_data in buckets.items():
                cell = BayesianCell()
                cell.count = cell_data.get("count", 0)
                cell.successes = cell_data.get("successes", 0)
                obj.tables[axis][bucket] = cell
        return obj


class TwoSpeedLearner:
    """
    Two-speed learning: fast (per trade) + deep (periodic batch).

    Fast layer: Bayesian tables + SGD classifier
      - Updates on every closed trade
      - Provides real-time predictions from trade 30+

    Deep layer: GBM + Random Forest
      - Retrained every 20 new trades
      - Provides non-linear interaction detection
      - SHAP explanations

    Prediction: Weighted ensemble.
    """

    def __init__(self, strategy_id: str = "", detector_type: str = "generic"):
        self.strategy_id = strategy_id
        self.detector_type = detector_type  # "rr_killer" | "early_exit" | "entry_filter"

        # ── Fast layer ──
        self.bayesian = BayesianConditionalTables()
        self._sgd_classifier = None    # sklearn SGDClassifier
        self._sgd_fitted = False
        self._fast_trade_count = 0

        # ── Deep layer ──
        self._gbm_model = None
        self._rf_model = None
        self._feature_names: List[str] = []
        self._last_deep_train_count = 0
        self._deep_retrain_threshold = 20
        self._deep_model_score = 0.0

        # Accumulated training data for deep retraining
        self._X_history: List[np.ndarray] = []
        self._y_history: List[int] = []

        # Optional callback: persist_callback(strategy_id, detector_type, model_blob_bytes)
        # Set by the caller (e.g., ContinuousLearner) to save models to DB/disk after training
        self._persist_callback: Optional[Any] = None

    # ── Fast layer updates ────────────────────────────────────────

    def update_fast(
        self,
        features: Dict[str, float],
        outcome: bool,
    ) -> None:
        """
        Called on EVERY closed trade. Updates fast layer only.
        O(1) — milliseconds.
        """
        # Bucketize features for Bayesian tables
        buckets = self._features_to_buckets(features)
        self.bayesian.update(buckets, outcome)

        # Update SGD classifier
        X = self._features_to_vector(features)
        self._update_sgd(X, outcome)

        # Accumulate for deep layer
        self._X_history.append(X)
        self._y_history.append(1 if outcome else 0)
        self._fast_trade_count += 1

    def _update_sgd(self, X: np.ndarray, outcome: bool) -> None:
        """Incremental update of SGD logistic regression."""
        try:
            from sklearn.linear_model import SGDClassifier
        except ImportError:
            return

        if self._sgd_classifier is None:
            self._sgd_classifier = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=0.001,
                warm_start=True,
                random_state=42,
            )

        try:
            self._sgd_classifier.partial_fit(
                X.reshape(1, -1),
                [1 if outcome else 0],
                classes=[0, 1],
            )
            self._sgd_fitted = True
        except Exception as e:
            logger.debug("SGD partial_fit error: %s", e)

    # ── Deep layer retraining ─────────────────────────────────────

    def should_retrain_deep(self) -> bool:
        """Check if we have enough new trades to retrain."""
        new_trades = self._fast_trade_count - self._last_deep_train_count
        return new_trades >= self._deep_retrain_threshold

    def retrain_deep(self) -> Dict[str, Any]:
        """
        Full batch retrain of GBM + RF on accumulated trade history.
        Returns training report with feature importances.
        """
        if len(self._X_history) < 50:
            return {"status": "insufficient_data", "count": len(self._X_history)}

        try:
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError:
            return {"status": "sklearn_unavailable"}

        X = np.vstack(self._X_history)
        y = np.array(self._y_history)

        # Check class balance
        min_class = min(y.sum(), (1 - y).sum())
        if min_class < 10:
            return {"status": "class_imbalance", "min_class_count": int(min_class)}

        n_samples = len(y)

        # Train GBM
        gbm = GradientBoostingClassifier(
            n_estimators=min(100, max(30, n_samples // 5)),
            max_depth=min(4, max(2, n_samples // 50)),
            min_samples_leaf=max(3, n_samples // 25),
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

        # Train RF
        rf = RandomForestClassifier(
            n_estimators=min(100, max(30, n_samples // 3)),
            max_depth=min(5, max(2, n_samples // 40)),
            min_samples_leaf=max(3, n_samples // 20),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        try:
            gbm.fit(X, y)
            rf.fit(X, y)

            # Time-series cross-validation (no lookahead bias)
            # Require ≥ 40 samples per fold to produce meaningful scores.
            # With n_splits folds, TimeSeriesSplit gives ~n/(n_splits+1) samples
            # in the first training fold — ensure that is ≥ 40.
            from sklearn.model_selection import TimeSeriesSplit
            MIN_FOLD_TRAIN = 40
            max_splits = max(1, (len(X) - MIN_FOLD_TRAIN) // MIN_FOLD_TRAIN)
            n_splits = min(5, max_splits)
            if n_splits >= 2 and min_class >= 5:
                tscv = TimeSeriesSplit(n_splits=n_splits)
                gbm_score = float(np.mean(cross_val_score(gbm, X, y, cv=tscv)))
                rf_score = float(np.mean(cross_val_score(rf, X, y, cv=tscv)))
            else:
                # Not enough data for meaningful CV — use single temporal holdout
                split = int(len(X) * 0.8)
                if split >= MIN_FOLD_TRAIN and len(X) - split >= 10:
                    gbm_score = float(GradientBoostingClassifier(
                        n_estimators=gbm.n_estimators, max_depth=gbm.max_depth,
                        min_samples_leaf=gbm.min_samples_leaf, learning_rate=gbm.learning_rate,
                        subsample=gbm.subsample, random_state=42,
                    ).fit(X[:split], y[:split]).score(X[split:], y[split:]))
                    rf_score = float(RandomForestClassifier(
                        n_estimators=rf.n_estimators, max_depth=rf.max_depth,
                        min_samples_leaf=rf.min_samples_leaf, class_weight="balanced",
                        random_state=42, n_jobs=-1,
                    ).fit(X[:split], y[:split]).score(X[split:], y[split:]))
                else:
                    gbm_score = float(gbm.score(X, y))
                    rf_score = float(rf.score(X, y))

            self._gbm_model = gbm
            self._rf_model = rf
            self._last_deep_train_count = self._fast_trade_count
            self._deep_model_score = max(gbm_score, rf_score)

            # Persist deep models immediately after training
            model_blob = self.dump_deep_models()
            if model_blob is not None and self._persist_callback is not None:
                try:
                    self._persist_callback(self.strategy_id, self.detector_type, model_blob)
                except Exception as _pe:
                    logger.warning("Model persistence callback failed: %s", _pe)

            report = {
                "status": "trained",
                "n_samples": n_samples,
                "gbm_score": round(gbm_score, 3),
                "rf_score": round(rf_score, 3),
                "gbm_importances": self._get_importances(gbm),
                "rf_importances": self._get_importances(rf),
                "model_persisted": model_blob is not None,
            }

            logger.info(
                "Deep retrain complete for %s/%s: GBM=%.3f RF=%.3f on %d samples",
                self.strategy_id, self.detector_type,
                gbm_score, rf_score, n_samples,
            )
            return report

        except Exception as e:
            logger.warning("Deep retrain failed: %s", e)
            return {"status": "training_failed", "error": str(e)}

    def _get_importances(self, model: Any) -> Dict[str, float]:
        """Get feature importances as {name: importance}."""
        importances = getattr(model, "feature_importances_", None)
        if importances is None or not self._feature_names:
            return {}
        return {
            self._feature_names[i]: round(float(importances[i]), 4)
            for i in range(min(len(self._feature_names), len(importances)))
        }

    # ── Prediction (ensemble) ─────────────────────────────────────

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Ensemble prediction from both layers.
        Fast layer always available. Deep layer if trained.
        """
        buckets = self._features_to_buckets(features)
        fast_pred = self.bayesian.predict(buckets)

        online_pred = None
        if self._sgd_fitted and self._sgd_classifier is not None:
            try:
                X = self._features_to_vector(features).reshape(1, -1)
                online_pred = float(self._sgd_classifier.predict_proba(X)[0, 1])
            except Exception:
                pass

        deep_pred = None
        if self._gbm_model is not None:
            try:
                X = self._features_to_vector(features).reshape(1, -1)
                gbm_pred = float(self._gbm_model.predict_proba(X)[0, 1])
                rf_pred = float(self._rf_model.predict_proba(X)[0, 1]) if self._rf_model else gbm_pred
                deep_pred = (gbm_pred + rf_pred) / 2
            except Exception:
                pass

        # Weighted ensemble
        if deep_pred is not None:
            # Deep model weight increases with training data
            deep_weight = min(0.5, self._last_deep_train_count / 200)
            fast_weight = 1.0 - deep_weight

            if online_pred is not None:
                fast_combined = (fast_pred + online_pred) / 2
            else:
                fast_combined = fast_pred

            final = fast_combined * fast_weight + deep_pred * deep_weight
        elif online_pred is not None:
            final = (fast_pred + online_pred) / 2
        else:
            final = fast_pred

        return {
            "probability": round(final, 3),
            "fast_layer": round(fast_pred, 3),
            "online_layer": round(online_pred, 3) if online_pred is not None else None,
            "deep_layer": round(deep_pred, 3) if deep_pred is not None else None,
            "confidence": self._compute_prediction_confidence(fast_pred, online_pred, deep_pred),
            "trade_count": self._fast_trade_count,
            "deep_trained": self._gbm_model is not None,
        }

    def _compute_prediction_confidence(
        self,
        fast: float,
        online: Optional[float],
        deep: Optional[float],
    ) -> float:
        """
        Confidence based on:
        - Agreement between layers
        - Amount of training data
        - Distance from 0.5 (uncertainty)
        """
        preds = [fast]
        if online is not None:
            preds.append(online)
        if deep is not None:
            preds.append(deep)

        # Agreement: low std = high agreement
        agreement = 1.0 - min(float(np.std(preds)) * 3, 1.0) if len(preds) > 1 else 0.5

        # Data factor: more trades = more confidence
        data_factor = min(1.0, self._fast_trade_count / 100)

        # Signal strength: distance from 0.5
        avg_pred = float(np.mean(preds))
        signal = abs(avg_pred - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1

        confidence = agreement * 0.4 + data_factor * 0.3 + signal * 0.3
        return round(min(0.95, confidence), 3)

    # ── Strong signals (for rule generation) ──────────────────────

    def get_strong_signals(
        self,
        threshold: float = 0.7,
        min_count: int = 5,
    ) -> List[Dict]:
        """Get conditions where the fast layer detects strong signals."""
        return self.bayesian.get_strong_signals(threshold, min_count)

    # ── Feature conversion helpers ────────────────────────────────

    def _features_to_buckets(self, features: Dict[str, float]) -> Dict[str, str]:
        """Convert feature dict to bucket dict for Bayesian tables."""
        buckets = {}

        # Map feature names to bucket axis names
        bucket_map = {
            "rsi_at_peak": "rsi_bucket",
            "rsi_at_entry": "rsi_bucket",
            "volume_at_peak_vs_breakout": "volume_bucket",
            "candle_exhaustion_score": "exhaustion_bucket",
            "hour_ist": "time_bucket",
            "atr_percentile": "atr_bucket",
            "conviction_score": "conviction_bucket",
            "streak_at_entry": "streak_bucket",
        }

        for feat_name, axis_name in bucket_map.items():
            if feat_name in features:
                buckets[axis_name] = _feature_to_bucket(feat_name, features[feat_name])

        # Categorical features — find the active one-hot
        for prefix in ["regime", "peak_regime"]:
            for regime in ["trending_up", "trending_down", "range", "ranging",
                           "volatile", "high_volatility", "breakout", "unknown"]:
                key = f"{prefix}_{regime}"
                if features.get(key, 0) == 1.0:
                    buckets["regime_bucket"] = regime
                    break

        for session in ["opening", "morning", "afternoon", "closing"]:
            if features.get(f"session_{session}", 0) == 1.0:
                buckets["session_type"] = session
                break

        # Direction
        if "direction" in features:
            buckets["direction"] = "long" if features["direction"] > 0 else "short"

        return buckets

    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy vector for SGD/deep models."""
        if not self._feature_names:
            self._feature_names = sorted(features.keys())

        return np.array([features.get(fn, 0.0) for fn in self._feature_names])

    def set_feature_names(self, names: List[str]) -> None:
        """Set feature names (call before training or after loading)."""
        self._feature_names = names

    # ── Serialization ─────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Serialize learner state for persistence (Bayesian tables + metadata)."""
        return {
            "strategy_id": self.strategy_id,
            "detector_type": self.detector_type,
            "bayesian": self.bayesian.to_dict(),
            "feature_names": self._feature_names,
            "fast_trade_count": self._fast_trade_count,
            "last_deep_train_count": self._last_deep_train_count,
            "deep_model_score": self._deep_model_score,
            "sgd_fitted": self._sgd_fitted,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load learner state from persistence."""
        self.strategy_id = state.get("strategy_id", "")
        self.detector_type = state.get("detector_type", "generic")
        self.bayesian = BayesianConditionalTables.from_dict(state.get("bayesian", {}))
        self._feature_names = state.get("feature_names", [])
        self._fast_trade_count = state.get("fast_trade_count", 0)
        self._last_deep_train_count = state.get("last_deep_train_count", 0)
        self._deep_model_score = state.get("deep_model_score", 0.0)
        self._sgd_fitted = state.get("sgd_fitted", False)

    @staticmethod
    def _feature_schema_hash(feature_names: List[str]) -> str:
        """
        SHA-256 of the sorted feature name list.

        Stored alongside each model blob. On load, if the hash doesn't
        match the current feature extractor's output, the blob is stale
        and must be discarded — otherwise predictions run on wrong columns.
        """
        import hashlib
        key = "|".join(sorted(feature_names))
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def dump_deep_models(self) -> Optional[bytes]:
        """
        Serialize deep GBM + RF models to bytes via joblib.

        Returns None if models haven't been trained yet.
        Call after every deep retrain and store the bytes in DB or disk.
        """
        if self._gbm_model is None and self._rf_model is None:
            return None
        import io
        import joblib
        buf = io.BytesIO()
        joblib.dump(
            {
                "gbm": self._gbm_model,
                "rf": self._rf_model,
                "sgd": self._sgd_classifier,
                "feature_names": self._feature_names,
                "feature_schema_hash": self._feature_schema_hash(self._feature_names),
                "deep_model_score": self._deep_model_score,
            },
            buf,
        )
        return buf.getvalue()

    def load_deep_models(self, blob: bytes) -> bool:
        """
        Restore deep GBM + RF models from bytes previously saved by dump_deep_models().

        Returns True if loaded successfully, False if the blob was discarded
        due to feature schema mismatch (caller should trigger a fresh retrain).

        Call on startup before the first prediction to avoid cold-start.
        """
        import io
        import joblib
        data = joblib.load(io.BytesIO(blob))

        # Validate feature schema — reject stale blobs silently
        saved_hash = data.get("feature_schema_hash", "")
        current_hash = self._feature_schema_hash(self._feature_names) if self._feature_names else ""
        if saved_hash and current_hash and saved_hash != current_hash:
            logger.warning(
                "Deep model blob for %s/%s discarded: feature schema changed "
                "(saved=%s, current=%s). Fresh retrain required.",
                self.strategy_id, self.detector_type, saved_hash, current_hash,
            )
            return False

        self._gbm_model = data.get("gbm")
        self._rf_model = data.get("rf")
        if data.get("sgd") is not None:
            self._sgd_classifier = data["sgd"]
            self._sgd_fitted = True
        if data.get("feature_names"):
            self._feature_names = data["feature_names"]
        if data.get("deep_model_score") is not None:
            self._deep_model_score = data["deep_model_score"]
        logger.info(
            "Loaded deep models for %s/%s (score=%.3f, schema=%s)",
            self.strategy_id,
            self.detector_type,
            self._deep_model_score,
            saved_hash,
        )
        return True
