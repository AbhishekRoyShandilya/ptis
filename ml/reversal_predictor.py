"""
Live Reversal Killer Predictor
================================

Predicts P(reversal_to_loss) for an open position based on its current state.

Two modes (auto-selected):
  - ML mode (xgboost installed AND ≥500 closed trades available):
      Pooled XGBoost classifier across ALL strategies, with strategy_id as a
      categorical feature. This is data-efficient at low samples — XGBoost
      learns "for strategy X in regime Y, reversal prob is high" via
      feature interactions.

  - Rule-based mode (always available, used as cold-start fallback):
      Hand-tuned heuristics derived from the existing RRKillerDetector
      research + standard market microstructure knowledge.

Live use:
  AVT calls `predict(position)` every scan cycle. If P > threshold (default 0.7),
  emit COGNITIVE_ALERT "Book partial — N% reversal probability at this R given
  regime+RSI+time".

Real-money safety:
  - Predictor only ALERTS, never auto-exits trades. Trader decides.
  - Returns (probability, method) tuple so caller can show "based on ML"
    vs "based on rules" — transparency over confidence.
  - Probabilities are NOT calibrated to true frequencies until ≥500 trades —
    treat low-sample outputs as ranking, not absolute probability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── XGBoost availability check (mirrors ml/regime_classifier.py pattern) ──

try:
    import xgboost as _xgb  # type: ignore
    _XGB_AVAILABLE = True
except ImportError:
    _xgb = None
    _XGB_AVAILABLE = False
    logger.info(
        "xgboost not installed — ReversalPredictor will use rule-based mode. "
        "Install with: pip install xgboost scikit-learn"
    )

try:
    from sklearn.preprocessing import LabelEncoder  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import roc_auc_score  # type: ignore
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────

# Minimum closed trades required to switch from rule-based to ML mode.
# Chosen for sample-efficiency: 500 pooled trades across ~5 strategies = ~100/strategy
# which is enough for XGBoost to learn meaningful interactions when strategy_id
# is included as a categorical feature.
MIN_TRADES_FOR_ML: int = 500

# Default alert threshold — emit COGNITIVE_ALERT when P > this
DEFAULT_ALERT_THRESHOLD: float = 0.70

# Reversal definition: a trade "reversed to loss" if final R < REVERSAL_R_THRESHOLD
# AFTER it had reached the MFE level we're scoring at. Matches Phase 2 cliff.
REVERSAL_R_THRESHOLD: float = 0.5

# Feature columns used by both rule-based and ML modes
FEATURE_COLUMNS: List[str] = [
    "regime",            # categorical
    "htf_trend_15m",     # categorical (Phase 1)
    "htf_trend_1h",      # categorical (Phase 1)
    "time_slot",         # categorical: morning/midday/afternoon/close
    "rsi",               # 0-100
    "volume_ratio",      # current bar vol / 20-bar avg
    "current_r",         # R-multiple right now
    "mfe_so_far",        # max favorable R seen so far
    "hold_minutes",      # how long held
    "atr_percentile",    # 0-100
    "strategy_id",       # categorical — pooled model uses this
]


# ── Output dataclass ───────────────────────────────────────────────

@dataclass
class ReversalPrediction:
    """Output of a single live prediction."""
    probability: float          # 0.0 - 1.0
    method: str                 # "ml_xgboost" | "rule_based"
    confidence: str             # "high" | "moderate" | "low"
    reason: str                 # human-readable explanation
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def is_alert(self, threshold: float = DEFAULT_ALERT_THRESHOLD) -> bool:
        return self.probability >= threshold


# ── Rule-based predictor (always available) ────────────────────────

def _rule_based_probability(features: Dict[str, Any]) -> Tuple[float, str]:
    """
    Hand-tuned probability estimate from observable trade state.

    Heuristics derived from RRKillerDetector research + market structure:
    - Trending regimes: low reversal probability (trend persists)
    - Ranging regimes: HIGH reversal once MFE > 1R (mean reversion)
    - Volatile regimes: highest reversal risk (whipsaw)
    - RSI extremes + meaningful R = late-stage move (added risk)
    - Late-day (afternoon/close) + extended MFE = profit-taking risk

    Returns (probability, reason_text).
    """
    regime      = (features.get("regime") or "").lower()
    htf_1h      = (features.get("htf_trend_1h") or "").lower()
    current_r   = float(features.get("current_r", 0.0) or 0.0)
    mfe         = float(features.get("mfe_so_far", current_r) or current_r)
    rsi         = float(features.get("rsi", 50.0) or 50.0)
    time_slot   = (features.get("time_slot") or "").lower()

    reasons: List[str] = []

    # ── Base probability: regime × MFE ──
    if regime in ("trending_up", "trending_down", "trending_aligned", "trending_bullish", "trending_bearish"):
        if mfe < 2.0:
            base = 0.12
        elif mfe < 3.0:
            base = 0.20
        else:
            base = 0.30  # late-stage trends still reverse
        reasons.append(f"Trending regime — trend persists more often than not (base {int(base*100)}%)")

    elif regime in ("ranging", "range_breakout_attempt", "mean_reverting"):
        if mfe < 1.0:
            base = 0.25
            reasons.append(f"Ranging — moderate risk at low MFE (base {int(base*100)}%)")
        elif mfe < 1.5:
            base = 0.50
            reasons.append(f"Ranging + MFE {mfe:.1f}R — range trades fade past 1R (base {int(base*100)}%)")
        else:
            base = 0.70
            reasons.append(f"Ranging + MFE {mfe:.1f}R — historically high reversal past 1.5R (base {int(base*100)}%)")

    elif regime in ("volatile", "volatile_top", "volatile_expansion", "high_volatility"):
        if mfe < 1.0:
            base = 0.35
            reasons.append(f"Volatile regime — whipsaw risk even at low MFE (base {int(base*100)}%)")
        elif mfe < 2.0:
            base = 0.60
            reasons.append(f"Volatile + MFE {mfe:.1f}R — give-back probability climbing (base {int(base*100)}%)")
        else:
            base = 0.75
            reasons.append(f"Volatile + MFE {mfe:.1f}R — historically high give-back (base {int(base*100)}%)")

    else:
        base = 0.35
        reasons.append(f"Unknown regime — using neutral base ({int(base*100)}%)")

    # ── HTF disagreement adds risk (LTF momentum vs HTF chop) ──
    if htf_1h in ("ranging", "volatile") and regime not in ("ranging", "volatile"):
        base += 0.10
        reasons.append("HTF (1h) is choppy — adds reversal risk on LTF moves")

    # ── RSI extreme + meaningful R = late-stage ──
    if mfe >= 1.0:
        if rsi > 75:
            base += 0.12
            reasons.append(f"RSI {rsi:.0f} overbought at +{mfe:.1f}R — overextended")
        elif rsi < 25:
            base += 0.12
            reasons.append(f"RSI {rsi:.0f} oversold at +{mfe:.1f}R — short side overextended")

    # ── Late-day + extended MFE = profit-taking window ──
    if time_slot in ("afternoon", "close", "power_hour") and mfe >= 1.5:
        base += 0.08
        reasons.append("Late-day + extended MFE — institutional profit-taking window")

    # ── Clip to valid probability range ──
    probability = max(0.0, min(1.0, base))
    reason_text = "; ".join(reasons) if reasons else "No specific risk factors identified"
    return probability, reason_text


# ── Feature extraction from a position ─────────────────────────────

def extract_features_from_position(
    position: Any,
    market_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract feature dict from an AVT position object + optional live market state.

    Position is expected to have (loosely):
      - strategy_id, regime_at_entry, entry_time
      - feature_vector_at_entry (dict from AVT engine.py)

    Market state (provided fresh per call):
      - regime, rsi, volume_ratio, current_r, mfe_so_far, atr_percentile,
        htf_trend_15m, htf_trend_1h
    """
    features: Dict[str, Any] = {}
    market_state = market_state or {}

    # Strategy
    features["strategy_id"] = getattr(position, "strategy_id", "") or ""

    # Pull live market state first (most recent / authoritative)
    for key in ("regime", "htf_trend_15m", "htf_trend_1h", "rsi", "volume_ratio",
                "current_r", "mfe_so_far", "atr_percentile"):
        if key in market_state:
            features[key] = market_state[key]

    # Fall back to entry-time snapshot for missing live values
    entry_features = getattr(position, "feature_vector_at_entry", None) or {}
    for key, val in entry_features.items():
        features.setdefault(key, val)

    # Time slot from current time (IST-aware would be better; keep simple)
    now = datetime.now()
    hour = now.hour
    if hour < 11:
        features["time_slot"] = "morning"
    elif hour < 13:
        features["time_slot"] = "midday"
    elif hour < 14.5:
        features["time_slot"] = "afternoon"
    else:
        features["time_slot"] = "close"

    # Hold minutes
    entry_time = getattr(position, "entry_time", None)
    if entry_time and isinstance(entry_time, datetime):
        # Strip tz if needed for subtraction
        try:
            delta = datetime.now(entry_time.tzinfo) - entry_time
        except Exception:
            delta = datetime.now() - entry_time.replace(tzinfo=None)
        features["hold_minutes"] = int(delta.total_seconds() / 60)
    else:
        features.setdefault("hold_minutes", 0)

    # Defaults for any missing numeric features
    for key in ("rsi", "volume_ratio", "current_r", "mfe_so_far", "atr_percentile"):
        features.setdefault(key, 0.0 if key != "rsi" else 50.0)

    # Defaults for missing categorical features
    for key in ("regime", "htf_trend_15m", "htf_trend_1h"):
        features.setdefault(key, "")

    return features


# ── Main predictor class ───────────────────────────────────────────

class ReversalPredictor:
    """
    Live reversal probability predictor.

    Auto-selects mode based on (a) xgboost availability and (b) training data size.
    Designed to be called from AVT's `_evaluate_exit()` for partial-book alerts.

    Cold-start: rule-based only.
    Steady-state (≥500 trades + xgboost installed): pooled XGBoost.
    """

    def __init__(
        self,
        trade_loader: Optional[Callable[[], List[Dict[str, Any]]]] = None,
        alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
        min_trades_for_ml: int = MIN_TRADES_FOR_ML,
    ):
        """
        Args:
            trade_loader: callable returning list of training records.
                Each record must have:
                  - all FEATURE_COLUMNS (excluding 'strategy_id' if not pooling)
                  - 'did_reverse_to_loss': 0 or 1 (target label)
                Called when maybe_train() is invoked.
            alert_threshold: P >= this → is_alert returns True
            min_trades_for_ml: switch from rule to ML at this trade count
        """
        self._trade_loader = trade_loader
        self.alert_threshold = alert_threshold
        self.min_trades_for_ml = min_trades_for_ml

        # ML state (None until trained)
        self._xgb_model: Optional[Any] = None
        self._label_encoders: Dict[str, Any] = {}  # per categorical feature
        self._feature_order: List[str] = list(FEATURE_COLUMNS)
        self._last_train_count: int = 0
        self._last_train_auc: Optional[float] = None
        self._last_feature_importance: Dict[str, float] = {}

    @property
    def ml_available(self) -> bool:
        return _XGB_AVAILABLE and _SKLEARN_AVAILABLE

    @property
    def is_ml_active(self) -> bool:
        return self._xgb_model is not None

    @property
    def stats(self) -> Dict[str, Any]:
        """Diagnostics for logs / dashboard."""
        return {
            "ml_available": self.ml_available,
            "is_ml_active": self.is_ml_active,
            "last_train_count": self._last_train_count,
            "last_train_auc": self._last_train_auc,
            "alert_threshold": self.alert_threshold,
            "feature_importance": self._last_feature_importance,
        }

    # ── Live prediction ──

    def predict(
        self,
        position: Any,
        market_state: Optional[Dict[str, Any]] = None,
    ) -> ReversalPrediction:
        """Main entry point — called by AVT _evaluate_exit() per scan cycle."""
        features = extract_features_from_position(position, market_state)
        return self.predict_from_features(features)

    def predict_from_features(self, features: Dict[str, Any]) -> ReversalPrediction:
        """
        Predict from a feature dict (used internally + for testing).
        Auto-selects ML if model is loaded, else rule-based.
        """
        if self._xgb_model is not None:
            try:
                prob = self._xgb_predict(features)
                # Confidence depends on training sample size + recency
                conf = "high" if self._last_train_count >= 1000 else "moderate"
                top_features = self._top_contributing_features(features)
                auc_str = f"{self._last_train_auc:.2f}" if self._last_train_auc else "n/a"
                reason = (
                    f"ML model ({self._last_train_count} trades, AUC={auc_str}). "
                    f"Top features: {top_features}"
                )
                return ReversalPrediction(
                    probability=prob,
                    method="ml_xgboost",
                    confidence=conf,
                    reason=reason,
                    feature_importance=dict(self._last_feature_importance),
                )
            except Exception as e:
                logger.warning("XGBoost predict failed (%s) — falling back to rule-based", e)

        # Rule-based path
        prob, reason = _rule_based_probability(features)
        return ReversalPrediction(
            probability=prob,
            method="rule_based",
            confidence="moderate",  # rule-based is always 'moderate' confidence
            reason=reason,
        )

    # ── Training (manual or scheduled) ──

    def maybe_train(self) -> Dict[str, Any]:
        """
        Try to train the ML model. Returns status dict.

        Triggered by:
          - Scheduler (nightly during Phase 4 loop)
          - Manual API call
          - First call after enough trades accumulated

        Skips if:
          - xgboost/sklearn missing
          - trade_loader not configured
          - <min_trades_for_ml available
        """
        status: Dict[str, Any] = {
            "trained": False,
            "method": None,
            "n_trades": 0,
            "auc": None,
            "reason": "",
        }

        if not self.ml_available:
            status["reason"] = "xgboost/sklearn not installed — staying on rule-based"
            return status

        if self._trade_loader is None:
            status["reason"] = "no trade_loader configured"
            return status

        try:
            trades = self._trade_loader()
        except Exception as e:
            status["reason"] = f"trade_loader failed: {e}"
            return status

        n = len(trades)
        status["n_trades"] = n
        if n < self.min_trades_for_ml:
            status["reason"] = (
                f"only {n} trades, need {self.min_trades_for_ml} — staying on rule-based"
            )
            return status

        try:
            auc = self._train_xgb(trades)
            self._last_train_count = n
            self._last_train_auc = auc
            status["trained"] = True
            status["method"] = "xgboost_pooled"
            status["auc"] = auc
            status["reason"] = f"trained on {n} trades, holdout AUC={auc:.3f}"
            logger.info(
                "ReversalPredictor trained (pooled XGBoost): n=%d, AUC=%.3f", n, auc,
            )
        except Exception as e:
            status["reason"] = f"training failed: {e}"
            logger.error("ReversalPredictor training failed: %s", e, exc_info=True)

        return status

    # ── XGBoost internals (only used if ml_available) ──

    def _train_xgb(self, trades: List[Dict[str, Any]]) -> float:
        """Train pooled XGBoost on closed-trade data. Returns holdout AUC."""
        if not self.ml_available:
            raise RuntimeError("xgboost/sklearn not available")

        import numpy as np

        # Build X, y
        X_rows: List[List[float]] = []
        y_rows: List[int] = []
        for t in trades:
            try:
                row = self._encode_row(t, fit=True)
                label = int(t.get("did_reverse_to_loss", 0))
                X_rows.append(row)
                y_rows.append(label)
            except Exception as e:
                logger.debug("Skipping trade due to encoding error: %s", e)
                continue

        if len(X_rows) < self.min_trades_for_ml:
            raise ValueError(f"too few valid rows after encoding ({len(X_rows)})")

        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y_rows, dtype=int)

        # Stratified split (preserve class balance)
        if len(set(y.tolist())) < 2:
            raise ValueError("training data has only one class — need both reversed and not-reversed examples")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )

        model = _xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
        )
        model.fit(X_train, y_train)

        # Holdout AUC
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))

        # Feature importance — use 'gain' (most informative)
        try:
            importance = model.get_booster().get_score(importance_type="gain")
            # Map back from f0, f1, ... to feature names
            self._last_feature_importance = {
                self._feature_order[int(k[1:])]: float(v)
                for k, v in importance.items()
                if k.startswith("f") and int(k[1:]) < len(self._feature_order)
            }
        except Exception:
            self._last_feature_importance = {}

        self._xgb_model = model
        return auc

    def _xgb_predict(self, features: Dict[str, Any]) -> float:
        """Run a single live prediction through the trained XGB model."""
        import numpy as np
        row = self._encode_row(features, fit=False)
        X = np.asarray([row], dtype=float)
        proba = self._xgb_model.predict_proba(X)[0, 1]
        return float(proba)

    def _encode_row(self, record: Dict[str, Any], fit: bool) -> List[float]:
        """
        Encode a record (dict) into a feature row matching self._feature_order.
        Categorical features get LabelEncoded; missing/unknown values map to -1.
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("sklearn not available")

        row: List[float] = []
        for col in self._feature_order:
            val = record.get(col, None)
            if col in ("regime", "htf_trend_15m", "htf_trend_1h", "time_slot", "strategy_id"):
                # Categorical — encode
                str_val = str(val) if val is not None else ""
                if fit:
                    enc = self._label_encoders.get(col)
                    if enc is None:
                        enc = LabelEncoder()
                        # Fit will be handled in batch below — defer for now
                        self._label_encoders[col] = enc
                    # During fit, just collect; we'll do the actual fit_transform later
                    # For simplicity, do incremental: maintain a dict of seen values
                    # (Cleaner than re-fitting per row — see batch path below)
                    classes = list(getattr(enc, "classes_", [])) if hasattr(enc, "classes_") else []
                    if str_val not in classes:
                        classes.append(str_val)
                        enc.classes_ = __import__("numpy").asarray(classes)
                    row.append(float(list(enc.classes_).index(str_val)))
                else:
                    enc = self._label_encoders.get(col)
                    if enc is None or not hasattr(enc, "classes_"):
                        row.append(-1.0)
                    else:
                        classes = list(enc.classes_)
                        row.append(float(classes.index(str_val)) if str_val in classes else -1.0)
            else:
                # Numeric
                try:
                    row.append(float(val) if val is not None else 0.0)
                except (TypeError, ValueError):
                    row.append(0.0)
        return row

    def _top_contributing_features(self, features: Dict[str, Any], n: int = 3) -> str:
        """Return string with top N features by importance, given live feature values."""
        if not self._last_feature_importance:
            return "(unknown)"
        # Sort by importance, take top N
        sorted_imp = sorted(
            self._last_feature_importance.items(), key=lambda kv: kv[1], reverse=True,
        )[:n]
        parts = []
        for name, _ in sorted_imp:
            val = features.get(name, "?")
            parts.append(f"{name}={val}")
        return ", ".join(parts)


# ── Module-level singleton ─────────────────────────────────────────

_singleton: Optional[ReversalPredictor] = None


def get_reversal_predictor(
    trade_loader: Optional[Callable[[], List[Dict[str, Any]]]] = None,
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
) -> ReversalPredictor:
    """Get the module singleton. First call sets up the trade loader."""
    global _singleton
    if _singleton is None:
        _singleton = ReversalPredictor(
            trade_loader=trade_loader,
            alert_threshold=alert_threshold,
        )
    elif trade_loader is not None and _singleton._trade_loader is None:
        # Late-bind the loader if not set initially
        _singleton._trade_loader = trade_loader
    return _singleton
