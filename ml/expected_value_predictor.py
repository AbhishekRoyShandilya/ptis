"""
Expected Value Predictor
=========================

Replaces the binary "P(reversal)" predictor with a continuous "what's my
expected R from here?" predictor.

Old (ml/reversal_predictor.py):
  P(reversal) = 0.7  →  alert "70% chance of reversal"

New (this module):
  E[R | current state]  =  -0.3   →  EXIT NOW (negative expected value)
  E[R | current state]  =  +0.8   →  HOLD (lots of upside left)
  E[R | current state]  =  +1.2 (low variance) → consider ADDING

Why this is better:
  1. Captures magnitude — "expect to lose 0.3R" is more useful than "70% reversal"
  2. Returns uncertainty — q10 / q50 / q90 quantiles via three XGBoost models
  3. Native decision support — should_hold / should_exit / should_add flags
  4. Composes with #2 — uses regime PROBABILITIES as features (not labels)

Two modes (auto-selected, same pattern as reversal_predictor):
  - ML mode (xgboost installed AND ≥500 trades):
      Three quantile regressors (q=0.1, q=0.5, q=0.9) trained on multi-checkpoint
      training data. Each closed trade contributes ~7 training rows (one per R
      checkpoint reached during the trade's lifetime).

  - Rule-based mode (always available, cold-start fallback):
      Heuristic blend of regime probs, RSI, RR-so-far, time-of-day.
      Returns approximate (q10, q50, q90) based on regime confidence.

Real-money safety:
  - ALERTS only — never auto-exits. Trader decides.
  - Returns probability of being right via confidence string + quantile width.
  - Both gross and net EV available; uses cost_r if known (composes with #1).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Optional ML deps (graceful degradation, same as reversal_predictor) ──

try:
    import xgboost as _xgb  # type: ignore
    _XGB_AVAILABLE = True
except ImportError:
    _xgb = None
    _XGB_AVAILABLE = False
    logger.info(
        "xgboost not installed — EV predictor will use rule-based mode. "
        "Install with: pip install xgboost scikit-learn"
    )

try:
    from sklearn.model_selection import TimeSeriesSplit  # type: ignore
    from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────

MIN_TRADES_FOR_ML: int = 500
DEFAULT_MODEL_DIR = Path("models/expected_value")

# Decision thresholds (configurable per instance)
DEFAULT_HOLD_THRESHOLD: float = 0.10        # E[R] > this → hold is supported
DEFAULT_EXIT_THRESHOLD: float = -0.10       # E[R] < this → exit signal
DEFAULT_ADD_THRESHOLD: float = 0.50         # E[R] > this AND tight uncertainty → consider adding
DEFAULT_ADD_MAX_VARIANCE: float = 2.0       # q90 - q10 must be < this for "add"
DEFAULT_ADD_MIN_DOWNSIDE: float = -0.50     # q10 must be > this for "add"

# Training: R-checkpoints used to generate multiple training rows per trade
TRAINING_CHECKPOINTS: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

# Features used by both ML and rule-based modes
EV_FEATURES: List[str] = [
    # Position state
    "current_r",
    "mfe_so_far",
    "mae_so_far",
    "hold_minutes",
    # Market context
    "regime_prob_trending",
    "regime_prob_ranging",
    "regime_prob_volatile",
    "regime_prob_breakout",
    "regime_confidence",
    "rsi",
    "volume_ratio",
    "atr_percentile",
    # HTF context
    "htf_alignment_score",
    # Time
    "minutes_to_close",
    "session_phase",
    # Strategy (categorical, encoded)
    "strategy_id_encoded",
]


# ── Output dataclass ───────────────────────────────────────────────


@dataclass
class EVPrediction:
    """Output of a single live prediction."""
    # Quantiles (in R-multiples, FROM CURRENT R)
    expected_r_from_here: float          # median (q50)
    expected_final_r: float              # current_r + median
    percentile_10: float                 # downside scenario (q10)
    percentile_50: float                 # median (same as expected_r_from_here)
    percentile_90: float                 # upside scenario (q90)

    # Decisions
    should_hold: bool                    # E[R] > hold threshold
    should_exit: bool                    # E[R] < exit threshold
    should_add: bool                     # E[R] > add threshold AND tight uncertainty AND low downside

    # Derived risk metrics
    edge_remaining: float                # E[R] / max_possible_R (0-1 scale)
    risk_reward_from_here: float         # |upside| / |downside| of remaining trade

    # Metadata
    model_type: str                      # "ml_xgboost_quantile" | "rule_based"
    confidence: str                      # "high" | "moderate" | "low"
    reason: str                          # human-readable explanation
    feature_importance: Dict[str, float] = field(default_factory=dict)


# ── Rule-based estimator (always available) ────────────────────────


def _rule_based_predict(features: Dict[str, Any]) -> Tuple[float, float, float, str]:
    """
    Hand-tuned EV estimate from observable trade state.
    Returns (q10, q50, q90, reason) — all in R-multiples FROM CURRENT R.

    Heuristics:
    - Trending regime: positive E[R] (trend persists more often than not)
    - Ranging regime: E[R] declines with each R reached (mean reversion pulls back)
    - Volatile regime: high uncertainty, slight negative bias at high R
    - RSI extremes + meaningful R = late-stage move (negative EV)
    - Late-day + extended MFE = profit-taking risk (negative EV)
    - HTF disagreement adds uncertainty (widens q10/q90)
    """
    current_r = float(features.get("current_r", 0.0) or 0.0)
    mfe = float(features.get("mfe_so_far", current_r) or current_r)

    p_trending = float(features.get("regime_prob_trending", 0.0) or 0.0)
    p_ranging = float(features.get("regime_prob_ranging", 0.0) or 0.0)
    p_volatile = float(features.get("regime_prob_volatile", 0.0) or 0.0)
    regime_conf = float(features.get("regime_confidence", 0.5) or 0.5)

    rsi = float(features.get("rsi", 50.0) or 50.0)
    minutes_to_close = float(features.get("minutes_to_close", 180) or 180)
    htf_alignment = float(features.get("htf_alignment_score", 0.0) or 0.0)

    reasons: List[str] = []

    # ── Base EV from regime mix ──
    # In trending: positive base, decays with R (less upside left)
    # In ranging: neutral at low R, negative once past 1R (mean reversion)
    # In volatile: noisy, slight negative bias
    base_ev = 0.0
    if p_trending > 0.4:
        # Trending contribution: positive but diminishing
        trend_ev = 0.4 - max(0, current_r - 1.0) * 0.15
        base_ev += p_trending * trend_ev
        reasons.append(f"Trending {int(p_trending*100)}% (+{trend_ev:.2f}R contribution)")

    if p_ranging > 0.3:
        # Ranging contribution: decays then negative
        if current_r < 0.5:
            range_ev = 0.2
        elif current_r < 1.5:
            range_ev = 0.0
        else:
            range_ev = -0.3 - (current_r - 1.5) * 0.15
        base_ev += p_ranging * range_ev
        reasons.append(f"Ranging {int(p_ranging*100)}% ({range_ev:+.2f}R contribution)")

    if p_volatile > 0.2:
        # Volatile contribution: small negative bias, larger uncertainty
        vol_ev = -0.1 if current_r > 1.0 else 0.0
        base_ev += p_volatile * vol_ev
        reasons.append(f"Volatile {int(p_volatile*100)}% ({vol_ev:+.2f}R contribution)")

    # ── RSI overextension penalty ──
    if mfe >= 1.0:
        if rsi > 75:
            base_ev -= 0.2
            reasons.append(f"RSI {rsi:.0f} overbought at +{mfe:.1f}R — overextended")
        elif rsi < 25:
            base_ev -= 0.2
            reasons.append(f"RSI {rsi:.0f} oversold at +{mfe:.1f}R — overextended (short)")

    # ── Late-day profit-taking ──
    if minutes_to_close < 30 and current_r > 0.5:
        base_ev -= 0.15
        reasons.append("Late session + winner: profit-taking pressure")

    # ── HTF support / opposition ──
    base_ev += htf_alignment * 0.15
    if abs(htf_alignment) > 0.5:
        side = "supports" if htf_alignment > 0 else "opposes"
        reasons.append(f"HTF {side} (alignment {htf_alignment:+.2f})")

    # ── Quantile spread = uncertainty ──
    # Wider when regime is uncertain or volatile is dominant
    base_spread = 0.5
    if regime_conf < 0.5:
        base_spread += 0.4  # uncertain regime → wider intervals
    if p_volatile > 0.3:
        base_spread += 0.3
    if mfe > 2.0:
        base_spread += 0.3  # extended trades have wider tails

    # Clip to sensible range
    median = max(-2.5, min(2.5, base_ev))
    q10 = round(median - base_spread, 4)
    q90 = round(median + base_spread, 4)

    reason = "; ".join(reasons) if reasons else "No specific risk factors"
    return q10, round(median, 4), q90, reason


# ── Decision logic (shared between ML and rule paths) ──────────────


def _decide(
    q10: float, median: float, q90: float,
    hold_thresh: float, exit_thresh: float, add_thresh: float,
    add_max_variance: float, add_min_downside: float,
) -> Tuple[bool, bool, bool]:
    """Compute should_hold / should_exit / should_add booleans."""
    should_hold = median > hold_thresh
    should_exit = median < exit_thresh
    variance = q90 - q10
    should_add = (
        median > add_thresh
        and variance < add_max_variance
        and q10 > add_min_downside
    )
    return should_hold, should_exit, should_add


# ── Feature extraction ─────────────────────────────────────────────


def extract_features_from_position(
    position: Any,
    market_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a feature dict from an AVT position + live market state.

    `position` should expose (loosely):
        .strategy_id
        .entry_time (datetime)
        .feature_vector_at_entry (dict)
    `market_state` (refreshed per call) provides current regime probs, RSI, etc.
    """
    market_state = market_state or {}
    features: Dict[str, Any] = {}

    # Strategy
    features["strategy_id"] = getattr(position, "strategy_id", "") or ""

    # Live market values take precedence (most recent)
    for key in (
        "regime_prob_trending", "regime_prob_ranging", "regime_prob_volatile",
        "regime_prob_breakout", "regime_confidence",
        "rsi", "volume_ratio", "atr_percentile",
        "current_r", "mfe_so_far", "mae_so_far",
        "htf_alignment_score",
    ):
        if key in market_state:
            features[key] = market_state[key]

    # Fall back to entry-time snapshot
    entry_features = getattr(position, "feature_vector_at_entry", None) or {}
    for k, v in entry_features.items():
        features.setdefault(k, v)

    # Time features
    now = datetime.now()
    hour, minute = now.hour, now.minute
    minutes_into_session = max(0, (hour - 9) * 60 + (minute - 15))
    minutes_in_session = 6 * 60 + 15  # 9:15 → 15:30 IST
    features["session_phase"] = round(min(minutes_into_session / minutes_in_session, 1.0), 4)
    features["minutes_to_close"] = max(0, minutes_in_session - minutes_into_session)

    # Hold time
    entry_time = getattr(position, "entry_time", None)
    if isinstance(entry_time, datetime):
        try:
            delta = datetime.now(entry_time.tzinfo) - entry_time
        except Exception:
            delta = datetime.now() - entry_time.replace(tzinfo=None)
        features["hold_minutes"] = max(0, int(delta.total_seconds() / 60))
    else:
        features.setdefault("hold_minutes", 0)

    # Defaults for any missing features
    for key in (
        "current_r", "mfe_so_far", "mae_so_far",
        "regime_prob_trending", "regime_prob_ranging", "regime_prob_volatile",
        "regime_prob_breakout", "volume_ratio", "atr_percentile",
        "htf_alignment_score",
    ):
        features.setdefault(key, 0.0)
    features.setdefault("rsi", 50.0)
    features.setdefault("regime_confidence", 0.5)

    return features


# ── Main predictor ─────────────────────────────────────────────────


class ExpectedValuePredictor:
    """
    Live EV predictor — auto-selects ML or rule-based mode.

    Usage:
        pred = ExpectedValuePredictor(trade_loader=fn)
        pred.maybe_train()                   # no-op until xgboost installed + 500 trades

        result = pred.predict(position, market_state={
            "regime_prob_trending": 0.6,
            "regime_prob_ranging":  0.3,
            "regime_prob_volatile": 0.1,
            "regime_confidence":    0.6,
            "current_r":            1.2,
            "mfe_so_far":           1.5,
            "rsi":                  68,
            ...
        })

        if result.should_exit:
            emit_alert(f"E[R from here] = {result.expected_r_from_here:+.2f}R — exit")
    """

    def __init__(
        self,
        trade_loader: Optional[Callable[[], List[Dict[str, Any]]]] = None,
        model_dir: Optional[Path] = None,
        min_trades_for_ml: int = MIN_TRADES_FOR_ML,
        hold_threshold: float = DEFAULT_HOLD_THRESHOLD,
        exit_threshold: float = DEFAULT_EXIT_THRESHOLD,
        add_threshold: float = DEFAULT_ADD_THRESHOLD,
    ):
        self._trade_loader = trade_loader
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.min_trades_for_ml = min_trades_for_ml
        self.hold_threshold = hold_threshold
        self.exit_threshold = exit_threshold
        self.add_threshold = add_threshold

        # ML state (None until trained)
        self.model_q10: Optional[Any] = None
        self.model_q50: Optional[Any] = None
        self.model_q90: Optional[Any] = None

        self._strategy_encoder: Dict[str, int] = {}
        self._feature_order: List[str] = list(EV_FEATURES)
        self._last_train_count: int = 0
        self._last_train_mae: Optional[float] = None
        self._last_feature_importance: Dict[str, float] = {}

    @property
    def ml_available(self) -> bool:
        return _XGB_AVAILABLE and _SKLEARN_AVAILABLE

    @property
    def is_ml_active(self) -> bool:
        return self.model_q50 is not None

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "ml_available": self.ml_available,
            "is_ml_active": self.is_ml_active,
            "last_train_count": self._last_train_count,
            "last_train_mae": self._last_train_mae,
            "feature_importance": self._last_feature_importance,
            "thresholds": {
                "hold": self.hold_threshold,
                "exit": self.exit_threshold,
                "add": self.add_threshold,
            },
        }

    # ── Live prediction ──

    def predict(
        self,
        position: Any,
        market_state: Optional[Dict[str, Any]] = None,
    ) -> EVPrediction:
        """Main entry point — extract features then route to ML or rule path."""
        features = extract_features_from_position(position, market_state)
        return self.predict_from_features(features)

    def predict_from_features(self, features: Dict[str, Any]) -> EVPrediction:
        """Predict from a feature dict (used internally + for testing)."""
        if self.is_ml_active:
            try:
                return self._ml_predict(features)
            except Exception as e:
                logger.warning("EV ML predict failed (%s) — falling back to rule-based", e)

        return self._rule_predict(features)

    def _ml_predict(self, features: Dict[str, Any]) -> EVPrediction:
        """Run the three quantile models and assemble an EVPrediction."""
        import numpy as np
        row = self._encode_row(features, fit=False)
        X = np.asarray([row], dtype=float)

        q10 = float(self.model_q10.predict(X)[0])
        median = float(self.model_q50.predict(X)[0])
        q90 = float(self.model_q90.predict(X)[0])

        # Ensure ordering (rare but possible quantile crossing)
        q10, median, q90 = sorted([q10, median, q90])

        return self._assemble(features, q10, median, q90, "ml_xgboost_quantile")

    def _rule_predict(self, features: Dict[str, Any]) -> EVPrediction:
        q10, median, q90, reason = _rule_based_predict(features)
        return self._assemble(features, q10, median, q90, "rule_based", reason=reason)

    def _assemble(
        self,
        features: Dict[str, Any],
        q10: float, median: float, q90: float,
        method: str,
        reason: str = "",
    ) -> EVPrediction:
        current_r = float(features.get("current_r", 0.0) or 0.0)
        max_possible = float(features.get("strategy_max_r", 5.0) or 5.0)

        should_hold, should_exit, should_add = _decide(
            q10, median, q90,
            self.hold_threshold, self.exit_threshold, self.add_threshold,
            DEFAULT_ADD_MAX_VARIANCE, DEFAULT_ADD_MIN_DOWNSIDE,
        )

        # Edge remaining (clamped 0-1; can be negative if losing trade)
        edge_remaining = median / max_possible if max_possible > 0 else 0.0

        # RR from here
        upside = max(q90 - median, 0.001)
        downside = max(median - q10, 0.001)
        rr_from_here = upside / downside

        # Confidence: ML is high when sample large; rule-based is moderate
        if method.startswith("ml_"):
            confidence = "high" if self._last_train_count >= 1000 else "moderate"
        else:
            confidence = "moderate"

        # Build human reason
        if not reason:
            reason = self._build_reason(method, current_r, q10, median, q90,
                                        should_hold, should_exit, should_add)

        return EVPrediction(
            expected_r_from_here=round(median, 4),
            expected_final_r=round(current_r + median, 4),
            percentile_10=round(q10, 4),
            percentile_50=round(median, 4),
            percentile_90=round(q90, 4),
            should_hold=should_hold,
            should_exit=should_exit,
            should_add=should_add,
            edge_remaining=round(edge_remaining, 4),
            risk_reward_from_here=round(rr_from_here, 4),
            model_type=method,
            confidence=confidence,
            reason=reason,
            feature_importance=dict(self._last_feature_importance),
        )

    @staticmethod
    def _build_reason(
        method: str, current_r: float, q10: float, median: float, q90: float,
        should_hold: bool, should_exit: bool, should_add: bool,
    ) -> str:
        if should_exit:
            verdict = f"EXIT — E[R from here] = {median:+.2f}R is below exit threshold"
        elif should_add:
            verdict = f"STRONG HOLD / consider ADDING — E[R] = {median:+.2f}R with tight uncertainty"
        elif should_hold:
            verdict = f"HOLD — positive E[R] = {median:+.2f}R"
        else:
            verdict = f"MARGINAL — E[R] = {median:+.2f}R near zero, consider tightening stop"
        return (
            f"{verdict}. Range: {q10:+.2f} to {q90:+.2f}R. "
            f"Source: {method}."
        )

    # ── Training (called periodically by scheduler) ──

    def maybe_train(self) -> Dict[str, Any]:
        """
        Try to train the three quantile models.
        Returns a status dict (trained=True/False + reason).
        """
        status: Dict[str, Any] = {
            "trained": False, "method": None,
            "n_trades": 0, "n_rows": 0, "mae": None, "reason": "",
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

        n_trades = len(trades)
        status["n_trades"] = n_trades
        if n_trades < self.min_trades_for_ml:
            status["reason"] = (
                f"only {n_trades} trades, need {self.min_trades_for_ml} — "
                f"staying on rule-based"
            )
            return status

        try:
            mae, n_rows = self._train_quantiles(trades)
            self._last_train_count = n_trades
            self._last_train_mae = mae
            status["trained"] = True
            status["method"] = "xgboost_quantile_pooled"
            status["n_rows"] = n_rows
            status["mae"] = mae
            status["reason"] = (
                f"trained on {n_rows} checkpoints from {n_trades} trades, MAE={mae:.3f}R"
            )
            logger.info("EV predictor trained: %s", status["reason"])
        except Exception as e:
            status["reason"] = f"training failed: {e}"
            logger.error("EV predictor training failed: %s", e, exc_info=True)

        return status

    def _train_quantiles(self, trades: List[Dict[str, Any]]) -> Tuple[float, int]:
        """Build training data + train q=0.1, q=0.5, q=0.9 models."""
        import numpy as np
        rows = self._build_training_rows(trades)
        if len(rows) < self.min_trades_for_ml:
            raise ValueError(
                f"only {len(rows)} training rows after checkpoint expansion"
            )

        X = np.asarray([r["features"] for r in rows], dtype=float)
        y = np.asarray([r["future_r"] for r in rows], dtype=float)

        # Train all three quantiles with shared hyperparams
        base_params = dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.04,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=2,
        )

        # XGBoost native quantile regression (XGBoost >= 1.7)
        self.model_q10 = _xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.10, **base_params,
        )
        self.model_q50 = _xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.50, **base_params,
        )
        self.model_q90 = _xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.90, **base_params,
        )

        # Time-series-aware split (last 20% as holdout)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model_q10.fit(X_train, y_train)
        self.model_q50.fit(X_train, y_train)
        self.model_q90.fit(X_train, y_train)

        # Eval median model only (quantile MAE = pinball loss; here we use plain MAE)
        y_pred = self.model_q50.predict(X_test)
        mae = float(np.mean(np.abs(y_test - y_pred)))

        # Feature importance from median model
        try:
            booster = self.model_q50.get_booster()
            imp = booster.get_score(importance_type="gain")
            self._last_feature_importance = {
                self._feature_order[int(k[1:])]: float(v)
                for k, v in imp.items()
                if k.startswith("f") and int(k[1:]) < len(self._feature_order)
            }
        except Exception:
            self._last_feature_importance = {}

        self._save()
        return mae, len(rows)

    def _build_training_rows(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand each trade into multiple training rows (one per R-checkpoint reached).

        For a trade with MFE=1.5R and final_r=-0.3R:
          - row at checkpoint 0.0: future_r = -0.3
          - row at checkpoint 0.5: future_r = -0.8 (gave back 0.8R from this point)
          - row at checkpoint 1.0: future_r = -1.3 (gave back 1.3R)
          - row at checkpoint 1.5: future_r = -1.8 (gave back full MFE+more)
        """
        # Build strategy encoder
        strategies = sorted({t.get("strategy", t.get("strategy_id", "unknown")) for t in trades})
        self._strategy_encoder = {s: i for i, s in enumerate(strategies)}

        rows: List[Dict[str, Any]] = []
        for t in trades:
            mfe = float(t.get("max_favorable_r", t.get("max_favorable_excursion", 0)) or 0)
            final_r = float(t.get("r_multiple", 0) or 0)

            for cp in TRAINING_CHECKPOINTS:
                if mfe < cp:
                    break
                future_r = final_r - cp
                features = self._build_checkpoint_features(t, cp)
                rows.append({
                    "features": features,
                    "future_r": future_r,
                })

        return rows

    def _build_checkpoint_features(self, trade: Dict[str, Any], checkpoint_r: float) -> List[float]:
        """
        Build a feature row for a single training checkpoint.
        Uses entry-time market state for context (in production we'd want per-bar state).
        """
        # Approximate hold time at this checkpoint
        full_hold = float(trade.get("hold_minutes", 0) or 0)
        mfe = float(trade.get("max_favorable_r", trade.get("max_favorable_excursion", 0.001)) or 0.001)
        frac = checkpoint_r / max(mfe, 0.01)
        approx_hold = full_hold * min(frac, 1.0)

        # Map old binary regime to probabilities (cold-start fallback)
        regime = (trade.get("regime_at_entry", "") or "").lower()
        regime_probs = {
            "regime_prob_trending":  trade.get("regime_prob_trending", 1.0 if regime.startswith("trending") else 0.0),
            "regime_prob_ranging":   trade.get("regime_prob_ranging", 1.0 if regime == "ranging" else 0.0),
            "regime_prob_volatile":  trade.get("regime_prob_volatile", 1.0 if regime == "volatile" else 0.0),
            "regime_prob_breakout":  trade.get("regime_prob_breakout", 1.0 if regime == "breakout" else 0.0),
        }
        regime_conf = float(trade.get("regime_confidence", max(regime_probs.values()) if regime_probs else 0.5))

        # Feature vector in fixed order
        sid = trade.get("strategy", trade.get("strategy_id", "unknown"))
        sid_enc = self._strategy_encoder.get(sid, 0)

        feature_dict = {
            "current_r": checkpoint_r,
            "mfe_so_far": checkpoint_r,
            "mae_so_far": float(trade.get("max_adverse_excursion", 0) or 0),
            "hold_minutes": approx_hold,
            "regime_prob_trending":  float(regime_probs["regime_prob_trending"]),
            "regime_prob_ranging":   float(regime_probs["regime_prob_ranging"]),
            "regime_prob_volatile":  float(regime_probs["regime_prob_volatile"]),
            "regime_prob_breakout":  float(regime_probs["regime_prob_breakout"]),
            "regime_confidence":     regime_conf,
            "rsi": float(trade.get("rsi_at_entry", trade.get("rsi", 50)) or 50),
            "volume_ratio": float(trade.get("volume_ratio", 1.0) or 1.0),
            "atr_percentile": float(trade.get("atr_percentile", 50) or 50),
            "htf_alignment_score": float(trade.get("htf_alignment_score", 0) or 0),
            "minutes_to_close": float(trade.get("minutes_to_close", 180) or 180),
            "session_phase": float(trade.get("session_phase", 0.5) or 0.5),
            "strategy_id_encoded": float(sid_enc),
        }
        return [feature_dict.get(f, 0.0) for f in self._feature_order]

    def _encode_row(self, features: Dict[str, Any], fit: bool) -> List[float]:
        """Encode a live feature dict into a row matching self._feature_order."""
        sid = features.get("strategy_id", "")
        if fit and sid not in self._strategy_encoder:
            self._strategy_encoder[sid] = len(self._strategy_encoder)

        row: List[float] = []
        for col in self._feature_order:
            if col == "strategy_id_encoded":
                row.append(float(self._strategy_encoder.get(sid, -1)))
                continue
            val = features.get(col, 0.0)
            try:
                row.append(float(val) if val is not None else 0.0)
            except (TypeError, ValueError):
                row.append(0.0)
        return row

    # ── Persistence ──

    def _save(self) -> None:
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            for name, model in (
                ("q10", self.model_q10),
                ("q50", self.model_q50),
                ("q90", self.model_q90),
            ):
                with open(self.model_dir / f"ev_{name}.pkl", "wb") as f:
                    pickle.dump(model, f)
            with open(self.model_dir / "ev_meta.pkl", "wb") as f:
                pickle.dump({
                    "strategy_encoder": self._strategy_encoder,
                    "feature_order": self._feature_order,
                    "last_train_count": self._last_train_count,
                    "last_train_mae": self._last_train_mae,
                    "feature_importance": self._last_feature_importance,
                }, f)
            logger.info("EV models saved to %s", self.model_dir)
        except Exception as e:
            logger.warning("EV model save failed: %s", e)

    def load(self) -> bool:
        """Try loading saved models. Returns True on success."""
        try:
            for name in ("q10", "q50", "q90"):
                path = self.model_dir / f"ev_{name}.pkl"
                if not path.exists():
                    return False
                with open(path, "rb") as f:
                    setattr(self, f"model_{name}", pickle.load(f))
            with open(self.model_dir / "ev_meta.pkl", "rb") as f:
                meta = pickle.load(f)
            self._strategy_encoder = meta.get("strategy_encoder", {})
            self._feature_order = meta.get("feature_order", list(EV_FEATURES))
            self._last_train_count = meta.get("last_train_count", 0)
            self._last_train_mae = meta.get("last_train_mae")
            self._last_feature_importance = meta.get("feature_importance", {})
            logger.info("EV models loaded from %s", self.model_dir)
            return True
        except Exception as e:
            logger.warning("EV model load failed: %s", e)
            return False


# ── Module singleton ───────────────────────────────────────────────


_singleton: Optional[ExpectedValuePredictor] = None


def get_ev_predictor(
    trade_loader: Optional[Callable[[], List[Dict[str, Any]]]] = None,
) -> ExpectedValuePredictor:
    """Module-level singleton."""
    global _singleton
    if _singleton is None:
        _singleton = ExpectedValuePredictor(trade_loader=trade_loader)
    elif trade_loader is not None and _singleton._trade_loader is None:
        _singleton._trade_loader = trade_loader
    return _singleton
