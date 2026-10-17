# improvement_lab/rr_killer_detector.py
"""
R:R Killer Detector
===================

Finds trades where price reached a favorable R threshold (1.5R, 2R, 2.5R, 3R)
but then reversed back, eating into profits. Correlates with regime, time,
volume, RSI to find the WHY — not just the what.

Rule-based (30+ trades):
  Groups reversal trades by conditions, flags clusters with >40% reversal rate.

ML hybrid (100+ trades):
  GBM classifier predicts P(reversal | trade at X:R favorable).
  SHAP explains which features drive reversals.
  Finds non-obvious interactions (e.g. ranging + RSI>70 + afternoon = 82% reversal).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from improvement_lab.models import (
    RRKillerPattern,
    ImprovementCandidate,
    ImprovementType,
    PatternType,
    DetectionMethod,
    TradeFeatureVector,
)
from improvement_lab.feature_extractor import FeatureExtractor, _safe_float
from improvement_lab.explainer import ImprovementExplainer

logger = logging.getLogger(__name__)

# R thresholds to check for reversal
R_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]

# Minimum sample to flag a pattern
MIN_SAMPLE_RULE = 5
MIN_SAMPLE_ML = 15

# Reversal rate thresholds
REVERSAL_RATE_WARN = 0.40    # flag at 40%
REVERSAL_RATE_CRITICAL = 0.60  # strong signal at 60%

# Minimum trades for ML
ML_THRESHOLD = 100


class RRKillerDetector:
    """
    Detects R:R killer patterns — trades that reach favorable R then reverse.

    Dual-mode:
      1. Rule-based groupby analysis (from 30 trades)
      2. GBM reversal predictor (from 100 trades)
    """

    def __init__(self):
        self._extractor = FeatureExtractor()
        self._explainer = ImprovementExplainer()
        self._reversal_model = None
        self._feature_names: List[str] = []

    def detect(
        self,
        trades: List[Any],
        strategy_id: str,
        shadow_data: Optional[Dict[str, Dict[str, float]]] = None,
        min_trades: int = 30,
    ) -> Tuple[List[RRKillerPattern], List[ImprovementCandidate]]:
        """
        Detect R:R killer patterns.

        Args:
            trades: closed Trade objects (must have r_multiple, pnl)
            strategy_id: which strategy to analyze
            shadow_data: optional {trade_id: {max_favorable_r, max_adverse_r}} from shadow
            min_trades: minimum trades before analysis

        Returns:
            (patterns, candidates)
        """
        if len(trades) < min_trades:
            return [], []

        # Compute MFE in R-multiples for each trade
        trade_mfe = self._compute_mfe_map(trades, shadow_data)

        # ── Rule-based detection ──
        rule_patterns = self._rule_based_detect(
            trades, strategy_id, trade_mfe,
        )

        # ── ML detection (if enough data) ──
        ml_patterns = []
        if len(trades) >= ML_THRESHOLD:
            ml_patterns = self._ml_detect(trades, strategy_id, trade_mfe)

        # Merge, deduplicate, convert to candidates
        all_patterns = self._merge_patterns(rule_patterns, ml_patterns)
        candidates = self._patterns_to_candidates(all_patterns, strategy_id)

        return all_patterns, candidates

    # ── Rule-based detection ─────────────────────────────────────

    def _rule_based_detect(
        self,
        trades: List[Any],
        strategy_id: str,
        trade_mfe: Dict[str, float],
    ) -> List[RRKillerPattern]:
        """
        Group trades by R-threshold reached × condition, find reversals.
        """
        patterns = []

        for threshold in R_THRESHOLDS:
            # Split: trades that reached this R vs not
            reached = []
            for t in trades:
                tid = str(getattr(t, "id", ""))
                mfe_r = trade_mfe.get(tid)
                if mfe_r is not None and mfe_r >= threshold:
                    final_r = _safe_float(getattr(t, "r_multiple", 0))
                    reached.append((t, mfe_r, final_r))

            if len(reached) < MIN_SAMPLE_RULE:
                continue

            # How many reversed (gave back more than 50% of their peak R)?
            reversed_trades = [
                (t, mfe, final)
                for t, mfe, final in reached
                if final < threshold * 0.5  # gave back more than half
            ]

            reversal_rate = len(reversed_trades) / len(reached)
            if reversal_rate < REVERSAL_RATE_WARN:
                continue

            # Find common conditions among reversed trades
            conditions = self._cluster_conditions(
                [t for t, _, _ in reversed_trades]
            )

            avg_giveback = np.mean([
                mfe - final for _, mfe, final in reversed_trades
            ])

            pattern = RRKillerPattern(
                strategy_id=strategy_id,
                trigger_r=threshold,
                reversal_rate=round(reversal_rate, 3),
                avg_r_given_back=round(float(avg_giveback), 2),
                condition_fingerprint=conditions,
                sample_size=len(reached),
                confidence=min(0.95, reversal_rate * (len(reached) / 30)),
                trade_ids=[str(getattr(t, "id", "")) for t, _, _ in reversed_trades],
                suggested_fix=self._suggest_fix(threshold, reversal_rate, conditions),
                detection_method=DetectionMethod.CONDITIONAL_FREQUENCY.value,
            )
            patterns.append(pattern)

        return patterns

    # ── ML detection ─────────────────────────────────────────────

    def _ml_detect(
        self,
        trades: List[Any],
        strategy_id: str,
        trade_mfe: Dict[str, float],
    ) -> List[RRKillerPattern]:
        """
        Train GBM to predict P(reversal | at favorable R).
        Uses SHAP to find the non-obvious condition combos.
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            logger.info("scikit-learn not available, skipping ML detection")
            return []

        patterns = []

        for threshold in R_THRESHOLDS:
            # Build dataset: only trades that reached this threshold
            X_rows = []
            y_labels = []
            trade_ids = []

            vectors = self._extractor.extract_batch(trades)
            for vec in vectors:
                mfe_r = trade_mfe.get(vec.trade_id)
                if mfe_r is None or mfe_r < threshold:
                    continue

                final_r = vec.final_r or 0.0
                reversed_flag = 1 if final_r < threshold * 0.5 else 0

                X_rows.append(vec)
                y_labels.append(reversed_flag)
                trade_ids.append(vec.trade_id)

            if len(X_rows) < MIN_SAMPLE_ML:
                continue

            y = np.array(y_labels)
            reversal_rate = float(y.mean())
            if reversal_rate < 0.2:  # not enough reversals to learn from
                continue

            X, feature_names = self._extractor.vectors_to_matrix(X_rows)

            # Train GBM
            model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=max(3, len(X_rows) // 20),
                learning_rate=0.1,
                random_state=42,
            )

            # Temporal out-of-sample evaluation (no lookahead bias)
            split_idx = int(len(X) * 0.8)
            if split_idx >= 10 and len(X) - split_idx >= 5:
                from sklearn.base import clone as _clone
                _eval_model = _clone(model)
                _eval_model.fit(X[:split_idx], y[:split_idx])
                _oos_score = float(_eval_model.score(X[split_idx:], y[split_idx:]))
                logger.info(
                    "RRKiller temporal OOS accuracy: %.3f (%d train / %d test)",
                    _oos_score, split_idx, len(X) - split_idx,
                )

            model.fit(X, y)

            # Get SHAP explanation
            explanation = self._explainer.explain_tree_model(
                model, X, feature_names, top_k=6,
            )

            # Find high-reversal clusters from SHAP
            # Trades where model predicts > 0.7 reversal probability
            proba = model.predict_proba(X)[:, 1]
            high_risk_mask = proba > 0.7
            high_risk_count = int(high_risk_mask.sum())

            if high_risk_count >= 3:
                high_risk_ids = [
                    trade_ids[i] for i in range(len(trade_ids)) if high_risk_mask[i]
                ]

                # Extract dominant conditions from top SHAP features
                ml_conditions = {}
                for feat_info in explanation.get("top_features", [])[:3]:
                    fname = feat_info["feature"]
                    if feat_info["direction"] == "positive":  # drives reversal
                        # Find the most common value for this feature among high-risk trades
                        high_risk_X = X[high_risk_mask]
                        fidx = feature_names.index(fname) if fname in feature_names else -1
                        if fidx >= 0:
                            mean_val = float(np.mean(high_risk_X[:, fidx]))
                            ml_conditions[fname] = round(mean_val, 2)

                avg_giveback = 0.0
                giveback_trades = [
                    (trade_mfe.get(tid, 0), vec.final_r or 0)
                    for tid, vec in zip(trade_ids, X_rows)
                    if tid in high_risk_ids
                ]
                if giveback_trades:
                    avg_giveback = float(np.mean([
                        mfe - final for mfe, final in giveback_trades
                    ]))

                pattern = RRKillerPattern(
                    strategy_id=strategy_id,
                    trigger_r=threshold,
                    reversal_rate=round(float(proba[high_risk_mask].mean()), 3),
                    avg_r_given_back=round(avg_giveback, 2),
                    condition_fingerprint=ml_conditions,
                    sample_size=len(X_rows),
                    confidence=round(float(model.score(X, y)), 3),
                    trade_ids=high_risk_ids,
                    suggested_fix=self._suggest_fix(
                        threshold,
                        float(proba[high_risk_mask].mean()),
                        ml_conditions,
                    ),
                    detection_method=DetectionMethod.REVERSAL_PREDICTOR.value,
                    shap_explanation={
                        f["feature"]: f["importance"]
                        for f in explanation.get("top_features", [])
                    },
                )
                patterns.append(pattern)

            # Store model for prediction
            self._reversal_model = model
            self._feature_names = feature_names

        return patterns

    def predict_reversal_probability(
        self,
        trade: Any,
        current_r: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Real-time prediction: given a trade currently at X:R favorable,
        what's the probability it will reverse?

        Returns None if model not trained yet.
        """
        if self._reversal_model is None:
            return None

        vec = self._extractor.extract(trade, context=context)
        X, _ = self._extractor.vectors_to_matrix([vec])

        # Ensure feature alignment
        if X.shape[1] != len(self._feature_names):
            return None

        proba = float(self._reversal_model.predict_proba(X)[0, 1])

        return {
            "reversal_probability": round(proba, 3),
            "current_r": current_r,
            "recommendation": (
                "consider_partial_exit" if proba > 0.7
                else "monitor" if proba > 0.5
                else "hold"
            ),
            "confidence": round(float(self._reversal_model.score(
                X, np.array([1 if proba > 0.5 else 0])
            )), 2) if proba > 0 else 0.0,
        }

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _compute_mfe_map(
        trades: List[Any],
        shadow_data: Optional[Dict[str, Dict[str, float]]],
    ) -> Dict[str, float]:
        """
        Build trade_id → MFE in R-multiples map.

        Sources (priority order):
          1. Shadow data (most accurate — tracked tick by tick)
          2. Trade fields (if available)
          3. Estimation from entry/exit/SL
        """
        mfe_map = {}
        for t in trades:
            tid = str(getattr(t, "id", ""))
            risk = abs(_safe_float(getattr(t, "risk_amount", 0)))
            if risk <= 0:
                # Estimate risk from entry - SL
                entry = _safe_float(getattr(t, "entry_price", 0))
                sl = _safe_float(getattr(t, "stop_loss", 0))
                if entry and sl:
                    risk = abs(entry - sl) * _safe_float(getattr(t, "position_size", 1))

            # Try shadow data first
            if shadow_data and tid in shadow_data:
                mfe_abs = shadow_data[tid].get("max_favorable", 0)
                if risk > 0:
                    mfe_map[tid] = mfe_abs / risk
                continue

            # Estimate from price data
            if risk > 0:
                r_multiple = _safe_float(getattr(t, "r_multiple", 0))
                # MFE is at least the final R if positive
                if r_multiple > 0:
                    mfe_map[tid] = r_multiple * 1.2  # rough estimate
                else:
                    # If lost, MFE was likely small
                    mfe_map[tid] = max(0, r_multiple + 0.5)

        return mfe_map

    @staticmethod
    def _cluster_conditions(trades: List[Any]) -> Dict[str, Any]:
        """Find the most common regime, session, time among a set of trades."""
        regimes = defaultdict(int)
        sessions = defaultdict(int)
        hours = defaultdict(int)

        for t in trades:
            regime = getattr(t, "market_regime", None)
            if regime:
                r_val = regime.value if hasattr(regime, "value") else str(regime)
                regimes[r_val] += 1

            session = getattr(t, "session_type", None)
            if session:
                s_val = session.value if hasattr(session, "value") else str(session)
                sessions[s_val] += 1

            entry_time = getattr(t, "entry_time", None)
            if entry_time:
                hour = entry_time.hour if hasattr(entry_time, "hour") else 10
                hours[hour] += 1

        conditions = {}
        if regimes:
            top_regime = max(regimes, key=regimes.get)
            if regimes[top_regime] / len(trades) > 0.5:
                conditions["regime"] = top_regime
                conditions["regime_pct"] = round(regimes[top_regime] / len(trades), 2)

        if sessions:
            top_session = max(sessions, key=sessions.get)
            if sessions[top_session] / len(trades) > 0.5:
                conditions["session"] = top_session

        if hours:
            top_hour = max(hours, key=hours.get)
            if hours[top_hour] / len(trades) > 0.4:
                conditions["peak_hour"] = top_hour

        return conditions

    @staticmethod
    def _suggest_fix(
        trigger_r: float,
        reversal_rate: float,
        conditions: Dict[str, Any],
    ) -> str:
        """Generate actionable fix suggestion."""
        partial_r = max(1.0, trigger_r - 0.5)
        trail_r = max(1.0, trigger_r - 1.0)

        parts = []
        if reversal_rate > 0.6:
            parts.append(
                f"Book 50% at {partial_r:.1f}R "
                f"(price reverses {reversal_rate:.0%} of the time after reaching {trigger_r:.1f}R)"
            )
        else:
            parts.append(
                f"Activate trailing stop at {trail_r:.1f}R "
                f"({reversal_rate:.0%} reversal rate after {trigger_r:.1f}R)"
            )

        if conditions.get("regime"):
            parts.append(
                f"Especially in {conditions['regime']} markets "
                f"({conditions.get('regime_pct', 0):.0%} of reversals)"
            )

        if conditions.get("peak_hour"):
            parts.append(f"Peak reversal hour: {conditions['peak_hour']}:00 IST")

        return ". ".join(parts)

    @staticmethod
    def _merge_patterns(
        rule_patterns: List[RRKillerPattern],
        ml_patterns: List[RRKillerPattern],
    ) -> List[RRKillerPattern]:
        """Merge rule-based and ML patterns, preferring ML when both exist for same R threshold."""
        merged = {}
        for p in rule_patterns:
            merged[p.trigger_r] = p
        for p in ml_patterns:
            key = p.trigger_r
            if key in merged:
                # ML has higher confidence and better explanation → replace
                p.detection_method = DetectionMethod.REVERSAL_PREDICTOR.value
                merged[key] = p
            else:
                merged[key] = p
        return list(merged.values())

    @staticmethod
    def _patterns_to_candidates(
        patterns: List[RRKillerPattern],
        strategy_id: str,
    ) -> List[ImprovementCandidate]:
        """Convert RRKillerPatterns to unified ImprovementHypotheses."""
        candidates = []
        for p in patterns:
            cond_str = ", ".join(f"{k}={v}" for k, v in p.condition_fingerprint.items())
            candidates.append(ImprovementCandidate(
                strategy_id=strategy_id,
                improvement_type=ImprovementType.RR_KILLER_FIX.value,
                analysis_layer="pattern",
                pattern_type=(
                    PatternType.HYBRID.value if p.shap_explanation
                    else PatternType.RULE_BASED.value
                ),
                detection_method=p.detection_method,
                title=f"R:R killer at {p.trigger_r:.1f}R — {p.reversal_rate:.0%} reversal rate",
                description=(
                    f"Out of {p.sample_size} trades that reached {p.trigger_r:.1f}R favorable, "
                    f"{p.reversal_rate:.0%} reversed and gave back an average of {p.avg_r_given_back:.1f}R. "
                    f"Conditions: {cond_str}"
                ),
                expected_impact_r=round(p.avg_r_given_back * p.reversal_rate * 0.5, 2),
                confidence=p.confidence,
                evidence={
                    "trigger_r": p.trigger_r,
                    "reversal_rate": p.reversal_rate,
                    "avg_r_given_back": p.avg_r_given_back,
                    "condition": p.condition_fingerprint,
                    "sample_size": p.sample_size,
                },
                suggested_action=p.suggested_fix,
                proposed_rule={
                    "type": "partial_exit",
                    "trigger_r": p.trigger_r,
                    "partial_pct": 0.5,
                    "condition": p.condition_fingerprint,
                },
                shap_explanation=p.shap_explanation,
                explained_trade_ids=p.trade_ids,
                variant_id=p.variant_id,
                variant_concentration=p.variant_concentration,
                behavioral_classification=p.behavioral_classification,
            ))
        return candidates
