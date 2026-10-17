# improvement_lab/early_exit_detector.py
"""
Early Exit Pattern Detector
============================

Finds condition clusters that predict stop-loss hit with high probability.
If pattern X preceded SL hit in >70% of cases, the system suggests
exiting immediately when pattern X forms — instead of waiting for full SL.

Rule-based (30+ trades):
  Cross-tabulates each condition axis with SL-hit rate.
  Flags combos where SL_hit_rate >= 70% and sample >= 5.

ML hybrid (100+ trades):
  GBM classifier predicts P(SL hit | features).
  SHAP identifies which feature combos are the strongest predictors.
  Discovers non-obvious patterns humans wouldn't spot.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from improvement_lab.models import (
    EarlyExitPattern,
    ImprovementCandidate,
    ImprovementType,
    PatternType,
    DetectionMethod,
)
from improvement_lab.feature_extractor import FeatureExtractor, _safe_float
from improvement_lab.explainer import ImprovementExplainer

logger = logging.getLogger(__name__)

# Thresholds
SL_HIT_R_THRESHOLD = -0.9     # R <= this = stop loss hit
MIN_SAMPLE_RULE = 5
MIN_SAMPLE_ML = 15
SL_RATE_THRESHOLD = 0.70       # flag if SL hit >= 70% of the time
ML_THRESHOLD = 100

# Condition axes to segment by
CONDITION_AXES = [
    "market_regime", "session_type", "direction", "emotion_pre",
]


class EarlyExitDetector:
    """
    Detects patterns that predict stop-loss hit.

    Dual-mode:
      1. Rule-based conditional frequency (from 30 trades)
      2. GBM classifier with SHAP (from 100 trades)
    """

    def __init__(self):
        self._extractor = FeatureExtractor()
        self._explainer = ImprovementExplainer()
        self._sl_model = None
        self._feature_names: List[str] = []

    def detect(
        self,
        trades: List[Any],
        strategy_id: str,
        min_trades: int = 30,
    ) -> Tuple[List[EarlyExitPattern], List[ImprovementCandidate]]:
        """
        Detect early exit patterns.

        Returns:
            (patterns, candidates)
        """
        if len(trades) < min_trades:
            return [], []

        # ── Rule-based ──
        rule_patterns = self._rule_based_detect(trades, strategy_id)

        # ── ML ──
        ml_patterns = []
        if len(trades) >= ML_THRESHOLD:
            ml_patterns = self._ml_detect(trades, strategy_id)

        all_patterns = self._merge_patterns(rule_patterns, ml_patterns)
        candidates = self._patterns_to_candidates(all_patterns, strategy_id)

        return all_patterns, candidates

    # ── Rule-based ───────────────────────────────────────────────

    def _rule_based_detect(
        self,
        trades: List[Any],
        strategy_id: str,
    ) -> List[EarlyExitPattern]:
        """
        Cross-tabulate condition axes with SL-hit rate.
        Also checks 2-axis combos (e.g. regime + session).
        """
        patterns = []
        total_sl_rate = self._overall_sl_rate(trades)

        # Single-axis segmentation
        for axis in CONDITION_AXES:
            segments = self._segment_by_axis(trades, axis)
            for value, segment_trades in segments.items():
                if len(segment_trades) < MIN_SAMPLE_RULE:
                    continue
                sl_rate = self._sl_rate(segment_trades)
                if sl_rate >= SL_RATE_THRESHOLD and sl_rate > total_sl_rate + 0.15:
                    avg_adverse = self._avg_adverse_r(segment_trades)
                    patterns.append(EarlyExitPattern(
                        strategy_id=strategy_id,
                        condition_fingerprint={axis: value},
                        sl_hit_rate=round(sl_rate, 3),
                        avg_adverse_r=round(avg_adverse, 2),
                        sample_size=len(segment_trades),
                        confidence=self._compute_confidence(
                            sl_rate, len(segment_trades), total_sl_rate,
                        ),
                        trade_ids=[str(getattr(t, "id", "")) for t in segment_trades],
                        suggested_fix=self._suggest_fix({axis: value}, sl_rate),
                        detection_method=DetectionMethod.CONDITIONAL_FREQUENCY.value,
                    ))

        # Two-axis combos (e.g. ranging + afternoon)
        for ax_a, ax_b in combinations(CONDITION_AXES, 2):
            combo_segments = self._segment_by_combo(trades, ax_a, ax_b)
            for (val_a, val_b), segment_trades in combo_segments.items():
                if len(segment_trades) < MIN_SAMPLE_RULE:
                    continue
                sl_rate = self._sl_rate(segment_trades)
                if sl_rate >= SL_RATE_THRESHOLD and sl_rate > total_sl_rate + 0.20:
                    avg_adverse = self._avg_adverse_r(segment_trades)
                    patterns.append(EarlyExitPattern(
                        strategy_id=strategy_id,
                        condition_fingerprint={ax_a: val_a, ax_b: val_b},
                        sl_hit_rate=round(sl_rate, 3),
                        avg_adverse_r=round(avg_adverse, 2),
                        sample_size=len(segment_trades),
                        confidence=self._compute_confidence(
                            sl_rate, len(segment_trades), total_sl_rate,
                        ),
                        trade_ids=[str(getattr(t, "id", "")) for t in segment_trades],
                        suggested_fix=self._suggest_fix(
                            {ax_a: val_a, ax_b: val_b}, sl_rate,
                        ),
                        detection_method=DetectionMethod.CONDITIONAL_FREQUENCY.value,
                    ))

        return patterns

    # ── ML detection ─────────────────────────────────────────────

    def _ml_detect(
        self,
        trades: List[Any],
        strategy_id: str,
    ) -> List[EarlyExitPattern]:
        """
        Train GBM to predict P(SL hit | features).
        Use SHAP to find top condition clusters.
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            logger.info("scikit-learn not available, skipping ML detection")
            return []

        patterns = []
        vectors = self._extractor.extract_batch(trades)

        # Build target: SL hit or not
        y = np.array([
            1 if v.hit_sl else 0
            for v in vectors
        ])

        if y.sum() < MIN_SAMPLE_ML or (1 - y).sum() < MIN_SAMPLE_ML:
            return []  # too imbalanced

        X, feature_names = self._extractor.vectors_to_matrix(vectors)

        # Train GBM
        model = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=4,
            min_samples_leaf=max(3, len(X) // 25),
            learning_rate=0.1,
            subsample=0.8,
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
                "EarlyExit temporal OOS accuracy: %.3f (%d train / %d test)",
                _oos_score, split_idx, len(X) - split_idx,
            )

        model.fit(X, y)

        # SHAP explanation
        explanation = self._explainer.explain_tree_model(
            model, X, feature_names, top_k=8,
        )

        # Store for real-time prediction
        self._sl_model = model
        self._feature_names = feature_names

        # Find high-SL-probability clusters
        proba = model.predict_proba(X)[:, 1]
        high_sl_mask = proba > 0.75
        high_sl_count = int(high_sl_mask.sum())

        if high_sl_count < 3:
            return []

        # Extract dominant condition from SHAP
        ml_conditions = {}
        for feat_info in explanation.get("top_features", [])[:4]:
            fname = feat_info["feature"]
            if feat_info["direction"] == "positive":  # drives SL hit
                fidx = feature_names.index(fname) if fname in feature_names else -1
                if fidx >= 0:
                    high_sl_X = X[high_sl_mask]
                    mean_val = float(np.mean(high_sl_X[:, fidx]))
                    ml_conditions[fname] = round(mean_val, 2)

        high_sl_ids = [
            vectors[i].trade_id for i in range(len(vectors)) if high_sl_mask[i]
        ]
        sl_rate = float(y[high_sl_mask].mean()) if high_sl_count > 0 else 0
        avg_adverse = float(np.mean([
            vectors[i].final_r or 0
            for i in range(len(vectors))
            if high_sl_mask[i]
        ]))

        patterns.append(EarlyExitPattern(
            strategy_id=strategy_id,
            condition_fingerprint=ml_conditions,
            sl_hit_rate=round(float(proba[high_sl_mask].mean()), 3),
            avg_adverse_r=round(avg_adverse, 2),
            sample_size=len(vectors),
            confidence=round(float(model.score(X, y)), 3),
            trade_ids=high_sl_ids,
            suggested_fix=self._suggest_fix(ml_conditions, sl_rate),
            detection_method=DetectionMethod.GRADIENT_BOOSTING.value,
            shap_explanation={
                f["feature"]: f["importance"]
                for f in explanation.get("top_features", [])
            },
        ))

        # Check interaction effects
        interactions = explanation.get("interaction_effects", [])
        if interactions:
            for ix in interactions[:2]:
                combo_cond = {ix["feature_a"]: "high", ix["feature_b"]: "high"}
                patterns.append(EarlyExitPattern(
                    strategy_id=strategy_id,
                    condition_fingerprint=combo_cond,
                    sl_hit_rate=round(float(proba[high_sl_mask].mean()), 3),
                    avg_adverse_r=round(avg_adverse, 2),
                    sample_size=high_sl_count,
                    confidence=round(abs(ix["correlation"]), 3),
                    trade_ids=high_sl_ids[:5],
                    suggested_fix=(
                        f"When {ix['feature_a']} and {ix['feature_b']} are both elevated, "
                        f"SL hit rate spikes. Consider tighter stop or skip."
                    ),
                    detection_method=DetectionMethod.GRADIENT_BOOSTING.value,
                    shap_explanation={
                        ix["feature_a"]: abs(ix["correlation"]),
                        ix["feature_b"]: abs(ix["correlation"]),
                    },
                ))

        return patterns

    def predict_sl_probability(
        self,
        trade: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Real-time: predict probability this trade will hit SL.
        Returns None if model not trained.
        """
        if self._sl_model is None:
            return None

        vec = self._extractor.extract(trade, context=context)
        X, _ = self._extractor.vectors_to_matrix([vec])

        if X.shape[1] != len(self._feature_names):
            return None

        proba = float(self._sl_model.predict_proba(X)[0, 1])

        return {
            "sl_hit_probability": round(proba, 3),
            "recommendation": (
                "exit_immediately" if proba > 0.85
                else "tighten_stop" if proba > 0.7
                else "monitor" if proba > 0.5
                else "hold"
            ),
        }

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _get_axis_value(trade: Any, axis: str) -> Optional[str]:
        val = getattr(trade, axis, None)
        if val is None:
            return None
        return val.value if hasattr(val, "value") else str(val)

    def _segment_by_axis(
        self, trades: List[Any], axis: str,
    ) -> Dict[str, List[Any]]:
        segments: Dict[str, List[Any]] = defaultdict(list)
        for t in trades:
            val = self._get_axis_value(t, axis)
            if val:
                segments[val].append(t)
        return segments

    def _segment_by_combo(
        self, trades: List[Any], ax_a: str, ax_b: str,
    ) -> Dict[Tuple[str, str], List[Any]]:
        segments: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
        for t in trades:
            va = self._get_axis_value(t, ax_a)
            vb = self._get_axis_value(t, ax_b)
            if va and vb:
                segments[(va, vb)].append(t)
        return segments

    @staticmethod
    def _sl_rate(trades: List[Any]) -> float:
        if not trades:
            return 0.0
        sl_count = sum(
            1 for t in trades
            if _safe_float(getattr(t, "r_multiple", 0)) <= SL_HIT_R_THRESHOLD
        )
        return sl_count / len(trades)

    def _overall_sl_rate(self, trades: List[Any]) -> float:
        return self._sl_rate(trades)

    @staticmethod
    def _avg_adverse_r(trades: List[Any]) -> float:
        r_values = [
            _safe_float(getattr(t, "r_multiple", 0))
            for t in trades
            if _safe_float(getattr(t, "r_multiple", 0)) < 0
        ]
        return float(np.mean(r_values)) if r_values else 0.0

    @staticmethod
    def _compute_confidence(sl_rate: float, sample_size: int, baseline: float) -> float:
        """Confidence based on effect size and sample."""
        effect = sl_rate - baseline
        sample_factor = min(1.0, sample_size / 30)
        return round(min(0.95, effect * sample_factor * 2), 3)

    @staticmethod
    def _suggest_fix(conditions: Dict[str, Any], sl_rate: float) -> str:
        cond_str = ", ".join(f"{k}={v}" for k, v in conditions.items())
        if sl_rate > 0.85:
            return (
                f"When {cond_str}: SL hit {sl_rate:.0%} of the time. "
                f"Exit immediately on first adverse signal instead of waiting for full SL."
            )
        return (
            f"When {cond_str}: SL hit {sl_rate:.0%} of the time. "
            f"Consider tightening stop by 30-50% or skipping these setups."
        )

    @staticmethod
    def _merge_patterns(
        rule: List[EarlyExitPattern], ml: List[EarlyExitPattern],
    ) -> List[EarlyExitPattern]:
        # Keep both — ML patterns may overlap but have SHAP explanations
        return rule + ml

    @staticmethod
    def _patterns_to_candidates(
        patterns: List[EarlyExitPattern], strategy_id: str,
    ) -> List[ImprovementCandidate]:
        candidates = []
        for p in patterns:
            cond_str = ", ".join(f"{k}={v}" for k, v in p.condition_fingerprint.items())
            candidates.append(ImprovementCandidate(
                strategy_id=strategy_id,
                improvement_type=ImprovementType.EARLY_EXIT_FILTER.value,
                analysis_layer="pattern",
                pattern_type=(
                    PatternType.HYBRID.value if p.shap_explanation
                    else PatternType.RULE_BASED.value
                ),
                detection_method=p.detection_method,
                title=f"SL predictor: {cond_str} → {p.sl_hit_rate:.0%} SL hit rate",
                description=(
                    f"When {cond_str}, stop loss is hit {p.sl_hit_rate:.0%} of the time "
                    f"(avg loss: {p.avg_adverse_r:.1f}R). Sample: {p.sample_size} trades."
                ),
                expected_impact_r=round(abs(p.avg_adverse_r) * p.sl_hit_rate * 0.3, 2),
                confidence=p.confidence,
                evidence={
                    "condition": p.condition_fingerprint,
                    "sl_hit_rate": p.sl_hit_rate,
                    "avg_adverse_r": p.avg_adverse_r,
                    "sample_size": p.sample_size,
                },
                suggested_action=p.suggested_fix,
                proposed_rule={
                    "type": "early_exit",
                    "condition": p.condition_fingerprint,
                    "exit_r": round(-abs(p.avg_adverse_r) * 0.3, 2),
                },
                shap_explanation=p.shap_explanation,
                explained_trade_ids=p.trade_ids,
                variant_id=p.variant_id,
                variant_concentration=p.variant_concentration,
                behavioral_classification=p.behavioral_classification,
            ))
        return candidates
