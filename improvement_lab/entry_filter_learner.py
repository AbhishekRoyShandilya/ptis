# improvement_lab/entry_filter_learner.py
"""
Entry Filter Learner
====================

Finds conditions where a strategy consistently fails — regime, time,
emotion, volatility, or combinations — and suggests skipping trades
under those conditions.

Rule-based (30+ trades):
  Segments trades by each axis, computes win rate per segment.
  Flags where win_rate < baseline * 0.6 with N >= 5.

ML hybrid (100+ trades):
  Random Forest classifier on features → win/loss.
  SHAP reveals which features have the strongest negative impact.
  Converts top negative features into interpretable filter rules.

The key insight: the ML layer finds INTERACTION effects that simple
segmentation misses, e.g. "breakouts fail only when volatility is low
AND it's the afternoon session AND you're on a losing streak."
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from improvement_lab.models import (
    EntryFilter,
    ImprovementCandidate,
    ImprovementType,
    PatternType,
    DetectionMethod,
)
from improvement_lab.feature_extractor import FeatureExtractor, _safe_float
from improvement_lab.explainer import ImprovementExplainer

logger = logging.getLogger(__name__)

# Thresholds
MIN_SAMPLE_RULE = 5
MIN_SAMPLE_ML = 15
WR_DROP_THRESHOLD = 0.6  # flag if segment WR < baseline * 0.6
ML_THRESHOLD = 100

# Segmentation axes
SEGMENT_AXES = [
    "market_regime", "session_type", "direction",
    "emotion_pre", "day_of_week",
]


class EntryFilterLearner:
    """
    Learns entry filters — conditions to skip trades.

    Dual-mode:
      1. Rule-based segmentation (from 30 trades)
      2. Random Forest with SHAP (from 100 trades)
    """

    def __init__(self):
        self._extractor = FeatureExtractor()
        self._explainer = ImprovementExplainer()
        self._filter_model = None
        self._feature_names: List[str] = []

    def learn(
        self,
        trades: List[Any],
        strategy_id: str,
        min_trades: int = 30,
    ) -> Tuple[List[EntryFilter], List[ImprovementCandidate]]:
        """
        Learn entry filters from trade history.

        Returns:
            (filters, candidates)
        """
        if len(trades) < min_trades:
            return [], []

        baseline_wr = self._compute_win_rate(trades)

        # ── Rule-based ──
        rule_filters = self._rule_based_learn(trades, strategy_id, baseline_wr)

        # ── ML ──
        ml_filters = []
        if len(trades) >= ML_THRESHOLD:
            ml_filters = self._ml_learn(trades, strategy_id, baseline_wr)

        all_filters = self._merge_filters(rule_filters, ml_filters)
        candidates = self._filters_to_candidates(all_filters, strategy_id, baseline_wr)

        return all_filters, candidates

    # ── Rule-based ───────────────────────────────────────────────

    def _rule_based_learn(
        self,
        trades: List[Any],
        strategy_id: str,
        baseline_wr: float,
    ) -> List[EntryFilter]:
        """Segment trades by condition axes, find underperforming segments."""
        filters = []
        wr_threshold = baseline_wr * WR_DROP_THRESHOLD

        # Single axis
        for axis in SEGMENT_AXES:
            segments = self._segment_by_axis(trades, axis)
            for value, segment_trades in segments.items():
                if len(segment_trades) < MIN_SAMPLE_RULE:
                    continue
                wr = self._compute_win_rate(segment_trades)
                if wr < wr_threshold:
                    lift = wr - baseline_wr
                    filters.append(EntryFilter(
                        strategy_id=strategy_id,
                        filter_type=axis,
                        filter_condition={axis: value},
                        trades_matching=len(segment_trades),
                        win_rate_when_active=round(wr, 3),
                        baseline_win_rate=round(baseline_wr, 3),
                        lift=round(lift, 3),
                        confidence=self._compute_confidence(
                            wr, baseline_wr, len(segment_trades),
                        ),
                        detection_method=DetectionMethod.CONDITIONAL_FREQUENCY.value,
                    ))

        # Two-axis combos
        for ax_a, ax_b in combinations(SEGMENT_AXES, 2):
            segments = self._segment_by_combo(trades, ax_a, ax_b)
            for (val_a, val_b), seg_trades in segments.items():
                if len(seg_trades) < MIN_SAMPLE_RULE:
                    continue
                wr = self._compute_win_rate(seg_trades)
                if wr < wr_threshold and wr < baseline_wr - 0.15:
                    lift = wr - baseline_wr
                    filters.append(EntryFilter(
                        strategy_id=strategy_id,
                        filter_type="composite",
                        filter_condition={ax_a: val_a, ax_b: val_b},
                        trades_matching=len(seg_trades),
                        win_rate_when_active=round(wr, 3),
                        baseline_win_rate=round(baseline_wr, 3),
                        lift=round(lift, 3),
                        confidence=self._compute_confidence(
                            wr, baseline_wr, len(seg_trades),
                        ),
                        detection_method=DetectionMethod.CONDITIONAL_FREQUENCY.value,
                    ))

        # Sort by lift (most negative first)
        filters.sort(key=lambda f: f.lift)
        return filters

    # ── ML detection ─────────────────────────────────────────────

    def _ml_learn(
        self,
        trades: List[Any],
        strategy_id: str,
        baseline_wr: float,
    ) -> List[EntryFilter]:
        """
        Train Random Forest to predict win/loss.
        Use SHAP to find features with strongest negative impact.
        Convert to interpretable entry filters.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            logger.info("scikit-learn not available, skipping ML learning")
            return []

        filters = []
        vectors = self._extractor.extract_batch(trades)

        y = np.array([1 if v.was_winner else 0 for v in vectors])

        if y.sum() < MIN_SAMPLE_ML or (1 - y).sum() < MIN_SAMPLE_ML:
            return []

        X, feature_names = self._extractor.vectors_to_matrix(vectors)

        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=max(3, len(X) // 20),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Temporal out-of-sample evaluation (no lookahead bias)
        split_idx = int(len(X) * 0.8)
        if split_idx >= 10 and len(X) - split_idx >= 5:
            from sklearn.base import clone as _clone
            _eval_model = _clone(model)
            _eval_model.fit(X[:split_idx], y[:split_idx])
            _oos_score = float(_eval_model.score(X[split_idx:], y[split_idx:]))
            logger.info(
                "EntryFilter temporal OOS accuracy: %.3f (%d train / %d test)",
                _oos_score, split_idx, len(X) - split_idx,
            )

        model.fit(X, y)

        # SHAP explanation
        explanation = self._explainer.explain_tree_model(
            model, X, feature_names, top_k=10,
        )

        self._filter_model = model
        self._feature_names = feature_names

        # Find features that HURT win rate (negative SHAP direction)
        negative_features = [
            f for f in explanation.get("top_features", [])
            if f.get("direction") == "negative" and f["importance"] > 0.02
        ]

        if not negative_features:
            return []

        # For each negative feature, find the condition that kills performance
        proba = model.predict_proba(X)[:, 1]  # P(win)

        for feat_info in negative_features[:5]:
            fname = feat_info["feature"]
            fidx = feature_names.index(fname) if fname in feature_names else -1
            if fidx < 0:
                continue

            feature_values = X[:, fidx]

            # Find the value range where win probability drops
            # Split into quartiles
            if feature_values.std() < 0.01:
                continue  # constant feature

            # Binary features (one-hot)
            unique_vals = np.unique(feature_values)
            if len(unique_vals) == 2:
                # One-hot encoded — check which value is bad
                for val in unique_vals:
                    mask = feature_values == val
                    if mask.sum() < MIN_SAMPLE_RULE:
                        continue
                    segment_wr = float(y[mask].mean())
                    if segment_wr < baseline_wr * WR_DROP_THRESHOLD:
                        condition_val = "active" if val == 1.0 else "inactive"
                        filters.append(EntryFilter(
                            strategy_id=strategy_id,
                            filter_type="ml_discovered",
                            filter_condition={fname: condition_val},
                            trades_matching=int(mask.sum()),
                            win_rate_when_active=round(segment_wr, 3),
                            baseline_win_rate=round(baseline_wr, 3),
                            lift=round(segment_wr - baseline_wr, 3),
                            confidence=round(feat_info["importance"], 3),
                            detection_method=DetectionMethod.FEATURE_IMPORTANCE.value,
                            shap_explanation={
                                f["feature"]: f["importance"]
                                for f in explanation.get("top_features", [])
                            },
                        ))
            else:
                # Continuous feature — find the bad quartile
                q25 = np.percentile(feature_values, 25)
                q75 = np.percentile(feature_values, 75)

                # Check low quartile
                low_mask = feature_values <= q25
                if low_mask.sum() >= MIN_SAMPLE_RULE:
                    low_wr = float(y[low_mask].mean())
                    if low_wr < baseline_wr * WR_DROP_THRESHOLD:
                        filters.append(EntryFilter(
                            strategy_id=strategy_id,
                            filter_type="ml_discovered",
                            filter_condition={fname: f"<= {q25:.2f}"},
                            trades_matching=int(low_mask.sum()),
                            win_rate_when_active=round(low_wr, 3),
                            baseline_win_rate=round(baseline_wr, 3),
                            lift=round(low_wr - baseline_wr, 3),
                            confidence=round(feat_info["importance"], 3),
                            detection_method=DetectionMethod.FEATURE_IMPORTANCE.value,
                            shap_explanation={
                                f["feature"]: f["importance"]
                                for f in explanation.get("top_features", [])
                            },
                        ))

                # Check high quartile
                high_mask = feature_values >= q75
                if high_mask.sum() >= MIN_SAMPLE_RULE:
                    high_wr = float(y[high_mask].mean())
                    if high_wr < baseline_wr * WR_DROP_THRESHOLD:
                        filters.append(EntryFilter(
                            strategy_id=strategy_id,
                            filter_type="ml_discovered",
                            filter_condition={fname: f">= {q75:.2f}"},
                            trades_matching=int(high_mask.sum()),
                            win_rate_when_active=round(high_wr, 3),
                            baseline_win_rate=round(baseline_wr, 3),
                            lift=round(high_wr - baseline_wr, 3),
                            confidence=round(feat_info["importance"], 3),
                            detection_method=DetectionMethod.FEATURE_IMPORTANCE.value,
                            shap_explanation={
                                f["feature"]: f["importance"]
                                for f in explanation.get("top_features", [])
                            },
                        ))

        # Check interaction-based filters
        interactions = explanation.get("interaction_effects", [])
        for ix in interactions[:2]:
            if ix.get("type") == "amplifying":
                # Both features together hurt performance
                filters.append(EntryFilter(
                    strategy_id=strategy_id,
                    filter_type="ml_interaction",
                    filter_condition={
                        ix["feature_a"]: "elevated",
                        ix["feature_b"]: "elevated",
                        "interaction": ix["type"],
                    },
                    trades_matching=0,  # interaction-level, not segment-level
                    win_rate_when_active=0.0,
                    baseline_win_rate=round(baseline_wr, 3),
                    lift=round(-abs(ix["correlation"]) * 0.3, 3),
                    confidence=round(abs(ix["correlation"]), 3),
                    detection_method=DetectionMethod.FEATURE_IMPORTANCE.value,
                    shap_explanation={
                        ix["feature_a"]: abs(ix["correlation"]),
                        ix["feature_b"]: abs(ix["correlation"]),
                    },
                ))

        filters.sort(key=lambda f: f.lift)
        return filters

    def predict_win_probability(
        self,
        trade: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Real-time: predict P(win) for a trade before entry.
        Returns None if model not trained.
        """
        if self._filter_model is None:
            return None

        vec = self._extractor.extract(trade, context=context)
        X, _ = self._extractor.vectors_to_matrix([vec])

        if X.shape[1] != len(self._feature_names):
            return None

        proba = float(self._filter_model.predict_proba(X)[0, 1])

        return {
            "win_probability": round(proba, 3),
            "recommendation": (
                "skip" if proba < 0.25
                else "reduce_size" if proba < 0.35
                else "proceed" if proba < 0.6
                else "high_conviction"
            ),
        }

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _get_axis_value(trade: Any, axis: str) -> Optional[str]:
        if axis == "day_of_week":
            entry_time = getattr(trade, "entry_time", None)
            if entry_time and hasattr(entry_time, "weekday"):
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                return days[entry_time.weekday()] if entry_time.weekday() < 5 else None
            return None
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
    def _compute_win_rate(trades: List[Any]) -> float:
        if not trades:
            return 0.0
        winners = sum(
            1 for t in trades
            if _safe_float(getattr(t, "pnl", 0)) > 0
        )
        return winners / len(trades)

    @staticmethod
    def _compute_confidence(wr: float, baseline: float, sample: int) -> float:
        """Confidence from effect size + sample."""
        effect = abs(baseline - wr)
        sample_factor = min(1.0, sample / 30)
        return round(min(0.95, effect * sample_factor * 3), 3)

    @staticmethod
    def _merge_filters(
        rule: List[EntryFilter], ml: List[EntryFilter],
    ) -> List[EntryFilter]:
        return rule + ml

    @staticmethod
    def _filters_to_candidates(
        filters: List[EntryFilter],
        strategy_id: str,
        baseline_wr: float,
    ) -> List[ImprovementCandidate]:
        candidates = []
        for f in filters:
            cond_str = ", ".join(f"{k}={v}" for k, v in f.filter_condition.items())
            candidates.append(ImprovementCandidate(
                strategy_id=strategy_id,
                improvement_type=ImprovementType.ENTRY_FILTER.value,
                analysis_layer="pattern",
                pattern_type=(
                    PatternType.HYBRID.value if f.shap_explanation
                    else PatternType.RULE_BASED.value
                ),
                detection_method=f.detection_method,
                title=f"Skip when {cond_str} — WR drops to {f.win_rate_when_active:.0%}",
                description=(
                    f"When {cond_str}, win rate drops from {baseline_wr:.0%} to "
                    f"{f.win_rate_when_active:.0%} ({f.trades_matching} trades). "
                    f"Skipping these trades would have saved "
                    f"~{abs(f.lift) * f.trades_matching:.1f}R total."
                ),
                expected_impact_r=round(abs(f.lift) * 0.5, 2),
                confidence=f.confidence,
                evidence={
                    "condition": f.filter_condition,
                    "win_rate": f.win_rate_when_active,
                    "baseline_wr": f.baseline_win_rate,
                    "lift": f.lift,
                    "sample_size": f.trades_matching,
                },
                suggested_action=(
                    f"Add filter: skip trades when {cond_str}. "
                    f"This would improve win rate from {f.win_rate_when_active:.0%} "
                    f"back toward baseline {baseline_wr:.0%}."
                ),
                proposed_rule={
                    "type": "entry_filter",
                    "condition": f.filter_condition,
                },
                shap_explanation=f.shap_explanation,
                variant_id=f.variant_id,
                variant_concentration=f.variant_concentration,
            ))
        return candidates
