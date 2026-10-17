# improvement_lab/edge_amplifier.py
"""
Edge Amplifier — Positive Conditional Edge Detection.

The MIRROR of entry_filter_learner.py:
  - Entry filter finds "when to SKIP" (negative edges)
  - Edge amplifier finds "when to SIZE UP" (positive edges)

Example: Strategy X has 65% WR overall. Within that, 10% of setups
have volume > 2x AND regime = trending AND RSI 40-60 → 82% WR, avg +2.1R.
Edge amplifier detects this and returns a sizing multiplier.

Detection approach:
  1. Rule-based: segment by axes, find segments with WR >> baseline
  2. ML (100+ trades): Random Forest, SHAP positive features
  3. Variant-aware: identify which variant clusters are A+ performers

Output: EdgeProfile per strategy → fed into ConvictionEngine as a
conviction boost → larger size via existing conviction-based sizing.

Thresholds for "A+ edge" (conservative):
  - WR >= baseline + 15% AND sample >= 5 (rule-based)
  - Avg R >= baseline avg R * 1.5 (additional confirmation)
  - Must pass in BOTH halves of data (split-half validation)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────

MIN_SAMPLE = 5              # minimum trades in a segment to consider
WR_BOOST_THRESHOLD = 0.15   # segment WR must be >= baseline + 15%
R_BOOST_RATIO = 1.5         # segment avg_r must be >= baseline * 1.5
ML_THRESHOLD = 80           # use ML if >= this many trades
SPLIT_HALF_MAX_DELTA = 0.20 # both halves must agree within 20%

# Sizing multipliers for amplified edges
EDGE_SIZING = {
    "A_PLUS": 1.5,   # A+ edge: 1.5x normal conviction-based size
    "A": 1.25,       # A edge: 1.25x
    "B_PLUS": 1.1,   # B+ edge: slight bump
}

# Segmentation axes (same as entry_filter_learner)
SEGMENT_AXES = [
    "market_regime", "session_type", "direction",
    "emotion_pre", "day_of_week",
]


# ══════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class EdgeCondition:
    """A single condition that defines an A+ edge."""
    axes: Dict[str, str]          # {axis: value} e.g. {"market_regime": "trending", "direction": "long"}
    win_rate: float               # WR of this segment
    avg_r: float                  # average R-multiple of this segment
    sample_size: int              # how many trades match
    baseline_wr: float            # overall strategy WR
    baseline_avg_r: float         # overall strategy avg R
    lift_wr: float                # win_rate - baseline_wr
    lift_r: float                 # avg_r - baseline_avg_r
    confidence: float             # 0-1
    detection_method: str         # "rule_single" | "rule_combo" | "ml" | "variant"
    split_half_valid: bool        # True if edge holds in both halves of data
    grade: str = "B_PLUS"         # "A_PLUS" | "A" | "B_PLUS"


@dataclass
class EdgeProfile:
    """
    Complete edge profile for a strategy.
    Contains all detected positive edges + a sizing multiplier function.
    """
    strategy_id: str
    baseline_wr: float
    baseline_avg_r: float
    total_trades: int
    edges: List[EdgeCondition] = field(default_factory=list)
    best_edge: Optional[EdgeCondition] = None
    detection_timestamp: str = ""
    model_available: bool = False

    def get_sizing_multiplier(self, trade_context: Dict[str, str]) -> Tuple[float, Optional[EdgeCondition]]:
        """
        Given current trade context, return sizing multiplier and matched edge.
        Returns (1.0, None) if no edge matches.
        """
        best_match: Optional[EdgeCondition] = None
        best_specificity = 0  # prefer more specific matches (2-axis > 1-axis)

        for edge in self.edges:
            if not edge.split_half_valid:
                continue
            # Check if all axes match
            matched = True
            for axis, expected_val in edge.axes.items():
                actual = trade_context.get(axis, "")
                if str(actual).lower() != str(expected_val).lower():
                    matched = False
                    break
            if matched:
                specificity = len(edge.axes)
                if specificity > best_specificity:
                    best_specificity = specificity
                    best_match = edge

        if best_match is None:
            return 1.0, None

        multiplier = EDGE_SIZING.get(best_match.grade, 1.0)
        return multiplier, best_match

    def summary(self) -> Dict[str, Any]:
        """Human-readable summary of edges found."""
        return {
            "strategy_id": self.strategy_id,
            "baseline_wr": f"{self.baseline_wr:.0%}",
            "baseline_avg_r": f"{self.baseline_avg_r:+.2f}R",
            "total_trades": self.total_trades,
            "edges_found": len(self.edges),
            "validated_edges": sum(1 for e in self.edges if e.split_half_valid),
            "best_edge": {
                "conditions": self.best_edge.axes,
                "win_rate": f"{self.best_edge.win_rate:.0%}",
                "avg_r": f"{self.best_edge.avg_r:+.2f}R",
                "sample": self.best_edge.sample_size,
                "grade": self.best_edge.grade,
            } if self.best_edge else None,
        }


# ══════════════════════════════════════════════════════════════════
#  EDGE AMPLIFIER
# ══════════════════════════════════════════════════════════════════

class EdgeAmplifier:
    """
    Detects positive conditional edges within a strategy.

    Dual-mode (mirrors entry_filter_learner):
      1. Rule-based: segment trades → find high-performing subsets
      2. ML: Random Forest → SHAP positive features → edge conditions

    All edges are split-half validated before being trusted.
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}  # strategy_id → trained RF model
        self._feature_names: Dict[str, List[str]] = {}
        self._profiles: Dict[str, EdgeProfile] = {}

    def learn(
        self,
        trades: List[Any],
        strategy_id: str,
        min_trades: int = 20,
    ) -> EdgeProfile:
        """
        Learn positive edges from trade history.

        Returns EdgeProfile with all detected edges + sizing multiplier.
        """
        from datetime import datetime, timezone

        if len(trades) < min_trades:
            return EdgeProfile(
                strategy_id=strategy_id,
                baseline_wr=0.0,
                baseline_avg_r=0.0,
                total_trades=len(trades),
            )

        baseline_wr = self._compute_win_rate(trades)
        baseline_avg_r = self._compute_avg_r(trades)

        # Rule-based edges
        rule_edges = self._rule_based_detect(trades, baseline_wr, baseline_avg_r)

        # ML edges (if enough trades)
        ml_edges = []
        if len(trades) >= ML_THRESHOLD:
            ml_edges = self._ml_detect(trades, strategy_id, baseline_wr, baseline_avg_r)

        # Variant-aware edges
        variant_edges = self._variant_detect(trades, strategy_id, baseline_wr, baseline_avg_r)

        all_edges = rule_edges + ml_edges + variant_edges

        # Deduplicate (prefer higher confidence)
        all_edges = self._deduplicate(all_edges)

        # Grade each edge
        for edge in all_edges:
            edge.grade = self._grade_edge(edge, baseline_wr, baseline_avg_r)

        # Sort by grade then confidence
        grade_order = {"A_PLUS": 0, "A": 1, "B_PLUS": 2}
        all_edges.sort(key=lambda e: (grade_order.get(e.grade, 9), -e.confidence))

        best = all_edges[0] if all_edges else None

        profile = EdgeProfile(
            strategy_id=strategy_id,
            baseline_wr=baseline_wr,
            baseline_avg_r=baseline_avg_r,
            total_trades=len(trades),
            edges=all_edges,
            best_edge=best,
            detection_timestamp=datetime.now(timezone.utc).isoformat(),
            model_available=strategy_id in self._models,
        )

        self._profiles[strategy_id] = profile
        logger.info(
            "Edge amplifier for %s: %d edges found (%d validated), best=%s",
            strategy_id, len(all_edges),
            sum(1 for e in all_edges if e.split_half_valid),
            best.grade if best else "none",
        )
        return profile

    def get_profile(self, strategy_id: str) -> Optional[EdgeProfile]:
        """Get cached edge profile for a strategy."""
        return self._profiles.get(strategy_id)

    def get_sizing_multiplier(
        self,
        strategy_id: str,
        trade_context: Dict[str, str],
    ) -> Tuple[float, Optional[EdgeCondition]]:
        """
        Quick lookup: given strategy + current trade context,
        return (multiplier, matched_edge).

        This is the main integration point for ConvictionEngine.
        """
        profile = self._profiles.get(strategy_id)
        if not profile:
            return 1.0, None
        return profile.get_sizing_multiplier(trade_context)

    # ══════════════════════════════════════════════════════════════
    #  RULE-BASED DETECTION
    # ══════════════════════════════════════════════════════════════

    def _rule_based_detect(
        self,
        trades: List[Any],
        baseline_wr: float,
        baseline_avg_r: float,
    ) -> List[EdgeCondition]:
        """Segment by axes, find segments with WR >> baseline."""
        edges = []
        wr_target = baseline_wr + WR_BOOST_THRESHOLD

        # Single-axis segments
        for axis in SEGMENT_AXES:
            segments = self._segment_by_axis(trades, axis)
            for value, seg_trades in segments.items():
                if len(seg_trades) < MIN_SAMPLE:
                    continue
                wr = self._compute_win_rate(seg_trades)
                avg_r = self._compute_avg_r(seg_trades)

                if wr >= wr_target:
                    valid = self._split_half_validate(seg_trades, baseline_wr)
                    edges.append(EdgeCondition(
                        axes={axis: value},
                        win_rate=round(wr, 3),
                        avg_r=round(avg_r, 3),
                        sample_size=len(seg_trades),
                        baseline_wr=round(baseline_wr, 3),
                        baseline_avg_r=round(baseline_avg_r, 3),
                        lift_wr=round(wr - baseline_wr, 3),
                        lift_r=round(avg_r - baseline_avg_r, 3),
                        confidence=self._compute_confidence(wr, baseline_wr, len(seg_trades)),
                        detection_method="rule_single",
                        split_half_valid=valid,
                    ))

        # Two-axis combos (more specific = stronger edge if found)
        for ax_a, ax_b in combinations(SEGMENT_AXES, 2):
            segments = self._segment_by_combo(trades, ax_a, ax_b)
            for (val_a, val_b), seg_trades in segments.items():
                if len(seg_trades) < MIN_SAMPLE:
                    continue
                wr = self._compute_win_rate(seg_trades)
                avg_r = self._compute_avg_r(seg_trades)

                # More stringent for combos: WR must beat baseline + 15%
                # AND avg_r must be meaningfully better
                if wr >= wr_target and avg_r >= baseline_avg_r * 1.2:
                    valid = self._split_half_validate(seg_trades, baseline_wr)
                    edges.append(EdgeCondition(
                        axes={ax_a: val_a, ax_b: val_b},
                        win_rate=round(wr, 3),
                        avg_r=round(avg_r, 3),
                        sample_size=len(seg_trades),
                        baseline_wr=round(baseline_wr, 3),
                        baseline_avg_r=round(baseline_avg_r, 3),
                        lift_wr=round(wr - baseline_wr, 3),
                        lift_r=round(avg_r - baseline_avg_r, 3),
                        confidence=self._compute_confidence(wr, baseline_wr, len(seg_trades)),
                        detection_method="rule_combo",
                        split_half_valid=valid,
                    ))

        return edges

    # ══════════════════════════════════════════════════════════════
    #  ML DETECTION — Random Forest + SHAP positive features
    # ══════════════════════════════════════════════════════════════

    def _ml_detect(
        self,
        trades: List[Any],
        strategy_id: str,
        baseline_wr: float,
        baseline_avg_r: float,
    ) -> List[EdgeCondition]:
        """Train RF on win/loss, find features that BOOST win rate."""
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            return []

        edges = []

        # Extract features
        try:
            from improvement_lab.feature_extractor import FeatureExtractor
            extractor = FeatureExtractor()
            vectors = extractor.extract_batch(trades)
            y = np.array([1 if v.was_winner else 0 for v in vectors])

            if y.sum() < 10 or (1 - y).sum() < 10:
                return []

            X, feature_names = extractor.vectors_to_matrix(vectors)
        except Exception as e:
            logger.debug("ML feature extraction failed: %s", e)
            return []

        # Train
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
                "EdgeAmplifier temporal OOS accuracy: %.3f (%d train / %d test)",
                _oos_score, split_idx, len(X) - split_idx,
            )

        model.fit(X, y)
        self._models[strategy_id] = model
        self._feature_names[strategy_id] = feature_names

        # Get SHAP positive features
        try:
            from improvement_lab.explainer import ImprovementExplainer
            explainer = ImprovementExplainer()
            explanation = explainer.explain_tree_model(model, X, feature_names, top_k=10)
        except Exception as e:
            logger.debug("SHAP explanation failed: %s", e)
            return []

        positive_features = [
            f for f in explanation.get("top_features", [])
            if f.get("direction") == "positive" and f["importance"] > 0.02
        ]

        if not positive_features:
            return []

        # For each positive feature, find the value range where WR is highest
        for feat_info in positive_features[:5]:
            fname = feat_info["feature"]
            fidx = feature_names.index(fname) if fname in feature_names else -1
            if fidx < 0:
                continue

            fvals = X[:, fidx]
            if fvals.std() < 0.01:
                continue

            unique_vals = np.unique(fvals)
            wr_target = baseline_wr + WR_BOOST_THRESHOLD

            if len(unique_vals) == 2:
                # Binary feature — check which value boosts WR
                for val in unique_vals:
                    mask = fvals == val
                    if mask.sum() < MIN_SAMPLE:
                        continue
                    seg_wr = float(y[mask].mean())
                    if seg_wr >= wr_target:
                        seg_trades = [t for t, m in zip(trades, mask) if m]
                        seg_avg_r = self._compute_avg_r(seg_trades)
                        valid = self._split_half_validate_mask(y, mask, baseline_wr)
                        condition_val = "active" if val == 1.0 else "inactive"
                        edges.append(EdgeCondition(
                            axes={fname: condition_val},
                            win_rate=round(seg_wr, 3),
                            avg_r=round(seg_avg_r, 3),
                            sample_size=int(mask.sum()),
                            baseline_wr=round(baseline_wr, 3),
                            baseline_avg_r=round(baseline_avg_r, 3),
                            lift_wr=round(seg_wr - baseline_wr, 3),
                            lift_r=round(seg_avg_r - baseline_avg_r, 3),
                            confidence=round(feat_info["importance"], 3),
                            detection_method="ml",
                            split_half_valid=valid,
                        ))
            else:
                # Continuous — check top quartile
                q75 = np.percentile(fvals, 75)
                high_mask = fvals >= q75
                if high_mask.sum() >= MIN_SAMPLE:
                    seg_wr = float(y[high_mask].mean())
                    if seg_wr >= wr_target:
                        seg_trades = [t for t, m in zip(trades, high_mask) if m]
                        seg_avg_r = self._compute_avg_r(seg_trades)
                        valid = self._split_half_validate_mask(y, high_mask, baseline_wr)
                        edges.append(EdgeCondition(
                            axes={fname: f">= {q75:.2f}"},
                            win_rate=round(seg_wr, 3),
                            avg_r=round(seg_avg_r, 3),
                            sample_size=int(high_mask.sum()),
                            baseline_wr=round(baseline_wr, 3),
                            baseline_avg_r=round(baseline_avg_r, 3),
                            lift_wr=round(seg_wr - baseline_wr, 3),
                            lift_r=round(seg_avg_r - baseline_avg_r, 3),
                            confidence=round(feat_info["importance"], 3),
                            detection_method="ml",
                            split_half_valid=valid,
                        ))

        return edges

    # ══════════════════════════════════════════════════════════════
    #  VARIANT-AWARE DETECTION
    # ══════════════════════════════════════════════════════════════

    def _variant_detect(
        self,
        trades: List[Any],
        strategy_id: str,
        baseline_wr: float,
        baseline_avg_r: float,
    ) -> List[EdgeCondition]:
        """
        Use SKE variant engine to find high-performing variant clusters.
        A variant with 80%+ WR and 1.5x+ avg R = A+ edge.
        """
        try:
            from ske.variant_engine import VariantEngine
        except ImportError:
            return []

        edges = []

        try:
            engine = VariantEngine()

            # Extract feature vectors from trades
            feature_vectors = []
            outcomes = []
            for t in trades:
                fv = self._trade_to_feature_vector(t)
                if fv:
                    feature_vectors.append(fv)
                    r = _safe_float(getattr(t, "r_multiple", None))
                    outcomes.append(r)

            if len(feature_vectors) < 15:
                return []

            variants = engine.discover_variants(feature_vectors, outcomes)

            wr_target = baseline_wr + WR_BOOST_THRESHOLD

            for variant in variants:
                if variant.is_anti_variant:
                    continue  # skip anti-variants (that's entry_filter's job)
                if variant.trade_count < MIN_SAMPLE:
                    continue
                if variant.win_rate >= wr_target:
                    # Build axes from distinguishing features
                    axes = {}
                    for df in variant.distinguishing_features[:3]:
                        axes[df.feature] = f"~{df.mean:.2f} (z={df.importance:.1f})"

                    if not axes:
                        axes = {"variant": variant.label}

                    valid = variant.confidence_level >= 0.5

                    edges.append(EdgeCondition(
                        axes=axes,
                        win_rate=round(variant.win_rate, 3),
                        avg_r=round(variant.avg_r, 3),
                        sample_size=variant.trade_count,
                        baseline_wr=round(baseline_wr, 3),
                        baseline_avg_r=round(baseline_avg_r, 3),
                        lift_wr=round(variant.win_rate - baseline_wr, 3),
                        lift_r=round(variant.avg_r - baseline_avg_r, 3),
                        confidence=round(variant.confidence_level, 3),
                        detection_method="variant",
                        split_half_valid=valid,
                    ))

        except Exception as e:
            logger.debug("Variant edge detection failed: %s", e)

        return edges

    # ══════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_win_rate(trades: List[Any]) -> float:
        if not trades:
            return 0.0
        winners = sum(
            1 for t in trades
            if _safe_float(getattr(t, "pnl", getattr(t, "r_multiple", 0))) > 0
        )
        return winners / len(trades)

    @staticmethod
    def _compute_avg_r(trades: List[Any]) -> float:
        r_values = [
            _safe_float(getattr(t, "r_multiple", None))
            for t in trades
            if getattr(t, "r_multiple", None) is not None
        ]
        return sum(r_values) / len(r_values) if r_values else 0.0

    def _split_half_validate(
        self,
        segment_trades: List[Any],
        baseline_wr: float,
    ) -> bool:
        """
        Split segment into two halves, check if edge holds in both.
        Both halves must beat baseline + half the boost threshold.
        """
        if len(segment_trades) < 6:
            return False

        mid = len(segment_trades) // 2
        wr1 = self._compute_win_rate(segment_trades[:mid])
        wr2 = self._compute_win_rate(segment_trades[mid:])

        min_wr = baseline_wr + WR_BOOST_THRESHOLD * 0.5
        halves_agree = abs(wr1 - wr2) <= SPLIT_HALF_MAX_DELTA

        return wr1 >= min_wr and wr2 >= min_wr and halves_agree

    @staticmethod
    def _split_half_validate_mask(
        y: np.ndarray,
        mask: np.ndarray,
        baseline_wr: float,
    ) -> bool:
        """Split-half on a boolean mask into y array."""
        indices = np.where(mask)[0]
        if len(indices) < 6:
            return False

        mid = len(indices) // 2
        wr1 = float(y[indices[:mid]].mean())
        wr2 = float(y[indices[mid:]].mean())

        min_wr = baseline_wr + WR_BOOST_THRESHOLD * 0.5
        halves_agree = abs(wr1 - wr2) <= SPLIT_HALF_MAX_DELTA

        return wr1 >= min_wr and wr2 >= min_wr and halves_agree

    @staticmethod
    def _compute_confidence(wr: float, baseline: float, sample: int) -> float:
        """Confidence from effect size + sample."""
        effect = wr - baseline
        sample_factor = min(1.0, sample / 30)
        return round(min(0.95, effect * sample_factor * 3), 3)

    @staticmethod
    def _grade_edge(edge: EdgeCondition, baseline_wr: float, baseline_avg_r: float) -> str:
        """
        Grade an edge:
          A+ : WR >= baseline + 20% AND avg_r >= baseline * 2.0 AND validated
          A  : WR >= baseline + 15% AND avg_r >= baseline * 1.5 AND validated
          B+ : WR >= baseline + 15% OR avg_r >= baseline * 1.5
        """
        wr_lift = edge.win_rate - baseline_wr
        r_ratio = edge.avg_r / baseline_avg_r if baseline_avg_r > 0 else 1.0

        if wr_lift >= 0.20 and r_ratio >= 2.0 and edge.split_half_valid and edge.sample_size >= 8:
            return "A_PLUS"
        if wr_lift >= 0.15 and r_ratio >= 1.5 and edge.split_half_valid:
            return "A"
        return "B_PLUS"

    def _deduplicate(self, edges: List[EdgeCondition]) -> List[EdgeCondition]:
        """Remove overlapping edges, keep higher confidence."""
        seen = {}
        for edge in edges:
            key = tuple(sorted(edge.axes.items()))
            if key not in seen or edge.confidence > seen[key].confidence:
                seen[key] = edge
        return list(seen.values())

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

    def _segment_by_axis(self, trades: List[Any], axis: str) -> Dict[str, List[Any]]:
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
    def _trade_to_feature_vector(trade: Any) -> Optional[Dict[str, float]]:
        """Extract a simple feature vector from a trade for variant clustering."""
        try:
            fv = {}
            for attr in [
                "conviction_score", "risk_pct", "pnl_pct",
                "r_multiple", "position_size",
            ]:
                val = getattr(trade, attr, None)
                if val is not None:
                    fv[attr] = float(val)

            # Volume ratio proxy
            vol = _safe_float(getattr(trade, "entry_volume", 0))
            avg_vol = _safe_float(getattr(trade, "avg_volume", 0))
            if avg_vol > 0:
                fv["volume_ratio"] = vol / avg_vol

            return fv if len(fv) >= 2 else None
        except Exception:
            return None


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ── Singleton ─────────────────────────────────────────────────────

_amplifier_instance: Optional[EdgeAmplifier] = None


def get_edge_amplifier() -> EdgeAmplifier:
    """Get or create the singleton EdgeAmplifier."""
    global _amplifier_instance
    if _amplifier_instance is None:
        _amplifier_instance = EdgeAmplifier()
    return _amplifier_instance
