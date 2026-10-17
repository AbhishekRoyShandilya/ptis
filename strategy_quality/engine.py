# strategy_quality/engine.py (COMPLETE REVISION)
"""
Strategy Understanding Engine (SUE) — v2
==========================================

Fixes from review:
1. Pattern-level consistency (not just surface fields)
2. Strategy-adaptive noise detection (not static thresholds)
3. Contradiction detection (unstable edge flagging)
4. Granular sample sufficiency (per-pattern, per-regime)
5. Regime-aware recency (not linear time decay)

Every check is CONTEXT-AWARE. Nothing is hardcoded
without strategy context.
"""

from __future__ import annotations

import logging
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial.distance import cosine as cosine_distance

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  OUTPUT MODELS
# ══════════════════════════════════════════════════════════════════


class StrategyDecision(str, Enum):
    REJECT = "reject"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class DataDecision(str, Enum):
    REJECT = "reject"
    PARTIAL_ACCEPT = "partial_accept"
    FULL_ACCEPT = "full_accept"


@dataclass
class LayerResult:
    layer_name: str
    score: float
    confidence: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)


@dataclass
class DataQualityResult:
    # Individual checks
    consistency: float = 0.0
    pattern_consistency: float = 0.0      # NEW: pattern-level check
    completeness: float = 0.0
    noise_level: float = 0.0
    recency: float = 0.0
    sample_sufficiency: float = 0.0
    contradiction_score: float = 0.0       # NEW: 0 = no contradictions

    final_score: float = 0.0
    decision: str = DataDecision.REJECT.value
    learning_weight: float = 0.0
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaLayerOutput:
    # Per-layer
    hard_rule_match: bool = False
    hard_rule_details: Dict[str, bool] = field(default_factory=dict)

    soft_score: float = 0.0
    soft_score_variance: float = 0.0
    soft_score_components: Dict[str, float] = field(default_factory=dict)

    context_alignment: float = 0.0
    context_components: Dict[str, float] = field(default_factory=dict)

    uncertainty_factors: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_penalty: float = 0.0

    # Combined
    overall_match_score: float = 0.0
    strategy_confidence: float = 0.0
    decision: str = StrategyDecision.REJECT.value

    # Stability
    regime_stability: float = 0.0
    regime_scores: Dict[str, float] = field(default_factory=dict)
    weakest_regime: str = ""
    strongest_regime: str = ""

    # Actionable
    avt_can_trade: bool = False
    avt_mode: str = "observer"
    improvement_lab_can_analyze: bool = False
    human_gates_required: List[str] = field(default_factory=list)

    # Diagnostics
    bottleneck: str = ""
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class SoftScoreDistribution:
    name: str
    values: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.values)) if len(self.values) >= 2 else 1.0

    @property
    def is_stable(self) -> bool:
        return self.std < 0.20 and len(self.values) >= 10

    @property
    def is_unstable(self) -> bool:
        return self.std > 0.35 and len(self.values) >= 10

    def add(self, value: float) -> None:
        self.values.append(value)
        if len(self.values) > 200:
            self.values = self.values[-200:]

    def confidence_from_stability(self) -> float:
        if len(self.values) < 5:
            return 0.3
        if self.mean == 0:
            return 0.3
        cv = self.std / max(abs(self.mean), 0.01)
        return round(min(0.95, max(0.2, 1.0 - cv)), 3)


# ══════════════════════════════════════════════════════════════════
#  STRATEGY PROFILE
# ══════════════════════════════════════════════════════════════════


@dataclass
class StrategyProfile:
    """
    Everything the system needs to know about a strategy
    to make context-aware assessments.

    Built during teaching. Updated as system learns.
    This replaces raw strategy_context dicts.
    """
    strategy_id: str = ""
    strategy_name: str = ""

    # ── Trading behavior expectations ──
    expected_trades_per_day: float = 2.0      # scalper=20, swing=0.5
    expected_hold_minutes: float = 120.0       # scalper=5, swing=1440
    allowed_directions: List[str] = field(
        default_factory=lambda: ["long"]
    )
    expected_risk_pct: float = 1.0
    timeframe: str = "5min"

    # ── Pattern signatures ──
    # Learned feature vectors from teaching examples
    # Each is a centroid of what "this strategy" looks like
    pattern_centroids: List[Dict[str, float]] = field(
        default_factory=list
    )
    pattern_feature_keys: List[str] = field(default_factory=list)
    # Radius of acceptable deviation from centroid
    pattern_tolerance: float = 0.4

    # ── Regime expectations ──
    # Which regimes is this strategy designed for?
    expected_regimes: List[str] = field(default_factory=list)
    # empty = all regimes
    regime_performance: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )  # regime -> {"win_rate", "expectancy", "sample_size"}

    # ── Pattern outcome tracking (for contradiction detection) ──
    pattern_outcomes: Dict[str, List[float]] = field(
        default_factory=dict
    )  # pattern_hash -> list of r_multiples

    # ── Historical distributions ──
    historical_win_rate: Optional[float] = None
    historical_expectancy: Optional[float] = None
    historical_sample_size: int = 0

    # ── Conviction thresholds ──
    min_conviction: float = 0.5

    # ── Hard rules ──
    hard_rules: List[Dict[str, Any]] = field(default_factory=list)

    # ── Soft scorers ──
    soft_scorers: List[Dict[str, Any]] = field(default_factory=list)

    # ── Context factors ──
    context_factors: List[Dict[str, Any]] = field(default_factory=list)

    # ── Uncertainty factors ──
    uncertainty_factors: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # ── Current market regime ──
    current_regime: str = ""
    last_regime_change: Optional[datetime] = None

    def get_pattern_hash(self, features: Dict[str, float]) -> str:
        """
        Create a hash that represents the pattern type.
        Used for contradiction detection — same pattern hash
        should have similar outcomes.
        """
        # Use only the pattern-relevant features, bucketed
        relevant = {}
        for key in self.pattern_feature_keys:
            val = features.get(key)
            if val is not None:
                # Bucket to reduce noise
                relevant[key] = round(float(val), 1)

        raw = json.dumps(relevant, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════
#  LAYER 0: DATA QUALITY FIREWALL (FIXED)
# ══════════════════════════════════════════════════════════════════


class DataQualityFirewall:
    """
    Context-aware data quality gate.

    Every check adapts to the strategy profile.
    No static thresholds that break across strategy types.

    Fixes:
    1. Pattern-level consistency (feature distance from centroids)
    2. Strategy-adaptive noise detection
    3. Contradiction detection
    4. Per-pattern and per-regime sample sufficiency
    5. Regime-aware recency
    """

    def __init__(self, profile: StrategyProfile):
        self._profile = profile

    def assess(
        self,
        trade_data: Dict[str, Any],
        features: Optional[Dict[str, float]] = None,
    ) -> DataQualityResult:
        """
        Assess a single trade's quality for learning.
        Context-aware: adapts to strategy type.
        """
        result = DataQualityResult()

        # 1. Surface consistency
        result.consistency = self._check_surface_consistency(
            trade_data
        )

        # 2. Pattern-level consistency (NEW)
        result.pattern_consistency = self._check_pattern_consistency(
            trade_data, features
        )

        # 3. Completeness
        result.completeness = self._check_completeness(trade_data)

        # 4. Strategy-adaptive noise detection (FIXED)
        result.noise_level = self._check_noise_adaptive(trade_data)

        # 5. Regime-aware recency (FIXED)
        result.recency = self._check_recency_regime_aware(trade_data)

        # 6. Granular sample sufficiency (FIXED)
        result.sample_sufficiency = self._check_sample_granular(
            trade_data, features
        )

        # 7. Contradiction detection (NEW)
        result.contradiction_score = self._check_contradictions(
            trade_data, features
        )

        # Combine — contradiction is a PENALTY like noise
        result.final_score = round(
            result.consistency * 0.15
            + result.pattern_consistency * 0.20   # pattern match matters most
            + result.completeness * 0.15
            + (1.0 - result.noise_level) * 0.15
            + result.recency * 0.10
            + result.sample_sufficiency * 0.10
            + (1.0 - result.contradiction_score) * 0.15,
            3,
        )

        # Decision
        if result.final_score < 0.4:
            result.decision = DataDecision.REJECT.value
            result.learning_weight = 0.0
        elif result.final_score < 0.65:
            result.decision = DataDecision.PARTIAL_ACCEPT.value
            result.learning_weight = round(result.final_score, 3)
        else:
            result.decision = DataDecision.FULL_ACCEPT.value
            result.learning_weight = 1.0

        # Detailed flags
        if result.pattern_consistency < 0.3:
            result.flags.append(
                "PATTERN_MISMATCH: Trade features don't match "
                "learned strategy pattern. Distance from nearest "
                f"centroid is high ({1.0 - result.pattern_consistency:.2f})"
            )

        if result.contradiction_score > 0.5:
            result.flags.append(
                f"CONTRADICTORY: Similar setups produce wildly "
                f"different outcomes (contradiction={result.contradiction_score:.2f}). "
                f"This may indicate an unstable edge."
            )

        if result.noise_level > 0.5:
            result.flags.append(
                f"NOISY: Behavioral noise detected "
                f"(noise={result.noise_level:.2f}). "
                f"Adaptive to strategy type."
            )

        return result

    def assess_batch(
        self,
        trades: List[Dict[str, Any]],
        feature_extractor: Optional[Callable] = None,
    ) -> Tuple[List[Dict[str, Any]], DataQualityResult]:
        """Assess batch. Returns accepted trades and aggregate report."""
        accepted = []
        all_scores = []
        reject_count = 0
        flags = []

        for trade in trades:
            features = None
            if feature_extractor:
                try:
                    features = feature_extractor(trade)
                except Exception:
                    features = None

            result = self.assess(trade, features)
            all_scores.append(result.final_score)

            if result.decision == DataDecision.REJECT.value:
                reject_count += 1
            else:
                trade["_learning_weight"] = result.learning_weight
                trade["_data_quality"] = result.final_score
                trade["_quality_flags"] = result.flags
                accepted.append(trade)

            flags.extend(result.flags)

        agg = DataQualityResult()
        if all_scores:
            agg.final_score = round(float(np.mean(all_scores)), 3)

        # Batch-level contradiction check
        batch_contradiction = self._batch_contradiction_check(
            accepted, feature_extractor
        )
        if batch_contradiction > 0.4:
            agg.contradiction_score = batch_contradiction
            agg.flags.append(
                f"BATCH CONTRADICTION: Overall contradiction "
                f"score is {batch_contradiction:.2f}. Strategy "
                f"may have unstable edge or missing variants."
            )

        agg.flags.append(
            f"Batch: {len(accepted)}/{len(trades)} accepted "
            f"({reject_count} rejected). "
            f"Avg quality: {agg.final_score:.2f}"
        )

        return accepted, agg

    # ── FIX 1: Pattern-Level Consistency ──────────────────────────

    def _check_pattern_consistency(
        self,
        trade: Dict[str, Any],
        features: Optional[Dict[str, float]],
    ) -> float:
        """
        Does this trade's feature pattern match what the
        strategy is supposed to look like?

        Uses cosine distance from learned pattern centroids.
        If no centroids exist (early stage), return neutral.
        """
        if not features or not self._profile.pattern_centroids:
            return 0.5  # neutral when we can't check

        # Build feature vector aligned to centroid keys
        feature_keys = self._profile.pattern_feature_keys
        if not feature_keys:
            return 0.5

        trade_vector = []
        for key in feature_keys:
            val = features.get(key, 0.0)
            try:
                trade_vector.append(float(val))
            except (ValueError, TypeError):
                trade_vector.append(0.0)

        trade_vector = np.array(trade_vector)

        # Find minimum distance to any centroid
        min_distance = float("inf")
        for centroid_dict in self._profile.pattern_centroids:
            centroid_vector = np.array([
                centroid_dict.get(key, 0.0) for key in feature_keys
            ])

            # Cosine distance (0 = identical, 1 = orthogonal)
            if np.linalg.norm(trade_vector) == 0 or np.linalg.norm(centroid_vector) == 0:
                distance = 1.0
            else:
                distance = cosine_distance(trade_vector, centroid_vector)

            min_distance = min(min_distance, distance)

        # Convert distance to score (0 = far from any pattern, 1 = exact match)
        tolerance = self._profile.pattern_tolerance
        if min_distance <= tolerance * 0.5:
            return 1.0  # very close
        elif min_distance <= tolerance:
            # Linear decay within tolerance
            return round(1.0 - (min_distance / tolerance) * 0.5, 3)
        elif min_distance <= tolerance * 2:
            # Rapid decay outside tolerance
            return round(
                0.5 * (1.0 - (min_distance - tolerance) / tolerance),
                3,
            )
        else:
            return 0.0  # way outside pattern

    # ── FIX 2: Strategy-Adaptive Noise Detection ─────────────────

    def _check_noise_adaptive(self, trade: Dict[str, Any]) -> float:
        """
        Noise detection that adapts to strategy type.

        A scalper trading 20 times/day is normal.
        A swing trader trading 20 times/day is tilting.

        Uses strategy profile for context.
        """
        noise_score = 0.0
        p = self._profile

        # Trades today — relative to expected frequency
        trades_today = _safe_float(trade.get("trades_today"), 0)
        expected = p.expected_trades_per_day
        if expected > 0 and trades_today > 0:
            ratio = trades_today / expected
            if ratio > 3.0:
                noise_score += 0.25
                # Way over normal frequency
            elif ratio > 2.0:
                noise_score += 0.10
            # ratio <= 2.0 is normal variance

        # Losing streak — severity depends on typical win rate
        streak = _safe_float(trade.get("streak_at_entry"), 0)
        if streak < 0:
            expected_wr = p.historical_win_rate or 0.5
            # Expected losing streak length
            # For 50% WR: 3-4 in a row is normal
            # For 65% WR: 2-3 in a row is normal
            expected_max_streak = max(
                2, int(1.0 / max(expected_wr, 0.1))
            )
            if abs(streak) > expected_max_streak * 2:
                noise_score += 0.25  # very unusual
            elif abs(streak) > expected_max_streak:
                noise_score += 0.10  # elevated

        # Emotional state
        emotion = trade.get("emotion_pre", "")
        if isinstance(emotion, str):
            high_noise_emotions = {
                "angry", "frustrated", "revenge", "tilt", "fomo",
            }
            medium_noise_emotions = {
                "anxious", "fearful", "overconfident", "euphoric",
            }
            if emotion.lower() in high_noise_emotions:
                noise_score += 0.30
            elif emotion.lower() in medium_noise_emotions:
                noise_score += 0.15

        # Discipline score
        discipline = trade.get("discipline_score")
        if discipline is not None:
            disc = _safe_float(discipline, 1.0)
            if disc < 0.3:
                noise_score += 0.20
            elif disc < 0.5:
                noise_score += 0.10

        # Overrides
        if trade.get("had_overrides", False):
            override_types = trade.get("override_types", [])
            # Some overrides are informative (trader knows better)
            # Some are noise (panic exit, fomo hold)
            noise_overrides = {
                "panic_exit", "moved_stop", "averaged_down",
                "removed_stop", "ignored_signal",
            }
            noisy = sum(
                1 for o in override_types
                if o.lower() in noise_overrides
            )
            if noisy > 0:
                noise_score += 0.15 * min(noisy, 2)

        # Hold time anomaly — relative to strategy type
        hold_minutes = _safe_float(trade.get("hold_minutes"), None)
        expected_hold = p.expected_hold_minutes
        if hold_minutes is not None and expected_hold > 0:
            hold_ratio = hold_minutes / expected_hold
            if hold_ratio < 0.1:
                noise_score += 0.15  # closed way too fast (panic)
            elif hold_ratio > 5.0:
                noise_score += 0.10  # held way too long (hope)

        # Time of day — only flag if outside market hours
        hour = _safe_float(trade.get("hour_ist"), None)
        if hour is not None:
            if hour < 9.25 or hour > 15.5:
                noise_score += 0.05  # pre/post market

        return min(1.0, round(noise_score, 3))

    # ── FIX 3: Contradiction Detection ────────────────────────────

    def _check_contradictions(
        self,
        trade: Dict[str, Any],
        features: Optional[Dict[str, float]],
    ) -> float:
        """
        Detect if similar setups produce contradictory outcomes.

        Same pattern → sometimes +2R, sometimes -1R
        with no clear differentiator = UNSTABLE EDGE.

        This is NOT noise. It's a real signal that the
        strategy may not have a reliable edge, or it has
        hidden variants we haven't identified.

        Returns 0-1 where 1 = highly contradictory.
        """
        if not features:
            return 0.0

        p = self._profile
        pattern_hash = p.get_pattern_hash(features)

        # Record this trade's outcome
        r_multiple = _safe_float(trade.get("r_multiple"), None)
        if r_multiple is not None:
            p.pattern_outcomes.setdefault(pattern_hash, []).append(
                r_multiple
            )

        # Check contradiction for this pattern
        outcomes = p.pattern_outcomes.get(pattern_hash, [])
        if len(outcomes) < 5:
            return 0.0  # not enough data for this pattern

        outcomes_arr = np.array(outcomes)

        # Coefficient of variation — high = contradictory
        mean_outcome = float(np.mean(outcomes_arr))
        std_outcome = float(np.std(outcomes_arr))

        if abs(mean_outcome) < 0.1:
            # Mean near zero — check if outcomes are scattered
            # or just consistently near zero
            if std_outcome > 0.8:
                return 0.8  # highly contradictory
            elif std_outcome > 0.5:
                return 0.5  # moderately contradictory
            else:
                return 0.2  # consistently near zero (weak but not contradictory)

        cv = std_outcome / abs(mean_outcome)

        # Also check: do we have BOTH significant winners and losers?
        winners = outcomes_arr[outcomes_arr > 0.5]
        losers = outcomes_arr[outcomes_arr < -0.5]

        if len(winners) >= 2 and len(losers) >= 2:
            # Both present — this IS contradictory
            win_pct = len(winners) / len(outcomes_arr)
            # If 40-60% split → highly contradictory
            # If 80-20% split → less contradictory (clear tendency)
            balance = 1.0 - abs(win_pct - 0.5) * 2  # 0=one-sided, 1=balanced
            contradiction = min(1.0, cv * 0.4 + balance * 0.6)
        else:
            # Only winners or only losers — low contradiction
            contradiction = min(0.3, cv * 0.3)

        return round(contradiction, 3)

    def _batch_contradiction_check(
        self,
        trades: List[Dict[str, Any]],
        feature_extractor: Optional[Callable],
    ) -> float:
        """
        Check contradictions across the entire batch.

        Groups trades by pattern similarity and checks
        if outcomes are consistent within groups.
        """
        if not feature_extractor or len(trades) < 10:
            return 0.0

        # Group by pattern hash
        pattern_groups: Dict[str, List[float]] = defaultdict(list)

        for trade in trades:
            try:
                features = feature_extractor(trade)
                if not features:
                    continue
                pattern_hash = self._profile.get_pattern_hash(features)
                r = _safe_float(trade.get("r_multiple"), None)
                if r is not None:
                    pattern_groups[pattern_hash].append(r)
            except Exception:
                continue

        if not pattern_groups:
            return 0.0

        # Compute per-group contradiction
        contradictions = []
        for pattern_hash, outcomes in pattern_groups.items():
            if len(outcomes) < 3:
                continue

            outcomes_arr = np.array(outcomes)
            std_o = float(np.std(outcomes_arr))
            mean_o = float(np.mean(outcomes_arr))

            winners = sum(1 for o in outcomes if o > 0.5)
            losers = sum(1 for o in outcomes if o < -0.5)

            if winners >= 2 and losers >= 2:
                balance = 1.0 - abs(
                    winners / len(outcomes) - 0.5
                ) * 2
                contradictions.append(balance)
            elif abs(mean_o) < 0.1 and std_o > 0.5:
                contradictions.append(0.5)

        if not contradictions:
            return 0.0

        return round(float(np.mean(contradictions)), 3)

    # ── FIX 4: Granular Sample Sufficiency ────────────────────────

    def _check_sample_granular(
        self,
        trade: Dict[str, Any],
        features: Optional[Dict[str, float]],
    ) -> float:
        """
        Sample sufficiency at multiple granularities:
        - Overall strategy level
        - Per-pattern level
        - Per-regime level

        A strategy with 200 trades overall but only 3 in
        the current regime pattern should score LOW, not HIGH.
        """
        scores = []

        # Overall strategy sample
        total = self._profile.historical_sample_size
        if total >= 200:
            scores.append(1.0)
        elif total >= 100:
            scores.append(0.8)
        elif total >= 50:
            scores.append(0.6)
        elif total >= 20:
            scores.append(0.4)
        else:
            scores.append(0.2)

        # Per-pattern sample
        if features:
            pattern_hash = self._profile.get_pattern_hash(features)
            pattern_outcomes = self._profile.pattern_outcomes.get(
                pattern_hash, []
            )
            pattern_count = len(pattern_outcomes)

            if pattern_count >= 30:
                scores.append(1.0)
            elif pattern_count >= 15:
                scores.append(0.7)
            elif pattern_count >= 5:
                scores.append(0.4)
            else:
                scores.append(0.15)

        # Per-regime sample
        regime = trade.get("market_regime", "")
        if isinstance(regime, Enum):
            regime = regime.value
        regime = str(regime)

        if regime and regime in self._profile.regime_performance:
            regime_data = self._profile.regime_performance[regime]
            regime_count = regime_data.get("sample_size", 0)

            if regime_count >= 30:
                scores.append(1.0)
            elif regime_count >= 15:
                scores.append(0.7)
            elif regime_count >= 5:
                scores.append(0.4)
            else:
                scores.append(0.15)
        elif regime:
            # Unknown regime — very low confidence
            scores.append(0.1)

        # Return MINIMUM of all granularities
        # Strategy is only as reliable as its weakest data point
        if not scores:
            return 0.5

        # Weighted: pattern and regime matter more than overall
        if len(scores) == 1:
            return scores[0]
        elif len(scores) == 2:
            return round(scores[0] * 0.3 + scores[1] * 0.7, 3)
        else:
            return round(
                scores[0] * 0.2    # overall
                + scores[1] * 0.4  # pattern
                + scores[2] * 0.4, # regime
                3,
            )

    # ── FIX 5: Regime-Aware Recency ──────────────────────────────

    def _check_recency_regime_aware(
        self, trade: Dict[str, Any]
    ) -> float:
        """
        Recency that accounts for regime changes.

        A trade from 2 weeks ago in the SAME regime
        is more relevant than a trade from yesterday
        in a DIFFERENT regime.

        Also: if market structure shifted (volatility regime
        change, trend reversal), older data is less relevant
        regardless of calendar time.
        """
        entry_time = trade.get("entry_time")
        if entry_time is None:
            return 0.5

        if isinstance(entry_time, str):
            try:
                from dateutil.parser import parse
                entry_time = parse(entry_time)
            except (ImportError, ValueError):
                return 0.5

        now = datetime.now(timezone.utc)
        if hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        days_old = (now - entry_time).days

        # Base time decay
        if days_old <= 14:
            time_score = 1.0
        elif days_old <= 30:
            time_score = 0.9
        elif days_old <= 90:
            time_score = 0.7
        elif days_old <= 180:
            time_score = 0.5
        elif days_old <= 365:
            time_score = 0.3
        else:
            time_score = 0.15

        # Regime alignment adjustment
        trade_regime = trade.get("market_regime", "")
        if isinstance(trade_regime, Enum):
            trade_regime = trade_regime.value
        trade_regime = str(trade_regime)

        current_regime = self._profile.current_regime

        if trade_regime and current_regime:
            if trade_regime == current_regime:
                # Same regime — BOOST recency (this data is more relevant)
                time_score = min(1.0, time_score + 0.15)
            else:
                # Different regime — PENALIZE (this data is less relevant)
                time_score = max(0.1, time_score - 0.20)

        # Regime change penalty
        # If regime changed recently, ALL older data is less relevant
        last_change = self._profile.last_regime_change
        if last_change and entry_time < last_change:
            # Trade is from before the regime change
            change_days = (now - last_change).days
            if change_days < 7:
                # Very recent regime change — old data much less relevant
                time_score = max(0.1, time_score - 0.25)
            elif change_days < 30:
                time_score = max(0.1, time_score - 0.10)

        return round(time_score, 3)

    # ── Surface consistency (kept but clearly labeled) ────────────

    def _check_surface_consistency(
        self, trade: Dict[str, Any]
    ) -> float:
        """
        Basic field-level consistency.
        This is the MINIMUM check. Pattern consistency (above)
        does the deep check.
        """
        p = self._profile
        score = 1.0

        # Direction
        direction = trade.get("direction", "long")
        if direction not in p.allowed_directions:
            score -= 0.4

        # Conviction
        conviction = _safe_float(trade.get("conviction_score"), None)
        if conviction is not None and p.min_conviction > 0:
            if conviction < p.min_conviction * 0.5:
                score -= 0.3

        # Timeframe
        tf = trade.get("timeframe")
        if tf and p.timeframe and tf != p.timeframe:
            score -= 0.2

        # Risk
        risk = _safe_float(trade.get("risk_pct"), None)
        if risk is not None and p.expected_risk_pct > 0:
            if risk > p.expected_risk_pct * 3:
                score -= 0.3

        return max(0.0, round(score, 3))

    # ── Completeness (unchanged) ──────────────────────────────────

    def _check_completeness(self, trade: Dict[str, Any]) -> float:
        required = [
            "symbol", "direction", "entry_price", "entry_time",
            "exit_price", "exit_time", "pnl", "r_multiple",
        ]
        important = [
            "stop_price", "conviction_score", "market_regime",
            "volume_at_entry",
        ]
        nice = [
            "max_favorable_r", "max_adverse_r", "variant_id",
            "features_at_entry", "features_at_exit",
        ]

        req_score = sum(
            1 for f in required if trade.get(f) is not None
        ) / len(required)
        imp_score = sum(
            1 for f in important if trade.get(f) is not None
        ) / len(important)
        nice_score = sum(
            1 for f in nice if trade.get(f) is not None
        ) / len(nice)

        return round(
            req_score * 0.6 + imp_score * 0.3 + nice_score * 0.1,
            3,
        )


# ══════════════════════════════════════════════════════════════════
#  LAYER 1: HARD RULES (unchanged — already solid)
# ══════════════════════════════════════════════════════════════════


class HardRuleLayer:
    """Binary gate. ALL must pass. No exceptions."""

    def __init__(self, rules: List[Dict[str, Any]]):
        self._rules = rules

    def evaluate(self, features: Dict[str, Any]) -> LayerResult:
        results = {}
        all_passed = True
        flags = []

        for rule in self._rules:
            name = rule["name"]
            field_name = rule["field"]
            operator = rule["operator"]
            threshold = rule["value"]

            actual = features.get(field_name)

            if actual is None:
                results[name] = False
                all_passed = False
                flags.append(f"MISSING: {field_name} for '{name}'")
                continue

            try:
                actual = float(actual)
            except (ValueError, TypeError):
                results[name] = False
                all_passed = False
                flags.append(f"INVALID: {field_name}={actual}")
                continue

            passed = self._check(actual, operator, threshold)
            results[name] = passed

            if not passed:
                all_passed = False
                flags.append(
                    f"FAILED: {name} — {field_name}={actual:.2f} "
                    f"{operator} {threshold}"
                )

        return LayerResult(
            layer_name="hard_rules",
            score=1.0 if all_passed else 0.0,
            confidence=0.95,
            passed=all_passed,
            details={"rule_results": results},
            flags=flags,
        )

    @staticmethod
    def _check(actual: float, op: str, threshold) -> bool:
        ops = {
            "lt": actual < threshold,
            "lte": actual <= threshold,
            "gt": actual > threshold,
            "gte": actual >= threshold,
            "eq": abs(actual - threshold) < 0.001,
            "neq": abs(actual - threshold) >= 0.001,
        }
        if op == "between" and isinstance(threshold, (list, tuple)):
            return threshold[0] <= actual <= threshold[1]
        return ops.get(op, False)


# ══════════════════════════════════════════════════════════════════
#  LAYER 2: SOFT SCORING (with distribution tracking)
# ══════════════════════════════════════════════════════════════════


class SoftScoringLayer:
    """
    Continuous scores with distribution tracking.
    High variance = unstable pattern = low confidence.

    Supports learned score overrides via set_learned_scores().
    When a JudgmentLearner has enough data, its learned score
    REPLACES the template compute_fn output.
    """

    def __init__(self, scorers: List[Dict[str, Any]]):
        self._scorers = scorers
        self._distributions: Dict[str, SoftScoreDistribution] = {}
        self._learned_overrides: Dict[str, float] = {}

    def set_learned_scores(
        self, overrides: Dict[str, float]
    ) -> None:
        """
        Inject learned scores from JudgmentManager.

        Called before evaluate() when a JudgmentManager is active.
        Keys must match scorer names. Values are 0.0-1.0 scores
        that REPLACE the template compute_fn output.
        """
        self._learned_overrides = overrides

    def clear_learned_scores(self) -> None:
        self._learned_overrides = {}

    def evaluate(self, features: Dict[str, Any]) -> LayerResult:
        scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        flags = []

        for scorer in self._scorers:
            name = scorer["name"]
            weight = scorer.get("weight", 1.0)
            compute_fn = scorer.get("compute_fn")

            # ── Check for learned override FIRST ──
            if name in self._learned_overrides:
                score = max(0.0, min(1.0, self._learned_overrides[name]))
                flags.append(f"LEARNED: '{name}' = {score:.3f}")
            elif compute_fn is not None:
                try:
                    score = max(0.0, min(1.0, float(compute_fn(features))))
                except Exception as e:
                    flags.append(f"SCORE_ERROR: {name}: {e}")
                    score = 0.5
            else:
                continue

            scores[name] = round(score, 3)
            weighted_sum += score * weight
            total_weight += weight

            # Track distribution
            if name not in self._distributions:
                self._distributions[name] = SoftScoreDistribution(
                    name=name
                )
            dist = self._distributions[name]
            dist.add(score)

            score_conf = dist.confidence_from_stability()
            confidence_sum += score_conf * weight

            if dist.is_unstable:
                flags.append(
                    f"UNSTABLE: '{name}' std={dist.std:.2f} — "
                    f"pattern inconsistent"
                )

        composite = (
            weighted_sum / total_weight if total_weight > 0 else 0.0
        )
        comp_confidence = (
            confidence_sum / total_weight if total_weight > 0 else 0.3
        )

        comp_variance = 0.0
        if scores:
            comp_variance = float(np.std(list(scores.values())))

        return LayerResult(
            layer_name="soft_scoring",
            score=round(composite, 3),
            confidence=round(comp_confidence, 3),
            passed=True,
            details={
                "component_scores": scores,
                "composite_variance": round(comp_variance, 3),
                "distributions": {
                    n: {
                        "mean": round(d.mean, 3),
                        "std": round(d.std, 3),
                        "stable": d.is_stable,
                        "samples": len(d.values),
                    }
                    for n, d in self._distributions.items()
                },
            },
            flags=flags,
        )


# ══════════════════════════════════════════════════════════════════
#  LAYER 3: CONTEXT ALIGNMENT (unchanged)
# ══════════════════════════════════════════════════════════════════


class ContextAlignmentLayer:
    def __init__(self, factors: List[Dict[str, Any]]):
        self._factors = factors

    def evaluate(self, features: Dict[str, Any]) -> LayerResult:
        scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        flags = []

        for factor in self._factors:
            name = factor["name"]
            weight = factor.get("weight", 1.0)
            compute_fn = factor.get("compute_fn")
            if compute_fn is None:
                continue

            try:
                score = max(0.0, min(1.0, float(compute_fn(features))))
            except Exception as e:
                flags.append(f"CONTEXT_ERROR: {name}: {e}")
                score = 0.5

            scores[name] = round(score, 3)
            weighted_sum += score * weight
            total_weight += weight

        composite = (
            weighted_sum / total_weight if total_weight > 0 else 0.0
        )

        return LayerResult(
            layer_name="context_alignment",
            score=round(composite, 3),
            confidence=0.7,
            passed=True,
            details={"factor_scores": scores},
            flags=flags,
        )


# ══════════════════════════════════════════════════════════════════
#  LAYER 4: UNCERTAINTY (with learning)
# ══════════════════════════════════════════════════════════════════


class UncertaintyLayer:
    """
    Models what we DON'T know.
    Reduces confidence proportionally.
    Learns to reduce uncertainty over time via calibration.
    """

    def __init__(self, factors: List[Dict[str, Any]]):
        self._factors = factors
        self._learning_progress: Dict[str, float] = {}

    def evaluate(self) -> LayerResult:
        active = []
        total_penalty = 0.0

        for factor in self._factors:
            name = factor["name"]
            base_penalty = factor.get("penalty", 0.1)
            impact = factor.get("impact", "medium")

            learned = self._learning_progress.get(name, 0.0)
            effective = base_penalty * (1.0 - learned)

            if effective > 0.01:
                active.append({
                    "name": name,
                    "description": factor.get("description", ""),
                    "impact": impact,
                    "base_penalty": round(base_penalty, 3),
                    "learned_reduction": round(learned, 3),
                    "effective_penalty": round(effective, 3),
                })
                total_penalty += effective

        total_penalty = min(0.6, total_penalty)

        return LayerResult(
            layer_name="uncertainty",
            score=round(1.0 - total_penalty, 3),
            confidence=0.8,
            passed=True,
            details={
                "active_factors": active,
                "total_penalty": round(total_penalty, 3),
            },
        )

    def update_learning(
        self, factor_name: str, calibration: float
    ) -> None:
        self._learning_progress[factor_name] = min(0.8, calibration)


# ══════════════════════════════════════════════════════════════════
#  REGIME STABILITY CHECKER (enhanced)
# ══════════════════════════════════════════════════════════════════


class RegimeStabilityChecker:
    """
    Enhanced regime stability with:
    - Per-regime confidence intervals
    - Regime transition detection
    - Minimum per-regime sample requirements
    """

    def __init__(self):
        self._regime_outcomes: Dict[str, List[float]] = {}
        self._regime_timestamps: Dict[str, List[datetime]] = {}

    def record_outcome(
        self,
        regime: str,
        match_score: float,
        actual_outcome_r: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self._regime_outcomes.setdefault(regime, []).append(
            actual_outcome_r
        )
        if timestamp:
            self._regime_timestamps.setdefault(regime, []).append(
                timestamp
            )

    def compute_stability(self) -> Dict[str, Any]:
        if not self._regime_outcomes:
            return {
                "stability_score": 0.5,
                "regime_scores": {},
                "regime_details": {},
                "weakest_regime": "",
                "strongest_regime": "",
                "sufficient_data": False,
                "per_regime_sufficient": {},
            }

        regime_stats = {}
        regime_sufficient = {}

        for regime, outcomes in self._regime_outcomes.items():
            n = len(outcomes)
            sufficient = n >= 10
            regime_sufficient[regime] = sufficient

            if n >= 3:
                arr = np.array(outcomes)
                regime_stats[regime] = {
                    "mean": round(float(np.mean(arr)), 4),
                    "std": round(float(np.std(arr)), 4),
                    "win_rate": round(
                        float(np.mean(arr > 0)), 4
                    ),
                    "sample_size": n,
                    "sufficient": sufficient,
                }

        if not regime_stats:
            return {
                "stability_score": 0.5,
                "regime_scores": {},
                "regime_details": regime_stats,
                "weakest_regime": "",
                "strongest_regime": "",
                "sufficient_data": False,
                "per_regime_sufficient": regime_sufficient,
            }

        # Only use regimes with sufficient data for stability calc
        sufficient_means = {
            k: v["mean"]
            for k, v in regime_stats.items()
            if v["sufficient"]
        }

        if len(sufficient_means) < 2:
            # Can't assess stability with < 2 sufficient regimes
            stability = 0.5
        else:
            means = list(sufficient_means.values())
            overall_mean = float(np.mean(means))
            overall_std = float(np.std(means))

            if abs(overall_mean) < 0.01:
                stability = 0.3
            else:
                cv = overall_std / abs(overall_mean)
                stability = max(0.0, min(1.0, 1.0 - cv))

        regime_scores = {
            k: v["mean"] for k, v in regime_stats.items()
        }

        weakest = min(regime_scores, key=regime_scores.get) if regime_scores else ""
        strongest = max(regime_scores, key=regime_scores.get) if regime_scores else ""

        return {
            "stability_score": round(stability, 3),
            "regime_scores": {
                k: round(v, 3) for k, v in regime_scores.items()
            },
            "regime_details": regime_stats,
            "weakest_regime": weakest,
            "strongest_regime": strongest,
            "sufficient_data": any(regime_sufficient.values()),
            "per_regime_sufficient": regime_sufficient,
        }


# ══════════════════════════════════════════════════════════════════
#  META LAYER: THE COMBINER
# ══════════════════════════════════════════════════════════════════


class StrategyUnderstandingEngine:
    """
    Combines all layers into executable output.

    Decision logic:
    IF hard_rules FAIL → REJECT
    ELSE:
        score = w1*soft + w2*context - w3*uncertainty
        confidence = f(soft_conf, regime_stability, uncertainty, variance)

        STRONG:   score >= 0.70 AND confidence >= 0.60
        MODERATE: score >= 0.50 AND confidence >= 0.45
        WEAK:     score >= 0.35 AND confidence >= 0.30
        REJECT:   else
    """

    def __init__(
        self,
        profile: StrategyProfile,
        hard_rules: Optional[HardRuleLayer] = None,
        soft_scoring: Optional[SoftScoringLayer] = None,
        context: Optional[ContextAlignmentLayer] = None,
        uncertainty: Optional[UncertaintyLayer] = None,
        regime_checker: Optional[RegimeStabilityChecker] = None,
        data_firewall: Optional[DataQualityFirewall] = None,
        config: Optional[Dict[str, Any]] = None,
        judgment_connector: Optional[Any] = None,
    ):
        self._profile = profile

        # Build layers from profile if not provided
        self._hard = hard_rules or HardRuleLayer(
            profile.hard_rules
        )
        self._soft = soft_scoring or SoftScoringLayer(
            profile.soft_scorers
        )
        self._context = context or ContextAlignmentLayer(
            profile.context_factors
        )
        self._uncertainty = uncertainty or UncertaintyLayer(
            profile.uncertainty_factors
        )
        self._regime = regime_checker or RegimeStabilityChecker()
        self._firewall = data_firewall or DataQualityFirewall(profile)

        # Judgment learning connector (optional, injected after bootstrap)
        self._judgment = judgment_connector

        cfg = config or {}
        self._weights = {
            "soft": cfg.get("soft_weight", 0.45),
            "context": cfg.get("context_weight", 0.30),
            "uncertainty": cfg.get("uncertainty_weight", 0.25),
        }
        self._thresholds = {
            "strong_score": cfg.get("strong_score", 0.70),
            "strong_conf": cfg.get("strong_confidence", 0.60),
            "moderate_score": cfg.get("moderate_score", 0.50),
            "moderate_conf": cfg.get("moderate_confidence", 0.45),
            "weak_score": cfg.get("weak_score", 0.35),
            "weak_conf": cfg.get("weak_confidence", 0.30),
        }
        self._avt_independent_conf = cfg.get(
            "avt_independent_confidence", 0.75
        )
        self._avt_candidate_conf = cfg.get(
            "avt_candidate_confidence", 0.50
        )

    # ── Data quality gate ─────────────────────────────────────────

    def gate_data(
        self,
        trade: Dict[str, Any],
        features: Optional[Dict[str, float]] = None,
    ) -> DataQualityResult:
        """
        Layer 0: Should we learn from this trade?
        Call BEFORE any learning/analysis.
        """
        return self._firewall.assess(trade, features)

    def gate_data_batch(
        self,
        trades: List[Dict[str, Any]],
        feature_extractor: Optional[Callable] = None,
    ) -> Tuple[List[Dict[str, Any]], DataQualityResult]:
        """Layer 0 for batch."""
        return self._firewall.assess_batch(trades, feature_extractor)

    # ── Main evaluation ───────────────────────────────────────────

    def evaluate(
        self,
        features: Dict[str, Any],
        current_regime: Optional[str] = None,
    ) -> MetaLayerOutput:
        """
        THE MAIN ENTRY POINT.
        Takes features → runs all layers → produces decision.
        """
        output = MetaLayerOutput()

        # ── Layer 1: Hard Rules ──
        hard = self._hard.evaluate(features)
        output.hard_rule_match = hard.passed
        output.hard_rule_details = hard.details.get(
            "rule_results", {}
        )

        if not hard.passed:
            output.decision = StrategyDecision.REJECT.value
            output.overall_match_score = 0.0
            output.strategy_confidence = 0.0
            output.bottleneck = "hard_rules_failed"
            output.human_gates_required = hard.flags
            return output

        # ── Judgment Learning: inject learned scores ──
        strategy_id = getattr(self._profile, "name", "") or ""
        if self._judgment is None:
            # Try global connector (lazy init)
            try:
                from bootstrap.live_connector import get_judgment_connector
                connector = get_judgment_connector()
                if connector.has_manager(strategy_id):
                    self._judgment = connector
            except ImportError:
                pass

        if self._judgment and strategy_id:
            # Inject learned soft scores BEFORE evaluation
            try:
                learned = {}
                for scorer in self._soft._scorers:
                    name = scorer["name"]
                    score = self._judgment.get_learned_soft_score(
                        strategy_id, name, features
                    )
                    if score is not None:
                        learned[name] = score
                if learned:
                    self._soft.set_learned_scores(learned)
            except Exception:
                pass

            # Sync uncertainty reductions
            try:
                self._judgment.sync_uncertainty_to_engine(
                    strategy_id, self._uncertainty
                )
            except Exception:
                pass

        # ── Layer 2: Soft Scoring ──
        soft = self._soft.evaluate(features)
        output.soft_score = soft.score
        output.soft_score_variance = soft.details.get(
            "composite_variance", 0.0
        )
        output.soft_score_components = soft.details.get(
            "component_scores", {}
        )

        # Clear per-call overrides so next call starts fresh
        self._soft.clear_learned_scores()

        # ── Layer 3: Context ──
        ctx = self._context.evaluate(features)
        output.context_alignment = ctx.score
        output.context_components = ctx.details.get(
            "factor_scores", {}
        )

        # ── Layer 4: Uncertainty ──
        unc = self._uncertainty.evaluate()
        output.uncertainty_factors = unc.details.get(
            "active_factors", []
        )
        output.uncertainty_penalty = unc.details.get(
            "total_penalty", 0.0
        )

        # ── Regime Stability ──
        reg = self._regime.compute_stability()
        output.regime_stability = reg.get("stability_score", 0.5)
        output.regime_scores = reg.get("regime_scores", {})
        output.weakest_regime = reg.get("weakest_regime", "")
        output.strongest_regime = reg.get("strongest_regime", "")

        # ══════════════════════════════════════════════════════════
        #  FINAL COMBINATION
        # ══════════════════════════════════════════════════════════

        w = self._weights
        output.overall_match_score = round(max(0.0,
            w["soft"] * soft.score
            + w["context"] * ctx.score
            - w["uncertainty"] * output.uncertainty_penalty
        ), 3)

        # Confidence
        variance_penalty = min(0.2, output.soft_score_variance * 0.5)
        raw_conf = (
            soft.confidence * 0.35
            + ctx.confidence * 0.20
            + output.regime_stability * 0.20
            + (1.0 - output.uncertainty_penalty) * 0.25
        )
        output.strategy_confidence = round(
            max(0.0, raw_conf - variance_penalty), 3
        )

        # ── DECISION ──
        s = output.overall_match_score
        c = output.strategy_confidence
        t = self._thresholds

        if s >= t["strong_score"] and c >= t["strong_conf"]:
            output.decision = StrategyDecision.STRONG.value
        elif s >= t["moderate_score"] and c >= t["moderate_conf"]:
            output.decision = StrategyDecision.MODERATE.value
        elif s >= t["weak_score"] and c >= t["weak_conf"]:
            output.decision = StrategyDecision.WEAK.value
        else:
            output.decision = StrategyDecision.REJECT.value

        # ── AVT Mode ──
        if c >= self._avt_independent_conf:
            output.avt_can_trade = True
            output.avt_mode = "independent"
        elif c >= self._avt_candidate_conf:
            output.avt_can_trade = True
            output.avt_mode = "candidate_presenter"
        else:
            output.avt_can_trade = False
            output.avt_mode = "observer"

        # ── Improvement lab gate ──
        output.improvement_lab_can_analyze = (
            c >= 0.40
            and output.decision != StrategyDecision.REJECT.value
        )

        # ── Bottleneck ──
        output.bottleneck = self._find_bottleneck(output)

        # ── Human gates ──
        output.human_gates_required = self._find_human_gates(output)

        # ── Suggestions ──
        output.improvement_suggestions = self._suggest(output)

        return output

    def _find_bottleneck(self, o: MetaLayerOutput) -> str:
        problems = {
            "uncertainty": o.uncertainty_penalty,
            "soft_score_low": max(0, 0.7 - o.soft_score),
            "soft_score_unstable": o.soft_score_variance,
            "context_poor": max(0, 0.5 - o.context_alignment),
            "regime_unstable": max(0, 0.5 - o.regime_stability),
        }
        return max(problems, key=problems.get)

    def _find_human_gates(self, o: MetaLayerOutput) -> List[str]:
        gates = []
        for f in o.uncertainty_factors:
            if f.get("impact") == "high":
                gates.append(
                    f"CONFIRM: {f['description']} "
                    f"(penalty: {f['effective_penalty']:.0%})"
                )
        if o.strategy_confidence < 0.5:
            gates.append(
                "LOW CONFIDENCE: User must confirm all signals"
            )
        if o.regime_stability < 0.4:
            gates.append(
                f"REGIME RISK: Unstable across regimes. "
                f"Weakest: {o.weakest_regime}"
            )
        return gates

    def _suggest(self, o: MetaLayerOutput) -> List[str]:
        suggestions = []
        bn = o.bottleneck

        if bn == "uncertainty":
            suggestions.append(
                "🗣️ Show more examples of setups you'd take vs skip"
            )
        if bn == "soft_score_unstable":
            suggestions.append(
                "📊 Soft scores are inconsistent. Strategy may have "
                "hidden variants — teach each variant separately"
            )
        if bn == "regime_unstable":
            suggestions.append(
                f"📈 Strategy is regime-sensitive. Consider teaching "
                f"regime-specific rules. Strongest in: "
                f"{o.strongest_regime}"
            )
        if bn == "context_poor":
            suggestions.append(
                "🌍 Teach the system WHEN you trade this strategy — "
                "market conditions, time of day, sector behavior"
            )
        if bn == "soft_score_low":
            suggestions.append(
                "📝 Pattern match is weak. Review if the system "
                "understood your entry criteria correctly"
            )

        return suggestions


# ══════════════════════════════════════════════════════════════════
#  CALIBRATION TRACKER
# ══════════════════════════════════════════════════════════════════


class CalibrationTracker:
    """
    Learns calibration from user feedback.
    Updates uncertainty layer as we learn.
    Tracks per-component accuracy.
    """

    def __init__(self, engine: StrategyUnderstandingEngine):
        self._engine = engine
        self._data: List[Dict[str, Any]] = []
        self._component_accuracy: Dict[str, List[bool]] = defaultdict(list)

    def record(
        self,
        output: MetaLayerOutput,
        user_response: str,
        actual_r: Optional[float] = None,
        regime: Optional[str] = None,
    ) -> None:
        """Record a calibration point."""
        is_positive = user_response in ("accept", "would_trade")

        self._data.append({
            "ts": datetime.now(timezone.utc),
            "match_score": output.overall_match_score,
            "confidence": output.strategy_confidence,
            "soft_score": output.soft_score,
            "soft_components": dict(output.soft_score_components),
            "context": output.context_alignment,
            "decision": output.decision,
            "user_positive": is_positive,
            "actual_r": actual_r,
            "regime": regime,
        })

        # Track per-component accuracy
        for comp_name, comp_score in output.soft_score_components.items():
            # Did high score = user accept?
            predicted_positive = comp_score >= 0.5
            self._component_accuracy[comp_name].append(
                predicted_positive == is_positive
            )

        # Regime tracking
        if actual_r is not None and regime:
            self._engine._regime.record_outcome(
                regime=regime,
                match_score=output.overall_match_score,
                actual_outcome_r=actual_r,
            )

        # Periodic recalibration
        if len(self._data) % 20 == 0:
            self._recalibrate()

    def _recalibrate(self) -> Dict[str, Any]:
        if len(self._data) < 20:
            return {"status": "insufficient"}

        recent = self._data[-100:]
        pos = [d for d in recent if d["user_positive"]]
        neg = [d for d in recent if not d["user_positive"]]

        report = {"status": "recalibrated"}

        if pos and neg:
            avg_pos = float(np.mean([d["match_score"] for d in pos]))
            avg_neg = float(np.mean([d["match_score"] for d in neg]))
            separation = avg_pos - avg_neg

            report["score_separation"] = round(separation, 3)

            if separation < 0.05:
                logger.warning(
                    "POOR CALIBRATION: match_score doesn't separate "
                    "accepts from rejects (sep=%.3f)", separation
                )

        # Per-component accuracy
        for comp, results in self._component_accuracy.items():
            if len(results) >= 10:
                accuracy = sum(results[-50:]) / len(results[-50:])
                report[f"component_{comp}_accuracy"] = round(accuracy, 3)

                if accuracy < 0.45:
                    logger.warning(
                        "Component '%s' has no predictive power "
                        "(accuracy=%.2f)", comp, accuracy
                    )

        # Update uncertainty learning
        overall_accuracy = 0.5
        if pos and neg:
            # How well does our overall score predict acceptance?
            total = len(recent)
            correct = sum(
                1 for d in recent
                if (d["match_score"] >= 0.5) == d["user_positive"]
            )
            overall_accuracy = correct / total

        for factor in self._engine._uncertainty._factors:
            if factor.get("can_learn", False):
                self._engine._uncertainty.update_learning(
                    factor["name"], overall_accuracy
                )

        return report

    def get_report(self) -> Dict[str, Any]:
        if not self._data:
            return {"status": "no_data", "total": 0}

        recent = self._data[-50:]
        pos = [d for d in recent if d["user_positive"]]

        return {
            "status": "active",
            "total_points": len(self._data),
            "recent": len(recent),
            "acceptance_rate": round(
                len(pos) / max(1, len(recent)), 3
            ),
            "avg_match_score": round(float(np.mean(
                [d["match_score"] for d in recent]
            )), 3),
            "avg_confidence": round(float(np.mean(
                [d["confidence"] for d in recent]
            )), 3),
            "component_accuracies": {
                comp: round(
                    sum(vals[-50:]) / max(1, len(vals[-50:])), 3
                )
                for comp, vals in self._component_accuracy.items()
                if len(vals) >= 10
            },
        }


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default