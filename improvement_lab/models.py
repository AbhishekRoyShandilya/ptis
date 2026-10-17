# improvement_lab/models.py
"""
Improvement Lab Models
=======================

All enums, dataclasses, and types for the improvement lab.

The improvement lab is AVT's internal analysis engine.
It discovers hypotheses from AVT's clean trade data,
which are then proven/disproven by AVT's virtual trading.

The user NEVER sees unproven hypotheses.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ══════════════════════════════════════════════════════════════════
#  ENUMS
# ══════════════════════════════════════════════════════════════════

class HypothesisSource(str, Enum):
    """Where did this hypothesis come from?"""
    AVT = "avt"                # AVT's own trade analysis (clean data)
    USER_BEHAVIORAL = "user"   # User vs AVT comparison (for coaching only)


class ProofStatus(str, Enum):
    """Has this improvement been PROVEN by AVT's live virtual trading?"""
    UNPROVEN = "unproven"
    REJECTED_TRADEOFF = "rejected_tradeoff"
    APPLIED_TO_AVT = "applied"
    PROVING = "proving"
    PROVEN = "proven"
    DISPROVEN = "disproven"
    REVERTED = "reverted"
    PRESENTED_TO_USER = "presented"
    ADOPTED_BY_USER = "adopted"
    USER_REJECTED = "user_rejected"


class AnalysisLayer(str, Enum):
    """Ordered analysis layers."""
    REGIME = "regime"
    VARIANT = "variant"
    BEHAVIORAL = "behavioral"
    PATTERN = "pattern"


class ImprovementType(str, Enum):
    """Types of improvements the lab can discover."""
    RR_KILLER_FIX = "rr_killer_fix"
    EARLY_EXIT_FILTER = "early_exit_filter"
    ENTRY_FILTER = "entry_filter"
    EXIT_TIMING = "exit_timing"
    SIZE_ADJUSTMENT = "size_adjustment"
    VARIANT_AVOIDANCE = "variant_avoidance"
    VARIANT_REFINEMENT = "variant_refinement"
    REGIME_FILTER = "regime_filter"


class PatternType(str, Enum):
    RULE_BASED = "rule_based"
    FAST_LAYER = "fast_layer"
    DEEP_LAYER = "deep_layer"
    HYBRID = "hybrid"


class DetectionMethod(str, Enum):
    CONDITIONAL_FREQUENCY = "conditional_frequency"
    FEATURE_IMPORTANCE = "feature_importance"
    SHAP = "shap"
    RULE_BASED = "rule_based"


class ExperimentStatus(str, Enum):
    DETECTED = "detected"
    SHADOW_TESTING = "shadow_testing"
    SHADOW_PASSED = "shadow_passed"
    SHADOW_FAILED = "shadow_failed"
    INSIGHT_PRESENTED = "insight_presented"
    SUGGESTING = "suggesting"
    ACCEPTANCE_REVIEW = "acceptance_review"
    GRADUATED = "graduated"
    MONITORING = "monitoring"
    DEGRADING = "degrading"
    RETIRED = "retired"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class BehavioralClassification(str, Enum):
    MARKET_PATTERN = "market_pattern"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    MIXED = "mixed"


# ══════════════════════════════════════════════════════════════════
#  DATACLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class PatternClassification:
    """Whether a detected pattern is market-driven or behavioral."""
    classification: BehavioralClassification = BehavioralClassification.MARKET_PATTERN
    market_pct: float = 1.0
    behavioral_pct: float = 0.0
    action: str = ""  # "strategy_change" | "coaching"
    note: str = ""


@dataclass
class VariantConcentration:
    """Whether a pattern is concentrated in specific variants."""
    is_concentrated: bool = False
    primary_variant_id: Optional[str] = None
    primary_variant_label: str = ""
    variant_type: str = ""  # "anti_variant" | "marginal_variant" | "spread"
    concentration_pct: float = 0.0
    action: str = ""  # "stop_variant" | "variant_specific_fix" | "strategy_wide"


@dataclass
class TradeoffAnalysis:
    """Net impact analysis for a hypothesis."""
    trades_improved: int = 0
    trades_hurt: int = 0
    trades_neutral: int = 0
    net_expectancy_change_r: float = 0.0
    net_expectancy_change_pct: float = 0.0
    win_rate_change: float = 0.0
    runner_impact: float = 0.0        # impact on big winners
    probability_net_positive: float = 0.0
    recommendation: str = ""
    alternatives: List[TradeoffAlternative] = field(default_factory=list)


@dataclass
class TradeoffAlternative:
    """Alternative action for a tradeoff."""
    description: str = ""
    net_r: float = 0.0
    runner_cost_pct: float = 0.0


@dataclass
class CompositionResult:
    """Result of testing multiple improvements together."""
    recommended_combination: List[str] = field(default_factory=list)
    combined_net_r: float = 0.0
    interactions: List[CompositionInteraction] = field(default_factory=list)
    explanation: str = ""


@dataclass
class CompositionInteraction:
    """Interaction between two improvements."""
    improvement_a_id: str = ""
    improvement_b_id: str = ""
    interaction_type: str = ""  # "synergistic" | "antagonistic" | "independent"
    interaction_r: float = 0.0


@dataclass
class ValidationProgress:
    """Progress of a shadow experiment."""
    experiment_id: str = ""
    strategy_id: str = ""
    hypothesis_id: str = ""
    status: str = ExperimentStatus.DETECTED.value
    shadow_trades_count: int = 0
    trades_needed: int = 50
    consistency_rate: float = 0.0
    avg_r_saved: float = 0.0
    false_positive_rate: float = 0.0
    ready_to_graduate: bool = False


# ══════════════════════════════════════════════════════════════════
#  IMPROVEMENT HYPOTHESIS (core model)
# ══════════════════════════════════════════════════════════════════

@dataclass
class ImprovementHypothesis:
    """
    Improvement hypothesis discovered by AVT's improvement lab.

    LIFECYCLE:
    1. Lab discovers hypothesis from AVT's clean data
    2. Tradeoff analysis validates net positive expectancy
    3. AVT applies to its own trading (version bump)
    4. AVT trades with improvement for 50+ trades
    5. If proven (positive delta) → present to user with evidence
    6. If disproven (negative delta) → revert, never show user

    The user NEVER sees unproven hypotheses.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = ""
    improvement_type: str = ""
    analysis_layer: str = ""
    pattern_type: str = "rule_based"
    detection_method: str = ""
    title: str = ""
    description: str = ""
    expected_impact_r: float = 0.0
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""
    proposed_rule: Dict[str, Any] = field(default_factory=dict)
    shap_explanation: Optional[Dict[str, float]] = None

    # Source
    source: str = HypothesisSource.AVT.value

    # Variant awareness
    variant_id: Optional[str] = None
    variant_concentration: Optional[VariantConcentration] = None
    applies_to: str = "all_variants"

    # Behavioral classification (only used when analyzing user trades)
    behavioral_classification: Optional[PatternClassification] = None

    # Tradeoff (required before AVT applies)
    tradeoff: Optional[TradeoffAnalysis] = None

    # Explained trade IDs
    explained_trade_ids: List[str] = field(default_factory=list)

    # Old status field (for backward compat with existing orchestrator)
    status: str = ExperimentStatus.DETECTED.value

    # ── AVT Proof Lifecycle ──
    proof_status: str = ProofStatus.UNPROVEN.value

    # When AVT applied this improvement
    applied_to_avt_at: Optional[datetime] = None
    applied_at_version: Optional[str] = None

    # AVT's proof data (filled during proving period)
    proof_trades_count: int = 0
    proof_trades_needed: int = 50
    proof_start_date: Optional[datetime] = None
    proof_performance_before: Optional[Dict[str, float]] = None
    proof_performance_after: Optional[Dict[str, float]] = None
    proof_delta_r: Optional[float] = None
    proof_p_value: Optional[float] = None
    proof_example_trades: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # Revert tracking
    reverted_at: Optional[datetime] = None
    revert_reason: Optional[str] = None

    # User presentation (only if proven)
    presented_to_user_at: Optional[datetime] = None
    user_response: Optional[str] = None
    user_rejection_reason: Optional[str] = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ImprovementCandidate = pre-proof hypothesis (same shape, different name used by entry_filter_learner)
ImprovementCandidate = ImprovementHypothesis


# ══════════════════════════════════════════════════════════════════
#  TRADE FEATURE VECTOR
# ══════════════════════════════════════════════════════════════════

@dataclass
class TradeFeatureVector:
    """Feature vector extracted from a single trade for ML analysis."""
    trade_id: str = ""
    strategy_id: str = ""
    variant_id: Optional[str] = None
    features: Dict[str, float] = field(default_factory=dict)
    hit_sl: Optional[bool] = None
    was_winner: Optional[bool] = None
    final_r: Optional[float] = None
    had_overrides: bool = False
    override_types: List[str] = field(default_factory=list)
    shadow_final_r: Optional[float] = None
    shadow_max_favorable_r: Optional[float] = None


# ══════════════════════════════════════════════════════════════════
#  BAYESIAN CELL + TWO-SPEED STATE
# ══════════════════════════════════════════════════════════════════

@dataclass
class BayesianCell:
    """Mutable win/loss counter for a single Bayesian conditional table cell."""
    wins: int = 0
    losses: int = 0

    def update(self, outcome: bool) -> None:
        if outcome:
            self.wins += 1
        else:
            self.losses += 1

    @property
    def n(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        if self.n == 0:
            return 0.5  # prior
        return self.wins / self.n


@dataclass
class TwoSpeedState:
    """Serialisable snapshot of a TwoSpeedLearner's internal state."""
    fast_trade_count: int = 0
    deep_trade_count: int = 0
    fast_accuracy: float = 0.0
    deep_accuracy: float = 0.0
    ensemble_weight_fast: float = 0.7
    ensemble_weight_deep: float = 0.3


# ══════════════════════════════════════════════════════════════════
#  ALTERNATIVE IMPROVEMENT + INTERACTION EFFECT
# ══════════════════════════════════════════════════════════════════

@dataclass
class AlternativeImprovement:
    """A partial-exit alternative considered during tradeoff simulation."""
    description: str = ""
    net_r: float = 0.0
    runner_cost_pct: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionEffect:
    """Synergy / antagonism between two improvements applied together."""
    improvement_a_id: str = ""
    improvement_b_id: str = ""
    individual_sum_r: float = 0.0
    combined_r: float = 0.0
    interaction_r: float = 0.0
    interaction_type: str = ""   # "synergistic" or "antagonistic"
    note: str = ""


# ══════════════════════════════════════════════════════════════════
#  PATTERN TYPES (EarlyExitPattern, RRKillerPattern)
# ══════════════════════════════════════════════════════════════════

@dataclass
class EarlyExitPattern:
    """Condition fingerprint where SL hit rate is abnormally high."""
    strategy_id: str = ""
    condition_fingerprint: Dict[str, Any] = field(default_factory=dict)
    sl_hit_rate: float = 0.0
    avg_adverse_r: float = 0.0
    sample_size: int = 0
    confidence: float = 0.0
    trade_ids: List[str] = field(default_factory=list)
    suggested_fix: str = ""
    detection_method: str = ""
    shap_explanation: Optional[Dict[str, float]] = None


@dataclass
class RRKillerPattern:
    """Conditions where trades reverse after reaching a favorable R threshold."""
    strategy_id: str = ""
    trigger_r: float = 0.0
    reversal_rate: float = 0.0
    avg_r_given_back: float = 0.0
    condition_fingerprint: Dict[str, Any] = field(default_factory=dict)
    sample_size: int = 0
    confidence: float = 0.0
    trade_ids: List[str] = field(default_factory=list)
    suggested_fix: str = ""
    detection_method: str = ""
    shap_explanation: Optional[Dict[str, float]] = None


# ══════════════════════════════════════════════════════════════════
#  ENTRY FILTER
# ══════════════════════════════════════════════════════════════════

@dataclass
class EntryFilter:
    """A learned filter that identifies conditions where entry should be skipped."""
    strategy_id: str = ""
    filter_type: str = ""                    # axis name or "composite"
    filter_condition: Dict[str, Any] = field(default_factory=dict)
    trades_matching: int = 0
    win_rate_when_active: float = 0.0
    baseline_win_rate: float = 0.0
    lift: float = 0.0                        # negative = bad condition
    confidence: float = 0.0
    detection_method: str = ""
    shap_explanation: Optional[str] = None   # set by ML path


# ══════════════════════════════════════════════════════════════════
#  IMPROVEMENT LAB REPORT
# ══════════════════════════════════════════════════════════════════

@dataclass
class ImprovementLabReport:
    """Analysis report from AVT's improvement lab."""
    strategy_id: str = ""
    strategy_version: str = ""
    source: str = HypothesisSource.AVT.value
    total_trades: int = 0
    sufficient_data: bool = False

    # Backward compat with existing analyze() method
    quality_filtered_trades: int = 0

    analysis_mode: str = "rule_based"
    baseline_win_rate: float = 0.0
    baseline_expectancy: float = 0.0

    # Findings by layer
    regime_findings: List[ImprovementHypothesis] = field(
        default_factory=list
    )
    variant_findings: List[ImprovementHypothesis] = field(
        default_factory=list
    )
    pattern_findings: List[ImprovementHypothesis] = field(
        default_factory=list
    )
    behavioral_findings: List[ImprovementHypothesis] = field(
        default_factory=list
    )

    # All candidates (used by existing analyze() for backward compat)
    candidates: List[ImprovementHypothesis] = field(
        default_factory=list
    )

    # Filtered results (used by analyze_avt_trades)
    actionable_hypotheses: List[ImprovementHypothesis] = field(
        default_factory=list
    )
    rejected_hypotheses: List[ImprovementHypothesis] = field(
        default_factory=list
    )

    # Composition
    composition: Optional[CompositionResult] = None

    # Validation progress (for shadow experiments in existing flow)
    validation_progress: List[ValidationProgress] = field(
        default_factory=list
    )

    # Stats
    total_explained_trades: int = 0
    total_unexplained_trades: int = 0
    total_estimated_impact_r: float = 0.0

    analyzed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
