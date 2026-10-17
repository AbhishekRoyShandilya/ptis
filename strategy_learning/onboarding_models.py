"""
============================================================
STRATEGY ONBOARDING ENGINE (SOE) — DATA MODELS
Personal Intelligence Trading System (PTIS)
============================================================
File: soe/models.py
============================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime
import uuid
import json


# ═══════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════

class ExampleLabel(Enum):
    """How user labels each example"""
    GOOD_SETUP = "good_setup"           # Setup user would take
    GOOD_WORKED = "good_worked"         # Took it, it worked
    GOOD_FAILED = "good_failed"         # Took it, it failed (still valid setup)
    BAD_SKIPPED = "bad_skipped"         # Saw it, skipped it
    BAD_WOULD_FAIL = "bad_would_fail"   # Would have been a loss
    AVOID = "avoid"                     # Red flag setup
    BORDERLINE = "borderline"           # Not sure / edge case

    @property
    def is_positive(self) -> bool:
        return self in (
            ExampleLabel.GOOD_SETUP,
            ExampleLabel.GOOD_WORKED,
            ExampleLabel.GOOD_FAILED  # Setup was valid, outcome doesn't change that
        )

    @property
    def is_negative(self) -> bool:
        return self in (
            ExampleLabel.BAD_SKIPPED,
            ExampleLabel.BAD_WOULD_FAIL,
            ExampleLabel.AVOID
        )

    @property
    def binary_label(self) -> Optional[int]:
        """1 = positive (take), 0 = negative (skip), None = ambiguous"""
        if self.is_positive:
            return 1
        elif self.is_negative:
            return 0
        return None  # BORDERLINE


class OnboardingPhase(Enum):
    NARRATIVE_INTAKE = "narrative_intake"
    EXAMPLE_COLLECTION = "example_collection"
    FEATURE_EXTRACTION = "feature_extraction"
    DATASET_BUILDING = "dataset_building"
    PATTERN_LEARNING = "pattern_learning"
    UNDERSTANDING_VALIDATION = "understanding_validation"
    SKO_GENERATION = "sko_generation"
    ADDITIONAL_EXAMPLES = "additional_examples"
    COMPLETE = "complete"


class FeatureDiscriminatorStrength(Enum):
    STRONG = "strong"       # p < 0.01, effect_size > 0.8
    MODERATE = "moderate"   # p < 0.05, effect_size > 0.5
    WEAK = "weak"           # p < 0.1, effect_size > 0.3
    NONE = "none"           # not discriminating


# ═══════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════

@dataclass
class TrainingExample:
    """Single labeled example provided by user during onboarding"""
    example_id: str = field(default_factory=lambda: f"ex_{uuid.uuid4().hex[:8]}")
    strategy_id: str = ""

    # User-provided metadata
    symbol: str = ""
    timeframe: str = ""
    date: str = ""                          # "2024-01-10"
    entry_time: Optional[str] = None        # "09:45"
    screenshot_path: Optional[str] = None

    # User labels
    label: ExampleLabel = ExampleLabel.GOOD_SETUP
    label_confidence: float = 0.8           # How sure user is (0-1)
    explanation: str = ""                   # Why this label
    outcome: Optional[str] = None           # "+2R", "-0.5R", etc.
    outcome_r: Optional[float] = None       # Numeric R-multiple

    # System-extracted
    candle_data: Optional[pd.DataFrame] = None  # Fetched OHLCV data
    features: Dict[str, float] = field(default_factory=dict)
    context_features: Dict[str, Any] = field(default_factory=dict)
    explanation_tags: List[str] = field(default_factory=list)
    explanation_feature_map: Dict[str, List[str]] = field(default_factory=dict)

    # Processing metadata
    data_quality: str = "pending"           # "good" / "partial" / "bad"
    processing_notes: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FeatureDiscriminator:
    """How well a single feature separates GOOD from BAD examples"""
    feature_name: str
    strength: FeatureDiscriminatorStrength = FeatureDiscriminatorStrength.NONE

    # Statistical measures
    good_mean: float = 0.0
    good_std: float = 0.0
    bad_mean: float = 0.0
    bad_std: float = 0.0
    effect_size: float = 0.0        # Cohen's d
    p_value: float = 1.0            # t-test p-value
    auc: float = 0.5                # ROC AUC for this feature alone

    # Threshold discovery
    optimal_threshold: Optional[float] = None
    threshold_direction: str = "below"  # "below" = good < threshold, "above" = good > threshold
    threshold_accuracy: float = 0.5

    # User language mapping
    user_terms: List[str] = field(default_factory=list)  # e.g., ["clean", "smooth"]

    def to_human_readable(self) -> str:
        direction = "lower" if self.threshold_direction == "below" else "higher"
        return (
            f"{self.feature_name}: GOOD setups have {direction} values "
            f"(good avg: {self.good_mean:.3f} vs bad avg: {self.bad_mean:.3f}, "
            f"effect size: {self.effect_size:.2f})"
        )


@dataclass
class ContextDependency:
    """Discovered dependency between context and strategy performance"""
    context_name: str               # e.g., "regime", "time_of_day"
    context_value: str              # e.g., "bearish", "morning"
    positive_count: int = 0         # GOOD examples in this context
    negative_count: int = 0         # BAD examples in this context
    total_count: int = 0
    win_rate: float = 0.0
    significance: float = 1.0       # p-value from chi-square test
    is_significant: bool = False
    direction: str = "neutral"      # "favorable" / "unfavorable" / "neutral"
    insight: str = ""               # Human-readable insight


@dataclass
class PatternDiscovery:
    """Complete results from pattern analysis of training examples"""
    strategy_id: str
    n_examples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    n_borderline: int = 0

    # Feature analysis
    feature_discriminators: Dict[str, FeatureDiscriminator] = field(default_factory=dict)
    top_discriminating_features: List[str] = field(default_factory=list)
    feature_importance_ranking: List[Tuple[str, float]] = field(default_factory=list)

    # Context analysis
    context_dependencies: List[ContextDependency] = field(default_factory=dict)

    # Interaction effects
    interaction_effects: List[Dict[str, Any]] = field(default_factory=list)

    # Decision boundaries
    decision_boundaries: Dict[str, Dict] = field(default_factory=dict)

    # Model results
    model_accuracy: float = 0.0
    model_accuracy_std: float = 0.0
    cross_val_scores: List[float] = field(default_factory=list)

    # Confidence
    overall_confidence: float = 0.0
    data_sufficiency: float = 0.0

    # What user says vs what data shows
    user_mentioned_features: List[str] = field(default_factory=list)
    data_discovered_features: List[str] = field(default_factory=list)
    hidden_features: List[str] = field(default_factory=list)  # Data shows, user didn't mention


@dataclass
class UnderstandingGap:
    """Something the system is NOT confident about"""
    gap_type: str               # "missing_context", "ambiguous_feature", "conflicting_examples"
    description: str
    severity: str               # "high" / "medium" / "low"
    suggested_action: str       # "Show me 3 examples of..."
    feature_involved: Optional[str] = None
    context_involved: Optional[str] = None


@dataclass
class UnderstandingReport:
    """Complete understanding summary presented to user"""
    strategy_id: str

    # What system learned
    good_setup_profile: Dict[str, str] = field(default_factory=dict)  # feature → description
    bad_setup_profile: Dict[str, str] = field(default_factory=dict)
    context_insights: List[str] = field(default_factory=list)

    # Confidence
    overall_confidence: float = 0.0
    feature_confidence: Dict[str, float] = field(default_factory=dict)

    # Gaps
    understanding_gaps: List[UnderstandingGap] = field(default_factory=list)

    # Quiz results
    quiz_accuracy: Optional[float] = None
    quiz_results: List[Dict] = field(default_factory=list)

    # Recommendations
    additional_examples_needed: List[str] = field(default_factory=list)

    def to_display(self) -> str:
        """Human-readable understanding report"""
        lines = []
        lines.append("=" * 60)
        lines.append("📊 STRATEGY UNDERSTANDING REPORT")
        lines.append("=" * 60)
        lines.append("")

