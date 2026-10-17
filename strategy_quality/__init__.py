# strategy_quality/__init__.py
"""
Strategy Quality — Governance & Validation Layer
==================================================

Sits ABOVE the execution layer (SKE).
Answers: "Is this strategy good enough to trade?"

Three responsibilities:
  1. Health Gate — validate strategy before AVT trades it
  2. Calibration — verify AVT's interpretation matches user intent
  3. Edge Monitor — detect degradation during autonomous trading

Core classes:
  - StrategyUnderstandingEngine (SUE) — 5-layer evaluation pipeline
  - DataQualityFirewall — context-aware trade data quality gate
  - PatternAwareFirewall — ML-enhanced data quality with clustering
  - CalibrationTracker — learns from user feedback
  - DynamicPatternClusterer — discovers trade pattern variants
  - ContradictionDetector — regime-aware contradiction detection
  - FeatureScaler — online z-score/min-max/quantile scaling
"""

from strategy_quality.engine import (
    CalibrationTracker,
    ContextAlignmentLayer,
    DataDecision,
    DataQualityFirewall,
    DataQualityResult,
    HardRuleLayer,
    LayerResult,
    MetaLayerOutput,
    RegimeStabilityChecker,
    SoftScoringLayer,
    SoftScoreDistribution,
    StrategyDecision,
    StrategyProfile,
    StrategyUnderstandingEngine,
    UncertaintyLayer,
)
from strategy_quality.pattern_intelligence import (
    ClusteringResult,
    ContradictionDetector,
    DynamicPatternClusterer,
    FeatureScaler,
    FeatureStats,
    PatternAwareFirewall,
    PatternCluster,
)

__all__ = [
    # Engine
    "StrategyUnderstandingEngine",
    "StrategyProfile",
    "StrategyDecision",
    "DataDecision",
    "DataQualityFirewall",
    "DataQualityResult",
    "CalibrationTracker",
    "MetaLayerOutput",
    "LayerResult",
    "HardRuleLayer",
    "SoftScoringLayer",
    "SoftScoreDistribution",
    "ContextAlignmentLayer",
    "UncertaintyLayer",
    "RegimeStabilityChecker",
    # Pattern Intelligence
    "FeatureScaler",
    "FeatureStats",
    "DynamicPatternClusterer",
    "PatternCluster",
    "ClusteringResult",
    "ContradictionDetector",
    "PatternAwareFirewall",
]
