# strategy_learning/__init__.py
"""
AI-Learned Strategy Understanding Module.

Instead of hard-coded rules, this module builds a 'latent understanding'
of the user's trading strategies from:
  - Chart screenshots analysed by Vision AI
  - User annotations (entry, stop, target, reasoning)
  - Interactive teaching sessions with Q&A
  - Live trade outcome feedback
  - Example-based ML learning (20-25 labeled examples)

The learned understanding powers:
  - Trade journal analysis (was the user on-strategy?)
  - Live trade co-planning (real-time advice)
  - Override detection (deviation from learned pattern)
  - Coaching (personalised, data-driven feedback)
  - Eventual autonomous execution (intention-based, not just rule-based)

Onboarding Enhancement Modules (ML-based learning):
  - pattern_learner             : Statistical pattern discovery from examples
  - onboarding_feature_extractor: Numerical feature extraction from candle data
  - regime_builder              : Regime description → measurable conditions
  - confidence_decomposition    : Per-component confidence breakdown
  - understanding_validator     : Gap detection + quiz validation
  - learning_separation         : Separates user INTENT from statistical EDGE
  - avt_bridge                  : Bridges learning to AVT autonomous scoring
  - counterfactual_tester       : What-if scenario testing
  - onboarding_orchestrator     : Top-level onboarding workflow
  - onboarding_sko_builder      : Builds Strategy Knowledge Object
  - ptis_integration            : Connects to SUE, JudgmentLearner, etc.
  - onboarding_models           : Training example data models
"""


from strategy_learning.models import (
    BatchTrainingReport,
    CandlePatternObservation,
    ConsistentPattern,
    CoplanRequest,
    CoplanResponse,
    LearnedEntryCriterion,
    LearnedExitRules,
    PriceLevel,
    ProfileConfidence,
    SimilarExample,
    StrategyContradiction,
    StrategyProfileSummary,
    SynthesisedRule,
    TradeAnalysisRequest,
    TradeAnalysisResponse,
    TradeDeviation,
    TradeOutcomeFeedback,
    UserAnnotation,
    VisionAnalysis,
)

__all__ = [
    # Vision
    "CandlePatternObservation",
    "PriceLevel",
    "VisionAnalysis",
    # Annotations
    "UserAnnotation",
    # Profile
    "ProfileConfidence",
    "LearnedEntryCriterion",
    "LearnedExitRules",
    "StrategyProfileSummary",
    # Co-plan
    "CoplanRequest",
    "CoplanResponse",
    "SimilarExample",
    # Trade analysis
    "TradeAnalysisRequest",
    "TradeAnalysisResponse",
    "TradeDeviation",
    # Feedback loop
    "TradeOutcomeFeedback",
    # Batch training
    "BatchTrainingReport",
    "ConsistentPattern",
    "StrategyContradiction",
    "SynthesisedRule",
]