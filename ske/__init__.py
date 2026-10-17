# ske/__init__.py
"""
Strategy Knowledge Engine (SKE) — encodes trading strategies as rich
Strategy Knowledge Objects (SKOs) with fuzzy scoring, variant clustering,
conviction-based sizing, and real-time signal generation.
"""

from ske.scoring import (
    gaussian_score,
    trapezoid_score,
    sigmoid_score,
    asymmetric_score,
    compute_feature_score,
    compute_conviction,
)
from ske.models import (
    ScoringParameter,
    FeatureScore,
    ConvictionResult,
    RegimeFingerprint,
    VariantInfo,
    DQALResult,
    SignalOutput,
)

__all__ = [
    # Scoring
    "gaussian_score",
    "trapezoid_score",
    "sigmoid_score",
    "asymmetric_score",
    "compute_feature_score",
    "compute_conviction",
    # Models
    "ScoringParameter",
    "FeatureScore",
    "ConvictionResult",
    "RegimeFingerprint",
    "VariantInfo",
    "DQALResult",
    "SignalOutput",
]
