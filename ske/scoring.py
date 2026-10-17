# ske/scoring.py
"""
Fuzzy scoring functions for the Strategy Knowledge Engine.

Each function maps a raw feature value to a continuous [0.0, 1.0] score.
The conviction pipeline aggregates individual feature scores into a
weighted conviction score → conviction band → sizing recommendation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from core.enums import ConvictionBand, ScoringFunctionType
from ske.models import ConvictionResult, FeatureScore, ScoringParameter


# ══════════════════════════════════════════════
#  SCORING FUNCTIONS
# ══════════════════════════════════════════════

def gaussian_score(value: float, center: float, spread: float) -> float:
    """
    Gaussian (bell-curve) score: peaks at 1.0 when value == center,
    falls off symmetrically with width controlled by spread.

    score = exp(-0.5 * ((value - center) / spread)²)
    """
    if spread <= 0:
        return 1.0 if value == center else 0.0
    z = (value - center) / spread
    return max(0.0, min(1.0, math.exp(-0.5 * z * z)))


def trapezoid_score(
    value: float,
    lower: float,
    upper: float,
    spread: float,
) -> float:
    """
    Trapezoid score: 1.0 inside [lower, upper], linear ramp outside.

    The ramp extends `spread` units on each side:
      - Below lower: ramps from 0 at (lower - spread) to 1 at lower
      - Above upper: ramps from 1 at upper to 0 at (upper + spread)
    """
    if spread <= 0:
        spread = 1e-9

    if lower <= value <= upper:
        return 1.0
    elif value < lower:
        if value <= lower - spread:
            return 0.0
        return (value - (lower - spread)) / spread
    else:  # value > upper
        if value >= upper + spread:
            return 0.0
        return ((upper + spread) - value) / spread


def sigmoid_score(value: float, center: float, slope: float) -> float:
    """
    Sigmoid score: S-curve centered at `center` with steepness `slope`.

    score = 1 / (1 + exp(-slope * (value - center)))

    Positive slope → higher values score higher.
    Negative slope → lower values score higher.
    """
    x = -slope * (value - center)
    # Clamp to avoid overflow
    x = max(-500.0, min(500.0, x))
    return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(x))))


def asymmetric_score(
    value: float,
    center: float,
    left_spread: float,
    right_spread: float,
) -> float:
    """
    Asymmetric Gaussian: different spreads on left vs right of center.
    Useful when a feature has different tolerance for under- vs over-shooting.
    """
    if value <= center:
        spread = left_spread if left_spread > 0 else 1e-9
    else:
        spread = right_spread if right_spread > 0 else 1e-9

    z = (value - center) / spread
    return max(0.0, min(1.0, math.exp(-0.5 * z * z)))


# ══════════════════════════════════════════════
#  DISPATCH
# ══════════════════════════════════════════════

def compute_feature_score(value: float, param: ScoringParameter) -> float:
    """
    Score a single feature value using the configured scoring function.
    Returns a float in [0.0, 1.0].
    """
    if param.type == ScoringFunctionType.GAUSSIAN:
        score = gaussian_score(
            value,
            center=param.center or 0.0,
            spread=param.spread or 1.0,
        )
    elif param.type == ScoringFunctionType.TRAPEZOID:
        score = trapezoid_score(
            value,
            lower=param.lower or 0.0,
            upper=param.upper or 1.0,
            spread=param.trap_spread or param.spread or 0.5,
        )
    elif param.type == ScoringFunctionType.SIGMOID:
        score = sigmoid_score(
            value,
            center=param.center or 0.0,
            slope=param.slope or 1.0,
        )
    elif param.type == ScoringFunctionType.ASYMMETRIC:
        score = asymmetric_score(
            value,
            center=param.center or 0.0,
            left_spread=param.left_spread or param.spread or 1.0,
            right_spread=param.right_spread or param.spread or 1.0,
        )
    else:
        score = 0.0

    if param.invert:
        score = 1.0 - score

    return max(0.0, min(1.0, score))


# ══════════════════════════════════════════════
#  CONVICTION AGGREGATION
# ══════════════════════════════════════════════

# Default conviction map (can be overridden per-SKO)
_DEFAULT_CONVICTION_MAP: Dict[str, List[float]] = {
    "skip":       [0.0, 0.30],
    "small":      [0.30, 0.50],
    "standard":   [0.50, 0.70],
    "aggressive": [0.70, 0.85],
    "max":        [0.85, 1.0],
}

# Sizing multipliers per conviction band
_SIZING_MULTIPLIERS: Dict[ConvictionBand, float] = {
    ConvictionBand.SKIP: 0.0,
    ConvictionBand.SMALL: 0.5,
    ConvictionBand.STANDARD: 1.0,
    ConvictionBand.AGGRESSIVE: 1.5,
    ConvictionBand.MAX_CONVICTION: 2.0,
}


def _resolve_band(
    score: float,
    conviction_map: Dict[str, List[float]],
) -> ConvictionBand:
    """Map a weighted score to a conviction band."""
    for band_name in ("max", "aggressive", "standard", "small", "skip"):
        bounds = conviction_map.get(band_name, [])
        if len(bounds) == 2 and bounds[0] <= score <= bounds[1]:
            # Map string to enum
            band_enum_map = {
                "skip": ConvictionBand.SKIP,
                "small": ConvictionBand.SMALL,
                "standard": ConvictionBand.STANDARD,
                "aggressive": ConvictionBand.AGGRESSIVE,
                "max": ConvictionBand.MAX_CONVICTION,
            }
            return band_enum_map.get(band_name, ConvictionBand.STANDARD)

    # Fallback based on raw score
    if score < 0.30:
        return ConvictionBand.SKIP
    elif score < 0.50:
        return ConvictionBand.SMALL
    elif score < 0.70:
        return ConvictionBand.STANDARD
    elif score < 0.85:
        return ConvictionBand.AGGRESSIVE
    else:
        return ConvictionBand.MAX_CONVICTION


def compute_conviction(
    feature_scores: List[FeatureScore],
    conviction_map: Optional[Dict[str, List[float]]] = None,
) -> ConvictionResult:
    """
    Aggregate individual feature scores into a weighted conviction result.

    1. Weighted average of all feature scores → raw conviction (0–1)
    2. Map to conviction band via conviction_map
    3. Look up sizing multiplier

    Returns ConvictionResult with band, sizing, and all individual scores.
    """
    cmap = conviction_map or _DEFAULT_CONVICTION_MAP

    if not feature_scores:
        return ConvictionResult(
            weighted_score=0.0,
            conviction_band=ConvictionBand.SKIP,
            individual_scores=[],
            sizing_recommendation="skip",
            size_multiplier=0.0,
        )

    total_weight = sum(fs.weight for fs in feature_scores)
    if total_weight <= 0:
        total_weight = 1.0

    weighted_sum = sum(fs.normalized_score * fs.weight for fs in feature_scores)
    weighted_score = max(0.0, min(1.0, weighted_sum / total_weight))

    band = _resolve_band(weighted_score, cmap)
    multiplier = _SIZING_MULTIPLIERS.get(band, 1.0)

    return ConvictionResult(
        weighted_score=weighted_score,
        conviction_band=band,
        individual_scores=feature_scores,
        sizing_recommendation=band.value,
        size_multiplier=multiplier,
    )
