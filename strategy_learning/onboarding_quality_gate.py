# strategy_learning/onboarding_quality_gate.py
"""
Onboarding Quality Gate — Garbage Strategy Detection
======================================================

┌─────────────────────────────────────────────────────────────┐
│  PURPOSE: Detect low-quality teaching examples EARLY.       │
│                                                             │
│  This gates at ONBOARDING TIME, not execution time.         │
│  It checks structural coherence of examples — NOT outcomes. │
│                                                             │
│  RULE: This module checks if examples are LEARNABLE.        │
│  It does NOT check if the strategy is PROFITABLE.           │
│  Profitability is AVT's job (via edge discovery).           │
│                                                             │
│  What it catches:                                           │
│   - Random/incoherent screenshots (no structure)            │
│   - Examples too dissimilar to form a pattern               │
│   - Missing critical annotations                            │
│   - Contradictory labels on similar setups                  │
│                                                             │
│  What it does NOT do:                                       │
│   ❌ Reject based on win rate or expectancy                 │
│   ❌ Judge strategy quality by outcomes                     │
│   ❌ Override user intent                                   │
└─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Thresholds ────────────────────────────────────────────────

# Minimum examples needed for any learning
MIN_EXAMPLES = 5

# Feature variance: examples that are all identical or all random
# are both useless for learning
MIN_FEATURE_VARIANCE = 0.01  # too uniform → no discrimination
MAX_FEATURE_VARIANCE = 10.0  # too noisy → no coherent pattern

# Label imbalance: need both TAKE and SKIP examples
MIN_LABEL_RATIO = 0.15  # at least 15% of each label

# Annotation completeness
MIN_ANNOTATION_COMPLETENESS = 0.3  # at least 30% of fields filled


# ── Output Models ─────────────────────────────────────────────

@dataclass
class QualityIssue:
    """A single quality issue detected in teaching examples."""
    category: str       # "coherence" | "variance" | "labels" | "annotations"
    severity: str       # "info" | "warning" | "critical"
    description: str
    suggestion: str     # actionable fix

    def to_dict(self) -> Dict[str, str]:
        return {
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "suggestion": self.suggestion,
        }


@dataclass
class OnboardingQualityReport:
    """
    Quality assessment of teaching examples.

    NOTE: This report is INFORMATIONAL. Even if quality is low,
    the system accepts the strategy and provides guidance to improve.
    The only exception is when examples are completely unusable
    (e.g., zero valid screenshots).
    """
    # Overall assessment
    overall_score: float = 0.0          # 0-1
    is_learnable: bool = True           # Can we learn anything?
    quality_level: str = "acceptable"   # "excellent" | "good" | "acceptable" | "poor"

    # Component scores
    coherence_score: float = 0.0        # Are examples structurally similar?
    variance_score: float = 0.0         # Is there enough feature variance?
    label_balance_score: float = 0.0    # TAKE vs SKIP balance
    annotation_score: float = 0.0       # How complete are annotations?

    # Issues found
    issues: List[QualityIssue] = field(default_factory=list)

    # Counts
    total_examples: int = 0
    valid_examples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 3),
            "is_learnable": self.is_learnable,
            "quality_level": self.quality_level,
            "scores": {
                "coherence": round(self.coherence_score, 3),
                "variance": round(self.variance_score, 3),
                "label_balance": round(self.label_balance_score, 3),
                "annotations": round(self.annotation_score, 3),
            },
            "total_examples": self.total_examples,
            "valid_examples": self.valid_examples,
            "issues": [i.to_dict() for i in self.issues],
        }


class OnboardingQualityGate:
    """
    Evaluates teaching example quality during onboarding.

    Checks if the examples are structurally coherent enough to learn from.
    Does NOT evaluate strategy profitability (that's AVT's job).
    """

    def assess(
        self,
        examples: List[Dict[str, Any]],
        feature_keys: Optional[List[str]] = None,
    ) -> OnboardingQualityReport:
        """
        Assess teaching example quality.

        Args:
            examples: List of teaching examples, each with:
                - 'features': Dict[str, float] — extracted features
                - 'label': str — "take" or "skip"
                - 'annotation': Dict — user annotations
            feature_keys: Optional list of feature names to check.

        Returns:
            OnboardingQualityReport with scores and issues.
        """
        report = OnboardingQualityReport(
            total_examples=len(examples),
        )

        # ── Check minimum count ──
        if len(examples) < MIN_EXAMPLES:
            report.is_learnable = False
            report.quality_level = "poor"
            report.issues.append(QualityIssue(
                category="count",
                severity="critical",
                description=(
                    f"Only {len(examples)} examples provided. "
                    f"Need at least {MIN_EXAMPLES} for learning."
                ),
                suggestion="Add more teaching examples with screenshots and annotations.",
            ))
            return report

        # Filter to examples with features
        valid = [e for e in examples if e.get("features")]
        report.valid_examples = len(valid)

        if len(valid) < MIN_EXAMPLES:
            report.is_learnable = False
            report.quality_level = "poor"
            report.issues.append(QualityIssue(
                category="features",
                severity="critical",
                description=(
                    f"Only {len(valid)}/{len(examples)} examples have "
                    f"extractable features."
                ),
                suggestion=(
                    "Ensure screenshots show clear chart data. "
                    "Low quality or cropped images produce no features."
                ),
            ))
            return report

        # Auto-detect feature keys if not provided
        if not feature_keys:
            feature_keys = sorted(set().union(
                *(e["features"].keys() for e in valid if e.get("features"))
            ))

        # ── Run quality checks ──
        report.coherence_score = self._check_coherence(valid, feature_keys, report)
        report.variance_score = self._check_variance(valid, feature_keys, report)
        report.label_balance_score = self._check_label_balance(valid, report)
        report.annotation_score = self._check_annotations(valid, report)

        # ── Overall score ──
        report.overall_score = (
            report.coherence_score * 0.30
            + report.variance_score * 0.25
            + report.label_balance_score * 0.25
            + report.annotation_score * 0.20
        )

        # ── Quality level ──
        if report.overall_score >= 0.75:
            report.quality_level = "excellent"
        elif report.overall_score >= 0.55:
            report.quality_level = "good"
        elif report.overall_score >= 0.35:
            report.quality_level = "acceptable"
        else:
            report.quality_level = "poor"

        # Only mark as unlearnable if truly hopeless
        report.is_learnable = report.overall_score >= 0.15

        return report

    def _check_coherence(
        self,
        examples: List[Dict],
        feature_keys: List[str],
        report: OnboardingQualityReport,
    ) -> float:
        """
        Check if examples are structurally coherent.

        Coherent: examples cluster together (similar features).
        Incoherent: examples are randomly scattered (no learnable pattern).
        """
        if len(feature_keys) < 2:
            return 0.5  # Can't assess with too few features

        # Build feature matrix
        matrix = []
        for ex in examples:
            row = [float(ex["features"].get(k, 0.0)) for k in feature_keys]
            matrix.append(row)

        X = np.array(matrix)
        if X.shape[0] < 3:
            return 0.5

        # Compute pairwise distances
        from scipy.spatial.distance import pdist

        try:
            distances = pdist(X, metric="euclidean")
        except Exception:
            return 0.5

        if len(distances) == 0:
            return 0.5

        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))

        # Coefficient of variation of distances
        # Low CV = tight cluster = coherent
        # High CV = scattered = incoherent
        cv = std_dist / max(mean_dist, 1e-6)

        if cv < 0.3:
            score = 0.9  # Very tight cluster
        elif cv < 0.6:
            score = 0.7  # Reasonable coherence
        elif cv < 1.0:
            score = 0.5  # Moderate scatter
        else:
            score = 0.3  # High scatter
            report.issues.append(QualityIssue(
                category="coherence",
                severity="warning",
                description=(
                    "Teaching examples are structurally scattered — "
                    "features vary widely across examples."
                ),
                suggestion=(
                    "Consider focusing on one specific setup type. "
                    "If you trade multiple setups, teach each one separately."
                ),
            ))

        return score

    def _check_variance(
        self,
        examples: List[Dict],
        feature_keys: List[str],
        report: OnboardingQualityReport,
    ) -> float:
        """
        Check that examples have enough variance to be discriminative.

        If all TAKE examples look identical to all SKIP examples,
        there's nothing to learn. Need variance BETWEEN labels.
        """
        take_examples = [e for e in examples
                         if str(e.get("label", "")).lower() in ("take", "would_take")]
        skip_examples = [e for e in examples
                         if str(e.get("label", "")).lower() in ("skip", "would_skip")]

        if not take_examples or not skip_examples:
            return 0.5  # Can't assess without both labels

        # Per-feature variance within and between groups
        discriminative_features = 0
        total_features = 0

        for key in feature_keys:
            take_vals = [float(e["features"].get(key, 0.0)) for e in take_examples]
            skip_vals = [float(e["features"].get(key, 0.0)) for e in skip_examples]

            if not take_vals or not skip_vals:
                continue

            total_features += 1
            take_mean = np.mean(take_vals)
            skip_mean = np.mean(skip_vals)
            overall_std = np.std(take_vals + skip_vals)

            if overall_std < MIN_FEATURE_VARIANCE:
                continue  # No variance at all

            # Effect size (Cohen's d)
            pooled_std = max(overall_std, 1e-6)
            effect_size = abs(take_mean - skip_mean) / pooled_std

            if effect_size > 0.3:
                discriminative_features += 1

        if total_features == 0:
            return 0.5

        ratio = discriminative_features / total_features

        if ratio >= 0.3:
            score = 0.9  # Many discriminative features
        elif ratio >= 0.15:
            score = 0.7  # Some discrimination
        elif ratio >= 0.05:
            score = 0.5  # Weak discrimination
        else:
            score = 0.2  # TAKE and SKIP look identical
            report.issues.append(QualityIssue(
                category="variance",
                severity="warning",
                description=(
                    "TAKE and SKIP examples have very similar features — "
                    "the system may struggle to distinguish them."
                ),
                suggestion=(
                    "Ensure your TAKE and SKIP examples represent clearly "
                    "different market situations. Add more diverse examples."
                ),
            ))

        return score

    def _check_label_balance(
        self,
        examples: List[Dict],
        report: OnboardingQualityReport,
    ) -> float:
        """
        Check that labels are reasonably balanced.

        Need both TAKE and SKIP examples to learn discrimination.
        """
        labels = [str(e.get("label", "")).lower() for e in examples]
        total = len(labels)
        if total == 0:
            return 0.0

        take_count = sum(1 for l in labels if l in ("take", "would_take"))
        skip_count = sum(1 for l in labels if l in ("skip", "would_skip"))
        unlabeled = total - take_count - skip_count

        if take_count == 0 or skip_count == 0:
            report.issues.append(QualityIssue(
                category="labels",
                severity="critical",
                description=(
                    f"Examples only have one label type: "
                    f"{take_count} TAKE, {skip_count} SKIP."
                ),
                suggestion=(
                    "Add examples of setups you would SKIP or NOT TAKE. "
                    "The system needs both positive and negative examples."
                ),
            ))
            return 0.1

        minority_ratio = min(take_count, skip_count) / total

        if minority_ratio >= 0.3:
            score = 1.0
        elif minority_ratio >= MIN_LABEL_RATIO:
            score = 0.7
        else:
            score = 0.4
            report.issues.append(QualityIssue(
                category="labels",
                severity="warning",
                description=(
                    f"Label imbalance: {take_count} TAKE vs {skip_count} SKIP "
                    f"(minority is {minority_ratio:.0%})."
                ),
                suggestion=(
                    "Add more examples of the underrepresented label. "
                    "Aim for at least 30% of each type."
                ),
            ))

        if unlabeled > total * 0.2:
            score *= 0.8
            report.issues.append(QualityIssue(
                category="labels",
                severity="info",
                description=f"{unlabeled} examples have no label.",
                suggestion="Label all examples as TAKE or SKIP for best learning.",
            ))

        return score

    def _check_annotations(
        self,
        examples: List[Dict],
        report: OnboardingQualityReport,
    ) -> float:
        """
        Check annotation completeness.

        Richer annotations = better understanding = better learning.
        """
        annotation_fields = [
            "direction", "reasoning", "what_makes_this_work",
            "what_would_invalidate", "setup_quality", "timeframe", "symbol",
        ]

        completeness_scores = []
        for ex in examples:
            annot = ex.get("annotation", {})
            if isinstance(annot, dict):
                filled = sum(
                    1 for f in annotation_fields
                    if annot.get(f) and str(annot[f]).strip()
                )
                completeness_scores.append(filled / len(annotation_fields))
            else:
                completeness_scores.append(0.0)

        if not completeness_scores:
            return 0.5

        avg_completeness = float(np.mean(completeness_scores))

        if avg_completeness >= 0.6:
            score = 1.0
        elif avg_completeness >= MIN_ANNOTATION_COMPLETENESS:
            score = 0.7
        elif avg_completeness > 0.1:
            score = 0.4
            report.issues.append(QualityIssue(
                category="annotations",
                severity="info",
                description=(
                    f"Annotations are {avg_completeness:.0%} complete on average."
                ),
                suggestion=(
                    "Adding reasoning (why you'd take/skip this setup) "
                    "significantly improves learning quality."
                ),
            ))
        else:
            score = 0.2
            report.issues.append(QualityIssue(
                category="annotations",
                severity="warning",
                description="Most examples have no annotations.",
                suggestion=(
                    "Add annotations: direction, reasoning, what makes it work, "
                    "what would invalidate it."
                ),
            ))

        return score
