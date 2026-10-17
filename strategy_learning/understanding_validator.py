"""
============================================================
UNDERSTANDING VALIDATOR
Proves system understood correctly via quiz + gap detection
============================================================
File: soe/understanding_validator.py
============================================================
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("SOE.UnderstandingValidator")


@dataclass
class QuizQuestion:
    """A single quiz question to validate understanding"""
    question_id: str
    features: Dict[str, float]
    system_prediction: str          # "would_take" / "would_skip"
    system_confidence: float
    system_reasoning: str
    
    # User response (filled after user answers)
    user_answer: Optional[str] = None       # "take" / "skip"
    user_explanation: Optional[str] = None
    is_correct: Optional[bool] = None
    
    # Display info
    key_features_shown: Dict[str, str] = field(default_factory=dict)


class UnderstandingValidator:
    """
    Validates that the system correctly understood the user's strategy.
    
    Three validation methods:
    1. SUMMARY: System tells user what it learned → user confirms/corrects
    2. QUIZ: System predicts on new setups → user validates
    3. GAP DETECTION: System identifies what it's NOT sure about
    
    This is the QUALITY GATE before SKO is generated.
    """

    def __init__(
        self,
        pattern_learner: 'OnboardingPatternLearner',
        pattern_discovery: 'PatternDiscovery'
    ):
        self.learner = pattern_learner
        self.discovery = pattern_discovery
        self.quiz_results: List[QuizQuestion] = []

    # ═══════════════════════════════════════════════════════
    # 1. UNDERSTANDING SUMMARY
    # ═══════════════════════════════════════════════════════

    def generate_understanding_report(
        self,
        regime_definition: Optional['RegimeDefinition'] = None
    ) -> 'UnderstandingReport':
        """
        Generate a complete report of what system learned.
        User reviews and confirms/corrects.
        """
        report = UnderstandingReport(
            strategy_id=self.discovery.strategy_id
        )

        # ── Good setup profile ──
        for fname in self.discovery.top_discriminating_features[:8]:
            disc = self.discovery.feature_discriminators.get(fname)
            if disc is None:
                continue

            if disc.strength in (
                FeatureDiscriminatorStrength.STRONG,
                FeatureDiscriminatorStrength.MODERATE
            ):
                readable_name = self._feature_to_readable(fname)
                direction = "lower" if disc.good_mean < disc.bad_mean else "higher"

                report.good_setup_profile[readable_name] = (
                    f"You prefer {direction} values "
                    f"(avg: {disc.good_mean:.3f}, "
                    f"threshold: ~{disc.optimal_threshold:.3f})"
                )

                # Also add to bad profile (inverse)
                bad_direction = "higher" if direction == "lower" else "lower"
                report.bad_setup_profile[readable_name] = (
                    f"You avoid when {bad_direction} "
                    f"(avg in skipped: {disc.bad_mean:.3f})"
                )

        # ── Context insights ──
        for dep in self.discovery.context_dependencies:
            if dep.is_significant:
                report.context_insights.append(dep.insight)

        # ── Interaction effects ──
        for interaction in self.discovery.interaction_effects[:3]:
            report.context_insights.append(interaction["insight"])

        # ── Hidden features (data found, user didn't mention) ──
        if self.discovery.hidden_features:
            for hf in self.discovery.hidden_features[:3]:
                readable = self._feature_to_readable(hf)
                disc = self.discovery.feature_discriminators.get(hf)
                if disc and disc.strength != FeatureDiscriminatorStrength.NONE:
                    report.context_insights.append(
                        f"🔍 HIDDEN PATTERN: {readable} also differs between "
                        f"your TAKE vs SKIP setups (effect size: {disc.effect_size:.2f}) "
                        f"— you may not consciously notice this"
                    )

        # ── Gaps ──
        report.understanding_gaps = self.detect_understanding_gaps()

        # ── Feature confidence ──
        for fname, disc in self.discovery.feature_discriminators.items():
            report.feature_confidence[fname] = disc.auc

        # ── Overall confidence ──
        report.overall_confidence = self.discovery.overall_confidence

        # ── Recommendations for more examples ──
        for gap in report.understanding_gaps:
            if gap.severity in ("high", "medium"):
                report.additional_examples_needed.append(
                    gap.suggested_action
                )

        return report

    # ═══════════════════════════════════════════════════════
    # 2. QUIZ MODE
    # ═══════════════════════════════════════════════════════

    def generate_quiz(
        self,
        candidate_features_list: List[Dict[str, float]],
        n_questions: int = 5
    ) -> List[QuizQuestion]:
        """
        Generate quiz questions from candidate setups.
        
        System predicts user's decision, then user validates.
        
        Args:
            candidate_features_list: List of feature dicts for potential setups
                                    (from market scan or historical data)
            n_questions: How many questions to ask
        """
        questions = []

        # Select diverse candidates (mix of predicted TAKE and SKIP)
        predictions = []
        for i, features in enumerate(candidate_features_list):
            pred = self.learner.predict(features)
            predictions.append((i, features, pred))

        # Sort by confidence — pick some high-confidence and some borderline
        predictions.sort(key=lambda x: x[2]["confidence"], reverse=True)

        selected = []

        # Pick 2 high-confidence TAKE predictions
        take_preds = [p for p in predictions if p[2]["prediction"] == "would_take"]
        selected.extend(take_preds[:2])

        # Pick 2 high-confidence SKIP predictions
        skip_preds = [p for p in predictions if p[2]["prediction"] == "would_skip"]
        selected.extend(skip_preds[:2])

        # Pick 1-2 borderline predictions (confidence near 0.5)
        borderline = sorted(
            predictions,
            key=lambda x: abs(x[2]["confidence"] - 0.5)
        )
        for b in borderline:
            if b not in selected and len(selected) < n_questions:
                selected.append(b)

        # Create quiz questions
        for idx, (i, features, pred) in enumerate(selected[:n_questions]):
            # Get key features to show user
            key_features = self._get_key_features_for_display(features)

            question = QuizQuestion(
                question_id=f"quiz_{idx}",
                features=features,
                system_prediction=pred["prediction"],
                system_confidence=pred["confidence"],
                system_reasoning=pred["explanation"],
                key_features_shown=key_features
            )
            questions.append(question)

        self.quiz_results = questions
        return questions

    def submit_quiz_answer(
        self,
        question_id: str,
        user_answer: str,
        user_explanation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user's answer to a quiz question.
        
        Returns feedback on whether system was right.
        """
        question = None
        for q in self.quiz_results:
            if q.question_id == question_id:
                question = q
                break

        if question is None:
            return {"error": f"Question {question_id} not found"}

        # Normalize answer
        answer_map = {
            "take": "would_take", "yes": "would_take",
            "skip": "would_skip", "no": "would_skip",
            "would_take": "would_take", "would_skip": "would_skip"
        }
        normalized_answer = answer_map.get(user_answer.lower(), user_answer.lower())

        question.user_answer = normalized_answer
        question.user_explanation = user_explanation
        question.is_correct = (
            question.system_prediction == normalized_answer
        )

        # Build feedback
        if question.is_correct:
            feedback = {
                "correct": True,
                "message": f"✅ Correct! I predicted '{question.system_prediction}' "
                          f"with {question.system_confidence:.0%} confidence.",
                "reasoning": question.system_reasoning
            }
        else:
            feedback = {
                "correct": False,
                "message": f"❌ I predicted '{question.system_prediction}' "
                          f"but you said '{normalized_answer}'. "
                          f"Learning from this mistake.",
                "reasoning": question.system_reasoning,
                "correction_note": (
                    f"User explanation: {user_explanation}"
                    if user_explanation else "No explanation provided"
                )
            }

        return feedback

    def get_quiz_results(self) -> Dict[str, Any]:
        """Get overall quiz performance"""
        answered = [q for q in self.quiz_results if q.user_answer is not None]

        if not answered:
            return {"status": "no_answers", "accuracy": 0.0}

        correct = sum(1 for q in answered if q.is_correct)
        total = len(answered)
        accuracy = correct / total

        # Analyze errors
        errors = [q for q in answered if not q.is_correct]
        error_analysis = []
        for err in errors:
            error_analysis.append({
                "question_id": err.question_id,
                "system_said": err.system_prediction,
                "user_said": err.user_answer,
                "confidence": err.system_confidence,
                "user_explanation": err.user_explanation
            })

        # Overall assessment
        if accuracy >= 0.8:
            assessment = "STRONG understanding — ready for live filtering"
        elif accuracy >= 0.6:
            assessment = "MODERATE understanding — needs some refinement"
        elif accuracy >= 0.4:
            assessment = "WEAK understanding — need more examples"
        else:
            assessment = "POOR understanding — significant retraining needed"

        return {
            "status": "complete",
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "assessment": assessment,
            "errors": error_analysis,
            "confidence_calibration": self._check_confidence_calibration(answered)
        }

    def _check_confidence_calibration(
        self, answered_questions: List[QuizQuestion]
    ) -> Dict[str, Any]:
        """
        Check if system's confidence matches actual accuracy.
        
        Good calibration: 80% confidence → correct 80% of the time
        Bad calibration: 90% confidence but only correct 50% of the time
        """
        high_conf = [q for q in answered_questions if q.system_confidence > 0.7]
        low_conf = [q for q in answered_questions if q.system_confidence <= 0.7]

        high_acc = (
            np.mean([q.is_correct for q in high_conf])
            if high_conf else 0
        )
        low_acc = (
            np.mean([q.is_correct for q in low_conf])
            if low_conf else 0
        )

        calibration = "good" if abs(high_acc - 0.8) < 0.2 else "needs_adjustment"

        return {
            "high_confidence_accuracy": float(high_acc),
            "low_confidence_accuracy": float(low_acc),
            "calibration": calibration,
            "is_well_calibrated": calibration == "good"
        }

    # ═══════════════════════════════════════════════════════
    # 3. GAP DETECTION
    # ═══════════════════════════════════════════════════════

    def detect_understanding_gaps(self) -> List['UnderstandingGap']:
        """
        Identify what the system is NOT confident about.
        
        Checks for:
        1. Features with low discrimination (can't tell TAKE from SKIP)
        2. Missing context coverage (e.g., no examples in uptrend)
        3. Conflicting examples (same features, different labels)
        4. Low-data regions (parts of feature space with no examples)
        5. User-mentioned concepts not captured in features
        """
        gaps = []

        # ── Gap Type 1: Weak Feature Understanding ──
        important_feature_groups = {
            "candle_quality": [
                "wick_ratio_mean", "body_consistency",
                "body_size_variance", "body_to_range_ratio"
            ],
            "compression_quality": [
                "range_contraction_r2", "squeeze_score",
                "compression_candle_count", "range_volatility"
            ],
            "volume_behavior": [
                "volume_trend_slope", "volume_ratio_last5",
                "volume_spike_count", "volume_consistency"
            ],
            "location": [
                "distance_to_nearest_support",
                "distance_to_nearest_resistance",
                "price_percentile_in_day_range"
            ],
            "momentum_trend": [
                "rsi_at_setup", "momentum_score",
                "trend_direction_score", "direction_entropy"
            ]
        }

        for group_name, features in important_feature_groups.items():
            group_strengths = []
            for fname in features:
                disc = self.discovery.feature_discriminators.get(fname)
                if disc:
                    group_strengths.append(disc.effect_size)
                else:
                    group_strengths.append(0)

            avg_strength = np.mean(group_strengths) if group_strengths else 0

            if avg_strength < 0.3:
                readable_group = group_name.replace("_", " ").title()
                gaps.append(UnderstandingGap(
                    gap_type="weak_feature_group",
                    description=(
                        f"I can't clearly distinguish your preferences for "
                        f"{readable_group}. The examples don't show a clear "
                        f"pattern in this area."
                    ),
                    severity="high" if avg_strength < 0.15 else "medium",
                    suggested_action=(
                        f"Show me 3-5 examples where {readable_group} "
                        f"was the MAIN reason you took or skipped a setup."
                    ),
                    feature_involved=group_name
                ))

        # ── Gap Type 2: Missing Context Coverage ──
        context_coverage = self._analyze_context_coverage()
        for context_name, missing_values in context_coverage.items():
            if missing_values:
                gaps.append(UnderstandingGap(
                    gap_type="missing_context",
                    description=(
                        f"I have no examples in these {context_name} conditions: "
                        f"{', '.join(missing_values)}"
                    ),
                    severity="medium",
                    suggested_action=(
                        f"Show me examples when market is "
                        f"{', '.join(missing_values)}"
                    ),
                    context_involved=context_name
                ))

        # ── Gap Type 3: Conflicting Examples ──
        conflicts = self._find_conflicting_examples()
        if conflicts:
            gaps.append(UnderstandingGap(
                gap_type="conflicting_examples",
                description=(
                    f"Found {len(conflicts)} pairs of similar setups where "
                    f"you made opposite decisions. This suggests hidden "
                    f"factors I'm not capturing yet."
                ),
                severity="high",
                suggested_action=(
                    "For the conflicting pairs, can you explain what "
                    "SPECIFICALLY was different? Even subtle things like "
                    "'the feel of momentum' or 'how price approached the level'."
                )
            ))

        # ── Gap Type 4: Low-Data Regions ──
        if self.discovery.n_examples < 20:
            gaps.append(UnderstandingGap(
                gap_type="insufficient_data",
                description=(
                    f"Only {self.discovery.n_examples} examples. "
                    f"I need at least 20 for reliable pattern learning."
                ),
                severity="high",
                suggested_action=(
                    f"Please provide {20 - self.discovery.n_examples} "
                    f"more labeled examples."
                )
            ))

        # ── Gap Type 5: Class Imbalance ──
        if self.discovery.n_examples > 0:
            pos_ratio = self.discovery.n_positive / self.discovery.n_examples
            if pos_ratio > 0.8:
                gaps.append(UnderstandingGap(
                    gap_type="class_imbalance",
                    description=(
                        f"Most of your examples ({self.discovery.n_positive}/"
                        f"{self.discovery.n_examples}) are TAKE setups. "
                        f"I need more SKIP examples to learn what you avoid."
                    ),
                    severity="high",
                    suggested_action=(
                        "Show me 5-8 examples of setups you would SKIP "
                        "or AVOID, with explanations of why."
                    )
                ))
            elif pos_ratio < 0.2:
                gaps.append(UnderstandingGap(
                    gap_type="class_imbalance",
                    description=(
                        f"Most examples are SKIP ({self.discovery.n_negative}/"
                        f"{self.discovery.n_examples}). "
                        f"I need more TAKE examples to learn what you like."
                    ),
                    severity="high",
                    suggested_action=(
                        "Show me 5-8 examples of setups you would TAKE, "
                        "with explanations of why they're good."
                    )
                ))

        # ── Gap Type 6: User Mentioned but Not Captured ──
        user_terms = set()
        for ex in self._get_examples():
            user_terms.update(ex.explanation_tags)

        captured_features = set(
            f for f, d in self.discovery.feature_discriminators.items()
            if d.strength != FeatureDiscriminatorStrength.NONE
        )

        # Check if user mentioned concepts that aren't captured
        uncaptured_terms = self._find_uncaptured_concepts(
            user_terms, captured_features
        )
        if uncaptured_terms:
            gaps.append(UnderstandingGap(
                gap_type="uncaptured_concept",
                description=(
                    f"You mentioned these concepts but I can't find "
                    f"statistical patterns for them: {', '.join(uncaptured_terms)}. "
                    f"Either I need more examples or these are too subtle "
                    f"for current feature extraction."
                ),
                severity="medium",
                suggested_action=(
                    f"For each of [{', '.join(uncaptured_terms[:3])}], "
                    f"show me 2 examples where this was the KEY factor."
                )
            ))

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        gaps.sort(key=lambda g: severity_order.get(g.severity, 3))

        return gaps

    def _analyze_context_coverage(self) -> Dict[str, List[str]]:
        """Check which context values are missing from examples"""
        expected_contexts = {
            "regime": ["bullish", "bearish", "sideways"],
            "volatility": ["high_volatility", "low_volatility", "medium_volatility"],
            "time_bucket": ["morning", "midday", "afternoon"],
            "volume_condition": ["declining_volume", "increasing_volume", "stable_volume"]
        }

        observed = defaultdict(set)
        for ex in self._get_examples():
            features = ex.features
            if features.get("trend_direction_score", 0) > 0.3:
                observed["regime"].add("bullish")
            elif features.get("trend_direction_score", 0) < -0.3:
                observed["regime"].add("bearish")
            else:
                observed["regime"].add("sideways")

            rv = features.get("range_volatility", 0.5)
            if rv > 0.7:
                observed["volatility"].add("high_volatility")
            elif rv < 0.3:
                observed["volatility"].add("low_volatility")
            else:
                observed["volatility"].add("medium_volatility")

            tb = features.get("time_bucket", 1)
            if tb <= 1:
                observed["time_bucket"].add("morning")
            elif tb <= 2:
                observed["time_bucket"].add("midday")
            else:
                observed["time_bucket"].add("afternoon")

            vt = features.get("volume_trend_slope", 0)
            if vt < -0.3:
                observed["volume_condition"].add("declining_volume")
            elif vt > 0.3:
                observed["volume_condition"].add("increasing_volume")
            else:
                observed["volume_condition"].add("stable_volume")

        missing = {}
        for context_name, expected_vals in expected_contexts.items():
            missing_vals = [
                v for v in expected_vals
                if v not in observed.get(context_name, set())
            ]
            if missing_vals:
                missing[context_name] = missing_vals

        return missing

    def _find_conflicting_examples(self) -> List[Dict]:
        """
        Find pairs of examples with similar features but opposite labels.
        These indicate hidden factors the system isn't capturing.
        """
        conflicts = []
        examples = self._get_examples()

        if len(examples) < 5:
            return conflicts

        # Get feature vectors
        feature_names = list(self.discovery.feature_discriminators.keys())
        if not feature_names:
            return conflicts

        vectors = []
        for ex in examples:
            vec = [ex.features.get(f, 0) for f in feature_names]
            vectors.append(vec)

        vectors = np.array(vectors)

        # Normalize for distance computation
        stds = np.std(vectors, axis=0)
        stds = np.where(stds == 0, 1, stds)
        normalized = vectors / stds

        # Find similar pairs with opposite labels
        for i in range(len(examples)):
            for j in range(i + 1, len(examples)):
                # Euclidean distance in normalized space
                dist = np.sqrt(np.sum((normalized[i] - normalized[j]) ** 2))

                # Check if similar but opposite label
                label_i = examples[i].label.binary_label if hasattr(examples[i].label, 'binary_label') else None
                label_j = examples[j].label.binary_label if hasattr(examples[j].label, 'binary_label') else None

                if label_i is not None and label_j is not None:
                    if label_i != label_j and dist < 2.0:  # Similar features
                        conflicts.append({
                            "example_1": examples[i].example_id,
                            "example_2": examples[j].example_id,
                            "distance": float(dist),
                            "label_1": label_i,
                            "label_2": label_j,
                            "explanation_1": examples[i].explanation,
                            "explanation_2": examples[j].explanation
                        })

        # Sort by distance (most similar = most conflicting)
        conflicts.sort(key=lambda x: x["distance"])
        return conflicts[:5]  # Top 5 conflicts

    def _find_uncaptured_concepts(
        self,
        user_terms: set,
        captured_features: set
    ) -> List[str]:
        """Find user concepts that aren't reflected in statistical patterns"""
        uncaptured = []

        # Map terms to feature groups
        term_to_features = {
            "clean": {"wick_ratio_mean", "body_consistency", "body_size_variance"},
            "smooth": {"body_size_variance", "direction_entropy", "range_contraction_r2"},
            "choppy": {"direction_entropy", "body_size_variance"},
            "compression": {"range_contraction_r2", "squeeze_score"},
            "volume_dying": {"volume_trend_slope", "volume_ratio_last5"},
            "near_support": {"distance_to_nearest_support"},
            "near_resistance": {"distance_to_nearest_resistance"},
            "momentum": {"rsi_at_setup", "momentum_score"},
            "trending": {"trend_direction_score", "adx"},
        }

        for term in user_terms:
            expected_features = term_to_features.get(term, set())
            if expected_features and not expected_features.intersection(captured_features):
                uncaptured.append(term)

        return uncaptured

    def _get_key_features_for_display(
        self, features: Dict[str, float]
    ) -> Dict[str, str]:
        """Get human-readable key features for quiz display"""
        display = {}

        top_features = self.discovery.top_discriminating_features[:8]

        for fname in top_features:
            val = features.get(fname)
            if val is None:
                continue

            readable = self._feature_to_readable(fname)
            disc = self.discovery.feature_discriminators.get(fname)

            if disc:
                # Show relative to learned thresholds
                if disc.optimal_threshold is not None:
                    quality = (
                        "✅" if (
                            (disc.threshold_direction == "below" and val <= disc.optimal_threshold) or
                            (disc.threshold_direction == "above" and val >= disc.optimal_threshold)
                        ) else "⚠️"
                    )
                    display[readable] = f"{quality} {val:.3f} (threshold: {disc.optimal_threshold:.3f})"
                else:
                    display[readable] = f"{val:.3f}"
            else:
                display[readable] = f"{val:.3f}"

        return display

    def _feature_to_readable(self, feature_name: str) -> str:
        """Convert feature name to human-readable format"""
        readable_map = {
            "wick_ratio_mean": "Wick Ratio (avg)",
            "wick_ratio_max": "Wick Ratio (max)",
            "body_consistency": "Body Consistency",
            "body_size_variance": "Body Size Variance",
            "avg_body_size": "Average Body Size",
            "body_to_range_ratio": "Body-to-Range Ratio",
            "range_contraction_r2": "Range Contraction (R²)",
            "squeeze_score": "Squeeze Score",
            "compression_candle_count": "Compression Candles",
            "range_volatility": "Range Volatility",
            "volume_trend_slope": "Volume Trend",
            "volume_ratio_last5": "Recent Volume Ratio",
            "volume_spike_count": "Volume Spikes",
            "volume_consistency": "Volume Consistency",
            "direction_entropy": "Direction Entropy",
            "direction_consistency": "Direction Consistency",
            "rsi_at_setup": "RSI",
            "momentum_score": "Momentum",
            "distance_to_nearest_support": "Distance to Support (ATR)",
            "distance_to_nearest_resistance": "Distance to Resistance (ATR)",
            "price_percentile_in_day_range": "Price Position in Day Range",
            "trend_direction_score": "Trend Direction",
            "adx": "ADX (Trend Strength)",
        }
        return readable_map.get(
            feature_name,
            feature_name.replace("_", " ").title()
        )

    def _get_examples(self) -> List:
        """Get examples from pattern learner (helper)"""
        # This would reference the stored examples
        # In practice, passed during initialization or stored
        return getattr(self, '_examples', [])

    def set_examples(self, examples: List):
        """Set examples for gap detection analysis"""
        self._examples = examples
