"""
============================================================
ONBOARDING PATTERN LEARNER
The core ML engine that learns from labeled examples
============================================================
File: soe/pattern_learner.py
============================================================

This is WHERE the actual learning happens.

From 25 labeled examples:
→ Discovers what separates GOOD from BAD
→ Trains initial model
→ Identifies feature importance
→ Finds context dependencies
→ Produces PatternDiscovery object
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from scipy import stats
from collections import defaultdict
import logging
import warnings

logger = logging.getLogger("SOE.PatternLearner")

# Suppress sklearn warnings for small datasets
warnings.filterwarnings('ignore', category=UserWarning)


class OnboardingPatternLearner:
    """
    Learns patterns from user's labeled onboarding examples.
    
    CRITICAL DESIGN PRINCIPLE:
    ══════════════════════════
    This learns USER INTENT, not EDGE.
    
    Input: 25 examples labeled as WOULD_TAKE / WOULD_SKIP
    Output: Model that predicts "would user take this setup?"
    
    This does NOT predict win/loss.
    That's AVT's job with 500+ samples.
    """

    # Minimum examples for different analysis types
    MIN_FOR_BASIC = 10      # Basic discriminator analysis
    MIN_FOR_MODEL = 15      # Train a model
    MIN_FOR_CONTEXT = 20    # Context dependency analysis
    MIN_FOR_ROBUST = 25     # Robust cross-validated model

    def __init__(self):
        self.trained_model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.pattern_discovery: Optional[PatternDiscovery] = None

    def learn(
        self,
        examples: List['TrainingExample'],
        strategy_id: str
    ) -> PatternDiscovery:
        """
        Main learning method. Takes labeled examples, produces discovery.
        
        This is the HEART of onboarding intelligence.
        """
        logger.info(f"Starting pattern learning for {strategy_id} "
                    f"with {len(examples)} examples")

        # Initialize discovery
        discovery = PatternDiscovery(strategy_id=strategy_id)

        # ── Step 1: Prepare dataset ──
        dataset = self._prepare_dataset(examples)
        if dataset is None:
            logger.warning("Cannot prepare dataset. Insufficient valid examples.")
            discovery.overall_confidence = 0.0
            discovery.data_sufficiency = 0.0
            return discovery

        X, y, feature_names, valid_examples = dataset
        discovery.n_examples = len(valid_examples)
        discovery.n_positive = int(np.sum(y == 1))
        discovery.n_negative = int(np.sum(y == 0))

        logger.info(f"Dataset prepared: {len(X)} examples, "
                    f"{discovery.n_positive} positive, "
                    f"{discovery.n_negative} negative, "
                    f"{len(feature_names)} features")

        # ── Step 2: Feature discrimination analysis ──
        if len(X) >= self.MIN_FOR_BASIC:
            discovery.feature_discriminators = self._analyze_feature_discrimination(
                X, y, feature_names
            )
            discovery.top_discriminating_features = self._get_top_features(
                discovery.feature_discriminators
            )
            logger.info(f"Top discriminating features: "
                       f"{discovery.top_discriminating_features[:5]}")

        # ── Step 3: Context dependency analysis ──
        if len(X) >= self.MIN_FOR_CONTEXT:
            discovery.context_dependencies = self._analyze_context_dependencies(
                valid_examples
            )

        # ── Step 4: Train classification model ──
        if len(X) >= self.MIN_FOR_MODEL:
            model_results = self._train_model(X, y, feature_names)
            self.trained_model = model_results["model"]
            self.scaler = model_results["scaler"]
            self.feature_names = feature_names

            discovery.model_accuracy = model_results["accuracy"]
            discovery.model_accuracy_std = model_results["accuracy_std"]
            discovery.cross_val_scores = model_results["cv_scores"]
            discovery.feature_importance_ranking = model_results["feature_importance"]

            logger.info(f"Model trained. Accuracy: {discovery.model_accuracy:.2%} "
                       f"± {discovery.model_accuracy_std:.2%}")

        # ── Step 5: Interaction effects ──
        if len(X) >= self.MIN_FOR_ROBUST:
            discovery.interaction_effects = self._find_interaction_effects(
                X, y, feature_names
            )

        # ── Step 6: Decision boundary discovery ──
        if discovery.feature_discriminators:
            discovery.decision_boundaries = self._discover_decision_boundaries(
                X, y, feature_names, discovery.feature_discriminators
            )

        # ── Step 7: Cross-reference with user explanations ──
        user_mentioned = self._extract_user_mentioned_features(valid_examples)
        data_discovered = [f for f, _ in discovery.feature_importance_ranking[:10]]

        discovery.user_mentioned_features = user_mentioned
        discovery.data_discovered_features = data_discovered
        discovery.hidden_features = [
            f for f in data_discovered if f not in user_mentioned
        ]

        if discovery.hidden_features:
            logger.info(f"Hidden features discovered (user didn't mention): "
                       f"{discovery.hidden_features}")

        # ── Step 8: Compute overall confidence ──
        discovery.data_sufficiency = self._compute_data_sufficiency(
            len(X), discovery.n_positive, discovery.n_negative
        )
        discovery.overall_confidence = self._compute_overall_confidence(discovery)

        self.pattern_discovery = discovery
        return discovery

    def _prepare_dataset(
        self,
        examples: List['TrainingExample']
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], List['TrainingExample']]]:
        """
        Convert examples into feature matrix X and label vector y.
        
        Only includes examples with:
        - Valid features (not empty)
        - Clear binary label (TAKE or SKIP, not borderline)
        """
        valid_examples = []
        feature_dicts = []
        labels = []

        for ex in examples:
            # Must have features
            if not ex.features:
                logger.debug(f"Skipping {ex.example_id}: no features")
                continue

            # Must have clear binary intent
            binary = ex.label.binary_label if hasattr(ex.label, 'binary_label') else None
            if binary is None:
                logger.debug(f"Skipping {ex.example_id}: ambiguous label")
                continue

            valid_examples.append(ex)
            feature_dicts.append(ex.features)
            labels.append(binary)

        if len(valid_examples) < self.MIN_FOR_BASIC:
            return None

        # Get common feature names across all examples
        all_feature_names = set()
        for fd in feature_dicts:
            all_feature_names.update(fd.keys())

        # Filter to features present in at least 70% of examples
        min_presence = int(len(valid_examples) * 0.7)
        feature_names = sorted([
            f for f in all_feature_names
            if sum(1 for fd in feature_dicts if f in fd) >= min_presence
        ])

        if len(feature_names) < 3:
            logger.warning("Too few common features across examples")
            return None

        # Build matrix
        X = np.zeros((len(valid_examples), len(feature_names)))
        for i, fd in enumerate(feature_dicts):
            for j, fname in enumerate(feature_names):
                X[i, j] = fd.get(fname, 0.0)

        y = np.array(labels)

        # Remove constant features
        non_constant = np.std(X, axis=0) > 1e-10
        X = X[:, non_constant]
        feature_names = [f for f, nc in zip(feature_names, non_constant) if nc]

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y, feature_names, valid_examples

    def _analyze_feature_discrimination(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, 'FeatureDiscriminator']:
        """
        For each feature, analyze how well it separates TAKE from SKIP.
        
        Uses:
        - t-test for significance
        - Cohen's d for effect size
        - AUC for discrimination power
        - Optimal threshold for decision boundary
        """
        discriminators = {}

        positive_mask = y == 1
        negative_mask = y == 0

        for i, fname in enumerate(feature_names):
            values = X[:, i]
            pos_values = values[positive_mask]
            neg_values = values[negative_mask]

            disc = FeatureDiscriminator(feature_name=fname)

            if len(pos_values) < 3 or len(neg_values) < 3:
                disc.strength = FeatureDiscriminatorStrength.NONE
                discriminators[fname] = disc
                continue

            # Basic statistics
            disc.good_mean = float(np.mean(pos_values))
            disc.good_std = float(np.std(pos_values))
            disc.bad_mean = float(np.mean(neg_values))
            disc.bad_std = float(np.std(neg_values))

            # t-test
            try:
                t_stat, p_value = stats.ttest_ind(
                    pos_values, neg_values, equal_var=False
                )
                disc.p_value = float(p_value)
            except Exception:
                disc.p_value = 1.0

            # Cohen's d (effect size)
            pooled_std = np.sqrt(
                (np.std(pos_values)**2 + np.std(neg_values)**2) / 2
            )
            if pooled_std > 1e-10:
                disc.effect_size = float(
                    abs(np.mean(pos_values) - np.mean(neg_values)) / pooled_std
                )
            else:
                disc.effect_size = 0.0

            # AUC (how well this single feature predicts)
            disc.auc = self._compute_single_feature_auc(values, y)

            # Optimal threshold
            threshold, direction, accuracy = self._find_optimal_threshold(
                values, y
            )
            disc.optimal_threshold = threshold
            disc.threshold_direction = direction
            disc.threshold_accuracy = accuracy

            # Classify strength
            if disc.p_value < 0.01 and disc.effect_size > 0.8:
                disc.strength = FeatureDiscriminatorStrength.STRONG
            elif disc.p_value < 0.05 and disc.effect_size > 0.5:
                disc.strength = FeatureDiscriminatorStrength.MODERATE
            elif disc.p_value < 0.1 and disc.effect_size > 0.3:
                disc.strength = FeatureDiscriminatorStrength.WEAK
            else:
                disc.strength = FeatureDiscriminatorStrength.NONE

            discriminators[fname] = disc

        return discriminators

    def _compute_single_feature_auc(
        self, values: np.ndarray, labels: np.ndarray
    ) -> float:
        """Compute AUC for a single feature (manual implementation)"""
        try:
            # Sort by feature value
            sorted_indices = np.argsort(values)
            sorted_labels = labels[sorted_indices]

            # Compute AUC via trapezoidal method
            n_pos = np.sum(labels == 1)
            n_neg = np.sum(labels == 0)

            if n_pos == 0 or n_neg == 0:
                return 0.5

            # Count concordant pairs
            concordant = 0
            for i in range(len(sorted_labels)):
                if sorted_labels[i] == 1:
                    # Count negative examples with lower feature value
                    concordant += np.sum(sorted_labels[:i] == 0)

            auc = concordant / (n_pos * n_neg)

            # AUC should be >= 0.5 (flip if feature direction is reversed)
            return max(auc, 1 - auc)

        except Exception:
            return 0.5

    def _find_optimal_threshold(
        self, values: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, str, float]:
        """
        Find the threshold that best separates positive from negative.
        Returns: (threshold, direction, accuracy)
        """
        best_threshold = float(np.median(values))
        best_accuracy = 0.5
        best_direction = "below"

        # Try percentile-based thresholds
        for percentile in range(10, 95, 5):
            threshold = float(np.percentile(values, percentile))

            # Try "good is below threshold"
            preds_below = (values <= threshold).astype(int)
            acc_below = np.mean(preds_below == labels)

            # Try "good is above threshold"
            preds_above = (values > threshold).astype(int)
            acc_above = np.mean(preds_above == labels)

            if acc_below > best_accuracy:
                best_accuracy = acc_below
                best_threshold = threshold
                best_direction = "below"

            if acc_above > best_accuracy:
                best_accuracy = acc_above
                best_threshold = threshold
                best_direction = "above"

        return best_threshold, best_direction, float(best_accuracy)

    def _train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Train classification model to predict user intent.
        
        Uses Gradient Boosting with careful regularization
        for small datasets (15-50 examples).
        
        IMPORTANT: This predicts "would user TAKE this setup?"
                   NOT "will this trade WIN?"
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score, LeaveOneOut
            from sklearn.model_selection import StratifiedKFold
        except ImportError:
            logger.error("sklearn not installed. Using fallback.")
            return self._train_fallback_model(X, y, feature_names)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_samples = len(X)

        # ── Model configuration based on dataset size ──
        if n_samples < 20:
            # Very small dataset — very conservative
            model = GradientBoostingClassifier(
                n_estimators=20,
                max_depth=2,
                learning_rate=0.1,
                min_samples_split=3,
                min_samples_leaf=2,
                subsample=0.8,
                max_features=min(5, X.shape[1]),
                random_state=42
            )
            # Use Leave-One-Out for very small datasets
            cv = LeaveOneOut()
        elif n_samples < 30:
            model = GradientBoostingClassifier(
                n_estimators=30,
                max_depth=2,
                learning_rate=0.1,
                min_samples_split=4,
                min_samples_leaf=2,
                subsample=0.8,
                max_features=min(8, X.shape[1]),
                random_state=42
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                max_features=min(10, X.shape[1]),
                random_state=42
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # ── Cross-validation ──
        try:
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            accuracy = float(np.mean(cv_scores))
            accuracy_std = float(np.std(cv_scores))
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}. Using simple split.")
            accuracy = 0.5
            accuracy_std = 0.2
            cv_scores = [0.5]

        # ── Train final model on all data ──
        model.fit(X_scaled, y)

        # ── Feature importance ──
        importances = model.feature_importances_
        feature_importance = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "model": model,
            "scaler": scaler,
            "accuracy": accuracy,
            "accuracy_std": accuracy_std,
            "cv_scores": cv_scores.tolist() if hasattr(cv_scores, 'tolist') else list(cv_scores),
            "feature_importance": feature_importance,
            "n_features": len(feature_names)
        }

    def _train_fallback_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Fallback when sklearn is not available.
        Uses simple Bayesian approach.
        """
        # Simple: for each feature, compute mean for positive and negative
        pos_mask = y == 1
        neg_mask = y == 0

        pos_means = np.mean(X[pos_mask], axis=0)
        neg_means = np.mean(X[neg_mask], axis=0)
        pos_stds = np.std(X[pos_mask], axis=0) + 1e-10
        neg_stds = np.std(X[neg_mask], axis=0) + 1e-10

        class SimpleBayesianModel:
            def __init__(self, pos_m, neg_m, pos_s, neg_s, prior):
                self.pos_means = pos_m
                self.neg_means = neg_m
                self.pos_stds = pos_s
                self.neg_stds = neg_s
                self.prior = prior

            def predict_proba(self, X_input):
                results = []
                for x in X_input:
                    # Log-likelihood
                    pos_ll = np.sum(
                        -0.5 * ((x - self.pos_means) / self.pos_stds) ** 2
                    )
                    neg_ll = np.sum(
                        -0.5 * ((x - self.neg_means) / self.neg_stds) ** 2
                    )
                    pos_ll += np.log(self.prior)
                    neg_ll += np.log(1 - self.prior)

                    # Convert to probability
                    max_ll = max(pos_ll, neg_ll)
                    pos_prob = np.exp(pos_ll - max_ll)
                    neg_prob = np.exp(neg_ll - max_ll)
                    total = pos_prob + neg_prob

                    results.append([neg_prob/total, pos_prob/total])
                return np.array(results)

            def predict(self, X_input):
                proba = self.predict_proba(X_input)
                return np.argmax(proba, axis=1)

        prior = np.mean(y)
        model = SimpleBayesianModel(pos_means, neg_means, pos_stds, neg_stds, prior)

        # Compute accuracy
        predictions = model.predict(X)
        accuracy = float(np.mean(predictions == y))

        # Feature importance (based on effect size)
        effect_sizes = np.abs(pos_means - neg_means) / (
            np.sqrt((pos_stds**2 + neg_stds**2) / 2)
        )
        feature_importance = sorted(
            zip(feature_names, effect_sizes),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "model": model,
            "scaler": None,
            "accuracy": accuracy,
            "accuracy_std": 0.1,
            "cv_scores": [accuracy],
            "feature_importance": [(f, float(imp)) for f, imp in feature_importance],
            "n_features": len(feature_names)
        }

    def _analyze_context_dependencies(
        self,
        examples: List['TrainingExample']
    ) -> List['ContextDependency']:
        """
        Discover context-dependent patterns.
        
        Checks: Does intent change based on:
        - Time of day
        - Regime / trend
        - Volatility level
        - Specific symbols
        """
        dependencies = []

        # Group examples by context dimensions
        context_dimensions = {
            "time_bucket": lambda e: e.features.get("time_bucket", "unknown"),
            "regime": lambda e: self._classify_regime(e.features),
            "volatility": lambda e: self._classify_volatility(e.features),
            "symbol": lambda e: e.symbol,
            "volume_condition": lambda e: self._classify_volume(e.features),
        }

        for dim_name, dim_extractor in context_dimensions.items():
            # Group examples by this dimension
            groups = defaultdict(lambda: {"positive": 0, "negative": 0, "total": 0})

            for ex in examples:
                try:
                    context_val = str(dim_extractor(ex))
                except Exception:
                    continue

                is_positive = ex.label.is_positive if hasattr(ex.label, 'is_positive') else True
                groups[context_val]["total"] += 1
                if is_positive:
                    groups[context_val]["positive"] += 1
                else:
                    groups[context_val]["negative"] += 1

            # Analyze each group
            for context_val, counts in groups.items():
                if counts["total"] < 3:
                    continue

                take_rate = counts["positive"] / counts["total"]

                # Chi-square test against overall rate
                overall_positive = sum(
                    1 for ex in examples
                    if hasattr(ex.label, 'is_positive') and ex.label.is_positive
                )
                overall_rate = overall_positive / len(examples) if examples else 0.5

                # Simple significance test
                try:
                    observed = [counts["positive"], counts["negative"]]
                    expected = [
                        counts["total"] * overall_rate,
                        counts["total"] * (1 - overall_rate)
                    ]
                    if min(expected) > 0:
                        chi2, p_value = stats.chisquare(observed, expected)
                    else:
                        p_value = 1.0
                except Exception:
                    p_value = 1.0

                # Determine direction
                if take_rate > overall_rate + 0.15:
                    direction = "favorable"
                elif take_rate < overall_rate - 0.15:
                    direction = "unfavorable"
                else:
                    direction = "neutral"

                dep = ContextDependency(
                    context_name=dim_name,
                    context_value=context_val,
                    positive_count=counts["positive"],
                    negative_count=counts["negative"],
                    total_count=counts["total"],
                    win_rate=take_rate,
                    significance=float(p_value),
                    is_significant=p_value < 0.1,
                    direction=direction,
                    insight=self._generate_context_insight(
                        dim_name, context_val, take_rate, direction
                    )
                )
                dependencies.append(dep)

        return dependencies

    def _find_interaction_effects(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Find pairs of features that interact
        (together predict better than individually).
        """
        interactions = []

        # Only check top 10 features to avoid combinatorial explosion
        n_check = min(10, len(feature_names))

        for i in range(n_check):
            for j in range(i+1, n_check):
                # Individual AUCs
                auc_i = self._compute_single_feature_auc(X[:, i], y)
                auc_j = self._compute_single_feature_auc(X[:, j], y)
                max_individual = max(auc_i, auc_j)

                # Combined AUC (simple: product of features)
                combined = X[:, i] * X[:, j]
                auc_combined = self._compute_single_feature_auc(combined, y)

                # Check if combination is meaningfully better
                if auc_combined > max_individual + 0.05:
                    interactions.append({
                        "feature_1": feature_names[i],
                        "feature_2": feature_names[j],
                        "auc_1": float(auc_i),
                        "auc_2": float(auc_j),
                        "auc_combined": float(auc_combined),
                        "synergy": float(auc_combined - max_individual),
                        "insight": (
                            f"{feature_names[i]} + {feature_names[j]} together "
                            f"predict {auc_combined:.0%} vs "
                            f"{max_individual:.0%} individually"
                        )
                    })

        # Sort by synergy
        interactions.sort(key=lambda x: x["synergy"], reverse=True)
        return interactions[:5]  # Top 5 interactions

    def _discover_decision_boundaries(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        discriminators: Dict[str, 'FeatureDiscriminator']
    ) -> Dict[str, Dict]:
        """
        For each important feature, find the precise boundary
        where user's decision changes from TAKE to SKIP.
        """
        boundaries = {}

        for fname, disc in discriminators.items():
            if disc.strength in (
                FeatureDiscriminatorStrength.STRONG,
                FeatureDiscriminatorStrength.MODERATE
            ):
                # Find the feature index
                if fname not in feature_names:
                    continue
                idx = feature_names.index(fname)

                # Get values at the boundary (sorted)
                sorted_indices = np.argsort(X[:, idx])
                sorted_values = X[sorted_indices, idx]
                sorted_labels = y[sorted_indices]

                # Find where labels transition
                transition_zone = self._find_transition_zone(
                    sorted_values, sorted_labels
                )

                boundaries[fname] = {
                    "threshold": disc.optimal_threshold,
                    "direction": disc.threshold_direction,
                    "accuracy": disc.threshold_accuracy,
                    "transition_zone": transition_zone,
                    "good_range": {
                        "min": float(np.min(X[y == 1, idx])),
                        "max": float(np.max(X[y == 1, idx])),
                        "mean": disc.good_mean,
                        "std": disc.good_std
                    },
                    "bad_range": {
                        "min": float(np.min(X[y == 0, idx])),
                        "max": float(np.max(X[y == 0, idx])),
                        "mean": disc.bad_mean,
                        "std": disc.bad_std
                    }
                }

        return boundaries

    def _find_transition_zone(
        self,
        sorted_values: np.ndarray,
        sorted_labels: np.ndarray
    ) -> Dict[str, float]:
        """Find where labels transition from mostly one class to another"""
        n = len(sorted_values)
        if n < 5:
            return {"start": float(sorted_values[0]),
                    "end": float(sorted_values[-1]),
                    "width": float(sorted_values[-1] - sorted_values[0])}

        # Sliding window: find where positive ratio changes most
        window = max(3, n // 5)
        ratios = []
        for i in range(n - window + 1):
            ratio = np.mean(sorted_labels[i:i+window])
            ratios.append(ratio)

        ratios = np.array(ratios)
        # Find maximum change
        if len(ratios) > 1:
            changes = np.abs(np.diff(ratios))
            max_change_idx = np.argmax(changes)
            transition_start = float(sorted_values[max_change_idx])
            transition_end = float(sorted_values[min(max_change_idx + window, n-1)])
        else:
            transition_start = float(sorted_values[n//3])
            transition_end = float(sorted_values[2*n//3])

        return {
            "start": transition_start,
            "end": transition_end,
            "width": transition_end - transition_start
        }

    def _extract_user_mentioned_features(
        self,
        examples: List['TrainingExample']
    ) -> List[str]:
        """Get list of features user explicitly mentioned in explanations"""
        all_mapped = set()
        for ex in examples:
            if ex.explanation_feature_map:
                for feature_list in ex.explanation_feature_map.values():
                    all_mapped.update(feature_list)
        return list(all_mapped)

    def _get_top_features(
        self,
        discriminators: Dict[str, 'FeatureDiscriminator']
    ) -> List[str]:
        """Get top discriminating features sorted by strength"""
        scored = []
        for fname, disc in discriminators.items():
            score = disc.effect_size * disc.auc
            if disc.p_value < 0.05:
                score *= 1.5
            scored.append((fname, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in scored[:15]]

    def _compute_data_sufficiency(
        self,
        n_total: int,
        n_positive: int,
        n_negative: int
    ) -> float:
        """How sufficient is the dataset for reliable learning"""
        # Size factor
        size_score = min(1.0, n_total / 30)

        # Balance factor
        if n_total > 0:
            min_class = min(n_positive, n_negative)
            balance = min_class / max(n_positive, n_negative, 1)
            balance_score = min(1.0, balance / 0.4)  # 40% ratio is acceptable
        else:
            balance_score = 0.0

        # Combined
        return size_score * 0.6 + balance_score * 0.4

    def _compute_overall_confidence(
        self,
        discovery: 'PatternDiscovery'
    ) -> float:
        """Compute overall confidence in the learned patterns"""
        factors = []

        # Model accuracy factor
        if discovery.model_accuracy > 0:
            factors.append(discovery.model_accuracy)

        # Data sufficiency
        factors.append(discovery.data_sufficiency)

        # Number of strong discriminators
        strong_count = sum(
            1 for d in discovery.feature_discriminators.values()
            if d.strength in (
                FeatureDiscriminatorStrength.STRONG,
                FeatureDiscriminatorStrength.MODERATE
            )
        )
        disc_score = min(1.0, strong_count / 5)
        factors.append(disc_score)

        return float(np.mean(factors)) if factors else 0.0

    # ── Helper classifiers for context analysis ──

    @staticmethod
    def _classify_regime(features: Dict[str, float]) -> str:
        trend = features.get("trend_direction_score", 0)
        if trend > 0.3:
            return "bullish"
        elif trend < -0.3:
            return "bearish"
        return "sideways"

    @staticmethod
    def _classify_volatility(features: Dict[str, float]) -> str:
        vol = features.get("range_volatility", 0)
        if vol > 0.7:
            return "high_volatility"
        elif vol < 0.3:
            return "low_volatility"
        return "medium_volatility"

    @staticmethod
    def _classify_volume(features: Dict[str, float]) -> str:
        vol_trend = features.get("volume_trend_slope", 0)
        if vol_trend < -0.3:
            return "declining_volume"
        elif vol_trend > 0.3:
            return "increasing_volume"
        return "stable_volume"

    def predict(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Predict user's intent for a new setup.
        
        Returns:
            {
                "prediction": "would_take" | "would_skip",
                "confidence": 0.78,
                "feature_contributions": {...},
                "explanation": "..."
            }
        """
        if self.trained_model is None:
            return {
                "prediction": "uncertain",
                "confidence": 0.0,
                "explanation": "Model not yet trained"
            }

        # Build feature vector
        vector = np.array([
            features.get(f, 0.0) for f in self.feature_names
        ]).reshape(1, -1)

        # Scale if scaler exists
        if self.scaler is not None:
            vector = self.scaler.transform(vector)

        # Predict
        proba = self.trained_model.predict_proba(vector)[0]
        prediction = "would_take" if proba[1] > 0.5 else "would_skip"
        confidence = float(max(proba))

        # Feature contributions (if model has feature_importances_)
        contributions = {}
        if hasattr(self.trained_model, 'feature_importances_'):
            for fname, imp in zip(
                self.feature_names,
                self.trained_model.feature_importances_
            ):
                if imp > 0.01:
                    val = features.get(fname, 0)
                    contributions[fname] = {
                        "importance": float(imp),
                        "value": float(val),
                        "direction": self._get_contribution_direction(fname, val)
                    }

        # Generate explanation
        explanation = self._generate_prediction_explanation(
            prediction, confidence, contributions
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probability_take": float(proba[1]),
            "probability_skip": float(proba[0]),
            "feature_contributions": contributions,
            "explanation": explanation
        }

    def _get_contribution_direction(
        self, feature_name: str, value: float
    ) -> str:
        """Is this feature value pushing toward TAKE or SKIP?"""
        if self.pattern_discovery is None:
            return "unknown"

        disc = self.pattern_discovery.feature_discriminators.get(feature_name)
        if disc is None:
            return "unknown"

        # Compare value to good/bad means
        dist_to_good = abs(value - disc.good_mean)
        dist_to_bad = abs(value - disc.bad_mean)

        if dist_to_good < dist_to_bad:
            return "toward_take"
        else:
            return "toward_skip"

    def _generate_prediction_explanation(
        self,
        prediction: str,
        confidence: float,
        contributions: Dict
    ) -> str:
        """Generate human-readable explanation"""
        top_contribs = sorted(
            contributions.items(),
            key=lambda x: x[1]["importance"],
            reverse=True
        )[:3]

        if prediction == "would_take":
            parts = [f"Prediction: TAKE (confidence: {confidence:.0%})"]
            take_reasons = [
                f"{fname} = {info['value']:.3f}"
                for fname, info in top_contribs
                if info["direction"] == "toward_take"
            ]
            if take_reasons:
                parts.append("Supporting: " + ", ".join(take_reasons))
        else:
            parts = [f"Prediction: SKIP (confidence: {confidence:.0%})"]
            skip_reasons = [
                f"{fname} = {info['value']:.3f}"
                for fname, info in top_contribs
                if info["direction"] == "toward_skip"
            ]
            if skip_reasons:
                parts.append("Against: " + ", ".join(skip_reasons))

        return " | ".join(parts)
