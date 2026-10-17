"""
============================================================
ONBOARDING SKO BUILDER
Converts everything learned into a production-ready SKO
============================================================
File: soe/sko_builder.py
============================================================

This is where ALL onboarding intelligence gets packaged
into a single object that the rest of PTIS consumes.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json

logger = logging.getLogger("SOE.SKOBuilder")


class OnboardingSKOBuilder:
    """
    Builds the final Strategy Knowledge Object from:
    - IntakeAssessor results (hard rules from text)
    - PatternDiscovery results (learned from examples)  
    - RegimeDefinition (formalized market conditions)
    - ConfidenceDecomposition (per-component confidence)
    - UnderstandingReport (validated understanding)
    
    The output SKO feeds directly into:
    - StrategyUnderstandingEngine (existing)
    - AVT scoring (via AVTBridge)
    - JudgmentLearner (as initial priors)
    """

    def build(
        self,
        strategy_id: str,
        intake_results: Optional[Dict] = None,
        pattern_discovery: Optional['PatternDiscovery'] = None,
        pattern_learner: Optional['OnboardingPatternLearner'] = None,
        regime_definition: Optional['RegimeDefinition'] = None,
        confidence_decomposition: Optional['ConfidenceDecomposition'] = None,
        understanding_report: Optional['UnderstandingReport'] = None,
        quiz_results: Optional[Dict] = None
    ) -> 'OnboardingSKO':
        """
        Build the complete OnboardingSKO.
        
        This is the FINAL output of the entire onboarding process.
        """
        logger.info(f"Building OnboardingSKO for {strategy_id}")

        sko = OnboardingSKO(strategy_id=strategy_id)

        # ── 1. Hard Rules (from IntakeAssessor) ──
        if intake_results:
            sko.hard_rules = self._extract_hard_rules(intake_results)
            logger.info(f"Hard rules from intake: {len(sko.hard_rules)}")

        # ── 2. Learned Soft Scorers (from PatternDiscovery) ──
        if pattern_discovery and pattern_learner:
            sko.soft_scorers = self._build_soft_scorers(pattern_discovery)
            sko.learned_model = pattern_learner.trained_model
            sko.learned_scaler = pattern_learner.scaler
            sko.feature_names = pattern_learner.feature_names

            # Feature importance
            sko.feature_importance = {
                fname: float(imp)
                for fname, imp in pattern_discovery.feature_importance_ranking
            }

            # Decision boundaries
            sko.decision_boundaries = pattern_discovery.decision_boundaries

            logger.info(
                f"Soft scorers built. Model accuracy: "
                f"{pattern_discovery.model_accuracy:.2%}"
            )

        # ── 3. Context Factors (from PatternDiscovery) ──
        if pattern_discovery:
            sko.context_factors = self._build_context_factors(
                pattern_discovery
            )

        # ── 4. Regime Dependency (from RegimeDefinition) ──
        if regime_definition:
            sko.regime_dependency = self._build_regime_dependency(
                regime_definition
            )

        # ── 5. Uncertainty Map ──
        if confidence_decomposition:
            sko.uncertainty_map = self._build_uncertainty_map(
                confidence_decomposition, pattern_discovery
            )

        # ── 6. Bayesian Priors (for JudgmentLearner) ──
        if pattern_discovery:
            sko.bayesian_priors = self._build_bayesian_priors(
                pattern_discovery
            )

        # ── 7. Training Metadata ──
        sko.training_metadata = self._build_metadata(
            pattern_discovery, quiz_results
        )

        # ── 8. Attach full objects for reference ──
        sko.pattern_discovery = pattern_discovery
        sko.understanding_report = understanding_report

        # ── 9. Validate SKO quality ──
        quality = self._validate_sko_quality(sko)
        sko.training_metadata["sko_quality"] = quality

        logger.info(
            f"OnboardingSKO built. Quality: {quality['overall_quality']:.2%}. "
            f"Confidence: {sko.get_confidence():.2%}"
        )

        return sko

    def _extract_hard_rules(
        self, intake_results: Dict
    ) -> List[Dict[str, Any]]:
        """Extract hard rules from IntakeAssessor results"""
        rules = []

        # From intake components classified as "rule"
        components = intake_results.get("components", [])
        for comp in components:
            if comp.get("type") == "rule":
                rule = {
                    "description": comp.get("description", ""),
                    "feature": comp.get("feature", ""),
                    "operator": comp.get("operator", ""),
                    "value": comp.get("value", None),
                    "source": "intake_assessor",
                    "confidence": comp.get("confidence", 0.5)
                }
                if rule["feature"] and rule["operator"] and rule["value"] is not None:
                    rules.append(rule)

        return rules

    def _build_soft_scorers(
        self, discovery: 'PatternDiscovery'
    ) -> Dict[str, Any]:
        """Build soft scoring configuration from learned patterns"""
        scorers = {
            "source": "learned_from_examples",
            "n_examples": discovery.n_examples,
            "accuracy": discovery.model_accuracy,
            "feature_profiles": {}
        }

        for fname, disc in discovery.feature_discriminators.items():
            if disc.strength == FeatureDiscriminatorStrength.NONE:
                continue

            scorers["feature_profiles"][fname] = {
                "strength": disc.strength.value,
                "good_mean": disc.good_mean,
                "good_std": disc.good_std,
                "bad_mean": disc.bad_mean,
                "bad_std": disc.bad_std,
                "effect_size": disc.effect_size,
                "threshold": disc.optimal_threshold,
                "threshold_direction": disc.threshold_direction,
                "threshold_accuracy": disc.threshold_accuracy,
                "auc": disc.auc
            }

        return scorers

    def _build_context_factors(
        self, discovery: 'PatternDiscovery'
    ) -> Dict[str, Any]:
        """Build context factor configuration"""
        factors = {
            "dependencies": [],
            "interaction_effects": discovery.interaction_effects,
            "hidden_features": discovery.hidden_features
        }

        for dep in discovery.context_dependencies:
            if dep.is_significant:
                factors["dependencies"].append({
                    "context": dep.context_name,
                    "value": dep.context_value,
                    "direction": dep.direction,
                    "take_rate": dep.win_rate,
                    "significance": dep.significance,
                    "insight": dep.insight
                })

        return factors

    def _build_regime_dependency(
        self, regime_def: 'RegimeDefinition'
    ) -> Dict[str, Any]:
        """Build regime dependency configuration"""
        return {
            "regime_type": regime_def.regime_type.value,
            "conditions": [
                {
                    "indicator": c.indicator,
                    "operator": c.operator,
                    "value": c.value,
                    "weight": c.weight,
                    "learned": c.learned
                }
                for c in regime_def.conditions
            ],
            "confidence": regime_def.confidence,
            "source": regime_def.source,
            "favorable_regimes": regime_def.favorable_regimes,
            "unfavorable_regimes": regime_def.unfavorable_regimes
        }

    def _build_uncertainty_map(
        self,
        confidence: 'ConfidenceDecomposition',
        discovery: Optional['PatternDiscovery']
    ) -> Dict[str, Any]:
        """Build uncertainty map for each component"""
        uncertainty = {
            "high_confidence_features": [],
            "medium_confidence_features": [],
            "low_confidence_features": [],
            "needs_more_examples": [],
            "component_uncertainty": {
                "structure": 1.0 - confidence.structure_confidence,
                "judgment": 1.0 - confidence.judgment_confidence,
                "context": 1.0 - confidence.context_confidence,
                "regime": 1.0 - confidence.regime_confidence,
                "location": 1.0 - confidence.location_confidence,
            }
        }

        if discovery:
            for fname, disc in discovery.feature_discriminators.items():
                if disc.auc > 0.75:
                    uncertainty["high_confidence_features"].append(fname)
                elif disc.auc > 0.6:
                    uncertainty["medium_confidence_features"].append(fname)
                elif disc.strength != FeatureDiscriminatorStrength.NONE:
                    uncertainty["low_confidence_features"].append(fname)
                else:
                    uncertainty["needs_more_examples"].append(fname)

        return uncertainty

    def _build_bayesian_priors(
        self, discovery: 'PatternDiscovery'
    ) -> Dict[str, Dict]:
        """
        Build Bayesian priors for JudgmentLearner integration.
        
        These serve as STARTING POINT for the live learning system.
        Instead of starting from flat priors, JudgmentLearner
        starts from onboarding-learned distributions.
        """
        priors = {}

        for fname, disc in discovery.feature_discriminators.items():
            if disc.strength == FeatureDiscriminatorStrength.NONE:
                continue

            # Build prior distribution for this feature
            priors[fname] = {
                "take_distribution": {
                    "mean": disc.good_mean,
                    "std": max(disc.good_std, 0.01),
                    "n_observations": discovery.n_positive
                },
                "skip_distribution": {
                    "mean": disc.bad_mean,
                    "std": max(disc.bad_std, 0.01),
                    "n_observations": discovery.n_negative
                },
                "prior_strength": self._compute_prior_strength(disc),
                "update_rate": self._compute_update_rate(disc)
            }

        return priors

    def _compute_prior_strength(
        self, disc: 'FeatureDiscriminator'
    ) -> float:
        """
        How strongly should JudgmentLearner weight these priors?
        
        Strong discrimination + many examples = strong prior
        Weak discrimination = weak prior (easy to override with new data)
        """
        strength_map = {
            FeatureDiscriminatorStrength.STRONG: 0.8,
            FeatureDiscriminatorStrength.MODERATE: 0.5,
            FeatureDiscriminatorStrength.WEAK: 0.3,
            FeatureDiscriminatorStrength.NONE: 0.1,
        }
        return strength_map.get(disc.strength, 0.1)

    def _compute_update_rate(
        self, disc: 'FeatureDiscriminator'
    ) -> float:
        """
        How quickly should new data update these priors?
        
        Low confidence = fast update (learn quickly from new data)
        High confidence = slow update (stable, don't overreact)
        """
        if disc.strength == FeatureDiscriminatorStrength.STRONG:
            return 0.05  # Slow update — we're confident
        elif disc.strength == FeatureDiscriminatorStrength.MODERATE:
            return 0.1
        elif disc.strength == FeatureDiscriminatorStrength.WEAK:
            return 0.2   # Fast update — not confident, learn quickly
        else:
            return 0.3   # Very fast — basically uncertain

    def _build_metadata(
        self,
        discovery: Optional['PatternDiscovery'],
        quiz_results: Optional[Dict]
    ) -> Dict[str, Any]:
        """Build training metadata"""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "onboarding_v1",
            "n_examples": discovery.n_examples if discovery else 0,
            "n_positive": discovery.n_positive if discovery else 0,
            "n_negative": discovery.n_negative if discovery else 0,
            "model_accuracy": discovery.model_accuracy if discovery else 0,
            "model_accuracy_std": discovery.model_accuracy_std if discovery else 0,
            "data_sufficiency": discovery.data_sufficiency if discovery else 0,
            "overall_confidence": discovery.overall_confidence if discovery else 0,
        }

        if quiz_results:
            metadata["quiz_accuracy"] = quiz_results.get("accuracy", 0)
            metadata["quiz_total"] = quiz_results.get("total", 0)
            metadata["quiz_assessment"] = quiz_results.get("assessment", "")

        return metadata

    def _validate_sko_quality(self, sko: 'OnboardingSKO') -> Dict[str, Any]:
        """
        Validate the quality of the built SKO.
        
        Checks:
        - Minimum components present
        - Model quality
        - Data sufficiency
        - Balance of components
        """
        quality_checks = {}

        # Has hard rules?
        quality_checks["has_hard_rules"] = len(sko.hard_rules) > 0

        # Has learned model?
        quality_checks["has_model"] = sko.learned_model is not None

        # Model accuracy acceptable?
        acc = sko.training_metadata.get("model_accuracy", 0)
        quality_checks["model_accuracy_acceptable"] = acc > 0.6

        # Has sufficient features?
        quality_checks["has_features"] = len(sko.feature_names) >= 5

        # Has context factors?
        quality_checks["has_context"] = bool(sko.context_factors.get("dependencies"))

        # Has regime definition?
        quality_checks["has_regime"] = bool(sko.regime_dependency)

        # Has uncertainty map?
        quality_checks["has_uncertainty"] = bool(sko.uncertainty_map)

        # Has Bayesian priors?
        quality_checks["has_priors"] = len(sko.bayesian_priors) >= 3

        # Overall quality score
        passed = sum(1 for v in quality_checks.values() if v)
        total = len(quality_checks)
        quality_checks["overall_quality"] = passed / total

        # Quality tier
        ratio = passed / total
        if ratio >= 0.85:
            quality_checks["tier"] = "production_ready"
        elif ratio >= 0.65:
            quality_checks["tier"] = "usable_with_caution"
        elif ratio >= 0.45:
            quality_checks["tier"] = "needs_improvement"
        else:
            quality_checks["tier"] = "insufficient"

        return quality_checks
