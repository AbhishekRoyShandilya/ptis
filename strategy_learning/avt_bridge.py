"""
============================================================
ONBOARDING → AVT SCORE BRIDGE
Defines EXACTLY how onboarding output feeds AVT scoring
============================================================

┌─────────────────────────────────────────────────────────────┐
│  ARCHITECTURAL RULE: ONE-WAY DATA FLOW ONLY                 │
│                                                             │
│  Direction:  Onboarding → AVT    ✅ (this module)           │
│  Direction:  AVT → Onboarding    ❌ FORBIDDEN               │
│                                                             │
│  AVT improvements STAY inside AVT.                          │
│  They optionally become SUGGESTIONS for the user.           │
│  They NEVER auto-update the user's strategy profile.        │
│                                                             │
│  If AVT discovers a better filter:                          │
│    ✅ Apply to AVT scoring config                           │
│    ✅ Show to user via Competition Engine as suggestion      │
│    ❌ DO NOT modify StrategyProfile or onboarding results   │
└─────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class AVTScoringConfig:
    """
    Configuration that bridges onboarding understanding
    to AVT's scoring system.
    
    This is the FINAL OUTPUT of onboarding that AVT uses.
    
    AVT Score = hard_rules_gate × (
        structure_score × w1 +
        judgment_score × w2 +
        context_score × w3 +
        regime_score × w4 +
        location_score × w5
    ) - uncertainty_penalty
    """
    strategy_id: str
    
    # ── Gate Layer (binary pass/fail) ──
    hard_rules: List[Dict[str, Any]] = field(default_factory=list)
    # e.g., [{"feature": "range_contraction_r2", "operator": ">", "value": 0.5}]
    
    # ── Structure Score ──
    structure_model: Any = None         # Trained model for structure quality
    structure_features: List[str] = field(default_factory=list)
    structure_weight: float = 0.25
    structure_confidence: float = 0.0
    
    # ── Judgment Score ──
    judgment_model: Any = None          # Trained model for subjective quality
    judgment_features: List[str] = field(default_factory=list)
    judgment_weight: float = 0.25
    judgment_confidence: float = 0.0
    
    # ── Context Score ──
    context_rules: List[Dict] = field(default_factory=list)
    context_weight: float = 0.20
    context_confidence: float = 0.0
    
    # ── Regime Score ──
    regime_definition: Any = None       # RegimeDefinition object
    regime_weight: float = 0.15
    regime_confidence: float = 0.0
    
    # ── Location Score ──
    location_preferences: Dict[str, Any] = field(default_factory=dict)
    location_weight: float = 0.15
    location_confidence: float = 0.0
    
    # ── Uncertainty ──
    uncertainty_penalties: Dict[str, float] = field(default_factory=dict)
    overall_uncertainty: float = 0.3
    
    # ── Metadata ──
    onboarding_accuracy: float = 0.0
    n_training_examples: int = 0
    last_updated: str = ""


class OnboardingToAVTBridge:
    """
    Converts onboarding results into AVT-ready scoring configuration.
    
    This is the CRITICAL LINK between:
    - Onboarding (understanding user's strategy)
    - AVT (autonomously evaluating setups)
    """
    
    def build_avt_config(
        self,
        onboarding_sko: 'OnboardingSKO',
        confidence_decomposition: 'ConfidenceDecomposition',
        pattern_discovery: 'PatternDiscovery',
        regime_definition: 'RegimeDefinition'
    ) -> AVTScoringConfig:
        """Build AVT scoring config from onboarding results"""
        
        config = AVTScoringConfig(
            strategy_id=onboarding_sko.strategy_id
        )
        
        # ── Hard Rules (Gate) ──
        config.hard_rules = self._build_hard_rules(onboarding_sko, pattern_discovery)
        
        # ── Structure Score ──
        config.structure_model = onboarding_sko.learned_model
        config.structure_features = [
            f for f, imp in pattern_discovery.feature_importance_ranking
            if f in self._get_structure_features()
        ]
        config.structure_weight = 0.25
        config.structure_confidence = confidence_decomposition.structure_confidence
        
        # ── Judgment Score ──
        config.judgment_model = onboarding_sko.learned_model  # Same model, different features
        config.judgment_features = [
            f for f, imp in pattern_discovery.feature_importance_ranking
            if f in self._get_judgment_features()
        ]
        config.judgment_weight = 0.25
        config.judgment_confidence = confidence_decomposition.judgment_confidence
        
        # ── Context Score ──
        config.context_rules = self._build_context_rules(pattern_discovery)
        config.context_weight = 0.20
        config.context_confidence = confidence_decomposition.context_confidence
        
        # ── Regime Score ──
        config.regime_definition = regime_definition
        config.regime_weight = 0.15
        config.regime_confidence = confidence_decomposition.regime_confidence
        
        # ── Location Score ──
        config.location_preferences = self._build_location_preferences(
            pattern_discovery
        )
        config.location_weight = 0.15
        config.location_confidence = confidence_decomposition.location_confidence
        
        # ── Uncertainty ──
        config.uncertainty_penalties = self._compute_uncertainty_penalties(
            confidence_decomposition
        )
        config.overall_uncertainty = 1.0 - confidence_decomposition.overall_confidence
        
        # ── Adjust weights based on confidence ──
        # Components with higher confidence get more weight
        self._adjust_weights_by_confidence(config)
        
        # ── Metadata ──
        config.onboarding_accuracy = pattern_discovery.model_accuracy
        config.n_training_examples = pattern_discovery.n_examples
        
        return config
    
    def _build_hard_rules(
        self,
        sko: 'OnboardingSKO',
        discovery: 'PatternDiscovery'
    ) -> List[Dict]:
        """
        Build hard rules (binary gate).
        Only from features with STRONG discrimination and clear thresholds.
        """
        rules = []
        
        # From SKO hard rules (user explicitly stated)
        for rule in sko.hard_rules:
            rules.append(rule)
        
        # From discovered patterns — ONLY if very strong
        for fname, disc in discovery.feature_discriminators.items():
            if (disc.strength == FeatureDiscriminatorStrength.STRONG and
                disc.optimal_threshold is not None and
                disc.threshold_accuracy > 0.85):
                
                rules.append({
                    "feature": fname,
                    "operator": "<" if disc.threshold_direction == "below" else ">",
                    "value": disc.optimal_threshold,
                    "source": "learned",
                    "confidence": disc.threshold_accuracy
                })
        
        return rules
    
    def _build_context_rules(
        self, discovery: 'PatternDiscovery'
    ) -> List[Dict]:
        """Build context scoring rules from discovered dependencies"""
        rules = []
        
        for dep in discovery.context_dependencies:
            if dep.is_significant:
                rules.append({
                    "context": dep.context_name,
                    "value": dep.context_value,
                    "direction": dep.direction,
                    "win_rate_impact": dep.win_rate,
                    "significance": dep.significance
                })
        
        return rules
    
    def _build_location_preferences(
        self, discovery: 'PatternDiscovery'
    ) -> Dict[str, Any]:
        """Build location preference profile"""
        location_features = [
            "distance_to_nearest_support",
            "distance_to_nearest_resistance",
            "price_percentile_in_day_range"
        ]
        
        preferences = {}
        for fname in location_features:
            disc = discovery.feature_discriminators.get(fname)
            if disc and disc.strength != FeatureDiscriminatorStrength.NONE:
                preferences[fname] = {
                    "preferred_range": (
                        disc.good_mean - disc.good_std,
                        disc.good_mean + disc.good_std
                    ),
                    "optimal": disc.good_mean,
                    "avoid_range": (
                        disc.bad_mean - disc.bad_std,
                        disc.bad_mean + disc.bad_std
                    )
                }
        
        return preferences
    
    def _compute_uncertainty_penalties(
        self, decomp: 'ConfidenceDecomposition'
    ) -> Dict[str, float]:
        """
        Compute per-component uncertainty penalties.
        Lower confidence = higher penalty.
        """
        return {
            "structure": max(0, 0.5 - decomp.structure_confidence) * 0.5,
            "judgment": max(0, 0.5 - decomp.judgment_confidence) * 0.5,
            "context": max(0, 0.5 - decomp.context_confidence) * 0.3,
            "regime": max(0, 0.5 - decomp.regime_confidence) * 0.3,
            "location": max(0, 0.5 - decomp.location_confidence) * 0.3,
        }
    
    def _adjust_weights_by_confidence(self, config: AVTScoringConfig):
        """
        Dynamically adjust weights: more confident components get more weight.
        This prevents low-confidence components from dominating.
        """
        raw_weights = {
            "structure": config.structure_weight * (0.5 + config.structure_confidence * 0.5),
            "judgment": config.judgment_weight * (0.5 + config.judgment_confidence * 0.5),
            "context": config.context_weight * (0.5 + config.context_confidence * 0.5),
            "regime": config.regime_weight * (0.5 + config.regime_confidence * 0.5),
            "location": config.location_weight * (0.5 + config.location_confidence * 0.5),
        }
        
        total = sum(raw_weights.values())
        if total > 0:
            config.structure_weight = raw_weights["structure"] / total
            config.judgment_weight = raw_weights["judgment"] / total
            config.context_weight = raw_weights["context"] / total
            config.regime_weight = raw_weights["regime"] / total
            config.location_weight = raw_weights["location"] / total
    
    def compute_avt_score(
        self,
        config: AVTScoringConfig,
        features: Dict[str, float],
        regime_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        THIS IS THE FUNCTION AVT CALLS FOR EVERY SETUP.
        
        Computes final score using onboarding-learned config.
        
        Returns:
            {
                "final_score": 0.72,
                "hard_rules_passed": True,
                "component_scores": {
                    "structure": 0.81,
                    "judgment": 0.68,
                    "context": 0.75,
                    "regime": 0.82,
                    "location": 0.70
                },
                "uncertainty_penalty": 0.08,
                "decision": "candidate",  # | "strong" | "skip" | "uncertain"
                "avt_mode": "candidate_presenter",
                "explanation": "..."
            }
        """
        result = {
            "hard_rules_passed": True,
            "component_scores": {},
            "uncertainty_penalty": 0.0,
            "rule_failures": []
        }
        
        # ── Step 1: Hard Rules Gate ──
        for rule in config.hard_rules:
            feature_val = features.get(rule["feature"])
            if feature_val is None:
                continue
            
            op = rule["operator"]
            threshold = rule["value"]
            
            passed = (
                (op == ">" and feature_val > threshold) or
                (op == "<" and feature_val < threshold) or
                (op == ">=" and feature_val >= threshold) or
                (op == "<=" and feature_val <= threshold)
            )
            
            if not passed:
                result["hard_rules_passed"] = False
                result["rule_failures"].append(
                    f"{rule['feature']} = {feature_val:.3f} "
                    f"(need {op} {threshold})"
                )
        
        if not result["hard_rules_passed"]:
            result["final_score"] = 0.0
            result["decision"] = "skip"
            result["avt_mode"] = "observer"
            result["explanation"] = (
                f"Failed hard rules: {', '.join(result['rule_failures'])}"
            )
            return result
        
        # ── Step 2: Component Scores ──
        
        # Structure score (from model)
        structure_score = self._compute_model_score(
            config.structure_model, features, config.structure_features
        )
        result["component_scores"]["structure"] = structure_score
        
        # Judgment score
        judgment_score = self._compute_model_score(
            config.judgment_model, features, config.judgment_features
        )
        result["component_scores"]["judgment"] = judgment_score
        
        # Context score
        context_score = self._compute_context_score(
            config.context_rules, features
        )
        result["component_scores"]["context"] = context_score
        
        # Regime score
        regime_score = 0.5  # neutral default
        if config.regime_definition:
            regime_eval = config.regime_definition.evaluate(regime_data)
            regime_score = regime_eval["regime_match"]
        result["component_scores"]["regime"] = regime_score
        
        # Location score
        location_score = self._compute_location_score(
            config.location_preferences, features
        )
        result["component_scores"]["location"] = location_score
        
        # ── Step 3: Weighted Combination ──
        weighted_score = (
            structure_score * config.structure_weight +
            judgment_score * config.judgment_weight +
            context_score * config.context_weight +
            regime_score * config.regime_weight +
            location_score * config.location_weight
        )
        
        # ── Step 4: Uncertainty Penalty ──
        total_penalty = sum(config.uncertainty_penalties.values())
        result["uncertainty_penalty"] = total_penalty
        
        final_score = max(0, weighted_score - total_penalty)
        result["final_score"] = final_score
        
        # ── Step 5: Decision ──
        if final_score >= 0.75:
            result["decision"] = "strong"
            result["avt_mode"] = "independent"
        elif final_score >= 0.55:
            result["decision"] = "candidate"
            result["avt_mode"] = "candidate_presenter"
        elif final_score >= 0.35:
            result["decision"] = "uncertain"
            result["avt_mode"] = "candidate_presenter"
        else:
            result["decision"] = "skip"
            result["avt_mode"] = "observer"
        
        # ── Step 6: Explanation ──
        result["explanation"] = self._generate_explanation(result, config)
        
        return result
    
    def _compute_model_score(
        self, model, features: Dict, feature_names: List[str]
    ) -> float:
        """Get probability from trained model"""
        if model is None or not feature_names:
            return 0.5
        
        try:
            vector = np.array([
                features.get(f, 0) for f in feature_names
            ]).reshape(1, -1)
            proba = model.predict_proba(vector)[0]
            return float(proba[1])  # probability of "would_take"
        except Exception:
            return 0.5
    
    def _compute_context_score(
        self, context_rules: List[Dict], features: Dict
    ) -> float:
        """Compute context alignment score"""
        if not context_rules:
            return 0.5
        
        scores = []
        for rule in context_rules:
            context_val = features.get(rule.get("context", ""))
            if context_val is not None:
                if rule["direction"] == "favorable":
                    scores.append(0.8)
                elif rule["direction"] == "unfavorable":
                    scores.append(0.2)
                else:
                    scores.append(0.5)
        
        return np.mean(scores) if scores else 0.5
    
    def _compute_location_score(
        self, preferences: Dict, features: Dict
    ) -> float:
        """Score based on price location preferences"""
        if not preferences:
            return 0.5
        
        scores = []
        for feature_name, pref in preferences.items():
            val = features.get(feature_name)
            if val is None:
                continue
            
            optimal = pref["optimal"]
            low, high = pref["preferred_range"]
            
            if low <= val <= high:
                # Within preferred range
                distance_from_optimal = abs(val - optimal) / max(abs(high - low), 0.01)
                scores.append(0.7 + 0.3 * (1 - distance_from_optimal))
            else:
                # Outside preferred range
                if val < low:
                    distance = (low - val) / max(abs(low), 0.01)
                else:
                    distance = (val - high) / max(abs(high), 0.01)
                scores.append(max(0.1, 0.5 - distance * 0.3))
        
        return np.mean(scores) if scores else 0.5
    
    def _generate_explanation(
        self, result: Dict, config: AVTScoringConfig
    ) -> str:
        """Generate human-readable explanation of score"""
        scores = result["component_scores"]
        
        # Find strongest and weakest
        strongest = max(scores, key=scores.get)
        weakest = min(scores, key=scores.get)
        
        explanation = (
            f"Score: {result['final_score']:.2f} | "
            f"Best: {strongest} ({scores[strongest]:.2f}) | "
            f"Weakest: {weakest} ({scores[weakest]:.2f})"
        )
        
        if result["uncertainty_penalty"] > 0.1:
            explanation += f" | Uncertainty penalty: -{result['uncertainty_penalty']:.2f}"
        
        return explanation


    @staticmethod
    def _get_structure_features() -> set:
        return {
            "wick_ratio_mean", "body_consistency", "body_size_variance",
            "range_contraction_r2", "squeeze_score", "compression_candle_count",
            "avg_body_size", "body_to_range_ratio", "range_volatility"
        }
    
    @staticmethod
    def _get_judgment_features() -> set:
        return {
            "direction_entropy", "volume_consistency", "range_volatility",
            "body_direction_ratio", "wick_ratio_max",
            "consecutive_same_direction_max"
        }
