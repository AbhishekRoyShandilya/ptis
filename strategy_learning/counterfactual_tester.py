"""
============================================================
COUNTERFACTUAL TESTER (Hook for P1)
============================================================
File: soe/counterfactual_tester.py
============================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CounterfactualQuestion:
    """A 'what if' question to deepen understanding"""
    feature_name: str
    original_value: float
    modified_value: float
    question_text: str
    predicted_change: str       # "would_still_take" | "would_now_skip" | "uncertain"
    confidence: float = 0.0
    user_answer: Optional[str] = None


class CounterfactualTester:
    """
    Generates 'what if' questions to test and deepen understanding.
    
    Uses the learned model to identify:
    - Boundary features (small change = different decision)
    - Robust features (large change still same decision)
    - Hidden dependencies
    """
    
    def __init__(self, learned_model=None, feature_importance: Dict[str, float] = None):
        self.model = learned_model
        self.feature_importance = feature_importance or {}
    
    def generate_counterfactuals(
        self,
        base_features: Dict[str, float],
        top_n: int = 5
    ) -> List[CounterfactualQuestion]:
        """
        Generate counterfactual questions for the most important features.
        
        For each important feature:
        - Nudge it toward the "bad" direction
        - Ask user: "Would you still take it?"
        """
        questions = []
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        for feature_name, importance in sorted_features:
            if feature_name not in base_features:
                continue
            
            original = base_features[feature_name]
            
            # Generate meaningful perturbation
            modified, direction_text = self._generate_perturbation(
                feature_name, original
            )
            
            if modified is None:
                continue
            
            # Predict what model thinks
            predicted = self._predict_with_change(
                base_features, feature_name, modified
            )
            
            question = CounterfactualQuestion(
                feature_name=feature_name,
                original_value=original,
                modified_value=modified,
                question_text=(
                    f"Your setup has {feature_name} = {original:.3f}. "
                    f"What if it was {modified:.3f} instead ({direction_text})? "
                    f"Would you still take this trade?"
                ),
                predicted_change=predicted["decision"],
                confidence=predicted["confidence"]
            )
            questions.append(question)
        
        return questions
    
    def _generate_perturbation(
        self, feature_name: str, original: float
    ) -> Tuple[Optional[float], str]:
        """Generate meaningful 'what if' change for a feature"""
        
        # Feature-specific perturbations
        perturbation_map = {
            "wick_ratio_mean": (0.15, "more wicky candles"),
            "body_size_variance": (0.15, "more inconsistent bodies"),
            "range_contraction_r2": (-0.2, "less clean compression"),
            "volume_trend_slope": (0.3, "volume increasing instead"),
            "direction_entropy": (0.2, "more random direction"),
            "distance_to_nearest_support": (1.5, "further from support"),
            "distance_to_nearest_resistance": (-1.0, "closer to resistance"),
            "squeeze_score": (-0.2, "weaker squeeze"),
        }
        
        if feature_name in perturbation_map:
            delta, text = perturbation_map[feature_name]
            return original + delta, text
        
        # Generic: move 30% toward unfavorable
        return original * 1.3, "30% worse"
    
    def _predict_with_change(
        self,
        base_features: Dict[str, float],
        changed_feature: str,
        new_value: float
    ) -> Dict[str, Any]:
        """Predict decision with one feature changed"""
        if self.model is None:
            return {"decision": "uncertain", "confidence": 0.0}
        
        modified = base_features.copy()
        modified[changed_feature] = new_value
        
        try:
            # Use model to predict
            feature_vector = np.array([
                modified.get(f, 0) for f in sorted(modified.keys())
            ]).reshape(1, -1)
            
            proba = self.model.predict_proba(feature_vector)[0]
            predicted_class = int(np.argmax(proba))
            confidence = float(np.max(proba))
            
            return {
                "decision": "would_still_take" if predicted_class == 1 
                           else "would_now_skip",
                "confidence": confidence
            }
        except Exception:
            return {"decision": "uncertain", "confidence": 0.0}
    
    def process_user_answers(
        self,
        questions: List[CounterfactualQuestion]
    ) -> Dict[str, Any]:
        """
        Process user's answers to counterfactual questions.
        Returns insights about decision boundaries.
        """
        insights = []
        boundary_features = []
        robust_features = []
        
        for q in questions:
            if q.user_answer is None:
                continue
            
            if q.user_answer == "no":  # User would now skip
                boundary_features.append({
                    "feature": q.feature_name,
                    "boundary_near": q.modified_value,
                    "sensitivity": "high"
                })
                insights.append(
                    f"{q.feature_name} is a BOUNDARY feature — "
                    f"changing from {q.original_value:.3f} to "
                    f"{q.modified_value:.3f} flips your decision"
                )
            else:  # User would still take
                robust_features.append({
                    "feature": q.feature_name,
                    "robust_range": (q.original_value, q.modified_value),
                    "sensitivity": "low"
                })
                insights.append(
                    f"{q.feature_name} is ROBUST — "
                    f"even at {q.modified_value:.3f} you'd still take it"
                )
        
        return {
            "boundary_features": boundary_features,
            "robust_features": robust_features,
            "insights": insights,
            "model_accuracy_on_counterfactuals": self._check_model_accuracy(questions)
        }
    
    def _check_model_accuracy(
        self, questions: List[CounterfactualQuestion]
    ) -> float:
        """How well did model predict user's counterfactual answers"""
        answered = [q for q in questions if q.user_answer is not None]
        if not answered:
            return 0.0
        
        correct = 0
        for q in answered:
            model_says_skip = q.predicted_change == "would_now_skip"
            user_says_skip = q.user_answer == "no"
            if model_says_skip == user_says_skip:
                correct += 1
        
        return correct / len(answered)
