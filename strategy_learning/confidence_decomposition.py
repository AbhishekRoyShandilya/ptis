"""
============================================================
CONFIDENCE DECOMPOSITION
Breaks overall confidence into component-level confidence
============================================================
File: soe/confidence_decomposition.py
============================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ConfidenceDecomposition:
    """
    Instead of: confidence = 76%
    
    Shows:
    ┌──────────────────────────────────────────┐
    │ Structure understanding:    82%          │
    │ Judgment understanding:     71%          │
    │ Context understanding:      68%          │
    │ Regime understanding:       74%          │
    │ Location understanding:     79%          │
    │                                          │
    │ OVERALL:                    76%          │
    │                                          │
    │ WEAKEST LINK: Context (68%)              │
    │ → Need more diverse context examples     │
    └──────────────────────────────────────────┘
    """
    
    # Component confidences (0-1 each)
    structure_confidence: float = 0.0       # Candle patterns, compression quality
    judgment_confidence: float = 0.0        # Subjective "clean/ugly" understanding
    context_confidence: float = 0.0         # Market context dependencies
    regime_confidence: float = 0.0          # When strategy works/fails
    location_confidence: float = 0.0        # Where in price structure
    
    # How each was measured
    structure_evidence: str = ""
    judgment_evidence: str = ""
    context_evidence: str = ""
    regime_evidence: str = ""
    location_evidence: str = ""
    
    # Gaps per component
    structure_gaps: List[str] = field(default_factory=list)
    judgment_gaps: List[str] = field(default_factory=list)
    context_gaps: List[str] = field(default_factory=list)
    regime_gaps: List[str] = field(default_factory=list)
    location_gaps: List[str] = field(default_factory=list)
    
    @property
    def overall_confidence(self) -> float:
        """Weighted average — weakest component pulls down more"""
        components = [
            (self.structure_confidence, 0.25),
            (self.judgment_confidence, 0.25),
            (self.context_confidence, 0.20),
            (self.regime_confidence, 0.15),
            (self.location_confidence, 0.15),
        ]
        
        weighted_sum = sum(c * w for c, w in components)
        
        # Penalty for any component below 50%
        min_component = min(c for c, _ in components)
        if min_component < 0.5:
            penalty = (0.5 - min_component) * 0.2
            weighted_sum -= penalty
        
        return max(0, min(1, weighted_sum))
    
    @property
    def weakest_component(self) -> str:
        components = {
            "structure": self.structure_confidence,
            "judgment": self.judgment_confidence,
            "context": self.context_confidence,
            "regime": self.regime_confidence,
            "location": self.location_confidence,
        }
        return min(components, key=components.get)
    
    @property
    def strongest_component(self) -> str:
        components = {
            "structure": self.structure_confidence,
            "judgment": self.judgment_confidence,
            "context": self.context_confidence,
            "regime": self.regime_confidence,
            "location": self.location_confidence,
        }
        return max(components, key=components.get)
    
    def get_improvement_recommendations(self) -> List[str]:
        """What should user provide to improve understanding"""
        recommendations = []
        
        if self.structure_confidence < 0.7:
            recommendations.append(
                "Show more examples with different candle structures — "
                "especially edge cases between 'clean' and 'messy'"
            )
            for gap in self.structure_gaps:
                recommendations.append(f"  → {gap}")
        
        if self.judgment_confidence < 0.7:
            recommendations.append(
                "Your subjective judgment needs more examples — "
                "show me setups where you ALMOST took it but didn't"
            )
            for gap in self.judgment_gaps:
                recommendations.append(f"  → {gap}")
        
        if self.context_confidence < 0.7:
            recommendations.append(
                "Show examples in different market conditions — "
                "trending, sideways, volatile"
            )
            for gap in self.context_gaps:
                recommendations.append(f"  → {gap}")
        
        if self.regime_confidence < 0.7:
            recommendations.append(
                "I need to see when you DON'T trade — "
                "show me market conditions where you sit out"
            )
            for gap in self.regime_gaps:
                recommendations.append(f"  → {gap}")
        
        if self.location_confidence < 0.7:
            recommendations.append(
                "Show more examples at different price locations — "
                "support, resistance, mid-range, extremes"
            )
            for gap in self.location_gaps:
                recommendations.append(f"  → {gap}")
        
        return recommendations
    
    def to_display(self) -> str:
        lines = []
        lines.append("┌──────────────────────────────────────────┐")
        lines.append("│   CONFIDENCE DECOMPOSITION               │")
        lines.append("├──────────────────────────────────────────┤")
        lines.append(f"│ Structure understanding:  {self.structure_confidence:>5.0%}         │")
        lines.append(f"│ Judgment understanding:   {self.judgment_confidence:>5.0%}         │")
        lines.append(f"│ Context understanding:    {self.context_confidence:>5.0%}         │")
        lines.append(f"│ Regime understanding:     {self.regime_confidence:>5.0%}         │")
        lines.append(f"│ Location understanding:   {self.location_confidence:>5.0%}         │")
        lines.append("├──────────────────────────────────────────┤")
        lines.append(f"│ OVERALL:                  {self.overall_confidence:>5.0%}         │")
        lines.append(f"│ WEAKEST: {self.weakest_component:<15}                │")
        lines.append("└──────────────────────────────────────────┘")
        return "\n".join(lines)


class ConfidenceComputer:
    """
    Computes confidence decomposition from onboarding results.
    """
    
    def compute(
        self,
        pattern_discovery: 'PatternDiscovery',
        regime_definition: 'RegimeDefinition',
        quiz_results: Optional[List[Dict]] = None,
        n_examples: int = 0
    ) -> ConfidenceDecomposition:
        
        decomp = ConfidenceDecomposition()
        
        # ── Structure Confidence ──
        # Based on: feature discriminators for structural features
        structure_features = [
            "wick_ratio_mean", "body_consistency", "body_size_variance",
            "range_contraction_r2", "squeeze_score", "compression_candle_count"
        ]
        structure_scores = []
        for f in structure_features:
            disc = pattern_discovery.feature_discriminators.get(f)
            if disc and disc.strength != FeatureDiscriminatorStrength.NONE:
                structure_scores.append(disc.auc)
        
        if structure_scores:
            decomp.structure_confidence = np.mean(structure_scores)
            decomp.structure_evidence = (
                f"Based on {len(structure_scores)} discriminating structural features"
            )
        else:
            decomp.structure_confidence = 0.3
            decomp.structure_gaps.append(
                "No clear structural patterns found between GOOD and BAD"
            )
        
        # ── Judgment Confidence ──
        # Based on: overall model accuracy on subjective features
        judgment_features = [
            "direction_entropy", "body_to_range_ratio",
            "volume_consistency", "range_volatility"
        ]
        judgment_scores = []
        for f in judgment_features:
            disc = pattern_discovery.feature_discriminators.get(f)
            if disc and disc.strength != FeatureDiscriminatorStrength.NONE:
                judgment_scores.append(disc.auc)
        
        if judgment_scores:
            decomp.judgment_confidence = np.mean(judgment_scores)
        else:
            decomp.judgment_confidence = 0.3
            decomp.judgment_gaps.append(
                "Subjective quality assessment not yet captured"
            )
        
        # ── Context Confidence ──
        # Based on: number of significant context dependencies found
        if pattern_discovery.context_dependencies:
            sig_contexts = [
                c for c in pattern_discovery.context_dependencies
                if c.is_significant
            ]
            decomp.context_confidence = min(0.9, 0.4 + len(sig_contexts) * 0.1)
            decomp.context_evidence = (
                f"Found {len(sig_contexts)} significant context dependencies"
            )
        else:
            decomp.context_confidence = 0.3
            decomp.context_gaps.append(
                "No context dependencies analyzed yet"
            )
        
        # ── Regime Confidence ──
        if regime_definition:
            decomp.regime_confidence = regime_definition.confidence
            if regime_definition.source == "hybrid":
                decomp.regime_evidence = "Learned from user description + examples"
            elif regime_definition.source == "learned_from_examples":
                decomp.regime_evidence = "Learned from examples only"
            else:
                decomp.regime_evidence = "From user description only"
                if regime_definition.confidence < 0.7:
                    decomp.regime_gaps.append(
                        "Regime definition based on description only — "
                        "needs validation from more examples"
                    )
        else:
            decomp.regime_confidence = 0.2
            decomp.regime_gaps.append("No regime definition available")
        
        # ── Location Confidence ──
        location_features = [
            "distance_to_nearest_support", "distance_to_nearest_resistance",
            "price_percentile_in_day_range", "atr_normalized_location"
        ]
        location_scores = []
        for f in location_features:
            disc = pattern_discovery.feature_discriminators.get(f)
            if disc and disc.strength != FeatureDiscriminatorStrength.NONE:
                location_scores.append(disc.auc)
        
        if location_scores:
            decomp.location_confidence = np.mean(location_scores)
        else:
            decomp.location_confidence = 0.3
            decomp.location_gaps.append(
                "Location preferences not yet clear from examples"
            )
        
        # ── Boost from quiz results ──
        if quiz_results:
            quiz_correct = sum(1 for q in quiz_results if q.get("correct"))
            quiz_total = len(quiz_results)
            if quiz_total > 0:
                quiz_accuracy = quiz_correct / quiz_total
                # Boost all components by quiz validation
                boost = (quiz_accuracy - 0.5) * 0.2  # +10% boost for 100% quiz
                decomp.structure_confidence = min(1, decomp.structure_confidence + boost)
                decomp.judgment_confidence = min(1, decomp.judgment_confidence + boost)
                decomp.context_confidence = min(1, decomp.context_confidence + boost)
        
        return decomp
