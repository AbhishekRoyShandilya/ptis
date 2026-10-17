from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger("SOE.LearningDomainSeparator")


class LearningDomain(Enum):
    """What system of PTIS should learn from this data"""
    INTENT = "intent"           # Onboarding learns this
    EDGE = "edge"               # AVT learns this
    BEHAVIOR = "behavior"       # Behavior detector learns this
    IMPROVEMENT = "improvement" # Improvement lab learns this


@dataclass
class LearningSignal:
    """A single piece of information routed to correct learning domain"""
    domain: LearningDomain
    signal_type: str            # "structure_preference", "outcome", etc.
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = ""            # "onboarding_example", "live_trade", etc.


class LearningDomainSeparator:
    """
    ENFORCES the critical separation:
    
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  ONBOARDING learns: USER INTENT                          │
    │  ├── What does user LOOK FOR?                           │
    │  ├── What does user CONSIDER valid?                     │
    │  ├── What does user SKIP and WHY?                       │
    │  └── What does user's JUDGMENT mean?                    │
    │                                                          │
    │  AVT discovers: STATISTICAL EDGE                         │
    │  ├── Which of user's setups actually WORK?              │
    │  ├── What HIDDEN features improve win rate?             │
    │  ├── When does strategy FAIL systematically?            │
    │  └── What improvements increase edge?                   │
    │                                                          │
    │  BEHAVIOR DETECTOR learns: USER PATTERNS                 │
    │  ├── When does user DEVIATE from strategy?              │
    │  ├── Emotional trading detection                        │
    │  ├── Revenge trading, FOMO, etc.                        │
    │  └── Risk management adherence                          │
    │                                                          │
    │  IMPROVEMENT LAB discovers: STRATEGY EVOLUTION           │
    │  ├── Which modifications improve edge?                  │
    │  ├── New conditions that help?                          │
    │  ├── Conditions that can be relaxed?                    │
    │  └── Regime-specific optimizations                      │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    
    WHY THIS MATTERS:
    
    ❌ Without separation:
       25 examples → "this pattern wins 67%"
       → NOISE → bad decisions → system fails
    
    ✅ With separation:
       25 examples → "user INTENDS to trade this pattern"
       500 AVT trades → "this pattern wins 63% when X"
       → SIGNAL → good decisions → system improves
    """

    # ═══════════════════════════════════════════════════════
    # ROUTING RULES: What goes where
    # ═══════════════════════════════════════════════════════

    ONBOARDING_SIGNALS = {
        # These are LEARNED during onboarding
        "structure_preference",     # What candle structures user likes
        "quality_judgment",         # What user considers "clean" vs "ugly"
        "location_preference",      # Where in price structure user trades
        "volume_interpretation",    # How user reads volume
        "pattern_recognition",      # What patterns user identifies
        "regime_preference",        # When user trades vs sits out
        "skip_reasons",            # Why user avoids certain setups
        "entry_criteria",          # What triggers entry
        "setup_classification",    # How user categorizes setups
    }

    AVT_SIGNALS = {
        # These are DISCOVERED by AVT from live/virtual trading
        "win_rate",                # Actual win rate
        "risk_reward_actual",      # Actual R-multiples
        "edge_by_regime",          # Edge varies by market condition
        "hidden_feature_impact",   # Features user doesn't know about
        "optimal_entry_timing",    # Best entry within a setup
        "optimal_exit_timing",     # Best exit strategy
        "position_sizing_edge",    # Size optimization
        "correlation_patterns",    # Multi-setup correlations
    }

    BEHAVIOR_SIGNALS = {
        # These detect user's psychological patterns
        "strategy_deviation",      # Took trade outside strategy
        "emotional_entry",         # Entered based on FOMO/revenge
        "early_exit",             # Exited too early (fear)
        "late_exit",              # Held too long (hope)
        "overtrading",            # Too many trades
        "sizing_deviation",       # Inconsistent position sizing
        "rule_violation",         # Broke own rules
    }

    def __init__(self):
        self.routing_log: List[Dict] = []

    def route_onboarding_example(
        self,
        example: 'TrainingExample'
    ) -> List[LearningSignal]:
        """
        Route a single onboarding example to correct learning domains.
        
        From one example, extracts MULTIPLE signals for DIFFERENT domains.
        """
        signals = []

        # ── ALWAYS goes to INTENT domain ──
        signals.append(LearningSignal(
            domain=LearningDomain.INTENT,
            signal_type="setup_classification",
            data={
                "features": example.features,
                "intent": example.label.value if hasattr(example.label, 'value') else str(example.label),
                "explanation_tags": example.explanation_tags,
                "context": example.context_features
            },
            confidence=example.label_confidence,
            source="onboarding_example"
        ))

        # ── Structure preference → INTENT ──
        if example.features:
            signals.append(LearningSignal(
                domain=LearningDomain.INTENT,
                signal_type="structure_preference",
                data={
                    "features": {
                        k: v for k, v in example.features.items()
                        if k in self._get_structure_feature_names()
                    },
                    "is_preferred": example.label.is_positive 
                                   if hasattr(example.label, 'is_positive') else True
                },
                confidence=example.label_confidence,
                source="onboarding_example"
            ))

        # ── Quality judgment → INTENT ──
        judgment_tags = [
            t for t in example.explanation_tags
            if t in ("clean", "smooth", "ugly", "choppy", "messy",
                     "tight", "wicky", "strong_bodies")
        ]
        if judgment_tags:
            signals.append(LearningSignal(
                domain=LearningDomain.INTENT,
                signal_type="quality_judgment",
                data={
                    "judgment_terms": judgment_tags,
                    "is_positive_judgment": any(
                        t in ("clean", "smooth", "tight", "strong_bodies")
                        for t in judgment_tags
                    ),
                    "mapped_features": {
                        k: v for k, v in example.features.items()
                        if k in self._get_judgment_feature_names()
                    }
                },
                confidence=example.label_confidence,
                source="onboarding_example"
            ))

        # ── Skip reasons → INTENT (very valuable) ──
        if hasattr(example.label, 'is_negative') and example.label.is_negative:
            signals.append(LearningSignal(
                domain=LearningDomain.INTENT,
                signal_type="skip_reasons",
                data={
                    "explanation": example.explanation,
                    "tags": example.explanation_tags,
                    "features_at_skip": example.features
                },
                confidence=example.label_confidence,
                source="onboarding_example"
            ))

        # ── Outcome data → STORED but routed to AVT domain ──
        # NOT used for strategy learning, only for future AVT reference
        if example.outcome_r is not None:
            signals.append(LearningSignal(
                domain=LearningDomain.EDGE,
                signal_type="outcome_reference",
                data={
                    "outcome_r": example.outcome_r,
                    "features": example.features,
                    "note": "FROM_ONBOARDING: Do NOT use for edge calculation. "
                           "Insufficient sample size. Store for future AVT analysis."
                },
                confidence=0.3,  # LOW confidence — this is reference only
                source="onboarding_example_outcome"
            ))

        # Log routing
        self.routing_log.append({
            "example_id": example.example_id,
            "signals_generated": len(signals),
            "domains_touched": list(set(s.domain.value for s in signals))
        })

        return signals

    def route_live_trade(
        self,
        trade_data: Dict[str, Any]
    ) -> List[LearningSignal]:
        """
        Route a LIVE trade to correct learning domains.
        Live trades feed EDGE and BEHAVIOR domains.
        """
        signals = []

        # ── Outcome → EDGE domain ──
        if "outcome_r" in trade_data:
            signals.append(LearningSignal(
                domain=LearningDomain.EDGE,
                signal_type="win_rate",
                data={
                    "outcome_r": trade_data["outcome_r"],
                    "features": trade_data.get("features", {}),
                    "regime": trade_data.get("regime", {}),
                    "context": trade_data.get("context", {})
                },
                confidence=1.0,  # Live data = high confidence
                source="live_trade"
            ))

        # ── Strategy adherence → BEHAVIOR domain ──
        if "strategy_match_score" in trade_data:
            match_score = trade_data["strategy_match_score"]
            if match_score < 0.5:
                signals.append(LearningSignal(
                    domain=LearningDomain.BEHAVIOR,
                    signal_type="strategy_deviation",
                    data={
                        "match_score": match_score,
                        "expected_features": trade_data.get("expected", {}),
                        "actual_features": trade_data.get("features", {}),
                        "deviation_details": trade_data.get("deviations", [])
                    },
                    confidence=0.8,
                    source="live_trade"
                ))

        return signals

    def validate_learning_request(
        self,
        domain: LearningDomain,
        signal_type: str,
        source: str
    ) -> Dict[str, Any]:
        """
        GUARD: Prevents wrong domain from learning wrong signals.
        
        Example: Prevents onboarding from learning outcomes.
        """
        allowed = True
        reason = ""

        if domain == LearningDomain.INTENT:
            if signal_type in self.AVT_SIGNALS:
                allowed = False
                reason = (
                    f"Signal '{signal_type}' belongs to AVT domain, "
                    f"not INTENT. Onboarding should NOT learn from outcomes."
                )
            if signal_type in self.BEHAVIOR_SIGNALS:
                allowed = False
                reason = (
                    f"Signal '{signal_type}' belongs to BEHAVIOR domain. "
                    f"Onboarding should NOT learn behavioral patterns."
                )

        elif domain == LearningDomain.EDGE:
            if source == "onboarding_example":
                if signal_type != "outcome_reference":
                    allowed = False
                    reason = (
                        f"Onboarding examples should NOT feed edge learning. "
                        f"Only 'outcome_reference' allowed for storage."
                    )

        return {
            "allowed": allowed,
            "reason": reason,
            "domain": domain.value,
            "signal_type": signal_type
        }

    def get_routing_summary(self) -> Dict[str, Any]:
        """Summary of all routing decisions"""
        domain_counts = {}
        for log in self.routing_log:
            for domain in log["domains_touched"]:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        return {
            "total_examples_routed": len(self.routing_log),
            "domain_distribution": domain_counts,
            "routing_log": self.routing_log[-10:]  # Last 10
        }

    @staticmethod
    def _get_structure_feature_names() -> set:
        return {
            "wick_ratio_mean", "wick_ratio_max", "body_consistency",
            "body_size_variance", "avg_body_size", "body_to_range_ratio",
            "range_contraction_r2", "squeeze_score", "compression_candle_count",
            "range_width_normalized", "range_volatility"
        }

    @staticmethod
    def _get_judgment_feature_names() -> set:
        return {
            "direction_entropy", "body_direction_ratio",
            "volume_consistency", "consecutive_same_direction_max",
            "wick_ratio_max", "body_size_variance",
            "range_volatility", "volume_trend_slope"
        }
