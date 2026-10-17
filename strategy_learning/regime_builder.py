"""
============================================================
REGIME DEFINITION BUILDER
Converts vague regime descriptions into measurable conditions
============================================================
File: soe/regime_builder.py
============================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np
import pandas as pd


class RegimeType(Enum):
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    TRANSITIONING = "transitioning"


@dataclass
class RegimeCondition:
    """Single measurable condition that defines a regime"""
    indicator: str          # "ema_50_slope", "adx", "price_vs_ema20"
    operator: str           # "above", "below", "between", "crossing"
    value: Any              # threshold or [low, high] for between
    weight: float = 1.0     # how important this condition is
    learned: bool = False   # was this learned from examples or stated by user


@dataclass
class RegimeDefinition:
    """Complete regime definition with measurable conditions"""
    regime_name: str
    regime_type: RegimeType
    conditions: List[RegimeCondition] = field(default_factory=list)
    confidence: float = 0.0         # how well defined
    source: str = "user_stated"     # "user_stated" | "learned_from_examples" | "hybrid"
    
    # Learned from examples
    favorable_regimes: List[str] = field(default_factory=list)
    unfavorable_regimes: List[str] = field(default_factory=list)
    
    def evaluate(self, market_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate current market data against regime definition.
        Returns regime match score and details.
        """
        if not self.conditions:
            return {
                "regime_match": 0.5,  # uncertain
                "conditions_met": 0,
                "conditions_total": 0,
                "regime_type": "unknown"
            }
        
        met = 0
        total_weight = 0
        weighted_met = 0
        details = []
        
        for condition in self.conditions:
            total_weight += condition.weight
            indicator_value = market_data.get(condition.indicator)
            
            if indicator_value is None:
                details.append({
                    "condition": condition.indicator,
                    "status": "missing_data",
                    "met": False
                })
                continue
            
            is_met = self._check_condition(indicator_value, condition)
            if is_met:
                met += 1
                weighted_met += condition.weight
            
            details.append({
                "condition": condition.indicator,
                "operator": condition.operator,
                "threshold": condition.value,
                "actual": indicator_value,
                "met": is_met
            })
        
        regime_match = weighted_met / total_weight if total_weight > 0 else 0.5
        
        return {
            "regime_match": regime_match,
            "conditions_met": met,
            "conditions_total": len(self.conditions),
            "details": details,
            "regime_type": self.regime_type.value
        }
    
    def _check_condition(
        self, value: float, condition: RegimeCondition
    ) -> bool:
        if condition.operator == "above":
            return value > condition.value
        elif condition.operator == "below":
            return value < condition.value
        elif condition.operator == "between":
            return condition.value[0] <= value <= condition.value[1]
        elif condition.operator == "crossing_above":
            return value > condition.value  # simplified
        elif condition.operator == "crossing_below":
            return value < condition.value
        return False


class RegimeDefinitionBuilder:
    """
    Builds formal regime definitions from:
    1. User's natural language descriptions
    2. Learned patterns from onboarding examples
    3. Statistical analysis of context
    
    Converts "works in bearish market" into:
    {
        conditions: [
            {indicator: "ema_50_slope", operator: "below", value: 0},
            {indicator: "price_vs_ema50", operator: "below", value: 0},
            {indicator: "adx", operator: "above", value: 20}
        ]
    }
    """
    
    # Mapping of natural language regime terms to measurable conditions
    REGIME_TERM_MAP = {
        "bearish": {
            "type": RegimeType.WEAK_DOWNTREND,
            "conditions": [
                RegimeCondition("ema_20_slope", "below", 0, weight=0.8),
                RegimeCondition("price_vs_ema50", "below", 0, weight=1.0),
                RegimeCondition("trend_direction_score", "below", -0.3, weight=0.9),
            ]
        },
        "strong_bearish": {
            "type": RegimeType.STRONG_DOWNTREND,
            "conditions": [
                RegimeCondition("ema_20_slope", "below", -0.5, weight=0.8),
                RegimeCondition("price_vs_ema50", "below", -1.0, weight=1.0),
                RegimeCondition("adx", "above", 25, weight=0.7),
                RegimeCondition("lower_highs_count", "above", 3, weight=0.9),
            ]
        },
        "bullish": {
            "type": RegimeType.WEAK_UPTREND,
            "conditions": [
                RegimeCondition("ema_20_slope", "above", 0, weight=0.8),
                RegimeCondition("price_vs_ema50", "above", 0, weight=1.0),
                RegimeCondition("trend_direction_score", "above", 0.3, weight=0.9),
            ]
        },
        "strong_bullish": {
            "type": RegimeType.STRONG_UPTREND,
            "conditions": [
                RegimeCondition("ema_20_slope", "above", 0.5, weight=0.8),
                RegimeCondition("price_vs_ema50", "above", 1.0, weight=1.0),
                RegimeCondition("adx", "above", 25, weight=0.7),
                RegimeCondition("higher_lows_count", "above", 3, weight=0.9),
            ]
        },
        "sideways": {
            "type": RegimeType.SIDEWAYS,
            "conditions": [
                RegimeCondition("adx", "below", 20, weight=1.0),
                RegimeCondition("ema_20_slope", "between", [-0.2, 0.2], weight=0.8),
                RegimeCondition("direction_entropy", "above", 0.7, weight=0.6),
            ]
        },
        "volatile": {
            "type": RegimeType.VOLATILE,
            "conditions": [
                RegimeCondition("atr_percentile", "above", 75, weight=1.0),
                RegimeCondition("range_volatility", "above", 0.5, weight=0.8),
            ]
        },
        "low_volatility": {
            "type": RegimeType.LOW_VOLATILITY,
            "conditions": [
                RegimeCondition("atr_percentile", "below", 25, weight=1.0),
                RegimeCondition("bb_width_percentile", "below", 20, weight=0.8),
            ]
        },
        "trending": {
            "type": RegimeType.STRONG_UPTREND,  # direction determined separately
            "conditions": [
                RegimeCondition("adx", "above", 25, weight=1.0),
                RegimeCondition("trend_strength", "above", 0.5, weight=0.9),
            ]
        }
    }
    
    def __init__(self):
        self.regime_definitions: Dict[str, RegimeDefinition] = {}
    
    def build_from_description(
        self, 
        strategy_id: str,
        user_description: str,
        regime_mentions: List[str]
    ) -> RegimeDefinition:
        """
        Build regime definition from user's natural language.
        
        Args:
            strategy_id: Strategy identifier
            user_description: Full strategy description
            regime_mentions: Extracted regime-related terms
                           e.g., ["bearish", "downtrend"]
        """
        all_conditions = []
        regime_type = RegimeType.SIDEWAYS  # default
        
        for mention in regime_mentions:
            mention_lower = mention.lower().strip()
            
            if mention_lower in self.REGIME_TERM_MAP:
                mapped = self.REGIME_TERM_MAP[mention_lower]
                regime_type = mapped["type"]
                all_conditions.extend(mapped["conditions"])
        
        # Deduplicate conditions (keep highest weight)
        deduped = {}
        for cond in all_conditions:
            key = cond.indicator
            if key not in deduped or cond.weight > deduped[key].weight:
                deduped[key] = cond
        
        definition = RegimeDefinition(
            regime_name=f"{strategy_id}_regime",
            regime_type=regime_type,
            conditions=list(deduped.values()),
            confidence=0.5,  # Low — from description only
            source="user_stated"
        )
        
        self.regime_definitions[strategy_id] = definition
        return definition
    
    def refine_from_examples(
        self,
        strategy_id: str,
        examples_with_regime_features: List[Dict[str, Any]]
    ) -> RegimeDefinition:
        """
        Refine regime definition using actual example data.
        
        Each example has:
        {
            "intent": "would_take" | "would_skip",
            "regime_features": {
                "ema_20_slope": -0.3,
                "price_vs_ema50": -1.2,
                "adx": 28,
                ...
            }
        }
        """
        existing = self.regime_definitions.get(strategy_id)
        
        # Separate TAKE vs SKIP examples
        take_examples = [
            e["regime_features"] for e in examples_with_regime_features
            if e["intent"] == "would_take"
        ]
        skip_examples = [
            e["regime_features"] for e in examples_with_regime_features
            if e["intent"] in ("would_skip", "would_avoid")
        ]
        
        if not take_examples:
            return existing or RegimeDefinition(
                regime_name=f"{strategy_id}_regime",
                regime_type=RegimeType.SIDEWAYS,
                confidence=0.0,
                source="insufficient_data"
            )
        
        # Analyze which regime features differ between TAKE and SKIP
        learned_conditions = []
        regime_features = [
            "ema_20_slope", "price_vs_ema50", "price_vs_ema20",
            "adx", "trend_direction_score", "trend_strength",
            "atr_percentile", "higher_lows_count", "lower_highs_count"
        ]
        
        for feature in regime_features:
            take_vals = [
                e.get(feature) for e in take_examples
                if e.get(feature) is not None
            ]
            skip_vals = [
                e.get(feature) for e in skip_examples
                if e.get(feature) is not None
            ]
            
            if len(take_vals) >= 3 and len(skip_vals) >= 3:
                take_mean = np.mean(take_vals)
                skip_mean = np.mean(skip_vals)
                take_std = np.std(take_vals) or 0.01
                
                # Cohen's d effect size
                pooled_std = np.sqrt(
                    (np.std(take_vals)**2 + np.std(skip_vals)**2) / 2
                ) or 0.01
                effect_size = abs(take_mean - skip_mean) / pooled_std
                
                if effect_size > 0.5:  # Medium+ effect
                    # Determine direction
                    if take_mean > skip_mean:
                        operator = "above"
                        # Threshold at midpoint minus some margin
                        threshold = take_mean - take_std * 0.5
                    else:
                        operator = "below"
                        threshold = take_mean + take_std * 0.5
                    
                    learned_conditions.append(RegimeCondition(
                        indicator=feature,
                        operator=operator,
                        value=round(threshold, 3),
                        weight=min(1.0, effect_size / 2),
                        learned=True
                    ))
        
        # Merge with existing definition
        if existing:
            # Keep user-stated conditions, add learned ones
            merged_conditions = {c.indicator: c for c in existing.conditions}
            for learned in learned_conditions:
                if learned.indicator in merged_conditions:
                    # Average the thresholds, increase weight
                    existing_cond = merged_conditions[learned.indicator]
                    merged_cond = RegimeCondition(
                        indicator=learned.indicator,
                        operator=learned.operator,
                        value=(existing_cond.value + learned.value) / 2
                              if isinstance(existing_cond.value, (int, float))
                              else learned.value,
                        weight=min(1.0, (existing_cond.weight + learned.weight) / 2 + 0.1),
                        learned=True
                    )
                    merged_conditions[learned.indicator] = merged_cond
                else:
                    merged_conditions[learned.indicator] = learned
            
            updated = RegimeDefinition(
                regime_name=existing.regime_name,
                regime_type=existing.regime_type,
                conditions=list(merged_conditions.values()),
                confidence=min(0.9, existing.confidence + 0.2),
                source="hybrid"
            )
        else:
            # Determine regime type from learned conditions
            regime_type = self._infer_regime_type(learned_conditions, take_examples)
            updated = RegimeDefinition(
                regime_name=f"{strategy_id}_regime",
                regime_type=regime_type,
                conditions=learned_conditions,
                confidence=0.6,
                source="learned_from_examples"
            )
        
        self.regime_definitions[strategy_id] = updated
        return updated
    
    def _infer_regime_type(
        self,
        conditions: List[RegimeCondition],
        take_examples: List[Dict]
    ) -> RegimeType:
        """Infer overall regime type from conditions and examples"""
        # Average trend direction across take examples
        trend_scores = [
            e.get("trend_direction_score", 0)
            for e in take_examples
            if e.get("trend_direction_score") is not None
        ]
        
        if not trend_scores:
            return RegimeType.SIDEWAYS
        
        avg_trend = np.mean(trend_scores)
        
        if avg_trend > 0.5:
            return RegimeType.STRONG_UPTREND
        elif avg_trend > 0.2:
            return RegimeType.WEAK_UPTREND
        elif avg_trend < -0.5:
            return RegimeType.STRONG_DOWNTREND
        elif avg_trend < -0.2:
            return RegimeType.WEAK_DOWNTREND
        else:
            return RegimeType.SIDEWAYS
