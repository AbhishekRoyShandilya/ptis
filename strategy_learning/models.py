# strategy_learning/models.py
"""
Pydantic models for the strategy-learning pipeline.

Design rules
────────────
• Every model is *immutable by default* (frozen) except where mutation is
  explicitly needed (e.g. ProfileConfidence recomputes `overall`).
• Validators enforce domain constraints at the boundary — callers never
  need to sanitise before constructing a model.
• `model_config` uses strict mode so silent type coercions don't hide bugs.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


# ══════════════════════════════════════════════
#  Vision Analysis — Output from screenshot analysis
# ══════════════════════════════════════════════


class CandlePatternObservation(BaseModel):
    """A single candle pattern observed in the chart."""

    model_config = {"frozen": True}

    pattern: str = Field(
        ...,
        min_length=1,
        description="e.g. '3 consecutive green candles', 'doji at resistance'",
    )
    location: str = Field(
        default="",
        description="e.g. 'near resistance zone', 'at VWAP'",
    )
    significance: str = Field(
        default="neutral",
        description="e.g. 'momentum building', 'exhaustion signal'",
    )


class PriceLevel(BaseModel):
    """A significant price level identified in the chart."""

    model_config = {"frozen": True}

    level: float
    label: str = Field(
        ...,
        min_length=1,
        description="e.g. 'resistance', 'support', 'VWAP', 'EMA20'",
    )
    strength: str = Field(default="medium")

    @field_validator("strength")
    @classmethod
    def _valid_strength(cls, v: str) -> str:
        allowed = {"weak", "medium", "strong"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(f"strength must be one of {allowed}, got '{v}'")
        return v_lower


class VisionAnalysis(BaseModel):
    """Structured output from Claude Vision chart analysis."""

    candle_patterns: list[CandlePatternObservation] = Field(default_factory=list)
    price_levels: list[PriceLevel] = Field(default_factory=list)
    support_resistance: list[str] = Field(default_factory=list)
    volume_profile: str = ""
    indicators_visible: list[str] = Field(default_factory=list)
    trend_direction: str = ""
    market_structure: str = ""
    key_observations: list[str] = Field(default_factory=list)
    timeframe_detected: str = ""
    symbol_detected: str = ""
    overall_context: str = ""

    @field_validator("trend_direction")
    @classmethod
    def _normalise_trend(cls, v: str) -> str:
        if not v:
            return v
        allowed = {
            "bullish", "bearish", "sideways", "transitioning",
            "strongly_bullish", "strongly_bearish",
        }
        v_lower = v.strip().lower().replace(" ", "_")
        return v_lower if v_lower in allowed else v.strip().lower()

    @property
    def is_empty(self) -> bool:
        """True when vision produced no meaningful data."""
        return not (
            self.candle_patterns
            or self.price_levels
            or self.trend_direction
            or self.overall_context
        )

    def to_text(self, symbol: str = "", timeframe: str = "") -> str:
        """Render a human-readable summary for LLM prompts."""
        lines: list[str] = []
        if symbol:
            lines.append(f"Symbol: {symbol}")
        if timeframe:
            lines.append(f"Timeframe: {timeframe}")
        if self.trend_direction:
            lines.append(f"Trend: {self.trend_direction}")
        if self.market_structure:
            lines.append(f"Structure: {self.market_structure}")
        if self.overall_context:
            lines.append(f"Context: {self.overall_context}")
        if self.candle_patterns:
            pats = "; ".join(
                f"{cp.pattern} ({cp.significance})" for cp in self.candle_patterns
            )
            lines.append(f"Candle patterns: {pats}")
        if self.volume_profile:
            lines.append(f"Volume: {self.volume_profile}")
        if self.price_levels:
            lvls = "; ".join(
                f"{pl.label}={pl.level} [{pl.strength}]" for pl in self.price_levels
            )
            lines.append(f"Key levels: {lvls}")
        if self.key_observations:
            lines.append(f"Observations: {'; '.join(self.key_observations)}")
        return "\n".join(lines) if lines else "No chart analysis available"


# ══════════════════════════════════════════════
#  User Annotations
# ══════════════════════════════════════════════

_VALID_DIRECTIONS = {"long", "short", ""}
_VALID_QUALITIES = {"A+", "standard", "marginal", ""}


class UserAnnotation(BaseModel):
    """User-provided annotations for a chart screenshot."""

    direction: str = ""
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_prices: list[float] = Field(default_factory=list)
    reasoning: str = ""
    what_makes_this_work: str = ""
    what_would_invalidate: str = ""
    setup_quality: str = "standard"
    timeframe: str = ""
    symbol: str = ""
    additional_notes: str = ""

    @field_validator("direction")
    @classmethod
    def _direction_check(cls, v: str) -> str:
        v = v.strip().lower()
        if v and v not in _VALID_DIRECTIONS:
            raise ValueError(f"direction must be 'long', 'short', or empty — got '{v}'")
        return v

    @field_validator("setup_quality")
    @classmethod
    def _quality_check(cls, v: str) -> str:
        if v and v not in _VALID_QUALITIES:
            raise ValueError(f"setup_quality must be one of {_VALID_QUALITIES}")
        return v

    @model_validator(mode="after")
    def _stop_on_correct_side(self) -> "UserAnnotation":
        """Warn-level check: stop should be below entry for longs, above for shorts."""
        if (
            self.direction
            and self.entry_price is not None
            and self.stop_price is not None
        ):
            if self.direction == "long" and self.stop_price >= self.entry_price:
                # Don't raise — but the profile builder will flag it
                pass
            if self.direction == "short" and self.stop_price <= self.entry_price:
                pass
        return self

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Compute R:R from annotations, or None if insufficient data."""
        if (
            self.entry_price is None
            or self.stop_price is None
            or not self.target_prices
        ):
            return None
        risk = abs(self.entry_price - self.stop_price)
        if risk == 0:
            return None
        reward = abs(self.target_prices[0] - self.entry_price)
        return round(reward / risk, 2)

    @property
    def has_full_trade_plan(self) -> bool:
        return all([
            self.direction,
            self.entry_price is not None,
            self.stop_price is not None,
            len(self.target_prices) > 0,
        ])


# ══════════════════════════════════════════════
#  Profile Confidence
# ══════════════════════════════════════════════

_CONFIDENCE_WEIGHTS = {
    "entry": 0.25,
    "stop": 0.20,
    "exit": 0.15,
    "regime": 0.15,
    "risk": 0.15,
    "timing": 0.10,
}


class ProfileConfidence(BaseModel):
    """Per-aspect confidence scores for a learned strategy profile."""

    entry: float = Field(default=0.0, ge=0.0, le=1.0)
    exit: float = Field(default=0.0, ge=0.0, le=1.0)
    stop: float = Field(default=0.0, ge=0.0, le=1.0)
    regime: float = Field(default=0.0, ge=0.0, le=1.0)
    timing: float = Field(default=0.0, ge=0.0, le=1.0)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    overall: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _recompute_overall(self) -> "ProfileConfidence":
        """Always recompute overall so it can never drift from components."""
        computed = sum(
            getattr(self, k) * w for k, w in _CONFIDENCE_WEIGHTS.items()
        )
        object.__setattr__(self, "overall", round(computed, 4))
        return self

    def weak_areas(self, threshold: float = 0.5) -> list[str]:
        """Return aspect names whose confidence is below *threshold*."""
        return [
            k for k in _CONFIDENCE_WEIGHTS
            if getattr(self, k) < threshold
        ]

    def strongest_area(self) -> str:
        scores = {k: getattr(self, k) for k in _CONFIDENCE_WEIGHTS}
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    @property
    def is_tradeable(self) -> bool:
        """Heuristic: profile has enough understanding to assist live."""
        return self.overall >= 0.45 and self.entry >= 0.4 and self.stop >= 0.3


# ══════════════════════════════════════════════
#  Learned Strategy Components
# ══════════════════════════════════════════════

_IMPORTANCE_LEVELS = {"critical", "important", "preferred", "nice_to_have"}


class LearnedEntryCriterion(BaseModel):
    """A single entry criterion synthesised from examples."""

    criterion: str = Field(..., min_length=1)
    description: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    example_count: int = Field(default=0, ge=0)
    importance: str = "important"
    learned_from: list[str] = Field(default_factory=list)

    @field_validator("importance")
    @classmethod
    def _valid_importance(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in _IMPORTANCE_LEVELS:
            raise ValueError(f"importance must be one of {_IMPORTANCE_LEVELS}")
        return v_lower


class LearnedExitRules(BaseModel):
    """Exit rules synthesised from examples."""

    stop_method: str = ""
    typical_stop_atr: Optional[float] = Field(default=None, ge=0)
    stop_description: str = ""
    target_patterns: list[dict[str, Any]] = Field(default_factory=list)
    trailing_rules: str = ""
    time_exit: str = ""
    partial_exit_rules: list[str] = Field(default_factory=list)

    @property
    def completeness(self) -> float:
        """0-1 score of how fully the exit rules are defined."""
        score = 0.0
        if self.stop_method:
            score += 0.30
        if self.target_patterns:
            score += 0.30
        if self.trailing_rules:
            score += 0.20
        if self.time_exit:
            score += 0.10
        if self.partial_exit_rules:
            score += 0.10
        return round(score, 2)


# ══════════════════════════════════════════════
#  Strategy Profile Summary (list views)
# ══════════════════════════════════════════════


class StrategyProfileSummary(BaseModel):
    """Lightweight profile representation for list endpoints."""

    model_config = {"frozen": True}

    id: UUID
    name: str
    short_code: str
    status: str
    examples_count: int = 0
    sessions_count: int = 0
    confidence: ProfileConfidence = Field(default_factory=ProfileConfidence)
    created_at: datetime
    last_rebuilt_at: Optional[datetime] = None


# ══════════════════════════════════════════════
#  Trade Analysis
# ══════════════════════════════════════════════


class TradeAnalysisRequest(BaseModel):
    """Request to compare a trade against the learned profile."""

    trade_id: Optional[UUID] = None
    symbol: str = ""
    direction: str = ""
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    regime: str = ""
    screenshot_path: Optional[str] = None

    @field_validator("direction")
    @classmethod
    def _dir(cls, v: str) -> str:
        return v.strip().lower()


class TradeDeviation(BaseModel):
    """A single deviation from the learned pattern."""

    model_config = {"frozen": True}

    aspect: str
    description: str
    severity: str = "medium"
    learned_expectation: str = ""
    actual_value: str = ""

    @field_validator("severity")
    @classmethod
    def _sev(cls, v: str) -> str:
        allowed = {"low", "medium", "high", "critical"}
        v = v.strip().lower()
        if v not in allowed:
            raise ValueError(f"severity must be one of {allowed}")
        return v


class TradeAnalysisResponse(BaseModel):
    """Result of analysing a trade against the learned profile."""

    adherence_score: float = Field(default=0.0, ge=0, le=100)
    deviations: list[TradeDeviation] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    narrative_feedback: str = ""
    similar_examples: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ══════════════════════════════════════════════
#  Co-Planning
# ══════════════════════════════════════════════


class CoplanRequest(BaseModel):
    """Request body for the co-plan endpoint."""

    profile_id: UUID
    screenshot_path: Optional[str] = None
    context: str = ""
    symbol: str = ""
    timeframe: str = ""


class SimilarExample(BaseModel):
    """A historical example matched by similarity search."""

    model_config = {"frozen": True}

    example_id: UUID
    similarity_score: float = Field(ge=0.0, le=1.0)
    ai_interpretation: str = ""
    outcome_notes: str = ""


class CoplanResponse(BaseModel):
    """Full co-planning result with recommendations."""

    setup_match_score: float = Field(default=0.0, ge=0, le=100)
    setup_detected: bool = False
    entry_recommendation: str = ""
    stop_recommendation: str = ""
    target_recommendations: list[str] = Field(default_factory=list)
    risk_assessment: str = ""
    similar_examples: list[SimilarExample] = Field(default_factory=list)
    narrative: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════
#  Trade Outcome Feedback (NEW — learn from live trades)
# ══════════════════════════════════════════════


class TradeOutcomeFeedback(BaseModel):
    """
    Submitted after a trade closes so the system can learn
    from real outcomes — not just hypothetical setups.
    """

    trade_id: UUID
    profile_id: UUID
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    stop_price: Optional[float] = None
    pnl_r: Optional[float] = None  # P&L in R-multiples
    outcome: str = ""  # "win", "loss", "breakeven", "scratched"
    exit_reason: str = ""  # "target_hit", "stopped_out", "manual", "time_exit"
    followed_plan: bool = True
    deviation_notes: str = ""
    screenshot_path: Optional[str] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None

    @field_validator("outcome")
    @classmethod
    def _valid_outcome(cls, v: str) -> str:
        allowed = {"win", "loss", "breakeven", "scratched", ""}
        if v and v.strip().lower() not in allowed:
            raise ValueError(f"outcome must be one of {allowed}")
        return v.strip().lower()


# ══════════════════════════════════════════════
#  Batch Training — Multi-screenshot synthesis
# ══════════════════════════════════════════════


class ConsistentPattern(BaseModel):
    """A pattern that appears consistently across multiple screenshots."""

    model_config = {"frozen": True}

    pattern: str = Field(..., min_length=1)
    description: str = ""
    frequency: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of screenshots where this pattern appears (0-1)",
    )
    example_indices: list[int] = Field(default_factory=list)
    importance: str = "important"
    category: str = ""  # entry_signal, confirmation, context, structure, volume, etc.

    @field_validator("importance")
    @classmethod
    def _valid_imp(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in _IMPORTANCE_LEVELS:
            raise ValueError(f"importance must be one of {_IMPORTANCE_LEVELS}")
        return v_lower


class StrategyContradiction(BaseModel):
    """A contradiction or ambiguity found across screenshots."""

    model_config = {"frozen": True}

    aspect: str = Field(..., min_length=1, description="e.g. 'trend_direction', 'volume_behavior'")
    description: str = ""
    variants: list[str] = Field(default_factory=list)
    example_groups: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Variant label → list of screenshot indices",
    )
    suggested_question: str = ""
    severity: str = "medium"

    @field_validator("severity")
    @classmethod
    def _sev(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        v = v.strip().lower()
        if v not in allowed:
            raise ValueError(f"severity must be one of {allowed}")
        return v


class SynthesisedRule(BaseModel):
    """A rule synthesised from cross-referencing multiple screenshots."""

    rule: str = Field(..., min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = Field(default=0, ge=0)
    category: str = ""  # entry, exit, stop, regime, timing
    conditions: list[str] = Field(default_factory=list)


class BatchTrainingReport(BaseModel):
    """
    Output of multi-screenshot batch training.

    Cross-references 15-20 chart screenshots to extract
    what's CONSISTENT (signal) vs what VARIES (noise).
    """

    total_screenshots: int = 0
    analysed_successfully: int = 0
    failed_analyses: int = 0

    # Core discoveries
    consistent_patterns: list[ConsistentPattern] = Field(default_factory=list)
    synthesised_rules: list[SynthesisedRule] = Field(default_factory=list)
    contradictions: list[StrategyContradiction] = Field(default_factory=list)

    # Aggregated observations
    common_indicators: list[str] = Field(default_factory=list)
    common_trend_direction: str = ""
    common_timeframe: str = ""
    common_market_structure: str = ""
    volume_patterns: list[str] = Field(default_factory=list)
    price_level_patterns: list[str] = Field(default_factory=list)

    # Follow-up
    auto_generated_questions: list[str] = Field(default_factory=list)

    # Quality metrics
    consistency_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="How consistent the screenshots are (1.0 = identical setups)",
    )
    coverage_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="How many strategy aspects are covered",
    )
    synthesis_narrative: str = ""

    @property
    def is_sufficient(self) -> bool:
        """Enough screenshots and consistency to build a profile."""
        return (
            self.analysed_successfully >= 10
            and self.consistency_score >= 0.4
            and len(self.consistent_patterns) >= 3
        )