# ske/models.py
"""
Pydantic models for the Strategy Knowledge Engine.
Used across API boundaries, internal pipelines, and serialization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from core.enums import ConvictionBand, ScoringFunctionType, SKOStatus


# ── Scoring Configuration ──

class ScoringParameter(BaseModel):
    """Configuration for a single fuzzy scoring function on one feature."""
    type: ScoringFunctionType
    weight: float = 1.0
    invert: bool = False  # True → score = 1 - score

    # Gaussian
    center: Optional[float] = None
    spread: Optional[float] = None

    # Trapezoid
    lower: Optional[float] = None
    upper: Optional[float] = None
    trap_spread: Optional[float] = None  # falloff width outside [lower, upper]

    # Sigmoid
    slope: Optional[float] = None

    # Asymmetric
    left_spread: Optional[float] = None
    right_spread: Optional[float] = None


class FeatureScore(BaseModel):
    """Result of scoring a single feature."""
    feature_name: str
    raw_value: float
    normalized_score: float = Field(ge=0.0, le=1.0)
    weight: float = 1.0


class ConvictionResult(BaseModel):
    """Aggregated conviction from all feature scores."""
    weighted_score: float = Field(ge=0.0, le=1.0)
    conviction_band: ConvictionBand
    individual_scores: List[FeatureScore] = []
    sizing_recommendation: str = "standard"
    size_multiplier: float = 1.0


# ── Regime Fingerprint ──

class RegimeFingerprint(BaseModel):
    """Multi-dimensional market regime snapshot (each dimension 0–1)."""
    index_trend: float = Field(0.5, ge=0.0, le=1.0)       # 0=strong bear, 1=strong bull
    volatility_level: float = Field(0.5, ge=0.0, le=1.0)   # ATR percentile
    market_breadth: float = Field(0.5, ge=0.0, le=1.0)     # advances / total
    sector_participation: float = Field(0.5, ge=0.0, le=1.0)
    fii_dii_flow: float = Field(0.5, ge=0.0, le=1.0)       # 0=strong sell, 1=strong buy

    def to_vector(self) -> List[float]:
        return [
            self.index_trend,
            self.volatility_level,
            self.market_breadth,
            self.sector_participation,
            self.fii_dii_flow,
        ]


# ── Variants ──

class DistinguishingFeature(BaseModel):
    feature: str
    mean: float
    std: float
    importance: float = 1.0


class VariantInfo(BaseModel):
    """
    Cluster information for an SKO variant.

    Variants can be:
    - **System-discovered**: auto-clustered by the VariantEngine from trade data
    - **User-taught**: explicitly taught by the trader through teaching sessions,
      with narrative, regime affinity, and custom scoring overrides

    The `provenance` field distinguishes origins. User-taught variants carry
    richer metadata that feeds into the coaching and competition layers.
    """

    label: str
    centroid: Dict[str, float] = {}
    trade_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    expectancy: float = 0.0
    distinguishing_features: List[DistinguishingFeature] = []
    is_anti_variant: bool = False

    # ── User-taught variant enrichment ──

    narrative: str = ""
    """
    Natural-language explanation of what makes this variant work.
    E.g., "Morning breakout on high-volume gap-ups where NIFTY is trending.
    Works best when pre-market volume is 2x average and the stock holds VWAP."
    Used for LLM context, coaching, and embedding-based retrieval.
    """

    regime_affinity: Optional[RegimeFingerprint] = None
    """
    Market regime where this variant thrives. Compared against the live
    regime fingerprint to boost/dampen conviction scores.
    """

    scoring_overrides: Dict[str, Any] = {}
    """
    Per-variant scoring parameter overrides. When a trade matches this
    variant, these overrides are merged on top of the SKO's base scoring_config.
    Allows the system to tighten or loosen feature thresholds per variant.
    Example: {"rsi": {"center": 35, "spread": 8}, "volume_ratio": {"weight": 2.0}}
    """

    provenance: str = "system"
    """
    Origin of this variant:
    - "system"      — auto-discovered by VariantEngine clustering
    - "user_taught" — explicitly taught by the trader in a teaching session
    - "hybrid"      — system-discovered, then refined by user
    """

    taught_examples_count: int = 0
    """Number of teaching examples explicitly linked to this variant."""

    performance_history: List[Dict[str, Any]] = []
    """
    Rolling performance snapshots for trend analysis.
    Each entry: {"period": "2024-W03", "win_rate": 0.72, "avg_r": 1.3,
                 "trades": 11, "expectancy": 0.65}
    """

    tags: List[str] = []
    """User-defined labels. E.g., ["morning", "gap-up", "high-volume"]"""

    preferred_timeframes: List[str] = []
    """Timeframes where this variant is most relevant. E.g., ["5m", "15m"]"""

    confidence_level: float = 0.0
    """
    System's confidence in this variant (0.0–1.0).
    Based on trade_count, data quality, and consistency of performance.
    Higher values → more weight in conviction scoring.
    """



# ── DQAL ──

class DQALGateResult(BaseModel):
    gate_name: str
    score: float = Field(ge=0.0, le=100.0)
    passed: bool = True
    issues: List[str] = []


class DQALResult(BaseModel):
    """Data Quality Assurance Layer validation output."""
    gate_scores: List[DQALGateResult] = []
    overall_score: float = Field(ge=0.0, le=100.0)
    grade: str = "F"  # A, B, C, D, F
    issues: List[str] = []
    recommendations: List[str] = []


# ── Signal Output ──

class EntryZone(BaseModel):
    lower: float
    upper: float
    sweet_spot: float


class SignalOutput(BaseModel):
    """Output from the signal generator for one symbol against one SKO."""
    sko_id: UUID
    sko_name: str = ""
    symbol: str
    timestamp: datetime

    conviction_score: float = Field(ge=0.0, le=1.0)
    conviction_band: ConvictionBand
    individual_scores: List[FeatureScore] = []

    sizing_recommendation: str = "standard"
    size_multiplier: float = 1.0

    entry_zone: Optional[EntryZone] = None
    stop_level: Optional[float] = None
    target_levels: List[float] = []

    regime_fingerprint: Optional[RegimeFingerprint] = None
    regime_match_score: float = 0.0

    matched_variant: Optional[str] = None
    variant_similarity: float = 0.0
    variant_provenance: Optional[str] = None  # "system" | "user_taught" | "hybrid"
    variant_conviction_boost: float = 0.0     # adjustment from per-variant scoring
    variant_regime_match: float = 0.0         # how well current regime fits variant affinity
    variant_narrative: Optional[str] = None   # user's explanation of the variant


# ── API Request/Response Models ──

class SKOCreateRequest(BaseModel):
    name: str
    short_code: str
    description: Optional[str] = None
    tags: List[str] = []
    scoring_config: Dict[str, Any] = {}
    conviction_map: Dict[str, List[float]] = {}
    entry_zones: List[Dict[str, Any]] = []
    exit_rules: Dict[str, Any] = {}
    risk_parameters: Dict[str, Any] = {}


class SKOUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    scoring_config: Optional[Dict[str, Any]] = None
    conviction_map: Optional[Dict[str, Any]] = None
    entry_zones: Optional[List[Dict[str, Any]]] = None
    exit_rules: Optional[Dict[str, Any]] = None
    risk_parameters: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class SKOSummary(BaseModel):
    id: UUID
    name: str
    short_code: str
    status: SKOStatus
    tier: int = 0
    total_trades: int = 0
    win_rate: Optional[float] = None
    expectancy: Optional[float] = None
    dqal_grade: Optional[str] = None
    tags: List[str] = []
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SKODetail(SKOSummary):
    description: Optional[str] = None
    narrative: Optional[str] = None
    profit_factor: Optional[float] = None
    scoring_config: Dict[str, Any] = {}
    conviction_map: Dict[str, Any] = {}
    entry_zones: List[Dict[str, Any]] = []
    exit_rules: Dict[str, Any] = {}
    risk_parameters: Dict[str, Any] = {}
    regime_fingerprint: Dict[str, Any] = {}
    time_preferences: Dict[str, Any] = {}
    dqal_score: Optional[float] = None
    variant_count: int = 0
    signal_count: int = 0
    updated_at: Optional[datetime] = None
    last_signal_at: Optional[datetime] = None


class SignalRequest(BaseModel):
    symbol: str
    candles: Optional[List[Dict[str, Any]]] = None  # if not provided, fetched internally


class BatchSignalRequest(BaseModel):
    symbols: List[str]


class ScoreRequest(BaseModel):
    """Ad-hoc scoring: pass feature values + scoring params → get score."""
    feature_values: Dict[str, float]
    scoring_config: Dict[str, Any]
    conviction_map: Optional[Dict[str, List[float]]] = None


# ── Trade Quality Gate Models ──

class TQWComponents(BaseModel):
    """Individual components feeding into the Trade Quality Weight."""
    discipline_score: Optional[float] = None       # 0-100 from attribution
    decision_quality_score: Optional[float] = None  # 0-100 from DQ evaluator
    journal_quality_score: Optional[float] = None   # 0-100 derived from journal
    override_penalty: Optional[float] = None        # 0-1 penalty factor

    def available_count(self) -> int:
        count = 0
        if self.discipline_score is not None:
            count += 1
        if self.decision_quality_score is not None:
            count += 1
        if self.journal_quality_score is not None:
            count += 1
        if self.override_penalty is not None:
            count += 1
        return count


class TQWResult(BaseModel):
    """Output of the Adaptive TQW Computer."""
    composite_tqw: float = Field(ge=0.0, le=1.0)
    components: TQWComponents
    data_confidence: float = Field(ge=0.0, le=1.0)
    weights_used: Dict[str, float] = {}
    excluded: bool = False
    exclusion_reason: Optional[str] = None


class AlignmentResult(BaseModel):
    """Output of the Strategy Alignment Checker."""
    aligned: bool = True
    similarity_score: float = 0.0
    centroid_confidence: float = 0.0
    threshold_used: float = 0.5
    activation_mode: str = "disabled"  # disabled, soft, moderate, strict
    nearest_variant: Optional[str] = None
    anti_variant_match: bool = False


class QualityGateResult(BaseModel):
    """Full output of the 4-layer quality gate pipeline."""
    trade_id: Optional[str] = None
    learning_weight: float = Field(ge=0.0, le=1.0, default=1.0)
    tqw: TQWResult
    alignment: AlignmentResult
    human_override: Optional[float] = None  # None = no override
    layer1_passed: bool = True   # behavioral hard filter
    layer2_tqw: float = 1.0     # composite TQW
    layer3_aligned: bool = True  # strategy alignment
    layer4_override: Optional[float] = None
    exclusion_reasons: List[str] = []
    route: str = "filtered"  # "filtered" | "unfiltered_only" | "excluded"


class TierEvaluation(BaseModel):
    """Tier progression evaluation using effective sample size."""
    raw_count: int = 0
    filtered_count: int = 0
    effective_count: float = 0.0
    current_tier: str = "hypothesis"
    eligible_tier: str = "hypothesis"
    promoted: bool = False
    requirements_met: Dict[str, bool] = {}


class DivergenceSnapshot(BaseModel):
    """Point-in-time divergence between filtered and unfiltered models."""
    filtered_win_rate: Optional[float] = None
    unfiltered_win_rate: Optional[float] = None
    filtered_expectancy_r: Optional[float] = None
    unfiltered_expectancy_r: Optional[float] = None
    behavioral_cost_per_trade_r: Optional[float] = None
    centroid_divergence_cosine: Optional[float] = None
    divergence_interpretation: Optional[str] = None
    effective_sample_size: Optional[float] = None
    feature_divergence: Dict[str, Any] = {}
    exclusion_breakdown: Dict[str, int] = {}
