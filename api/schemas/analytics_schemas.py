# analytics_schemas.py
# api/schemas/analytics_schemas.py (additions for behavioral)
"""Behavioral analytics response schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

class DimensionScoreSchema(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    weight: float
    weighted_score: float
    factors: list[str]
    deductions: list[str]


class DecisionQualityReportSchema(BaseModel):
    trade_id: str
    timestamp: str
    composite_score: float = Field(ge=0.0, le=1.0)
    grade: str
    is_quality_trade: bool
    dimensions: dict[str, DimensionScoreSchema]
    strengths: list[str]
    weaknesses: list[str]
    coaching_notes: list[str]


class DecisionQualityTrendSchema(BaseModel):
    average_score: float
    trend: str  # "improving", "declining", "stable", "insufficient_data"
    dimension_averages: dict[str, float]
    quality_rate: float
    total_trades: int
    grade_distribution: dict[str, int]

class BaselineResponse(BaseModel):
    trade_frequency: dict = {}
    timing: dict = {}
    discipline: dict = {}
    risk_behavior: dict = {}
    emotional: dict = {}
    resilience: dict = {}


class ImprovementResponse(BaseModel):
    periods: list[dict] = []
    trends: dict = {}
    improvement_score: float = 0.0
    milestones: list[str] = []


class BiasDetectionResponse(BaseModel):
    bias_type: str
    confidence: float
    severity: str
    description: str
    evidence: list[str] = []
    recommendation: str = ""
    related_trades: int = 0


class DeviationAlertResponse(BaseModel):
    type: str
    severity: str
    message: str
    details: dict = {}
    recommendation: str = ""


class BehavioralDashboardResponse(BaseModel):
    baseline: Optional[BaselineResponse] = None
    improvement: Optional[ImprovementResponse] = None
    active_biases: list[BiasDetectionResponse] = []
    session_alerts: list[DeviationAlertResponse] = []
    discipline_score_today: Optional[float] = None
    clean_execution_rate_today: Optional[float] = None