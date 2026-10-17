# coplanner/models.py
"""
Co-Planner Intelligence Engine — Data Models.

Shared data contracts between all co-planner engine modules.
Uses PTIS enums where they exist; adds co-planner-specific ones.

RULE: These are engine-internal dataclasses, NOT SQLAlchemy models.
      DB models live in core/models.py; these are computation containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# Enums (co-planner specific — don't duplicate core/enums.py)
# ─────────────────────────────────────────────

class ConvictionLevel(str, Enum):
    HIGH = "HIGH_CONVICTION"
    MODERATE = "MODERATE"
    LOW = "LOW_CONVICTION"
    AVOID = "AVOID"


class RRGrade(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    TERRIBLE = "TERRIBLE"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"               # 30+ trades
    MODERATE = "MODERATE"       # 15-29 trades
    LOW = "LOW"                 # 5-14 trades
    INSUFFICIENT = "INSUFFICIENT"  # <5 trades


class WarningType(str, Enum):
    REGIME_MISMATCH = "REGIME_MISMATCH"
    LOW_CLUSTER_WR = "LOW_CLUSTER_WR"
    VOLATILITY_OUTSIDE = "VOLATILITY_OUTSIDE"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    POOR_RR = "POOR_RR"
    SAMPLE_SIZE_LOW = "SAMPLE_SIZE_LOW"
    STRATEGY_WEAK_IN_REGIME = "STRATEGY_WEAK_IN_REGIME"


class DataTrustTier(str, Enum):
    """Trade data quality tiers for matrix building."""
    AVT_AUTONOMOUS = "avt_autonomous"     # 100% trust
    USER_QUALITY = "user_quality"         # 70% trust — clean execution trades
    USER_ALL = "user_all"                 # 30% trust — includes emotional trades


# ─────────────────────────────────────────────
# Core Data Models
# ─────────────────────────────────────────────

@dataclass
class SetupData:
    """
    A single trading setup from watchlist scan.
    INPUT to conviction engine.
    """
    symbol: str
    strategy_name: str
    timeframe: str = ""
    pattern_data: Dict[str, Any] = field(default_factory=dict)
    cluster_id: Optional[str] = None
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    current_volume: float = 0.0
    avg_volume: float = 0.0
    current_regime: str = "unknown"
    scan_timestamp: datetime = field(default_factory=datetime.now)

    # Optional enrichment from watchlist_analyzer
    pattern_score: float = 0.0
    watchlist_meta: Dict[str, Any] = field(default_factory=dict)

    def compute_raw_rr(self) -> float:
        """Basic risk-reward from entry/SL/target."""
        if self.entry_price == 0 or self.stop_loss == 0:
            return 0.0
        risk = abs(self.entry_price - self.stop_loss)
        if risk == 0:
            return 0.0
        reward = abs(self.target_price - self.entry_price)
        return round(reward / risk, 2)


@dataclass
class ConvictionResult:
    """Output from ConvictionEngine for a single setup."""
    symbol: str
    strategy_name: str
    conviction_score: float                  # 0.0 to 1.0
    conviction_level: ConvictionLevel

    # Component breakdown
    components: Dict[str, float] = field(default_factory=dict)

    # Warnings
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    # Sizing recommendation
    sizing_modifier: float = 1.0             # 0.0 to 1.0
    verdict: str = ""

    # RR assessment (if prices available)
    rr_assessment: Optional[RRAssessment] = None

    computed_at: datetime = field(default_factory=datetime.now)


@dataclass
class RegimePerformanceStats:
    """Performance of a strategy in a specific regime."""
    strategy_name: str
    regime: str
    wins: int = 0
    losses: int = 0
    total_trades: int = 0
    win_rate: float = 0.0
    avg_rr: float = 0.0
    total_rr: float = 0.0
    max_win_rr: float = 0.0
    max_loss_rr: float = 0.0
    avg_holding_period: float = 0.0          # hours
    sample_confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    last_updated: datetime = field(default_factory=datetime.now)

    def normalized_score(self) -> float:
        """
        WR + expectancy blended into 0-1, pulled toward 0.5
        when confidence is low.
        """
        if self.total_trades == 0:
            return 0.5

        wr_component = self.win_rate
        expectancy = self._compute_expectancy()
        exp_normalized = max(0.0, min(1.0, (expectancy + 1.0) / 2.0))
        conf_weight = self._confidence_weight()

        raw = wr_component * 0.5 + exp_normalized * 0.5
        return raw * conf_weight + 0.5 * (1.0 - conf_weight)

    def _compute_expectancy(self) -> float:
        if self.total_trades == 0:
            return 0.0
        avg_win = self.total_rr / self.wins if self.wins > 0 else 0.0
        avg_loss = abs(self.total_rr) / self.losses if self.losses > 0 else 0.0
        return (self.win_rate * avg_win) - ((1.0 - self.win_rate) * avg_loss)

    def _confidence_weight(self) -> float:
        if self.total_trades >= 30:
            return 0.95
        elif self.total_trades >= 15:
            return 0.80
        elif self.total_trades >= 5:
            return 0.60
        return 0.30


@dataclass
class RRAssessment:
    """Output from RR Enforcer."""
    expected_rr: float
    grade: RRGrade
    acceptable: bool
    message: str
    sizing_modifier: float                   # 0.0 to 1.0
    suggested_target: Optional[float] = None
    suggestion_message: Optional[str] = None


@dataclass
class AvoidanceAdvice:
    """When strategy-regime matrix says AVOID."""
    avoid: bool
    strategy_name: str
    current_regime: str
    reason: str = ""
    current_wr: float = 0.0
    sample_size: int = 0
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TradeRecord:
    """
    A completed trade used for matrix building.
    Mapped from PTIS Trade ORM model by db_interface.
    """
    trade_id: str
    symbol: str
    strategy_name: str
    timeframe: str
    regime_at_entry: str
    entry_price: float
    exit_price: float
    stop_loss: float
    target_price: float
    entry_time: datetime
    exit_time: datetime
    outcome: str                             # 'win' or 'loss'
    actual_rr: float
    volume_at_entry: float = 0.0
    cluster_id: Optional[str] = None
    trust_tier: DataTrustTier = DataTrustTier.USER_ALL


@dataclass
class BehavioralCheckResult:
    """Output from BehavioralPlannerGuard."""
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    risk_score: int = 0
    max_risk_score: int = 10
    verdict: str = ""
    proceed: bool = True


@dataclass
class RankedOpportunity:
    """A single ranked setup from OpportunityRanker."""
    rank: int
    symbol: str
    strategy_name: str
    composite_score: float
    conviction: ConvictionResult
    rr_assessment: Optional[RRAssessment] = None
    regime_stats: Optional[RegimePerformanceStats] = None
    avoidance: Optional[AvoidanceAdvice] = None

    # Score breakdown
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    # Action
    action: str = ""                         # TAKE, CAUTIOUS, SKIP
    sizing: float = 0.0
    explanation: str = ""


@dataclass
class PlanSessionResult:
    """Complete output from CoPlannerEngine.plan_session()."""
    behavioral_check: BehavioralCheckResult
    ranked_opportunities: List[RankedOpportunity] = field(default_factory=list)
    regime_warnings: List[Dict[str, Any]] = field(default_factory=list)
    recommendation: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    total_scanned: int = 0
    actionable_count: int = 0
    computed_at: datetime = field(default_factory=datetime.now)
