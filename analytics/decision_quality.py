"""
PTIS Decision Quality Engine
============================

Evaluates trade decision quality independent of P&L across six
scored dimensions.  Produces a composite score, execution grade,
and actionable improvement insights.

Dimensions
----------
1. Setup Alignment    – trade ↔ strategy match
2. Timing Quality     – entry / signal timing
3. Risk Discipline    – risk‑rule adherence
4. Emotional Stability– emotional grounding
5. Market Context     – environment fit
6. Exit Discipline    – exit‑plan honour

Design
------
• Deterministic (no ML)
• Config‑driven weights / thresholds
• Graceful degradation on partial data
• Fully explainable outputs
• Event‑bus compatible
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════

class DimensionName(str, Enum):
    """Scored decision dimensions."""
    SETUP_ALIGNMENT = "setup_alignment"
    TIMING_QUALITY = "timing_quality"
    RISK_DISCIPLINE = "risk_discipline"
    EMOTIONAL_STABILITY = "emotional_stability"
    MARKET_CONTEXT = "market_context"
    EXIT_DISCIPLINE = "exit_discipline"


class ExecutionGrade(str, Enum):
    """Human‑readable execution classification."""
    TEXTBOOK = "textbook"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNDISCIPLINED = "undisciplined"


class InsightSeverity(str, Enum):
    """Priority level for an improvement insight."""
    CRITICAL = "critical"
    WARNING = "warning"
    SUGGESTION = "suggestion"
    POSITIVE = "positive"


class EmotionalStateTag(str, Enum):
    """Trader self‑reported or inferred emotional state."""
    CALM = "calm"
    FOCUSED = "focused"
    CONFIDENT = "confident"
    EXCITED = "excited"
    ANXIOUS = "anxious"
    FRUSTRATED = "frustrated"
    FEARFUL = "fearful"
    GREEDY = "greedy"
    REVENGE = "revenge"
    BORED = "bored"
    NEUTRAL = "neutral"


class TradeOutcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DimensionWeight:
    """Weight assigned to each scoring dimension."""
    setup_alignment: float = 0.20
    timing_quality: float = 0.15
    risk_discipline: float = 0.25
    emotional_stability: float = 0.15
    market_context: float = 0.10
    exit_discipline: float = 0.15

    def as_dict(self) -> Dict[DimensionName, float]:
        return {
            DimensionName.SETUP_ALIGNMENT: self.setup_alignment,
            DimensionName.TIMING_QUALITY: self.timing_quality,
            DimensionName.RISK_DISCIPLINE: self.risk_discipline,
            DimensionName.EMOTIONAL_STABILITY: self.emotional_stability,
            DimensionName.MARKET_CONTEXT: self.market_context,
            DimensionName.EXIT_DISCIPLINE: self.exit_discipline,
        }


@dataclass(frozen=True)
class GradeThresholds:
    """Score boundaries for execution grades."""
    textbook: float = 85.0
    good: float = 70.0
    acceptable: float = 55.0
    poor: float = 40.0
    # below poor → undisciplined


@dataclass(frozen=True)
class DecisionQualityConfig:
    """All tuneable knobs for the Decision Quality Engine."""
    weights: DimensionWeight = field(default_factory=DimensionWeight)
    grade_thresholds: GradeThresholds = field(default_factory=GradeThresholds)

    # Risk discipline
    max_position_size_deviation_pct: float = 0.25
    max_portfolio_heat_pct: float = 0.06
    min_reward_risk_ratio: float = 1.5

    # Timing
    max_entry_atr_slippage: float = 0.5  # half ATR
    preferred_session_minutes: Tuple[int, int] = (30, 120)  # minutes from open

    # Emotional
    revenge_trade_window_seconds: int = 300  # 5 min
    fomo_atr_threshold: float = 1.5  # move already > 1.5 ATR

    # Exit
    stop_slippage_tolerance_atr: float = 0.25
    premature_exit_captured_min: float = 0.5

    # Insight generation
    low_dimension_threshold: float = 50.0
    critical_dimension_threshold: float = 30.0


# ═══════════════════════════════════════════════════════════════════
# INPUT DATA MODEL
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RecentTradeSnapshot:
    """Lightweight summary of a preceding trade for streak/pattern detection."""
    trade_id: Optional[UUID] = None
    pnl: float = 0.0
    r_multiple: Optional[float] = None
    outcome: TradeOutcome = TradeOutcome.BREAKEVEN
    exit_time: Optional[datetime] = None
    emotional_state: Optional[EmotionalStateTag] = None
    position_size: Optional[float] = None


@dataclass
class TradeContext:
    """
    Aggregated context for a single trade evaluation.

    The caller (trade pipeline / context assembler) populates this
    from various system components.  Every field is Optional so
    the engine degrades gracefully when data is missing.
    """

    # ── identity ────────────────────────────────────────────────
    trade_id: Optional[UUID] = None

    # ── basic trade data ────────────────────────────────────────
    symbol: Optional[str] = None
    side: Optional[str] = None  # "long" | "short"
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    quantity: Optional[float] = None
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None

    # ── strategy / setup ────────────────────────────────────────
    strategy_name: Optional[str] = None
    strategy_match_score: Optional[float] = None  # 0‑100
    conditions_met: Optional[int] = None
    conditions_total: Optional[int] = None
    regime_compatible: Optional[bool] = None

    # ── risk planning ───────────────────────────────────────────
    planned_stop_price: Optional[float] = None
    actual_stop_price: Optional[float] = None
    planned_target_price: Optional[float] = None
    planned_position_size: Optional[float] = None
    actual_position_size: Optional[float] = None
    planned_risk_reward: Optional[float] = None
    portfolio_heat: Optional[float] = None  # fraction
    max_daily_loss_limit: Optional[float] = None
    current_daily_pnl: Optional[float] = None

    # ── emotional / behavioural ────────────────────────────────
    emotional_state: Optional[EmotionalStateTag] = None
    confidence_level: Optional[float] = None  # 0‑100
    notes: Optional[str] = None
    override_events: List[Dict[str, Any]] = field(default_factory=list)
    override_severity: Optional[float] = None  # 0‑1

    # ── market context ──────────────────────────────────────────
    atr: Optional[float] = None
    volatility_regime: Optional[str] = None  # "low" / "normal" / "high" / "extreme"
    trend_direction: Optional[str] = None    # "up" / "down" / "neutral"
    trend_strength: Optional[float] = None   # 0‑100
    vix_level: Optional[float] = None
    session_type: Optional[str] = None       # "open" / "mid" / "close" / "overnight"
    market_open_time: Optional[datetime] = None

    # ── shadow comparison ───────────────────────────────────────
    shadow_entry_price: Optional[float] = None
    shadow_exit_price: Optional[float] = None
    shadow_pnl: Optional[float] = None
    optimal_entry_price: Optional[float] = None
    optimal_exit_price: Optional[float] = None

    # ── recent history (for streak / pattern) ───────────────────
    recent_trades: List[RecentTradeSnapshot] = field(default_factory=list)
    trade_count_today: int = 0
    time_since_last_trade_seconds: Optional[float] = None

    # ── convenience helpers ─────────────────────────────────────

    @property
    def outcome(self) -> TradeOutcome:
        if self.pnl is None:
            return TradeOutcome.BREAKEVEN
        if self.pnl > 0:
            return TradeOutcome.WIN
        if self.pnl < 0:
            return TradeOutcome.LOSS
        return TradeOutcome.BREAKEVEN

    @property
    def has_strategy(self) -> bool:
        return self.strategy_name is not None and self.strategy_match_score is not None

    @property
    def has_shadow(self) -> bool:
        return self.shadow_entry_price is not None

    @property
    def recent_loss_streak(self) -> int:
        count = 0
        for t in self.recent_trades:
            if t.outcome == TradeOutcome.LOSS:
                count += 1
            else:
                break
        return count

    @property
    def recent_win_streak(self) -> int:
        count = 0
        for t in self.recent_trades:
            if t.outcome == TradeOutcome.WIN:
                count += 1
            else:
                break
        return count

    def minutes_from_open(self) -> Optional[float]:
        if self.entry_time and self.market_open_time:
            delta = (self.entry_time - self.market_open_time).total_seconds() / 60.0
            return max(delta, 0.0)
        return None


# ═══════════════════════════════════════════════════════════════════
# OUTPUT DATA MODELS
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DimensionScore:
    """Scored result for a single decision dimension."""
    name: DimensionName
    score: float  # 0 – 100
    weight: float
    confidence: float  # 0 – 1  (data completeness)
    components: Dict[str, float]  # named sub‑scores
    reasoning: Tuple[str, ...]

    @property
    def weighted_contribution(self) -> float:
        return self.score * self.weight * self.confidence


@dataclass(frozen=True)
class ImprovementInsight:
    """Actionable improvement suggestion."""
    insight_id: UUID
    dimension: DimensionName
    severity: InsightSeverity
    title: str
    description: str
    suggestion: str
    score_impact: float  # estimated score points lost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": str(self.insight_id),
            "dimension": self.dimension.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "suggestion": self.suggestion,
            "score_impact": round(self.score_impact, 1),
        }


@dataclass(frozen=True)
class DecisionQualityResult:
    """Complete engine output for one trade evaluation."""
    trade_id: Optional[UUID]
    evaluation_id: UUID
    evaluated_at: datetime
    final_score: float  # 0 – 100
    grade: ExecutionGrade
    dimensions: Tuple[DimensionScore, ...]
    insights: Tuple[ImprovementInsight, ...]
    outcome_independent: bool  # True = score valid regardless of P&L
    data_completeness: float  # 0 – 1

    def dimension(self, name: DimensionName) -> Optional[DimensionScore]:
        for d in self.dimensions:
            if d.name == name:
                return d
        return None

    @property
    def critical_insights(self) -> Tuple[ImprovementInsight, ...]:
        return tuple(i for i in self.insights if i.severity == InsightSeverity.CRITICAL)

    @property
    def positive_insights(self) -> Tuple[ImprovementInsight, ...]:
        return tuple(i for i in self.insights if i.severity == InsightSeverity.POSITIVE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": str(self.trade_id) if self.trade_id else None,
            "evaluation_id": str(self.evaluation_id),
            "evaluated_at": self.evaluated_at.isoformat(),
            "final_score": round(self.final_score, 1),
            "grade": self.grade.value,
            "data_completeness": round(self.data_completeness, 2),
            "dimensions": {
                d.name.value: {
                    "score": round(d.score, 1),
                    "weight": d.weight,
                    "confidence": round(d.confidence, 2),
                    "components": {k: round(v, 1) for k, v in d.components.items()},
                    "reasoning": list(d.reasoning),
                }
                for d in self.dimensions
            },
            "insights": [i.to_dict() for i in self.insights],
        }


# ═══════════════════════════════════════════════════════════════════
# BASE SCORER
# ═══════════════════════════════════════════════════════════════════

class BaseDimensionScorer(ABC):
    """Abstract base for all dimension scorers."""

    dimension: DimensionName  # subclass must set

    def __init__(self, config: DecisionQualityConfig) -> None:
        self.cfg = config

    @abstractmethod
    def score(self, ctx: TradeContext) -> DimensionScore:
        ...

    # ── helpers shared by all scorers ───────────────────────────

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _linear_decay(value: float, threshold: float) -> float:
        """Return 1.0 when value==0, decaying linearly to 0.0 at *threshold*."""
        if threshold <= 0:
            return 0.0
        return max(0.0, 1.0 - value / threshold)

    def _make_score(
        self,
        raw: float,
        confidence: float,
        components: Dict[str, float],
        reasoning: List[str],
    ) -> DimensionScore:
        weight = self.cfg.weights.as_dict()[self.dimension]
        return DimensionScore(
            name=self.dimension,
            score=self._clamp(raw),
            weight=weight,
            confidence=max(0.0, min(1.0, confidence)),
            components={k: self._clamp(v) for k, v in components.items()},
            reasoning=tuple(reasoning),
        )


# ═══════════════════════════════════════════════════════════════════
# DIMENSION 1 — SETUP ALIGNMENT
# ═══════════════════════════════════════════════════════════════════

class SetupAlignmentScorer(BaseDimensionScorer):
    """How well the trade matched a defined strategy setup."""

    dimension = DimensionName.SETUP_ALIGNMENT

    def score(self, ctx: TradeContext) -> DimensionScore:
        components: Dict[str, float] = {}
        reasoning: List[str] = []
        data_points = 0
        max_data_points = 4

        # ── 1. Strategy match quality (0–40) ───────────────────
        if ctx.has_strategy:
            match_raw = (ctx.strategy_match_score or 0.0) / 100.0
            components["match_quality"] = match_raw * 40.0
            reasoning.append(
                f"Strategy '{ctx.strategy_name}' matched at "
                f"{ctx.strategy_match_score:.0f}%."
            )
            data_points += 1
        else:
            components["match_quality"] = 15.0  # discretionary base
            reasoning.append("No strategy template matched — discretionary trade.")

        # ── 2. Condition satisfaction (0–30) ────────────────────
        if ctx.conditions_met is not None and ctx.conditions_total:
            ratio = ctx.conditions_met / ctx.conditions_total
            components["condition_satisfaction"] = ratio * 30.0
            reasoning.append(
                f"{ctx.conditions_met}/{ctx.conditions_total} conditions satisfied "
                f"({ratio:.0%})."
            )
            data_points += 1
        else:
            components["condition_satisfaction"] = 15.0
            reasoning.append("Condition data unavailable — neutral score applied.")

        # ── 3. Regime compatibility (0–15) ──────────────────────
        if ctx.regime_compatible is not None:
            components["regime_compat"] = 15.0 if ctx.regime_compatible else 3.0
            reasoning.append(
                "Regime compatible." if ctx.regime_compatible
                else "Trade taken in incompatible regime."
            )
            data_points += 1
        else:
            components["regime_compat"] = 8.0
            reasoning.append("Regime compatibility unknown.")

        # ── 4. Override penalty (0–15) ──────────────────────────
        if ctx.override_severity is not None:
            penalty = ctx.override_severity  # 0 = no override, 1 = severe
            components["override_adherence"] = 15.0 * (1.0 - penalty)
            if penalty > 0:
                reasoning.append(
                    f"Override detected (severity {penalty:.2f}) — "
                    f"deviations from strategy rules reduce score."
                )
            else:
                reasoning.append("No overrides detected.")
            data_points += 1
        elif ctx.override_events:
            avg_sev = sum(
                e.get("severity", 0.5) for e in ctx.override_events
            ) / len(ctx.override_events)
            components["override_adherence"] = 15.0 * (1.0 - avg_sev)
            reasoning.append(
                f"{len(ctx.override_events)} override event(s) detected."
            )
            data_points += 1
        else:
            components["override_adherence"] = 15.0
            data_points += 1

        raw = sum(components.values())
        confidence = data_points / max_data_points
        return self._make_score(raw, confidence, components, reasoning)


# ═══════════════════════════════════════════════════════════════════
# DIMENSION 2 — TIMING QUALITY
# ═══════════════════════════════════════════════════════════════════

class TimingQualityScorer(BaseDimensionScorer):
    """Was entry timing appropriate relative to signal and session?"""

    dimension = DimensionName.TIMING_QUALITY

    def score(self, ctx: TradeContext) -> DimensionScore:
        components: Dict[str, float] = {}
        reasoning: List[str] = []
        data_points = 0
        max_data_points = 4

        # ── 1. Entry price vs optimal / shadow (0–40) ──────────
        if ctx.has_shadow and ctx.atr and ctx.atr > 0 and ctx.entry_price is not None:
            optimal = ctx.optimal_entry_price or ctx.shadow_entry_price
            if optimal is not None:
                deviation_atr = abs(ctx.entry_price - optimal) / ctx.atr
                entry_score = 40.0 * self._linear_decay(
                    deviation_atr, self.cfg.max_entry_atr_slippage * 2.0
                )
                components["entry_precision"] = entry_score
                reasoning.append(
                    f"Entry deviated {deviation_atr:.2f} ATR from optimal "
                    f"({optimal:.2f})."
                )
                data_points += 1
            else:
                components["entry_precision"] = 20.0
        else:
            components["entry_precision"] = 20.0  # neutral
            reasoning.append("No shadow/optimal entry benchmark available.")

        # ── 2. Session window (0–25) ───────────────────────────
        mins = ctx.minutes_from_open()
        if mins is not None:
            lo, hi = self.cfg.preferred_session_minutes
            if lo <= mins <= hi:
                components["session_timing"] = 25.0
                reasoning.append(
                    f"Entry {mins:.0f} min from open — within preferred window."
                )
            elif mins < lo:
                components["session_timing"] = 15.0
                reasoning.append(
                    f"Entry {mins:.0f} min from open — early / pre‑session."
                )
            else:
                # Decay beyond preferred window
                excess = mins - hi
                components["session_timing"] = max(
                    5.0, 25.0 * self._linear_decay(excess, 180.0)
                )
                reasoning.append(
                    f"Entry {mins:.0f} min from open — late in session."
                )
            data_points += 1
        else:
            components["session_timing"] = 12.5
            reasoning.append("Session timing data unavailable.")

        # ── 3. Execution speed / patience (0–20) ───────────────
        if ctx.time_since_last_trade_seconds is not None:
            if ctx.time_since_last_trade_seconds < 30:
                components["patience"] = 5.0
                reasoning.append(
                    "Rapid re‑entry (<30 s since last trade) — possible impulse."
                )
            elif ctx.time_since_last_trade_seconds < 120:
                components["patience"] = 12.0
                reasoning.append("Quick follow‑up trade (< 2 min).")
            else:
                components["patience"] = 20.0
                reasoning.append("Adequate time between trades.")
            data_points += 1
        else:
            components["patience"] = 10.0

        # ── 4. FOMO / chasing indicator (0–15) ─────────────────
        if ctx.atr and ctx.atr > 0 and ctx.entry_price is not None:
            # Approximate: if price already moved > fomo_threshold ATR from
            # day's reference (e.g., prior close or VWAP), flag.
            # We use shadow_entry as proxy for "where signal fired".
            if ctx.shadow_entry_price is not None:
                move_atr = abs(ctx.entry_price - ctx.shadow_entry_price) / ctx.atr
                if move_atr > self.cfg.fomo_atr_threshold:
                    components["fomo_risk"] = 3.0
                    reasoning.append(
                        f"Price moved {move_atr:.1f} ATR before entry — "
                        f"possible chasing."
                    )
                else:
                    components["fomo_risk"] = 15.0
                data_points += 1
            else:
                components["fomo_risk"] = 8.0
        else:
            components["fomo_risk"] = 8.0

        raw = sum(components.values())
        confidence = data_points / max_data_points
        return self._make_score(raw, confidence, components, reasoning)


# ═══════════════════════════════════════════════════════════════════
# DIMENSION 3 — RISK DISCIPLINE
# ═══════════════════════════════════════════════════════════════════

class RiskDisciplineScorer(BaseDimensionScorer):
    """Were position sizing, stops, and risk rules followed?"""

    dimension = DimensionName.RISK_DISCIPLINE

    def score(self, ctx: TradeContext) -> DimensionScore:
        components: Dict[str, float] = {}
        reasoning: List[str] = []
        data_points = 0
        max_data_points = 5

        # ── 1. Position sizing adherence (0–30) ────────────────
        if (
            ctx.planned_position_size is not None
            and ctx.actual_position_size is not None
            and ctx.planned_position_size > 0
        ):
            dev = abs(ctx.actual_position_size - ctx.planned_position_size) / ctx.planned_position_size
            score_val = 30.0 * self._linear_decay(
                dev, self.cfg.max_position_size_deviation_pct * 2.0
            )
            components["position_sizing"] = score_val
            if dev > self.cfg.max_position_size_deviation_pct:
                direction = (
                    "oversized" if ctx.actual_position_size > ctx.planned_position_size
                    else "undersized"
                )
                reasoning.append(
                    f"Position {direction} by {dev:.0%} vs plan "
                    f"(planned {ctx.planned_position_size}, "
                    f"actual {ctx.actual_position_size})."
                )
            else:
                reasoning.append("Position size within acceptable range of plan.")
            data_points += 1
        else:
            components["position_sizing"] = 15.0
            reasoning.append("Position sizing plan data unavailable.")

        # ── 2. Stop‑loss placement (0–25) ──────────────────────
        if ctx.planned_stop_price is not None:
            if ctx.actual_stop_price is not None:
                if ctx.atr and ctx.atr > 0:
                    stop_dev_atr = abs(ctx.actual_stop_price - ctx.planned_stop_price) / ctx.atr
                    components["stop_placement"] = 25.0 * self._linear_decay(
                        stop_dev_atr, 1.0
                    )
                else:
                    price_dev = abs(ctx.actual_stop_price - ctx.planned_stop_price)
                    if ctx.planned_stop_price != 0:
                        pct_dev = price_dev / abs(ctx.planned_stop_price)
                        components["stop_placement"] = 25.0 * self._linear_decay(pct_dev, 0.05)
                    else:
                        components["stop_placement"] = 12.0
                reasoning.append(
                    f"Stop placed at {ctx.actual_stop_price:.2f} vs "
                    f"planned {ctx.planned_stop_price:.2f}."
                )
            else:
                components["stop_placement"] = 0.0
                reasoning.append(
                    "CRITICAL: No stop placed despite having a planned stop."
                )
            data_points += 1
        else:
            # No planned stop — check if any stop at all
            if ctx.actual_stop_price is not None:
                components["stop_placement"] = 18.0
                reasoning.append("Stop placed but no pre‑trade plan recorded.")
            else:
                components["stop_placement"] = 5.0
                reasoning.append("No stop loss placed and no plan recorded.")
            data_points += 1

        # ── 3. Risk‑reward ratio (0–20) ────────────────────────
        if ctx.planned_risk_reward is not None:
            if ctx.planned_risk_reward >= self.cfg.min_reward_risk_ratio:
                components["risk_reward"] = 20.0
                reasoning.append(
                    f"Risk:reward {ctx.planned_risk_reward:.1f}:1 meets minimum "
                    f"{self.cfg.min_reward_risk_ratio:.1f}:1."
                )
            elif ctx.planned_risk_reward >= 1.0:
                ratio = (ctx.planned_risk_reward - 1.0) / (self.cfg.min_reward_risk_ratio - 1.0)
                components["risk_reward"] = 8.0 + 12.0 * min(ratio, 1.0)
                reasoning.append(
                    f"Risk:reward {ctx.planned_risk_reward:.1f}:1 below ideal but ≥ 1:1."
                )
            else:
                components["risk_reward"] = 3.0
                reasoning.append(
                    f"Risk:reward {ctx.planned_risk_reward:.1f}:1 — "
                    f"negative expectancy setup."
                )
            data_points += 1
        else:
            components["risk_reward"] = 10.0
            reasoning.append("Risk:reward data unavailable.")

        # ── 4. Portfolio heat (0–15) ────────────────────────────
        if ctx.portfolio_heat is not None:
            if ctx.portfolio_heat <= self.cfg.max_portfolio_heat_pct:
                components["portfolio_heat"] = 15.0
                reasoning.append(
                    f"Portfolio heat {ctx.portfolio_heat:.1%} within limit "
                    f"({self.cfg.max_portfolio_heat_pct:.1%})."
                )
            else:
                excess = (
                    (ctx.portfolio_heat - self.cfg.max_portfolio_heat_pct)
                    / self.cfg.max_portfolio_heat_pct
                )
                components["portfolio_heat"] = 15.0 * self._linear_decay(excess, 1.0)
                reasoning.append(
                    f"Portfolio heat {ctx.portfolio_heat:.1%} exceeds limit "
                    f"({self.cfg.max_portfolio_heat_pct:.1%})."
                )
            data_points += 1
        else:
            components["portfolio_heat"] = 8.0

        # ── 5. Daily loss limit (0–10) ─────────────────────────
        if ctx.max_daily_loss_limit is not None and ctx.current_daily_pnl is not None:
            if ctx.current_daily_pnl >= -abs(ctx.max_daily_loss_limit):
                components["daily_limit"] = 10.0
                reasoning.append("Within daily loss limit.")
            else:
                components["daily_limit"] = 0.0
                reasoning.append(
                    "CRITICAL: Trading beyond daily loss limit "
                    f"(PnL {ctx.current_daily_pnl:.2f} vs "
                    f"limit {ctx.max_daily_loss_limit:.2f})."
                )
            data_points += 1
        else:
            components["daily_limit"] = 5.0

        raw = sum(components.values())
        confidence = data_points / max_data_points
        return self._make_score(raw, confidence, components, reasoning)


# ═══════════════════════════════════════════════════════════════════
# DIMENSION 4 — EMOTIONAL STABILITY
# ═══════════════════════════════════════════════════════════════════

# Emotional‑state scoring map
_EMOTION_SCORES: Dict[EmotionalStateTag, float] = {
    EmotionalStateTag.CALM: 30.0,
    EmotionalStateTag.FOCUSED: 30.0,
    EmotionalStateTag.CONFIDENT: 25.0,
    EmotionalStateTag.NEUTRAL: 22.0,
    EmotionalStateTag.EXCITED: 15.0,
    EmotionalStateTag.BORED: 12.0,
    EmotionalStateTag.ANXIOUS: 10.0,
    EmotionalStateTag.FRUSTRATED: 8.0,
    EmotionalStateTag.FEARFUL: 5.0,
    EmotionalStateTag.GREEDY: 5.0,
    EmotionalStateTag.REVENGE: 2.0,
}


class EmotionalStabilityScorer(BaseDimensionScorer):
    """Was the trade taken from an emotionally grounded state?"""

    dimension = DimensionName.EMOTIONAL_STABILITY

    def score(self, ctx: TradeContext) -> DimensionScore:
        components: Dict[str, float] = {}
        reasoning: List[str] = []
        data_points = 0
        max_data_points = 4

        # ── 1. Self‑reported / tagged emotional state (0–30) ───
        if ctx.emotional_state is not None:
            components["emotional_state"] = _EMOTION_SCORES.get(
                ctx.emotional_state, 15.0
            )
            reasoning.append(f"Emotional state tagged: {ctx.emotional_state.value}.")
            data_points += 1
        else:
            components["emotional_state"] = 15.0
            reasoning.append("No emotional state reported — neutral assumed.")

        # ── 2. Revenge‑trading indicator (0–25) ────────────────
        revenge_score = 25.0
        if ctx.recent_trades and ctx.time_since_last_trade_seconds is not None:
            last = ctx.recent_trades[0]
            if (
                last.outcome == TradeOutcome.LOSS
                and ctx.time_since_last_trade_seconds < self.cfg.revenge_trade_window_seconds
            ):
                urgency = self._linear_decay(
                    ctx.time_since_last_trade_seconds,
                    self.cfg.revenge_trade_window_seconds,
                )
                revenge_score = 25.0 * (1.0 - urgency * 0.8)
                reasoning.append(
                    f"Trade entered {ctx.time_since_last_trade_seconds:.0f}s "
                    f"after a loss — revenge‑trade risk elevated."
                )
            data_points += 1
        components["revenge_risk"] = revenge_score

        # ── 3. Streak influence (0–25) ─────────────────────────
                # ── 3. Streak influence (0–25) ─────────────────────────
        loss_streak = ctx.recent_loss_streak
        win_streak = ctx.recent_win_streak
        if loss_streak >= 3:
            components["streak_influence"] = 5.0
            reasoning.append(
                f"Currently on a {loss_streak}‑trade loss streak — "
                f"elevated tilt risk."
            )
            data_points += 1
        elif loss_streak == 2:
            components["streak_influence"] = 12.0
            reasoning.append("Two consecutive losses — monitor emotional state.")
            data_points += 1
        elif win_streak >= 4:
            components["streak_influence"] = 15.0
            reasoning.append(
                f"On a {win_streak}‑trade win streak — overconfidence risk."
            )
            data_points += 1
        else:
            components["streak_influence"] = 25.0
            reasoning.append("No concerning win/loss streak detected.")
            data_points += 1

        # ── 4. Overtrading signal (0–20) ───────────────────────
        if ctx.trade_count_today > 0:
            if ctx.trade_count_today <= 3:
                components["overtrading"] = 20.0
                reasoning.append(
                    f"Trade #{ctx.trade_count_today} today — within normal range."
                )
            elif ctx.trade_count_today <= 6:
                components["overtrading"] = 12.0
                reasoning.append(
                    f"Trade #{ctx.trade_count_today} today — elevated frequency."
                )
            elif ctx.trade_count_today <= 10:
                components["overtrading"] = 5.0
                reasoning.append(
                    f"Trade #{ctx.trade_count_today} today — possible overtrading."
                )
            else:
                components["overtrading"] = 1.0
                reasoning.append(
                    f"Trade #{ctx.trade_count_today} today — significant overtrading."
                )
            data_points += 1
        else:
            components["overtrading"] = 10.0

        raw = sum(components.values())
        confidence = data_points / max_data_points
        return self._make_score(raw, confidence, components, reasoning)


# ═══════════════════════════════════════════════════════════════════
# DIMENSION 5 — MARKET CONTEXT
# ═══════════════════════════════════════════════════════════════════

# Volatility regime scoring — higher score = better alignment awareness
_VOLATILITY_REGIME_BASE: Dict[str, float] = {
    "low": 20.0,
    "normal": 25.0,
    "high": 18.0,
    "extreme": 10.0,
}


class MarketContextScorer(BaseDimensionScorer):
    """Did the trade align with current market environment?"""

    dimension = DimensionName.MARKET_CONTEXT

    def score(self, ctx: TradeContext) -> DimensionScore:
        components: Dict[str, float] = {}
        reasoning: List[str] = []
        data_points = 0
        max_data_points = 5

        # ── 1. Volatility regime appropriateness (0–25) ────────
        if ctx.volatility_regime is not None:
            base = _VOLATILITY_REGIME_BASE.get(ctx.volatility_regime, 15.0)
            # If regime is known AND strategy is regime‑compatible → full score
            if ctx.regime_compatible is True:
                components["volatility_fit"] = 25.0
                reasoning.append(
                    f"Volatility regime '{ctx.volatility_regime}' is compatible "
                    f"with selected strategy."
                )
            elif ctx.regime_compatible is False:
                components["volatility_fit"] = max(3.0, base * 0.4)
                reasoning.append(
                    f"Volatility regime '{ctx.volatility_regime}' is NOT compatible "
                    f"with selected strategy — edge may be reduced."
                )
            else:
                components["volatility_fit"] = base
                reasoning.append(
                    f"Volatility regime: '{ctx.volatility_regime}'. "
                    f"Regime compatibility unknown."
                )
            data_points += 1
        else:
            components["volatility_fit"] = 12.5
            reasoning.append("Volatility regime data unavailable.")

        # ── 2. Trend alignment (0–25) ──────────────────────────
        if ctx.trend_direction is not None and ctx.side is not None:
            aligned = (
                (ctx.trend_direction == "up" and ctx.side == "long")
                or (ctx.trend_direction == "down" and ctx.side == "short")
            )
            counter_trend = (
                (ctx.trend_direction == "up" and ctx.side == "short")
                or (ctx.trend_direction == "down" and ctx.side == "long")
            )

            if aligned:
                # Bonus for strong trend alignment
                strength_bonus = 0.0
                if ctx.trend_strength is not None and ctx.trend_strength > 60:
                    strength_bonus = 5.0 * (ctx.trend_strength / 100.0)
                components["trend_alignment"] = min(25.0, 20.0 + strength_bonus)
                reasoning.append(
                    f"Trade direction ({ctx.side}) aligns with "
                    f"trend ({ctx.trend_direction}"
                    f"{', strength ' + str(round(ctx.trend_strength, 0)) if ctx.trend_strength else ''})."
                )
            elif counter_trend:
                # Counter‑trend isn't always wrong (reversals), but riskier
                components["trend_alignment"] = 8.0
                reasoning.append(
                    f"Counter‑trend trade: {ctx.side} vs trend {ctx.trend_direction}. "
                    f"Higher risk — requires strong setup conviction."
                )
            else:
                # neutral trend
                components["trend_alignment"] = 15.0
                reasoning.append(
                    f"Market trend neutral — directional edge unclear."
                )
            data_points += 1
        else:
            components["trend_alignment"] = 12.5
            reasoning.append("Trend data unavailable.")

        # ── 3. VIX / macro environment (0–15) ──────────────────
        if ctx.vix_level is not None:
            if ctx.vix_level < 15:
                components["macro_env"] = 15.0
                reasoning.append(f"VIX at {ctx.vix_level:.1f} — low fear environment.")
            elif ctx.vix_level < 20:
                components["macro_env"] = 13.0
                reasoning.append(f"VIX at {ctx.vix_level:.1f} — normal environment.")
            elif ctx.vix_level < 30:
                components["macro_env"] = 9.0
                reasoning.append(
                    f"VIX at {ctx.vix_level:.1f} — elevated uncertainty. "
                    f"Reduce sizing expectations."
                )
            else:
                components["macro_env"] = 5.0
                reasoning.append(
                    f"VIX at {ctx.vix_level:.1f} — extreme fear. "
                    f"Only high‑conviction setups warranted."
                )
            data_points += 1
        else:
            components["macro_env"] = 8.0

        # ── 4. Session type suitability (0–20) ─────────────────
        if ctx.session_type is not None:
            session_scores: Dict[str, float] = {
                "open": 18.0,     # high liquidity, high signal
                "mid": 15.0,     # good — avoid chop
                "close": 12.0,   # acceptable, time risk
                "overnight": 5.0, # wide spreads, event risk
            }
            s = session_scores.get(ctx.session_type, 10.0)
            # Strategies that explicitly target certain sessions get a boost
            if ctx.has_strategy and ctx.strategy_match_score and ctx.strategy_match_score > 70:
                s = min(20.0, s + 3.0)
            components["session_suitability"] = s
            reasoning.append(f"Session type: '{ctx.session_type}'.")
            data_points += 1
        else:
            components["session_suitability"] = 10.0
            reasoning.append("Session type not determined.")

        # ── 5. ATR‑relative sizing (0–15) ──────────────────────
        if ctx.atr and ctx.atr > 0 and ctx.planned_stop_price and ctx.entry_price:
            stop_distance = abs(ctx.entry_price - ctx.planned_stop_price)
            atr_ratio = stop_distance / ctx.atr
            if 0.3 <= atr_ratio <= 1.5:
                components["atr_relative_risk"] = 15.0
                reasoning.append(
                    f"Stop distance {atr_ratio:.2f} ATR — within reasonable range."
                )
            elif atr_ratio < 0.3:
                components["atr_relative_risk"] = 8.0
                reasoning.append(
                    f"Stop distance {atr_ratio:.2f} ATR — very tight, "
                    f"high whipsaw probability."
                )
            elif atr_ratio <= 2.5:
                components["atr_relative_risk"] = 10.0
                reasoning.append(
                    f"Stop distance {atr_ratio:.2f} ATR — wide stop, "
                    f"consider smaller size."
                )
            else:
                components["atr_relative_risk"] = 4.0
                reasoning.append(
                    f"Stop distance {atr_ratio:.2f} ATR — excessively wide."
                )
            data_points += 1
        else:
            components["atr_relative_risk"] = 8.0

        raw = sum(components.values())
        confidence = data_points / max_data_points
        return self._make_score(raw, confidence, components, reasoning)


# ═══════════════════════════════════════════════════════════════════
# DIMENSION 6 — EXIT DISCIPLINE
# ═══════════════════════════════════════════════════════════════════

class ExitDisciplineScorer(BaseDimensionScorer):
    """Did the trader follow exit rules and manage the trade properly?"""

    dimension = DimensionName.EXIT_DISCIPLINE

    def score(self, ctx: TradeContext) -> DimensionScore:
        components: Dict[str, float] = {}
        reasoning: List[str] = []
        data_points = 0
        max_data_points = 4

        # ── 1. Stop honour / stop slippage (0–35) ──────────────
        if ctx.planned_stop_price is not None and ctx.exit_price is not None:
            if ctx.pnl is not None and ctx.pnl < 0:
                # Losing trade — did we honour the stop?
                if ctx.side == "long":
                    expected_loss_exit = ctx.planned_stop_price
                    slipped_past = ctx.exit_price < expected_loss_exit
                else:
                    expected_loss_exit = ctx.planned_stop_price
                    slipped_past = ctx.exit_price > expected_loss_exit

                if ctx.atr and ctx.atr > 0:
                    slip_atr = abs(ctx.exit_price - expected_loss_exit) / ctx.atr
                    if slipped_past:
                        components["stop_honour"] = max(
                            0.0,
                            35.0 * self._linear_decay(
                                slip_atr,
                                self.cfg.stop_slippage_tolerance_atr * 4.0,
                            ),
                        )
                        reasoning.append(
                            f"Exit slipped {slip_atr:.2f} ATR past planned stop — "
                            f"stop not honoured cleanly."
                        )
                    else:
                        components["stop_honour"] = 35.0
                        reasoning.append("Loss exit at or better than planned stop.")
                else:
                    # No ATR — simple boolean check
                    if slipped_past:
                        components["stop_honour"] = 10.0
                        reasoning.append("Exit price worse than planned stop.")
                    else:
                        components["stop_honour"] = 35.0
                        reasoning.append("Loss exit respected planned stop level.")
                data_points += 1
            elif ctx.pnl is not None and ctx.pnl >= 0:
                # Winning trade — stop wasn't triggered, full credit
                components["stop_honour"] = 35.0
                reasoning.append("Winning trade — stop was not breached.")
                data_points += 1
            else:
                components["stop_honour"] = 18.0
        else:
            components["stop_honour"] = 12.0
            reasoning.append("Insufficient data to evaluate stop honour.")

        # ── 2. Target adherence / profit capture (0–30) ────────
        if (
            ctx.planned_target_price is not None
            and ctx.entry_price is not None
            and ctx.exit_price is not None
        ):
            planned_move = abs(ctx.planned_target_price - ctx.entry_price)
            actual_move = ctx.exit_price - ctx.entry_price
            if ctx.side == "short":
                actual_move = -actual_move  # normalize to positive for winning

            if planned_move > 0:
                capture_ratio = actual_move / planned_move
                if capture_ratio >= 0.8:
                    components["target_adherence"] = 30.0
                    reasoning.append(
                        f"Captured {capture_ratio:.0%} of planned move — "
                        f"excellent exit management."
                    )
                elif capture_ratio >= self.cfg.premature_exit_captured_min:
                    components["target_adherence"] = 15.0 + 15.0 * (
                        (capture_ratio - self.cfg.premature_exit_captured_min)
                        / (0.8 - self.cfg.premature_exit_captured_min)
                    )
                    reasoning.append(
                        f"Captured {capture_ratio:.0%} of planned move — "
                        f"adequate but left profit on table."
                    )
                elif capture_ratio > 0:
                    components["target_adherence"] = 8.0
                    reasoning.append(
                        f"Captured only {capture_ratio:.0%} of planned move — "
                        f"premature exit."
                    )
                else:
                    components["target_adherence"] = 3.0
                    reasoning.append(
                        f"Exited with loss despite price moving partway to target."
                    )
            else:
                components["target_adherence"] = 15.0
            data_points += 1
        else:
            components["target_adherence"] = 15.0
            reasoning.append("Target price not available for evaluation.")

        # ── 3. Shadow comparison (exit quality) (0–20) ─────────
        if (
            ctx.has_shadow
            and ctx.shadow_exit_price is not None
            and ctx.exit_price is not None
            and ctx.atr
            and ctx.atr > 0
        ):
            exit_dev_atr = abs(ctx.exit_price - ctx.shadow_exit_price) / ctx.atr
            # Determine if trader exit was worse or better
            if ctx.side == "long":
                trader_better = ctx.exit_price >= ctx.shadow_exit_price
            else:
                trader_better = ctx.exit_price <= ctx.shadow_exit_price

            if trader_better:
                components["shadow_exit_quality"] = 20.0
                reasoning.append(
                    "Exit price better than shadow trade exit — strong discipline."
                )
            else:
                components["shadow_exit_quality"] = max(
                    2.0,
                    20.0 * self._linear_decay(exit_dev_atr, 2.0),
                )
                reasoning.append(
                    f"Exit {exit_dev_atr:.2f} ATR worse than shadow trade exit."
                )
            data_points += 1
        else:
            components["shadow_exit_quality"] = 10.0

        # ── 4. Exit timing relative to trade duration (0–15) ───
        if ctx.entry_time and ctx.exit_time:
            duration = (ctx.exit_time - ctx.entry_time).total_seconds()
            if duration < 15:
                components["exit_timing"] = 5.0
                reasoning.append(
                    f"Trade held <15 seconds — panic exit or scalp not planned?"
                )
            elif duration < 60 and (ctx.strategy_name and "scalp" not in ctx.strategy_name.lower()):
                components["exit_timing"] = 8.0
                reasoning.append(
                    f"Trade held ~{duration:.0f}s — very short for "
                    f"non‑scalp strategy."
                )
            else:
                components["exit_timing"] = 15.0
                reasoning.append(
                    f"Trade duration ({duration / 60:.1f} min) reasonable."
                )
            data_points += 1
        else:
            components["exit_timing"] = 8.0

        raw = sum(components.values())
        confidence = data_points / max_data_points
        return self._make_score(raw, confidence, components, reasoning)


# ═══════════════════════════════════════════════════════════════════
# INSIGHT GENERATOR
# ═══════════════════════════════════════════════════════════════════

class InsightGenerator:
    """
    Analyses dimension scores and trade context to produce
    actionable improvement insights.
    """

    def __init__(self, config: DecisionQualityConfig) -> None:
        self.cfg = config

    def generate(
        self,
        dimensions: Sequence[DimensionScore],
        ctx: TradeContext,
    ) -> List[ImprovementInsight]:
        """Produce ordered list of insights from scored dimensions."""
        insights: List[ImprovementInsight] = []

        for dim in dimensions:
            insights.extend(self._insights_for_dimension(dim, ctx))

        # Add cross‑dimensional insights
        insights.extend(self._cross_dimensional_insights(dimensions, ctx))

        # Add positive reinforcement
        insights.extend(self._positive_reinforcement(dimensions, ctx))

        # Sort: critical first, then by score impact descending
        severity_order = {
            InsightSeverity.CRITICAL: 0,
            InsightSeverity.WARNING: 1,
            InsightSeverity.SUGGESTION: 2,
            InsightSeverity.POSITIVE: 3,
        }
        insights.sort(
            key=lambda i: (severity_order.get(i.severity, 9), -i.score_impact)
        )

        return insights

    # ── per‑dimension insight extraction ────────────────────────

    def _insights_for_dimension(
        self,
        dim: DimensionScore,
        ctx: TradeContext,
    ) -> List[ImprovementInsight]:
        insights: List[ImprovementInsight] = []

        if dim.score < self.cfg.critical_dimension_threshold:
            severity = InsightSeverity.CRITICAL
        elif dim.score < self.cfg.low_dimension_threshold:
            severity = InsightSeverity.WARNING
        else:
            return insights  # no negative insight needed

        impact = (100.0 - dim.score) * dim.weight

        # Dispatch per dimension
        dispatch = {
            DimensionName.SETUP_ALIGNMENT: self._setup_insight,
            DimensionName.TIMING_QUALITY: self._timing_insight,
            DimensionName.RISK_DISCIPLINE: self._risk_insight,
            DimensionName.EMOTIONAL_STABILITY: self._emotional_insight,
            DimensionName.MARKET_CONTEXT: self._market_insight,
            DimensionName.EXIT_DISCIPLINE: self._exit_insight,
        }

        generator = dispatch.get(dim.name)
        if generator:
            insight = generator(dim, ctx, severity, impact)
            if insight:
                insights.append(insight)

        return insights

    def _setup_insight(
        self,
        dim: DimensionScore,
        ctx: TradeContext,
        severity: InsightSeverity,
        impact: float,
    ) -> Optional[ImprovementInsight]:
        if not ctx.has_strategy:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=severity,
                title="No Strategy Template Matched",
                description=(
                    "This trade was not aligned with any defined strategy. "
                    "Discretionary trades without a framework reduce consistency."
                ),
                suggestion=(
                    "Before entering, identify which playbook setup applies. "
                    "If none match, consider passing on the trade."
                ),
                score_impact=impact,
            )

        if dim.components.get("condition_satisfaction", 100) < 15:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=severity,
                title="Low Condition Satisfaction",
                description=(
                    f"Only {ctx.conditions_met}/{ctx.conditions_total} setup "
                    f"conditions were met for '{ctx.strategy_name}'."
                ),
                suggestion=(
                    "Review which conditions failed and whether the trade "
                    "should have been taken. Build a pre‑trade checklist."
                ),
                score_impact=impact,
            )

        if dim.components.get("regime_compat", 100) < 5:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=severity,
                title="Strategy–Regime Mismatch",
                description=(
                    f"'{ctx.strategy_name}' was used in an incompatible "
                    f"market regime."
                ),
                suggestion=(
                    "Check regime compatibility before entering. "
                    "Your strategy works best in specific volatility/trend conditions."
                ),
                score_impact=impact,
            )

        return ImprovementInsight(
            insight_id=uuid4(),
            dimension=dim.name,
            severity=severity,
            title="Weak Setup Alignment",
            description=(
                f"Strategy match was {ctx.strategy_match_score:.0f}% "
                f"with override severity of "
                f"{ctx.override_severity or 0:.2f}."
            ),
            suggestion=(
                "Wait for higher‑probability setups where more conditions "
                "align. Patience is an edge."
            ),
            score_impact=impact,
        )

    def _timing_insight(
        self,
        dim: DimensionScore,
        ctx: TradeContext,
        severity: InsightSeverity,
        impact: float,
    ) -> Optional[ImprovementInsight]:
        if dim.components.get("patience", 100) < 8:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=severity,
                title="Impulsive Re‑Entry",
                description=(
                    f"Trade entered {ctx.time_since_last_trade_seconds:.0f}s "
                    f"after previous trade — may indicate impulse."
                ),
                suggestion=(
                    "Implement a mandatory cooldown timer between trades. "
                    "A 2‑minute pause allows emotional reset."
                ),
                score_impact=impact,
            )

        if dim.components.get("fomo_risk", 100) < 5:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=severity,
                title="Possible Chasing / FOMO Entry",
                description=(
                    "Price had already moved significantly before entry, "
                    "suggesting chasing behavior."
                ),
                suggestion=(
                    "Set price alerts at your entry zone BEFORE the move. "
                    "If the move happens without you, wait for a pullback."
                ),
                score_impact=impact,
            )

        return ImprovementInsight(
            insight_id=uuid4(),
            dimension=dim.name,
            severity=severity,
            title="Suboptimal Entry Timing",
            description="Entry timing scored below expectations.",
            suggestion=(
                "Review your entry trigger. Are you entering on confirmation "
                "or anticipation? Align with your strategy's signal."
            ),
            score_impact=impact,
        )

    def _risk_insight(
        self,
        dim: DimensionScore,
        ctx: TradeContext,
        severity: InsightSeverity,
        impact: float,
    ) -> Optional[ImprovementInsight]:
        if dim.components.get("stop_placement", 100) < 3:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=InsightSeverity.CRITICAL,
                title="No Stop Loss Placed",
                description=(
                    "Trade had a planned stop but no actual stop was placed. "
                    "This is the single largest risk management failure."
                ),
                suggestion=(
                    "ALWAYS place your stop before or immediately at entry. "
                    "Consider using bracket orders."
                ),
                score_impact=impact * 1.5,  # amplified
            )

        if dim.components.get("position_sizing", 100) < 10:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=severity,
                title="Position Size Deviation",
                description=(
                    f"Actual size ({ctx.actual_position_size}) deviated "
                    f"significantly from planned ({ctx.planned_position_size})."
                ),
                suggestion=(
                    "Use a position sizing calculator linked to your stop distance. "
                    "Pre‑compute size before entry."
                ),
                score_impact=impact,
            )

        if dim.components.get("daily_limit", 100) < 2:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=InsightSeverity.CRITICAL,
                title="Trading Beyond Daily Loss Limit",
                description=(
                    f"Current daily P&L ({ctx.current_daily_pnl:.2f}) "
                    f"has breached the max daily loss limit "
                    f"({ctx.max_daily_loss_limit:.2f})."
                ),
                suggestion=(
                    "STOP trading immediately. Walk away from the screen. "
                    "This is a hard rule — no exceptions."
                ),
                score_impact=impact * 2.0,
            )

        return ImprovementInsight(
            insight_id=uuid4(),
            dimension=dim.name,
            severity=severity,
            title="Risk Discipline Needs Improvement",
            description=(
                "Multiple risk parameters were outside acceptable bounds."
            ),
            suggestion=(
                "Review your risk management checklist before each trade: "
                "stop placement, position size, portfolio heat, and R:R ratio."
            ),
            score_impact=impact,
        )

    def _emotional_insight(
        self,
        dim: DimensionScore,
        ctx: TradeContext,
        severity: InsightSeverity,
        impact: float,
    ) -> Optional[ImprovementInsight]:
        if dim.components.get("revenge_risk", 100) < 10:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=InsightSeverity.CRITICAL,
                title="Revenge Trade Detected",
                description=(
                    "This trade was entered very quickly after a loss. "
                    "This pattern strongly correlates with further losses."
                ),
                suggestion=(
                    "After a loss, enforce a 5‑minute mandatory pause. "
                    "Re‑run your setup checklist. If you feel urgency to "
                    "'make it back,' that IS the signal to stop."
                ),
                score_impact=impact * 1.3,
            )

        if dim.components.get("overtrading", 100) < 6:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=severity,
                title="Overtrading Pattern",
                description=(
                    f"Trade #{ctx.trade_count_today} today suggests "
                    f"overtrading behavior."
                ),
                suggestion=(
                    "Set a maximum daily trade limit. Quality over quantity. "
                    "Review if all trades today met your setup criteria."
                ),
                score_impact=impact,
            )

        if ctx.emotional_state in (
            EmotionalStateTag.REVENGE,
            EmotionalStateTag.GREEDY,
            EmotionalStateTag.FEARFUL,
        ):
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=severity,
                title=f"Elevated Emotional State: {ctx.emotional_state.value.title()}",
                description=(
                    f"You tagged your state as '{ctx.emotional_state.value}'. "
                    f"Trading in this state significantly reduces decision quality."
                ),
                suggestion=(
                    "Step away, do a breathing exercise, and return only "
                    "when you can operate from a calm/focused state."
                ),
                score_impact=impact,
            )

        return ImprovementInsight(
            insight_id=uuid4(),
            dimension=dim.name,
            severity=severity,
            title="Emotional Stability Below Threshold",
            description="Emotional factors may have influenced this trade.",
            suggestion=(
                "Start tagging your emotional state pre‑trade to build "
                "self‑awareness. Review which emotions precede your worst trades."
            ),
            score_impact=impact,
        )

    def _market_insight(
        self,
        dim: DimensionScore,
        ctx: TradeContext,
        severity: InsightSeverity,
        impact: float,
    ) -> Optional[ImprovementInsight]:
        if dim.components.get("trend_alignment", 100) < 10:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=severity,
                title="Counter‑Trend Trade",
                description=(
                    f"Traded {ctx.side} against a {ctx.trend_direction} trend. "
                    f"Counter‑trend trades have lower win rates."
                ),
                suggestion=(
                    "Unless your strategy specifically targets reversals, "
                    "trade in the direction of the prevailing trend."
                ),
                score_impact=impact,
            )

        if dim.components.get("volatility_fit", 100) < 5:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=severity,
                title="Regime Mismatch",
                description=(
                    f"Current volatility regime '{ctx.volatility_regime}' "
                    f"is suboptimal for this strategy."
                ),
                suggestion=(
                    "Check the regime indicator before entering. "
                    "Some strategies only work in specific environments."
                ),
                score_impact=impact,
            )

        return ImprovementInsight(
            insight_id=uuid4(),
            dimension=dim.name,
            severity=severity,
            title="Market Context Misalignment",
            description="Market conditions were not ideal for this trade.",
            suggestion=(
                "Add a market context check to your pre‑trade routine: "
                "volatility, trend, session, and macro environment."
            ),
            score_impact=impact,
        )

    def _exit_insight(
        self,
        dim: DimensionScore,
        ctx: TradeContext,
        severity: InsightSeverity,
        impact: float,
    ) -> Optional[ImprovementInsight]:
        if dim.components.get("stop_honour", 100) < 12:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=InsightSeverity.CRITICAL,
                title="Stop Not Honoured",
                description=(
                    "Exit price was significantly worse than planned stop. "
                    "Either stop was moved or not placed."
                ),
                suggestion=(
                    "Use hard stops (broker‑held), not mental stops. "
                    "Moving stops wider is the #1 account killer."
                ),
                score_impact=impact * 1.5,
            )

        if dim.components.get("target_adherence", 100) < 10:
            return ImprovementInsight(
                insight_id=uuid4(),
                dimension=dim.name,
                severity=severity,
                title="Premature Exit",
                description=(
                    "Trade was closed well before reaching the planned target, "
                    "leaving significant profit potential unrealized."
                ),
                suggestion=(
                    "Use partial profit‑taking: scale out 50% at first target "
                    "and trail the remainder. This balances fear and greed."
                ),
                score_impact=impact,
            )

        return ImprovementInsight(
            insight_id=uuid4(),
            dimension=dim.name,
            severity=severity,
            title="Exit Discipline Needs Work",
            description="Exit execution deviated from the trade plan.",
            suggestion=(
                "Write your exit plan BEFORE entering. Include: "
                "stop level, target level, trail rules, and time stop."
            ),
            score_impact=impact,
        )

    # ── cross‑dimensional patterns ──────────────────────────────

    def _cross_dimensional_insights(
        self,
        dimensions: Sequence[DimensionScore],
        ctx: TradeContext,
    ) -> List[ImprovementInsight]:
        insights: List[ImprovementInsight] = []
        dim_map = {d.name: d for d in dimensions}

        # Pattern: good setup + poor exit → "gave back edge"
        setup = dim_map.get(DimensionName.SETUP_ALIGNMENT)
        exit_d = dim_map.get(DimensionName.EXIT_DISCIPLINE)
        if setup and exit_d and setup.score > 70 and exit_d.score < 40:
            insights.append(ImprovementInsight(
                insight_id=uuid4(),
                dimension=DimensionName.EXIT_DISCIPLINE,
                severity=InsightSeverity.WARNING,
                title="Good Setup Wasted by Poor Exit",
                description=(
                    "You identified a strong setup but did not execute the "
                    "exit plan. This is a common pattern where good analysis "
                    "is negated by emotional exit decisions."
                ),
                suggestion=(
                    "Focus your improvement on exit mechanics. "
                    "Your setup selection is solid — protect it with disciplined exits."
                ),
                score_impact=15.0,
            ))

        # Pattern: poor emotional + poor risk → "tilt cascade"
        emo = dim_map.get(DimensionName.EMOTIONAL_STABILITY)
        risk = dim_map.get(DimensionName.RISK_DISCIPLINE)
        if emo and risk and emo.score < 40 and risk.score < 40:
            insights.append(ImprovementInsight(
                insight_id=uuid4(),
                dimension=DimensionName.EMOTIONAL_STABILITY,
                severity=InsightSeverity.CRITICAL,
                title="Tilt Cascade Detected",
                description=(
                    "Both emotional stability and risk discipline scored "
                    "critically low. This combination is strongly correlated "
                    "with account drawdowns."
                ),
                suggestion=(
                    "Implement a circuit breaker: after 2 consecutive losses "
                    "or when emotional score drops below threshold, stop trading "
                    "for the session."
                ),
                score_impact=25.0,
            ))

        # Pattern: good market + good setup + losing trade → "execution gap"
        market = dim_map.get(DimensionName.MARKET_CONTEXT)
        if (
            setup and market
            and setup.score > 65 and market.score > 65
            and ctx.outcome == TradeOutcome.LOSS
        ):
            insights.append(ImprovementInsight(
                insight_id=uuid4(),
                dimension=DimensionName.TIMING_QUALITY,
                severity=InsightSeverity.SUGGESTION,
                title="Execution Gap — Good Conditions, Bad Outcome",
                description=(
                    "Setup and market conditions were favorable, but the "
                    "trade lost. This suggests the issue lies in execution "
                    "timing or exit management rather than analysis."
                ),
                suggestion=(
                    "Review the exact entry and exit mechanics. "
                    "Where precisely did execution diverge from the plan?"
                ),
                score_impact=8.0,
            ))

        return insights

    # ── positive reinforcement ──────────────────────────────────

    def _positive_reinforcement(
        self,
        dimensions: Sequence[DimensionScore],
        ctx: TradeContext,
    ) -> List[ImprovementInsight]:
        insights: List[ImprovementInsight] = []

        for dim in dimensions:
            if dim.score >= 85.0:
                insights.append(ImprovementInsight(
                    insight_id=uuid4(),
                    dimension=dim.name,
                    severity=InsightSeverity.POSITIVE,
                    title=f"Excellent {dim.name.value.replace('_', ' ').title()}",
                    description=(
                        f"Scored {dim.score:.0f}/100 — this is textbook execution "
                        f"in the {dim.name.value.replace('_', ' ')} dimension."
                    ),
                    suggestion="Keep doing this. Consistency here is your edge.",
                    score_impact=0.0,
                ))

        # Positive for disciplined losing trade
        if ctx.outcome == TradeOutcome.LOSS:
            avg_score = sum(d.score for d in dimensions) / max(len(dimensions), 1)
            if avg_score >= 70:
                insights.append(ImprovementInsight(
                    insight_id=uuid4(),
                    dimension=DimensionName.RISK_DISCIPLINE,
                    severity=InsightSeverity.POSITIVE,
                    title="Well‑Managed Loss",
                    description=(
                        "This was a losing trade with a high decision quality "
                        f"score ({avg_score:.0f}). Not every good decision makes "
                        f"money — but consistent good decisions compound."
                    ),
                    suggestion=(
                        "This is exactly how professional traders handle losses. "
                        "The P&L on one trade is noise; the process is signal."
                    ),
                    score_impact=0.0,
                ))

        return insights


# ═══════════════════════════════════════════════════════════════════
# EXECUTION CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

class ExecutionClassifier:
    """Maps composite score to human‑readable execution grade."""

    def __init__(self, thresholds: GradeThresholds) -> None:
        self.t = thresholds

    def classify(self, score: float) -> ExecutionGrade:
        if score >= self.t.textbook:
            return ExecutionGrade.TEXTBOOK
        if score >= self.t.good:
            return ExecutionGrade.GOOD
        if score >= self.t.acceptable:
            return ExecutionGrade.ACCEPTABLE
        if score >= self.t.poor:
            return ExecutionGrade.POOR
        return ExecutionGrade.UNDISCIPLINED


# ═══════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════

class DecisionQualityEngine:
    """
    Central engine that orchestrates dimension scoring,
    classification, and insight generation.

    Usage
    -----
    >>> config = DecisionQualityConfig()
    >>> engine = DecisionQualityEngine(config)
    >>> result = engine.evaluate(trade_context)
    >>> print(result.final_score, result.grade)
    """

    def __init__(self, config: Optional[DecisionQualityConfig] = None) -> None:
        self.config = config or DecisionQualityConfig()

        # Initialize scorers — ordered
        self._scorers: Tuple[BaseDimensionScorer, ...] = (
            SetupAlignmentScorer(self.config),
            TimingQualityScorer(self.config),
            RiskDisciplineScorer(self.config),
            EmotionalStabilityScorer(self.config),
            MarketContextScorer(self.config),
            ExitDisciplineScorer(self.config),
        )

        self._classifier = ExecutionClassifier(self.config.grade_thresholds)
        self._insight_gen = InsightGenerator(self.config)

        logger.info(
            "DecisionQualityEngine initialized with %d scorers | "
            "weights=%s",
            len(self._scorers),
            {d.value: w for d, w in self.config.weights.as_dict().items()},
        )

    # ── primary interface ───────────────────────────────────────

    def evaluate(self, ctx: TradeContext) -> DecisionQualityResult:
        """
        Evaluate decision quality for a single trade.

        Parameters
        ----------
        ctx : TradeContext
            Assembled context for the trade.

        Returns
        -------
        DecisionQualityResult
            Complete scored evaluation.
        """
        evaluation_id = uuid4()
        evaluated_at = datetime.utcnow()

        # 1. Score each dimension
        dimensions: List[DimensionScore] = []
        for scorer in self._scorers:
            try:
                dim_score = scorer.score(ctx)
                dimensions.append(dim_score)
            except Exception:
                logger.exception(
                    "Scorer %s failed for trade %s — assigning neutral.",
                    scorer.dimension.value,
                    ctx.trade_id,
                )
                dimensions.append(DimensionScore(
                    name=scorer.dimension,
                    score=50.0,
                    weight=self.config.weights.as_dict()[scorer.dimension],
                    confidence=0.0,
                    components={"error": 50.0},
                    reasoning=("Scorer failed — neutral score assigned.",),
                ))

        # 2. Compute composite score (confidence‑weighted)
        final_score = self._compute_composite(dimensions)

        # 3. Classify
        grade = self._classifier.classify(final_score)

        # 4. Data completeness
        data_completeness = self._compute_data_completeness(dimensions)

        # 5. Generate insights
        insights = self._insight_gen.generate(dimensions, ctx)

        result = DecisionQualityResult(
            trade_id=ctx.trade_id,
            evaluation_id=evaluation_id,
            evaluated_at=evaluated_at,
            final_score=round(final_score, 1),
            grade=grade,
            dimensions=tuple(dimensions),
            insights=tuple(insights),
            outcome_independent=True,
            data_completeness=data_completeness,
        )

        logger.info(
            "Trade %s evaluated: score=%.1f grade=%s completeness=%.0f%% "
            "insights=%d (critical=%d)",
            ctx.trade_id,
            result.final_score,
            result.grade.value,
            result.data_completeness * 100,
            len(result.insights),
            len(result.critical_insights),
        )

        return result

    # ── batch interface ─────────────────────────────────────────

    def evaluate_batch(
        self, contexts: Sequence[TradeContext]
    ) -> List[DecisionQualityResult]:
        """Evaluate multiple trades. Returns results in same order."""
        return [self.evaluate(ctx) for ctx in contexts]

    # ── internal ────────────────────────────────────────────────

    def _compute_composite(self, dimensions: Sequence[DimensionScore]) -> float:
        """
        Confidence‑weighted composite score.

        When a dimension has low confidence (sparse data), its
        weight is redistributed proportionally to high‑confidence
        dimensions so total always sums to 100.
        """
        total_effective_weight = 0.0
        weighted_sum = 0.0

        for d in dimensions:
            effective_weight = d.weight * d.confidence
            weighted_sum += d.score * effective_weight
            total_effective_weight += effective_weight

        if total_effective_weight <= 0:
            return 50.0  # complete data absence

        return weighted_sum / total_effective_weight

    @staticmethod
    def _compute_data_completeness(dimensions: Sequence[DimensionScore]) -> float:
        """Average confidence across all dimensions."""
        if not dimensions:
            return 0.0
        return sum(d.confidence for d in dimensions) / len(dimensions)


# ═══════════════════════════════════════════════════════════════════
# EVENT BUS INTEGRATION HELPER
# ═══════════════════════════════════════════════════════════════════

class DecisionQualityEventHandler:
    """
    Subscribes to trade completion events and triggers evaluation.

    Integration
    -----------
    >>> from core.events import event_bus
    >>> handler = DecisionQualityEventHandler(engine, event_bus)
    >>> handler.register()
    """

    def __init__(
        self,
        engine: DecisionQualityEngine,
        event_bus: Any,
        context_assembler: Optional[Any] = None,
    ) -> None:
        self.engine = engine
        self.event_bus = event_bus
        self.context_assembler = context_assembler

    def register(self) -> None:
        """Register event listeners."""
        self.event_bus.subscribe("trade.completed", self._on_trade_completed)
        self.event_bus.subscribe("trade.updated", self._on_trade_updated)
        logger.info("DecisionQualityEventHandler registered on event bus.")

    async def _on_trade_completed(self, event: Dict[str, Any]) -> None:
        """Handle trade completion event."""
        trade_id = event.get("trade_id")
        if trade_id is None:
            logger.warning("trade.completed event missing trade_id.")
            return

        try:
            if self.context_assembler:
                ctx = await self.context_assembler.assemble(trade_id)
            else:
                logger.warning(
                    "No context assembler configured — cannot evaluate trade %s.",
                    trade_id,
                )
                return

            result = self.engine.evaluate(ctx)

            # Emit evaluation event for downstream consumers
            await self.event_bus.publish("decision_quality.evaluated", {
                "trade_id": str(trade_id),
                "evaluation_id": str(result.evaluation_id),
                "final_score": result.final_score,
                "grade": result.grade.value,
                "critical_insights": len(result.critical_insights),
                "data_completeness": result.data_completeness,
            })

        except Exception:
            logger.exception(
                "Failed to evaluate decision quality for trade %s.", trade_id
            )

    async def _on_trade_updated(self, event: Dict[str, Any]) -> None:
        """Re‑evaluate when trade data is updated (e.g., exit added)."""
        if event.get("fields_updated") and "exit_price" in event["fields_updated"]:
            await self._on_trade_completed(event)


# ═══════════════════════════════════════════════════════════════════
# PERSISTENCE HELPER
# ═══════════════════════════════════════════════════════════════════

class DecisionQualityPersistence:
    """
    Persists DecisionQualityResult to the database.

    Designed to work with the existing SQLAlchemy async session
    from core.database.
    """

    def __init__(self, session_factory: Any) -> None:
        self._session_factory = session_factory

    async def save(self, result: DecisionQualityResult) -> None:
        """
        Save evaluation result to decision_quality_evaluations
        and decision_quality_dimensions tables.
        """
        from sqlalchemy import text

        async with self._session_factory() as session:
            # Main evaluation record
            await session.execute(
                text("""
                    INSERT INTO decision_quality_evaluations (
                        evaluation_id, trade_id, evaluated_at,
                        final_score, grade, data_completeness,
                        outcome_independent, raw_result
                    ) VALUES (
                        :evaluation_id, :trade_id, :evaluated_at,
                        :final_score, :grade, :data_completeness,
                        :outcome_independent, :raw_result
                    )
                    ON CONFLICT (evaluation_id) DO UPDATE SET
                        final_score = EXCLUDED.final_score,
                        grade = EXCLUDED.grade,
                        data_completeness = EXCLUDED.data_completeness,
                        raw_result = EXCLUDED.raw_result
                """),
                {
                    "evaluation_id": str(result.evaluation_id),
                    "trade_id": str(result.trade_id) if result.trade_id else None,
                    "evaluated_at": result.evaluated_at,
                    "final_score": result.final_score,
                    "grade": result.grade.value,
                    "data_completeness": result.data_completeness,
                    "outcome_independent": result.outcome_independent,
                    "raw_result": str(result.to_dict()),
                },
            )

            # Dimension scores
            for dim in result.dimensions:
                await session.execute(
                    text("""
                        INSERT INTO decision_quality_dimensions (
                            evaluation_id, dimension_name,
                            score, weight, confidence, components, reasoning
                        ) VALUES (
                            :evaluation_id, :dimension_name,
                            :score, :weight, :confidence,
                            :components, :reasoning
                        )
                        ON CONFLICT (evaluation_id, dimension_name) DO UPDATE SET
                            score = EXCLUDED.score,
                            confidence = EXCLUDED.confidence,
                            components = EXCLUDED.components,
                            reasoning = EXCLUDED.reasoning
                    """),
                    {
                        "evaluation_id": str(result.evaluation_id),
                        "dimension_name": dim.name.value,
                        "score": dim.score,
                        "weight": dim.weight,
                        "confidence": dim.confidence,
                        "components": str(dim.components),
                        "reasoning": str(list(dim.reasoning)),
                    },
                )

            # Insights
            for insight in result.insights:
                await session.execute(
                    text("""
                        INSERT INTO decision_quality_insights (
                            insight_id, evaluation_id, dimension_name,
                            severity, title, description,
                            suggestion, score_impact
                        ) VALUES (
                            :insight_id, :evaluation_id, :dimension_name,
                            :severity, :title, :description,
                            :suggestion, :score_impact
                        )
                        ON CONFLICT (insight_id) DO NOTHING
                    """),
                    {
                        "insight_id": str(insight.insight_id),
                        "evaluation_id": str(result.evaluation_id),
                        "dimension_name": insight.dimension.value,
                        "severity": insight.severity.value,
                        "title": insight.title,
                        "description": insight.description,
                        "suggestion": insight.suggestion,
                        "score_impact": insight.score_impact,
                    },
                )

            await session.commit()

        logger.debug(
            "Persisted evaluation %s for trade %s (%d dimensions, %d insights).",
            result.evaluation_id,
            result.trade_id,
            len(result.dimensions),
            len(result.insights),
        )

    async def load(self, trade_id: UUID) -> Optional[Dict[str, Any]]:
        """Load the latest evaluation for a trade."""
        from sqlalchemy import text

        async with self._session_factory() as session:
            row = await session.execute(
                text("""
                    SELECT raw_result FROM decision_quality_evaluations
                    WHERE trade_id = :trade_id
                    ORDER BY evaluated_at DESC
                    LIMIT 1
                """),
                {"trade_id": str(trade_id)},
            )
            result = row.fetchone()
            if result:
                import ast
                try:
                    return ast.literal_eval(result[0])
                except (ValueError, SyntaxError):
                    return None
        return None

    async def load_recent(
        self, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Load recent evaluations for analytics."""
        from sqlalchemy import text

        async with self._session_factory() as session:
            rows = await session.execute(
                text("""
                    SELECT trade_id, final_score, grade,
                           data_completeness, evaluated_at
                    FROM decision_quality_evaluations
                    ORDER BY evaluated_at DESC
                    LIMIT :limit
                """),
                {"limit": limit},
            )
            return [
                {
                    "trade_id": r[0],
                    "final_score": r[1],
                    "grade": r[2],
                    "data_completeness": r[3],
                    "evaluated_at": r[4].isoformat() if r[4] else None,
                }
                for r in rows.fetchall()
            ]


# ═══════════════════════════════════════════════════════════════════
# ANALYTICS AGGREGATOR
# ═══════════════════════════════════════════════════════════════════

class DecisionQualityAnalytics:
    """
    Aggregates decision quality evaluations for trend analysis,
    coaching summaries, and dashboard display.
    """

    @staticmethod
    def compute_averages(
        results: Sequence[DecisionQualityResult],
    ) -> Dict[str, float]:
        """Average scores across a set of evaluations."""
        if not results:
            return {
                "overall": 0.0,
                **{d.value: 0.0 for d in DimensionName},
            }

        total_score = sum(r.final_score for r in results)
        dim_totals: Dict[DimensionName, float] = {d: 0.0 for d in DimensionName}
        dim_counts: Dict[DimensionName, int] = {d: 0 for d in DimensionName}

        for r in results:
            for d in r.dimensions:
                dim_totals[d.name] += d.score
                dim_counts[d.name] += 1

        n = len(results)
        averages: Dict[str, float] = {"overall": round(total_score / n, 1)}
        for dim_name in DimensionName:
            count = dim_counts[dim_name]
            averages[dim_name.value] = (
                round(dim_totals[dim_name] / count, 1) if count > 0 else 0.0
            )

        return averages

    @staticmethod
    def grade_distribution(
        results: Sequence[DecisionQualityResult],
    ) -> Dict[str, int]:
        """Count evaluations per grade."""
        dist = {g.value: 0 for g in ExecutionGrade}
        for r in results:
            dist[r.grade.value] += 1
        return dist

    @staticmethod
    def weakest_dimensions(
        results: Sequence[DecisionQualityResult],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """Identify consistently weakest dimensions for coaching focus."""
        dim_scores: Dict[DimensionName, List[float]] = {
            d: [] for d in DimensionName
        }
        for r in results:
            for d in r.dimensions:
                dim_scores[d.name].append(d.score)

        averages = []
        for name, scores in dim_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                averages.append({
                    "dimension": name.value,
                    "average_score": round(avg, 1),
                    "sample_size": len(scores),
                    "min_score": round(min(scores), 1),
                    "below_50_pct": round(
                        sum(1 for s in scores if s < 50) / len(scores) * 100, 1
                    ),
                })

        averages.sort(key=lambda x: x["average_score"])
        return averages[:top_n]

    @staticmethod
    def score_trend(
        results: Sequence[DecisionQualityResult],
        window: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Rolling average of decision quality scores.
        Results should be in chronological order.
        """
        if len(results) < window:
            return [
                {
                    "index": i,
                    "trade_id": str(r.trade_id),
                    "score": r.final_score,
                    "rolling_avg": None,
                    "evaluated_at": r.evaluated_at.isoformat(),
                }
                for i, r in enumerate(results)
            ]

        trend: List[Dict[str, Any]] = []
        for i, r in enumerate(results):
            if i < window - 1:
                rolling = None
            else:
                window_scores = [
                    results[j].final_score for j in range(i - window + 1, i + 1)
                ]
                rolling = round(sum(window_scores) / window, 1)

            trend.append({
                "index": i,
                "trade_id": str(r.trade_id),
                "score": r.final_score,
                "rolling_avg": rolling,
                "evaluated_at": r.evaluated_at.isoformat(),
            })

        return trend

    @staticmethod
    def outcome_vs_quality_matrix(
        results: Sequence[DecisionQualityResult],
        contexts: Sequence[TradeContext],
    ) -> Dict[str, Dict[str, int]]:
        """
        Cross‑tabulate decision quality grade vs trade outcome.

        This is the KEY insight: separating process from outcome.
        A "good decision + loss" is expected; a "bad decision + win"
        is dangerous (reinforces bad habits).

        Returns
        -------
        Dict mapping grade → {win: N, loss: N, breakeven: N}
        """
        matrix: Dict[str, Dict[str, int]] = {
            g.value: {"win": 0, "loss": 0, "breakeven": 0}
            for g in ExecutionGrade
        }

        for result, ctx in zip(results, contexts):
            outcome_key = ctx.outcome.value
            matrix[result.grade.value][outcome_key] += 1

        return matrix

    @staticmethod
    def insight_frequency(
        results: Sequence[DecisionQualityResult],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Most frequently occurring insight titles across evaluations.
        Useful for identifying systemic issues.
        """
        from collections import Counter

        title_counter: Counter[str] = Counter()
        title_severity: Dict[str, InsightSeverity] = {}
        title_dimension: Dict[str, DimensionName] = {}

        for r in results:
            for insight in r.insights:
                if insight.severity == InsightSeverity.POSITIVE:
                    continue  # exclude positives from "problem" frequency
                title_counter[insight.title] += 1
                title_severity[insight.title] = insight.severity
                title_dimension[insight.title] = insight.dimension

        return [
            {
                "title": title,
                "count": count,
                "frequency_pct": round(count / max(len(results), 1) * 100, 1),
                "severity": title_severity[title].value,
                "dimension": title_dimension[title].value,
            }
            for title, count in title_counter.most_common(top_n)
        ]

    @staticmethod
    def improvement_over_time(
        results: Sequence[DecisionQualityResult],
        period_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Compare average scores across successive periods.
        Helps detect whether coaching is producing improvement.
        """
        if not results:
            return []

        periods: List[Dict[str, Any]] = []
        for i in range(0, len(results), period_size):
            chunk = results[i: i + period_size]
            if not chunk:
                break

            dim_avgs: Dict[str, float] = {}
            for dim_name in DimensionName:
                scores = []
                for r in chunk:
                    d = r.dimension(dim_name)
                    if d:
                        scores.append(d.score)
                dim_avgs[dim_name.value] = (
                    round(sum(scores) / len(scores), 1) if scores else 0.0
                )

            periods.append({
                "period_index": i // period_size,
                "trade_count": len(chunk),
                "avg_score": round(
                    sum(r.final_score for r in chunk) / len(chunk), 1
                ),
                "grade_distribution": {
                    g.value: sum(1 for r in chunk if r.grade == g)
                    for g in ExecutionGrade
                },
                "dimension_averages": dim_avgs,
                "start_date": chunk[0].evaluated_at.isoformat(),
                "end_date": chunk[-1].evaluated_at.isoformat(),
            })

        return periods

    @staticmethod
    def coaching_summary(
        results: Sequence[DecisionQualityResult],
    ) -> Dict[str, Any]:
        """
        Produce a summary suitable for the coaching agent's context.
        """
        if not results:
            return {
                "trade_count": 0,
                "avg_score": 0.0,
                "grade_distribution": {},
                "weakest_dimensions": [],
                "top_insights": [],
                "trend_direction": "insufficient_data",
            }

        avg = sum(r.final_score for r in results) / len(results)
        weakest = DecisionQualityAnalytics.weakest_dimensions(results)
        top_insights = DecisionQualityAnalytics.insight_frequency(results, top_n=5)
        grade_dist = DecisionQualityAnalytics.grade_distribution(results)

        # Simple trend: compare first half vs second half
        mid = len(results) // 2
        if mid > 0:
            first_half_avg = sum(r.final_score for r in results[:mid]) / mid
            second_half_avg = sum(
                r.final_score for r in results[mid:]
            ) / (len(results) - mid)
            delta = second_half_avg - first_half_avg
            if delta > 3.0:
                trend = "improving"
            elif delta < -3.0:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "trade_count": len(results),
            "avg_score": round(avg, 1),
            "grade_distribution": grade_dist,
            "weakest_dimensions": weakest,
            "top_insights": top_insights,
            "trend_direction": trend,
        }


# ═══════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONVENIENCE FUNCTIONS (used by api/routes/analytics.py)
# ═══════════════════════════════════════════════════════════════════

def score_trade_quality(
    trade,
    attribution=None,
    shadow=None,
    discipline=None,
    behavioral_events=None,
    checklist_results=None,
) -> "DecisionQualityResult":
    """
    Convenience wrapper: build a TradeContext from ORM objects and evaluate.
    """
    ctx = TradeContext(
        trade_id=getattr(trade, "id", None),
        symbol=getattr(trade, "symbol", None),
        side=getattr(trade, "direction", None),
        entry_price=getattr(trade, "entry_price", None),
        exit_price=getattr(trade, "exit_price", None),
        entry_time=getattr(trade, "entry_time", None),
        exit_time=getattr(trade, "exit_time", None),
        quantity=getattr(trade, "position_size", None),
        pnl=getattr(trade, "pnl", None),
        r_multiple=getattr(trade, "r_multiple", None),
        strategy_name=getattr(trade, "declared_strategy_id", None),
        stop_price=getattr(trade, "stop_loss", None),
        target_price=getattr(trade, "take_profit", None),
        account_balance=getattr(trade, "account_balance_at_entry", None),
        risk_amount=getattr(trade, "risk_amount", None),
        emotional_state=getattr(trade, "emotion_pre", None),
        session_type=getattr(trade, "session_type", None),
        checklist_results=checklist_results or [],
        behavioral_flags=behavioral_events or [],
    )
    engine = DecisionQualityEngine()
    return engine.evaluate(ctx)


def compute_decision_quality_trend(results, window: int = 10):
    """Rolling trend of decision quality scores."""
    if not results:
        return []
    engine = DecisionQualityEngine()
    return engine.score_trend(results, window=window)