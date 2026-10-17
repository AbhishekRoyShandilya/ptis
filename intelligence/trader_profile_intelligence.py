"""
Trader Profile Intelligence

Deep behavioral and style modeling of the trader. Goes beyond basic
profile settings to learn and model:

- Risk tolerance & sizing behavior patterns
- Timing preferences (when they trade best)
- Confirmation vs anticipation trading style
- Discipline patterns (when discipline breaks down)
- Emotional triggers (what situations cause problems)
- Cognitive style (analytical vs intuitive, patient vs aggressive)
- Performance context (conditions where they excel/struggle)

This profile is consumed by the discretionary query engine, coaching
agent, and alert system to personalize all intelligence output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
from sqlalchemy import select, and_, func, extract, cast, String, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.models import Trade, BehavioralEvent, DisciplineRecord, DecisionQualityRecord
from core.enums import TradeStatus

logger = logging.getLogger(__name__)


class TradingStyle(str):
    SCALPER = "scalper"
    DAY_TRADER = "day_trader"
    SWING_TRADER = "swing_trader"
    POSITION_TRADER = "position_trader"


class EntryStyle(str):
    ANTICIPATION = "anticipation"  # Enters before confirmation
    CONFIRMATION = "confirmation"  # Waits for full confirmation
    REACTIVE = "reactive"          # Chases after moves start
    MIXED = "mixed"


class RiskPersonality(str):
    CONSERVATIVE = "conservative"  # Small positions, tight stops
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"      # Larger positions, wider tolerance
    INCONSISTENT = "inconsistent"  # Varies significantly


class DisciplinePattern(str):
    CONSISTENT = "consistent"
    MORNING_STRONG = "morning_strong"     # Disciplined early, breaks down later
    LOSS_TRIGGERED = "loss_triggered"     # Discipline collapses after losses
    STREAK_DEPENDENT = "streak_dependent" # Overconfident on wins, scared on losses
    FATIGUE_DRIVEN = "fatigue_driven"     # Decays with number of trades


@dataclass
class TimingProfile:
    """When does this trader perform best?"""
    best_hours: list[int] = field(default_factory=list)       # Hours (UTC)
    worst_hours: list[int] = field(default_factory=list)
    best_day_of_week: list[int] = field(default_factory=list) # 0=Mon, 4=Fri
    avg_hold_time_minutes: float = 0.0
    preferred_session: str = "regular"  # "pre_market", "regular", "after_hours"
    optimal_trade_count: int = 3        # Trades per day before quality drops
    quality_decay_after_n_trades: Optional[int] = None


@dataclass
class RiskProfile:
    """How does this trader manage risk?"""
    personality: str = RiskPersonality.MODERATE
    avg_risk_per_trade: float = 0.01        # % of account
    risk_consistency: float = 0.5           # 0-1, how consistent
    avg_stop_distance_atr: float = 1.5      # Stop in ATR multiples
    stop_adherence_rate: float = 0.80       # How often stop is respected
    avg_rr_planned: float = 2.0
    avg_rr_actual: float = 1.5
    position_size_consistency: float = 0.5  # 0-1
    max_observed_risk: float = 0.05
    tends_to_widen_stops: bool = False
    tends_to_move_targets: bool = False


@dataclass
class EmotionalProfile:
    """Emotional patterns and triggers."""
    primary_triggers: list[str] = field(default_factory=list)
    # ^ e.g., ["consecutive_losses", "large_win", "missing_move", "slow_market"]

    post_loss_behavior: str = ""
    # ^ "reduces_size", "increases_size", "stops_trading", "revenge_trades"

    post_win_behavior: str = ""
    # ^ "stays_disciplined", "becomes_overconfident", "takes_break"

    tilt_frequency: float = 0.0  # How often tilt is detected (per week)
    avg_trades_before_tilt: Optional[int] = None
    recovery_time_minutes: float = 0.0  # How long to recover from tilt
    emotional_awareness_score: float = 0.5  # 0-1, self-awareness


@dataclass
class StrategyPreference:
    """Which strategies the trader uses and how well."""
    strategy_name: str
    frequency: float = 0.0        # % of total trades
    win_rate: float = 0.0
    avg_r: float = 0.0
    is_core: bool = False         # Primary strategy
    best_regime: Optional[str] = None
    worst_regime: Optional[str] = None
    execution_quality: float = 0.5  # 0-1


@dataclass
class TraderIntelligenceProfile:
    """
    Complete intelligence profile for a trader.

    This is the system's deep model of who this trader is,
    how they behave, and what they need.
    """
    trader_id: str
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Style classification
    trading_style: str = TradingStyle.DAY_TRADER
    entry_style: str = EntryStyle.MIXED
    cognitive_type: str = "analytical"  # "analytical", "intuitive", "systematic"

    # Risk
    risk: RiskProfile = field(default_factory=RiskProfile)

    # Timing
    timing: TimingProfile = field(default_factory=TimingProfile)

    # Emotional
    emotional: EmotionalProfile = field(default_factory=EmotionalProfile)

    # Discipline
    discipline_pattern: str = DisciplinePattern.CONSISTENT
    current_discipline_score: float = 0.70
    discipline_trend: str = "stable"  # "improving", "declining", "stable"

    # Strategy preferences
    strategy_preferences: list[StrategyPreference] = field(default_factory=list)
    core_strategies: list[str] = field(default_factory=list)

    # Performance context
    best_conditions: list[str] = field(default_factory=list)
    # ^ e.g., ["trending_bullish + morning", "breakout + high_volume"]
    worst_conditions: list[str] = field(default_factory=list)
    # ^ e.g., ["ranging + afternoon", "low_vol + many_trades"]

    # Decision quality
    avg_dq_score: float = 0.0
    weakest_dq_dimension: Optional[str] = None
    strongest_dq_dimension: Optional[str] = None

    # Growth areas
    primary_growth_area: Optional[str] = None
    improvement_velocity: float = 0.0  # Rate of improvement

    # Meta
    sample_size: int = 0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trader_id": self.trader_id,
            "generated_at": self.generated_at.isoformat(),
            "trading_style": self.trading_style,
            "entry_style": self.entry_style,
            "cognitive_type": self.cognitive_type,
            "risk": {
                "personality": self.risk.personality,
                "avg_risk_per_trade": self.risk.avg_risk_per_trade,
                "risk_consistency": round(self.risk.risk_consistency, 3),
                "stop_adherence_rate": round(self.risk.stop_adherence_rate, 3),
                "avg_rr_planned": self.risk.avg_rr_planned,
                "avg_rr_actual": round(self.risk.avg_rr_actual, 3),
                "tends_to_widen_stops": self.risk.tends_to_widen_stops,
            },
            "timing": {
                "best_hours": self.timing.best_hours,
                "worst_hours": self.timing.worst_hours,
                "avg_hold_time_minutes": round(self.timing.avg_hold_time_minutes, 1),
                "optimal_trade_count": self.timing.optimal_trade_count,
                "quality_decay_after_n_trades": self.timing.quality_decay_after_n_trades,
            },
            "emotional": {
                "primary_triggers": self.emotional.primary_triggers,
                "post_loss_behavior": self.emotional.post_loss_behavior,
                "post_win_behavior": self.emotional.post_win_behavior,
                "tilt_frequency": round(self.emotional.tilt_frequency, 2),
                "emotional_awareness_score": round(self.emotional.emotional_awareness_score, 3),
            },
            "discipline_pattern": self.discipline_pattern,
            "current_discipline_score": round(self.current_discipline_score, 3),
            "discipline_trend": self.discipline_trend,
            "core_strategies": self.core_strategies,
            "best_conditions": self.best_conditions,
            "worst_conditions": self.worst_conditions,
            "avg_dq_score": round(self.avg_dq_score, 3),
            "weakest_dq_dimension": self.weakest_dq_dimension,
            "strongest_dq_dimension": self.strongest_dq_dimension,
            "primary_growth_area": self.primary_growth_area,
            "sample_size": self.sample_size,
            "confidence": round(self.confidence, 3),
        }


class TraderProfileBuilder:
    """
    Builds the TraderIntelligenceProfile from historical data.

    This is computationally expensive — run periodically (daily)
    and cache the result. The profile is updated incrementally
    after each trade.
    """

    def __init__(self, db: AsyncSession, lookback_days: int = 90):
        self.db = db
        self.lookback_days = lookback_days

    async def build(self, trader_id: str) -> TraderIntelligenceProfile:
        """Build complete trader intelligence profile."""
        cutoff = datetime.utcnow() - timedelta(days=self.lookback_days)

        # Each fetch is wrapped in a savepoint so a failed query (e.g. missing table)
        # does NOT abort the outer transaction (asyncpg marks the whole tx as aborted
        # on any error; a savepoint ROLLBACK resets only that sub-transaction).
        async def _safe_fetch(coro):
            try:
                async with self.db.begin_nested():
                    return await coro
            except Exception as _e:
                logger.debug("Profile fetch skipped: %s", _e)
                return []

        trades = await _safe_fetch(self._fetch_trades(trader_id, cutoff))
        behavioral_events = await _safe_fetch(self._fetch_behavioral_events(trader_id, cutoff))
        discipline_records = await _safe_fetch(self._fetch_discipline_records(trader_id, cutoff))
        dq_records = await _safe_fetch(self._fetch_dq_records(trader_id, cutoff))

        if not trades:
            return TraderIntelligenceProfile(
                trader_id=trader_id,
                confidence=0.0,
                sample_size=0,
            )

        profile = TraderIntelligenceProfile(
            trader_id=trader_id,
            sample_size=len(trades),
        )

        # Build each component
        profile.trading_style = self._classify_trading_style(trades)
        profile.entry_style = self._classify_entry_style(trades)
        profile.cognitive_type = self._classify_cognitive_type(
            trades, behavioral_events
        )
        profile.risk = self._build_risk_profile(trades)
        profile.timing = self._build_timing_profile(trades, dq_records)
        profile.emotional = self._build_emotional_profile(
            trades, behavioral_events
        )
        profile.discipline_pattern = self._classify_discipline_pattern(
            trades, discipline_records, dq_records
        )

        # Discipline current state
        if discipline_records:
            recent = discipline_records[:5]
            profile.current_discipline_score = float(
                np.mean([r.score for r in recent])
            )
            if len(discipline_records) >= 10:
                first_half = np.mean([r.score for r in discipline_records[5:10]])
                second_half = np.mean([r.score for r in discipline_records[:5]])
                delta = second_half - first_half
                if delta > 0.05:
                    profile.discipline_trend = "improving"
                elif delta < -0.05:
                    profile.discipline_trend = "declining"
                else:
                    profile.discipline_trend = "stable"

        # Strategy preferences
        profile.strategy_preferences = self._build_strategy_preferences(trades)
        profile.core_strategies = [
            sp.strategy_name for sp in profile.strategy_preferences
            if sp.is_core
        ]

        # Performance context
        profile.best_conditions, profile.worst_conditions = (
            self._analyze_performance_context(trades)
        )

        # DQ analysis
        if dq_records:
            scores = [r.composite_score for r in dq_records if r.composite_score]
            if scores:
                profile.avg_dq_score = float(np.mean(scores))

            # Weakest/strongest dimensions
            dim_scores: dict[str, list[float]] = {}
            for r in dq_records:
                details = r.dimension_details or {}
                for dim_name, dim_data in details.items():
                    if isinstance(dim_data, dict) and "score" in dim_data:
                        if dim_name not in dim_scores:
                            dim_scores[dim_name] = []
                        dim_scores[dim_name].append(dim_data["score"])

            if dim_scores:
                dim_avgs = {
                    k: np.mean(v) for k, v in dim_scores.items()
                }
                profile.weakest_dq_dimension = min(dim_avgs, key=dim_avgs.get)
                profile.strongest_dq_dimension = max(dim_avgs, key=dim_avgs.get)
                profile.primary_growth_area = profile.weakest_dq_dimension

        # Confidence based on sample size
        if len(trades) >= 200:
            profile.confidence = 0.90
        elif len(trades) >= 100:
            profile.confidence = 0.75
        elif len(trades) >= 50:
            profile.confidence = 0.60
        elif len(trades) >= 20:
            profile.confidence = 0.40
        else:
            profile.confidence = 0.20

        return profile

    # ─── Data Fetchers ───────────────────────────────────────────

    async def _fetch_trades(self, trader_id: str, cutoff: datetime) -> list:
        # Use raw SQL to avoid asyncpg enum type resolution issues with tradestatus
        from types import SimpleNamespace
        import uuid as _uuid
        try:
            tid_uuid = _uuid.UUID(str(trader_id)) if not isinstance(trader_id, _uuid.UUID) else trader_id
        except ValueError:
            return []  # Non-UUID trader_id (e.g. "default-trader") → no trade history
        result = await self.db.execute(
            text("""
                SELECT id, symbol, direction, status::text as status,
                       entry_price, exit_price, entry_time, exit_time,
                       position_size, stop_loss, pnl, r_multiple, risk_pct,
                       market_regime, declared_strategy_id,
                       emotion_pre, conviction_score, notes, trader_id
                FROM trades
                WHERE trader_id = :tid
                  AND status::text = 'closed'
                  AND exit_time >= :cutoff
                ORDER BY exit_time DESC
            """).bindparams(tid=tid_uuid, cutoff=cutoff),
        )
        rows = result.mappings().all()
        # Wrap in SimpleNamespace so downstream getattr() calls work
        return [SimpleNamespace(**dict(r)) for r in rows]

    async def _fetch_behavioral_events(
        self, trader_id: str, cutoff: datetime
    ) -> list:
        try:
            import uuid as _uuid
            try:
                tid = _uuid.UUID(str(trader_id)) if not isinstance(trader_id, _uuid.UUID) else trader_id
            except ValueError:
                return []
            trade_ids_subq = select(Trade.id).where(Trade.trader_id == tid).scalar_subquery()
            stmt = (
                select(BehavioralEvent)
                .where(
                    and_(
                        BehavioralEvent.trade_id.in_(trade_ids_subq),
                        BehavioralEvent.session_date >= cutoff,
                    )
                )
                .order_by(BehavioralEvent.session_date.desc())
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.debug("Could not fetch behavioral events: %s", e)
            return []

    async def _fetch_discipline_records(
        self, trader_id: str, cutoff: datetime
    ) -> list:
        try:
            stmt = (
                select(DisciplineRecord)
                .where(
                    and_(
                        DisciplineRecord.trader_id == str(trader_id),
                        DisciplineRecord.created_at >= cutoff,
                    )
                )
                .order_by(DisciplineRecord.created_at.desc())
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.debug("Could not fetch discipline records: %s", e)
            return []

    async def _fetch_dq_records(
        self, trader_id: str, cutoff: datetime
    ) -> list:
        try:
          stmt = (
            select(DecisionQualityRecord)
            .where(
                and_(
                    DecisionQualityRecord.trader_id == str(trader_id),
                    DecisionQualityRecord.created_at >= cutoff,
                )
            )
            .order_by(DecisionQualityRecord.created_at.desc())
          )
          result = await self.db.execute(stmt)
          return list(result.scalars().all())
        except Exception as e:
            logger.debug("Could not fetch DQ records: %s", e)
            return []

    # ─── Classifiers ─────────────────────────────────────────────

    def _classify_trading_style(self, trades: list) -> str:
        """Classify based on average hold time."""
        hold_times = []
        for t in trades:
            entry = getattr(t, "entry_time", None)
            exit_t = getattr(t, "exit_time", None)
            if entry and exit_t:
                delta = (exit_t - entry).total_seconds() / 60.0
                hold_times.append(delta)

        if not hold_times:
            return TradingStyle.DAY_TRADER

        avg_hold = np.median(hold_times)
        if avg_hold < 5:
            return TradingStyle.SCALPER
        elif avg_hold < 240:  # 4 hours
            return TradingStyle.DAY_TRADER
        elif avg_hold < 7200:  # 5 days
            return TradingStyle.SWING_TRADER
        else:
            return TradingStyle.POSITION_TRADER

    def _classify_entry_style(self, trades: list) -> str:
        """
        Classify entry style by analyzing entry timing relative to signals.

        Anticipation: enters before full confirmation (early entries)
        Confirmation: waits for all conditions (precise entries)
        Reactive: chases after moves already started (late entries)
        """
        entry_qualities = []
        for t in trades:
            meta = getattr(t, "metadata", {}) or {}
            if meta.get("chased_entry"):
                entry_qualities.append("reactive")
            elif meta.get("early_entry"):
                entry_qualities.append("anticipation")
            else:
                entry_qualities.append("confirmation")

        if not entry_qualities:
            return EntryStyle.MIXED

        from collections import Counter
        counts = Counter(entry_qualities)
        total = len(entry_qualities)

        dominant = counts.most_common(1)[0]
        if dominant[1] / total >= 0.60:
            return dominant[0]
        return EntryStyle.MIXED

    def _classify_cognitive_type(
        self, trades: list, events: list
    ) -> str:
        """
        Classify cognitive/thinking style.

        Analytical: uses checklist, follows rules, data-driven
        Intuitive: relies on feel, pattern recognition, discretionary
        Systematic: highly rule-based, mechanical execution
        """
        checklist_usage = 0
        discretionary_count = 0
        for t in trades:
            meta = getattr(t, "metadata", {}) or {}
            if meta.get("pre_trade_journal") or meta.get("checklist_completed"):
                checklist_usage += 1
            if meta.get("discretionary_override") or meta.get("intuition_entry"):
                discretionary_count += 1

        total = len(trades) if trades else 1
        checklist_rate = checklist_usage / total
        discretionary_rate = discretionary_count / total

        if checklist_rate > 0.80 and discretionary_rate < 0.10:
            return "systematic"
        elif discretionary_rate > 0.40:
            return "intuitive"
        else:
            return "analytical"

    def _build_risk_profile(self, trades: list) -> RiskProfile:
        """Build risk profile from trade history."""
        risk_pcts = []
        rr_planned = []
        rr_actual = []
        stop_distances = []
        stop_respected = 0
        stop_widened = 0
        target_moved = 0

        for t in trades:
            rp = getattr(t, "risk_percent", None)
            if rp is not None:
                risk_pcts.append(rp)

            prr = getattr(t, "planned_rr", None)
            if prr is not None:
                rr_planned.append(prr)

            rm = getattr(t, "r_multiple", None)
            if rm is not None:
                rr_actual.append(rm)

            meta = getattr(t, "metadata", {}) or {}
            if meta.get("stop_hit") or meta.get("stop_respected"):
                stop_respected += 1
            if meta.get("stop_moved_against"):
                stop_widened += 1
            if meta.get("target_moved"):
                target_moved += 1

        profile = RiskProfile()

        if risk_pcts:
            arr = np.array(risk_pcts)
            profile.avg_risk_per_trade = float(np.mean(arr))
            profile.risk_consistency = 1.0 - min(1.0, float(np.std(arr) / (np.mean(arr) + 1e-10)))
            profile.max_observed_risk = float(np.max(arr))

            # Classify personality
            avg = profile.avg_risk_per_trade
            consistency = profile.risk_consistency
            if avg <= 0.01 and consistency > 0.7:
                profile.personality = RiskPersonality.CONSERVATIVE
            elif avg <= 0.02 and consistency > 0.5:
                profile.personality = RiskPersonality.MODERATE
            elif avg > 0.03 or consistency < 0.3:
                if consistency < 0.3:
                    profile.personality = RiskPersonality.INCONSISTENT
                else:
                    profile.personality = RiskPersonality.AGGRESSIVE
            else:
                profile.personality = RiskPersonality.MODERATE

        if rr_planned:
            profile.avg_rr_planned = float(np.mean(rr_planned))
        if rr_actual:
            profile.avg_rr_actual = float(np.mean(rr_actual))

        total_with_stops = stop_respected + stop_widened
        if total_with_stops > 0:
            profile.stop_adherence_rate = stop_respected / total_with_stops

        profile.tends_to_widen_stops = stop_widened > len(trades) * 0.15
        profile.tends_to_move_targets = target_moved > len(trades) * 0.15

        return profile

    def _build_timing_profile(
        self, trades: list, dq_records: list
    ) -> TimingProfile:
        """Build timing preference profile."""
        hour_performance: dict[int, list[float]] = {}
        dow_performance: dict[int, list[float]] = {}
        hold_times = []

        for t in trades:
            entry_time = getattr(t, "entry_time", None)
            r_mult = getattr(t, "r_multiple", None)
            exit_time = getattr(t, "exit_time", None)

            if entry_time and r_mult is not None:
                hour = entry_time.hour
                dow = entry_time.weekday()

                if hour not in hour_performance:
                    hour_performance[hour] = []
                hour_performance[hour].append(r_mult)

                if dow not in dow_performance:
                    dow_performance[dow] = []
                dow_performance[dow].append(r_mult)

            if entry_time and exit_time:
                hold = (exit_time - entry_time).total_seconds() / 60.0
                hold_times.append(hold)

        timing = TimingProfile()

        # Best/worst hours
        if hour_performance:
            hour_avgs = {
                h: np.mean(rs) for h, rs in hour_performance.items()
                if len(rs) >= 3
            }
            if hour_avgs:
                sorted_hours = sorted(hour_avgs.items(), key=lambda x: x[1], reverse=True)
                timing.best_hours = [h for h, _ in sorted_hours[:3] if hour_avgs[h] > 0]
                timing.worst_hours = [h for h, _ in sorted_hours[-3:] if hour_avgs[h] < 0]

        # Best day of week
        if dow_performance:
            dow_avgs = {
                d: np.mean(rs) for d, rs in dow_performance.items()
                if len(rs) >= 3
            }
            if dow_avgs:
                timing.best_day_of_week = [
                    d for d, avg in sorted(dow_avgs.items(), key=lambda x: x[1], reverse=True)[:2]
                    if avg > 0
                ]

        if hold_times:
            timing.avg_hold_time_minutes = float(np.median(hold_times))

        # Optimal trade count: find where quality starts dropping
        timing.optimal_trade_count = self._find_optimal_trade_count(trades, dq_records)
        timing.quality_decay_after_n_trades = timing.optimal_trade_count

        return timing

    def _find_optimal_trade_count(
        self, trades: list, dq_records: list
    ) -> int:
        """
        Find the daily trade count after which quality degrades.

        Groups trades by day, counts sequential trade quality,
        looks for the inflection point.
        """
        # Group trades by date
        daily_trades: dict[str, list] = {}
        for t in trades:
            entry = getattr(t, "entry_time", None) or getattr(t, "created_at", None)
            if entry:
                day_key = entry.strftime("%Y-%m-%d")
                if day_key not in daily_trades:
                    daily_trades[day_key] = []
                daily_trades[day_key].append(t)

        # For each position (1st trade, 2nd trade, etc.), compute avg R
        position_r: dict[int, list[float]] = {}
        for day, day_trades in daily_trades.items():
            for i, t in enumerate(sorted(day_trades, key=lambda x: getattr(x, "entry_time", datetime.min))):
                pos = i + 1
                r = getattr(t, "r_multiple", None)
                if r is not None:
                    if pos not in position_r:
                        position_r[pos] = []
                    position_r[pos].append(r)

        if not position_r:
            return 3  # Default

        # Find where avg R drops below zero or below overall average
        overall_avg = np.mean([r for rs in position_r.values() for r in rs])
        optimal = 3  # Default
        for pos in sorted(position_r.keys()):
            if len(position_r[pos]) >= 5:
                pos_avg = np.mean(position_r[pos])
                if pos_avg < 0 or pos_avg < overall_avg * 0.5:
                    optimal = max(1, pos - 1)
                    break
                optimal = pos

        return min(optimal, 10)

    def _build_emotional_profile(
        self, trades: list, events: list
    ) -> EmotionalProfile:
        """Build emotional trigger profile."""
        profile = EmotionalProfile()

        # Analyze behavioral events for triggers
        trigger_counts: dict[str, int] = {}
        for e in events:
            etype = getattr(e, "event_type", "")
            desc = getattr(e, "description", "")
            if etype in {"cognitive_bias", "tilt_detected", "deviation"}:
                trigger = desc or etype
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

        if trigger_counts:
            sorted_triggers = sorted(
                trigger_counts.items(), key=lambda x: x[1], reverse=True
            )
            profile.primary_triggers = [t for t, _ in sorted_triggers[:5]]

        # Tilt frequency
        tilt_events = [
            e for e in events
            if getattr(e, "event_type", "") == "tilt_detected"
        ]
        if events:
            days_span = max(
                1,
                (datetime.utcnow() - min(
                    getattr(e, "created_at", datetime.utcnow()) for e in events
                )).days,
            )
            weeks = days_span / 7.0
            profile.tilt_frequency = len(tilt_events) / max(weeks, 1)

        # Post-loss behavior analysis
        profile.post_loss_behavior = self._analyze_post_outcome_behavior(
            trades, outcome="loss"
        )
        profile.post_win_behavior = self._analyze_post_outcome_behavior(
            trades, outcome="win"
        )

        # Emotional awareness from journaling and self-reporting
        reported_states = sum(
            1 for t in trades
            if getattr(t, "emotional_state", None) is not None
        )
        total = len(trades) if trades else 1
        profile.emotional_awareness_score = min(1.0, reported_states / total)

        return profile

    def _analyze_post_outcome_behavior(
        self, trades: list, outcome: str
    ) -> str:
        """Analyze what the trader does after a win/loss."""
        sorted_trades = sorted(
            trades,
            key=lambda t: getattr(t, "entry_time", datetime.min)
        )

        post_outcome_r = []
        post_outcome_size_change = []

        for i in range(len(sorted_trades) - 1):
            current = sorted_trades[i]
            next_trade = sorted_trades[i + 1]

            current_outcome = getattr(current, "outcome", None)
            if str(current_outcome) != outcome:
                continue

            next_r = getattr(next_trade, "r_multiple", None)
            if next_r is not None:
                post_outcome_r.append(next_r)

            # Check timing — was next trade within same session?
            current_exit = getattr(current, "exit_time", None)
            next_entry = getattr(next_trade, "entry_time", None)
            if current_exit and next_entry:
                gap = (next_entry - current_exit).total_seconds() / 60.0
                if gap < 5:
                    post_outcome_size_change.append("rapid")

        if not post_outcome_r:
            return "insufficient_data"

        avg_post_r = np.mean(post_outcome_r)
        rapid_count = sum(1 for x in post_outcome_size_change if x == "rapid")
        rapid_rate = rapid_count / len(post_outcome_r) if post_outcome_r else 0

        if outcome == "loss":
            if rapid_rate > 0.30 and avg_post_r < 0:
                return "revenge_trades"
            elif rapid_rate > 0.20:
                return "increases_activity"
            elif avg_post_r > 0:
                return "stays_disciplined"
            else:
                return "continues_losing"
        else:  # win
            if avg_post_r > 0.5:
                return "stays_disciplined"
            elif avg_post_r < -0.3:
                return "becomes_overconfident"
            else:
                return "neutral"

    def _classify_discipline_pattern(
        self,
        trades: list,
        discipline_records: list,
        dq_records: list,
    ) -> str:
        """Classify when discipline typically breaks down."""
        # Group by position in daily sequence
        daily_dq: dict[str, list[tuple[int, float]]] = {}
        for r in dq_records:
            if not r.composite_score:
                continue
            created = getattr(r, "created_at", None)
            if not created:
                continue
            day_key = created.strftime("%Y-%m-%d")
            if day_key not in daily_dq:
                daily_dq[day_key] = []
            daily_dq[day_key].append((len(daily_dq[day_key]) + 1, r.composite_score))

        if not daily_dq:
            return DisciplinePattern.CONSISTENT

        # Check morning vs afternoon decay
        early_scores = []
        late_scores = []
        for day, seq in daily_dq.items():
            if len(seq) >= 3:
                mid = len(seq) // 2
                early_scores.extend([s for _, s in seq[:mid]])
                late_scores.extend([s for _, s in seq[mid:]])

        if early_scores and late_scores:
            early_avg = np.mean(early_scores)
            late_avg = np.mean(late_scores)
            if early_avg - late_avg > 0.10:
                return DisciplinePattern.MORNING_STRONG

        # Check post-loss decay
        # (Look for declining DQ after losses)
        post_loss_dq = []
        sorted_records = sorted(
            dq_records,
            key=lambda r: getattr(r, "created_at", datetime.min)
        )
        for i in range(1, len(sorted_records)):
            prev = sorted_records[i - 1]
            curr = sorted_records[i]
            if prev.composite_score and prev.composite_score < 0.5:
                if curr.composite_score:
                    post_loss_dq.append(curr.composite_score)

        if post_loss_dq and np.mean(post_loss_dq) < 0.55:
            return DisciplinePattern.LOSS_TRIGGERED

        # Check trade count fatigue
        count_dq: dict[int, list[float]] = {}
        for day, seq in daily_dq.items():
            for pos, score in seq:
                if pos not in count_dq:
                    count_dq[pos] = []
                count_dq[pos].append(score)

        if count_dq:
            pos_avgs = {
                pos: np.mean(scores) for pos, scores in count_dq.items()
                if len(scores) >= 3
            }
            if pos_avgs:
                positions = sorted(pos_avgs.keys())
                if len(positions) >= 3:
                    first_avg = pos_avgs.get(positions[0], 0)
                    last_avg = pos_avgs.get(positions[-1], 0)
                    if first_avg - last_avg > 0.12:
                        return DisciplinePattern.FATIGUE_DRIVEN

        return DisciplinePattern.CONSISTENT

    def _build_strategy_preferences(
        self, trades: list
    ) -> list[StrategyPreference]:
        """Build per-strategy preference profiles."""
        strat_data: dict[str, dict[str, Any]] = {}
        total_trades = len(trades)

        for t in trades:
            strat = getattr(t, "strategy", None) or "unattributed"
            if strat not in strat_data:
                strat_data[strat] = {
                    "count": 0, "wins": 0, "r_values": [],
                    "regimes": {},
                }
            strat_data[strat]["count"] += 1
            if getattr(t, "outcome", None) == "win":
                strat_data[strat]["wins"] += 1
            r = getattr(t, "r_multiple", None)
            if r is not None:
                strat_data[strat]["r_values"].append(r)
            regime = getattr(t, "market_regime", None)
            if regime:
                if regime not in strat_data[strat]["regimes"]:
                    strat_data[strat]["regimes"][regime] = []
                strat_data[strat]["regimes"][regime].append(r or 0)

        preferences = []
        for strat, data in strat_data.items():
            count = data["count"]
            r_vals = data["r_values"]

            # Best/worst regime
            best_regime = None
            worst_regime = None
            if data["regimes"]:
                regime_avgs = {
                    r: np.mean(vs) for r, vs in data["regimes"].items()
                    if len(vs) >= 3
                }
                if regime_avgs:
                    best_regime = max(regime_avgs, key=regime_avgs.get)
                    worst_regime = min(regime_avgs, key=regime_avgs.get)

            pref = StrategyPreference(
                strategy_name=strat,
                frequency=count / total_trades if total_trades > 0 else 0,
                win_rate=data["wins"] / count if count > 0 else 0,
                avg_r=float(np.mean(r_vals)) if r_vals else 0,
                is_core=count / total_trades >= 0.15,
                best_regime=best_regime,
                worst_regime=worst_regime,
            )
            preferences.append(pref)

        # Sort by frequency
        preferences.sort(key=lambda p: p.frequency, reverse=True)
        return preferences

    def _analyze_performance_context(
        self, trades: list
    ) -> tuple[list[str], list[str]]:
        """Identify conditions where trader excels and struggles."""
        context_r: dict[str, list[float]] = {}

        for t in trades:
            regime = getattr(t, "market_regime", None)
            entry_time = getattr(t, "entry_time", None)
            r = getattr(t, "r_multiple", None)
            strategy = getattr(t, "strategy", None)

            if r is None:
                continue

            # Build context keys
            contexts = []
            if regime:
                contexts.append(regime)
            if entry_time:
                hour = entry_time.hour
                if hour < 11:
                    contexts.append("morning")
                elif hour < 14:
                    contexts.append("midday")
                else:
                    contexts.append("afternoon")
            if strategy:
                contexts.append(strategy)

            # Store combinations
            for ctx in contexts:
                if ctx not in context_r:
                    context_r[ctx] = []
                context_r[ctx].append(r)

            # Two-way combinations
            if regime and entry_time:
                period = "morning" if entry_time.hour < 11 else "afternoon"
                combo = f"{regime}+{period}"
                if combo not in context_r:
                    context_r[combo] = []
                context_r[combo].append(r)

        # Rank contexts
        context_avgs = {
            ctx: float(np.mean(rs))
            for ctx, rs in context_r.items()
            if len(rs) >= 5
        }

        sorted_contexts = sorted(
            context_avgs.items(), key=lambda x: x[1], reverse=True
        )

        best = [ctx for ctx, avg in sorted_contexts[:5] if avg > 0.3]
        worst = [ctx for ctx, avg in sorted_contexts[-5:] if avg < -0.2]

        return best, worst