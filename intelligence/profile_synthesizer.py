"""
PTIS Trader Profile Synthesizer
==============================

Unifies multiple profile analysis streams into a cohesive trader intelligence
profile. Synthesizes data from:

- Decision Quality Engine (execution patterns, quality trends)
- Behavioral Detection (deviations, emotional triggers, discipline)
- Trade History (risk tolerance, timing preferences, outcomes)
- Strategy Attribution (preferred setups, regime compatibility)

The synthesizer creates a unified profile that evolves with each new trade
and provides personalization context for coaching agents and alerts.

Design
------
- Event-driven updates from trade pipeline
- Combines existing profile builders (no duplication)
- Incremental updates - O(1) per new trade
- Explainable classifications with evidence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session as DBSession

from core.events import Event, EventType, get_event_bus
from core.models import DecisionQualityRecord, Trade, BehavioralEvent, DisciplineRecord
from intelligence.trader_profile_intelligence import (
    TraderIntelligenceProfile, TraderProfileBuilder as IntelligenceBuilder
)
from behavioral.trader_profile import (
    TraderProfile as BehavioralProfile, TraderProfileBuilder as BehavioralBuilder
)
from analytics.decision_quality import DecisionQualityResult

logger = logging.getLogger(__name__)


@dataclass
class ProfileSynthesisConfig:
    """Configuration for profile synthesis."""
    # Update triggers
    update_on_trade_close: bool = True
    update_on_decision_quality: bool = True
    update_on_behavioral_event: bool = True
    
    # Synthesis weights
    intelligence_weight: float = 0.6
    behavioral_weight: float = 0.4
    
    # Minimum data thresholds
    min_trades_for_synthesis: int = 10
    min_decision_quality_scores: int = 5
    
    # Update frequency
    max_full_rebuild_frequency_hours: int = 24
    incremental_update_threshold_trades: int = 20


@dataclass
class SynthesisEvidence:
    """Evidence supporting a profile classification."""
    source: str  # "decision_quality", "behavioral", "trade_history"
    evidence_type: str  # "statistic", "pattern", "event"
    value: Any
    confidence: float  # 0-1
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UnifiedTraderProfile:
    """
    Unified trader profile synthesizing all intelligence sources.
    
    This is the canonical profile that feeds all downstream systems:
    - Coaching agent personalization
    - Alert threshold calibration  
    - Strategy recommendations
    - Risk management adjustments
    """
    
    profile_id: UUID = field(default_factory=uuid4)
    trader_id: str = ""
    synthesized_at: datetime = field(default_factory=datetime.utcnow)
    
    # Core classifications (synthesized)
    risk_tolerance: str = "moderate"  # very_conservative, conservative, moderate, aggressive, very_aggressive
    trading_style: str = "day_trader"  # scalper, day_trader, swing_trader, position_trader
    decision_style: str = "analytical"  # analytical, intuitive, systematic
    emotional_profile: str = "stable"  # stable, reactive, vulnerable, volatile
    
    # Behavioral patterns
    discipline_consistency: float = 0.5  # 0-1
    emotional_triggers: List[str] = field(default_factory=list)
    vulnerability_windows: List[str] = field(default_factory=list)
    
    # Performance context
    best_conditions: List[str] = field(default_factory=list)
    worst_conditions: List[str] = field(default_factory=list)
    optimal_trade_frequency: int = 3  # trades per session
    quality_decay_triggers: List[str] = field(default_factory=list)
    
    # Decision quality patterns
    avg_dq_score: float = 0.0
    dq_trend: str = "stable"  # improving, declining, stable
    weakest_dimensions: List[str] = field(default_factory=list)
    strongest_dimensions: List[str] = field(default_factory=list)
    
    # Risk behavior
    position_sizing_pattern: str = "consistent"  # consistent, erratic, conservative, aggressive
    stop_adherence_rate: float = 0.8
    risk_reward_tendency: str = "balanced"  # conservative, balanced, aggressive
    
    # Learning & adaptation
    improvement_velocity: float = 0.0  # rate of improvement per 100 trades
    adaptation_score: float = 0.5  # how well they learn from mistakes
    repeatability_score: float = 0.5  # how consistently they apply lessons
    
    # Meta
    data_confidence: float = 0.0  # overall confidence in profile
    sample_size: int = 0
    evidence: List[SynthesisEvidence] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "profile_id": str(self.profile_id),
            "trader_id": self.trader_id,
            "synthesized_at": self.synthesized_at.isoformat(),
            "classifications": {
                "risk_tolerance": self.risk_tolerance,
                "trading_style": self.trading_style,
                "decision_style": self.decision_style,
                "emotional_profile": self.emotional_profile,
            },
            "behavioral_patterns": {
                "discipline_consistency": round(self.discipline_consistency, 3),
                "emotional_triggers": self.emotional_triggers,
                "vulnerability_windows": self.vulnerability_windows,
            },
            "performance_context": {
                "best_conditions": self.best_conditions,
                "worst_conditions": self.worst_conditions,
                "optimal_trade_frequency": self.optimal_trade_frequency,
                "quality_decay_triggers": self.quality_decay_triggers,
            },
            "decision_quality": {
                "avg_score": round(self.avg_dq_score, 2),
                "trend": self.dq_trend,
                "weakest_dimensions": self.weakest_dimensions,
                "strongest_dimensions": self.strongest_dimensions,
            },
            "risk_behavior": {
                "position_sizing_pattern": self.position_sizing_pattern,
                "stop_adherence_rate": round(self.stop_adherence_rate, 3),
                "risk_reward_tendency": self.risk_reward_tendency,
            },
            "learning_adaptation": {
                "improvement_velocity": round(self.improvement_velocity, 3),
                "adaptation_score": round(self.adaptation_score, 3),
                "repeatability_score": round(self.repeatability_score, 3),
            },
            "meta": {
                "data_confidence": round(self.data_confidence, 3),
                "sample_size": self.sample_size,
                "last_updated": self.last_updated.isoformat(),
            }
        }


class TraderProfileSynthesizer:
    """
    Synthesizes multiple intelligence streams into a unified trader profile.
    
    Event-driven architecture:
    - Listens to trade close events
    - Incorporates decision quality evaluations  
    - Processes behavioral events
    - Updates profile incrementally
    """
    
    def __init__(self, config: Optional[ProfileSynthesisConfig] = None):
        self.cfg = config or ProfileSynthesisConfig()
        self._event_bus = get_event_bus()
        self._intelligence_builder = IntelligenceBuilder()
        self._behavioral_builder = BehavioralBuilder()
        
        # Subscribe to relevant events
        self._setup_event_listeners()
        
        logger.info("TraderProfileSynthesizer initialized")
    
    def _setup_event_listeners(self) -> None:
        """Subscribe to events that trigger profile updates."""
        if self.cfg.update_on_trade_close:
            self._event_bus.subscribe(EventType.TRADE_CLOSED, self._on_trade_closed)
        
        # Note: Decision quality events use BEHAVIORAL_EVENT type with custom data
        if self.cfg.update_on_decision_quality:
            self._event_bus.subscribe(EventType.BEHAVIORAL_EVENT, self._on_behavioral_event)
        
        if self.cfg.update_on_behavioral_event:
            self._event_bus.subscribe(EventType.BEHAVIORAL_EVENT, self._on_behavioral_event)
    
    async def synthesize_profile(
        self, 
        trader_id: str, 
        db: AsyncSession,
        force_rebuild: bool = False
    ) -> UnifiedTraderProfile:
        """
        Synthesize complete unified profile for a trader.
        
        Parameters
        ----------
        trader_id : str
            Trader identifier
        db : AsyncSession
            Database session
        force_rebuild : bool
            Force full rebuild instead of incremental update
        """
        logger.info(f"Synthesizing profile for trader {trader_id}")
        
        # Fetch data from all intelligence sources
        trades = await self._fetch_trades(db, trader_id)
        dq_records = await self._fetch_decision_quality_records(db, trader_id)
        behavioral_events = await self._fetch_behavioral_events(db, trader_id)
        discipline_records = await self._fetch_discipline_records(db, trader_id)
        
        if len(trades) < self.cfg.min_trades_for_synthesis:
            logger.warning(f"Insufficient trades for synthesis: {len(trades)} < {self.cfg.min_trades_for_synthesis}")
            return UnifiedTraderProfile(
                trader_id=trader_id,
                data_confidence=0.0,
                sample_size=len(trades)
            )
        
        # Build individual profiles
        intelligence_profile = await self._intelligence_builder.build(trader_id)
        behavioral_profile = self._behavioral_builder.build(trades)
        
        # Synthesize into unified profile
        unified = self._synthesize_profiles(
            trader_id=trader_id,
            intelligence_profile=intelligence_profile,
            behavioral_profile=behavioral_profile,
            trades=trades,
            dq_records=dq_records,
            behavioral_events=behavioral_events,
            discipline_records=discipline_records
        )
        
        logger.info(
            f"Profile synthesized for {trader_id}: "
            f"risk={unified.risk_tolerance}, style={unified.trading_style}, "
            f"confidence={unified.data_confidence:.2f}"
        )
        
        return unified
    
    def _synthesize_profiles(
        self,
        trader_id: str,
        intelligence_profile: TraderIntelligenceProfile,
        behavioral_profile: BehavioralProfile,
        trades: List[Trade],
        dq_records: List[DecisionQualityRecord],
        behavioral_events: List[BehavioralEvent],
        discipline_records: List[DisciplineRecord]
    ) -> UnifiedTraderProfile:
        """Synthesize multiple profiles into unified profile."""
        
        unified = UnifiedTraderProfile(trader_id=trader_id)
        unified.sample_size = len(trades)
        
        # Synthesize core classifications
        unified.risk_tolerance = self._synthesize_risk_tolerance(
            intelligence_profile, behavioral_profile, trades
        )
        unified.trading_style = self._synthesize_trading_style(
            intelligence_profile, behavioral_profile, trades
        )
        unified.decision_style = self._synthesize_decision_style(
            intelligence_profile, behavioral_profile
        )
        unified.emotional_profile = self._synthesize_emotional_profile(
            intelligence_profile, behavioral_profile, behavioral_events
        )
        
        # Synthesize behavioral patterns
        unified.discipline_consistency = self._synthesize_discipline_consistency(
            intelligence_profile, behavioral_profile, discipline_records
        )
        unified.emotional_triggers = self._extract_emotional_triggers(
            intelligence_profile, behavioral_events
        )
        unified.vulnerability_windows = self._identify_vulnerability_windows(
            intelligence_profile, behavioral_events, dq_records
        )
        
        # Synthesize performance context
        unified.best_conditions, unified.worst_conditions = self._analyze_performance_conditions(
            intelligence_profile, trades, dq_records
        )
        unified.optimal_trade_frequency = intelligence_profile.timing.optimal_trade_count
        unified.quality_decay_triggers = self._identify_quality_decay_triggers(
            intelligence_profile, dq_records, behavioral_events
        )
        
        # Synthesize decision quality patterns
        unified.avg_dq_score = intelligence_profile.avg_dq_score
        unified.dq_trend = intelligence_profile.discipline_trend
        unified.weakest_dimensions = [intelligence_profile.weakest_dq_dimension] if intelligence_profile.weakest_dq_dimension else []
        unified.strongest_dimensions = [intelligence_profile.strongest_dq_dimension] if intelligence_profile.strongest_dq_dimension else []
        
        # Synthesize risk behavior
        unified.position_sizing_pattern = self._analyze_position_sizing_pattern(
            behavioral_profile, trades
        )
        unified.stop_adherence_rate = intelligence_profile.risk.stop_adherence_rate
        unified.risk_reward_tendency = self._analyze_risk_reward_tendency(
            intelligence_profile, trades
        )
        
        # Calculate learning & adaptation metrics
        unified.improvement_velocity = self._calculate_improvement_velocity(dq_records)
        unified.adaptation_score = self._calculate_adaptation_score(
            behavioral_events, dq_records
        )
        unified.repeatability_score = self._calculate_repeatability_score(
            dq_records, behavioral_events
        )
        
        # Calculate overall confidence
        unified.data_confidence = self._calculate_overall_confidence(
            len(trades), len(dq_records), len(behavioral_events)
        )
        
        # Generate evidence
        unified.evidence = self._generate_synthesis_evidence(
            intelligence_profile, behavioral_profile, trades, dq_records
        )
        
        return unified
    
    def _synthesize_risk_tolerance(
        self, 
        intelligence_profile: TraderIntelligenceProfile,
        behavioral_profile: BehavioralProfile,
        trades: List[Trade]
    ) -> str:
        """Synthesize risk tolerance from multiple sources."""
        
        # Get risk assessments from each source
        intelligence_risk = intelligence_profile.risk.personality
        behavioral_risk = behavioral_profile.risk_tolerance.classification
        
        # Trade-based risk assessment
        avg_risk_pct = sum([t.risk_pct or 0 for t in trades]) / len(trades) if trades else 0
        max_risk_pct = max([t.risk_pct or 0 for t in trades]) if trades else 0
        
        # Weighted decision
        risk_scores = {
            "very_conservative": 1,
            "conservative": 2, 
            "moderate": 3,
            "aggressive": 4,
            "very_aggressive": 5,
            "inconsistent": 3  # Treat as moderate
        }
        
        intelligence_score = risk_scores.get(intelligence_risk, 3) * self.cfg.intelligence_weight
        behavioral_score = risk_scores.get(behavioral_risk, 3) * self.cfg.behavioral_weight
        
        # Adjust based on actual trading behavior
        if avg_risk_pct > 0.03:  # High average risk
            behavioral_score += 1
        elif avg_risk_pct < 0.01:  # Low average risk
            behavioral_score -= 1
        
        if max_risk_pct > 0.05:  # Very high risk taken
            behavioral_score += 0.5
        
        final_score = intelligence_score + behavioral_score
        
        # Map back to classification
        if final_score <= 1.5:
            return "very_conservative"
        elif final_score <= 2.5:
            return "conservative"
        elif final_score <= 3.5:
            return "moderate"
        elif final_score <= 4.5:
            return "aggressive"
        else:
            return "very_aggressive"
    
    def _synthesize_trading_style(
        self,
        intelligence_profile: TraderIntelligenceProfile,
        behavioral_profile: BehavioralProfile,
        trades: List[Trade]
    ) -> str:
        """Synthesize trading style from hold times and patterns."""
        
        # Use intelligence profile as primary (it has more sophisticated analysis)
        primary_style = intelligence_profile.trading_style
        
        # Validate with behavioral profile
        behavioral_style = behavioral_profile.trading_style.classification
        
        # If they agree, use that classification
        if primary_style == behavioral_style:
            return primary_style
        
        # If they disagree, analyze trade hold times directly
        hold_times = []
        for trade in trades:
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 60  # minutes
                hold_times.append(duration)
        
        if not hold_times:
            return primary_style  # Default to intelligence profile
        
        avg_hold = sum(hold_times) / len(hold_times)
        
        # Direct classification from hold times
        if avg_hold < 10:
            return "scalper"
        elif avg_hold < 240:  # 4 hours
            return "day_trader"
        elif avg_hold < 7200:  # 5 days
            return "swing_trader"
        else:
            return "position_trader"
    
    def _synthesize_decision_style(
        self,
        intelligence_profile: TraderIntelligenceProfile,
        behavioral_profile: BehavioralProfile
    ) -> str:
        """Synthesize decision style from cognitive patterns."""
        
        # Intelligence profile has cognitive_type analysis
        cognitive_type = intelligence_profile.cognitive_type
        
        # Behavioral profile might have entry style clues
        entry_style = behavioral_profile.entry_style.classification
        
        # Map to decision style
        if cognitive_type == "systematic":
            return "systematic"
        elif cognitive_type == "intuitive":
            return "intuitive"
        else:  # analytical
            # Check entry style for refinement
            if entry_style == "anticipation":
                return "analytical"  # Requires analysis to anticipate
            elif entry_style == "reactive":
                return "intuitive"  # More reactive/intuitive
            else:
                return "analytical"
    
    def _synthesize_emotional_profile(
        self,
        intelligence_profile: TraderIntelligenceProfile,
        behavioral_profile: BehavioralProfile,
        behavioral_events: List[BehavioralEvent]
    ) -> str:
        """Synthesize emotional profile from multiple indicators."""
        
        # Get emotional vulnerability assessments
        behavioral_vulnerability = behavioral_profile.emotional_vulnerability.classification
        
        # Analyze behavioral events for emotional patterns
        tilt_events = len([e for e in behavioral_events if "tilt" in e.event_type.value.lower()])
        emotional_events = len([e for e in behavioral_events if "emotional" in e.event_type.value.lower()])
        
        # Calculate emotional stability score
        total_events = len(behavioral_events)
        emotional_volatility = (tilt_events + emotional_events) / max(total_events, 1)
        
        # Map to emotional profile
        if emotional_volatility > 0.3:
            return "volatile"
        elif emotional_volatility > 0.15:
            return "vulnerable"
        elif emotional_volatility > 0.05:
            return "reactive"
        else:
            return "stable"
    
    def _synthesize_discipline_consistency(
        self,
        intelligence_profile: TraderIntelligenceProfile,
        behavioral_profile: BehavioralProfile,
        discipline_records: List[DisciplineRecord]
    ) -> float:
        """Calculate overall discipline consistency."""
        
        consistency_scores = []
        
        # From intelligence profile
        if intelligence_profile.sample_size > 0:
            consistency_scores.append(intelligence_profile.current_discipline_score / 100.0)
        
        # From behavioral profile
        if behavioral_profile.position_sizing_consistency.confidence > 0.3:
            sizing_consistency = behavioral_profile.position_sizing_consistency.score / 100.0
            consistency_scores.append(sizing_consistency)
        
        # From discipline records
        if discipline_records:
            recent_scores = [r.score for r in discipline_records[:10]]
            avg_discipline = sum(recent_scores) / len(recent_scores) / 100.0
            consistency_scores.append(avg_discipline)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
    
    def _extract_emotional_triggers(
        self,
        intelligence_profile: TraderIntelligenceProfile,
        behavioral_events: List[BehavioralEvent]
    ) -> List[str]:
        """Extract common emotional triggers."""
        
        triggers = set()
        
        # From intelligence profile
        triggers.update(intelligence_profile.emotional.primary_triggers)
        
        # From behavioral events
        for event in behavioral_events:
            if event.description:
                # Extract common trigger keywords
                desc = event.description.lower()
                if "loss" in desc and "consecutive" in desc:
                    triggers.add("consecutive_losses")
                if "revenge" in desc:
                    triggers.add("revenge_trading")
                if "fomo" in desc or "chase" in desc:
                    triggers.add("fomo_chasing")
                if "bored" in desc:
                    triggers.add("boredom_trading")
        
        return list(triggers)[:5]  # Top 5 triggers
    
    def _identify_vulnerability_windows(
        self,
        intelligence_profile: TraderIntelligenceProfile,
        behavioral_events: List[BehavioralEvent],
        dq_records: List[DecisionQualityRecord]
    ) -> List[str]:
        """Identify conditions where trader is most vulnerable."""
        
        vulnerabilities = []
        
        # Time-based vulnerabilities
        if intelligence_profile.discipline_pattern == "morning_strong":
            vulnerabilities.append("afternoon_fatigue")
        elif intelligence_profile.discipline_pattern == "fatigue_driven":
            vulnerabilities.append("late_session_decline")
        
        # Loss-based vulnerabilities
        if intelligence_profile.discipline_pattern == "loss_triggered":
            vulnerabilities.append("post_loss_escalation")
        
        # Low decision quality periods
        low_dq_periods = [r for r in dq_records if r.composite_score < 50]
        if len(low_dq_periods) > len(dq_records) * 0.3:
            vulnerabilities.append("frequent_process_breakdown")
        
        return vulnerabilities
    
    def _analyze_performance_conditions(
        self,
        intelligence_profile: TraderIntelligenceProfile,
        trades: List[Trade],
        dq_records: List[DecisionQualityRecord]
    ) -> Tuple[List[str], List[str]]:
        """Analyze best and worst performance conditions."""
        
        # Use intelligence profile analysis
        best = intelligence_profile.best_conditions.copy()
        worst = intelligence_profile.worst_conditions.copy()
        
        # Add decision quality insights
        high_dq_trades = []
        low_dq_trades = []
        
        for trade in trades:
            # Find corresponding DQ record
            dq = next((r for r in dq_records if r.trade_id == trade.id), None)
            if dq:
                if dq.composite_score >= 75:
                    high_dq_trades.append(trade)
                elif dq.composite_score < 50:
                    low_dq_trades.append(trade)
        
        # Analyze conditions for high/low quality trades
        if high_dq_trades:
            # Common conditions in high-quality trades
            morning_trades = len([t for t in high_dq_trades if t.session_type and "morning" in t.session_type.value])
            if morning_trades > len(high_dq_trades) * 0.6:
                best.append("morning_session")
        
        if low_dq_trades:
            # Common conditions in low-quality trades
            afternoon_trades = len([t for t in low_dq_trades if t.session_type and "afternoon" in t.session_type.value])
            if afternoon_trades > len(low_dq_trades) * 0.6:
                worst.append("afternoon_session")
        
        return best[:5], worst[:5]  # Top 5 each
    
    def _identify_quality_decay_triggers(
        self,
        intelligence_profile: TraderIntelligenceProfile,
        dq_records: List[DecisionQualityRecord],
        behavioral_events: List[BehavioralEvent]
    ) -> List[str]:
        """Identify triggers that cause decision quality to decay."""
        
        triggers = []
        
        # Trade count fatigue
        if intelligence_profile.timing.quality_decay_after_n_trades:
            triggers.append(f"trade_count_over_{intelligence_profile.timing.quality_decay_after_n_trades}")
        
        # Behavioral event correlation
        tilt_events = [e for e in behavioral_events if "tilt" in e.event_type.value.lower()]
        if tilt_events:
            triggers.append("emotional_tilt_detected")
        
        # Consecutive losses impact
        if intelligence_profile.discipline_pattern == "loss_triggered":
            triggers.append("consecutive_losses")
        
        return triggers
    
    def _analyze_position_sizing_pattern(
        self,
        behavioral_profile: BehavioralProfile,
        trades: List[Trade]
    ) -> str:
        """Analyze position sizing patterns."""
        
        # Use behavioral profile classification
        classification = behavioral_profile.position_sizing_consistency.classification
        
        if classification in ["consistent", "moderate", "erratic"]:
            return classification
        
        # Fallback to direct analysis
        position_pcts = [t.risk_pct for t in trades if t.risk_pct is not None]
        if len(position_pcts) < 2:
            return "consistent"
        
        # Calculate coefficient of variation
        import statistics
        mean_pct = statistics.mean(position_pcts)
        std_pct = statistics.stdev(position_pcts) if len(position_pcts) > 1 else 0
        cv = std_pct / mean_pct if mean_pct > 0 else 0
        
        if cv < 0.2:
            return "consistent"
        elif cv < 0.5:
            return "moderate"
        else:
            return "erratic"
    
    def _analyze_risk_reward_tendency(
        self,
        intelligence_profile: TraderIntelligenceProfile,
        trades: List[Trade]
    ) -> str:
        """Analyze risk-reward tendencies."""
        
        # From intelligence profile
        avg_rr_planned = intelligence_profile.risk.avg_rr_planned
        avg_rr_actual = intelligence_profile.risk.avg_rr_actual
        
        if avg_rr_planned >= 2.5:
            return "conservative"  # High R:R requirement
        elif avg_rr_planned <= 1.5:
            return "aggressive"  # Low R:R requirement
        else:
            return "balanced"
    
    def _calculate_improvement_velocity(
        self,
        dq_records: List[DecisionQualityRecord]
    ) -> float:
        """Calculate rate of improvement in decision quality."""
        
        if len(dq_records) < 10:
            return 0.0
        
        # Sort by date
        sorted_records = sorted(dq_records, key=lambda r: r.evaluated_at)
        
        # Compare first third vs last third
        first_third = sorted_records[:len(sorted_records)//3]
        last_third = sorted_records[-len(sorted_records)//3:]
        
        first_avg = sum(r.composite_score for r in first_third) / len(first_third)
        last_avg = sum(r.composite_score for r in last_third) / len(last_third)
        
        # Improvement per 100 trades (normalized)
        improvement = last_avg - first_avg
        trades_per_improvement = len(sorted_records) / 3
        velocity = (improvement / trades_per_improvement) * 100 if trades_per_improvement > 0 else 0
        
        return velocity
    
    def _calculate_adaptation_score(
        self,
        behavioral_events: List[BehavioralEvent],
        dq_records: List[DecisionQualityRecord]
    ) -> float:
        """Calculate how well trader adapts from mistakes."""
        
        # Look for reduction in behavioral events over time
        if len(behavioral_events) < 10:
            return 0.5
        
        # Split events in half and compare frequency
        sorted_events = sorted(behavioral_events, key=lambda e: e.detected_at)
        first_half = sorted_events[:len(sorted_events)//2]
        second_half = sorted_events[len(sorted_events)//2:]
        
        first_frequency = len(first_half)
        second_frequency = len(second_half)
        
        # Calculate adaptation (reduction in negative events)
        if first_frequency == 0:
            return 1.0
        
        reduction = (first_frequency - second_frequency) / first_frequency
        adaptation_score = max(0.0, min(1.0, reduction + 0.5))  # Normalize to 0-1
        
        return adaptation_score
    
    def _calculate_repeatability_score(
        self,
        dq_records: List[DecisionQualityRecord],
        behavioral_events: List[BehavioralEvent]
    ) -> float:
        """Calculate how consistently trader applies lessons."""
        
        if len(dq_records) < 5:
            return 0.5
        
        # Calculate consistency in decision quality scores
        scores = [r.composite_score for r in dq_records]
        import statistics
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0
        
        # Lower standard deviation = higher repeatability
        max_possible_std = 50.0  # Maximum reasonable standard deviation
        repeatability = 1.0 - (std_score / max_possible_std)
        
        return max(0.0, min(1.0, repeatability))
    
    def _calculate_overall_confidence(
        self,
        trade_count: int,
        dq_count: int,
        behavioral_count: int
    ) -> float:
        """Calculate overall confidence in profile synthesis."""
        
        # Base confidence from sample size
        if trade_count >= 200:
            base_confidence = 0.9
        elif trade_count >= 100:
            base_confidence = 0.8
        elif trade_count >= 50:
            base_confidence = 0.7
        elif trade_count >= 20:
            base_confidence = 0.5
        else:
            base_confidence = 0.3
        
        # Adjust based on data completeness
        data_completeness = min(1.0, (
            (trade_count / 50.0) +      # Trade history weight
            (dq_count / 20.0) +        # Decision quality weight
            (behavioral_count / 30.0)  # Behavioral events weight
        ) / 3.0)
        
        # Combine base confidence with data completeness
        overall_confidence = base_confidence * (0.5 + 0.5 * data_completeness)
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _generate_synthesis_evidence(
        self,
        intelligence_profile: TraderIntelligenceProfile,
        behavioral_profile: BehavioralProfile,
        trades: List[Trade],
        dq_records: List[DecisionQualityRecord]
    ) -> List[SynthesisEvidence]:
        """Generate evidence supporting synthesis decisions."""
        
        evidence = []
        
        # Risk tolerance evidence
        evidence.append(SynthesisEvidence(
            source="intelligence_profile",
            evidence_type="classification",
            value=intelligence_profile.risk.personality,
            confidence=intelligence_profile.confidence,
            description=f"Risk personality from {intelligence_profile.sample_size} trades"
        ))
        
        # Trading style evidence
        if trades:
            avg_hold = sum([
                (t.exit_time - t.entry_time).total_seconds() / 60 
                for t in trades 
                if t.entry_time and t.exit_time
            ]) / len(trades)
            evidence.append(SynthesisEvidence(
                source="trade_history",
                evidence_type="statistic",
                value=avg_hold,
                confidence=0.9,
                description=f"Average hold time: {avg_hold:.1f} minutes from {len(trades)} trades"
            ))
        
        # Decision quality evidence
        if dq_records:
            avg_dq = sum(r.composite_score for r in dq_records) / len(dq_records)
            evidence.append(SynthesisEvidence(
                source="decision_quality",
                evidence_type="statistic", 
                value=avg_dq,
                confidence=0.8,
                description=f"Average decision quality: {avg_dq:.1f} from {len(dq_records)} evaluations"
            ))
        
        return evidence
    
    # Event handlers
    async def _on_trade_closed(self, event: Event) -> None:
        """Handle trade close events - trigger profile update."""
        if event.trade_id:
            logger.debug(f"Trade closed event received for profile update: {event.trade_id}")
            # In production, would queue async profile update
            # For now, just log the event
            pass
    
    async def _on_behavioral_event(self, event: Event) -> None:
        """Handle behavioral events - check if profile update needed."""
        event_type = event.data.get("event_type", "")
        
        if event_type == "decision_quality_evaluated":
            logger.debug(f"Decision quality evaluated: {event.data.get('evaluation_id')}")
            # Could trigger incremental profile update
        
        # For other behavioral events, check severity
        severity = event.data.get("severity", "")
        if severity in ["high", "critical"]:
            logger.debug(f"High severity behavioral event: {event_type}")
            # Could trigger immediate profile update
    
    # Data fetchers (simplified for demo)
    async def _fetch_trades(self, db: AsyncSession, trader_id: str) -> List[Trade]:
        """Fetch trades for profile synthesis."""
        # Implementation would query database
        return []
    
    async def _fetch_decision_quality_records(
        self, 
        db: AsyncSession, 
        trader_id: str
    ) -> List[DecisionQualityRecord]:
        """Fetch decision quality records."""
        # Implementation would query database
        return []
    
    async def _fetch_behavioral_events(
        self, 
        db: AsyncSession, 
        trader_id: str
    ) -> List[BehavioralEvent]:
        """Fetch behavioral events."""
        # Implementation would query database  
        return []
    
    async def _fetch_discipline_records(
        self, 
        db: AsyncSession, 
        trader_id: str
    ) -> List[DisciplineRecord]:
        """Fetch discipline records."""
        # Implementation would query database
        return []


# Singleton instance
_profile_synthesizer: Optional[TraderProfileSynthesizer] = None

def get_profile_synthesizer() -> TraderProfileSynthesizer:
    """Get global profile synthesizer instance."""
    global _profile_synthesizer
    if _profile_synthesizer is None:
        _profile_synthesizer = TraderProfileSynthesizer()
    return _profile_synthesizer
