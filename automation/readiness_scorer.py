# readiness_scorer.py
# automation/readiness_scorer.py
"""
Automation Readiness Assessment.
Scores how ready each strategy is for semi or full automation.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_
import numpy as np

from core.models import Trade, AutomationReadinessScore
from core.enums import TradeStatus, AutomationPhase
from strategy_engine.strategy_dna.loader import StrategyDNALoader
from strategy_engine.strategy_dna.learning_updater import (
    StrategyDNALearningUpdater, StrategyPerformanceSnapshot
)
import logging

logger = logging.getLogger(__name__)


class AutomationReadinessScorer:
    """
    Assesses readiness for automation transition per strategy.

    Dimensions:
    1. Rule Clarity — how well-defined are the strategy rules?
    2. Rule Consistency — how consistently do you follow them?
    3. Performance Stability — is performance consistent?
    4. Edge Confidence — statistical evidence of edge?
    5. Behavioral Discipline — freedom from emotional trading?

    Phase Progression:
    1. Observation → 2. Awareness → 3. Alignment → 4. Semi-Auto → 5. Full Auto
    """

    # Phase requirements
    PHASE_REQUIREMENTS = {
        AutomationPhase.OBSERVATION: {
            "min_trades": 0,
            "min_discipline": 0,
            "min_consistency": 0,
        },
        AutomationPhase.AWARENESS: {
            "min_trades": 20,
            "min_discipline": 40,
            "min_consistency": 40,
        },
        AutomationPhase.ALIGNMENT: {
            "min_trades": 50,
            "min_discipline": 60,
            "min_consistency": 60,
        },
        AutomationPhase.SEMI_AUTO: {
            "min_trades": 100,
            "min_discipline": 75,
            "min_consistency": 70,
            "min_edge_confidence": 60,
        },
        AutomationPhase.FULL_AUTO: {
            "min_trades": 200,
            "min_discipline": 85,
            "min_consistency": 80,
            "min_edge_confidence": 75,
            "min_performance_stability": 70,
        },
    }

    def __init__(self, db: Session, dna_loader: StrategyDNALoader):
        self.db = db
        self.dna_loader = dna_loader

    def assess_strategy(
        self,
        strategy_name: str,
        lookback_days: int = 90,
    ) -> Dict:
        """Assess automation readiness for a single strategy."""
        dna = self.dna_loader.get(strategy_name)

        # Get performance data
        updater = StrategyDNALearningUpdater(self.db, self.dna_loader)
        snapshot = updater._compute_snapshot(strategy_name, lookback_days)

        # Score each dimension
        rule_clarity = self._score_rule_clarity(dna)
        rule_consistency = self._score_rule_consistency(snapshot, strategy_name)
        performance_stability = self._score_performance_stability(snapshot)
        edge_confidence = self._score_edge_confidence(snapshot)
        behavioral_discipline = self._score_behavioral_discipline(snapshot)

        # Overall score
        overall = np.mean([
            rule_clarity,
            rule_consistency,
            performance_stability,
            edge_confidence,
            behavioral_discipline,
        ])

        # Determine current and recommended phase
        current_phase = self._determine_phase(
            snapshot, overall, rule_consistency, edge_confidence,
            performance_stability, behavioral_discipline
        )

        # Identify blockers
        blockers = self._identify_blockers(
            snapshot, rule_clarity, rule_consistency,
            performance_stability, edge_confidence, behavioral_discipline
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_phase, blockers, snapshot
        )

        # Save to database
        score_record = AutomationReadinessScore(
            strategy_name=strategy_name,
            rule_clarity=rule_clarity,
            rule_consistency=rule_consistency,
            performance_stability=performance_stability,
            edge_confidence=edge_confidence,
            behavioral_discipline=behavioral_discipline,
            overall_readiness=overall,
            current_phase=current_phase,
            recommended_phase=current_phase,
            blockers=blockers,
            recommendations=recommendations,
        )
        self.db.add(score_record)
        self.db.flush()

        return {
            "strategy": strategy_name,
            "display_name": dna.display_name if dna else strategy_name,
            "scores": {
                "rule_clarity": round(rule_clarity, 1),
                "rule_consistency": round(rule_consistency, 1),
                "performance_stability": round(performance_stability, 1),
                "edge_confidence": round(edge_confidence, 1),
                "behavioral_discipline": round(behavioral_discipline, 1),
                "overall": round(overall, 1),
            },
            "current_phase": current_phase.value,
            "sample_size": snapshot.total_trades,
            "blockers": blockers,
            "recommendations": recommendations,
        }

    def _score_rule_clarity(self, dna) -> float:
        """Score how well-defined the strategy rules are."""
        if not dna:
            return 20

        score = 0

        # Has conditions defined
        if dna.setup_conditions:
            score += min(30, len(dna.setup_conditions) * 5)

        # Has critical conditions
        critical = [c for c in dna.setup_conditions if c.importance == "critical"]
        score += min(20, len(critical) * 7)

        # Has risk rules
        if dna.stop_method:
            score += 15
        if dna.risk_per_trade:
            score += 10

        # Has exit rules
        if dna.profit_taking_rules:
            score += 10
        if dna.time_exit_rules:
            score += 5

        # Has checklist
        if dna.checklist and len(dna.checklist) >= 5:
            score += 10

        return min(100, score)

    def _score_rule_consistency(
        self,
        snapshot: StrategyPerformanceSnapshot,
        strategy_name: str,
    ) -> float:
        """Score how consistently rules are followed."""
        if snapshot.total_trades < 10:
            return 30

        # Based on override rate
        total = snapshot.total_trades
        overrides = snapshot.trades_with_overrides
        clean_rate = (total - overrides) / total * 100

        # Based on checklist scores
        if snapshot.checklist_scores:
            avg_checklist = np.mean(snapshot.checklist_scores)
            checklist_factor = avg_checklist / 100
        else:
            checklist_factor = 0.5

        score = clean_rate * 0.6 + (checklist_factor * 100) * 0.4
        return min(100, score)

    def _score_performance_stability(
        self,
        snapshot: StrategyPerformanceSnapshot,
    ) -> float:
        """Score consistency of performance."""
        if snapshot.total_trades < 20:
            return 30

        if not snapshot.r_values:
            return 30

        # Low variance in R-multiples = more stable
        r_std = np.std(snapshot.r_values)
        r_mean = abs(np.mean(snapshot.r_values)) if np.mean(snapshot.r_values) != 0 else 1

        # Coefficient of variation (lower is more stable)
        cv = r_std / r_mean if r_mean > 0 else float('inf')

        if cv < 1:
            stability_score = 80
        elif cv < 2:
            stability_score = 60
        elif cv < 3:
            stability_score = 40
        else:
            stability_score = 20

        # Positive expectancy bonus
        if snapshot.expectancy > 0:
            stability_score += 20

        return min(100, stability_score)

    def _score_edge_confidence(
        self,
        snapshot: StrategyPerformanceSnapshot,
    ) -> float:
        """Score statistical confidence in the strategy's edge."""
        if snapshot.total_trades < 20:
            return 20

        score = 0

        # Win rate above 45%
        if snapshot.win_rate >= 55:
            score += 30
        elif snapshot.win_rate >= 45:
            score += 15

        # Positive expectancy
        if snapshot.expectancy > 0.1:
            score += 25
        elif snapshot.expectancy > 0:
            score += 10

        # Profit factor > 1.5
        if snapshot.profit_factor > 2.0:
            score += 25
        elif snapshot.profit_factor > 1.5:
            score += 15
        elif snapshot.profit_factor > 1.0:
            score += 5

        # Sample size bonus
        if snapshot.total_trades >= 100:
            score += 20
        elif snapshot.total_trades >= 50:
            score += 10

        return min(100, score)

    def _score_behavioral_discipline(
        self,
        snapshot: StrategyPerformanceSnapshot,
    ) -> float:
        """Score freedom from emotional trading."""
        if snapshot.total_trades < 10:
            return 30

        score = 50

        # Low override rate
        override_rate = snapshot.trades_with_overrides / max(snapshot.total_trades, 1)
        if override_rate < 0.1:
            score += 30
        elif override_rate < 0.2:
            score += 15
        elif override_rate > 0.4:
            score -= 20

        # Discipline edge positive (following rules helps)
        if snapshot.override_impact and snapshot.override_impact > 0:
            score += 20

        return min(100, max(0, score))

    def _determine_phase(
        self,
        snapshot: StrategyPerformanceSnapshot,
        overall: float,
        consistency: float,
        edge: float,
        stability: float,
        discipline: float,
    ) -> AutomationPhase:
        """Determine current automation phase."""
        trades = snapshot.total_trades

        if (trades >= 200 and overall >= 80 and discipline >= 85 and
            consistency >= 80 and edge >= 75 and stability >= 70):
            return AutomationPhase.FULL_AUTO

        if (trades >= 100 and overall >= 65 and discipline >= 75 and
            consistency >= 70 and edge >= 60):
            return AutomationPhase.SEMI_AUTO

        if trades >= 50 and overall >= 55 and discipline >= 60:
            return AutomationPhase.ALIGNMENT

        if trades >= 20 and overall >= 40:
            return AutomationPhase.AWARENESS

        return AutomationPhase.OBSERVATION

    def _identify_blockers(
        self,
        snapshot: StrategyPerformanceSnapshot,
        clarity: float,
        consistency: float,
        stability: float,
        edge: float,
        discipline: float,
    ) -> List[str]:
        """Identify what's blocking advancement."""
        blockers = []

        if snapshot.total_trades < 50:
            blockers.append(
                f"Need more trades ({snapshot.total_trades}/50 minimum for Phase 3)"
            )

        if clarity < 60:
            blockers.append("Strategy rules not well-defined enough — complete DNA profile")

        if consistency < 60:
            blockers.append("Rule adherence too low — focus on following your checklist")

        if stability < 50:
            blockers.append("Performance too inconsistent — review regime filtering")

        if edge < 50:
            blockers.append("Edge not statistically significant — need more data or strategy refinement")

        if discipline < 60:
            blockers.append("Emotional trading still present — work on behavioral discipline")

        return blockers

    def _generate_recommendations(
        self,
        phase: AutomationPhase,
        blockers: List[str],
        snapshot: StrategyPerformanceSnapshot,
    ) -> List[str]:
        """Generate specific recommendations for advancement."""
        recs = []

        if phase == AutomationPhase.OBSERVATION:
            recs.append("Keep logging every trade with full journal entries")
            recs.append("Complete the pre-trade checklist for every trade")
            recs.append("Review shadow trade comparisons daily")

        elif phase == AutomationPhase.AWARENESS:
            recs.append("Focus on improving checklist adherence to 80%+")
            recs.append("Study your divergence patterns — where do you deviate?")
            recs.append("Implement mandatory break after losses")

        elif phase == AutomationPhase.ALIGNMENT:
            recs.append("Target 90% rule adherence for 30 consecutive trades")
            recs.append("Review and update strategy DNA based on learned statistics")
            recs.append("Practice regime filtering — skip poor-fit conditions")

        elif phase == AutomationPhase.SEMI_AUTO:
            recs.append("Enable alert-based entry confirmations")
            recs.append("Test automated stop/target management on shadow trades")
            recs.append("Document every manual override and its outcome")

        return recs