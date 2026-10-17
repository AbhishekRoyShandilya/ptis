# coplanner/trade_quality_classifier.py
"""
INTELLIGENT TRADE QUALITY CLASSIFIER

Philosophy:
  This system is NOT a rigid rule checker.
  It's an INTELLIGENT evaluator that understands:

  1. INTENT   -> What was the trader trying to do?
  2. CONTEXT  -> What was happening in the market?
  3. PROCESS  -> Was the deviation reasonable?
  4. OUTCOME  -> Did the adaptation work? (informative, not punitive)
  5. PATTERN  -> Is this a recurring behavior or one-off?

KEY PRINCIPLES:
  - Adaptation != Emotion
  - Outcome doesn't define quality (process > result)
  - Traders evolve (classifier learns the user's growth)
  - Context matters everything
  - Flexibility IS an edge

CLASSIFICATION IS A SPECTRUM, NOT BINARY:
  0.0 ---- 0.5 ---- 1.0
  Pure       Deviation     Pure clean
  emotional  with cause    execution
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from coplanner.models import ConfidenceLevel, TradeRecord

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ENUMS & DATA MODELS
# ─────────────────────────────────────────────

class DeviationType(Enum):
    """Types of deviations from the original plan."""
    NONE = "none"
    SL_WIDENED = "sl_widened"
    SL_TIGHTENED = "sl_tightened"
    EARLY_EXIT_WIN = "early_exit_win"
    EARLY_EXIT_LOSS = "early_exit_loss"
    LATE_EXIT = "late_exit"
    OVERSIZED = "oversized"
    UNDERSIZED = "undersized"
    NO_PLAN = "no_plan"
    COUNTER_REGIME = "counter_regime"
    RAPID_ENTRY = "rapid_entry"
    MULTIPLE_ENTRIES = "multiple_entries"


class DeviationIntent(Enum):
    """
    ESTIMATED intent behind the deviation.
    This is the KEY innovation -- we try to understand WHY.
    """
    ADAPTATION = "adaptation"
    RISK_MANAGEMENT = "risk_mgmt"
    OPPORTUNITY = "opportunity"
    FEAR = "fear"
    GREED = "greed"
    FOMO = "fomo"
    REVENGE = "revenge"
    BOREDOM = "boredom"
    UNCLEAR = "unclear"
    DISCIPLINED = "disciplined"


@dataclass
class DeviationDetail:
    """Detailed breakdown of a single deviation."""
    deviation_type: DeviationType
    estimated_intent: DeviationIntent
    severity: float                  # 0.0 (harmless) to 1.0 (severe)
    reasoning: str
    context_factors: List[str] = field(default_factory=list)
    was_profitable: Optional[bool] = None

    @property
    def is_constructive(self) -> bool:
        """Was this deviation potentially constructive?"""
        return self.estimated_intent in (
            DeviationIntent.ADAPTATION,
            DeviationIntent.RISK_MANAGEMENT,
            DeviationIntent.OPPORTUNITY,
            DeviationIntent.DISCIPLINED,
        )


@dataclass
class QualityAssessment:
    """
    Complete quality assessment of a trade.
    quality_score is the primary measure (0.0 to 1.0).
    """
    trade_id: str
    quality_score: float
    quality_label: str

    deviations: List[DeviationDetail] = field(default_factory=list)
    total_deviation_severity: float = 0.0

    market_context_considered: bool = False
    context_notes: List[str] = field(default_factory=list)

    process_score: float = 0.5
    outcome_score: float = 0.5

    trust_weight: float = 0.5

    compared_to_user_average: str = "average"

    feedback_message: str = ""
    improvement_suggestion: str = ""

    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────────
# MAIN CLASSIFIER
# ─────────────────────────────────────────────

class TradeQualityClassifier:
    """
    Intelligent trade quality classifier.
    NOT a rule checker. An UNDERSTANDING engine.

    Usage:
        classifier = TradeQualityClassifier(db)
        assessment = classifier.assess(trade)
        tier, tag, reasons = classifier.classify(trade)  # DataTrustLayer compat
    """

    def __init__(
        self,
        db=None,
        user_history: Optional[List[TradeRecord]] = None,
        config: Optional[dict] = None,
    ):
        self.db = db
        self._user_history = user_history
        self.config = config or {}

        self._user_baselines: Optional[Dict[str, Any]] = None
        self._classification_cache: Dict[str, QualityAssessment] = {}
        self._compute_user_baselines()

    # ─────────────────────────────────────────────
    # MAIN CLASSIFICATION — DataTrustLayer compat
    # ─────────────────────────────────────────────

    def classify(self, trade: TradeRecord) -> Tuple:
        """
        Returns tuple for DataTrustLayer compatibility:
        (DataTier, TradeQualityTag, List[str] reasons)
        """
        assessment = self.assess(trade)
        from coplanner.data_trust_layer import DataTier, TradeQualityTag

        tier, tag = self._assessment_to_tier(assessment)
        reasons = [d.reasoning for d in assessment.deviations]
        if not reasons:
            reasons = [assessment.feedback_message]
        return tier, tag, reasons

    def assess(self, trade: TradeRecord) -> QualityAssessment:
        """
        Full quality assessment of a trade.

        FLOW:
        1. Detect deviations from plan
        2. Analyze CONTEXT of each deviation
        3. Estimate INTENT behind each deviation
        4. Score process quality (separate from outcome)
        5. Compute trust weight
        6. Generate feedback
        """
        if trade.trade_id in self._classification_cache:
            return self._classification_cache[trade.trade_id]

        # Step 1: Detect what happened differently from plan
        deviations = self._detect_deviations(trade)

        # Step 2: Analyze context
        context_notes = self._analyze_context(trade, deviations)

        # Step 3: Estimate intent
        self._estimate_intents(trade, deviations, context_notes)

        # Step 4: Score process quality
        process_score = self._score_process(deviations, trade)

        # Step 5: Score outcome (informative only)
        outcome_score = self._score_outcome(trade)

        # Step 6: Compute overall quality — process matters MORE
        quality_score = (
            process_score * 0.75
            + outcome_score * 0.15
            + self._consistency_bonus(trade) * 0.10
        )
        quality_score = max(0.0, min(1.0, quality_score))

        # Step 7: Compute trust weight
        trust_weight = self._compute_trust_weight(quality_score, deviations)

        # Step 8: Generate human feedback
        label = self._generate_label(quality_score, deviations)
        feedback = self._generate_feedback(quality_score, deviations, trade)
        suggestion = self._generate_suggestion(deviations, trade)

        # Step 9: Compare to user's average
        comparison = self._compare_to_average(quality_score)

        assessment = QualityAssessment(
            trade_id=trade.trade_id,
            quality_score=round(quality_score, 4),
            quality_label=label,
            deviations=deviations,
            total_deviation_severity=sum(d.severity for d in deviations),
            market_context_considered=bool(context_notes),
            context_notes=context_notes,
            process_score=round(process_score, 4),
            outcome_score=round(outcome_score, 4),
            trust_weight=round(trust_weight, 4),
            compared_to_user_average=comparison,
            feedback_message=feedback,
            improvement_suggestion=suggestion,
        )

        self._classification_cache[trade.trade_id] = assessment
        return assessment

    # ─────────────────────────────────────────────
    # STEP 1: DEVIATION DETECTION
    # ─────────────────────────────────────────────

    def _detect_deviations(self, trade: TradeRecord) -> List[DeviationDetail]:
        """Detect what deviated from the original plan. FACTUAL — no judgment yet."""
        deviations: List[DeviationDetail] = []

        has_plan = self._has_valid_plan(trade)
        if not has_plan:
            deviations.append(DeviationDetail(
                deviation_type=DeviationType.NO_PLAN,
                estimated_intent=DeviationIntent.UNCLEAR,
                severity=0.6,
                reasoning="No stop loss or target defined at entry",
                context_factors=["Missing trade plan data"],
            ))
            return deviations

        sl_deviation = self._analyze_sl_deviation(trade)
        if sl_deviation:
            deviations.append(sl_deviation)

        exit_deviation = self._analyze_exit_deviation(trade)
        if exit_deviation:
            deviations.append(exit_deviation)

        timing_deviation = self._analyze_timing(trade)
        if timing_deviation:
            deviations.append(timing_deviation)

        notes_deviations = self._analyze_notes(trade)
        deviations.extend(notes_deviations)

        if not deviations:
            deviations.append(DeviationDetail(
                deviation_type=DeviationType.NONE,
                estimated_intent=DeviationIntent.DISCIPLINED,
                severity=0.0,
                reasoning="Trade followed plan -- clean execution",
            ))

        return deviations

    def _has_valid_plan(self, trade: TradeRecord) -> bool:
        return bool(
            trade.stop_loss and trade.stop_loss > 0
            and trade.target_price and trade.target_price > 0
            and trade.entry_price and trade.entry_price > 0
        )

    def _analyze_sl_deviation(self, trade: TradeRecord) -> Optional[DeviationDetail]:
        """Analyze if stop loss was respected, widened, or tightened."""
        if trade.outcome != "loss":
            return None

        expected_loss = abs(trade.entry_price - trade.stop_loss)
        actual_loss = abs(trade.entry_price - trade.exit_price)
        if expected_loss == 0:
            return None

        loss_ratio = actual_loss / expected_loss

        if loss_ratio <= 1.05:
            return None
        elif loss_ratio <= 1.20:
            return DeviationDetail(
                deviation_type=DeviationType.SL_WIDENED,
                estimated_intent=DeviationIntent.UNCLEAR,
                severity=0.15,
                reasoning=(
                    f"Exit loss {loss_ratio:.0%} of planned SL -- "
                    f"minor deviation, possibly slippage or small adaptation"
                ),
                context_factors=["within_slippage_range"],
                was_profitable=False,
            )
        elif loss_ratio <= 1.50:
            return DeviationDetail(
                deviation_type=DeviationType.SL_WIDENED,
                estimated_intent=DeviationIntent.UNCLEAR,
                severity=0.40,
                reasoning=(
                    f"Exit loss {loss_ratio:.0%} of planned SL -- "
                    f"noticeable widening of stop"
                ),
                context_factors=["moderate_sl_breach"],
                was_profitable=False,
            )
        else:
            return DeviationDetail(
                deviation_type=DeviationType.SL_WIDENED,
                estimated_intent=DeviationIntent.UNCLEAR,
                severity=0.75,
                reasoning=(
                    f"Exit loss {loss_ratio:.0%} of planned SL -- "
                    f"significant stop widening"
                ),
                context_factors=["major_sl_breach"],
                was_profitable=False,
            )

    def _analyze_exit_deviation(self, trade: TradeRecord) -> Optional[DeviationDetail]:
        """Analyze exit behavior relative to target."""
        if not trade.target_price or trade.target_price == 0:
            return None

        potential_reward = abs(trade.target_price - trade.entry_price)
        if potential_reward == 0:
            return None

        # Handle direction
        if trade.target_price > trade.entry_price:
            actual_reward = trade.exit_price - trade.entry_price
        else:
            actual_reward = trade.entry_price - trade.exit_price

        capture_ratio = actual_reward / potential_reward

        if trade.outcome == "win":
            if capture_ratio >= 0.80:
                return None
            elif capture_ratio >= 0.50:
                return DeviationDetail(
                    deviation_type=DeviationType.EARLY_EXIT_WIN,
                    estimated_intent=DeviationIntent.UNCLEAR,
                    severity=0.10,
                    reasoning=f"Captured {capture_ratio:.0%} of planned target -- partial capture",
                    context_factors=["partial_profit_capture"],
                    was_profitable=True,
                )
            elif capture_ratio >= 0.20:
                return DeviationDetail(
                    deviation_type=DeviationType.EARLY_EXIT_WIN,
                    estimated_intent=DeviationIntent.UNCLEAR,
                    severity=0.30,
                    reasoning=f"Captured only {capture_ratio:.0%} of target -- very early exit",
                    context_factors=["minimal_profit_capture"],
                    was_profitable=True,
                )
            else:
                return DeviationDetail(
                    deviation_type=DeviationType.EARLY_EXIT_WIN,
                    estimated_intent=DeviationIntent.UNCLEAR,
                    severity=0.45,
                    reasoning=(
                        f"Captured only {capture_ratio:.0%} -- "
                        f"exited almost immediately despite move continuing"
                    ),
                    context_factors=["panic_exit_likely"],
                    was_profitable=True,
                )

        elif trade.outcome == "loss":
            sl_distance = abs(trade.entry_price - trade.stop_loss)
            exit_distance = abs(trade.entry_price - trade.exit_price)
            if sl_distance > 0 and exit_distance < sl_distance * 0.70:
                return DeviationDetail(
                    deviation_type=DeviationType.EARLY_EXIT_LOSS,
                    estimated_intent=DeviationIntent.RISK_MANAGEMENT,
                    severity=0.05,
                    reasoning=(
                        f"Cut loss early at {exit_distance / sl_distance:.0%} "
                        f"of planned SL distance -- potentially smart risk management"
                    ),
                    context_factors=["early_loss_cut"],
                    was_profitable=False,
                )

        return None

    def _analyze_timing(self, trade: TradeRecord) -> Optional[DeviationDetail]:
        """Analyze entry/exit timing."""
        if not trade.entry_time or not trade.exit_time:
            return None

        holding_seconds = (trade.exit_time - trade.entry_time).total_seconds()
        holding_minutes = holding_seconds / 60

        if holding_minutes < 1:
            return DeviationDetail(
                deviation_type=DeviationType.RAPID_ENTRY,
                estimated_intent=DeviationIntent.UNCLEAR,
                severity=0.50,
                reasoning=(
                    f"Trade lasted only {holding_seconds:.0f} seconds -- "
                    f"extremely rapid, likely impulsive entry/exit"
                ),
                context_factors=["ultra_short_duration"],
            )
        elif holding_minutes < 3:
            return DeviationDetail(
                deviation_type=DeviationType.RAPID_ENTRY,
                estimated_intent=DeviationIntent.UNCLEAR,
                severity=0.25,
                reasoning=(
                    f"Trade lasted {holding_minutes:.1f} minutes -- "
                    f"very short hold, could be impulsive or quick scalp"
                ),
                context_factors=["short_duration"],
            )
        return None

    def _analyze_notes(self, trade: TradeRecord) -> List[DeviationDetail]:
        """Analyze trade notes for emotional and adaptation signals."""
        deviations: List[DeviationDetail] = []
        notes = getattr(trade, "notes", "") or ""
        notes = notes.lower().strip()
        if not notes:
            return deviations

        # Clearly emotional signals
        emotional_strong = {
            "revenge": (DeviationIntent.REVENGE, 0.80, "Notes mention revenge trading"),
            "fomo": (DeviationIntent.FOMO, 0.70, "Notes mention FOMO"),
            "panic": (DeviationIntent.FEAR, 0.65, "Notes mention panic"),
            "angry": (DeviationIntent.REVENGE, 0.70, "Notes mention anger"),
            "frustrated": (DeviationIntent.REVENGE, 0.60, "Notes mention frustration"),
            "bored": (DeviationIntent.BOREDOM, 0.55, "Notes mention boredom"),
            "overtraded": (DeviationIntent.BOREDOM, 0.60, "Notes acknowledge overtrading"),
            "no setup": (DeviationIntent.BOREDOM, 0.65, "Notes say no real setup"),
            "random": (DeviationIntent.BOREDOM, 0.70, "Notes describe random entry"),
            "gut feel": (DeviationIntent.UNCLEAR, 0.30, "Notes mention gut feel"),
            "hope": (DeviationIntent.FEAR, 0.50, "Notes mention hope"),
            "scared": (DeviationIntent.FEAR, 0.60, "Notes mention being scared"),
            "impulsive": (DeviationIntent.FOMO, 0.65, "Notes acknowledge impulsive entry"),
        }

        # Positive adaptation signals (reduce severity)
        adaptation_signals = {
            "adapted": (DeviationIntent.ADAPTATION, -0.20, "Notes mention adaptation"),
            "structure changed": (DeviationIntent.ADAPTATION, -0.25, "Notes reference structure change"),
            "read the tape": (DeviationIntent.ADAPTATION, -0.20, "Notes mention tape reading"),
            "risk management": (DeviationIntent.RISK_MANAGEMENT, -0.25, "Notes reference risk management"),
            "reduced risk": (DeviationIntent.RISK_MANAGEMENT, -0.20, "Notes mention reducing risk"),
            "trailed stop": (DeviationIntent.ADAPTATION, -0.15, "Notes mention trailing stop"),
            "partial exit": (DeviationIntent.RISK_MANAGEMENT, -0.15, "Notes mention partial exit"),
            "scaling out": (DeviationIntent.RISK_MANAGEMENT, -0.15, "Notes mention scaling out"),
            "volatility picked up": (DeviationIntent.ADAPTATION, -0.15, "Notes reference volatility change"),
            "news event": (DeviationIntent.RISK_MANAGEMENT, -0.20, "Notes mention news-driven adaptation"),
        }

        for keyword, (intent, severity, reason) in emotional_strong.items():
            if keyword in notes:
                deviations.append(DeviationDetail(
                    deviation_type=DeviationType.NO_PLAN if intent == DeviationIntent.BOREDOM else DeviationType.NONE,
                    estimated_intent=intent,
                    severity=severity,
                    reasoning=reason,
                    context_factors=[f"keyword:{keyword}"],
                ))

        for keyword, (intent, severity_modifier, reason) in adaptation_signals.items():
            if keyword in notes:
                deviations.append(DeviationDetail(
                    deviation_type=DeviationType.NONE,
                    estimated_intent=intent,
                    severity=severity_modifier,  # NEGATIVE = reduces total severity
                    reasoning=reason,
                    context_factors=[f"positive:{keyword}"],
                ))

        return deviations

    # ─────────────────────────────────────────────
    # STEP 2: CONTEXT ANALYSIS
    # ─────────────────────────────────────────────

    def _analyze_context(
        self, trade: TradeRecord, deviations: List[DeviationDetail],
    ) -> List[str]:
        context: List[str] = []

        recent_context = self._get_recent_trading_context(trade)
        context.extend(recent_context)

        time_context = self._get_time_context(trade)
        if time_context:
            context.append(time_context)

        regime_context = self._get_regime_context(trade)
        if regime_context:
            context.append(regime_context)

        streak_context = self._get_streak_context(trade)
        if streak_context:
            context.append(streak_context)

        return context

    def _get_recent_trading_context(self, trade: TradeRecord) -> List[str]:
        context: List[str] = []
        if not self.db:
            return context
        try:
            recent = self.db.get_recent_trades(5)
            preceding = [
                t for t in recent
                if t.exit_time and trade.entry_time and t.exit_time < trade.entry_time
            ]
            if not preceding:
                return context

            recent_losses = sum(1 for t in preceding if t.outcome == "loss")
            recent_wins = sum(1 for t in preceding if t.outcome == "win")

            if recent_losses >= 3:
                context.append(
                    f"CONTEXT: {recent_losses} consecutive losses before this trade -- "
                    f"elevated revenge/recovery risk"
                )
            if recent_wins >= 4:
                context.append(
                    f"CONTEXT: {recent_wins} consecutive wins before this trade -- "
                    f"elevated overconfidence risk"
                )

            if len(preceding) >= 3 and preceding[0].entry_time and preceding[-1].entry_time:
                time_span = (preceding[0].entry_time - preceding[-1].entry_time).total_seconds() / 3600
                if time_span < 1:
                    context.append(
                        f"CONTEXT: {len(preceding)} trades in {time_span:.1f} hours -- "
                        f"possible overtrading"
                    )
        except Exception as e:
            logger.debug("Could not get recent trading context: %s", e)
        return context

    def _get_time_context(self, trade: TradeRecord) -> Optional[str]:
        if not trade.entry_time:
            return None
        hour = trade.entry_time.hour
        minute = trade.entry_time.minute
        if hour == 9 and minute < 30:
            return "CONTEXT: Entry during market open rush -- higher volatility, could justify adaptation"
        if 12 <= hour <= 13:
            return "CONTEXT: Entry during lunch hours -- typically lower conviction time"
        if hour == 15 and minute >= 15:
            return "CONTEXT: Entry in last 15 minutes -- compressed time, higher risk"
        return None

    def _get_regime_context(self, trade: TradeRecord) -> Optional[str]:
        regime = (trade.regime_at_entry or "").lower()
        if not regime or regime == "unknown":
            return None
        if "volatile" in regime or "choppy" in regime:
            return f"CONTEXT: Regime was '{regime}' -- deviations may be market-driven, not emotional"
        return f"CONTEXT: Regime at entry was '{regime}'"

    def _get_streak_context(self, trade: TradeRecord) -> Optional[str]:
        if not self.db:
            return None
        try:
            today_trades = [
                t for t in self.db.get_recent_trades(20)
                if t.entry_time and trade.entry_time
                and t.entry_time.date() == trade.entry_time.date()
            ]
            if len(today_trades) >= 8:
                return f"CONTEXT: {len(today_trades)} trades today -- significantly above normal"
            elif len(today_trades) >= 5:
                return f"CONTEXT: {len(today_trades)} trades today -- above average"
        except Exception:
            pass
        return None

    # ─────────────────────────────────────────────
    # STEP 3: INTENT ESTIMATION (THE INTELLIGENCE)
    # ─────────────────────────────────────────────

    def _estimate_intents(
        self, trade: TradeRecord, deviations: List[DeviationDetail], context: List[str],
    ) -> None:
        """Estimate the INTENT behind each deviation using context."""
        has_consecutive_losses = any("consecutive losses" in c for c in context)
        has_consecutive_wins = any("consecutive wins" in c for c in context)
        has_volatile_regime = any("volatile" in c.lower() for c in context)
        has_overtrading = any("overtrading" in c for c in context)
        has_market_open = any("market open" in c for c in context)

        for dev in deviations:
            if dev.estimated_intent != DeviationIntent.UNCLEAR:
                continue

            if dev.deviation_type == DeviationType.SL_WIDENED:
                if has_volatile_regime or has_market_open:
                    if dev.severity <= 0.30:
                        dev.estimated_intent = DeviationIntent.ADAPTATION
                        dev.reasoning += " -- Market was volatile, minor widening could be adaptation"
                        dev.severity *= 0.6
                    else:
                        dev.estimated_intent = DeviationIntent.FEAR
                        dev.reasoning += " -- Despite volatility, widening was excessive"
                elif has_consecutive_losses:
                    dev.estimated_intent = DeviationIntent.FEAR
                    dev.reasoning += " -- After consecutive losses, likely fear of another loss"
                    dev.severity *= 1.2
                else:
                    if dev.severity <= 0.20:
                        dev.estimated_intent = DeviationIntent.ADAPTATION
                        dev.reasoning += " -- Minor, could be slippage or adaptation"
                    else:
                        dev.estimated_intent = DeviationIntent.FEAR
                        dev.reasoning += " -- Significant widening suggests emotional response"

            elif dev.deviation_type == DeviationType.EARLY_EXIT_WIN:
                if has_consecutive_losses:
                    dev.estimated_intent = DeviationIntent.FEAR
                    dev.reasoning += " -- After losses, likely securing gains from fear"
                elif has_volatile_regime:
                    dev.estimated_intent = DeviationIntent.ADAPTATION
                    dev.reasoning += " -- Volatile regime, partial profit taking can be smart"
                    dev.severity *= 0.5
                elif "partial_profit_capture" in dev.context_factors:
                    dev.estimated_intent = DeviationIntent.RISK_MANAGEMENT
                    dev.reasoning += " -- Captured significant portion, reasonable exit"
                    dev.severity *= 0.5
                else:
                    user_early_exit_success = self._user_early_exit_pattern()
                    if user_early_exit_success and user_early_exit_success > 0.6:
                        dev.estimated_intent = DeviationIntent.ADAPTATION
                        dev.reasoning += (
                            f" -- User's early exits historically successful "
                            f"({user_early_exit_success:.0%}), likely developed skill"
                        )
                        dev.severity *= 0.4
                    else:
                        dev.estimated_intent = DeviationIntent.FEAR

            elif dev.deviation_type == DeviationType.EARLY_EXIT_LOSS:
                dev.estimated_intent = DeviationIntent.RISK_MANAGEMENT
                dev.reasoning += " -- Cutting losses before SL is often smart risk management"
                dev.severity = max(0, dev.severity * 0.3)

            elif dev.deviation_type == DeviationType.RAPID_ENTRY:
                if has_overtrading:
                    dev.estimated_intent = DeviationIntent.BOREDOM
                    dev.reasoning += " -- Combined with overtrading pattern, likely impulsive"
                elif has_consecutive_losses:
                    dev.estimated_intent = DeviationIntent.REVENGE
                    dev.reasoning += " -- Quick entry after losses suggests revenge trading"
                    dev.severity *= 1.3
                elif has_consecutive_wins:
                    dev.estimated_intent = DeviationIntent.FOMO
                    dev.reasoning += " -- Quick entry on winning streak, possible overconfidence"
                else:
                    dev.estimated_intent = DeviationIntent.UNCLEAR
                    dev.reasoning += " -- Rapid but no clear emotional trigger"
                    dev.severity *= 0.7

    # ─────────────────────────────────────────────
    # STEP 4 & 5: SCORING
    # ─────────────────────────────────────────────

    def _score_process(self, deviations: List[DeviationDetail], trade: TradeRecord) -> float:
        """
        Score the PROCESS quality.
        Start at 0.80 (assume decent until proven otherwise).
        """
        base_score = 0.80
        for dev in deviations:
            if dev.deviation_type == DeviationType.NONE and dev.severity <= 0:
                base_score += abs(dev.severity) * 0.5
            elif dev.deviation_type == DeviationType.NONE and dev.severity == 0:
                base_score += 0.05
            elif dev.is_constructive:
                if dev.severity <= 0.15:
                    base_score += 0.02
                else:
                    base_score -= dev.severity * 0.15
            else:
                base_score -= dev.severity * 0.5

        if self._has_valid_plan(trade):
            base_score += 0.05

        return max(0.0, min(1.0, base_score))

    def _score_outcome(self, trade: TradeRecord) -> float:
        """Score outcome -- INFORMATIVE, not punitive."""
        if trade.outcome == "win":
            if trade.actual_rr >= 2.0:
                return 0.90
            elif trade.actual_rr >= 1.0:
                return 0.75
            else:
                return 0.60
        elif trade.outcome == "loss":
            if trade.actual_rr >= -1.0:
                return 0.45
            elif trade.actual_rr >= -2.0:
                return 0.30
            else:
                return 0.15
        return 0.50

    def _consistency_bonus(self, trade: TradeRecord) -> float:
        """Bonus for consistent execution quality."""
        if not self.db:
            return 0.5
        try:
            recent = self.db.get_recent_trades(5)
            if len(recent) < 3:
                return 0.5
            clean_count = 0
            for t in recent:
                if t.trade_id in self._classification_cache:
                    cached = self._classification_cache[t.trade_id]
                    if cached.quality_score >= 0.65:
                        clean_count += 1
                else:
                    if t.stop_loss and t.stop_loss > 0 and t.target_price and t.target_price > 0:
                        clean_count += 1
            return min(1.0, clean_count / len(recent))
        except Exception:
            return 0.5

    # ─────────────────────────────────────────────
    # TRUST WEIGHT COMPUTATION
    # ─────────────────────────────────────────────

    def _compute_trust_weight(
        self, quality_score: float, deviations: List[DeviationDetail],
    ) -> float:
        has_smart_adaptation = any(
            d.is_constructive and d.severity < 0.15 for d in deviations
        )

        if quality_score >= 0.90:
            weight = 0.80
        elif quality_score >= 0.75:
            weight = 0.70
        elif quality_score >= 0.60:
            weight = 0.55
        elif quality_score >= 0.50:
            weight = 0.40
        elif quality_score >= 0.35:
            weight = 0.20
        else:
            weight = 0.05

        if has_smart_adaptation:
            weight = min(0.85, weight + 0.05)

        return weight

    # ─────────────────────────────────────────────
    # CONVERSION TO DataTrustLayer FORMAT
    # ─────────────────────────────────────────────

    def _assessment_to_tier(self, assessment: QualityAssessment):
        from coplanner.data_trust_layer import DataTier, TradeQualityTag
        score = assessment.quality_score
        if score >= 0.85:
            return DataTier.USER_CLEAN, TradeQualityTag.PRISTINE
        elif score >= 0.70:
            return DataTier.USER_CLEAN, TradeQualityTag.CLEAN
        elif score >= 0.55:
            return DataTier.USER_PARTIAL, TradeQualityTag.ACCEPTABLE
        elif score >= 0.40:
            return DataTier.USER_PARTIAL, TradeQualityTag.DEVIATED
        elif score >= 0.25:
            return DataTier.USER_EMOTIONAL, TradeQualityTag.EMOTIONAL
        else:
            return DataTier.USER_EMOTIONAL, TradeQualityTag.RANDOM

    # ─────────────────────────────────────────────
    # FEEDBACK GENERATION
    # ─────────────────────────────────────────────

    def _generate_label(self, score: float, deviations: List[DeviationDetail]) -> str:
        has_adaptation = any(d.is_constructive for d in deviations)
        has_emotional = any(
            d.estimated_intent in (
                DeviationIntent.FEAR, DeviationIntent.GREED,
                DeviationIntent.REVENGE, DeviationIntent.FOMO,
                DeviationIntent.BOREDOM,
            )
            for d in deviations
        )

        if score >= 0.85:
            return "Excellent -- clean execution with smart adaptation" if has_adaptation else "Excellent -- disciplined execution"
        elif score >= 0.70:
            return "Good -- reasonable adaptations made" if has_adaptation else "Good -- mostly followed plan"
        elif score >= 0.55:
            if has_adaptation and has_emotional:
                return "Mixed -- some smart moves, some emotional decisions"
            elif has_emotional:
                return "Below average -- emotional signals detected"
            return "Average -- moderate deviations from plan"
        elif score >= 0.40:
            return "Poor -- significant deviations, likely emotional"
        else:
            return "Very poor -- trade appears primarily emotional"

    def _generate_feedback(
        self, score: float, deviations: List[DeviationDetail], trade: TradeRecord,
    ) -> str:
        parts: List[str] = []
        good_deviations = [d for d in deviations if d.is_constructive]
        if good_deviations:
            parts.append("What went well:")
            for d in good_deviations:
                parts.append(f"  + {d.reasoning}")

        concerning = [
            d for d in deviations
            if not d.is_constructive and d.severity > 0.20
            and d.deviation_type != DeviationType.NONE
        ]
        if concerning:
            parts.append("Areas to watch:")
            for d in concerning:
                parts.append(f"  ! {d.reasoning}")

        if score >= 0.75:
            parts.append(f"Overall: Strong execution (score: {score:.2f})")
        elif score >= 0.55:
            parts.append(f"Overall: Acceptable execution (score: {score:.2f})")
        else:
            parts.append(f"Overall: Needs improvement (score: {score:.2f}). Focus on following your plan.")

        return "\n".join(parts)

    def _generate_suggestion(
        self, deviations: List[DeviationDetail], trade: TradeRecord,
    ) -> str:
        worst = None
        for d in deviations:
            if not d.is_constructive and d.severity > 0.20:
                if worst is None or d.severity > worst.severity:
                    worst = d

        if worst is None:
            return "Keep doing what you're doing -- execution is clean."

        suggestions = {
            DeviationIntent.FEAR: (
                "Try pre-defining your exit rules before entering. "
                "When fear hits during a trade, trust your pre-trade analysis."
            ),
            DeviationIntent.GREED: (
                "Consider using partial exits to lock in profits. "
                "This satisfies the desire for more while protecting gains."
            ),
            DeviationIntent.REVENGE: (
                "After losses, take a 30-minute break before next trade. "
                "Your win rate after cooling down is likely much higher."
            ),
            DeviationIntent.FOMO: (
                "Remind yourself: the market gives opportunities every day. "
                "Missing one trade costs nothing. Taking a bad one costs real money."
            ),
            DeviationIntent.BOREDOM: (
                "Quality over quantity. Track your per-trade quality score -- "
                "fewer, better trades almost always outperform many random ones."
            ),
            DeviationIntent.UNCLEAR: (
                "Try adding brief notes to your trades explaining your reasoning. "
                "This helps both you and the system understand your decisions."
            ),
        }

        return suggestions.get(
            worst.estimated_intent,
            "Review this trade's deviation and ask: 'Would I make the same decision again?'",
        )

    def _compare_to_average(self, score: float) -> str:
        if not self._user_baselines:
            return "average"
        avg = self._user_baselines.get("avg_quality", 0.5)
        std = self._user_baselines.get("std_quality", 0.15)
        if score > avg + std:
            return "above_average"
        elif score < avg - std:
            return "below_average"
        return "average"

    # ─────────────────────────────────────────────
    # USER BASELINES (Adaptive)
    # ─────────────────────────────────────────────

    def _compute_user_baselines(self) -> None:
        if not self.db:
            self._user_baselines = None
            return
        try:
            recent = self.db.get_recent_trades(50)
            if len(recent) < 10:
                self._user_baselines = None
                return
            scores = [self._quick_quality_estimate(t) for t in recent]
            avg = sum(scores) / len(scores)
            variance = sum((s - avg) ** 2 for s in scores) / len(scores)
            std = math.sqrt(variance)
            self._user_baselines = {
                "avg_quality": avg,
                "std_quality": std,
                "sample_size": len(scores),
                "best_quality": max(scores),
                "worst_quality": min(scores),
                "recent_trend": self._quality_trend(scores),
            }
            logger.info(
                "User baselines computed: avg=%.2f, std=%.2f, trend=%s",
                avg, std, self._user_baselines["recent_trend"],
            )
        except Exception as e:
            logger.warning("Could not compute user baselines: %s", e)
            self._user_baselines = None

    def _quick_quality_estimate(self, trade: TradeRecord) -> float:
        """Fast quality estimate without full assessment."""
        score = 0.70
        if not trade.stop_loss or trade.stop_loss == 0:
            score -= 0.25
        if not trade.target_price or trade.target_price == 0:
            score -= 0.15

        if trade.outcome == "loss" and trade.stop_loss and trade.entry_price:
            expected = abs(trade.entry_price - trade.stop_loss)
            actual = abs(trade.entry_price - trade.exit_price)
            if expected > 0 and actual > expected * 1.5:
                score -= 0.20

        emotional_words = ["fomo", "revenge", "panic", "angry", "random", "bored"]
        notes = getattr(trade, "notes", "") or ""
        notes = notes.lower()
        for word in emotional_words:
            if word in notes:
                score -= 0.15
                break

        adapt_words = ["adapted", "structure", "risk management", "trailed"]
        for word in adapt_words:
            if word in notes:
                score += 0.05
                break

        return max(0.0, min(1.0, score))

    @staticmethod
    def _quality_trend(scores: List[float]) -> str:
        if len(scores) < 10:
            return "insufficient_data"
        mid = len(scores) // 2
        first_half = sum(scores[:mid]) / mid
        second_half = sum(scores[mid:]) / (len(scores) - mid)
        diff = second_half - first_half
        if diff > 0.08:
            return "improving"
        elif diff < -0.08:
            return "declining"
        return "stable"

    def _user_early_exit_pattern(self) -> Optional[float]:
        """Check if user's early exits have historically been successful."""
        if not self.db:
            return None
        try:
            all_trades = self.db.get_all_completed_trades()
            early_exits = []
            for trade in all_trades:
                if (trade.outcome == "win"
                        and trade.target_price and trade.entry_price and trade.exit_price):
                    potential = abs(trade.target_price - trade.entry_price)
                    actual = abs(trade.exit_price - trade.entry_price)
                    if potential > 0 and actual < potential * 0.70:
                        early_exits.append(trade)
            if len(early_exits) < 5:
                return None
            profitable = sum(1 for t in early_exits if t.actual_rr > 0)
            return profitable / len(early_exits)
        except Exception:
            return None

    # ─────────────────────────────────────────────
    # EXECUTION PROFILE
    # ─────────────────────────────────────────────

    def get_execution_profile(self) -> Dict[str, Any]:
        """Comprehensive profile of user's execution quality."""
        if not self.db:
            return {"error": "No database connected"}

        trades = self.db.get_all_completed_trades()
        if len(trades) < 10:
            return {
                "status": "insufficient_data",
                "trades_analyzed": len(trades),
                "message": "Need at least 10 completed trades for meaningful profile.",
            }

        assessments: List[QualityAssessment] = []
        for trade in trades:
            try:
                assessments.append(self.assess(trade))
            except Exception as e:
                logger.debug("Could not assess trade %s: %s", trade.trade_id, e)

        if not assessments:
            return {"error": "Could not assess any trades"}

        scores = [a.quality_score for a in assessments]
        process_scores = [a.process_score for a in assessments]
        avg_quality = sum(scores) / len(scores)
        avg_process = sum(process_scores) / len(process_scores)

        all_deviations: List[DeviationDetail] = []
        for a in assessments:
            all_deviations.extend(a.deviations)

        intent_counts: Dict[str, int] = {}
        deviation_type_counts: Dict[str, int] = {}
        for d in all_deviations:
            intent_counts[d.estimated_intent.value] = intent_counts.get(d.estimated_intent.value, 0) + 1
            deviation_type_counts[d.deviation_type.value] = deviation_type_counts.get(d.deviation_type.value, 0) + 1

        emotional_intents = [
            DeviationIntent.FEAR, DeviationIntent.GREED,
            DeviationIntent.REVENGE, DeviationIntent.FOMO,
            DeviationIntent.BOREDOM,
        ]
        top_weakness = None
        top_weakness_count = 0
        for intent in emotional_intents:
            count = intent_counts.get(intent.value, 0)
            if count > top_weakness_count:
                top_weakness = intent.value
                top_weakness_count = count

        constructive = sum(1 for d in all_deviations if d.is_constructive)
        disciplined = sum(1 for d in all_deviations if d.estimated_intent == DeviationIntent.DISCIPLINED)

        clean_trades = [a for a in assessments if a.quality_score >= 0.65]
        emotional_trades = [a for a in assessments if a.quality_score < 0.40]

        clean_wr = (
            sum(1 for a in clean_trades if a.outcome_score > 0.5) / len(clean_trades)
            if clean_trades else 0
        )
        emotional_wr = (
            sum(1 for a in emotional_trades if a.outcome_score > 0.5) / len(emotional_trades)
            if emotional_trades else 0
        )

        trend = self._quality_trend(scores) if self._user_baselines else "unknown"

        return {
            "trades_analyzed": len(assessments),
            "average_quality": round(avg_quality, 3),
            "average_process": round(avg_process, 3),
            "quality_trend": trend,
            "execution_grade": self._compute_grade(avg_quality),
            "quality_distribution": {
                "excellent (0.85+)": sum(1 for s in scores if s >= 0.85),
                "good (0.70-0.85)": sum(1 for s in scores if 0.70 <= s < 0.85),
                "average (0.55-0.70)": sum(1 for s in scores if 0.55 <= s < 0.70),
                "poor (0.40-0.55)": sum(1 for s in scores if 0.40 <= s < 0.55),
                "emotional (0.40-)": sum(1 for s in scores if s < 0.40),
            },
            "deviation_intents": intent_counts,
            "deviation_types": deviation_type_counts,
            "constructive_deviations": constructive,
            "disciplined_trades": disciplined,
            "top_weakness": top_weakness,
            "top_weakness_count": top_weakness_count,
            "adaptation_skill": round(constructive / len(all_deviations), 2) if all_deviations else 0,
            "clean_trade_wr": round(clean_wr, 2) if clean_trades else None,
            "emotional_trade_wr": round(emotional_wr, 2) if emotional_trades else None,
            "quality_impact_message": self._quality_impact_message(clean_wr, emotional_wr, clean_trades, emotional_trades),
            "primary_suggestion": self._get_primary_suggestion(top_weakness, avg_quality, trend),
            "growth": {
                "trend": trend,
                "best_period_score": max(scores) if scores else 0,
                "recent_score": sum(scores[-10:]) / min(10, len(scores)) if scores else 0,
                "early_score": sum(scores[:10]) / min(10, len(scores)) if scores else 0,
                "improvement": round(
                    (sum(scores[-10:]) / min(10, len(scores))) - (sum(scores[:10]) / min(10, len(scores))), 3
                ) if len(scores) >= 20 else None,
            },
        }

    @staticmethod
    def _compute_grade(avg_quality: float) -> str:
        if avg_quality >= 0.85:
            return "A+"
        elif avg_quality >= 0.80:
            return "A"
        elif avg_quality >= 0.75:
            return "A-"
        elif avg_quality >= 0.70:
            return "B+"
        elif avg_quality >= 0.65:
            return "B"
        elif avg_quality >= 0.60:
            return "B-"
        elif avg_quality >= 0.55:
            return "C+"
        elif avg_quality >= 0.50:
            return "C"
        elif avg_quality >= 0.45:
            return "C-"
        elif avg_quality >= 0.40:
            return "D"
        return "F"

    def _quality_impact_message(
        self, clean_wr: float, emotional_wr: float,
        clean_trades: list, emotional_trades: list,
    ) -> str:
        if not clean_trades or not emotional_trades:
            return "Not enough data to compare clean vs emotional trade performance."
        if len(clean_trades) < 5 or len(emotional_trades) < 3:
            return "Building comparison data -- need more trades."

        wr_diff = clean_wr - emotional_wr
        if wr_diff > 0.15:
            return (
                f"Your clean trades win {clean_wr:.0%} of the time. "
                f"Your emotional trades win only {emotional_wr:.0%}. "
                f"That's a {wr_diff:.0%} gap -- eliminating emotional trades "
                f"would significantly improve your results."
            )
        elif wr_diff > 0.05:
            return (
                f"Clean trades: {clean_wr:.0%} WR vs Emotional trades: {emotional_wr:.0%} WR. "
                f"Noticeable gap -- emotional trades are dragging you down."
            )
        else:
            return (
                f"Interesting: your clean ({clean_wr:.0%}) and emotional ({emotional_wr:.0%}) "
                f"trade win rates are similar. Your challenge may be more about consistency than emotion."
            )

    def _get_primary_suggestion(
        self, top_weakness: Optional[str], avg_quality: float, trend: str,
    ) -> str:
        if trend == "improving":
            prefix = "You're improving! To accelerate: "
        elif trend == "declining":
            prefix = "Your execution quality has been slipping. Focus on: "
        else:
            prefix = "Key area to improve: "

        suggestions = {
            "fear": f"{prefix}Trust your analysis more. Pre-set your exits and don't touch them unless the STRUCTURE changes.",
            "greed": f"{prefix}Use partial exits -- take 50% at first target, trail the rest.",
            "revenge": f"{prefix}Implement a mandatory 30-min cooldown after losses.",
            "fomo": f"{prefix}Before entering, ask: 'Does this match my setup criteria?'",
            "boredom": f"{prefix}Set a maximum trades-per-day limit. Quality beats quantity.",
        }

        if top_weakness and top_weakness in suggestions:
            return suggestions[top_weakness]

        if avg_quality >= 0.75:
            return "Your execution is strong. Focus on maintaining consistency."

        return f"{prefix}Start adding brief notes to every trade."

    # ─────────────────────────────────────────────
    # EVOLUTION TRACKING
    # ─────────────────────────────────────────────

    def get_evolution_timeline(self, window_size: int = 20) -> List[Dict[str, Any]]:
        """Track how user's execution quality has evolved over time."""
        if not self.db:
            return []

        trades = self.db.get_all_completed_trades()
        if len(trades) < window_size:
            return [{
                "period": "all_time",
                "trades": len(trades),
                "avg_quality": (
                    sum(self._quick_quality_estimate(t) for t in trades) / len(trades) if trades else 0
                ),
                "message": "Not enough trades for timeline. Keep trading!",
            }]

        # Reverse to chronological order
        trades_chrono = list(reversed(trades))
        timeline: List[Dict[str, Any]] = []

        for i in range(0, len(trades_chrono) - window_size + 1, max(1, window_size // 4)):
            window = trades_chrono[i: i + window_size]
            scores = [self._quick_quality_estimate(t) for t in window]
            avg_score = sum(scores) / len(scores)

            emotional = sum(1 for s in scores if s < 0.40)
            clean = sum(1 for s in scores if s >= 0.70)

            start_date = window[0].entry_time
            end_date = window[-1].entry_time

            timeline.append({
                "period_start": start_date.isoformat() if start_date else "",
                "period_end": end_date.isoformat() if end_date else "",
                "trades": len(window),
                "avg_quality": round(avg_score, 3),
                "grade": self._compute_grade(avg_score),
                "clean_pct": round(clean / len(window), 2),
                "emotional_pct": round(emotional / len(window), 2),
            })

        for i in range(1, len(timeline)):
            diff = timeline[i]["avg_quality"] - timeline[i - 1]["avg_quality"]
            if diff > 0.05:
                timeline[i]["trend"] = "improving"
            elif diff < -0.05:
                timeline[i]["trend"] = "declining"
            else:
                timeline[i]["trend"] = "stable"

        if timeline:
            timeline[0]["trend"] = "baseline"

        return timeline

    # ─────────────────────────────────────────────
    # PATTERN DETECTION
    # ─────────────────────────────────────────────

    def detect_behavioral_patterns(self) -> Dict[str, Any]:
        """Detect recurring behavioral patterns in user's trading."""
        if not self.db:
            return {"error": "No database connected"}

        trades = self.db.get_all_completed_trades()
        if len(trades) < 20:
            return {"status": "insufficient_data", "message": "Need 20+ trades to detect patterns."}

        patterns: List[Dict[str, Any]] = []

        for detector in [
            self._detect_time_quality_pattern,
            self._detect_post_loss_pattern,
            self._detect_post_win_pattern,
            self._detect_day_of_week_pattern,
            self._detect_overtrading_pattern,
            self._detect_regime_quality_pattern,
        ]:
            pattern = detector(trades)
            if pattern:
                patterns.append(pattern)

        return {
            "status": "analyzed",
            "trades_analyzed": len(trades),
            "patterns_found": len(patterns),
            "patterns": patterns,
        }

    def _detect_time_quality_pattern(self, trades: List[TradeRecord]) -> Optional[Dict[str, Any]]:
        time_buckets: Dict[int, List[float]] = {}
        for trade in trades:
            if not trade.entry_time:
                continue
            hour = trade.entry_time.hour
            score = self._quick_quality_estimate(trade)
            time_buckets.setdefault(hour, []).append(score)

        hour_averages = {h: sum(s) / len(s) for h, s in time_buckets.items() if len(s) >= 3}
        if not hour_averages:
            return None

        best_hour = max(hour_averages, key=hour_averages.get)
        worst_hour = min(hour_averages, key=hour_averages.get)
        gap = hour_averages[best_hour] - hour_averages[worst_hour]

        if gap >= 0.12:
            return {
                "pattern_type": "TIME_OF_DAY",
                "severity": "MEDIUM" if gap < 0.20 else "HIGH",
                "description": (
                    f"Execution quality varies by time. Best: {best_hour}:00 "
                    f"({hour_averages[best_hour]:.2f}), Worst: {worst_hour}:00 ({hour_averages[worst_hour]:.2f})."
                ),
                "suggestion": f"Focus trading around {best_hour}:00 and be cautious around {worst_hour}:00.",
                "data": {"best_hour": best_hour, "worst_hour": worst_hour, "gap": round(gap, 3)},
            }
        return None

    def _detect_post_loss_pattern(self, trades: List[TradeRecord]) -> Optional[Dict[str, Any]]:
        sorted_trades = sorted([t for t in trades if t.entry_time], key=lambda t: t.entry_time)
        post_loss_scores: List[float] = []
        other_scores: List[float] = []

        for i in range(1, len(sorted_trades)):
            score = self._quick_quality_estimate(sorted_trades[i])
            if sorted_trades[i - 1].outcome == "loss":
                post_loss_scores.append(score)
            else:
                other_scores.append(score)

        if len(post_loss_scores) < 5 or not other_scores:
            return None

        avg_post_loss = sum(post_loss_scores) / len(post_loss_scores)
        avg_normal = sum(other_scores) / len(other_scores)
        drop = avg_normal - avg_post_loss

        if drop >= 0.10:
            return {
                "pattern_type": "POST_LOSS_DETERIORATION",
                "severity": "HIGH" if drop >= 0.18 else "MEDIUM",
                "description": (
                    f"Execution quality drops {drop:.0%} after a loss. "
                    f"Normal: {avg_normal:.2f}, After loss: {avg_post_loss:.2f}."
                ),
                "suggestion": "Implement a cooling period after losses. Even 15-30 minutes helps.",
                "data": {
                    "avg_quality_normal": round(avg_normal, 3),
                    "avg_quality_post_loss": round(avg_post_loss, 3),
                    "quality_drop": round(drop, 3),
                    "sample_size": len(post_loss_scores),
                },
            }
        return None

    def _detect_post_win_pattern(self, trades: List[TradeRecord]) -> Optional[Dict[str, Any]]:
        sorted_trades = sorted([t for t in trades if t.entry_time], key=lambda t: t.entry_time)
        post_streak_scores: List[float] = []
        normal_scores: List[float] = []

        for i in range(3, len(sorted_trades)):
            streak = all(sorted_trades[i - j - 1].outcome == "win" for j in range(3))
            score = self._quick_quality_estimate(sorted_trades[i])
            if streak:
                post_streak_scores.append(score)
            else:
                normal_scores.append(score)

        if len(post_streak_scores) < 3 or not normal_scores:
            return None

        avg_post = sum(post_streak_scores) / len(post_streak_scores)
        avg_normal = sum(normal_scores) / len(normal_scores)
        drop = avg_normal - avg_post

        if drop >= 0.08:
            return {
                "pattern_type": "POST_WIN_OVERCONFIDENCE",
                "severity": "MEDIUM",
                "description": (
                    f"After 3+ consecutive wins, execution quality drops by {drop:.0%}. "
                    f"Possible overconfidence."
                ),
                "suggestion": "After a winning streak, maintain your standard criteria. Wins don't change the market.",
                "data": {"avg_quality_normal": round(avg_normal, 3), "avg_quality_post_streak": round(avg_post, 3)},
            }
        return None

    def _detect_day_of_week_pattern(self, trades: List[TradeRecord]) -> Optional[Dict[str, Any]]:
        day_buckets: Dict[str, List[float]] = {}
        for trade in trades:
            if not trade.entry_time:
                continue
            day = trade.entry_time.strftime("%A")
            day_buckets.setdefault(day, []).append(self._quick_quality_estimate(trade))

        day_averages = {d: sum(s) / len(s) for d, s in day_buckets.items() if len(s) >= 3}
        if len(day_averages) < 3:
            return None

        best_day = max(day_averages, key=day_averages.get)
        worst_day = min(day_averages, key=day_averages.get)
        gap = day_averages[best_day] - day_averages[worst_day]

        if gap >= 0.12:
            return {
                "pattern_type": "DAY_OF_WEEK",
                "severity": "MEDIUM" if gap < 0.20 else "HIGH",
                "description": (
                    f"Quality varies by day. Best: {best_day} ({day_averages[best_day]:.2f}), "
                    f"Worst: {worst_day} ({day_averages[worst_day]:.2f})."
                ),
                "suggestion": f"Be extra disciplined on {worst_day}s, or trade lighter.",
                "data": {d: round(a, 3) for d, a in sorted(day_averages.items(), key=lambda x: x[1], reverse=True)},
            }
        return None

    def _detect_overtrading_pattern(self, trades: List[TradeRecord]) -> Optional[Dict[str, Any]]:
        daily_groups: Dict[str, List[TradeRecord]] = {}
        for trade in trades:
            if not trade.entry_time:
                continue
            date_key = trade.entry_time.date().isoformat()
            daily_groups.setdefault(date_key, []).append(trade)

        count_quality: Dict[str, List[float]] = {}
        for _, day_trades in daily_groups.items():
            count = len(day_trades)
            bucket = "1-2" if count <= 2 else "3-4" if count <= 4 else "5-6" if count <= 6 else "7+"
            scores = [self._quick_quality_estimate(t) for t in day_trades]
            avg_score = sum(scores) / len(scores)
            count_quality.setdefault(bucket, []).append(avg_score)

        bucket_averages = {b: sum(s) / len(s) for b, s in count_quality.items() if len(s) >= 2}

        for low, high in [("1-2", "7+"), ("1-2", "5-6"), ("3-4", "7+")]:
            if low in bucket_averages and high in bucket_averages:
                drop = bucket_averages[low] - bucket_averages[high]
                if drop >= 0.10:
                    return {
                        "pattern_type": "OVERTRADING",
                        "severity": "HIGH" if drop >= 0.18 else "MEDIUM",
                        "description": (
                            f"Quality drops {drop:.0%} on high-trade days ({high}/day) "
                            f"vs low-trade days ({low}/day). Fewer trades = better quality."
                        ),
                        "suggestion": "Consider setting a daily trade limit. Quality beats quantity.",
                        "data": {b: round(a, 3) for b, a in bucket_averages.items()},
                    }
        return None

    def _detect_regime_quality_pattern(self, trades: List[TradeRecord]) -> Optional[Dict[str, Any]]:
        regime_buckets: Dict[str, List[float]] = {}
        for trade in trades:
            regime = (trade.regime_at_entry or "unknown").lower()
            if regime == "unknown":
                continue
            regime_buckets.setdefault(regime, []).append(self._quick_quality_estimate(trade))

        regime_averages = {r: sum(s) / len(s) for r, s in regime_buckets.items() if len(s) >= 5}
        if len(regime_averages) < 2:
            return None

        best = max(regime_averages, key=regime_averages.get)
        worst = min(regime_averages, key=regime_averages.get)
        gap = regime_averages[best] - regime_averages[worst]

        if gap >= 0.10:
            return {
                "pattern_type": "REGIME_EXECUTION_QUALITY",
                "severity": "MEDIUM",
                "description": (
                    f"Execution quality varies by regime. Best in '{best}' ({regime_averages[best]:.2f}), "
                    f"Worst in '{worst}' ({regime_averages[worst]:.2f})."
                ),
                "suggestion": f"In '{worst}' regimes, reduce size or trade less frequently.",
                "data": {r: round(a, 3) for r, a in sorted(regime_averages.items(), key=lambda x: x[1], reverse=True)},
            }
        return None

    # ─────────────────────────────────────────────
    # BATCH OPERATIONS
    # ─────────────────────────────────────────────

    def classify_batch(self, trades: List[TradeRecord]) -> List[QualityAssessment]:
        assessments: List[QualityAssessment] = []
        for trade in trades:
            try:
                assessments.append(self.assess(trade))
            except Exception as e:
                logger.warning("Could not classify trade %s: %s", trade.trade_id, e)
        assessments.sort(key=lambda a: a.quality_score, reverse=True)
        return assessments

    def reclassify_all(self) -> Dict[str, Any]:
        self._classification_cache.clear()
        self._compute_user_baselines()
        if not self.db:
            return {"error": "No database"}
        trades = self.db.get_all_completed_trades()
        assessments = self.classify_batch(trades)
        return {
            "total_classified": len(assessments),
            "avg_quality": round(
                sum(a.quality_score for a in assessments) / len(assessments), 3
            ) if assessments else 0,
            "grade": self._compute_grade(
                sum(a.quality_score for a in assessments) / len(assessments)
            ) if assessments else "N/A",
        }

    def clear_cache(self) -> None:
        self._classification_cache.clear()

    def refresh_baselines(self) -> None:
        self._compute_user_baselines()

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cached_assessments": len(self._classification_cache),
            "baselines_computed": self._user_baselines is not None,
            "baseline_avg_quality": self._user_baselines.get("avg_quality", 0) if self._user_baselines else None,
            "baseline_trend": self._user_baselines.get("recent_trend", "unknown") if self._user_baselines else None,
        }
