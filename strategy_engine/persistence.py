# strategy_engine/persistence.py
"""
Persists attribution results, condition evaluations, override events,
and behavioral events to the database.

Bridges the attribution engine output to the ORM layer.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from core.enums import (
    BehavioralEventType,
    EmotionalState,
    MarketRegime,
    OverrideSeverity,
    OverrideType,
    SessionType,
)
from core.models import (
    BehavioralEvent,
    ConditionEvaluation,
    OverrideEvent,
    StrategyAttribution,
)
from strategy_engine.attribution import AttributionResult
from strategy_engine.override_detector import OverrideDetection, OverrideReport, SessionContext

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  OVERRIDE → BEHAVIORAL EVENT MAPPING
# ══════════════════════════════════════════════

OVERRIDE_TO_BEHAVIORAL: dict[OverrideType, BehavioralEventType] = {
    OverrideType.RAPID_RE_ENTRY: BehavioralEventType.RAPID_RE_ENTRY,
    OverrideType.OVERSIZING: BehavioralEventType.LOSS_CHASING,
    OverrideType.NO_STOP_SET: BehavioralEventType.IMPULSE_ENTRY,
    OverrideType.STOP_MOVED: BehavioralEventType.STOP_WIDENING,
}


# ══════════════════════════════════════════════
#  PERSISTENCE SERVICE
# ══════════════════════════════════════════════

class AttributionPersistence:
    """
    Writes complete attribution pipeline results to PostgreSQL.
    Creates linked records across strategy_attributions,
    condition_evaluations, override_events, and behavioral_events.
    """

    def persist(
        self,
        db: Session,
        trade_id: str,
        result: AttributionResult,
        session_context: SessionContext | None = None,
        emotional_state: EmotionalState | None = None,
        session_type: SessionType | None = None,
    ) -> StrategyAttribution:
        """
        Persist full attribution result to DB.
        Returns the created StrategyAttribution record.
        """
        trade_uuid = uuid.UUID(trade_id)

        # ═══════════════════════════════
        # 1. Strategy Attribution record
        # ═══════════════════════════════
        tier2_matches_json = []
        if result.tier2_output:
            tier2_matches_json = [
                m.to_dict() for m in result.tier2_output.matches
            ]

        tier3_summary = {}
        if result.tier3_override_report:
            tier3_summary = result.tier3_override_report.summary()

        attribution = StrategyAttribution(
            id=uuid.uuid4(),
            trade_id=trade_uuid,

            # Tier 1
            tier1_strategy_id=result.tier1_strategy_id,
            tier1_declared=result.tier1_declared,

            # Tier 2
            tier2_matches=tier2_matches_json,
            tier2_best_match_id=result.tier2_best_match_id,
            tier2_best_match_score=result.tier2_best_match_score,

            # Tier 3
            tier3_override_count=(
                result.tier3_override_report.total_count
                if result.tier3_override_report else 0
            ),
            tier3_max_severity=(
                result.tier3_override_report.max_severity
                if result.tier3_override_report else None
            ),
            tier3_summary=tier3_summary,

            # Resolved
            resolved_strategy_id=result.resolved_strategy_id,
            resolved_tier=result.resolved_tier,
            confidence_score=result.confidence_score,
            discipline_score=result.discipline_score,

            # Context
            market_regime_at_entry=(
                MarketRegime(result.tier2_output.matches[0].to_dict().get("regime"))
                if False else None  # Will be set from trade context
            ),
            evaluation_context={
                "insights": result.insights,
                "strategies_evaluated": (
                    result.tier2_output.strategies_evaluated
                    if result.tier2_output else 0
                ),
                "valid_matches": (
                    result.tier2_output.valid_matches
                    if result.tier2_output else 0
                ),
            },
        )

        db.add(attribution)
        db.flush()  # get attribution.id for FK references

        logger.info(
            "Persisted attribution %s for trade %s (strategy=%s, discipline=%.1f)",
            attribution.id, trade_id, result.resolved_strategy_id, result.discipline_score,
        )

        # ═══════════════════════════════
        # 2. Condition Evaluations
        # ═══════════════════════════════
        if result.tier2_output:
            self._persist_condition_evaluations(
                db, trade_uuid, result.tier2_output,
            )

        # ═══════════════════════════════
        # 3. Override Events
        # ═══════════════════════════════
        if result.tier3_override_report and result.tier3_override_report.has_overrides:
            self._persist_override_events(
                db, trade_uuid, attribution.id, result.tier3_override_report,
            )

        # ═══════════════════════════════
        # 4. Behavioral Events
        # ═══════════════════════════════
        self._persist_behavioral_events(
            db, trade_uuid, result, session_context, emotional_state, session_type,
        )

        db.commit()
        return attribution

    def _persist_condition_evaluations(
        self,
        db: Session,
        trade_uuid: uuid.UUID,
        matcher_output: Any,
    ) -> int:
        """Persist all condition evaluation results from Tier 2."""
        count = 0

        for match in matcher_output.matches:
            for cond_eval in match.condition_results:
                record = ConditionEvaluation(
                    id=uuid.uuid4(),
                    trade_id=trade_uuid,
                    strategy_id=match.strategy_id,
                    condition_type=cond_eval.result.condition_type,
                    condition_params=cond_eval.condition_def.params,
                    required=cond_eval.condition_def.required,
                    weight=cond_eval.condition_def.weight,
                    status=cond_eval.result.status,
                    confidence=cond_eval.result.confidence,
                    actual_value=cond_eval.result.actual_value,
                    expected_value=cond_eval.result.expected_value,
                    details=cond_eval.result.details,
                )
                db.add(record)
                count += 1

        logger.debug("Persisted %d condition evaluations for trade %s", count, trade_uuid)
        return count

    def _persist_override_events(
        self,
        db: Session,
        trade_uuid: uuid.UUID,
        attribution_id: uuid.UUID,
        report: OverrideReport,
    ) -> int:
        """Persist all override events from Tier 3."""
        count = 0

        for override in report.overrides:
            record = OverrideEvent(
                id=uuid.uuid4(),
                trade_id=trade_uuid,
                attribution_id=attribution_id,
                override_type=override.override_type,
                severity=override.severity,
                severity_score=override.severity_score,
                description=override.description,
                rule_expected=override.rule_expected,
                actual_value=override.actual_value,
                expected_value=override.expected_value,
                strategy_id=override.strategy_id,
                context=override.context,
            )
            db.add(record)
            count += 1

        logger.debug(
            "Persisted %d override events for trade %s (max_severity=%s)",
            count, trade_uuid, report.max_severity,
        )
        return count

    def _persist_behavioral_events(
        self,
        db: Session,
        trade_uuid: uuid.UUID,
        result: AttributionResult,
        session_context: SessionContext | None,
        emotional_state: EmotionalState | None,
        session_type: SessionType | None,
    ) -> int:
        """
        Derive behavioral events from override analysis and persist them.
        Also creates positive events for clean executions.
        """
        count = 0
        now = datetime.now(timezone.utc)
        session_date = now.replace(hour=0, minute=0, second=0, microsecond=0)

        override_report = result.tier3_override_report

        # ── Negative events from overrides ──
        if override_report and override_report.has_overrides:
            derived = self._derive_behavioral_events(
                trade_uuid, override_report, session_context, emotional_state, session_type,
            )
            for event in derived:
                db.add(event)
                count += 1

        # ── Positive event for clean execution ──
        if (
            override_report
            and not override_report.has_overrides
            and result.resolved_strategy_id
        ):
            clean_event = BehavioralEvent(
                id=uuid.uuid4(),
                trade_id=trade_uuid,
                session_date=session_date,
                event_type=BehavioralEventType.CLEAN_EXECUTION,
                severity_score=0.0,
                is_positive=True,
                description=(
                    f"Clean execution of '{result.resolved_strategy_id}' — "
                    f"all rules followed, discipline score {result.discipline_score:.0f}"
                ),
                trigger_conditions={
                    "strategy_id": result.resolved_strategy_id,
                    "confidence": result.confidence_score,
                    "discipline_score": result.discipline_score,
                },
                emotional_state=emotional_state,
                market_regime=None,
                session_type=session_type,
                pnl_at_detection=session_context.daily_pnl if session_context else None,
                trades_today_at_detection=(
                    session_context.trades_today if session_context else None
                ),
            )
            db.add(clean_event)
            count += 1

        # ── High discipline hold event (discipline > 90 with multiple trades) ──
        if (
            session_context
            and session_context.trades_today >= 3
            and result.discipline_score >= 90
        ):
            discipline_event = BehavioralEvent(
                id=uuid.uuid4(),
                trade_id=trade_uuid,
                session_date=session_date,
                event_type=BehavioralEventType.DISCIPLINE_HOLD,
                severity_score=0.0,
                is_positive=True,
                description=(
                    f"Maintained high discipline ({result.discipline_score:.0f}) "
                    f"across {session_context.trades_today} trades today"
                ),
                trigger_conditions={
                    "discipline_score": result.discipline_score,
                    "trades_today": session_context.trades_today,
                },
                emotional_state=emotional_state,
                session_type=session_type,
                pnl_at_detection=session_context.daily_pnl,
                trades_today_at_detection=session_context.trades_today,
            )
            db.add(discipline_event)
            count += 1

        logger.debug("Persisted %d behavioral events for trade %s", count, trade_uuid)
        return count

    def _derive_behavioral_events(
        self,
        trade_uuid: uuid.UUID,
        report: OverrideReport,
        session_context: SessionContext | None,
        emotional_state: EmotionalState | None,
        session_type: SessionType | None,
    ) -> list[BehavioralEvent]:
        """
        Convert significant overrides into behavioral events.
        Not all overrides produce behavioral events — only those
        indicating behavioral patterns (vs mechanical errors).
        """
        events = []
        now = datetime.now(timezone.utc)
        session_date = now.replace(hour=0, minute=0, second=0, microsecond=0)

        for override in report.overrides:
            behavioral_type = OVERRIDE_TO_BEHAVIORAL.get(override.override_type)
            if not behavioral_type:
                # Check severity — high/critical overrides always create events
                if override.severity not in (OverrideSeverity.HIGH, OverrideSeverity.CRITICAL):
                    continue
                behavioral_type = BehavioralEventType.IMPULSE_ENTRY

            trigger_conditions = dict(override.context)
            if session_context:
                trigger_conditions.update({
                    "consecutive_losses": session_context.consecutive_losses,
                    "daily_pnl": session_context.daily_pnl,
                    "trades_today": session_context.trades_today,
                })

            event = BehavioralEvent(
                id=uuid.uuid4(),
                trade_id=trade_uuid,
                session_date=session_date,
                event_type=behavioral_type,
                severity_score=override.severity_score,
                is_positive=False,
                description=override.description,
                trigger_conditions=trigger_conditions,
                emotional_state=emotional_state,
                market_regime=None,
                session_type=session_type,
                pnl_at_detection=session_context.daily_pnl if session_context else None,
                trades_today_at_detection=(
                    session_context.trades_today if session_context else None
                ),
            )
            events.append(event)

        # ── Compound behavioral detections ──

        # Revenge trading: rapid re-entry + consecutive losses >= 2
        rapid_overrides = [
            o for o in report.overrides
            if o.override_type == OverrideType.RAPID_RE_ENTRY
        ]
        if rapid_overrides and session_context and session_context.consecutive_losses >= 2:
            events.append(BehavioralEvent(
                id=uuid.uuid4(),
                trade_id=trade_uuid,
                session_date=session_date,
                event_type=BehavioralEventType.REVENGE_TRADE,
                severity_score=1.0,
                is_positive=False,
                description=(
                    f"Revenge trade detected: rapid re-entry after "
                    f"{session_context.consecutive_losses} consecutive losses"
                ),
                trigger_conditions={
                    "consecutive_losses": session_context.consecutive_losses,
                    "daily_pnl": session_context.daily_pnl,
                },
                emotional_state=emotional_state,
                session_type=session_type,
                pnl_at_detection=session_context.daily_pnl,
                trades_today_at_detection=session_context.trades_today,
            ))

        # Overtrade: daily limit exceeded
        # strategy_engine/persistence.py (continued from _derive_behavioral_events)

        # Overtrade: daily limit exceeded
        daily_limit_overrides = [
            o for o in report.overrides
            if "Daily trade limit" in o.description
        ]
        if daily_limit_overrides:
            events.append(BehavioralEvent(
                id=uuid.uuid4(),
                trade_id=trade_uuid,
                session_date=session_date,
                event_type=BehavioralEventType.OVERTRADE,
                severity_score=0.8,
                is_positive=False,
                description=(
                    f"Overtrading: exceeded daily trade limit "
                    f"({session_context.trades_today + 1 if session_context else '?'} trades)"
                ),
                trigger_conditions={
                    "trades_today": session_context.trades_today if session_context else None,
                    "daily_pnl": session_context.daily_pnl if session_context else None,
                },
                emotional_state=emotional_state,
                session_type=session_type,
                pnl_at_detection=session_context.daily_pnl if session_context else None,
                trades_today_at_detection=(
                    session_context.trades_today if session_context else None
                ),
            ))

        # FOMO: missing confirmations + high number of trades + entry outside time window
        missing_conf = [
            o for o in report.overrides
            if o.override_type == OverrideType.MISSING_CONFIRMATION
        ]
        time_violation = [
            o for o in report.overrides
            if o.override_type in (
                OverrideType.EARLY_ENTRY,
                OverrideType.TIME_WINDOW_VIOLATION,
            )
        ]
        if missing_conf and time_violation:
            events.append(BehavioralEvent(
                id=uuid.uuid4(),
                trade_id=trade_uuid,
                session_date=session_date,
                event_type=BehavioralEventType.FOMO_ENTRY,
                severity_score=0.9,
                is_positive=False,
                description=(
                    f"FOMO entry suspected: {len(missing_conf)} missing confirmations "
                    f"combined with time window violation"
                ),
                trigger_conditions={
                    "missing_confirmations": len(missing_conf),
                    "time_violations": len(time_violation),
                },
                emotional_state=emotional_state,
                session_type=session_type,
                pnl_at_detection=session_context.daily_pnl if session_context else None,
                trades_today_at_detection=(
                    session_context.trades_today if session_context else None
                ),
            ))

        return events