# strategy_learning/session_manager.py
"""
Learning Session Manager — full lifecycle for teaching sessions.

A teaching session is where a trader uploads chart screenshots, annotates
them, and teaches the system their strategy.  Multiple sessions gradually
build up the StrategyProfile.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.events import Event, EventType, get_event_bus
from core.models import LearningExample, LearningSession, StrategyProfile
from strategy_learning.models import UserAnnotation, VisionAnalysis

logger = logging.getLogger(__name__)

_VALID_SESSION_TYPES = {
    "initial_teaching",
    "refinement",
    "counter_examples",
    "regime_teaching",
    "exit_teaching",
    "outcome_review",
    "variant_teaching",
    "variant_refinement",
}

# Guard against accidental mega-sessions
_MAX_EXAMPLES_PER_SESSION = 50


class LearningSessionManager:
    """Manages teaching-session lifecycle and example storage."""

    def __init__(self, db: Session):
        self.db = db

    # ── session lifecycle ────────────────────────────────────

    def start_session(
        self,
        profile_id: UUID,
        session_type: str = "initial_teaching",
        notes: str = "",
    ) -> LearningSession:
        """Start a new teaching session."""
        if session_type not in _VALID_SESSION_TYPES:
            raise ValueError(
                f"Invalid session_type '{session_type}'. "
                f"Allowed: {sorted(_VALID_SESSION_TYPES)}"
            )

        profile = self.db.get(StrategyProfile, profile_id)
        if profile is None:
            raise ValueError(f"Strategy profile not found: {profile_id}")

        # Atomically check for an existing active session using FOR UPDATE
        active = self.db.execute(
            select(LearningSession)
            .where(
                LearningSession.strategy_profile_id == profile_id,
                LearningSession.status == "active",
            )
            .with_for_update(skip_locked=True)
        ).scalar_one_or_none()

        if active is not None:
            raise ValueError(
                f"Profile already has active session {active.id}. "
                "Complete or abandon it first."
            )

        session = LearningSession(
            id=uuid4(),
            strategy_profile_id=profile_id,
            session_type=session_type,
            status="active",
            notes=notes,
            examples_count=0,
        )
        self.db.add(session)

        if profile.status == "draft":
            profile.status = "learning"

        self.db.commit()
        self.db.refresh(session)

        logger.info(
            "Started session %s [%s] for profile '%s'",
            session.id, session_type, profile.name,
        )
        return session

    def complete_session(
        self,
        session_id: UUID,
        ai_summary: str = "",
    ) -> LearningSession:
        """Mark session complete and update profile counters."""
        session = self._get_active_session(session_id)

        if session.examples_count == 0:
            raise ValueError(
                "Cannot complete a session with zero examples. "
                "Add at least one example or abandon the session."
            )

        session.status = "completed"
        session.completed_at = datetime.now(timezone.utc)
        session.ai_summary = ai_summary

        # Update profile counters
        profile = self.db.get(StrategyProfile, session.strategy_profile_id)
        if profile:
            # Recount from source of truth rather than incrementing
            total_examples = self.db.execute(
                select(func.count(LearningExample.id))
                .join(LearningSession)
                .where(
                    LearningExample.strategy_profile_id == profile.id,
                    LearningSession.status.in_(["completed"]),
                )
            ).scalar_one()

            total_sessions = self.db.execute(
                select(func.count(LearningSession.id)).where(
                    LearningSession.strategy_profile_id == profile.id,
                    LearningSession.status == "completed",
                )
            ).scalar_one()

            profile.examples_count = total_examples
            profile.sessions_count = total_sessions

        self.db.commit()
        self.db.refresh(session)

        get_event_bus().publish(Event(
            event_type=EventType.LEARNING_SESSION_COMPLETED,
            source="session_manager",
            strategy_id=str(session.strategy_profile_id),
            data={
                "session_id": str(session_id),
                "examples_count": session.examples_count,
                "session_type": session.session_type,
            },
        ))

        logger.info(
            "Completed session %s — %d examples",
            session_id, session.examples_count,
        )
        return session

    def abandon_session(self, session_id: UUID) -> None:
        session = self._get_active_session(session_id)
        session.status = "abandoned"
        session.completed_at = datetime.now(timezone.utc)
        self.db.commit()
        logger.info("Abandoned session %s", session_id)

    # ── examples ─────────────────────────────────────────────

    def add_example(
        self,
        session_id: UUID,
        screenshot_path: str,
        screenshot_hash: str,
        vision_analysis: VisionAnalysis,
        user_annotations: Optional[UserAnnotation] = None,
        is_positive: bool = True,
        market_context: Optional[dict[str, Any]] = None,
    ) -> LearningExample:
        """Add a learning example to an active session."""
        session = self._get_active_session(session_id)

        if (session.examples_count or 0) >= _MAX_EXAMPLES_PER_SESSION:
            raise ValueError(
                f"Session has reached the maximum of {_MAX_EXAMPLES_PER_SESSION} examples. "
                "Complete this session and start a new one."
            )

        # Duplicate detection within profile
        existing = self.db.execute(
            select(LearningExample.id).where(
                LearningExample.strategy_profile_id == session.strategy_profile_id,
                LearningExample.screenshot_hash == screenshot_hash,
            )
        ).scalar_one_or_none()

        if existing is not None:
            raise ValueError(
                f"Duplicate screenshot (hash={screenshot_hash[:12]}…). "
                f"Already stored as example {existing}."
            )

        ai_interpretation = self._build_interpretation(vision_analysis, user_annotations)

        example = LearningExample(
            id=uuid4(),
            session_id=session_id,
            strategy_profile_id=session.strategy_profile_id,
            screenshot_path=screenshot_path,
            screenshot_hash=screenshot_hash,
            vision_analysis=vision_analysis.model_dump(),
            user_annotations=user_annotations.model_dump() if user_annotations else {},
            ai_interpretation=ai_interpretation,
            is_positive_example=is_positive,
            market_context=market_context or {},
        )
        self.db.add(example)

        session.examples_count = (session.examples_count or 0) + 1
        self.db.commit()
        self.db.refresh(example)

        get_event_bus().publish(Event(
            event_type=EventType.LEARNING_EXAMPLE_ADDED,
            source="session_manager",
            strategy_id=str(session.strategy_profile_id),
            data={
                "session_id": str(session_id),
                "example_id": str(example.id),
                "is_positive": is_positive,
                "total_in_session": session.examples_count,
            },
        ))

        logger.info(
            "Added example %s to session %s (%d in session)",
            example.id, session_id, session.examples_count,
        )
        return example

    def update_annotations(
        self,
        example_id: UUID,
        annotations: UserAnnotation,
    ) -> LearningExample:
        """Update or add annotations on an existing example."""
        example = self.db.get(LearningExample, example_id)
        if example is None:
            raise ValueError(f"Example not found: {example_id}")

        example.user_annotations = annotations.model_dump()

        # Rebuild interpretation with new annotations
        vision = (
            VisionAnalysis(**example.vision_analysis)
            if example.vision_analysis
            else None
        )
        example.ai_interpretation = self._build_interpretation(vision, annotations)

        self.db.commit()
        self.db.refresh(example)
        return example

    # ── queries ───────────────────────────────────────────────

    def get_session(self, session_id: UUID) -> Optional[LearningSession]:
        return self.db.get(LearningSession, session_id)

    def get_session_examples(self, session_id: UUID) -> list[LearningExample]:
        result = self.db.execute(
            select(LearningExample)
            .where(LearningExample.session_id == session_id)
            .order_by(LearningExample.created_at)
        )
        return list(result.scalars().all())

    def get_profile_examples(
        self,
        profile_id: UUID,
        *,
        positive_only: bool = False,
    ) -> list[LearningExample]:
        """All examples from completed sessions for a profile."""
        stmt = (
            select(LearningExample)
            .join(LearningSession)
            .where(
                LearningExample.strategy_profile_id == profile_id,
                LearningSession.status == "completed",
            )
        )
        if positive_only:
            stmt = stmt.where(LearningExample.is_positive_example.is_(True))
        stmt = stmt.order_by(LearningExample.created_at)
        return list(self.db.execute(stmt).scalars().all())

    def get_active_session_for_profile(
        self, profile_id: UUID
    ) -> Optional[LearningSession]:
        return self.db.execute(
            select(LearningSession).where(
                LearningSession.strategy_profile_id == profile_id,
                LearningSession.status == "active",
            )
        ).scalar_one_or_none()

    # ── internals ─────────────────────────────────────────────

    def _get_active_session(self, session_id: UUID) -> LearningSession:
        session = self.db.get(LearningSession, session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        if session.status != "active":
            raise ValueError(
                f"Session {session_id} is '{session.status}', expected 'active'"
            )
        return session

    def _build_interpretation(
        self,
        vision: Optional[VisionAnalysis],
        annotations: Optional[UserAnnotation],
    ) -> str:
        """
        Merge vision analysis and user annotations into a single
        narrative paragraph that serves as the primary text for
        embedding and LLM context.
        """
        parts: list[str] = []

        # ── vision-derived context ──
        if vision:
            if vision.overall_context:
                parts.append(f"Chart context: {vision.overall_context}")
            if vision.trend_direction:
                parts.append(f"Trend is {vision.trend_direction}.")
            if vision.market_structure:
                parts.append(f"Market structure: {vision.market_structure}.")
            if vision.candle_patterns:
                pats = "; ".join(
                    f"{cp.pattern} ({cp.significance})"
                    for cp in vision.candle_patterns
                )
                parts.append(f"Candle patterns observed: {pats}.")
            if vision.volume_profile:
                parts.append(f"Volume behaviour: {vision.volume_profile}.")
            if vision.price_levels:
                lvls = ", ".join(
                    f"{pl.label} at {pl.level}" for pl in vision.price_levels
                )
                parts.append(f"Key levels: {lvls}.")
            if vision.key_observations:
                parts.append(
                    "Additional observations: "
                    + "; ".join(vision.key_observations)
                    + "."
                )

        # ── user-supplied context ──
        if annotations:
            if annotations.direction:
                parts.append(f"Trader's bias: {annotations.direction.upper()}.")
            if annotations.entry_price is not None:
                parts.append(f"Planned entry at {annotations.entry_price}.")
            if annotations.stop_price is not None:
                parts.append(f"Stop at {annotations.stop_price}.")
            if annotations.target_prices:
                targets_str = ", ".join(str(t) for t in annotations.target_prices)
                parts.append(f"Targets: {targets_str}.")
            if annotations.reasoning:
                parts.append(f"Reasoning: {annotations.reasoning}")
            if annotations.what_makes_this_work:
                parts.append(f"Edge: {annotations.what_makes_this_work}")
            if annotations.what_would_invalidate:
                parts.append(f"Invalidation: {annotations.what_would_invalidate}")
            if annotations.setup_quality:
                parts.append(f"Setup quality rated: {annotations.setup_quality}.")
            rr = annotations.risk_reward_ratio
            if rr is not None:
                parts.append(f"Risk-reward ratio: {rr:.1f}R.")

        return " ".join(parts) if parts else "No interpretation available."