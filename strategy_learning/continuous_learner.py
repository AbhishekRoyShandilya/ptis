# strategy_learning/continuous_learner.py
"""
Continuous Learner — Automatically learns from closed trades.

When a trade closes, the continuous learner:
1. Matches the trade to a learned strategy profile (via declared_strategy_id or attribution)
2. Analyzes adherence against the learned profile
3. Records feedback (deviations, strengths, lessons) in TradeFeedback
4. Tracks outcome alignment (win/loss vs. profile expectations)
5. Publishes TRADE_INSIGHTS_READY when enough feedback accumulates

┌─────────────────────────────────────────────────────────────┐
│  ARCHITECTURAL RULE: FEEDBACK ONLY — NO PROFILE MUTATION    │
│                                                             │
│  This module NEVER modifies StrategyProfile (user intent).  │
│  It generates insights and feedback that inform:            │
│   - The Improvement Lab (suggestions for user)              │
│   - The AVT engine (edge discovery)                         │
│   - The coaching system (behavior feedback)                  │
│                                                             │
│  StrategyProfile = USER MIND   (set during onboarding)      │
│  AVT Edge Data  = MARKET TRUTH (discovered from live data)  │
│  They must NEVER overwrite each other.                      │
└─────────────────────────────────────────────────────────────┘

Subscribes to TRADE_CLOSED events on the event bus.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from agents.llm_client import LLMClient, LLMMessage
from config.settings import settings
from core.events import Event, EventType, get_event_bus
from core.models import StrategyProfile, TradeFeedback, Trade

logger = logging.getLogger(__name__)

# ── How many feedbacks before triggering a profile rebuild ──
FEEDBACK_REBUILD_THRESHOLD = 10

FEEDBACK_PROMPT = """You are analyzing a closed trade against the trader's learned strategy profile.

## Strategy Profile: {name}
{narrative}

Entry Criteria: {entry_criteria}
Exit Rules: {exit_rules}
Risk Parameters: {risk_parameters}
Regime Preferences: {regime_preferences}

## Closed Trade
Symbol: {symbol}
Direction: {direction}
Entry: {entry_price} → Exit: {exit_price}
P&L: {pnl} ({pnl_pct}%)
R-Multiple: {r_multiple}
Stop Loss: {stop_loss}
Take Profit: {take_profit}
Entry Time: {entry_time}
Exit Time: {exit_time}
Market Regime: {regime}
Emotion Pre-trade: {emotion_pre}
Conviction: {conviction}/10

## Instructions
Analyze this trade and return JSON:
{{
  "adherence_score": 0-100,
  "deviations": [
    {{"aspect": "entry|stop|exit|regime|timing|risk|sizing", "description": "...", "severity": "low|medium|high"}}
  ],
  "strengths": ["what was done well"],
  "lessons_learned": ["specific, actionable lessons from this trade"],
  "outcome_alignment": "win_as_expected|loss_as_expected|surprise_win|surprise_loss"
}}

Be specific to THIS trade and THIS strategy. Return ONLY JSON:"""


class ContinuousLearner:
    """
    Learns from every closed trade to generate insights and feedback.

    IMPORTANT: This class NEVER modifies StrategyProfile directly.
    It only records TradeFeedback and publishes insight events.
    """

    def __init__(self, db_factory, llm: Optional[LLMClient] = None):
        """
        Args:
            db_factory: callable that returns a new DB session (e.g. get_db generator)
            llm: optional LLM client override
        """
        self._db_factory = db_factory
        self._llm = llm
        self._subscribed = False

        # Per-strategy ContrastLearner instances (AVT edge discovery)
        # These accumulate win/loss feature distributions for the CLE.
        self._contrast_learners: Dict[str, Any] = {}

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(
                provider=settings.LLM_PROVIDER,
                model=settings.LLM_MODEL,
            )
        return self._llm

    def subscribe(self) -> None:
        """Subscribe to TRADE_CLOSED events on the event bus."""
        if self._subscribed:
            return
        bus = get_event_bus()
        bus.subscribe(EventType.TRADE_CLOSED, self._on_trade_closed)
        self._subscribed = True
        logger.info("ContinuousLearner subscribed to TRADE_CLOSED events")

    def _on_trade_closed(self, event: Event) -> None:
        """Handle trade closed event — analyze and record feedback."""
        trade_id = event.trade_id or event.data.get("trade_id")
        if not trade_id:
            return

        try:
            db = next(self._db_factory())
            try:
                self._process_trade(db, trade_id)
            finally:
                db.close()
        except Exception:
            logger.exception("ContinuousLearner failed processing trade %s", trade_id)

    def _process_trade(self, db: Session, trade_id) -> None:
        """Main processing: match trade to profile, analyze, store feedback."""
        trade = db.query(Trade).get(trade_id)
        if not trade:
            return

        # Find matching strategy profile
        profile = self._match_profile(db, trade)
        if not profile:
            logger.debug("No matching profile for trade %s", trade_id)
            return

        # Check if we already analyzed this trade
        existing = db.query(TradeFeedback).filter_by(trade_id=trade.id).first()
        if existing:
            return

        # Analyze trade against profile
        feedback = self._analyze_trade(db, trade, profile)
        if feedback:
            db.add(feedback)
            db.commit()

            # Publish event
            bus = get_event_bus()
            bus.publish(Event(
                event_type=EventType.TRADE_FEEDBACK_RECORDED,
                source="continuous_learner",
                trade_id=trade.id,
                data={
                    "profile_id": str(profile.id),
                    "adherence_score": feedback.adherence_score,
                    "was_profitable": feedback.was_profitable,
                    "outcome_alignment": feedback.outcome_alignment,
                },
            ))

            # Feed trade data to ContrastLearner (AVT edge discovery)
            self._feed_contrast_learner(trade, profile)

            # Check if we should publish insights-ready event
            self._check_insight_threshold(db, profile)

    def _match_profile(self, db: Session, trade: Trade) -> Optional[StrategyProfile]:
        """Find the strategy profile that matches this trade."""
        # 1. Direct match via declared_strategy_id
        if trade.declared_strategy_id:
            profile = db.query(StrategyProfile).filter(
                StrategyProfile.short_code == trade.declared_strategy_id,
                StrategyProfile.status == "active",
            ).first()
            if profile:
                return profile

        # 2. Match via attribution resolved_strategy_id
        if trade.attribution and trade.attribution.resolved_strategy_id:
            profile = db.query(StrategyProfile).filter(
                StrategyProfile.short_code == trade.attribution.resolved_strategy_id,
                StrategyProfile.status == "active",
            ).first()
            if profile:
                return profile

        # 3. If only one active profile, use it (single-strategy trader)
        active_profiles = db.query(StrategyProfile).filter(
            StrategyProfile.status == "active"
        ).all()
        if len(active_profiles) == 1:
            return active_profiles[0]

        return None

    def _analyze_trade(
        self, db: Session, trade: Trade, profile: StrategyProfile
    ) -> Optional[TradeFeedback]:
        """Use LLM to analyze trade against profile and create feedback."""
        prompt = FEEDBACK_PROMPT.format(
            name=profile.name,
            narrative=profile.narrative or "Not yet learned",
            entry_criteria=json.dumps(profile.entry_criteria or [], indent=2),
            exit_rules=json.dumps(profile.exit_rules or {}, indent=2),
            risk_parameters=json.dumps(profile.risk_parameters or {}, indent=2),
            regime_preferences=json.dumps(profile.regime_preferences or {}, indent=2),
            symbol=trade.symbol or "Unknown",
            direction=trade.direction.value if trade.direction else "Unknown",
            entry_price=trade.entry_price or 0,
            exit_price=trade.exit_price or 0,
            pnl=f"{trade.pnl:.2f}" if trade.pnl is not None else "N/A",
            pnl_pct=f"{trade.pnl_pct:.2f}" if trade.pnl_pct is not None else "N/A",
            r_multiple=f"{trade.r_multiple:.2f}" if trade.r_multiple is not None else "N/A",
            stop_loss=trade.stop_loss or "None",
            take_profit=trade.take_profit or "None",
            entry_time=str(trade.entry_time) if trade.entry_time else "Unknown",
            exit_time=str(trade.exit_time) if trade.exit_time else "Unknown",
            regime=trade.market_regime.value if trade.market_regime else "Unknown",
            emotion_pre=trade.emotion_pre.value if trade.emotion_pre else "Unknown",
            conviction=trade.conviction_score or "N/A",
        )

        try:
            from agents.prompts import TRADE_REVIEWER

            llm = self._get_llm()
            _msgs = [
                LLMMessage(role="system", content=TRADE_REVIEWER),
                LLMMessage(role="user", content=prompt),
            ]
            # Synchronous call for event handler context
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        llm.chat(
                            messages=_msgs,
                            temperature=0.2,
                            max_tokens=800,
                            caller="continuous_learner",
                        )
                    ).result()
            else:
                result = asyncio.run(llm.chat(
                    messages=_msgs,
                    temperature=0.2,
                    max_tokens=800,
                    caller="continuous_learner",
                ))

            data = self._parse_llm_response(result.content)
        except Exception:
            logger.exception("LLM analysis failed for trade %s", trade.id)
            # Fallback: basic statistical analysis
            data = self._statistical_fallback(trade, profile)

        feedback = TradeFeedback(
            trade_id=trade.id,
            strategy_profile_id=profile.id,
            adherence_score=data.get("adherence_score", 50),
            deviations=data.get("deviations", []),
            strengths=data.get("strengths", []),
            lessons_learned=data.get("lessons_learned", []),
            outcome_alignment=data.get("outcome_alignment", "unknown"),
            pnl=trade.pnl,
            r_multiple=trade.r_multiple,
            was_profitable=(trade.pnl or 0) > 0,
        )
        return feedback

    def _parse_llm_response(self, raw: str) -> dict:
        """Parse JSON response from LLM."""
        content = raw.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse feedback JSON")
            return {"adherence_score": 50, "outcome_alignment": "unknown"}

    def _statistical_fallback(self, trade: Trade, profile: StrategyProfile) -> dict:
        """Basic analysis when LLM is unavailable."""
        deviations = []
        strengths = []

        risk_params = profile.risk_parameters or {}
        regime_prefs = profile.regime_preferences or {}

        # Check regime alignment
        if trade.market_regime and regime_prefs.get("preferred"):
            regime_val = trade.market_regime.value if hasattr(trade.market_regime, 'value') else str(trade.market_regime)
            if regime_val not in regime_prefs["preferred"]:
                deviations.append({
                    "aspect": "regime",
                    "description": f"Traded in {regime_val}, preferred: {regime_prefs['preferred']}",
                    "severity": "medium",
                })
            else:
                strengths.append(f"Traded in preferred regime: {regime_val}")

        # Check risk sizing
        max_risk = risk_params.get("max_risk_pct", 2.0)
        if trade.stop_loss and trade.entry_price and trade.entry_price > 0:
            actual_risk_pct = abs(trade.entry_price - trade.stop_loss) / trade.entry_price * 100
            if actual_risk_pct > max_risk * 1.5:
                deviations.append({
                    "aspect": "risk",
                    "description": f"Risk {actual_risk_pct:.1f}% exceeds profile max {max_risk}%",
                    "severity": "high",
                })

        # Outcome alignment
        was_profitable = (trade.pnl or 0) > 0
        has_deviations = len(deviations) > 0
        if was_profitable and not has_deviations:
            alignment = "win_as_expected"
        elif not was_profitable and has_deviations:
            alignment = "loss_as_expected"
        elif was_profitable and has_deviations:
            alignment = "surprise_win"
        else:
            alignment = "surprise_loss"

        adherence = max(0, 100 - len(deviations) * 25)

        return {
            "adherence_score": adherence,
            "deviations": deviations,
            "strengths": strengths,
            "lessons_learned": [],
            "outcome_alignment": alignment,
        }

    def _check_insight_threshold(self, db: Session, profile: StrategyProfile) -> None:
        """
        Check if we have enough new feedback to publish an insights-ready event.

        ARCHITECTURAL RULE: This publishes TRADE_INSIGHTS_READY, NOT
        STRATEGY_PROFILE_UPDATED. These insights feed the Improvement Lab
        and coaching system — they do NOT trigger profile rebuilds.

        Profile (user intent) is ONLY modified through explicit
        onboarding sessions, never through trade outcome feedback.
        """
        unprocessed_count = db.query(TradeFeedback).filter(
            TradeFeedback.strategy_profile_id == profile.id,
            TradeFeedback.profile_updated == False,
        ).count()

        if unprocessed_count >= FEEDBACK_REBUILD_THRESHOLD:
            logger.info(
                "Profile %s has %d new feedbacks — publishing insights-ready event "
                "(NOTE: this does NOT trigger profile mutation)",
                profile.name, unprocessed_count,
            )
            bus = get_event_bus()
            bus.publish(Event(
                event_type=EventType.TRADE_INSIGHTS_READY,
                source="continuous_learner",
                data={
                    "profile_id": str(profile.id),
                    "reason": "feedback_threshold_reached",
                    "feedback_count": unprocessed_count,
                    "note": "Insights available for Improvement Lab analysis. "
                            "DO NOT use to modify StrategyProfile.",
                },
            ))

            # Mark feedbacks as processed (for threshold tracking only)
            db.query(TradeFeedback).filter(
                TradeFeedback.strategy_profile_id == profile.id,
                TradeFeedback.profile_updated == False,
            ).update({"profile_updated": True})
            db.commit()

    # ── Contrast Learner Feed (AVT Edge Discovery) ────────────

    def _feed_contrast_learner(
        self, trade: Trade, profile: StrategyProfile
    ) -> None:
        """
        Feed trade features and outcome to the ContrastLearner.

        This builds the win/loss feature distributions that the CLE
        uses to discover what actually drives outcomes. The CLE's
        insights then feed into self_improver.py → prove-it cycle.

        ARCHITECTURAL NOTE: This feeds INTO AVT's edge discovery.
        It does NOT modify the user's StrategyProfile.
        """
        try:
            from avt.contrast_learner import ContrastLearner, OutcomeExample
        except ImportError:
            return  # CLE not available

        strategy_key = str(profile.id)

        # Get or create CLE instance for this strategy
        if strategy_key not in self._contrast_learners:
            self._contrast_learners[strategy_key] = ContrastLearner(
                strategy_id=strategy_key
            )

        cle = self._contrast_learners[strategy_key]

        # Extract trade features for contrast analysis
        features = self._extract_trade_features(trade)
        if not features:
            return

        # Compute R-multiple outcome
        r_multiple = float(trade.r_multiple) if trade.r_multiple else 0.0

        # Determine regime (if available from trade metadata)
        regime = None
        if hasattr(trade, 'metadata') and isinstance(trade.metadata, dict):
            regime = trade.metadata.get('regime')

        example = OutcomeExample(
            features=features,
            outcome=r_multiple,
            regime=regime,
            timestamp=trade.exit_time or trade.entry_time,
            symbol=trade.symbol,
            strategy_id=strategy_key,
        )

        cle.add_example(example)
        logger.debug(
            "Fed trade %s to ContrastLearner (strategy %s, %s, %.2fR). "
            "CLE total: %d examples",
            trade.id, profile.name, example.label, r_multiple,
            len(cle.examples),
        )

    @staticmethod
    def _extract_trade_features(trade: Trade) -> dict:
        """
        Extract numerical features from a trade for contrast analysis.

        These features are what the CLE compares between wins and losses
        to discover outcome drivers.
        """
        features = {}
        try:
            if trade.entry_price and trade.stop_loss:
                entry = float(trade.entry_price)
                stop = float(trade.stop_loss)
                if entry > 0:
                    features["stop_distance_pct"] = abs(entry - stop) / entry

            if trade.entry_price and trade.exit_price:
                entry = float(trade.entry_price)
                exit_p = float(trade.exit_price)
                if entry > 0:
                    features["move_pct"] = (exit_p - entry) / entry

            if hasattr(trade, 'hold_duration_minutes') and trade.hold_duration_minutes:
                features["hold_duration"] = float(trade.hold_duration_minutes)

            if trade.r_multiple:
                features["r_multiple"] = float(trade.r_multiple)

            # Extract any stored feature metadata
            if hasattr(trade, 'metadata') and isinstance(trade.metadata, dict):
                for key, val in trade.metadata.items():
                    if isinstance(val, (int, float)) and key not in (
                        'pnl', 'commission'
                    ):
                        features[key] = float(val)

        except (TypeError, ValueError, AttributeError):
            pass

        return features

    def get_contrast_learner(self, strategy_id: str) -> Optional[Any]:
        """
        Get the ContrastLearner instance for a strategy.

        Used by self_improver.py to access CLE insights:
            cle = continuous_learner.get_contrast_learner(strategy_id)
            if cle and cle.has_sufficient_data:
                adjustments = cle.suggest_weight_adjustments(current_weights)
        """
        return self._contrast_learners.get(strategy_id)

    def get_profile_insights(self, db: Session, profile_id: UUID) -> dict:
        """Get aggregated insights from all feedback for a profile."""
        feedbacks = db.query(TradeFeedback).filter(
            TradeFeedback.strategy_profile_id == profile_id,
        ).order_by(TradeFeedback.created_at.desc()).all()

        if not feedbacks:
            return {"total_trades": 0, "message": "No trades analyzed yet"}

        total = len(feedbacks)
        wins = sum(1 for f in feedbacks if f.was_profitable)
        avg_adherence = sum(f.adherence_score or 0 for f in feedbacks) / total

        # Common deviations
        deviation_counts: dict[str, int] = {}
        for f in feedbacks:
            for d in (f.deviations or []):
                aspect = d.get("aspect", "unknown")
                deviation_counts[aspect] = deviation_counts.get(aspect, 0) + 1

        # Outcome alignment breakdown
        alignments: dict[str, int] = {}
        for f in feedbacks:
            a = f.outcome_alignment or "unknown"
            alignments[a] = alignments.get(a, 0) + 1

        # Recent lessons
        recent_lessons = []
        for f in feedbacks[:5]:
            recent_lessons.extend(f.lessons_learned or [])

        # Adherence trend (last 10 vs. first 10)
        if total >= 10:
            early = feedbacks[-10:]
            recent = feedbacks[:10]
            early_avg = sum(f.adherence_score or 0 for f in early) / 10
            recent_avg = sum(f.adherence_score or 0 for f in recent) / 10
            trend = "improving" if recent_avg > early_avg + 3 else (
                "declining" if recent_avg < early_avg - 3 else "stable"
            )
        else:
            trend = "insufficient_data"

        return {
            "total_trades": total,
            "win_rate": wins / total if total else 0,
            "avg_adherence": round(avg_adherence, 1),
            "adherence_trend": trend,
            "common_deviations": sorted(
                deviation_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "outcome_alignment": alignments,
            "recent_lessons": recent_lessons[:10],
            "avg_r_multiple": round(
                sum(f.r_multiple or 0 for f in feedbacks) / total, 2
            ) if total else 0,
        }


# ── Module-level singleton ──
_learner: ContinuousLearner | None = None


def get_continuous_learner(db_factory=None) -> ContinuousLearner:
    global _learner
    if _learner is None:
        if db_factory is None:
            from core.database import get_db
            db_factory = get_db
        _learner = ContinuousLearner(db_factory)
    return _learner
