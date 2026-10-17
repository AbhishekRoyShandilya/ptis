# core/trade_pipeline.py
"""
Central trade pipeline — wires together:
  Trade ingestion → Attribution → Shadow creation → Event publishing → DB persistence

This is the single entry point for all new trades entering the system.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.orm import Session as DBSession

from core.enums import (
    EmotionalState,
    MarketRegime,
    SessionType,
    TradeDirection,
    TradeStatus,
)
from core.events import Event, EventType, get_event_bus
from core.models import Trade
from core.session_manager import get_session_manager
from strategy_engine.attribution import AttributionEngine, AttributionResult
from strategy_engine.conditions.base_condition import MarketSnapshot, TradeContext
from strategy_engine.override_detector import SessionContext
from strategy_engine.persistence import AttributionPersistence
from strategy_engine.strategy_registry import get_registry
from analytics.decision_quality import DecisionQualityEngine

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  TRADE INPUT DTO
# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
#  TRADE INPUT DTO
class TradeInput:
    """Structured input for the trade pipeline."""

    def __init__(
        self,
        symbol: str,
        direction: TradeDirection,
        entry_price: float,
        position_size: float,
        entry_time: Optional[datetime] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        declared_strategy_id: Optional[str] = None,
        emotion_pre: Optional[EmotionalState] = None,
        conviction_score: Optional[float] = None,
        notes: Optional[str] = None,
        screenshot_path: Optional[str] = None,
        account_balance: Optional[float] = None,
        source: str = "manual",
        broker: Optional[str] = None,
        external_id: Optional[str] = None,
        trader_id: Optional[str] = None,
    ):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.position_size = position_size
        self.entry_time = entry_time or datetime.now(timezone.utc)
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.declared_strategy_id = declared_strategy_id
        self.emotion_pre = emotion_pre
        self.conviction_score = conviction_score
        self.notes = notes
        self.screenshot_path = screenshot_path
        self.account_balance = account_balance
        self.source = source
        self.broker = broker
        self.external_id = external_id
        self.trader_id = trader_id

    @property
    def risk_amount(self) -> Optional[float]:
        if self.stop_loss and self.entry_price:
            dist = abs(self.entry_price - self.stop_loss)
            return dist * self.position_size
        return None

    @property
    def risk_pct(self) -> Optional[float]:
        if self.risk_amount and self.account_balance and self.account_balance > 0:
            return (self.risk_amount / self.account_balance) * 100
        return None


# ══════════════════════════════════════════════
#  PIPELINE OUTPUT
# ══════════════════════════════════════════════

class TradeResult:
    """Complete result from the trade pipeline."""

    def __init__(self):
        self.trade: Optional[Trade] = None
        self.trade_id: Optional[UUID] = None
        self.attribution: Optional[AttributionResult] = None
        self.shadow_ids: list[UUID] = []
        self.insights: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    @property
    def success(self) -> bool:
        return self.trade is not None and len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "trade_id": str(self.trade_id) if self.trade_id else None,
            "attribution": self.attribution.to_dict() if self.attribution else None,
            "shadow_trade_ids": [str(sid) for sid in self.shadow_ids],
            "insights": self.insights,
            "warnings": self.warnings,
            "errors": self.errors,
        }


# ══════════════════════════════════════════════
#  MARKET DATA PROVIDER (interface)
# ══════════════════════════════════════════════

class MarketDataProvider:
    """
    Interface for providing market snapshots.
    Override with real implementation (broker API, data service).
    """

    def get_snapshot(self, symbol: str, timestamp: datetime) -> MarketSnapshot:
        """Return market data snapshot for attribution evaluation."""
        # Default: minimal snapshot — subclass for real data
        return MarketSnapshot(
            symbol=symbol,
            timestamp=timestamp,
        )


# ══════════════════════════════════════════════
#  TRADE PIPELINE
# ══════════════════════════════════════════════

class TradePipeline:
    """
    Orchestrates the complete trade entry flow:

    1. Validate input
    2. Create Trade record
    3. Update session state
    4. Get market snapshot
    5. Run 3-tier attribution
    6. Persist attribution + overrides + behavioral events
    7. Publish TRADE_ENTERED event (triggers shadow creation)
    8. Return complete result
    """

    def __init__(
        self,
        market_data_provider: Optional[MarketDataProvider] = None,
    ):
        self._registry = get_registry()
        if not self._registry.is_initialized:
            self._registry.initialize()
        self._session_mgr = get_session_manager()
        self._event_bus = get_event_bus()
        self._attribution_engine = AttributionEngine(self._registry)
        self._attribution_persistence = AttributionPersistence()
        self._market_data = market_data_provider or MarketDataProvider()

    def enter_trade(self, db: DBSession, input: TradeInput) -> TradeResult:
        """Process a new trade entry through the full pipeline."""
        result = TradeResult()

        try:
            # ═══════════════════════════
            # 1. VALIDATE
            # ═══════════════════════════
            validation_errors = self._validate(input)
            if validation_errors:
                result.errors = validation_errors
                return result

            # ═══════════════════════════
            # 2. CREATE TRADE RECORD
            # ═══════════════════════════
            trade = self._create_trade(db, input)
            result.trade = trade
            result.trade_id = trade.id

            # ═══════════════════════════
            # 3. UPDATE SESSION STATE
            # ═══════════════════════════
            if input.account_balance:
                self._session_mgr.set_account_balance(input.account_balance)

            self._session_mgr.on_trade_entered(
                trade_id=trade.id,
                symbol=trade.symbol,
                entry_time=trade.entry_time,
                risk_amount=input.risk_amount,
            )

            # ═══════════════════════════
            # 4. GET MARKET SNAPSHOT
            # ═══════════════════════════
            market = self._market_data.get_snapshot(
                symbol=input.symbol,
                timestamp=input.entry_time,
            )

            # ═══════════════════════════
            # 5. BUILD TRADE CONTEXT
            # ═══════════════════════════
            session_ctx = self._session_mgr.to_session_context()
            trade_ctx = self._build_trade_context(input, trade, session_ctx)

            # ═══════════════════════════
            # 6. RUN ATTRIBUTION
            # ═══════════════════════════
            attribution = self._attribution_engine.evaluate(
                trade=trade_ctx,
                market=market,
                session=session_ctx,
            )
            result.attribution = attribution
            result.insights = list(attribution.insights)

            # ═══════════════════════════
            # 7. PERSIST ATTRIBUTION
            # ═══════════════════════════
            session_type = self._detect_session_type(input.entry_time)

            self._attribution_persistence.persist(
                db=db,
                trade_id=str(trade.id),
                result=attribution,
                session_context=session_ctx,
                emotional_state=input.emotion_pre,
                session_type=session_type,
            )

            # Update trade with resolved strategy
            if attribution.resolved_strategy_id:
                trade.declared_strategy_id = (
                    trade.declared_strategy_id or attribution.resolved_strategy_id
                )
            trade.session_type = session_type
            trade.market_regime = market.regime if market.regime != MarketRegime.UNKNOWN else None
            db.commit()

            # ═══════════════════════════
            # 8. PUBLISH EVENT
            # ═══════════════════════════
            self._event_bus.publish(Event(
                event_type=EventType.TRADE_ENTERED,
                source="trade_pipeline",
                trade_id=trade.id,
                strategy_id=attribution.resolved_strategy_id,
                data={
                    "symbol": trade.symbol,
                    "direction": trade.direction.value,
                    "entry_price": trade.entry_price,
                    "position_size": trade.position_size,
                    "stop_loss": trade.stop_loss,
                    "take_profit": trade.take_profit,
                    "declared_strategy_id": input.declared_strategy_id,
                    "resolved_strategy_id": attribution.resolved_strategy_id,
                    "discipline_score": attribution.discipline_score,
                    "confidence_score": attribution.confidence_score,
                    "override_count": (
                        attribution.tier3_override_report.total_count
                        if attribution.tier3_override_report else 0
                    ),
                },
            ))

            # ═══════════════════════════
            # 9. PUBLISH ATTRIBUTION EVENT
            # ═══════════════════════════
            self._event_bus.publish(Event(
                event_type=EventType.ATTRIBUTION_COMPLETED,
                source="trade_pipeline",
                trade_id=trade.id,
                strategy_id=attribution.resolved_strategy_id,
                data=attribution.to_dict(),
            ))

            # Publish behavioral events if overrides detected
            if (
                attribution.tier3_override_report
                and attribution.tier3_override_report.has_overrides
            ):
                self._event_bus.publish(Event(
                    event_type=EventType.OVERRIDE_DETECTED,
                    source="trade_pipeline",
                    trade_id=trade.id,
                    data=attribution.tier3_override_report.to_dict(),
                ))

            # ═══════════════════════════
            # 10. WARNINGS
            # ═══════════════════════════
            if attribution.discipline_score < 50:
                result.warnings.append(
                    f"⚠️ Low discipline score ({attribution.discipline_score:.0f}/100)"
                )
            if self._session_mgr.consecutive_losses >= 3:
                result.warnings.append(
                    f"🔴 {self._session_mgr.consecutive_losses} consecutive losses — "
                    f"consider stepping away"
                )

            logger.info(
                "Trade pipeline complete: %s %s %s @ %.2f | strategy=%s | discipline=%.0f",
                trade.symbol,
                trade.direction.value,
                trade.status.value,
                trade.entry_price,
                attribution.resolved_strategy_id or "unattributed",
                attribution.discipline_score,
            )

        except Exception as e:
            logger.error("Trade pipeline failed: %s", e, exc_info=True)
            result.errors.append(f"Pipeline error: {str(e)}")
            db.rollback()

        return result

    def close_trade(
        self,
        db: DBSession,
        trade_id: UUID,
        exit_price: float,
        exit_time: Optional[datetime] = None,
        emotion_post: Optional[EmotionalState] = None,
        exit_notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Process a trade close through the pipeline."""
        trade = db.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            return {"success": False, "error": "Trade not found"}

        if trade.status != TradeStatus.OPEN:
            return {"success": False, "error": f"Trade is {trade.status.value}, not open"}

        exit_time = exit_time or datetime.now(timezone.utc)

        # Calculate P&L
        if trade.direction == TradeDirection.LONG:
            pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            pnl = (trade.entry_price - exit_price) * trade.position_size

        pnl -= (trade.commission or 0.0)

        # R-multiple
        r_multiple = None
        if trade.risk_amount and trade.risk_amount > 0:
            r_multiple = pnl / trade.risk_amount

        pnl_pct = None
        if trade.account_balance_at_entry and trade.account_balance_at_entry > 0:
            pnl_pct = (pnl / trade.account_balance_at_entry) * 100

        # Update trade
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.r_multiple = r_multiple
        trade.status = TradeStatus.CLOSED
        trade.emotion_post = emotion_post

        if exit_notes:
            trade.notes = (
                f"{trade.notes}\n\n--- Exit Notes ---\n{exit_notes}"
                if trade.notes else exit_notes
            )

        db.commit()

        # Update session state
        self._session_mgr.on_trade_closed(
            trade_id=trade.id,
            symbol=trade.symbol,
            pnl=pnl,
            exit_time=exit_time,
        )

        # Update strategy registry state
        if trade.declared_strategy_id:
            self._registry.update_daily_pnl(trade.declared_strategy_id, pnl)

        # Backfill behavioral event outcomes
        self._backfill_behavioral_outcomes(db, trade)

        # Market Context Enrichment
        self._enrich_market_context(db, trade)

        # Decision Quality Evaluation
        self._evaluate_decision_quality(db, trade)

        # Profile Update (async - doesn't block trade close)
        self._trigger_profile_update(trade)

        # Pattern Learning (async - doesn't block trade close)
        self._trigger_pattern_learning(trade)

        # Insight Memory (async - doesn't block trade close)
        self._trigger_insight_generation(trade)

        # Publish close event
        self._event_bus.publish(Event(
            event_type=EventType.TRADE_CLOSED,
            source="trade_pipeline",
            trade_id=trade.id,
            strategy_id=trade.declared_strategy_id,
            data={
                "symbol": trade.symbol,
                "direction": trade.direction.value,
                "pnl": pnl,
                "r_multiple": r_multiple,
                "exit_price": exit_price,
                "duration_seconds": (exit_time - trade.entry_time).total_seconds(),
            },
        ))

        logger.info(
            "Trade closed: %s %s P&L=$%.2f (%.1fR)",
            trade.symbol,
            trade.direction.value,
            pnl,
            r_multiple or 0.0,
        )

        return {
            "success": True,
            "trade_id": str(trade.id),
            "pnl": round(pnl, 2),
            "r_multiple": round(r_multiple, 2) if r_multiple else None,
            "pnl_pct": round(pnl_pct, 2) if pnl_pct else None,
        }

    # ── Internal helpers ──

    def _validate(self, input: TradeInput) -> list[str]:
        errors = []
        if not input.symbol:
            errors.append("Symbol is required")
        if input.entry_price <= 0:
            errors.append("Entry price must be positive")
        if input.position_size <= 0:
            errors.append("Position size must be positive")
        if input.stop_loss is not None:
            if input.direction == TradeDirection.LONG and input.stop_loss >= input.entry_price:
                errors.append("Long stop loss must be below entry price")
            elif input.direction == TradeDirection.SHORT and input.stop_loss <= input.entry_price:
                errors.append("Short stop loss must be above entry price")
        if input.declared_strategy_id:
            if not self._registry.get(input.declared_strategy_id):
                # Log as warning only — don't block the trade.
                # Registry may not be initialized yet or strategy is a custom ID.
                logger.warning(
                    "Declared strategy '%s' not found in registry — trade accepted with unknown strategy",
                    input.declared_strategy_id,
                )
        return errors

    def _create_trade(self, db: DBSession, input: TradeInput) -> Trade:
        trade = Trade(
            symbol=input.symbol.upper(),
            direction=input.direction,
            status=TradeStatus.OPEN,
            entry_price=input.entry_price,
            entry_time=input.entry_time,
            position_size=input.position_size,
            stop_loss=input.stop_loss,
            take_profit=input.take_profit,
            risk_amount=input.risk_amount,
            risk_pct=input.risk_pct,
            account_balance_at_entry=input.account_balance,
            declared_strategy_id=input.declared_strategy_id,
            emotion_pre=input.emotion_pre,
            conviction_score=input.conviction_score,
            notes=input.notes,
            setup_screenshot=input.screenshot_path,
            source=input.source,
            broker=input.broker,
            external_id=input.external_id,
            trader_id=UUID(input.trader_id) if input.trader_id else None,
        )
        db.add(trade)
        db.flush()
        return trade

    def _build_trade_context(
        self,
        input: TradeInput,
        trade: Trade,
        session: SessionContext,
    ) -> TradeContext:
        time_since_last = None
        if session.last_trade_time:
            time_since_last = (
                input.entry_time - session.last_trade_time
            ).total_seconds()

        return TradeContext(
            symbol=input.symbol,
            direction=input.direction,
            entry_price=input.entry_price,
            entry_time=input.entry_time,
            position_size=input.position_size,
            stop_loss=input.stop_loss,
            take_profit=input.take_profit,
            risk_amount=input.risk_amount,
            risk_pct=input.risk_pct,
            account_balance=input.account_balance,
            declared_strategy_id=input.declared_strategy_id,
            session_type=self._detect_session_type(input.entry_time),
            market_regime=self._session_mgr.current_regime,
            trades_today=session.trades_today,
            daily_pnl=session.daily_pnl,
            consecutive_losses=session.consecutive_losses,
            time_since_last_trade_seconds=time_since_last,
            last_trade_pnl=session.last_trade_pnl,
        )

    def _detect_session_type(self, entry_time: datetime) -> Optional[SessionType]:
        from zoneinfo import ZoneInfo
        from config.settings import settings
        try:
            local = entry_time.astimezone(ZoneInfo(settings.MARKET_TIMEZONE))
        except Exception:
            return None

        hour, minute = local.hour, local.minute
        time_val = hour * 60 + minute

        # NSE session boundaries (IST)
        if time_val < 9 * 60 + 15:
            return SessionType.PRE_MARKET
        elif time_val < 11 * 60 + 30:
            return SessionType.MORNING
        elif time_val < 14 * 60:
            return SessionType.MIDDAY
        elif time_val < 15 * 60 + 15:
            return SessionType.AFTERNOON
        elif time_val < 15 * 60 + 30:
            return SessionType.CLOSE
        else:
            return SessionType.AFTER_HOURS


    def _backfill_behavioral_outcomes(self, db: DBSession, trade: Trade) -> None:
        """Update behavioral events with trade outcome after close."""
        from core.models import BehavioralEvent

        events = (
            db.query(BehavioralEvent)
            .filter(
                BehavioralEvent.trade_id == trade.id,
                BehavioralEvent.outcome_validated == False,
            )
            .all()
        )

        for event in events:
            event.outcome_pnl = trade.pnl
            event.outcome_r = trade.r_multiple
            event.outcome_validated = True

        if events:
            db.commit()
            logger.debug(
                "Backfilled %d behavioral event outcomes for trade %s",
                len(events), trade.id,
            )

    def _evaluate_decision_quality(self, db: DBSession, trade: Trade) -> None:
        """Evaluate decision quality for a closed trade and persist results."""
        try:
            from analytics.decision_quality import DecisionQualityEngine, TradeContext
            from core.models import DecisionQualityRecord
            
            # Build trade context for decision quality evaluation
            context = self._build_decision_quality_context(trade)
            
            # Run decision quality evaluation
            dq_engine = DecisionQualityEngine()
            result = dq_engine.evaluate(context)
            
            # Create and persist decision quality record
            dq_record = DecisionQualityRecord(
                trade_id=trade.id,
                composite_score=result.final_score,
                grade=result.grade.value,
                data_completeness=result.data_completeness,
                outcome_independent=result.outcome_independent,
                dimension_details={
                    dim.name.value: {
                        "score": dim.score,
                        "weight": dim.weight,
                        "confidence": dim.confidence,
                        "components": dim.components,
                        "reasoning": list(dim.reasoning)
                    }
                    for dim in result.dimensions
                },
                insights=[
                    {
                        "insight_id": str(insight.insight_id),
                        "dimension": insight.dimension.value,
                        "severity": insight.severity.value,
                        "title": insight.title,
                        "description": insight.description,
                        "suggestion": insight.suggestion,
                        "score_impact": insight.score_impact
                    }
                    for insight in result.insights
                ],
                is_quality_trade=result.final_score >= 70.0,
                is_process_violation=result.final_score < 40.0,
                coaching_notes=self._generate_coaching_notes(result)
            )
            
            db.add(dq_record)
            db.commit()
            
            # Publish decision quality event
            self._event_bus.publish(Event(
                event_type=EventType.BEHAVIORAL_EVENT,  # Using existing event type
                source="trade_pipeline",
                trade_id=trade.id,
                data={
                    "event_type": "decision_quality_evaluated",
                    "evaluation_id": dq_record.evaluation_id,
                    "composite_score": result.final_score,
                    "grade": result.grade.value,
                    "is_quality_trade": dq_record.is_quality_trade,
                    "is_process_violation": dq_record.is_process_violation,
                    "coaching_notes": dq_record.coaching_notes,
                    "insights_count": len(result.insights),
                    "critical_insights": len(result.critical_insights)
                }
            ))
            
            logger.info(
                "Decision quality evaluated for trade %s: score=%.1f (%s) - %s",
                trade.id,
                result.final_score,
                result.grade.value,
                "quality trade" if dq_record.is_quality_trade else "needs improvement"
            )
            
        except Exception as e:
            logger.error("Decision quality evaluation failed: %s", e, exc_info=True)
            try:
                db.rollback()
            except Exception:
                pass

    def _build_decision_quality_context(self, trade: Trade) -> TradeContext:
        """Build TradeContext for decision quality evaluation from Trade model."""
        from analytics.decision_quality import TradeContext as DQTradeContext
        from analytics.decision_quality import RecentTradeSnapshot, TradeOutcome
        
        # Get recent trades for context (simplified - in production would query properly)
        recent_trades = [
            RecentTradeSnapshot(
                trade_id=trade.id,
                pnl=trade.pnl or 0.0,
                r_multiple=trade.r_multiple,
                outcome=TradeOutcome.WIN if (trade.pnl or 0) > 0 else TradeOutcome.LOSS if (trade.pnl or 0) < 0 else TradeOutcome.BREAKEVEN,
                exit_time=trade.exit_time,
                emotional_state=trade.emotion_post,
                position_size=trade.position_size
            )
        ]
        
        return DQTradeContext(
            trade_id=trade.id,
            symbol=trade.symbol,
            side=trade.direction.value,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            quantity=trade.position_size,
            pnl=trade.pnl,
            r_multiple=trade.r_multiple,
            
            # Strategy information
            strategy_name=trade.declared_strategy_id,
            conditions_met=None,  # Would come from attribution
            conditions_total=None,
            regime_compatible=trade.market_regime is not None,
            
            # Risk planning
            planned_stop_price=trade.stop_loss,
            actual_stop_price=trade.stop_loss,  # Simplified
            planned_target_price=trade.take_profit,
            planned_position_size=None,  # Would be calculated
            actual_position_size=trade.position_size,
            planned_risk_reward=None,  # Would be calculated
            portfolio_heat=None,  # Would come from session
            max_daily_loss_limit=None,  # Would come from session
            current_daily_pnl=None,  # Would come from session
            
            # Emotional/behavioral
            emotional_state=trade.emotion_post,
            confidence_level=trade.conviction_score,
            notes=trade.notes,
            override_events=[],  # Would come from attribution
            override_severity=None,
            
            # Market context (simplified)
            atr=None,  # Would come from market data
            volatility_regime=trade.market_regime.value if trade.market_regime else None,
            trend_direction=None,  # Would come from market data
            trend_strength=None,  # Would come from market data
            vix_level=None,  # Would come from market data
            session_type=trade.session_type.value if trade.session_type else None,
            market_open_time=None,  # Would be calculated
            
            # Shadow comparison (not available in basic implementation)
            shadow_entry_price=None,
            shadow_exit_price=None,
            shadow_pnl=None,
            optimal_entry_price=None,
            optimal_exit_price=None,
            
            # Recent history
            recent_trades=recent_trades,
            trade_count_today=1,  # Simplified
            time_since_last_trade_seconds=None,
        )

    def _generate_coaching_notes(self, result) -> str:
        """Generate coaching notes from decision quality result."""
        notes = []
        
        # Grade-based summary
        if result.final_score >= 85:
            notes.append("Excellent execution across all dimensions.")
        elif result.final_score >= 70:
            notes.append("Good overall execution with minor areas for improvement.")
        elif result.final_score >= 55:
            notes.append("Acceptable execution but several dimensions need attention.")
        elif result.final_score >= 40:
            notes.append("Poor execution - significant process violations detected.")
        else:
            notes.append("Undisciplined trading - major process breakdown.")
        
        # Critical insights
        critical_insights = [i for i in result.insights if i.severity.value == "critical"]
        if critical_insights:
            notes.append(f"Critical issues: {', '.join(i.title for i in critical_insights[:2])}")
        
        # Weakest dimension
        if result.dimensions:
            weakest = min(result.dimensions, key=lambda d: d.score)
            if weakest.score < 50:
                notes.append(f"Primary focus area: {weakest.name.value.replace('_', ' ').title()}")
        
        return " | ".join(notes) if notes else "Trade executed."

    def _trigger_profile_update(self, trade: Trade) -> None:
        """Trigger async profile update for trader."""
        try:
            # In production, this would be an async task
            # For now, just publish an event for the profile service
            self._event_bus.publish(Event(
                event_type=EventType.BEHAVIORAL_EVENT,
                source="trade_pipeline",
                trade_id=trade.id,
                data={
                    "event_type": "trade_closed_profile_update",
                    "trader_id": str(trade.trader_id) if trade.trader_id else "unknown",
                    "symbol": trade.symbol,
                    "pnl": trade.pnl,
                    "r_multiple": trade.r_multiple,
                },
            ))
        except Exception as e:
            logger.debug("Profile update trigger failed: %s", e)

    def _enrich_market_context(self, db: DBSession, trade: Trade) -> None:
        """Enrich trade with market context (regime, session type) after close."""
        try:
            if trade.session_type is None:
                trade.session_type = self._detect_session_type(trade.exit_time or trade.entry_time)
            if trade.market_regime is None and self._session_mgr.current_regime:
                trade.market_regime = self._session_mgr.current_regime
            db.commit()
        except Exception as e:
            logger.debug("Market context enrichment failed: %s", e)

    def _trigger_pattern_learning(self, trade: Trade) -> None:
        """Trigger pattern learning from closed trade."""
        try:
            self._event_bus.publish(Event(
                event_type=EventType.TRADE_CLOSED,
                source="trade_pipeline.pattern_learning",
                trade_id=trade.id,
                data={
                    "event_type": "pattern_learning_trigger",
                    "symbol": trade.symbol,
                    "pnl": trade.pnl,
                    "r_multiple": trade.r_multiple,
                    "strategy_id": trade.declared_strategy_id,
                },
            ))
        except Exception as e:
            logger.debug("Pattern learning trigger failed: %s", e)

    def _trigger_insight_generation(self, trade: Trade) -> None:
        """Trigger insight/memory generation from closed trade."""
        try:
            self._event_bus.publish(Event(
                event_type=EventType.BEHAVIORAL_EVENT,
                source="trade_pipeline.insights",
                trade_id=trade.id,
                data={
                    "event_type": "insight_generation_trigger",
                    "trader_id": str(trade.trader_id) if trade.trader_id else "unknown",
                    "symbol": trade.symbol,
                    "pnl": trade.pnl,
                    "r_multiple": trade.r_multiple,
                },
            ))
        except Exception as e:
            logger.debug("Insight generation trigger failed: %s", e)