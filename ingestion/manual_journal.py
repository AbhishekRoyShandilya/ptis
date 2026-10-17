# manual_journal.py
# ingestion/manual_journal.py
from datetime import date, datetime
from typing import Optional, List, Dict
from sqlalchemy.orm import Session
from core.models import (
    Trade, JournalEntry, TradeScreenshot,
    TradeDirection, TradeStatus, SetupType,
    EmotionalState, MarketRegime, SessionType
)
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class TradeInput(BaseModel):
    """Structured input for manual trade entry."""
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime
    exit_time: Optional[datetime] = None
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: Optional[str] = None
    setup_type: Optional[str] = None
    timeframe: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class JournalInput(BaseModel):
    """Structured input for journal entry."""
    # Pre-trade
    pre_trade_emotion: Optional[str] = None
    pre_trade_confidence: Optional[int] = Field(None, ge=1, le=10)
    pre_trade_notes: Optional[str] = None

    # Post-trade
    post_trade_emotion: Optional[str] = None
    post_trade_notes: Optional[str] = None

    # Decision context
    market_bias: Optional[str] = None
    key_levels_noted: Optional[List[float]] = None
    news_events: Optional[str] = None
    was_planned: Optional[bool] = None
    followed_rules: Optional[bool] = None
    rule_violations: Optional[List[str]] = None

    # Session context
    sleep_quality: Optional[int] = Field(None, ge=1, le=10)
    energy_level: Optional[int] = Field(None, ge=1, le=10)
    focus_level: Optional[int] = Field(None, ge=1, le=10)
    stress_level: Optional[int] = Field(None, ge=1, le=10)

    # Lessons
    lessons_learned: Optional[str] = None
    what_went_well: Optional[str] = None
    what_to_improve: Optional[str] = None


class ManualJournalService:
    """Service for manual trade logging and journaling."""

    def __init__(self, db: Session):
        self.db = db

    def log_trade_with_journal(
        self,
        trade_input: TradeInput,
        journal_input: Optional[JournalInput] = None,
        screenshot_paths: Optional[List[str]] = None
    ) -> Dict:
        """Log a complete trade with journal entry and screenshots."""

        # Create trade
        trade = self._create_trade(trade_input)
        self.db.add(trade)
        self.db.flush()  # get trade ID

        result = {"trade_id": trade.id}

        # Create journal entry
        if journal_input:
            journal = self._create_journal(trade.id, trade_input.entry_time.date(), journal_input)
            self.db.add(journal)
            result["journal_id"] = journal.id

        # Attach screenshots
        if screenshot_paths:
            for path in screenshot_paths:
                ss = TradeScreenshot(
                    trade_id=trade.id,
                    file_path=path,
                    screenshot_type="context"
                )
                self.db.add(ss)

        self.db.flush()
        logger.info(f"Logged trade {trade.id} for {trade.symbol}")

        return result

    def add_daily_journal(
        self,
        entry_date: date,
        journal_input: JournalInput
    ) -> JournalEntry:
        """Add a standalone daily journal entry (not tied to a specific trade)."""
        journal = self._create_journal(None, entry_date, journal_input)
        self.db.add(journal)
        self.db.flush()
        return journal

    def update_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: datetime,
        post_emotion: Optional[str] = None,
        post_notes: Optional[str] = None
    ) -> Trade:
        """Update trade with exit information."""
        trade = self.db.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            raise ValueError(f"Trade {trade_id} not found")

        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.status = TradeStatus.CLOSED

        # Calculate R multiple
        if trade.stop_loss and trade.stop_loss != trade.entry_price:
            risk_per_unit = abs(trade.entry_price - trade.stop_loss)
            if trade.direction == TradeDirection.LONG:
                trade.actual_r_multiple = (exit_price - trade.entry_price) / risk_per_unit
            else:
                trade.actual_r_multiple = (trade.entry_price - exit_price) / risk_per_unit

        # Update journal if exists
        if trade.journal_entry and post_emotion:
            trade.journal_entry.post_trade_emotion = EmotionalState(post_emotion)
            trade.journal_entry.post_trade_notes = post_notes

        self.db.flush()
        return trade

    def _create_trade(self, ti: TradeInput) -> Trade:
        direction = TradeDirection(ti.direction.lower())
        status = TradeStatus.CLOSED if ti.exit_price else TradeStatus.OPEN

        # Calculate initial risk
        initial_risk = None
        if ti.stop_loss:
            initial_risk = abs(ti.entry_price - ti.stop_loss) * ti.quantity

        # Calculate R multiple
        actual_r = None
        if ti.exit_price and ti.stop_loss and ti.stop_loss != ti.entry_price:
            risk_per_unit = abs(ti.entry_price - ti.stop_loss)
            if direction == TradeDirection.LONG:
                actual_r = (ti.exit_price - ti.entry_price) / risk_per_unit
            else:
                actual_r = (ti.entry_price - ti.exit_price) / risk_per_unit

        return Trade(
            symbol=ti.symbol.upper().strip(),
            direction=direction,
            entry_price=ti.entry_price,
            exit_price=ti.exit_price,
            entry_time=ti.entry_time,
            exit_time=ti.exit_time,
            status=status,
            quantity=ti.quantity,
            position_value=ti.entry_price * ti.quantity,
            stop_loss=ti.stop_loss,
            take_profit=ti.take_profit,
            initial_risk=initial_risk,
            actual_r_multiple=actual_r,
            strategy_name=ti.strategy_name,
            setup_type=SetupType(ti.setup_type) if ti.setup_type else None,
            timeframe=ti.timeframe,
            tags=ti.tags,
            notes=ti.notes,
            source="manual",
        )

    def _create_journal(
        self,
        trade_id: Optional[str],
        entry_date: date,
        ji: JournalInput
    ) -> JournalEntry:
        return JournalEntry(
            trade_id=trade_id,
            entry_date=entry_date,
            pre_trade_emotion=EmotionalState(ji.pre_trade_emotion) if ji.pre_trade_emotion else None,
            pre_trade_confidence=ji.pre_trade_confidence,
            pre_trade_notes=ji.pre_trade_notes,
            post_trade_emotion=EmotionalState(ji.post_trade_emotion) if ji.post_trade_emotion else None,
            post_trade_notes=ji.post_trade_notes,
            market_bias=ji.market_bias,
            key_levels_noted=ji.key_levels_noted,
            news_events=ji.news_events,
            was_planned=ji.was_planned,
            followed_rules=ji.followed_rules,
            rule_violations=ji.rule_violations,
            sleep_quality=ji.sleep_quality,
            energy_level=ji.energy_level,
            focus_level=ji.focus_level,
            stress_level=ji.stress_level,
            lessons_learned=ji.lessons_learned,
            what_went_well=ji.what_went_well,
            what_to_improve=ji.what_to_improve,
        )