# core/models.py
"""
SQLAlchemy ORM models for PTIS.
Only showing NEW tables for attribution + behavioral events.
Assumes Trade, JournalEntry, etc. already exist.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, Text, DateTime, Date,
    ForeignKey, JSON, Enum as SAEnum, Index, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

from core.enums import (
    TradeDirection, TradeStatus, MarketRegime, EmotionalState,
    SessionType, AttributionTier, OverrideType, OverrideSeverity,
    BehavioralEventType, ConditionStatus, SKOStatus, ConvictionBand,
    AutomationPhase, SetupType,
)

Base = declarative_base()


# ══════════════════════════════════════════════
#  USER
# ══════════════════════════════════════════════

class User(Base):
    """User account for the PTIS platform."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)

    # Onboarding
    onboarding_completed = Column(Boolean, default=False)
    onboarding_data = Column(JSONB, default=dict)  # experience, style, risk prefs, etc.

    tokens_invalidated_at = Column(DateTime(timezone=True), nullable=True)  # logout-all: reject tokens issued before this

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


# ══════════════════════════════════════════════
#  TRADE (reference — assumed to exist)
# ══════════════════════════════════════════════

class Trade(Base):
    __tablename__ = "trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(SAEnum(TradeDirection), nullable=False)
    status = Column(SAEnum(TradeStatus), default=TradeStatus.OPEN)

    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    entry_time = Column(DateTime(timezone=True), nullable=False, index=True)
    exit_time = Column(DateTime(timezone=True), nullable=True)

    position_size = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)

    # P&L
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    r_multiple = Column(Float, nullable=True)
    commission = Column(Float, default=0.0)

    # Risk
    risk_amount = Column(Float, nullable=True)
    risk_pct = Column(Float, nullable=True)
    account_balance_at_entry = Column(Float, nullable=True)

    # Context
    session_type = Column(SAEnum(SessionType), nullable=True)
    market_regime = Column(SAEnum(MarketRegime), nullable=True)

    # User-declared strategy (Tier 1)
    declared_strategy_id = Column(String(50), nullable=True, index=True)

    # Emotion
    emotion_pre = Column(SAEnum(EmotionalState), nullable=True)
    emotion_post = Column(SAEnum(EmotionalState), nullable=True)
    conviction_score = Column(Float, nullable=True)  # 1-10

    # Notes
    notes = Column(Text, nullable=True)
    setup_screenshot = Column(String(500), nullable=True)

    # Source
    source = Column(String(50), default="manual")  # manual, csv, broker_api
    broker = Column(String(50), nullable=True)
    external_id = Column(String(100), nullable=True)

    # Owner
    trader_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    attribution = relationship("StrategyAttribution", back_populates="trade", uselist=False, cascade="all, delete-orphan")
    overrides = relationship("OverrideEvent", back_populates="trade", cascade="all, delete-orphan")
    behavioral_events = relationship("BehavioralEvent", back_populates="trade", cascade="all, delete-orphan")
    condition_evaluations = relationship("ConditionEvaluation", back_populates="trade", cascade="all, delete-orphan")
    decision_quality = relationship("DecisionQualityRecord", back_populates="trade", uselist=False, cascade="all, delete-orphan")
    journal_entry = relationship("JournalEntry", back_populates="trade", uselist=False, cascade="all, delete-orphan")
    trader = relationship("User")

    __table_args__ = (
        Index("ix_trades_symbol_entry", "symbol", "entry_time"),
        Index("ix_trades_strategy_entry", "declared_strategy_id", "entry_time"),
    )


# ══════════════════════════════════════════════
#  STRATEGY ATTRIBUTION
# ══════════════════════════════════════════════

class StrategyAttribution(Base):
    """
    3-tier attribution result for a trade.
    Links declared strategy, auto-match results, and override analysis.
    """
    __tablename__ = "strategy_attributions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id", ondelete="CASCADE"), nullable=False, unique=True)

    # ── Tier 1: User Declared ──
    tier1_strategy_id = Column(String(50), nullable=True)
    tier1_declared = Column(Boolean, default=False)

    # ── Tier 2: Auto-Matched ──
    tier2_matches = Column(JSONB, default=list)
    # Structure: [{strategy_id, score, required_met, total_met, total_conditions, details}]
    tier2_best_match_id = Column(String(50), nullable=True)
    tier2_best_match_score = Column(Float, nullable=True)

    # ── Tier 3: Override Analysis ──
    tier3_override_count = Column(Integer, default=0)
    tier3_max_severity = Column(SAEnum(OverrideSeverity), nullable=True)
    tier3_summary = Column(JSONB, default=dict)
    # Structure: {total_overrides, by_type: {type: count}, severity_distribution: {...}}

    # ── Resolved Attribution ──
    resolved_strategy_id = Column(String(50), nullable=True, index=True)
    resolved_tier = Column(SAEnum(AttributionTier), nullable=True)
    confidence_score = Column(Float, nullable=True)  # 0.0 - 1.0
    discipline_score = Column(Float, nullable=True)  # 0.0 - 100.0

    # ── Metadata ──
    evaluated_at = Column(DateTime(timezone=True), server_default=func.now())
    market_regime_at_entry = Column(SAEnum(MarketRegime), nullable=True)
    evaluation_context = Column(JSONB, default=dict)
    # Stores: available data flags, regime, session, etc.

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    trade = relationship("Trade", back_populates="attribution")

    __table_args__ = (
        Index("ix_attr_resolved_strategy", "resolved_strategy_id"),
        Index("ix_attr_discipline", "discipline_score"),
    )


# ══════════════════════════════════════════════
#  CONDITION EVALUATIONS
# ══════════════════════════════════════════════

class ConditionEvaluation(Base):
    """
    Individual condition evaluation result for a trade against a strategy.
    Provides granular visibility into which conditions matched.
    """
    __tablename__ = "condition_evaluations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id", ondelete="CASCADE"), nullable=False)
    strategy_id = Column(String(50), nullable=False, index=True)

    condition_type = Column(String(50), nullable=False)
    condition_params = Column(JSONB, default=dict)
    required = Column(Boolean, default=True)
    weight = Column(Float, default=1.0)

    status = Column(SAEnum(ConditionStatus), nullable=False)
    confidence = Column(Float, nullable=True)
    actual_value = Column(String(200), nullable=True)
    expected_value = Column(String(200), nullable=True)
    details = Column(Text, nullable=True)

    evaluated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    trade = relationship("Trade", back_populates="condition_evaluations")

    __table_args__ = (
        Index("ix_cond_eval_trade_strategy", "trade_id", "strategy_id"),
    )


# ══════════════════════════════════════════════
#  OVERRIDE EVENTS
# ══════════════════════════════════════════════

class OverrideEvent(Base):
    """
    Records a specific rule deviation / override detected on a trade.
    """
    __tablename__ = "override_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id", ondelete="CASCADE"), nullable=False)
    attribution_id = Column(UUID(as_uuid=True), ForeignKey("strategy_attributions.id", ondelete="SET NULL"), nullable=True)

    override_type = Column(SAEnum(OverrideType), nullable=False, index=True)
    severity = Column(SAEnum(OverrideSeverity), nullable=False)
    severity_score = Column(Float, nullable=False)  # 0.0 - 1.0

    description = Column(Text, nullable=False)
    rule_expected = Column(Text, nullable=True)
    actual_value = Column(String(200), nullable=True)
    expected_value = Column(String(200), nullable=True)

    strategy_id = Column(String(50), nullable=True)
    context = Column(JSONB, default=dict)

    detected_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    trade = relationship("Trade", back_populates="overrides")

    __table_args__ = (
        Index("ix_override_trade_type", "trade_id", "override_type"),
        Index("ix_override_severity", "severity"),
        Index("ix_override_detected", "detected_at"),
    )


# ══════════════════════════════════════════════
#  BEHAVIORAL EVENTS
# ══════════════════════════════════════════════

class BehavioralEvent(Base):
    """
    Detected behavioral patterns — both negative deviations and positive discipline events.
    Forms the behavioral dataset for pattern learning.
    """
    __tablename__ = "behavioral_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id", ondelete="SET NULL"), nullable=True)
    session_date = Column(DateTime, nullable=False, index=True)

    event_type = Column(SAEnum(BehavioralEventType), nullable=False, index=True)
    severity_score = Column(Float, nullable=False)  # 0.0 - 1.0 (for neg) or benefit score (for pos)
    is_positive = Column(Boolean, default=False)

    description = Column(Text, nullable=False)

    # Detection context
    trigger_conditions = Column(JSONB, default=dict)
    # e.g., {"consecutive_losses": 3, "time_since_last_trade_min": 2, "pnl_today": -500}

    # Correlated state
    emotional_state = Column(SAEnum(EmotionalState), nullable=True)
    market_regime = Column(SAEnum(MarketRegime), nullable=True)
    session_type = Column(SAEnum(SessionType), nullable=True)
    pnl_at_detection = Column(Float, nullable=True)
    trades_today_at_detection = Column(Integer, nullable=True)

    # Outcome tracking (filled after trade closes)
    outcome_pnl = Column(Float, nullable=True)
    outcome_r = Column(Float, nullable=True)
    outcome_validated = Column(Boolean, default=False)
    # True = the behavioral flag was justified by outcome

    detected_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    trade = relationship("Trade", back_populates="behavioral_events")

    __table_args__ = (
        Index("ix_behav_session_type", "session_date", "event_type"),
        Index("ix_behav_detected", "detected_at"),
    )


# ══════════════════════════════════════════════
#  DECISION QUALITY RECORDS
# ══════════════════════════════════════════════

class DecisionQualityRecord(Base):
    """Decision quality evaluation results for a trade."""
    __tablename__ = "decision_quality_evaluations"

    evaluation_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id"), nullable=True, index=True)
    trader_id = Column(String(50), nullable=False, index=True)
    
    evaluated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    # Composite results
    composite_score = Column(Float, nullable=False)
    grade = Column(String(20), nullable=False)
    data_completeness = Column(Float, nullable=False, default=0.0)
    outcome_independent = Column(Boolean, nullable=False, default=True)
    
    # Detailed results (JSON)
    dimension_details = Column(JSON, nullable=True)  # Dimension scores and details
    insights = Column(JSON, nullable=True)  # Improvement insights
    
    # Coaching context
    is_quality_trade = Column(Boolean, nullable=False, default=True)
    is_process_violation = Column(Boolean, nullable=False, default=False)
    coaching_notes = Column(Text, nullable=True)
    
    # Relationships
    trade = relationship("Trade", back_populates="decision_quality")

    __table_args__ = (
        Index("ix_dq_evaluated_at", "evaluated_at"),
        Index("ix_dq_grade", "grade"),
        Index("ix_dq_trader", "trader_id", "evaluated_at"),
    )


class DisciplineRecord(Base):
    """Discipline scoring records for traders."""
    __tablename__ = "discipline_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    trader_id = Column(String(50), nullable=False, index=True)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id"), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    score = Column(Float, nullable=False)  # 0-100
    trend = Column(String(20), nullable=True)  # improving, declining, stable

    # Component scores
    rule_adherence = Column(Float, nullable=True)
    emotional_control = Column(Float, nullable=True)
    consistency = Column(Float, nullable=True)

    # Context
    session_date = Column(Date, nullable=False)
    trades_in_session = Column(Integer, nullable=True)

    # Relationships
    trade = relationship("Trade")

    __table_args__ = (
        Index("ix_discipline_trader_date", "trader_id", "session_date"),
        Index("ix_discipline_created", "created_at"),
    )


# ══════════════════════════════════════════════
#  JOURNAL ENTRY
# ══════════════════════════════════════════════

class JournalEntry(Base):
    """Pre- and post-trade journal: plan, mental state, lessons."""
    __tablename__ = "journal_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id", ondelete="CASCADE"),
                      nullable=False, unique=True, index=True)

    # Pre-trade
    plan = Column(Text, nullable=True)
    was_planned = Column(Boolean, default=True)
    pre_trade_confidence = Column(Integer, nullable=True)   # 1-10
    pre_trade_emotion = Column(SAEnum(EmotionalState), nullable=True)
    sleep_quality = Column(Integer, nullable=True)          # 1-10
    energy_level = Column(Integer, nullable=True)           # 1-10
    stress_level = Column(Integer, nullable=True)           # 1-10

    # Post-trade
    post_trade_emotion = Column(SAEnum(EmotionalState), nullable=True)
    followed_plan = Column(Boolean, nullable=True)
    would_take_again = Column(Boolean, nullable=True)
    what_went_well = Column(Text, nullable=True)
    what_went_wrong = Column(Text, nullable=True)
    lesson = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    trade = relationship("Trade", back_populates="journal_entry")


# ══════════════════════════════════════════════
#  AUTOMATION RULE
# ══════════════════════════════════════════════

class AutomationRule(Base):
    """
    Configurable guardrail / automation rule.
    rule_type: loss_limit | daily_drawdown | max_trades | loss_streak |
               emotion_block | discipline_floor | time_block | circuit_breaker
    action:    pause | stop | alert | reduce_size
    """
    __tablename__ = "automation_rules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    rule_type = Column(String(50), nullable=False, index=True)
    trigger_value = Column(Float, nullable=False, default=0.0)
    action = Column(String(30), nullable=False, default="alert")
    enabled = Column(Boolean, default=True, index=True)
    description = Column(Text, nullable=True)
    triggered_count = Column(Integer, default=0)

    # Owner
    trader_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index("ix_automation_enabled_type", "enabled", "rule_type"),
    )


# ══════════════════════════════════════════════
#  KNOWLEDGE ENTRY  (RAG knowledge base)
# ══════════════════════════════════════════════

class KnowledgeEntry(Base):
    """
    Personal trading knowledge — lessons, rules, strategies, insights.
    Indexed into the vector store by rag/indexer.py for semantic retrieval.
    """
    __tablename__ = "knowledge_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    category = Column(String(30), nullable=False, index=True)
    # "strategy" | "lesson" | "pattern" | "insight" | "rule" | "review"

    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(50), default="manual")
    # "manual" | "trade_analysis" | "ai_<agent_name>"

    tags = Column(JSONB, default=list)   # list of strings

    # Owner
    trader_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)

    # Usage tracking
    relevance_count = Column(Integer, default=0)
    last_referenced = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index("ix_knowledge_category", "category"),
        Index("ix_knowledge_created", "created_at"),
    )


# ══════════════════════════════════════════════
#  MARKET REGIME LOG
# ══════════════════════════════════════════════

class MarketRegimeLog(Base):
    """
    Timestamped record of each regime change (manual or auto-detected).
    Used by regime routes to track current/historical market context.
    """
    __tablename__ = "market_regime_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True,
                       server_default=func.now())
    regime = Column(SAEnum(MarketRegime), nullable=False)
    confidence = Column(Float, default=0.8)
    indicators = Column(JSONB, default=dict)
    # e.g. {"source": "manual", "notes": "...", "atr_percentile": 0.72}

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_regime_log_timestamp", "timestamp"),
    )


# ══════════════════════════════════════════════
#  CONVERSATION SESSION & MESSAGES
# ══════════════════════════════════════════════

class ConversationSession(Base):
    """
    A coaching conversation session with the AI agent.
    One session = one continuous conversation thread.
    """
    __tablename__ = "conversation_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trader_id = Column(String(100), nullable=False, index=True)
    mode = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    extra_data = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)

    messages = relationship("ConversationMessage", back_populates="session",
                            cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_conv_session_trader", "trader_id"),
        Index("ix_conv_session_active", "is_active"),
    )


class ConversationMessage(Base):
    """A single message (turn) within a coaching conversation session."""
    __tablename__ = "conversation_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("conversation_sessions.id",
                        ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)   # "user" | "assistant"
    content = Column(Text, nullable=False)
    extra_data = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ConversationSession", back_populates="messages")

    __table_args__ = (
        Index("ix_conv_message_session", "session_id"),
        Index("ix_conv_message_created", "created_at"),
    )


# ══════════════════════════════════════════════
#  STRATEGY LEARNING — AI-Learned Strategy Profiles
# ══════════════════════════════════════════════

class StrategyProfile(Base):
    """
    A learned strategy profile built from user-provided screenshots and annotations.
    Replaces hard-coded PersonalStrategy classes with AI-derived understanding.
    """
    __tablename__ = "strategy_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    short_code = Column(String(20), nullable=False, unique=True)
    status = Column(String(20), nullable=False, default="draft")  # draft, learning, active, archived

    # Owner
    trader_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)

    # Learned understanding
    narrative = Column(Text, nullable=True)  # evolving NL description, rebuilt after each session
    entry_criteria = Column(JSONB, default=list)  # [{criterion, description, confidence, example_count}]
    exit_rules = Column(JSONB, default=dict)  # {stop_method, typical_stop_atr, target_patterns}
    risk_parameters = Column(JSONB, default=dict)  # {typical_risk_pct, typical_rr}
    regime_preferences = Column(JSONB, default=dict)  # {preferred: [{regime, confidence}], avoid: [...]}
    time_preferences = Column(JSONB, default=dict)  # learned time-of-day patterns
    confidence_scores = Column(JSONB, default=dict)  # {entry: 0.85, exit: 0.40, stop: 0.70, overall: 0.65}

    # Counters
    examples_count = Column(Integer, default=0)
    sessions_count = Column(Integer, default=0)

    # Timestamps
    last_rebuilt_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Flexible metadata
    tags = Column(JSONB, default=list)
    profile_metadata = Column("metadata", JSONB, default=dict)

    # Relationships
    learning_sessions = relationship("LearningSession", back_populates="profile", cascade="all, delete-orphan")
    learning_examples = relationship("LearningExample", back_populates="profile", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_strategy_profile_status", "status"),
        Index("ix_strategy_profile_short_code", "short_code"),
    )


class LearningSession(Base):
    """
    A teaching session where user provides strategy examples.
    Groups multiple LearningExamples into one learning batch.
    """
    __tablename__ = "learning_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_profile_id = Column(UUID(as_uuid=True), ForeignKey("strategy_profiles.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), nullable=False, default="active")  # active, completed, abandoned
    session_type = Column(String(30), nullable=False, default="initial_teaching")  # initial_teaching, refinement, correction

    # Session data
    examples_count = Column(Integer, default=0)
    notes = Column(Text, nullable=True)
    ai_summary = Column(Text, nullable=True)  # AI's summary of what was learned

    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    profile = relationship("StrategyProfile", back_populates="learning_sessions")
    examples = relationship("LearningExample", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_learning_session_profile", "strategy_profile_id"),
        Index("ix_learning_session_status", "status"),
    )


class LearningExample(Base):
    """
    A single teaching example: screenshot + AI vision analysis + user annotations.
    The building blocks from which StrategyProfile is constructed.
    """
    __tablename__ = "learning_examples"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id", ondelete="CASCADE"), nullable=False)
    strategy_profile_id = Column(UUID(as_uuid=True), ForeignKey("strategy_profiles.id", ondelete="CASCADE"), nullable=False)

    # Screenshot
    screenshot_path = Column(String(500), nullable=False)
    screenshot_hash = Column(String(64), nullable=True)  # SHA-256 for dedup

    # AI analysis
    vision_analysis = Column(JSONB, default=dict)  # structured Claude Vision output
    ai_interpretation = Column(Text, nullable=True)  # narrative combining vision + annotations

    # User input
    user_annotations = Column(JSONB, default=dict)  # entry, stop, target, direction, reasoning
    is_positive_example = Column(Boolean, default=True)  # False = counter-example ("NOT a valid setup")

    # Context
    market_context = Column(JSONB, default=dict)  # regime, time, volatility
    embedding_id = Column(String(100), nullable=True)  # ChromaDB vector reference

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("LearningSession", back_populates="examples")
    profile = relationship("StrategyProfile", back_populates="learning_examples")

    __table_args__ = (
        Index("ix_learning_example_session", "session_id"),
        Index("ix_learning_example_profile", "strategy_profile_id"),
    )


# ══════════════════════════════════════════════
#  WATCHLISTS — Master + Strategy-linked
# ══════════════════════════════════════════════

class Watchlist(Base):
    """
    A watchlist of symbols. Can be:
    - master: user-curated universe of stocks to monitor
    - strategy: auto-populated by system matching stocks to a learned strategy
    """
    __tablename__ = "watchlists"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    watchlist_type = Column(String(20), nullable=False, default="master")  # master | strategy
    strategy_profile_id = Column(UUID(as_uuid=True), ForeignKey("strategy_profiles.id"), nullable=True)

    # Owner
    trader_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    auto_scan = Column(Boolean, default=True)  # whether system auto-analyzes this watchlist
    last_scanned_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    items = relationship("WatchlistItem", back_populates="watchlist", cascade="all, delete-orphan")
    strategy_profile = relationship("StrategyProfile", backref="watchlists")

    __table_args__ = (
        Index("ix_watchlist_type", "watchlist_type"),
        Index("ix_watchlist_strategy", "strategy_profile_id"),
    )


class WatchlistItem(Base):
    """A single symbol in a watchlist with analysis metadata."""
    __tablename__ = "watchlist_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    watchlist_id = Column(UUID(as_uuid=True), ForeignKey("watchlists.id", ondelete="CASCADE"), nullable=False)
    symbol = Column(String(30), nullable=False)
    added_by = Column(String(20), default="user")  # user | system
    status = Column(String(20), default="active")  # active | filtered_out | matched | archived

    # Analysis results (filled by watchlist analyzer)
    match_score = Column(Float, nullable=True)  # 0-1 how well it fits the strategy
    match_reason = Column(Text, nullable=True)  # why system thinks this fits
    last_analysis = Column(JSONB, default=dict)  # regime, trend, pattern, levels
    regime_at_add = Column(String(30), nullable=True)

    # Source tracking (which master watchlist item triggered this)
    source_watchlist_id = Column(UUID(as_uuid=True), nullable=True)
    source_item_id = Column(UUID(as_uuid=True), nullable=True)

    added_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    watchlist = relationship("Watchlist", back_populates="items")

    __table_args__ = (
        UniqueConstraint("watchlist_id", "symbol", name="uq_watchlist_symbol"),
        Index("ix_watchlist_item_symbol", "symbol"),
        Index("ix_watchlist_item_status", "status"),
    )


class TradeFeedback(Base):
    """
    Continuous learning: feedback from closed trades analyzed against learned profiles.
    Each record = one trade's lessons fed back into the strategy profile.
    """
    __tablename__ = "trade_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id"), nullable=False)
    strategy_profile_id = Column(UUID(as_uuid=True), ForeignKey("strategy_profiles.id"), nullable=False)

    # Analysis
    adherence_score = Column(Float, nullable=True)  # 0-100
    deviations = Column(JSONB, default=list)  # [{aspect, description, severity}]
    strengths = Column(JSONB, default=list)
    lessons_learned = Column(JSONB, default=list)  # extracted insights
    outcome_alignment = Column(String(20), nullable=True)  # win_as_expected, loss_as_expected, surprise_win, surprise_loss

    # Trade outcome snapshot
    pnl = Column(Float, nullable=True)
    r_multiple = Column(Float, nullable=True)
    was_profitable = Column(Boolean, nullable=True)

    # Profile impact
    profile_updated = Column(Boolean, default=False)  # did this feedback trigger profile rebuild
    confidence_delta = Column(JSONB, default=dict)  # change in confidence scores

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    trade = relationship("Trade", backref="trade_feedback")
    strategy_profile = relationship("StrategyProfile", backref="trade_feedbacks")

    __table_args__ = (
        Index("ix_trade_feedback_trade", "trade_id"),
        Index("ix_trade_feedback_profile", "strategy_profile_id"),
    )


# ══════════════════════════════════════════════
#  STRATEGY KNOWLEDGE ENGINE (SKE)
# ══════════════════════════════════════════════

class StrategyKnowledgeObject(Base):
    """
    A Strategy Knowledge Object (SKO) — rich quantitative representation of a
    trading strategy using fuzzy scoring, variant clustering, and conviction-based sizing.
    """
    __tablename__ = "strategy_knowledge_objects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    short_code = Column(String(30), nullable=False, unique=True)
    status = Column(SAEnum(SKOStatus), nullable=False, default=SKOStatus.HYPOTHESIS)

    # Owner
    trader_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)

    # Descriptions
    description = Column(Text, nullable=True)
    narrative = Column(Text, nullable=True)  # evolving NL description

    # Tier & stats
    tier = Column(Integer, default=0)
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    expectancy = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)

    # Fuzzy scoring config: {feature_name: {type, center, spread, weight, invert, ...}}
    scoring_config = Column(JSONB, default=dict)

    # Conviction mapping: {skip: [0, 0.3], small: [0.3, 0.5], ...}
    conviction_map = Column(JSONB, default=dict)

    # Zone-based entry: [{feature, lower_bound, upper_bound, sweet_spot}]
    entry_zones = Column(JSONB, default=list)

    # Exit rules: {stop_method, atr_multiple, target_zones, trailing}
    exit_rules = Column(JSONB, default=dict)

    # Risk: {base_risk_pct, max_risk_pct, rr_minimum}
    risk_parameters = Column(JSONB, default=dict)

    # Multi-dimensional regime fingerprint: {preferred: [...], avoid: [...]}
    regime_fingerprint = Column(JSONB, default=dict)

    # IST session performance
    time_preferences = Column(JSONB, default=dict)

    # DQAL
    dqal_score = Column(Float, nullable=True)
    dqal_grade = Column(String(1), nullable=True)

    # Metadata
    tags = Column(JSONB, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_signal_at = Column(DateTime(timezone=True), nullable=True)

    # ── Divergence snapshot (cached from dual-model comparison) ──
    filtered_win_rate = Column(Float, nullable=True)
    unfiltered_win_rate = Column(Float, nullable=True)
    filtered_expectancy_r = Column(Float, nullable=True)
    unfiltered_expectancy_r = Column(Float, nullable=True)
    behavioral_cost_per_trade_r = Column(Float, nullable=True)
    centroid_divergence_cosine = Column(Float, nullable=True)
    divergence_interpretation = Column(String(50), nullable=True)  # improving, stable, regressing
    divergence_computed_at = Column(DateTime(timezone=True), nullable=True)
    effective_sample_size = Column(Float, nullable=True)

    # ── Quality gate config ──
    quality_gate_config = Column(JSONB, default=dict)
    # {half_life_days, min_tqw_threshold, alignment_mode, ...}
    temporal_decay_half_life = Column(Float, nullable=True, default=120.0)

    # Relationships
    variants = relationship("SKOVariant", back_populates="sko", cascade="all, delete-orphan")
    signals = relationship("SKOSignal", back_populates="sko", cascade="all, delete-orphan")
    divergence_history = relationship("StrategyDivergenceHistory", back_populates="sko", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_sko_status_tier", "status", "tier"),
        Index("ix_sko_short_code", "short_code"),
    )


class SKOVariant(Base):
    """
    A discovered variant (or anti-variant) within an SKO — a cluster of trades
    sharing similar feature-vector patterns with distinct performance characteristics.
    """
    __tablename__ = "sko_variants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sko_id = Column(UUID(as_uuid=True), ForeignKey("strategy_knowledge_objects.id", ondelete="CASCADE"), nullable=False)
    variant_type = Column(String(20), nullable=False, default="variant")  # variant | anti_variant

    label = Column(String(100), nullable=False)  # e.g., "high_compression_breakout"
    centroid = Column(JSONB, default=dict)  # feature vector centroid
    trade_count = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    avg_r = Column(Float, nullable=True)
    expectancy = Column(Float, nullable=True)

    # [{feature, mean, std, importance}]
    distinguishing_features = Column(JSONB, default=list)

    # Enrichment fields (matching ske/models.py VariantInfo)
    narrative = Column(Text, nullable=True)
    regime_affinity = Column(JSONB, nullable=True)  # serialized RegimeFingerprint
    scoring_overrides = Column(JSONB, nullable=True)
    provenance = Column(String(20), nullable=False, server_default="system")  # system | user_taught | hybrid
    taught_examples_count = Column(Integer, default=0)
    tags = Column(JSONB, nullable=True)  # list of strings
    preferred_timeframes = Column(JSONB, nullable=True)  # list of strings
    confidence_level = Column(Float, default=0.0)

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    sko = relationship("StrategyKnowledgeObject", back_populates="variants")

    __table_args__ = (
        Index("ix_sko_variant_sko", "sko_id"),
        Index("ix_sko_variant_type", "variant_type"),
        Index("ix_sko_variant_provenance", "provenance"),
    )


class SKOSignal(Base):
    """
    A real-time signal generated by evaluating market data against an SKO's
    fuzzy scoring config. Tracks conviction, sizing, and eventual outcome.
    """
    __tablename__ = "sko_signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sko_id = Column(UUID(as_uuid=True), ForeignKey("strategy_knowledge_objects.id", ondelete="CASCADE"), nullable=False)
    variant_id = Column(UUID(as_uuid=True), ForeignKey("sko_variants.id", ondelete="SET NULL"), nullable=True)

    symbol = Column(String(30), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    # Feature data
    feature_vector = Column(JSONB, default=dict)
    individual_scores = Column(JSONB, default=dict)  # {feature: score}

    # Conviction
    conviction_score = Column(Float, nullable=False)
    conviction_band = Column(String(20), nullable=False)

    # Sizing
    sizing_recommendation = Column(String(20), nullable=True)
    size_multiplier = Column(Float, nullable=True)

    # Regime context
    regime_fingerprint = Column(JSONB, default=dict)
    regime_match_score = Column(Float, nullable=True)

    # Trade levels
    entry_zone = Column(JSONB, default=dict)  # {lower, upper, sweet_spot}
    stop_level = Column(Float, nullable=True)
    target_levels = Column(JSONB, default=list)

    # Status & outcome
    status = Column(String(20), nullable=False, default="pending")  # pending, taken, skipped, expired
    outcome_r = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    sko = relationship("StrategyKnowledgeObject", back_populates="signals")
    variant = relationship("SKOVariant")

    __table_args__ = (
        Index("ix_sko_signal_sko", "sko_id"),
        Index("ix_sko_signal_symbol", "symbol"),
        Index("ix_sko_signal_status", "status"),
        Index("ix_sko_signal_created", "created_at"),
    )


# ══════════════════════════════════════════════
#  STRATEGY DIVERGENCE HISTORY
# ══════════════════════════════════════════════

class StrategyDivergenceHistory(Base):
    """
    Time-series record of divergence between filtered (quality-gated) and
    unfiltered (all trades) SKO models. Tracks behavioral cost over time.
    """
    __tablename__ = "strategy_divergence_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sko_id = Column(UUID(as_uuid=True), ForeignKey("strategy_knowledge_objects.id", ondelete="CASCADE"), nullable=False)

    computed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Raw counts
    raw_trade_count = Column(Integer, nullable=False, default=0)
    filtered_trade_count = Column(Integer, nullable=False, default=0)
    excluded_trade_count = Column(Integer, nullable=False, default=0)
    effective_sample_size = Column(Float, nullable=True)

    # Unfiltered model stats (all trades)
    unfiltered_win_rate = Column(Float, nullable=True)
    unfiltered_expectancy_r = Column(Float, nullable=True)
    unfiltered_profit_factor = Column(Float, nullable=True)
    unfiltered_avg_winner_r = Column(Float, nullable=True)
    unfiltered_avg_loser_r = Column(Float, nullable=True)
    unfiltered_sharpe = Column(Float, nullable=True)

    # Filtered model stats (quality-gated trades only)
    filtered_win_rate = Column(Float, nullable=True)
    filtered_expectancy_r = Column(Float, nullable=True)
    filtered_profit_factor = Column(Float, nullable=True)
    filtered_avg_winner_r = Column(Float, nullable=True)
    filtered_avg_loser_r = Column(Float, nullable=True)
    filtered_sharpe = Column(Float, nullable=True)

    # Divergence metrics
    win_rate_delta = Column(Float, nullable=True)  # filtered - unfiltered
    expectancy_delta_r = Column(Float, nullable=True)
    profit_factor_delta = Column(Float, nullable=True)
    behavioral_cost_per_trade_r = Column(Float, nullable=True)
    centroid_divergence_cosine = Column(Float, nullable=True)

    # Interpretation
    divergence_interpretation = Column(String(50), nullable=True)
    # "improving" = behavioral cost shrinking, "stable", "regressing" = cost growing

    # Feature-level divergence (which features shift most between filtered/unfiltered)
    feature_divergence = Column(JSONB, default=dict)
    # {feature_name: {filtered_mean, unfiltered_mean, shift, significance}}

    # Exclusion breakdown
    exclusion_breakdown = Column(JSONB, default=dict)
    # {behavioral_exclusion: N, low_tqw: N, alignment_fail: N, override_downweight: N}

    # Relationships
    sko = relationship("StrategyKnowledgeObject", back_populates="divergence_history")

    __table_args__ = (
        Index("ix_divergence_sko_computed", "sko_id", "computed_at"),
        Index("ix_divergence_computed", "computed_at"),
    )


# ══════════════════════════════════════════════
#  DAILY PERFORMANCE SUMMARY
# ══════════════════════════════════════════════

class DailyPerformanceSummary(Base):
    """Aggregated daily trading performance metrics."""
    __tablename__ = "daily_performance_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(Date, nullable=False, index=True)

    # Owner
    trader_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)

    total_trades = Column(Integer, default=0)
    winners = Column(Integer, default=0)
    losers = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    total_r = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    avg_r = Column(Float, default=0.0)

    rule_adherence_rate = Column(Float, default=100.0)
    avg_decision_quality = Column(Float, nullable=True)
    deviations_count = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ══════════════════════════════════════════════
#  AUTOMATION READINESS SCORE
# ══════════════════════════════════════════════

class AutomationReadinessScore(Base):
    """Per-strategy automation readiness assessment."""
    __tablename__ = "automation_readiness_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_name = Column(String(100), nullable=False, index=True)

    # Owner
    trader_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)

    rule_clarity = Column(Float, default=0.0)
    rule_consistency = Column(Float, default=0.0)
    performance_stability = Column(Float, default=0.0)
    edge_confidence = Column(Float, default=0.0)
    behavioral_discipline = Column(Float, default=0.0)
    overall_readiness = Column(Float, default=0.0)

    current_phase = Column(SAEnum(AutomationPhase), default=AutomationPhase.OBSERVATION)
    recommended_phase = Column(SAEnum(AutomationPhase), default=AutomationPhase.OBSERVATION)

    blockers = Column(JSONB, default=list)
    recommendations = Column(JSONB, default=list)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_readiness_strategy_created", "strategy_name", "created_at"),
    )


# ══════════════════════════════════════════════
#  TOKEN BLACKLIST (for JWT revocation)
# ══════════════════════════════════════════════

class TokenBlacklist(Base):
    """Revoked JWT tokens — checked on every authenticated request."""
    __tablename__ = "token_blacklist"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    jti = Column(String(64), nullable=False, unique=True, index=True)  # JWT ID
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_type = Column(String(10), nullable=False, default="access")  # access | refresh
    expires_at = Column(DateTime(timezone=True), nullable=False)  # auto-cleanup after expiry
    revoked_at = Column(DateTime(timezone=True), server_default=func.now())


# ══════════════════════════════════════════════
#  ORDER INTENTS (execution durability journal)
# ══════════════════════════════════════════════

class OrderIntent(Base):
    """
    Every order is written here BEFORE it is sent to the broker.
    This ensures durability: if the process crashes after broker submission
    but before event publish, the intent record proves the order was placed.

    Idempotency: callers should provide a stable idempotency_key (e.g.
    f"{trade_id}:{direction}:{quantity}"). Duplicate keys return the cached
    result without re-submitting.
    """
    __tablename__ = "order_intents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    idempotency_key = Column(String(128), nullable=False, unique=True, index=True)

    # Order details snapshot
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)      # long | short
    quantity = Column(Integer, nullable=False)
    order_type = Column(String(20), nullable=False)
    price = Column(Float, nullable=True)
    product = Column(String(20), nullable=True)
    tag = Column(String(100), nullable=True)
    trade_id = Column(String(100), nullable=True, index=True)

    # Execution context
    broker_name = Column(String(50), nullable=False, default="paper")
    is_live = Column(Boolean, default=False)
    request_payload = Column(JSONB, server_default="{}")

    # Lifecycle
    status = Column(String(20), nullable=False, server_default="PENDING")
    # PENDING → SUBMITTED → FILLED | REJECTED | CANCELLED | ERROR
    broker_order_id = Column(String(100), nullable=True)
    response_payload = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_order_intents_symbol_created", "symbol", "created_at"),
        Index("ix_order_intents_status", "status"),
    )


# ══════════════════════════════════════════════
#  SYSTEM FLAGS (kill switch + operator controls)
# ══════════════════════════════════════════════

class SystemFlag(Base):
    """
    Key/value store for operator-controlled system flags.

    Kill switch:
        key="kill_switch", value="active"  → all live order submission blocked
        DELETE or value="inactive"         → orders allowed again
    """
    __tablename__ = "system_flags"

    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=False, default="")
    reason = Column(Text, nullable=True)    # human-readable note set by operator
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ══════════════════════════════════════════════
#  RESEARCH FINDINGS (Phase 4 + 5 outputs)
# ══════════════════════════════════════════════

class ResearchFinding(Base):
    """
    Findings emitted by the research agent — decay alerts, emergence alerts,
    regime target shifts, reversal cliffs, and meta-correlations.

    Real-money safety:
      - status starts as 'pending_review'
      - NEVER auto-applied to live strategies — trader must accept
      - Approved findings still go through the AVT proof cycle before live use
    """
    __tablename__ = "research_findings"

    id = Column(String(64), primary_key=True)

    # Categorization
    type = Column(String(40), nullable=False, index=True)
    # Values: decay_detected | edge_emerging | regime_target_shift |
    #         reversal_cliff | new_pattern | regime_shift (meta) | time_slot_risk (meta)

    # Context
    strategy_id = Column(String(64), nullable=True, index=True)
    symbol = Column(String(32), nullable=True)
    detected_at = Column(DateTime(timezone=True), nullable=False, index=True)

    # Statistics
    confidence = Column(String(16), nullable=False)   # high|moderate|low
    p_value = Column(Float, nullable=True)
    sample_size = Column(Integer, nullable=True)

    # Human-readable
    title = Column(String(255), nullable=False)
    summary = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    suggested_action = Column(Text, nullable=True)

    # Full detector payload
    payload = Column(JSONB, nullable=False, default=dict)

    # Workflow
    status = Column(String(20), nullable=False, default="pending_review", index=True)
    # Values: pending_review | accepted | rejected | in_proof_cycle | proven | expired
    status_history = Column(JSONB, nullable=False, default=list)

    # Cross-correlation linkage (for meta-findings)
    related_finding_ids = Column(JSONB, nullable=True)

    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_research_findings_strategy_status", "strategy_id", "status"),
    )


# ══════════════════════════════════════════════
#  USER-INITIATED CONTINUOUS RESEARCH (Phase 4 enhanced)
# ══════════════════════════════════════════════

class ResearchTask(Base):
    """
    User-initiated continuous discovery task.

    User clicks "Start Continuous Research" with a theme + scope.
    Background worker enumerates (symbol, pattern_type) candidates and runs
    the existing discovery pipeline on each, persisting validated results
    as DiscoveryDraft rows.

    No automatic trading anywhere in this loop — drafts await user "Send to AVT".
    """
    __tablename__ = "research_tasks"

    id = Column(String(64), primary_key=True)
    trader_id = Column(String(64), nullable=True, index=True)

    # Task config (set by user at creation time)
    theme = Column(String(255), nullable=True)            # human hint, e.g. "momentum on FNO"
    scope = Column(String(64), nullable=False)            # fno | nifty50 | comma-separated tickers
    symbols = Column(JSONB, nullable=False, default=list)
    pattern_types = Column(JSONB, nullable=False, default=list)
    lookback_years = Column(Integer, nullable=False, default=2)
    similarity_threshold = Column(Float, nullable=False, default=0.85)

    # Validation thresholds — drafts must beat these to be saved
    min_sample_size = Column(Integer, nullable=False, default=30)
    min_win_rate = Column(Float, nullable=False, default=0.55)
    min_avg_r = Column(Float, nullable=False, default=0.5)

    # Progress
    status = Column(String(20), nullable=False, default="pending", index=True)
    # pending | running | stopped | completed | failed
    candidates_planned = Column(Integer, nullable=False, default=0)
    candidates_completed = Column(Integer, nullable=False, default=0)
    candidates_validated = Column(Integer, nullable=False, default=0)
    current_candidate = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)

    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    stopped_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class DiscoveryDraft(Base):
    """
    A validated strategy candidate awaiting user decision.

    Created when ContinuousResearchTask finds a (symbol, pattern_type) combo
    that passes the task's validation thresholds (sample_size, win_rate, avg_r).

    Workflow:
      pending_review → user clicks "Send to AVT" → strategy YAML written →
      AVT picks up via existing hot-reload → AVT virtual-trades it.
    """
    __tablename__ = "discovery_drafts"

    id = Column(String(64), primary_key=True)
    task_id = Column(String(64), nullable=True, index=True)
    trader_id = Column(String(64), nullable=True, index=True)

    # Source reference
    discovery_id = Column(String(64), nullable=True)
    symbol = Column(String(32), nullable=False)
    pattern_type = Column(String(64), nullable=False)

    # Backtest stats (denormalized for fast listing)
    sample_size = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)
    avg_r = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=True)
    best_regime = Column(String(64), nullable=True)
    best_time_slot = Column(String(64), nullable=True)
    htf_alignment = Column(String(64), nullable=True)       # from Phase 1
    optimal_target_r = Column(Float, nullable=True)         # from Phase 2
    reversal_cliff_r = Column(Float, nullable=True)         # from Phase 2

    # Strategy YAML
    strategy_name = Column(String(255), nullable=False)
    strategy_yaml = Column(Text, nullable=False)
    strategy_dna = Column(JSONB, nullable=False, default=dict)

    # Workflow
    status = Column(String(20), nullable=False, default="pending_review", index=True)
    # pending_review | sent_to_avt | discarded
    sent_to_avt_at = Column(DateTime(timezone=True), nullable=True)
    avt_strategy_id = Column(String(64), nullable=True)
    yaml_path = Column(String(512), nullable=True)
    user_note = Column(Text, nullable=True)

    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_discovery_drafts_task_status", "task_id", "status"),
        Index("ix_discovery_drafts_trader_status", "trader_id", "status"),
    )


# ══════════════════════════════════════════════
#  BEHAVIORAL AGENT — Trade Replay + Profile
# ══════════════════════════════════════════════

class TradeReplay(Base):
    """
    Per-trade reconstruction of market state at entry/exit timestamps.
    One row per trade. Reproducible: same trade always produces same replay.
    Migration 031.
    """
    __tablename__ = "trade_replays"

    id = Column(String(64), primary_key=True)  # str(trade_id) for portability
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id", ondelete="CASCADE"),
                      nullable=False, unique=True, index=True)

    tier = Column(String(16), nullable=False, default="standard")
    # full | standard | sample

    # Quick-access denormalized fields
    symbol = Column(String(32), nullable=False, index=True)
    entry_time = Column(DateTime(timezone=True), nullable=False, index=True)
    exit_time = Column(DateTime(timezone=True), nullable=True)

    regime_at_entry = Column(String(40), nullable=True)
    regime_at_exit = Column(String(40), nullable=True)
    trend_alignment = Column(String(20), nullable=True)

    peak_r = Column(Float, nullable=True)
    final_r = Column(Float, nullable=True)
    early_exit_severity = Column(Float, nullable=True)
    move_pct_complete_at_entry = Column(Float, nullable=True)
    rsi_at_entry = Column(Float, nullable=True)

    # Full payload (snapshots, observations, excursion path) as JSONB
    payload = Column(JSONB, nullable=False, default=dict)

    candles_fetched = Column(Integer, nullable=False, default=0)
    fetch_errors = Column(JSONB, nullable=True)

    replayed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_trade_replays_symbol_entry", "symbol", "entry_time"),
    )


class BehaviorProfile(Base):
    """
    Per-trader rolling synthesis of behavioral patterns from trade replays.

    Real-money safety: capped to 1-3 strengths + 1-3 weaknesses + 1 biggest_issue.
    Don't lecture — surface only the highest-leverage patterns.
    """
    __tablename__ = "behavior_profiles"

    id = Column(String(64), primary_key=True)
    trader_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                       nullable=False, unique=True, index=True)

    strengths = Column(JSONB, nullable=False, default=list)
    weaknesses = Column(JSONB, nullable=False, default=list)
    biggest_issue = Column(JSONB, nullable=True)
    # {pattern, frequency, impact_text, sample_size, ...}

    best_strategy = Column(String(64), nullable=True)
    worst_pattern = Column(String(64), nullable=True)

    discipline_score = Column(Float, nullable=True)
    causal_summary = Column(JSONB, nullable=False, default=list)
    # Top 3 cause-tag patterns with frequencies

    archetype = Column(String(64), nullable=True)
    # From ml/behavior_clustering.py result

    trades_analyzed_count = Column(Integer, nullable=False, default=0)
    first_trade_at = Column(DateTime(timezone=True), nullable=True)
    last_trade_at = Column(DateTime(timezone=True), nullable=True)
    last_analyzed_at = Column(DateTime(timezone=True), nullable=True)
    last_full_recompute_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class StrategyContextPerformance(Base):
    """
    Edge-decay matrix (Phase 2): per (trader, strategy, context) performance,
    aggregated EOD from the CLEAN AVT strategy-research record
    (marketdata/strategy_research) — perfect-discipline executions, NOT raw user
    trades (which are behaviourally contaminated).

    context_key examples: 'regime:breakout', 'vol:high', 'session:morning',
    'dir:long'. This is how PTIS detects edge decay —
    "ORB edge falling in ranging regimes, VWAP edge rising".
    """
    __tablename__ = "strategy_context_performance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    trader_id = Column(String(50), nullable=False, index=True)
    strategy_code = Column(String(64), nullable=False)
    context_key = Column(String(64), nullable=False)

    # Full-window stats
    trade_count = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=True)        # 0-1
    avg_r = Column(Float, nullable=True)
    expectancy = Column(Float, nullable=True)      # mean R (== avg_r here)
    total_r = Column(Float, nullable=True)

    # Recent window (the edge-decay signal)
    last_n = Column(Integer, nullable=True)
    last_n_win_rate = Column(Float, nullable=True)
    last_n_avg_r = Column(Float, nullable=True)
    edge_trend = Column(String(16), nullable=True)  # strengthening|weakening|stable

    sample_window_start = Column(DateTime(timezone=True), nullable=True)
    sample_window_end = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("trader_id", "strategy_code", "context_key", name="uq_scp_trader_strat_ctx"),
        Index("ix_scp_trader_strategy", "trader_id", "strategy_code"),
    )


class StrategySymbolPerformance(Base):
    """
    Per-(strategy, stock[, regime, market_state]) performance — the INITIAL RANKING
    baseline. Answers "which stocks does ORB work best on, in which context?" so a
    freshly-taught strategy isn't ranked blind on day 1.

    SOURCE INVARIANT — BENCHMARK REPLAY ONLY: aggregated from the CLEAN AVT
    strategy-research record (marketdata/strategy_research), i.e. frozen-v1.0
    perfect-discipline executions. NEVER from user trades — user mistakes must not
    pollute strategy ranking (the Strategy ≠ User boundary).

    Confidence layer (effective_sample_size, confidence_score) ensures a thin
    3-trade/100% cell never outranks a 150-trade/58% cell.
    """
    __tablename__ = "strategy_symbol_performance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    trader_id = Column(String(50), nullable=False, index=True)
    strategy_code = Column(String(64), nullable=False)
    symbol = Column(String(32), nullable=False)

    # Context (nullable / sentinel for the not-yet-populated dims):
    #   regime: composite regime, or '*' for the symbol-overall (all-regime) row
    #   market_state: compression|expansion|trend|exhaustion (populated later; '' for now)
    #   strategy_version: '1.0' for benchmark; future per-version intelligence
    regime = Column(String(32), nullable=True)
    market_state = Column(String(32), nullable=True)
    strategy_version = Column(String(16), nullable=True)

    # Full-window stats
    trade_count = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=True)        # 0-1
    avg_r = Column(Float, nullable=True)
    expectancy = Column(Float, nullable=True)      # mean R (== avg_r here)
    total_r = Column(Float, nullable=True)

    # Recent window (recency signal)
    last_n = Column(Integer, nullable=True)
    last_n_avg_r = Column(Float, nullable=True)

    # Confidence — so small samples don't outrank large ones
    effective_sample_size = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)  # 0-1

    sample_window_start = Column(DateTime(timezone=True), nullable=True)
    sample_window_end = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "trader_id", "strategy_code", "symbol", "regime", "market_state",
            "strategy_version", name="uq_ssp_trader_strat_symbol_ctx",
        ),
        Index("ix_ssp_trader_strategy", "trader_id", "strategy_code"),
        Index("ix_ssp_trader_strategy_symbol", "trader_id", "strategy_code", "symbol"),
    )