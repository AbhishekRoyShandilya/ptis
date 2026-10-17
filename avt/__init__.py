# avt/__init__.py
"""
Autonomous Virtual Trader (AVT) Module
========================================

The AVT is the single intelligence brain of PTIS:

  1. TRADES strategies with virtual money in live market (engine.py)
  2. OBSERVES user trades for behavioral issues (observer.py)
  3. ANALYZES its own clean data via improvement_lab (self_improver.py)
  4. PROVES improvements by trading with them for 50+ trades
  5. ONLY THEN presents proven improvements to the user
  6. COMMUNICATES in natural language (conversation.py)

Architecture:
  engine.py         — scan/rank/filter/execute/manage cycle
  observer.py       — behavioral checks ONLY (discipline, overtrading, SL widen)
  triage.py         — message queue for user (no hypothesis routing)
  self_improver.py  — bridge: improvement_lab → AVT engine → proof → user
  conversation.py   — NL interface (template + LLM)
  competition.py    — user vs AVT comparison with evidence
  presenter.py      — evidence-based improvement narratives
  stock_ranker.py   — multi-dimensional stock selection
  models.py         — all data models
  db_models.py      — SQLAlchemy ORM (5 tables)
  repository.py     — database CRUD layer
  integrations.py   — DB persistence, events, WebSocket, scheduler
  market_data.py    — live market data adapters (YFinance, Broker)

improvement_lab is AVT's INTERNAL analysis engine.
It only analyzes AVT's own clean trades, never user trades directly.
"""

from avt.models import (
    AVTConfig,
    ActiveImprovement,
    CompetitionReport,
    ExitDecision,
    ImprovementPresentation,
    LIFECYCLE_ALLOCATION,
    LifecycleInfo,
    PerformanceDifference,
    PerformanceSnapshot,
    RankedCandidate,
    ScanCycleLog,
    SetupCandidate,
    SkippedSetup,
    StrategyGrade,
    StrategyLifecycleState,
    StrategyVersion,
    VirtualPortfolio,
    VirtualTrade,
    compute_performance,
)
from avt.engine import (
    AutonomousVirtualTrader,
    MarketDataInterface,
    ReplayMarketData,
    get_avt,
    init_avt,
)
from avt.stock_ranker import StockSelectionRanker
from avt.self_improver import AVTSelfImprover
from avt.competition import CompetitionEngine
from avt.presenter import AVTPresenter
from avt.integrations import (
    AVTEventHook,
    AVTPersistenceHook,
    AVTScheduler,
    setup_avt_full,
    wire_avt_engine,
    wire_avt_to_websocket,
    reload_avt_state,
)
from avt.repository import (
    AVTCompetitionRepository,
    AVTImprovementRepository,
    AVTScanLogRepository,
    AVTTradeRepository,
    AVTVersionRepository,
)
from avt.db_models import (
    AVTCalibrationSignalRow,
    AVTCompetitionSnapshotRow,
    AVTImprovementRow,
    AVTScanLogRow,
    AVTStrategyLifecycleRow,
    AVTStrategyVersionRow,
    AVTVirtualTradeRow,
)
from avt.strategy_lifecycle import StrategyHealthGate, StrategyLifecycleManager
from avt.calibration_tracker import SignalCalibrationTracker
from avt.edge_monitor import ContinuousEdgeMonitor
from avt.contrast_learner import (
    ContrastLearner,
    ContrastInsight,
    FeatureContrast,
    OutcomeExample,
    WeightAdjustment,
    ImprovementSuggestion,
)
from avt.observer import AVTObserver, Observation, ObservationType
from avt.triage import AVTTriage
from avt.conversation import AVTConversation

__all__ = [
    # Engine
    "AutonomousVirtualTrader",
    "MarketDataInterface",
    "ReplayMarketData",
    "get_avt",
    "init_avt",
    # Stock Ranker
    "StockSelectionRanker",
    # Self-Improver
    "AVTSelfImprover",
    # Competition
    "CompetitionEngine",
    # Presenter
    "AVTPresenter",
    # Integrations
    "AVTEventHook",
    "AVTPersistenceHook",
    "AVTScheduler",
    "setup_avt_full",
    "wire_avt_engine",
    "wire_avt_to_websocket",
    "reload_avt_state",
    # Repository
    "AVTTradeRepository",
    "AVTImprovementRepository",
    "AVTVersionRepository",
    "AVTScanLogRepository",
    "AVTCompetitionRepository",
    # DB Models
    "AVTVirtualTradeRow",
    "AVTStrategyVersionRow",
    "AVTImprovementRow",
    "AVTScanLogRow",
    "AVTCompetitionSnapshotRow",
    "AVTStrategyLifecycleRow",
    "AVTCalibrationSignalRow",
    # Lifecycle
    "StrategyHealthGate",
    "StrategyLifecycleManager",
    "SignalCalibrationTracker",
    "ContinuousEdgeMonitor",
    "StrategyLifecycleState",
    "StrategyGrade",
    "LifecycleInfo",
    "LIFECYCLE_ALLOCATION",
    # Pydantic Models
    "AVTConfig",
    "ActiveImprovement",
    "CompetitionReport",
    "ExitDecision",
    "ImprovementPresentation",
    "PerformanceDifference",
    "PerformanceSnapshot",
    "RankedCandidate",
    "ScanCycleLog",
    "SetupCandidate",
    "SkippedSetup",
    "StrategyVersion",
    "VirtualPortfolio",
    "VirtualTrade",
    "compute_performance",
    # Observer + Triage + Conversation
    "AVTObserver",
    "Observation",
    "ObservationType",
    "AVTTriage",
    "AVTConversation",
    # Contrast Learning (edge discovery)
    "ContrastLearner",
    "ContrastInsight",
    "FeatureContrast",
    "OutcomeExample",
    "WeightAdjustment",
    "ImprovementSuggestion",
]
