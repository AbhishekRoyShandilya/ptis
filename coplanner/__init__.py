# coplanner/__init__.py
"""
Co-Planner Intelligence Layer — Engine-Driven Trading Intelligence.

ENGINE computes → LLM explains. Scoring by LLM: NEVER. Ranking by LLM: NEVER.

Modules:
    engine          — CoPlannerEngine orchestrator (main entry point)
    conviction_engine — 5-component numerical conviction scoring
    strategy_regime_matrix — strategy × regime statistical backbone
    rr_enforcer     — risk-reward gate with dynamic conviction adjustment
    cross_stock_comparator — head-to-head setup comparison
    opportunity_ranker — 7-factor multi-dimensional ranking
    behavioral_guard — planning-stage behavioral checks
    db_interface    — SQLAlchemy adapter for co-planner data
    models          — data models and enums
"""

from coplanner.engine import CoPlannerEngine
from coplanner.models import (
    BehavioralCheckResult,
    ConvictionLevel,
    ConvictionResult,
    PlanSessionResult,
    RankedOpportunity,
    RRAssessment,
    SetupData,
    TradeRecord,
)
from coplanner.data_trust_layer import DataTrustLayer
from coplanner.trade_quality_classifier import TradeQualityClassifier
from coplanner.integration_adapter import (
    CoPlanEngineAdapter,
    TradeJournalAdapter,
    WatchlistAdapter,
)

__all__ = [
    "CoPlannerEngine",
    "BehavioralCheckResult",
    "ConvictionLevel",
    "ConvictionResult",
    "PlanSessionResult",
    "RankedOpportunity",
    "RRAssessment",
    "SetupData",
    "TradeRecord",
    "DataTrustLayer",
    "TradeQualityClassifier",
    "CoPlanEngineAdapter",
    "TradeJournalAdapter",
    "WatchlistAdapter",
]
