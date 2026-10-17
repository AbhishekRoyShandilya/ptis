# improvement_lab/__init__.py
"""
Improvement Lab — AVT's Internal Analysis Engine
==================================================

Analyzes AVT's own clean virtual trade data to discover improvement
hypotheses. These are NOT shown to the user directly.

Flow:
  AVT trades cleanly → improvement_lab analyzes AVT's data
  → discovers hypotheses → AVT applies top hypothesis
  → AVT trades with it for 50+ trades → proven or reverted
  → ONLY proven improvements are presented to the user

Analysis Order (simple before complex):
  1. Regime Check — is the failure in a specific regime?
  2. Variant Check — is it concentrated in an SKE variant?
  3. (Behavioral skipped for AVT — AVT has no behavioral noise)
  4. Pattern Check — ML pattern detection (RR killer, early exit, entry filter)

Two-Speed Learning:
  Fast layer: Bayesian tables + SGD (per trade, milliseconds)
  Deep layer: GBM + RF (batch retrain every 20 trades)

Key entry point for AVT:
  orchestrator.analyze_avt_trades(trades, strategy_id, version)
"""

from improvement_lab.models import (
    AnalysisLayer,
    CompositionResult,
    ExperimentStatus,
    HypothesisSource,
    ImprovementHypothesis,
    ImprovementLabReport,
    ImprovementType,
    PatternClassification,
    PatternType,
    ProofStatus,
    TradeoffAnalysis,
    ValidationProgress,
)
from improvement_lab.orchestrator import ImprovementLabOrchestrator

__all__ = [
    "AnalysisLayer",
    "CompositionResult",
    "ExperimentStatus",
    "HypothesisSource",
    "ImprovementHypothesis",
    "ImprovementLabReport",
    "ImprovementLabOrchestrator",
    "ImprovementType",
    "PatternClassification",
    "PatternType",
    "ProofStatus",
    "TradeoffAnalysis",
    "ValidationProgress",
]
