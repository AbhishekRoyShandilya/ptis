# bootstrap/__init__.py
"""
Bootstrap Understanding System
================================

Solves the cold-start problem: teaches PTIS a user's trading strategy
from day one using NL description + pattern library + fast backtest
+ iterative correction loop + judgment learning.

Components:
    PatternLibrary         — Curated domain knowledge about trading patterns
    LLMStrategyInterpreter — NL description → structured StrategyDraft
    FastBacktestEngine     — Generate synthetic trade history
    WarmStarter            — Initialize downstream systems from synthetic data
    JudgmentLearner        — Learn user's implicit judgment from accept/reject
    BootstrapOrchestrator  — End-to-end pipeline coordinator
"""

from bootstrap.pattern_library import PatternLibrary, PatternTemplate
from bootstrap.nl_interpreter import LLMStrategyInterpreter, StrategyDraft
from bootstrap.fast_backtester import FastBacktestEngine, BacktestResult, SyntheticTrade
from bootstrap.warm_starter import WarmStarter, WarmStartReport, CorrectionLoop
from bootstrap.judgment_learner import JudgmentLearner, JudgmentManager, MicroFeatureComputer
from bootstrap.orchestrator import BootstrapOrchestrator, BootstrapSession
from bootstrap.adapters import build_bootstrap_adapters
from bootstrap.live_connector import LiveJudgmentConnector, get_judgment_connector

__all__ = [
    # Core pipeline
    "PatternLibrary",
    "PatternTemplate",
    "LLMStrategyInterpreter",
    "StrategyDraft",
    "FastBacktestEngine",
    "BacktestResult",
    "SyntheticTrade",
    "WarmStarter",
    "WarmStartReport",
    "CorrectionLoop",
    # Judgment learning
    "JudgmentLearner",
    "JudgmentManager",
    "MicroFeatureComputer",
    # Live connector (Phase 2)
    "LiveJudgmentConnector",
    "get_judgment_connector",
    # Orchestration
    "BootstrapOrchestrator",
    "BootstrapSession",
    "build_bootstrap_adapters",
]
