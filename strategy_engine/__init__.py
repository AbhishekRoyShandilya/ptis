# __init__.py
# Expose commonly used strategy engine components for easier imports

from .indicators import IndicatorEngine
from .consolidation import ConsolidationDetector
from .breakout_logic import BreakoutValidator
from .risk_rules import AdaptiveRiskManager
from .confidence import ConfidenceScorer
from .regime_detector import RegimeState, RegimeDetector

__all__ = [
    "IndicatorEngine",
    "ConsolidationDetector",
    "BreakoutValidator",
    "AdaptiveRiskManager",
    "ConfidenceScorer",
    "RegimeState",
    "RegimeDetector",
]
