"""
Research Agent — Autonomous strategy research & decay/emergence detection.

Phase 5 (built first):
  - decay_detector.py: rolling Mann-Whitney + CUSUM + Page-Hinkley with majority vote
  - emergence_detector.py: regime+strategy co-emergence detection
  - models.py: shared finding dataclasses

Phase 4 (built later):
  - scheduler.py: nightly autonomous exploration loop
  - explorer.py: 40/30/30 explore-exploit policy
  - aggregator.py: cross-correlation meta-findings
  - report_generator.py: weekly markdown report
"""

from research_agent.models import (
    DecayFinding,
    EmergenceFinding,
    DecayConfidence,
)
from research_agent.decay_detector import DecayDetector
from research_agent.emergence_detector import EmergenceDetector
from research_agent.confidence_tracker import (
    ConfidencePoint,
    ConfidenceCurve,
    ConfidenceAlert,
    ConfidenceTracker,
    get_confidence_tracker,
)

__all__ = [
    # Decay (binary alarm)
    "DecayFinding",
    "DecayConfidence",
    "DecayDetector",
    # Emergence
    "EmergenceFinding",
    "EmergenceDetector",
    # Confidence (continuous trajectory)
    "ConfidencePoint",
    "ConfidenceCurve",
    "ConfidenceAlert",
    "ConfidenceTracker",
    "get_confidence_tracker",
]
