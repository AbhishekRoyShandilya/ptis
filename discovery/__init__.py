# discovery/ — Pattern Discovery Engine
#
# Turns natural-language market observations into backtested,
# regime-aware Strategy DNA files.
#
# 7-stage pipeline:
#   1. Intent Parser        — NL → PatternQuery
#   2. Event Extractor      — OHLCV → PatternFingerprint
#   3. Similarity Scanner   — cross-stock fingerprint matching
#   4. Statistical Backtester — forward-simulate matched events
#   5. Regime Correlator    — regime + time-of-day analysis
#   6. Strategy Generator   — LLM-powered YAML generation
#   7. Validator + Storer   — human review → RAG index

from discovery.models import PatternQuery, PatternFingerprint, DiscoveryReport
from discovery.orchestrator import PatternDiscoveryOrchestrator

__all__ = [
    "PatternQuery",
    "PatternFingerprint",
    "DiscoveryReport",
    "PatternDiscoveryOrchestrator",
]
