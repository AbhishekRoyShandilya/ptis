# discovery/models.py
"""
Shared data structures for the Pattern Discovery Engine.
Every stage reads/writes these — keeps the pipeline loosely coupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import numpy as np


# ══════════════════════════════════════════════
#  ENUMS
# ══════════════════════════════════════════════

class PatternType(str, Enum):
    MOMENTUM_BREAKOUT = "momentum_breakout"
    VOLUME_BREAKOUT = "volume_breakout"
    PULLBACK_ENTRY = "pullback_entry"
    REVERSAL = "reversal"
    GAP_PLAY = "gap_play"
    RANGE_BREAKOUT = "range_breakout"
    VWAP_RECLAIM = "vwap_reclaim"
    SQUEEZE_BREAKOUT = "squeeze_breakout"
    TREND_CONTINUATION = "trend_continuation"
    CUSTOM = "custom"


class DiscoveryStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    SCANNING = "scanning"
    BACKTESTING = "backtesting"
    CORRELATING = "correlating"
    GENERATING = "generating"
    REVIEW = "review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


# ══════════════════════════════════════════════
#  STAGE 1 OUTPUT: Parsed intent
# ══════════════════════════════════════════════

@dataclass
class PatternQuery:
    """Structured output from the Intent Parser (Stage 1)."""
    query_id: UUID = field(default_factory=uuid4)
    raw_input: str = ""

    # Extracted fields
    symbol: str = ""                        # e.g. "RELIANCE"
    timeframe: str = "15m"                  # e.g. "15m", "5m", "1h"
    event_time: Optional[str] = None        # e.g. "10:30" (IST)
    event_date: Optional[str] = None        # e.g. "2026-03-10"
    pattern_type: PatternType = PatternType.CUSTOM
    description: str = ""                   # human description of what happened
    scope: str = "fno"                      # "fno", "nifty50", "all", custom list
    similarity_threshold: float = 0.90      # 0.0 – 1.0
    lookback_years: int = 3

    def to_dict(self) -> dict:
        return {
            "query_id": str(self.query_id),
            "raw_input": self.raw_input,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "event_time": self.event_time,
            "event_date": self.event_date,
            "pattern_type": self.pattern_type.value,
            "description": self.description,
            "scope": self.scope,
            "similarity_threshold": self.similarity_threshold,
            "lookback_years": self.lookback_years,
        }


# ══════════════════════════════════════════════
#  STAGE 2 OUTPUT: Pattern fingerprint
# ══════════════════════════════════════════════

@dataclass
class PatternFingerprint:
    """
    12-dimensional vector that captures the *shape* of a market event.

    Dimensions (all normalised 0–1 or z-scored):
        0  body_ratio         — |close-open| / (high-low)  of breakout candle
        1  upper_wick_ratio   — (high - max(open,close)) / (high-low)
        2  lower_wick_ratio   — (min(open,close) - low) / (high-low)
        3  volume_surge       — breakout_vol / SMA(vol, 20)
        4  atr_expansion      — current_atr / SMA(atr, 20)
        5  momentum_score     — close-to-close change / ATR (how many ATRs it moved)
        6  consolidation_bars — # of bars in tight range before breakout (normalised)
        7  range_compression  — (range of last 10 bars) / (range of last 30 bars)
        8  trend_alignment    — EMA20 slope direction (+1 aligned, -1 counter, 0 flat)
        9  rsi_at_trigger     — RSI(14) at breakout bar (0–1 rescaled)
       10  vwap_distance      — (close - VWAP) / ATR  (signed)
       11  relative_volume    — volume / 20-day avg vol (intraday session-adjusted)
    """
    vector: np.ndarray = field(default_factory=lambda: np.zeros(12))
    labels: List[str] = field(default_factory=lambda: [
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
        "volume_surge", "atr_expansion", "momentum_score",
        "consolidation_bars", "range_compression", "trend_alignment",
        "rsi_at_trigger", "vwap_distance", "relative_volume",
    ])

    # Context for the source event
    symbol: str = ""
    timeframe: str = ""
    timestamp: Optional[datetime] = None
    bar_index: int = -1             # which bar in the candle DataFrame
    direction: int = 1              # +1 long breakout, -1 short breakout
    raw_ohlcv: Optional[Dict] = None  # the actual breakout candle data

    @property
    def dimension(self) -> int:
        return len(self.vector)

    def cosine_similarity(self, other: "PatternFingerprint") -> float:
        """Cosine similarity between two fingerprints (0–1)."""
        a, b = self.vector, other.vector
        dot = np.dot(a, b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.clip(dot / denom, 0, 1))

    def weighted_similarity(self, other: "PatternFingerprint",
                            weights: Optional[np.ndarray] = None) -> float:
        """Weighted cosine similarity — lets you emphasise volume, momentum etc."""
        if weights is None:
            # Default weights: momentum + volume matter most
            weights = np.array([
                1.0,   # body_ratio
                0.5,   # upper_wick
                0.5,   # lower_wick
                1.5,   # volume_surge
                1.2,   # atr_expansion
                1.5,   # momentum_score
                1.0,   # consolidation_bars
                1.0,   # range_compression
                1.2,   # trend_alignment
                0.8,   # rsi
                0.8,   # vwap_distance
                1.3,   # relative_volume
            ])
        a = self.vector * weights
        b = other.vector * weights
        dot = np.dot(a, b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.clip(dot / denom, 0, 1))

    def to_dict(self) -> dict:
        return {
            "vector": self.vector.tolist(),
            "labels": self.labels,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "bar_index": self.bar_index,
            "direction": self.direction,
        }


# ══════════════════════════════════════════════
#  STAGE 3 OUTPUT: Matched historical event
# ══════════════════════════════════════════════

@dataclass
class MatchedEvent:
    """A single historical occurrence that matched the query fingerprint."""
    fingerprint: PatternFingerprint
    similarity: float               # 0.0 – 1.0
    symbol: str = ""
    date: Optional[datetime] = None
    time_of_day: Optional[str] = None  # "HH:MM"
    bar_index: int = -1


# ══════════════════════════════════════════════
#  STAGE 4 OUTPUT: Backtest result per match
# ══════════════════════════════════════════════

@dataclass
class BacktestResult:
    """Forward simulation result for a single matched event."""
    match: MatchedEvent
    # P&L metrics
    r_multiple: float = 0.0         # (exit - entry) / risk, signed — NET of costs after CostModel applied
    pnl_pct: float = 0.0           # % move from entry
    won: bool = False
    # Timing
    hold_bars: int = 0              # how many bars the trade was held
    hold_minutes: int = 0
    # Drawdown
    max_adverse_excursion: float = 0.0   # worst intra-trade drawdown in R (gross — price level)
    max_favorable_excursion: float = 0.0 # best intra-trade peak in R (gross — price level)
    # Context
    exit_reason: str = ""           # "target", "stop", "time", "eod"
    regime_at_entry: str = ""
    time_of_day: str = ""
    # Higher-Timeframe context (added by MultiTFBacktester wrapper)
    htf_trend_15m: str = ""         # dominant 15m regime label (backward compat)
    htf_trend_1h: str = ""          # dominant 1h regime label (backward compat)
    htf_regime: str = ""            # composite regime (consensus across TFs)
    # Probabilistic regime (added when ProbabilisticRegimeClassifier is wired in)
    regime_probs_at_entry: dict = field(default_factory=dict)  # {label: prob, ...} at entry TF
    htf_regime_probs_15m: dict = field(default_factory=dict)   # 15m regime distribution
    htf_regime_probs_1h: dict = field(default_factory=dict)    # 1h regime distribution
    regime_confidence: float = 0.0  # max(probs) — how decisive the regime is
    # Transaction cost adjustment (added by CostModel.adjust_backtest_results)
    r_multiple_gross: float = 0.0   # original r_multiple BEFORE costs subtracted
    cost_r: float = 0.0             # transaction cost expressed as fraction of 1R
    cost_breakdown: dict = field(default_factory=dict)  # ₹ breakdown of brokerage/slippage/STT/etc


# ══════════════════════════════════════════════
#  STAGE 5 OUTPUT: Regime + time analysis
# ══════════════════════════════════════════════

@dataclass
class RegimeBreakdown:
    """Performance stats for a single market regime."""
    regime: str
    count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    median_r: float = 0.0
    profit_factor: float = 0.0
    best_r: float = 0.0
    worst_r: float = 0.0
    why_works: str = ""
    why_fails: str = ""


@dataclass
class TimeSlotBreakdown:
    """Performance stats for a time-of-day window."""
    slot: str                       # e.g. "09:15-10:00"
    count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    label: str = ""                 # "GOLDEN WINDOW", "AVOID", etc.


@dataclass
class CorrelationReport:
    """Full regime + time analysis output."""
    regime_breakdown: List[RegimeBreakdown] = field(default_factory=list)
    time_breakdown: List[TimeSlotBreakdown] = field(default_factory=list)
    best_regime: str = ""
    worst_regime: str = ""
    best_time_window: str = ""
    regime_recommendation: str = ""
    time_recommendation: str = ""


# ══════════════════════════════════════════════
#  STAGE 6 OUTPUT: Generated strategy
# ══════════════════════════════════════════════

@dataclass
class GeneratedStrategy:
    """LLM-generated strategy definition ready for human review."""
    strategy_name: str = ""
    display_name: str = ""
    short_code: str = ""
    category: str = ""
    yaml_content: str = ""          # full YAML string
    yaml_dict: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0         # how confident the system is (0–1)
    llm_reasoning: str = ""         # LLM's explanation of its choices


# ══════════════════════════════════════════════
#  FINAL OUTPUT: Discovery Report
# ══════════════════════════════════════════════

@dataclass
class DiscoveryReport:
    """Complete output of the Pattern Discovery pipeline."""
    discovery_id: UUID = field(default_factory=uuid4)
    status: DiscoveryStatus = DiscoveryStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())

    # Inputs
    query: Optional[PatternQuery] = None

    # Stage outputs
    fingerprint: Optional[PatternFingerprint] = None
    matches: List[MatchedEvent] = field(default_factory=list)
    backtest_results: List[BacktestResult] = field(default_factory=list)
    correlation: Optional[CorrelationReport] = None
    generated_strategy: Optional[GeneratedStrategy] = None

    # Aggregate stats
    total_matches: int = 0
    overall_win_rate: float = 0.0
    overall_avg_r: float = 0.0
    overall_profit_factor: float = 0.0
    median_hold_minutes: int = 0
    avg_mae: float = 0.0

    # Top stocks
    top_stocks: List[Dict[str, Any]] = field(default_factory=list)

    # Human review
    accepted: Optional[bool] = None
    reviewer_notes: str = ""
    stored_strategy_id: Optional[str] = None

    # Errors
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "discovery_id": str(self.discovery_id),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "query": self.query.to_dict() if self.query else None,
            "fingerprint": self.fingerprint.to_dict() if self.fingerprint else None,
            "total_matches": self.total_matches,
            "overall_win_rate": round(self.overall_win_rate, 3),
            "overall_avg_r": round(self.overall_avg_r, 2),
            "overall_profit_factor": round(self.overall_profit_factor, 2),
            "median_hold_minutes": self.median_hold_minutes,
            "avg_mae": round(self.avg_mae, 2),
            "top_stocks": self.top_stocks,
            "correlation": {
                "regime_breakdown": [
                    {
                        "regime": rb.regime,
                        "count": rb.count,
                        "win_rate": round(rb.win_rate, 3),
                        "avg_r": round(rb.avg_r, 2),
                        "profit_factor": round(rb.profit_factor, 2),
                        "why_works": rb.why_works,
                        "why_fails": rb.why_fails,
                    }
                    for rb in (self.correlation.regime_breakdown if self.correlation else [])
                ],
                "time_breakdown": [
                    {
                        "slot": tb.slot,
                        "count": tb.count,
                        "win_rate": round(tb.win_rate, 3),
                        "avg_r": round(tb.avg_r, 2),
                        "label": tb.label,
                    }
                    for tb in (self.correlation.time_breakdown if self.correlation else [])
                ],
                "best_regime": self.correlation.best_regime if self.correlation else "",
                "worst_regime": self.correlation.worst_regime if self.correlation else "",
                "best_time_window": self.correlation.best_time_window if self.correlation else "",
            } if self.correlation else None,
            "generated_strategy": {
                "strategy_name": self.generated_strategy.strategy_name,
                "display_name": self.generated_strategy.display_name,
                "short_code": self.generated_strategy.short_code,
                "category": self.generated_strategy.category,
                "yaml_content": self.generated_strategy.yaml_content,
                "confidence": round(self.generated_strategy.confidence, 2),
                "llm_reasoning": self.generated_strategy.llm_reasoning,
            } if self.generated_strategy else None,
            "accepted": self.accepted,
            "reviewer_notes": self.reviewer_notes,
            "stored_strategy_id": self.stored_strategy_id,
            "error": self.error,
        }
