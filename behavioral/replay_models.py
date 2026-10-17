"""
Behavioral Agent — Replay Data Models
======================================

Dataclasses describing the result of replaying a single trade backwards
through historical market data.

The Trade Replay Engine reconstructs:
  - Market state at entry timestamp (price levels, indicators, regime)
  - Market state at exit timestamp
  - Bar-by-bar excursion between entry and exit (for MFE/MAE)
  - Peak favorable / adverse points (with timestamps)

Three replay tiers — driven by trade age (set in TradeReplayEngine.replay_trade):
  - FULL     (≤ 30 days):   1m + 5m + 15m + 1h candles, all indicators
  - STANDARD (31-90 days):  15m + 1h candles, core indicators
  - SAMPLE   (> 90 days):   summary only — entry/exit prices vs SMA20

These dataclasses describe the FULL output. STANDARD/SAMPLE results leave
the higher-tier fields as None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── Tier labels ─────────────────────────────────────────────────────


REPLAY_TIER_FULL = "full"
REPLAY_TIER_STANDARD = "standard"
REPLAY_TIER_SAMPLE = "sample"


# ── Indicator snapshot at a point in time ──────────────────────────


@dataclass
class IndicatorSnapshot:
    """Indicator values at a single bar timestamp."""
    timestamp: datetime
    timeframe: str               # "1m" | "5m" | "15m" | "1h"

    # Price levels
    close: float
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    volume: Optional[float] = None

    # Moving averages
    ema_9: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    sma_20: Optional[float] = None

    # Volatility / momentum
    rsi_14: Optional[float] = None
    atr_14: Optional[float] = None
    atr_pct: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width_pct: Optional[float] = None

    # Volume / VWAP (intraday)
    vwap: Optional[float] = None
    volume_ratio: Optional[float] = None  # current vol / 20-bar avg

    # Regime
    regime_label: Optional[str] = None    # dominant probabilistic regime
    regime_probs: Dict[str, float] = field(default_factory=dict)
    regime_confidence: Optional[float] = None


# ── Bar-by-bar excursion ───────────────────────────────────────────


@dataclass
class ExcursionPoint:
    """A single in-trade snapshot used to compute MFE / MAE / regime shifts."""
    timestamp: datetime
    bar_index: int               # 0 = entry bar, +1 = next bar, etc.
    close: float
    r_at_this_bar: float         # signed R-multiple from entry to this bar's close
    mfe_so_far: float            # max favorable R reached up to this bar
    mae_so_far: float            # max adverse R reached up to this bar


# ── Behavioral observations from replay ────────────────────────────


@dataclass
class ReplayObservations:
    """
    Observations derived purely from market data + trade timestamps.
    These feed the CausalLinker to produce per-trade explanations.

    NOT a behavior detection itself — that's the linker's job. These are
    just the raw signals (e.g., "yes, price reached +2R MFE before exit").
    """
    # Late entry signals
    move_pct_complete_at_entry: Optional[float] = None  # 0-1 (0 = at start, 1 = at end)
    bars_since_breakout: Optional[int] = None            # bars elapsed since breakout/setup formation
    rsi_at_entry: Optional[float] = None                 # for late-entry detection

    # Early exit signals
    peak_r_during_trade: float = 0.0                     # max R-mult reached (gross)
    peak_r_timestamp: Optional[datetime] = None
    final_r: float = 0.0                                  # actual exit R
    early_exit_severity: float = 0.0                      # 0-1: peak_r - final_r normalized

    # Regime state
    regime_at_entry: str = ""
    regime_at_exit: str = ""
    regime_changed: bool = False

    # Trade-vs-trend alignment
    trend_alignment_at_entry: Optional[str] = None       # "with_trend" | "against_trend" | "no_trend"
    above_vwap_at_entry: Optional[bool] = None
    above_ema20_at_entry: Optional[bool] = None
    above_ema50_at_entry: Optional[bool] = None


# ── Top-level replay output ────────────────────────────────────────


@dataclass
class ReplayedTrade:
    """
    Complete replay output for a single trade.

    Stored 1:1 with `trades` table via the new `trade_replays` table
    (Phase D migration). Reproducible — same trade always produces same replay.
    """
    trade_id: str                                # UUID of the trade row (string form)
    symbol: str
    direction: str                               # "long" | "short"
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    stop_loss: Optional[float] = None

    # Replay tier (drives how rich the output is)
    tier: str = REPLAY_TIER_STANDARD

    # Market state snapshots
    snapshot_at_entry: Optional[IndicatorSnapshot] = None      # primary timeframe (15m by default)
    snapshot_at_exit: Optional[IndicatorSnapshot] = None
    snapshot_at_peak_favorable: Optional[IndicatorSnapshot] = None

    # Higher-timeframe context (always populated when tier ≥ STANDARD)
    snapshot_15m_at_entry: Optional[IndicatorSnapshot] = None
    snapshot_1h_at_entry: Optional[IndicatorSnapshot] = None

    # Bar-by-bar excursion (FULL tier only)
    excursion_path: List[ExcursionPoint] = field(default_factory=list)

    # Aggregated observations (used by CausalLinker)
    observations: ReplayObservations = field(default_factory=ReplayObservations)

    # Replay quality / debug
    candles_fetched: int = 0
    fetch_errors: List[str] = field(default_factory=list)
    replay_duration_ms: int = 0
    replayed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for DB JSONB storage. Excludes excursion_path for STANDARD/SAMPLE."""
        def _snap(s: Optional[IndicatorSnapshot]) -> Optional[dict]:
            if s is None:
                return None
            return {
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                "timeframe": s.timeframe,
                "close": s.close, "high": s.high, "low": s.low,
                "open": s.open, "volume": s.volume,
                "ema_9": s.ema_9, "ema_20": s.ema_20, "ema_50": s.ema_50,
                "sma_20": s.sma_20, "rsi_14": s.rsi_14, "atr_14": s.atr_14,
                "atr_pct": s.atr_pct, "bb_upper": s.bb_upper, "bb_lower": s.bb_lower,
                "bb_width_pct": s.bb_width_pct,
                "vwap": s.vwap, "volume_ratio": s.volume_ratio,
                "regime_label": s.regime_label, "regime_probs": s.regime_probs,
                "regime_confidence": s.regime_confidence,
            }

        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "tier": self.tier,
            "snapshot_at_entry": _snap(self.snapshot_at_entry),
            "snapshot_at_exit": _snap(self.snapshot_at_exit),
            "snapshot_at_peak_favorable": _snap(self.snapshot_at_peak_favorable),
            "snapshot_15m_at_entry": _snap(self.snapshot_15m_at_entry),
            "snapshot_1h_at_entry": _snap(self.snapshot_1h_at_entry),
            "excursion_path": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "bar_index": e.bar_index,
                    "close": e.close,
                    "r_at_this_bar": e.r_at_this_bar,
                    "mfe_so_far": e.mfe_so_far,
                    "mae_so_far": e.mae_so_far,
                }
                for e in self.excursion_path
            ] if self.tier == REPLAY_TIER_FULL else [],
            "observations": {
                "move_pct_complete_at_entry": self.observations.move_pct_complete_at_entry,
                "bars_since_breakout": self.observations.bars_since_breakout,
                "rsi_at_entry": self.observations.rsi_at_entry,
                "peak_r_during_trade": self.observations.peak_r_during_trade,
                "peak_r_timestamp": (
                    self.observations.peak_r_timestamp.isoformat()
                    if self.observations.peak_r_timestamp else None
                ),
                "final_r": self.observations.final_r,
                "early_exit_severity": self.observations.early_exit_severity,
                "regime_at_entry": self.observations.regime_at_entry,
                "regime_at_exit": self.observations.regime_at_exit,
                "regime_changed": self.observations.regime_changed,
                "trend_alignment_at_entry": self.observations.trend_alignment_at_entry,
                "above_vwap_at_entry": self.observations.above_vwap_at_entry,
                "above_ema20_at_entry": self.observations.above_ema20_at_entry,
                "above_ema50_at_entry": self.observations.above_ema50_at_entry,
            },
            "candles_fetched": self.candles_fetched,
            "fetch_errors": self.fetch_errors,
            "replay_duration_ms": self.replay_duration_ms,
            "replayed_at": self.replayed_at.isoformat() if self.replayed_at else None,
        }
