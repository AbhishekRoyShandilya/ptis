# avt/models.py
"""
Autonomous Virtual Trader — Data Models
========================================

All dataclasses, enums, and type definitions for the AVT module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

# NSE FNO realistic slippage: 0.3% per leg (entry + exit = 0.6% round-trip).
# Accounts for bid/ask spread + market impact on liquid FNO contracts.
_SLIPPAGE_PER_LEG_PCT: float = 0.003


# ══════════════════════════════════════════════════════════════════
#  ENUMS
# ══════════════════════════════════════════════════════════════════

class AVTStatus(str, Enum):
    """Overall AVT lifecycle."""
    IDLE = "idle"
    SCANNING = "scanning"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class VirtualTradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"           # partially exited


class ExitType(str, Enum):
    STOP_LOSS = "stop_loss"
    STORY_BROKEN = "story_broken"
    TARGET_HIT = "target_hit"
    TRAILING_STOP = "trailing_stop"   # ATR trailing exit (opt-in via exit_rules.trail_atr)
    TIME_EXIT = "time_exit"
    IMPROVEMENT_PARTIAL_EXIT = "improvement_partial_exit"
    IMPROVEMENT_EARLY_EXIT = "improvement_early_exit"
    MANUAL_CLOSE = "manual_close"   # AVT operator override
    MARKET_CLOSE = "market_close"


class ImprovementStatus(str, Enum):
    """Lifecycle of an AVT-applied improvement."""
    APPLIED = "applied"
    EVALUATING = "evaluating"
    PROVEN = "proven"
    REVERTED = "reverted"
    ADOPTED_BY_USER = "adopted_by_user"


class DifferenceCategory(str, Enum):
    STOCK_SELECTION = "stock_selection"
    EXECUTION_QUALITY = "execution_quality"
    LEARNED_IMPROVEMENT = "learned_improvement"
    TRADE_AVOIDANCE = "trade_avoidance"
    TIMING = "timing"
    SIZING = "sizing"


class StrategyLifecycleState(str, Enum):
    """Strategy lifecycle from ingestion to autonomous trading."""
    INGESTED = "ingested"           # YAML loaded, not yet validated
    HEALTH_CHECK = "health_check"   # Validation running
    REJECTED = "rejected"           # Failed health check — user must fix
    PROBATION = "probation"         # No trade history — 25% allocation, 30-trade eval
    CALIBRATING = "calibrating"     # Signal-only mode, comparing to user trades
    AUTONOMOUS = "autonomous"       # Full trading, continuous monitoring
    DEGRADED = "degraded"           # Edge decay detected — 50% allocation
    PAUSED = "paused"               # Critical degradation — no new trades


class StrategyGrade(str, Enum):
    """Health gate grade after validation."""
    READY = "ready"           # All checks passed, edge verified
    PROBATION = "probation"   # Structural ok but no history to verify edge
    REJECTED = "rejected"     # Critical issues found


# Allocation multipliers by lifecycle state
LIFECYCLE_ALLOCATION = {
    StrategyLifecycleState.INGESTED: 0.0,
    StrategyLifecycleState.HEALTH_CHECK: 0.0,
    StrategyLifecycleState.REJECTED: 0.0,
    StrategyLifecycleState.PROBATION: 0.25,
    StrategyLifecycleState.CALIBRATING: 0.0,
    StrategyLifecycleState.AUTONOMOUS: 1.0,
    StrategyLifecycleState.DEGRADED: 0.5,
    StrategyLifecycleState.PAUSED: 0.0,
}


@dataclass
class LifecycleInfo:
    """Per-strategy lifecycle status for API/frontend."""
    strategy_id: str = ""
    state: str = StrategyLifecycleState.INGESTED.value
    grade: str = ""
    health_score: float = 0.0
    allocation_multiplier: float = 0.0

    # Health gate
    health_issues: List[Dict[str, Any]] = field(default_factory=list)

    # Calibration
    calibration_alignment: Optional[float] = None
    calibration_signals: int = 0
    calibration_comparisons: int = 0

    # Edge monitor
    degradation_level: str = "healthy"
    rolling_expectancy: Optional[float] = None
    translation_drift: Optional[float] = None
    baseline_expectancy: Optional[float] = None

    # Timestamps
    state_changed_at: Optional[datetime] = None
    validated_at: Optional[datetime] = None


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

@dataclass
class AVTConfig:
    """Configuration for the Autonomous Virtual Trader."""

    # Capital
    virtual_capital: float = 1_000_000.0

    # Position limits
    max_open_positions: int = 5              # tight: 5 positions max for clean isolation
    max_positions_per_strategy: int = 2      # 2 per strategy
    max_positions_per_sector: int = 2
    max_exposure_per_stock_pct: float = 8.0  # % of capital
    max_exposure_per_strategy_pct: float = 20.0   # was 40% — halved for risk control
    max_exposure_per_sector_pct: float = 20.0

    # Conviction thresholds — no trade below 0.50 conviction
    min_conviction_override: Optional[float] = 0.50

    # Capital sleeves (B.1): partition virtual_capital into INDEPENDENT pools by instrument so strategies run
    # concurrently off separate capital (a future never blocks an option, and each pool slots/draws-down on its
    # own). Empty → ONE implicit wildcard sleeve holding all capital = the historical shared pool (golden-safe).
    # Each entry: {"name": "futures", "capital": 200000, "instruments": ["future","equity"], "max_positions": 3}.
    sleeves: List[Dict[str, Any]] = field(default_factory=list)

    # Scan cycle
    scan_interval_seconds: int = 300  # 5 minutes default
    market_open_time: str = ""   # loaded from settings if empty
    market_close_time: str = ""  # loaded from settings if empty

    def __post_init__(self):
        # Pull market times from central settings if not explicitly overridden
        if not self.market_open_time or not self.market_close_time:
            from config.settings import settings
            if not self.market_open_time:
                self.market_open_time = settings.MARKET_OPEN_TIME
            if not self.market_close_time:
                self.market_close_time = settings.MARKET_CLOSE_TIME

    # Self-improvement
    min_trades_for_analysis: int = 30
    improvement_eval_interval_days: int = 7
    improvement_revert_threshold_r: float = -0.10
    improvement_proof_threshold_r: float = 0.05
    improvement_proof_min_trades: int = 20
    evidence_strength_threshold: float = 0.75

    # Stock quality weights
    stock_quality_weights: Dict[str, float] = field(default_factory=lambda: {
        "rvol": 0.35,
        "spread": 0.15,
        "range_remaining": 0.20,
        "trend_alignment": 0.30,
    })

    # Ranking weights
    ranking_weights: Dict[str, float] = field(default_factory=lambda: {
        "conviction": 0.30,
        "stock_quality": 0.25,
        "variant_performance": 0.20,
        "freshness": 0.10,
        "portfolio_fit": 0.15,
    })


# ══════════════════════════════════════════════════════════════════
#  VIRTUAL PORTFOLIO
# ══════════════════════════════════════════════════════════════════

@dataclass
class VirtualPortfolio:
    """Tracks AVT's virtual capital and positions."""

    starting_capital: float = 1_000_000.0
    cash: float = 1_000_000.0
    open_positions: List[VirtualTrade] = field(default_factory=list)
    closed_trades: List[VirtualTrade] = field(default_factory=list)

    # Tracking
    peak_equity: float = 1_000_000.0
    total_pnl: float = 0.0
    trade_count: int = 0

    # Optional accurate cost model (analytics.futures_charges.FuturesCostModel or any obj with
    # .slippage_bps + .round_trip(buy_notional, sell_notional)). Default None → the crude 0.6%
    # round-trip slippage below (user AVT untouched); the SYSTEM cascade tester opts into the real
    # F&O model (STT-dominated charges + realistic bps slippage). See scripts/research/cascade_forward.py.
    cost_model: Optional[Any] = None

    def _slip_pct(self) -> float:
        """Per-leg slippage fraction — the accurate bps model when a cost_model is set, else the crude 0.3%."""
        if self.cost_model is not None:
            return getattr(self.cost_model, "slippage_bps", 3.0) / 1e4
        return _SLIPPAGE_PER_LEG_PCT

    def _charges(self, entry_px: float, exit_px: float, qty: int) -> float:
        """Real F&O round-trip charges for `qty` (0 when no cost_model → slippage-only, as before)."""
        if self.cost_model is None or qty <= 0:
            return 0.0
        return self.cost_model.round_trip(entry_px * qty, exit_px * qty)

    @property
    def equity(self) -> float:
        unrealized = sum(p.unrealized_pnl for p in self.open_positions)
        return self.cash + unrealized

    @property
    def available_capital(self) -> float:
        allocated = sum(
            p.entry_price * p.remaining_quantity
            for p in self.open_positions
        )
        return self.cash - allocated

    @property
    def current_drawdown_pct(self) -> float:
        eq = self.equity
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - eq) / self.peak_equity)

    @property
    def open_risk_pct(self) -> float:
        """Total risk across all open positions as % of equity."""
        total_risk = sum(p.risk_amount for p in self.open_positions)
        eq = self.equity
        return (total_risk / eq * 100) if eq > 0 else 0.0

    def close_position(self, trade: VirtualTrade, exit_price: float,
                       exit_time: datetime, exit_type: str,
                       portion: float = 1.0) -> None:
        """Close (fully or partially) a position with slippage + (optional) real F&O charge deduction."""
        # Slippage per leg: crude 0.3% by default, or the accurate bps model when cost_model is set.
        slip = self._slip_pct()
        if trade.direction == "long":
            effective_exit = exit_price * (1.0 - slip)   # sell below mid
        else:
            effective_exit = exit_price * (1.0 + slip)   # buy back above mid

        if portion >= 1.0:
            # Full close
            charges = self._charges(trade.entry_price, effective_exit, trade.remaining_quantity)
            trade.exit_price = effective_exit
            trade.exit_time = exit_time
            trade.exit_type = exit_type
            trade.status = VirtualTradeStatus.CLOSED
            trade.realized_pnl = self._compute_pnl(trade, effective_exit, trade.remaining_quantity) - charges
            trade.r_multiple = self._compute_r(trade, effective_exit)
            self.cash += effective_exit * trade.remaining_quantity + trade.realized_pnl
            trade.remaining_quantity = 0
            self.open_positions = [p for p in self.open_positions if p.id != trade.id]
            self.closed_trades.append(trade)
            self.total_pnl += trade.realized_pnl
        else:
            # Partial close
            close_qty = int(trade.remaining_quantity * portion)
            if close_qty <= 0:
                return
            charges = self._charges(trade.entry_price, effective_exit, close_qty)
            partial_pnl = self._compute_pnl(trade, effective_exit, close_qty) - charges
            trade.partial_exits.append({
                "time": exit_time.isoformat(),
                "price": effective_exit,
                "quantity": close_qty,
                "pnl": partial_pnl,
                "reason": exit_type,
            })
            trade.remaining_quantity -= close_qty
            trade.status = VirtualTradeStatus.PARTIAL
            self.cash += effective_exit * close_qty - charges   # charges==0 when no cost_model (default unchanged)
            self.total_pnl += partial_pnl

        # Update peak equity
        eq = self.equity
        if eq > self.peak_equity:
            self.peak_equity = eq

    def add_position(self, trade: VirtualTrade) -> None:
        """Add a position, applying entry slippage to the stored entry price."""
        # Slippage is already in the market price at execution, but we make
        # entry adverse to further reflect hidden spread costs on illiquid fills.
        slip = self._slip_pct()
        if trade.direction == "long":
            trade.entry_price = trade.entry_price * (1.0 + slip)   # pay more on entry
        else:
            trade.entry_price = trade.entry_price * (1.0 - slip)   # short at lower bid
        self.open_positions.append(trade)
        self.trade_count += 1

    # ── OPTION accounting (P&L-only; separate from the futures share model above) ──
    # Options book on the premium and their NET already includes option slippage + real option charges (computed by
    # OpenOptionPosition.settle). So they must NOT go through add_position/close_position (which apply the FUTURES
    # slippage + FuturesCostModel charges). These two methods keep option cash/pnl correct and leave the futures
    # path byte-identical (the golden gate never touches them).
    def add_option_position(self, trade: VirtualTrade) -> None:
        """Open an OPTION position: no futures slippage (already in the booked premium fill); mark-to-market via
        `unrealized_pnl` (set by the engine's option manage), cash moves only by the NET on close."""
        self.open_positions.append(trade)
        self.trade_count += 1

    def close_option_position(self, trade: VirtualTrade, exit_prem: float, net: float,
                              exit_time: datetime, exit_type: str) -> None:
        """Close an OPTION at `exit_prem` with the option's OWN `net` (option slippage + option charges already
        included via OpenOptionPosition.settle)."""
        trade.exit_price = exit_prem
        trade.exit_time = exit_time
        trade.exit_type = exit_type
        trade.exit_reason = exit_type
        trade.status = VirtualTradeStatus.CLOSED
        trade.realized_pnl = net
        trade.unrealized_pnl = 0.0
        trade.remaining_quantity = 0
        self.cash += net
        self.total_pnl += net
        self.open_positions = [p for p in self.open_positions if p.id != trade.id]
        self.closed_trades.append(trade)
        eq = self.equity
        if eq > self.peak_equity:
            self.peak_equity = eq

    @staticmethod
    def _compute_pnl(trade: VirtualTrade, exit_price: float, qty: int) -> float:
        if trade.direction == "long":
            return (exit_price - trade.entry_price) * qty
        else:
            return (trade.entry_price - exit_price) * qty

    @staticmethod
    def _compute_r(trade: VirtualTrade, exit_price: float) -> float:
        if trade.risk_amount <= 0:
            return 0.0
        pnl = (exit_price - trade.entry_price) if trade.direction == "long" \
            else (trade.entry_price - exit_price)
        risk_per_share = abs(trade.entry_price - trade.stop_price)
        if risk_per_share <= 0:
            return 0.0
        return pnl / risk_per_share


# ══════════════════════════════════════════════════════════════════
#  VIRTUAL TRADE
# ══════════════════════════════════════════════════════════════════

@dataclass
class VirtualTrade:
    """A single virtual trade executed by the AVT."""

    id: UUID = field(default_factory=uuid4)
    trader_id: Optional[str] = None   # owner (multi-user attribution)
    strategy_id: str = ""
    strategy_version: str = "1.0"
    variant_id: Optional[str] = None
    symbol: str = ""
    direction: str = "long"    # "long" or "short"

    # Entry
    entry_time: Optional[datetime] = None
    entry_price: float = 0.0
    quantity: int = 0
    remaining_quantity: int = 0
    stop_price: float = 0.0
    entry_atr: Optional[float] = None   # entry-bar ATR (if the signal supplied one) → fixed base for the ATR trail
    target_prices: List[float] = field(default_factory=list)

    # Sizing
    risk_amount: float = 0.0   # $ at risk
    risk_pct: float = 0.0      # % of equity at risk

    # Signal metadata
    conviction_at_entry: float = 0.0
    feature_vector_at_entry: Dict[str, float] = field(default_factory=dict)
    regime_at_entry: Optional[Dict[str, float]] = None

    # Ranking metadata (WHY this stock was selected)
    ranking_position: int = 0
    ranking_score: float = 0.0
    ranking_explanation: str = ""
    competing_setups_count: int = 0

    # Active improvements at entry
    active_improvements: List[str] = field(default_factory=list)

    # Exit
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_type: Optional[str] = None
    exit_improvement_id: Optional[str] = None   # if exit was triggered by improvement

    # P&L
    realized_pnl: float = 0.0
    r_multiple: float = 0.0
    unrealized_pnl: float = 0.0
    partial_exits: List[Dict[str, Any]] = field(default_factory=list)

    # Excursion tracking
    max_favorable_r: float = 0.0
    max_favorable_price: float = 0.0
    max_favorable_time: Optional[datetime] = None
    max_adverse_r: float = 0.0
    features_at_peak: Optional[Dict[str, float]] = None

    # Instrument / option execution (M3). Defaults keep every underlying (future/equity) trade unchanged; an OPTION
    # position books on the option premium (entry_price = premium, quantity = lots × lot_size, symbol = underlying,
    # tradeable = option tradingsymbol) and its exit is driven by underlying_stop + the premium exit-at-spike trail.
    instrument: str = "future"                    # "future" | "equity" | "option" | "crypto"
    tradeable: Optional[str] = None               # what actually trades (option tradingsymbol); None → underlying
    option_strike: Optional[float] = None
    option_expiry: Optional[str] = None
    option_right: Optional[str] = None            # "CE" | "PE"
    option_lots: Optional[int] = None
    option_lot_size: Optional[int] = None
    underlying_stop: Optional[float] = None       # thesis-invalidation on the UNDERLYING (drives the option exit)

    # Status
    status: VirtualTradeStatus = VirtualTradeStatus.OPEN

    @property
    def is_open(self) -> bool:
        return self.status in (VirtualTradeStatus.OPEN, VirtualTradeStatus.PARTIAL)

    @property
    def is_winner(self) -> bool:
        return self.realized_pnl > 0

    @property
    def duration_minutes(self) -> Optional[float]:
        if self.entry_time and self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 60
        return None


# ══════════════════════════════════════════════════════════════════
#  SETUP CANDIDATES & RANKING
# ══════════════════════════════════════════════════════════════════

@dataclass
class SetupCandidate:
    """A detected setup before ranking."""
    symbol: str = ""
    strategy_id: str = ""
    signal: Any = None         # SignalOutput from ske
    timestamp: Optional[datetime] = None
    sector: str = "unknown"


@dataclass
class RankedCandidate:
    """A setup candidate after multi-dimensional ranking."""
    candidate: SetupCandidate = field(default_factory=SetupCandidate)
    composite_score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    ranking_explanation: str = ""
    rank: int = 0


# ══════════════════════════════════════════════════════════════════
#  EXIT DECISIONS
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExitDecision:
    """Result of evaluating exit conditions for an open position."""
    should_exit: bool = False
    exit_type: str = ""
    exit_portion: float = 1.0       # 1.0 = full, 0.5 = half
    reason: str = ""
    improvement_id: Optional[str] = None
    exit_price: Optional[float] = None  # explicit fill price (e.g. OHLC-aware stop/target in replay)


@dataclass
class StoryIntactResult:
    """Result of evaluating whether the trade thesis still holds."""
    story_broken: bool = False
    reason: str = ""
    confidence: float = 0.0


# ══════════════════════════════════════════════════════════════════
#  STRATEGY VERSIONING
# ══════════════════════════════════════════════════════════════════

@dataclass
class ActiveImprovement:
    """An improvement actively applied to AVT's strategy version."""

    id: UUID = field(default_factory=uuid4)
    hypothesis_id: Optional[str] = None
    strategy_id: str = ""

    # What
    type: str = ""             # "partial_exit_at_rr", "early_exit_pattern", "entry_filter", "regime_filter"
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    # When
    applied_at: Optional[datetime] = None
    applied_at_version: str = ""
    reverted_at: Optional[datetime] = None
    revert_reason: str = ""

    # Validation that triggered the apply
    validation_result: Optional[Dict[str, Any]] = None

    # Performance tracking
    trades_before_count: int = 0
    performance_before: Optional[PerformanceSnapshot] = None
    trades_after_count: int = 0
    performance_after: Optional[PerformanceSnapshot] = None

    # Status
    status: ImprovementStatus = ImprovementStatus.APPLIED
    evidence_strength: float = 0.0


@dataclass
class StrategyVersion:
    """
    A versioned copy of an SKO with applied improvements.

    AVT trades the VERSIONED strategy, not the raw SKO.
    Each improvement bumps the version.
    """

    base_sko: Any = None                                    # StrategyKnowledgeObject dict
    active_improvements: List[ActiveImprovement] = field(default_factory=list)
    version: str = "1.0"
    version_start_date: Optional[datetime] = None
    version_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def strategy_id(self) -> str:
        if self.base_sko:
            return str(self.base_sko.get("id", "")) if isinstance(self.base_sko, dict) \
                else str(getattr(self.base_sko, "id", ""))
        return ""

    @property
    def min_conviction(self) -> float:
        """Minimum conviction from SKO's conviction_map, default 0.3."""
        if self.base_sko:
            cmap = self.base_sko.get("conviction_map", {}) if isinstance(self.base_sko, dict) \
                else getattr(self.base_sko, "conviction_map", {})
            skip_bounds = cmap.get("SKIP", [0.0, 0.3])
            if isinstance(skip_bounds, list) and len(skip_bounds) == 2:
                return skip_bounds[1]
        return 0.3

    def get_active_sko(self) -> Dict[str, Any]:
        """Return the SKO dict (base, not modified — improvements are applied at runtime)."""
        if isinstance(self.base_sko, dict):
            return self.base_sko
        # Convert to dict if it's a model
        return {
            "id": getattr(self.base_sko, "id", None),
            "name": getattr(self.base_sko, "name", ""),
            "scoring_config": getattr(self.base_sko, "scoring_config", {}),
            "conviction_map": getattr(self.base_sko, "conviction_map", {}),
            "entry_zones": getattr(self.base_sko, "entry_zones", []),
            "exit_rules": getattr(self.base_sko, "exit_rules", {}),
            "risk_parameters": getattr(self.base_sko, "risk_parameters", {}),
        }

    def bump_version(self, reason: str = "") -> str:
        """Increment version and log history."""
        parts = self.version.split(".")
        minor = int(parts[1]) + 1 if len(parts) > 1 else 1
        new_ver = f"{parts[0]}.{minor}"
        self.version_history.append({
            "from": self.version,
            "to": new_ver,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.version = new_ver
        self.version_start_date = datetime.now(timezone.utc)
        return new_ver


# ══════════════════════════════════════════════════════════════════
#  PERFORMANCE SNAPSHOTS
# ══════════════════════════════════════════════════════════════════

@dataclass
class PerformanceSnapshot:
    """Point-in-time performance summary."""
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    expectancy_r: float = 0.0
    total_r: float = 0.0
    avg_winner_r: float = 0.0
    avg_loser_r: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_r: float = 0.0
    sharpe_estimate: float = 0.0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


def compute_performance(trades: List[VirtualTrade]) -> PerformanceSnapshot:
    """Compute performance snapshot from a list of closed virtual trades."""
    if not trades:
        return PerformanceSnapshot()

    closed = [t for t in trades if t.status == VirtualTradeStatus.CLOSED]
    if not closed:
        return PerformanceSnapshot()

    winners = [t for t in closed if t.r_multiple > 0]
    losers = [t for t in closed if t.r_multiple <= 0]
    rs = [t.r_multiple for t in closed]

    win_r = sum(t.r_multiple for t in winners) if winners else 0.0
    loss_r = abs(sum(t.r_multiple for t in losers)) if losers else 0.0

    import numpy as np
    rs_arr = np.array(rs) if rs else np.array([0.0])
    std = float(rs_arr.std()) if len(rs_arr) > 1 else 1.0
    mean = float(rs_arr.mean())

    times = [t.entry_time for t in closed if t.entry_time]

    return PerformanceSnapshot(
        trade_count=len(closed),
        win_count=len(winners),
        loss_count=len(losers),
        win_rate=len(winners) / len(closed) if closed else 0.0,
        expectancy_r=mean,
        total_r=sum(rs),
        avg_winner_r=win_r / len(winners) if winners else 0.0,
        avg_loser_r=loss_r / len(losers) if losers else 0.0,
        profit_factor=win_r / loss_r if loss_r > 0 else float("inf") if win_r > 0 else 0.0,
        max_drawdown_r=_compute_max_dd_r(rs),
        sharpe_estimate=mean / std if std > 0.01 else 0.0,
        period_start=min(times) if times else None,
        period_end=max(times) if times else None,
    )


def _compute_max_dd_r(rs: List[float]) -> float:
    """Maximum drawdown in R-multiple terms."""
    if not rs:
        return 0.0
    peak = 0.0
    cumulative = 0.0
    max_dd = 0.0
    for r in rs:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ══════════════════════════════════════════════════════════════════
#  COMPETITION REPORT
# ══════════════════════════════════════════════════════════════════

@dataclass
class UserStats:
    trade_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0


@dataclass
class AVTStats:
    trade_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0
    version_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TradeExample:
    """Concrete trade example for evidence presentation."""
    symbol: str = ""
    trade_date: Optional[datetime] = None
    with_improvement_r: float = 0.0
    without_improvement_r: float = 0.0
    difference: float = 0.0
    explanation: str = ""


@dataclass
class PerformanceDifference:
    """One category where AVT outperformed the user."""
    category: str = ""           # DifferenceCategory value
    impact_r: float = 0.0       # per-trade R impact
    description: str = ""
    example_trades: List[Any] = field(default_factory=list)
    actionable_insight: str = ""
    improvement_id: Optional[str] = None


@dataclass
class CompetitionReport:
    """Full AVT vs User comparison."""
    strategy_name: str = ""
    strategy_id: str = ""
    period_days: int = 30
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    user_stats: UserStats = field(default_factory=UserStats)
    avt_stats: AVTStats = field(default_factory=AVTStats)

    performance_gap: float = 0.0        # avt_expectancy - user_expectancy

    differences: List[PerformanceDifference] = field(default_factory=list)
    proven_improvements: List[ActiveImprovement] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
#  IMPROVEMENT PRESENTATION
# ══════════════════════════════════════════════════════════════════

@dataclass
class ImprovementPresentation:
    """Human-readable presentation of a proven improvement."""
    title: str = ""
    narrative: str = ""
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    examples: List[TradeExample] = field(default_factory=list)
    user_action_options: List[str] = field(default_factory=lambda: [
        "adopt_as_rule",
        "try_as_suggestion",
        "keep_virtual_only",
        "dismiss",
    ])
    avt_confidence: float = 0.0


# ══════════════════════════════════════════════════════════════════
#  SCAN CYCLE LOG
# ══════════════════════════════════════════════════════════════════

@dataclass
class ScanCycleLog:
    """Record of one scan cycle for debugging/audit."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stocks_scanned: int = 0
    candidates_found: int = 0
    ranked_count: int = 0
    filtered_count: int = 0
    executed_count: int = 0
    positions_managed: int = 0
    exits_triggered: int = 0
    duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
#  SKIPPED SETUP LOG (for "trades avoided" comparison)
# ══════════════════════════════════════════════════════════════════

@dataclass
class SkippedSetup:
    """Record of a setup AVT saw but chose not to trade."""
    symbol: str = ""
    strategy_id: str = ""
    timestamp: Optional[datetime] = None
    conviction: float = 0.0
    skip_reason: str = ""           # "low_conviction", "poor_stock_quality", etc.
    ranking_score: float = 0.0
    would_have_been_rank: int = 0
