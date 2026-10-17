# avt/engine.py
"""
Autonomous Virtual Trader — Core Engine
=========================================

The AVT runs the trader's strategies in a virtual portfolio using live
market data. It scans 200+ stocks, ranks setups, and executes virtually
with perfect discipline.

THREE advantages over the human:
  1. COVERAGE — scans full FNO universe simultaneously
  2. DISCIPLINE — no FOMO, no revenge, no hesitation, no panic
  3. INTELLIGENCE — ranks competing setups, picks the best ones

The AVT's trade record = what the strategy CAN DO when executed perfectly.
The gap between AVT and user = total improvement opportunity.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from avt.models import (
    AVTConfig,
    AVTStatus,
    ActiveImprovement,
    ExitDecision,
    ExitType,
    ScanCycleLog,
    SetupCandidate,
    SkippedSetup,
    StrategyVersion,
    StoryIntactResult,
    VirtualPortfolio,
    VirtualTrade,
    VirtualTradeStatus,
)

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────

def compute_r_multiple(entry: float, current: float, stop: float,
                       direction: str = "long") -> float:
    """Compute current R-multiple."""
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    if direction == "long":
        return (current - entry) / risk
    return (entry - current) / risk


# ══════════════════════════════════════════════════════════════════
#  MARKET DATA INTERFACE
# ══════════════════════════════════════════════════════════════════

class MarketDataInterface:
    """
    Abstract interface for real-time market data.
    Concrete implementations connect to broker feeds, CSV replay, etc.
    """

    async def get_current_price(self, symbol: str,
                                side: str = "last") -> float:
        """Get current price. side = 'buy'|'sell'|'last'."""
        raise NotImplementedError

    async def get_candles(self, symbol: str, timeframe: str = "5m",
                          count: int = 100) -> List[Dict[str, Any]]:
        """Get recent OHLCV candles for a symbol."""
        raise NotImplementedError

    async def get_spread_pct(self, symbol: str) -> float:
        """Get current bid-ask spread as % of price."""
        return 0.05  # default 0.05%

    async def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return "unknown"

    async def is_market_open(self) -> bool:
        """Check if market is currently open."""
        return True


class ReplayMarketData(MarketDataInterface):
    """
    Market data from historical candles — for testing and historical replay.

    A replay driver (avt/replay.py) loads lake candles per symbol, then steps a
    shared CLOCK forward through a unified bar timeline calling `seek(ts)` each
    step. `get_candles` only ever returns bars at/▸before the cursor (no
    lookahead); `get_current_bar` exposes the cursor bar's OHLC so the engine can
    do OHLC-aware intrabar stop/target fills.
    """

    def __init__(self):
        self._candle_store: Dict[str, List[Dict[str, Any]]] = {}
        self._timestamps: Dict[str, List[Any]] = {}   # parsed bar ts, sorted
        self._current_index: Dict[str, int] = {}
        self._sectors: Dict[str, str] = {}

    @staticmethod
    def _parse_ts(c: Dict[str, Any]):
        import pandas as pd
        return pd.Timestamp(c.get("ts") or c.get("timestamp"))

    def load_candles(self, symbol: str, candles: List[Dict[str, Any]],
                     sector: str = "unknown") -> None:
        self._candle_store[symbol] = candles
        self._timestamps[symbol] = [self._parse_ts(c) for c in candles]
        self._current_index[symbol] = len(candles) - 1
        self._sectors[symbol] = sector

    def advance(self, symbol: str, steps: int = 1) -> None:
        if symbol in self._current_index:
            self._current_index[symbol] = min(
                self._current_index[symbol] + steps,
                len(self._candle_store.get(symbol, [])) - 1,
            )

    def seek(self, ts) -> None:
        """Point-in-time: set every symbol's cursor to the latest bar with bar_ts <= ts."""
        import bisect
        import pandas as pd
        ts = pd.Timestamp(ts)
        for sym, stamps in self._timestamps.items():
            if not stamps:
                continue
            self._current_index[sym] = max(0, bisect.bisect_right(stamps, ts) - 1)

    def get_current_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """The cursor bar's OHLC dict (for OHLC-aware intrabar fills), or None."""
        candles = self._candle_store.get(symbol, [])
        idx = self._current_index.get(symbol, len(candles) - 1)
        if not candles or idx < 0:
            return None
        return candles[idx]

    async def get_current_price(self, symbol: str,
                                side: str = "last") -> float:
        bar = self.get_current_bar(symbol)
        return float(bar.get("close", 0.0)) if bar else 0.0

    async def get_candles(self, symbol: str, timeframe: str = "5m",
                          count: int = 100) -> List[Dict[str, Any]]:
        candles = self._candle_store.get(symbol, [])
        idx = self._current_index.get(symbol, len(candles) - 1)
        start = max(0, idx - count + 1)
        return candles[start:idx + 1]

    async def get_sector(self, symbol: str) -> str:
        return self._sectors.get(symbol, "unknown")


# ══════════════════════════════════════════════════════════════════
#  VIRTUAL POSITION MANAGER
# ══════════════════════════════════════════════════════════════════

class VirtualPositionManager:
    """Computes position sizes for virtual trades."""

    def __init__(self, portfolio: VirtualPortfolio):
        self.portfolio = portfolio

    def compute_size(
        self,
        conviction: float,
        entry_price: float,
        stop_price: float,
        config: AVTConfig,
        equity: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute position size based on conviction and risk parameters.

        `equity`/`drawdown_pct` override the portfolio-wide values when the caller sizes off a specific capital
        SLEEVE (B.1). Both default to the shared portfolio values → byte-identical to the historical path.

        Returns dict with 'shares', 'risk_amount', 'risk_pct'.
        """
        equity = self.portfolio.equity if equity is None else equity
        if equity <= 0 or entry_price <= 0:
            return {"shares": 0, "risk_amount": 0.0, "risk_pct": 0.0}

        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return {"shares": 0, "risk_amount": 0.0, "risk_pct": 0.0}

        # Base risk: 1% of equity, scaled by conviction
        # conviction 0.3→0.5x, 0.5→1.0x, 0.7→1.5x, 0.85+→2.0x
        if conviction >= 0.85:
            conv_mult = 2.0
        elif conviction >= 0.7:
            conv_mult = 1.5
        elif conviction >= 0.5:
            conv_mult = 1.0
        elif conviction >= 0.3:
            conv_mult = 0.5
        else:
            conv_mult = 0.25

        base_risk_pct = 1.0  # 1% per trade
        risk_pct = base_risk_pct * conv_mult

        # Drawdown scaling
        dd = self.portfolio.current_drawdown_pct if drawdown_pct is None else drawdown_pct
        if dd >= 0.20:
            risk_pct *= 0.0  # halt trading
        elif dd >= 0.15:
            risk_pct *= 0.25
        elif dd >= 0.10:
            risk_pct *= 0.50
        elif dd >= 0.05:
            risk_pct *= 0.75

        # Cap at max risk
        risk_pct = min(risk_pct, 2.0)

        risk_amount = equity * (risk_pct / 100.0)
        shares = int(risk_amount / stop_distance)

        # Check stock exposure limit
        position_value = shares * entry_price
        max_stock_value = equity * (config.max_exposure_per_stock_pct / 100.0)
        if position_value > max_stock_value:
            shares = int(max_stock_value / entry_price)
            risk_amount = shares * stop_distance

        return {
            "shares": max(0, shares),
            "risk_amount": risk_amount,
            "risk_pct": risk_pct,
        }


# ══════════════════════════════════════════════════════════════════
#  AUTONOMOUS VIRTUAL TRADER
# ══════════════════════════════════════════════════════════════════

class AutonomousVirtualTrader:
    """
    The AVT runs the trader's strategies better than the trader.

    It scans the full watchlist, ranks setups by composite desirability,
    applies portfolio filters, and executes virtually with perfect discipline.

    Improvements discovered by AVTSelfImprover are applied as additional
    exit/entry conditions on the versioned strategy.
    """

    def __init__(
        self,
        strategies: List[Dict[str, Any]],
        config: Optional[AVTConfig] = None,
        signal_generator: Any = None,
        feature_computer: Any = None,
        regime_engine: Any = None,
        market_data: Optional[MarketDataInterface] = None,
        stock_ranker: Any = None,       # injected StockSelectionRanker
        watchlist: Optional[List[str]] = None,
        trader_id: Optional[str] = None,   # owner — this engine trades only their strategies
    ):
        # Owner of this engine (multi-user): every virtual trade it opens is
        # attributed here, and it only ever loads this user's strategies.
        self.trader_id = str(trader_id) if trader_id is not None else None
        self.config = config or AVTConfig()
        self.portfolio = VirtualPortfolio(
            starting_capital=self.config.virtual_capital,
            cash=self.config.virtual_capital,
            peak_equity=self.config.virtual_capital,
        )

        # SKE infrastructure
        self.signal_gen = signal_generator
        self.features = feature_computer
        self.regime = regime_engine
        self.market = market_data or ReplayMarketData()

        # Stock ranker (injected or lazy-created)
        self._stock_ranker = stock_ranker

        # Strategies and versions
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.strategy_versions: Dict[str, StrategyVersion] = {}
        for s in strategies:
            sid = str(s.get("id", s.get("short_code", "")))
            self.strategies[sid] = s
            self.strategy_versions[sid] = StrategyVersion(
                base_sko=s,
                version="1.0",
                version_start_date=datetime.now(timezone.utc),
            )

        # Watchlist
        self.watchlist = watchlist or self._build_watchlist(strategies)

        # Position manager
        self.position_manager = VirtualPositionManager(self.portfolio)

        # Capital Operator (the spine seam: Intent → operator decides take/instrument/size → ExecutionEngine).
        # decide_open is advisory-reflex (always take at the preferred instrument) so the FUTURES golden is
        # unchanged; the money/account here are used ONLY by size_option (the options path). Options default to a
        # SAFE fixed-1-lot policy (matches the honest money test; no ballooning) — live/tester may override.
        from avt.capital.operator import AccountState, CapitalOperator
        from avt.execution.options import MoneyPolicy
        self._option_money = MoneyPolicy(mode="fixed", fixed_lots=1)
        self._capital_operator = CapitalOperator(
            money=self._option_money,
            account=AccountState(starting_capital=self.config.virtual_capital),
        )
        # Capital SLEEVES (B.1): independent per-instrument capital pools so strategies run concurrently off
        # separate capital. Default (no configured sleeves) = ONE implicit wildcard sleeve → is_partitioned False →
        # the engine keeps its byte-identical shared-pool path (futures golden unmoved). Only when the user
        # configures sleeves does the money math below route per-sleeve.
        from avt.capital.allocator import CapitalAllocator
        self._allocator = CapitalAllocator(
            self.portfolio, getattr(self.config, "sleeves", None), self.config.max_open_positions)
        # Options-in-engine (M3b): injected by the tester (ReplayOptionMarket) or integrations (LiveOptionMarket).
        # None → the engine cannot book options (safe default). Live option handles kept for the manage loop (A3).
        self._option_market: Any = None
        self._option_slip = 0.5
        self._option_ptrail = 0.30
        self._option_positions: Dict[Any, Any] = {}   # VirtualTrade.id → (OpenOptionPosition, contract)

        # Two-tier attention (B.2 — 100-stock scale): before the expensive per-symbol cascade priming
        # (prime_live narrates 5m/15m/1h), run cheap peripheral recognition on every symbol and pay for the full
        # signal only on the few that light up (a 5m birth). OFF by default → replay/golden byte-identical; the
        # LIVE engine turns it on (integrations). Fail-open + cascade-only, so it can never drop a real signal.
        self._two_tier_attention: bool = False
        self._lit_cache: Optional[set] = None   # per-cycle memo of lit symbols

        # Logging
        self.scan_logs: List[ScanCycleLog] = []
        self.skipped_setups: List[SkippedSetup] = []

        # State
        self.status = AVTStatus.IDLE
        self._cycle_count = 0

        # Callbacks
        self._on_trade_opened: Optional[Callable] = None
        self._on_trade_closed: Optional[Callable] = None

        # Lifecycle manager (set by integrations.py)
        self._lifecycle_manager: Any = None

        # Telemetry
        self._last_scan_at: Optional[datetime] = None

        # Clock (replay): when set, the engine reads "now" from this callable
        # (the current bar's timestamp) instead of wall-clock, so historical
        # replay stamps trades + evaluates time-exits at the bar's real time.
        # Live mode leaves it None → datetime.now(timezone.utc).
        self._clock: Optional[Callable[[], datetime]] = None
        # Whether the intraday market-close auto-exit applies (disabled for
        # daily/swing replay so multi-day holds aren't force-closed same day).
        self.force_intraday_close: bool = True

        # Per-strategy watchlists (strategy_id → symbols), mirroring the
        # co-planner's per-strategy stock targeting. Each strategy scans only
        # its own matched stocks; strategies without a list fall back to the
        # global watchlist. Refreshed each cycle via the optional provider so
        # newly distributed/matched stocks are picked up automatically.
        self.strategy_watchlists: Dict[str, List[str]] = {}
        self._strategy_watchlist_provider: Optional[Callable] = None

    # ── Watchlist ──────────────────────────────────────────────

    @staticmethod
    def _build_watchlist(strategies: List[Dict[str, Any]]) -> List[str]:
        """
        Build watchlist from strategy metadata.
        In production, this comes from the user's watchlist + FNO universe.
        """
        symbols = set()
        for s in strategies:
            # Check if strategy has a watchlist
            wl = s.get("watchlist", [])
            symbols.update(wl)
            # Check risk_parameters for universe
            rp = s.get("risk_parameters", {})
            universe = rp.get("universe", [])
            symbols.update(universe)
        return list(symbols)

    def set_watchlist(self, symbols: List[str]) -> None:
        """Set the global fallback watchlist dynamically."""
        self.watchlist = symbols

    # ── Clock (replay vs live) ─────────────────────────────────

    def _now(self) -> datetime:
        """Current time — bar timestamp in replay, wall-clock otherwise."""
        if self._clock is not None:
            try:
                return self._clock()
            except Exception:
                pass
        return datetime.now(timezone.utc)

    def set_clock(self, clock: Optional[Callable[[], datetime]],
                  force_intraday_close: bool = True) -> None:
        """Inject a replay clock. clock() returns the current bar's timestamp.
        force_intraday_close=False for daily/swing replay (no same-day exit)."""
        self._clock = clock
        self.force_intraday_close = force_intraday_close

    def _sleep_to_next_bar(self) -> float:
        """Seconds to sleep so the NEXT scan fires JUST AFTER the next candle-close boundary, instead of a
        free-running `interval` after the scan finishes. This aligns confirmations (and therefore entries) to
        the bar close — minimal post-close latency, so live fills sit near the intended close price. Epoch is
        used directly: the 5-min NSE bar closes (…09:20, 09:25…) are exact multiples of 300s in epoch time, so
        `epoch // interval` boundaries land on real bar closes regardless of the IST offset. LIVE only; a replay
        clock keeps the plain deterministic interval."""
        interval = max(1, int(self.config.scan_interval_seconds))
        if self._clock is not None:
            return float(interval)                        # replay: deterministic cadence, no wall-clock alignment
        import time as _t
        now = _t.time()
        epsilon = float(getattr(self.config, "scan_align_epsilon_s", 3.0))  # let the just-closed bar land in the feed
        nxt = (int(now) // interval + 1) * interval
        return max(1.0, (nxt - now) + epsilon)

    def _current_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """The market's current OHLC bar (replay only); None in live mode."""
        getter = getattr(self.market, "get_current_bar", None)
        if getter is None:
            return None
        try:
            return getter(symbol)
        except Exception:
            return None

    def set_strategy_watchlists(self, mapping: Dict[str, List[str]]) -> None:
        """Set per-strategy watchlists (strategy_id → symbols)."""
        self.strategy_watchlists = mapping or {}

    def set_strategy_watchlist_provider(self, provider: Optional[Callable]) -> None:
        """
        Set a callable returning the current per-strategy watchlist map.
        Called (in an executor) at the start of each scan cycle so AVT stays
        in sync with the master → per-strategy distribution.
        """
        self._strategy_watchlist_provider = provider

    # ── Main Scan Cycle ────────────────────────────────────────

    async def run_scan_cycle(self) -> ScanCycleLog:
        """
        Called every candle close (e.g., every 5 minutes).
        This is the MAIN LOOP.

        STEP 1: SCAN — generate signals for all stocks × strategies
        STEP 2: RANK — multi-dimensional ranking
        STEP 3: FILTER — apply portfolio limits
        STEP 4: EXECUTE — open virtual positions
        STEP 5: MANAGE — evaluate exits on open positions
        """
        t0 = time.monotonic()
        log = ScanCycleLog(stocks_scanned=len(self.watchlist))
        self.status = AVTStatus.SCANNING

        try:
            # Refresh per-strategy watchlists from the latest distribution
            if self._strategy_watchlist_provider is not None:
                try:
                    loop = asyncio.get_event_loop()
                    mapping = await loop.run_in_executor(
                        None, self._strategy_watchlist_provider
                    )
                    if mapping:
                        self.strategy_watchlists = mapping
                except Exception as e:
                    logger.debug("Per-strategy watchlist refresh failed: %s", e)

            # Get current regime
            current_regime = None
            if self.regime:
                try:
                    current_regime = await self._compute_regime()
                except Exception as e:
                    logger.warning("Regime computation failed: %s", e)

            regime_dict = None
            if current_regime:
                regime_dict = (
                    current_regime.dict()
                    if hasattr(current_regime, "dict")
                    else {"raw": str(current_regime)}
                )

            # ── STEP 1: SCAN (each strategy against its OWN watchlist) ──
            candidates = []
            scanned_symbols: set = set()
            from avt.templates import is_cascade_sko as _is_cascade
            # LAKE REFRESH (live): ONE throttled pass fetches today's 1m for the whole scan universe up front, so
            # everything below reads the warm lake/cache instead of each stage hammering Angel's rate-limited API
            # (the AB1021 death-spiral). Exact-same bars → exact-same signals. Union = global watchlist ∪ each
            # cascade SKO's own watchlist (e.g. the option indices) ∪ held symbols.
            _universe: set = set(self.watchlist or [])
            for _sid in self.strategy_versions:
                if _is_cascade(self.strategies.get(_sid, {})):
                    _universe.update(self.strategy_watchlists.get(_sid)
                                     or self.strategies.get(_sid, {}).get("watchlist") or self.watchlist)
            _universe.update(p.symbol for p in self.portfolio.open_positions)
            await self._refresh_lake_today(sorted(_universe))
            # Two-tier attention (B.2): compute the cheap peripheral "lit" set ONCE per cycle. Only cascade symbols
            # that light up (a 5m birth) get the expensive full signal; the rest are skipped. OFF → lit is None →
            # every symbol runs the full signal (byte-identical to before).
            # OFFLOAD to a thread — recognize over the whole watchlist (~20s at 50 stocks) is heavy sync pandas/IO;
            # running it on the event loop FREEZES every API request during the scan. to_thread keeps the app live.
            self._lit_cache = (await asyncio.to_thread(self._peripheral_lit)
                               if self._two_tier_attention else None)
            # PERF: pre-warm cascade narration for every symbol we're about to evaluate — CONCURRENTLY (threads),
            # so the serial per-strategy loop below hits warm O(1) lookups instead of narrating one symbol at a
            # time. Live candle fetches are I/O-bound (scale well on threads); the priming is per-symbol keyed +
            # throttle-guarded, so the serial _ensure_primed_live calls then skip. No detection change — only WHEN
            # the (identical) narration runs. Live dispatcher only; replay primes explicitly over its range.
            await self._prewarm_cascade_narration(_is_cascade)
            for strategy_id, version in self.strategy_versions.items():
                # SYSTEM cascade templates are pre-validated (frozen SPEC) — they bypass the taught-strategy
                # lifecycle governance (calibration/probation/allocation-scaling); the Capital Operator governs them.
                is_system = _is_cascade(self.strategies.get(strategy_id, {}))
                # Lifecycle gate (once per strategy)
                is_calibrating = False
                if self._lifecycle_manager and not is_system:
                    can, _reason = self._lifecycle_manager.can_trade(strategy_id)
                    lc_info = self._lifecycle_manager.get_lifecycle_info(strategy_id)
                    is_calibrating = bool(lc_info and lc_info.state == "calibrating")
                    if not can and not is_calibrating:
                        continue

                # Per-strategy watchlist: DB strategy-watchlist → the SKO's OWN pinned watchlist (e.g. the
                # index-options template's fixed NIFTY/BANKNIFTY/SENSEX universe) → the global fallback list.
                symbols = (self.strategy_watchlists.get(strategy_id)
                           or self.strategies.get(strategy_id, {}).get("watchlist")
                           or self.watchlist)

                is_cascade_strategy = is_system and _is_cascade(self.strategies.get(strategy_id, {}))
                for symbol in symbols:
                    scanned_symbols.add(symbol)
                    # Two-tier gate: skip the expensive cascade signal for symbols with NO peripheral activity.
                    # ONLY cascade strategies are gated (they require a 5m birth, which recognition sees); taught
                    # strategies always run (their signals don't need a birth). Fail-open lives in _peripheral_lit.
                    if (self._lit_cache is not None and is_cascade_strategy
                            and symbol not in self._lit_cache):
                        continue
                    try:
                        signal = await self._generate_signal(
                            symbol, version, current_regime
                        )
                        if signal is None:
                            continue

                        conv = getattr(signal, "conviction_score", 0.0)
                        min_conv = self.config.min_conviction_override or version.min_conviction

                        # If CALIBRATING, record signal but don't execute
                        if is_calibrating and conv >= min_conv:
                            cal = (
                                self._lifecycle_manager.get_calibration_tracker(strategy_id)
                                if self._lifecycle_manager else None
                            )
                            if cal:
                                cal.record_avt_signal(
                                    symbol=symbol,
                                    direction=getattr(signal, "direction", "long"),
                                    conviction=conv,
                                    details={"strategy_id": strategy_id},
                                )
                            continue  # don't add to execution candidates

                        if conv >= min_conv:
                            candidates.append(SetupCandidate(
                                symbol=symbol,
                                strategy_id=strategy_id,
                                signal=signal,
                                timestamp=datetime.now(timezone.utc),
                                sector=await self.market.get_sector(symbol),
                            ))
                        elif conv > 0.1:
                            # Log as skipped for comparison later
                            self.skipped_setups.append(SkippedSetup(
                                symbol=symbol,
                                strategy_id=strategy_id,
                                timestamp=datetime.now(timezone.utc),
                                conviction=conv,
                                skip_reason="low_conviction",
                                ranking_score=0.0,
                            ))

                    except Exception as e:
                        log.errors.append(f"{symbol}/{strategy_id}: {e}")
                        continue

            log.stocks_scanned = len(scanned_symbols)
            log.candidates_found = len(candidates)

            # ── STEP 2: RANK ──
            ranked = []
            if candidates and self._stock_ranker:
                ranked = await self._stock_ranker.rank_candidates(
                    candidates=candidates,
                    current_regime=current_regime,
                    current_positions=self.portfolio.open_positions,
                    strategy_versions=self.strategy_versions,
                )
            elif candidates:
                # Fallback: rank by conviction alone
                ranked = self._rank_by_conviction(candidates)

            log.ranked_count = len(ranked)

            # ── STEP 3: FILTER ──
            executable = self._apply_portfolio_filters(ranked)
            log.filtered_count = len(executable)

            # Log setups that were ranked but filtered out
            for rc in ranked:
                if rc not in executable:
                    self.skipped_setups.append(SkippedSetup(
                        symbol=rc.candidate.symbol,
                        strategy_id=rc.candidate.strategy_id,
                        timestamp=datetime.now(timezone.utc),
                        conviction=getattr(rc.candidate.signal, "conviction_score", 0.0),
                        skip_reason="portfolio_filter",
                        ranking_score=rc.composite_score,
                        would_have_been_rank=rc.rank,
                    ))

            # ── STEP 4: EXECUTE ──
            for rc in executable:
                try:
                    await self._open_virtual_position(rc, regime_dict)
                    log.executed_count += 1
                except Exception as e:
                    log.errors.append(f"Open {rc.candidate.symbol}: {e}")

            # ── STEP 5: MANAGE ──
            log.positions_managed = len(self.portfolio.open_positions)
            for position in list(self.portfolio.open_positions):
                try:
                    # OPTIONS (M3b): delegate the 1-min lifecycle to OpenOptionPosition (own exit rules); the generic
                    # underlying exit path below is for futures/equity only.
                    if getattr(position, "instrument", "future") == "option":
                        if await self._manage_option_position(position):
                            log.exits_triggered += 1
                        continue
                    exit_decision = await self._evaluate_exit(
                        position, current_regime
                    )
                    if exit_decision.should_exit:
                        await self._close_virtual_position(
                            position, exit_decision
                        )
                        log.exits_triggered += 1
                except Exception as e:
                    log.errors.append(f"Manage {position.symbol}: {e}")

            self.status = AVTStatus.RUNNING

        except Exception as e:
            logger.error("Scan cycle error: %s", e, exc_info=True)
            log.errors.append(str(e))
            self.status = AVTStatus.ERROR

        log.duration_ms = (time.monotonic() - t0) * 1000
        self.scan_logs.append(log)
        self._cycle_count += 1
        self._last_scan_at = self._now()

        # Keep logs bounded
        if len(self.scan_logs) > 500:
            self.scan_logs = self.scan_logs[-500:]
        if len(self.skipped_setups) > 5000:
            self.skipped_setups = self.skipped_setups[-5000:]

        return log

    # ── Signal Generation ──────────────────────────────────────

    async def _generate_signal(
        self,
        symbol: str,
        version: StrategyVersion,
        current_regime: Any,
    ) -> Any:
        """Generate signal for a symbol using the SKE signal generator."""
        if not self.signal_gen:
            return None

        candles = await self.market.get_candles(symbol, count=100)
        if not candles or len(candles) < 30:
            return None

        sko = version.get_active_sko()

        # SignalGenerator.evaluate() signature — OFFLOAD to a thread: for the cascade this runs prime_live
        # (5m/15m/1h narration), heavy sync work that would block the event loop (freezing all API requests).
        signal = await asyncio.to_thread(
            self.signal_gen.evaluate, sko=sko, candles=candles, symbol=symbol,
        )
        return signal

    async def _refresh_lake_today(self, symbols) -> None:
        """ONE controlled, throttled pass that fetches TODAY's 1-min bars for the scan universe into the lake and
        warms the candle_store's today-cache — so the rest of the scan (recognition, cascade priming, the serial
        signal loop) reads CACHED data instead of each hammering Angel's rate-limited historical API. 5m/15m/1h all
        resample from 1m, so one 1m fetch per symbol freshens every timeframe. Sequential + a small delay keeps us
        under the broker's limit (no burst → no AB1021 death-spiral). EXACT signals: this fetches the very same 1m
        bars the scan would have fetched itself — it only does them once, up front, gently. Live only."""
        if self._clock is not None:
            return
        from marketdata.candle_store import get_candle_store
        import pandas as pd
        cs = get_candle_store()
        now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
        lo = now.normalize() - pd.Timedelta(days=1)     # today + yesterday's tail; the warmup history is in the lake
        # BOUNDED concurrency: a full-session 1m fetch is ~3s of broker latency, so pure-sequential over ~33 symbols
        # would be ~100s. A small in-flight cap (default 5) overlaps the I/O to ~20s while staying well under the
        # broker's sustained rate limit (≈1.5 req/s here) — no burst, no AB1021.
        limit = max(1, int(getattr(self.config, "lake_refresh_concurrency", 5)))
        sem = asyncio.Semaphore(limit)
        t0 = time.monotonic()
        ok = 0

        async def _pull(sym):
            nonlocal ok
            async with sem:
                try:
                    await asyncio.to_thread(cs.get, sym, "1m", lo, now, True)  # fetch today, write lake, warm cache
                    ok += 1
                except Exception as e:
                    logger.debug("lake refresh %s failed: %s", sym, e)
        await asyncio.gather(*[_pull(s) for s in symbols])
        logger.info("AVT lake refresh: %d/%d symbols today-fresh in %.1fs", ok, len(symbols), time.monotonic() - t0)

    async def _prewarm_cascade_narration(self, is_cascade) -> None:
        """Concurrently (threads) prime the cascade narration for the symbols this cycle will evaluate, BEFORE the
        serial signal loop — so post-close latency is the parallel narration time, not the serial sum. Uses the
        dispatcher's own throttled `_ensure_primed_live` (per-symbol keys → thread-safe; the later serial calls
        then no-op). LIVE only. Bit-identical output — this changes only WHEN priming happens, never what it does."""
        sg = self.signal_gen
        ens = getattr(sg, "_ensure_primed_live", None)
        if ens is None or not getattr(sg, "live_prime", False) or self._clock is not None:
            return                                  # replay / non-cascade dispatcher: nothing to pre-warm
        syms: set = set()
        if self._lit_cache is not None:
            syms = set(self._lit_cache)             # two-tier ON: only the lit symbols get primed anyway
        else:
            for sid in self.strategy_versions:
                if is_cascade(self.strategies.get(sid, {})):
                    syms.update(self.strategy_watchlists.get(sid)
                                or self.strategies.get(sid, {}).get("watchlist") or self.watchlist)
        if not syms:
            return
        # IST-naive wall-clock: the lake/feed candles are IST-naive, and prime_live's closed-bar cutoff compares
        # against `now` — a UTC `now` would drop all of today's (IST-timestamped) bars. Match the serial path.
        import pandas as pd
        cur_ts = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)

        # BOUND the concurrency: each prime_live is several live candle fetches, and the broker (Angel) rate-limits
        # historical data hard ("Too many requests"/AB1021). A small semaphore keeps enough I/O overlap to be fast
        # WITHOUT flooding the API into timeouts+retries (which is far slower than a throttled serial-ish burst).
        limit = max(1, int(getattr(self.config, "prewarm_concurrency", 3)))
        sem = asyncio.Semaphore(limit)

        async def _one(sym):
            async with sem:
                try:
                    await asyncio.to_thread(ens, sym, cur_ts)
                except Exception as e:
                    logger.debug("cascade prewarm %s failed: %s", sym, e)
        await asyncio.gather(*[_one(s) for s in syms])

    def _peripheral_lit(self) -> set:
        """Two-tier attention (B.2): the cheap peripheral pass. Run `recognize` (5m BirthTracker only, ~0.1s warm)
        on every watchlist symbol and return the set that is 'lit' — i.e. could host a cascade signal (a 5m
        birth / active move this session). The expensive cascade priming then runs ONLY on these; quiet symbols
        are skipped. SAFETY: this is a strict SUPERSET of cascade candidates — a cascade REQUIRES a 5m birth, and
        recognition sees births on the same tracker family; anything recognition can't read (no warm candles,
        error) FAILS OPEN (kept lit → full read), so the gate can never drop a real signal. Held symbols are
        always kept lit. Runs once per cycle."""
        from intelligence.recognize import recognize, ATTENTION_NONE

        # union of every symbol this cycle scans (global ∪ per-strategy watchlists ∪ SKO-pinned watchlists),
        # plus anything we hold — otherwise a template pinned to its own universe (e.g. index options) would be
        # invisible to recognition and get gated out by the two-tier attention check.
        symbols: set = set(self.watchlist or [])
        for syms in (self.strategy_watchlists or {}).values():
            symbols.update(syms or [])
        for s in (self.strategies or {}).values():
            symbols.update(s.get("watchlist") or [])
        held = {p.symbol for p in self.portfolio.open_positions}

        # LIVE (no replay clock) → as_of=None uses the warm live-candle cache (loop-friendly). If a replay clock is
        # active, read the exact point-in-time window instead (keeps the gate correct if ever enabled off-live).
        as_of = self._now() if getattr(self, "_clock", None) is not None else None

        lit: set = set(held)
        for sym in symbols:
            try:
                r = recognize(sym, "5m", as_of=as_of)
            except Exception:
                lit.add(sym)                            # fail-open: never drop on a recognition error
                continue
            if (not r.ok) or r.active or r.n_births > 0 or r.attention != ATTENTION_NONE:
                lit.add(sym)
        return lit

    async def _compute_regime(self) -> Any:
        """Compute current market regime."""
        # RegimeFingerprintComputer.compute() needs candles + market_data
        # Use a broad market proxy (e.g., NIFTY50)
        candles = await self.market.get_candles("NIFTY50", count=100)
        if candles and len(candles) >= 20:
            return self.regime.compute(candles, {})
        return None

    # ── Ranking Fallback ───────────────────────────────────────

    @staticmethod
    def _rank_by_conviction(candidates: List[SetupCandidate]):
        """Simple fallback ranking when stock ranker not available."""
        from avt.models import RankedCandidate
        ranked = []
        for c in candidates:
            conv = getattr(c.signal, "conviction_score", 0.0)
            ranked.append(RankedCandidate(
                candidate=c,
                composite_score=conv,
                score_breakdown={"conviction": conv},
                ranking_explanation=f"Conviction: {conv:.2f}",
            ))
        ranked.sort(key=lambda r: r.composite_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        return ranked

    # ── Portfolio Filters ──────────────────────────────────────

    def _candidate_instrument(self, candidate) -> str:
        """The instrument a candidate would trade (via its template's preferred instrument) — for sleeve routing."""
        from avt.templates import preferred_instrument as _pref
        version = self.strategy_versions.get(candidate.strategy_id)
        if version is None:
            return "future"
        try:
            return _pref(version.get_active_sko()).value
        except Exception:
            return "future"

    def _apply_portfolio_filters(self, ranked) -> list:
        """
        Apply position limits:
          - Max N positions open simultaneously
          - Max exposure per stock
          - Max exposure per strategy
          - Max exposure per sector
          - Available virtual capital

        When capital sleeves are configured (`allocator.is_partitioned`), slots + capital are enforced PER SLEEVE
        so a full futures book never blocks an option; otherwise the historical global shared-pool path runs
        byte-identical (the futures golden is unmoved).
        """
        if self._allocator.is_partitioned:
            return self._apply_portfolio_filters_sleeved(ranked)

        executable = []
        cfg = self.config

        open_count = len(self.portfolio.open_positions)
        open_by_strategy: Dict[str, int] = {}
        open_by_sector: Dict[str, int] = {}
        open_symbols: set = set()

        for p in self.portfolio.open_positions:
            open_by_strategy[p.strategy_id] = open_by_strategy.get(p.strategy_id, 0) + 1
            open_symbols.add(p.symbol)

        for rc in ranked:
            c = rc.candidate

            # Already have position in this stock
            if c.symbol in open_symbols:
                continue

            # Max total positions
            if open_count + len(executable) >= cfg.max_open_positions:
                break

            # Max positions per strategy
            strat_count = open_by_strategy.get(c.strategy_id, 0) + \
                sum(1 for e in executable if e.candidate.strategy_id == c.strategy_id)
            if strat_count >= cfg.max_positions_per_strategy:
                continue

            # Max positions per sector
            sector_count = open_by_sector.get(c.sector, 0) + \
                sum(1 for e in executable if e.candidate.sector == c.sector)
            if sector_count >= cfg.max_positions_per_sector:
                continue

            # Capital check
            if self.portfolio.available_capital < self.portfolio.equity * 0.05:
                break

            executable.append(rc)
            open_symbols.add(c.symbol)

        return executable

    def _apply_portfolio_filters_sleeved(self, ranked) -> list:
        """Sleeve-aware variant of `_apply_portfolio_filters`: total-position slots and available-capital are
        enforced PER SLEEVE (per-strategy / per-sector / per-stock limits stay global). This is what lets futures
        and options run concurrently off separate pools — filling the futures sleeve's slots leaves the options
        sleeve free, and each sleeve's 5%-of-equity capital floor is checked against its OWN capital."""
        executable = []
        cfg = self.config
        alloc = self._allocator

        open_by_strategy: Dict[str, int] = {}
        open_by_sector: Dict[str, int] = {}
        open_symbols: set = set()
        for p in self.portfolio.open_positions:
            open_by_strategy[p.strategy_id] = open_by_strategy.get(p.strategy_id, 0) + 1
            open_symbols.add(p.symbol)

        # live per-sleeve open counts (by sleeve name) — futures slots and options slots are independent
        sleeve_open: Dict[str, int] = {}
        for sl in alloc.sleeves:
            sleeve_open[sl.name] = alloc.sleeve_open_count(sl)
        exec_by_sleeve: Dict[str, int] = {}

        for rc in ranked:
            c = rc.candidate
            if c.symbol in open_symbols:
                continue

            sleeve = alloc.sleeve_for(self._candidate_instrument(c))

            # Max total positions — PER SLEEVE
            taken = sleeve_open.get(sleeve.name, 0) + exec_by_sleeve.get(sleeve.name, 0)
            if taken >= sleeve.max_positions:
                continue

            # Max positions per strategy (global)
            strat_count = open_by_strategy.get(c.strategy_id, 0) + \
                sum(1 for e in executable if e.candidate.strategy_id == c.strategy_id)
            if strat_count >= cfg.max_positions_per_strategy:
                continue

            # Max positions per sector (global)
            sector_count = open_by_sector.get(c.sector, 0) + \
                sum(1 for e in executable if e.candidate.sector == c.sector)
            if sector_count >= cfg.max_positions_per_sector:
                continue

            # Capital check — against THIS SLEEVE's own capital
            if alloc.sleeve_available(sleeve) < alloc.sleeve_equity(sleeve) * 0.05:
                continue

            executable.append(rc)
            open_symbols.add(c.symbol)
            exec_by_sleeve[sleeve.name] = exec_by_sleeve.get(sleeve.name, 0) + 1

        return executable

    def _log_entry_slippage(self, candidate, signal, fill_price: float) -> None:
        """Record the LIVE execution slippage: intended entry (confirmation-bar close = signal.entry_ref) vs the
        actual fill (live LTP). This is the honest measurement of the post-close latency cost we agreed to watch;
        it's advisory only (never alters the trade). LIVE only (replay fills AT the close → slippage is 0)."""
        if self._clock is not None:
            return
        intended = float(getattr(signal, "entry_ref", 0.0) or 0.0)
        if intended <= 0 or fill_price <= 0:
            return
        try:
            import csv
            from pathlib import Path
            direction = "short" if getattr(signal, "direction", "long") == "short" else "long"
            # signed so POSITIVE = adverse (paid up on a long / sold lower on a short)
            slip_pts = (fill_price - intended) if direction == "long" else (intended - fill_price)
            atr = float(getattr(signal, "atr", 0.0) or 0.0)
            stop = float(getattr(signal, "stop_level", 0.0) or 0.0)
            risk = abs(intended - stop) if stop > 0 else 0.0
            row = {
                "ts": self._now().isoformat(), "trader": str(self.trader_id), "symbol": candidate.symbol,
                "strategy": candidate.strategy_id, "direction": direction,
                "intended": round(intended, 2), "fill": round(fill_price, 2),
                "slip_pts": round(slip_pts, 3), "slip_pct": round(100 * slip_pts / intended, 4),
                "slip_atr": round(slip_pts / atr, 4) if atr > 0 else "",
                "slip_R": round(slip_pts / risk, 4) if risk > 0 else "",
            }
            p = Path("data/avt_live_slippage.csv")
            p.parent.mkdir(parents=True, exist_ok=True)
            new = not p.exists()
            with p.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new:
                    w.writeheader()
                w.writerow(row)
            logger.info("AVT entry-slippage %s %s: intended %.2f → fill %.2f (%+.3f pts, %+.3f R)",
                        candidate.symbol, direction, intended, fill_price, slip_pts, row["slip_R"] or 0.0)
        except Exception as e:
            logger.debug("entry-slippage log failed for %s: %s", candidate.symbol, e)

    def _log_exit_slippage(self, position, intended_level: float, fill_price: float, reason: str) -> None:
        """Record LIVE EXIT slippage: the intended exit LEVEL (hard-stop price or trail level) vs the actual fill
        (live LTP at the scan that detected it). Symmetric with entry slippage — this measures the exit-side
        cost of a bar-cadence system (an intra-bar breach is only acted on at the next close). Advisory only,
        LIVE only (replay fills AT the level → 0). data/avt_live_exit_slippage.csv."""
        if self._clock is not None:
            return
        if not intended_level or intended_level <= 0 or fill_price <= 0:
            return
        try:
            import csv
            from pathlib import Path
            long = position.direction == "long"
            # signed so POSITIVE = adverse (sold below / covered above the intended exit level)
            slip_pts = (intended_level - fill_price) if long else (fill_price - intended_level)
            risk = abs(position.entry_price - position.stop_price) if position.stop_price else 0.0
            row = {
                "ts": self._now().isoformat(), "trader": str(self.trader_id), "symbol": position.symbol,
                "strategy": position.strategy_id, "direction": position.direction, "reason": reason,
                "intended": round(intended_level, 2), "fill": round(fill_price, 2),
                "slip_pts": round(slip_pts, 3), "slip_pct": round(100 * slip_pts / intended_level, 4),
                "slip_R": round(slip_pts / risk, 4) if risk > 0 else "",
            }
            p = Path("data/avt_live_exit_slippage.csv")
            p.parent.mkdir(parents=True, exist_ok=True)
            new = not p.exists()
            with p.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new:
                    w.writeheader()
                w.writerow(row)
            logger.info("AVT exit-slippage %s %s (%s): intended %.2f → fill %.2f (%+.3f pts, %+.3f R)",
                        position.symbol, position.direction, reason, intended_level, fill_price,
                        slip_pts, row["slip_R"] or 0.0)
        except Exception as e:
            logger.debug("exit-slippage log failed for %s: %s", position.symbol, e)

    # ── Open Position ──────────────────────────────────────────

    async def _open_virtual_position(self, rc, regime_dict: Optional[Dict]):
        """Open a virtual trade with full metadata recording."""
        candidate = rc.candidate
        version = self.strategy_versions[candidate.strategy_id]
        sko = version.get_active_sko()
        signal = candidate.signal

        # Get entry price — the LIVE fill (market at the first tick after the bar close). The strategy's INTENDED
        # entry is the confirmation-bar close (signal.entry_ref); the gap between them is the real execution
        # slippage we agreed to MEASURE live. Fill logic is unchanged — we only record the delta.
        entry_price = await self.market.get_current_price(
            candidate.symbol, side="buy"
        )
        if entry_price <= 0:
            return
        self._log_entry_slippage(candidate, signal, entry_price)

        # Get stop from signal or compute from exit_logic
        stop_price = getattr(signal, "stop_level", None)
        if not stop_price:
            stop_price = self._compute_stop_from_sko(sko, entry_price)
        if not stop_price or stop_price <= 0:
            stop_price = entry_price * 0.98  # fallback 2% stop

        # Get targets
        target_prices = getattr(signal, "target_levels", [])

        # Determine direction (needed to build the Intent)
        direction = "long"  # default; strategies may specify
        exit_rules = sko.get("exit_rules", {})
        if exit_rules.get("direction") == "short":
            direction = "short"

        # ── The spine seam: Strategy → INTENT → Capital Operator → Execution Engine ──
        # Build the Intent from the fully-derived entry/stop/direction, ask the operator (v0: advisory reflex →
        # take at the preferred instrument, so the futures/golden path is unchanged), then dispatch to the
        # instrument's ExecutionEngine for concrete sizing (FuturesExecution reproduces compute_size exactly).
        from avt.intent import Intent
        from avt.templates import is_cascade_sko as _is_cascade, preferred_instrument as _pref
        from avt.execution import for_instrument

        intent = Intent(
            symbol=candidate.symbol,
            direction=direction,
            confidence=float(getattr(signal, "conviction_score", 0.5)),
            entry_ref=entry_price,
            structural_stop=stop_price,
            atr=float(getattr(signal, "atr", 0.0) or 0.0),
            entry_ts=self._now(),
            preferred_instrument=_pref(sko),
            strategy_id=candidate.strategy_id,
            template=sko.get("template"),
        )

        decision = self._capital_operator.decide_open(intent, as_of=self._now())
        if decision.action != "take":
            logger.info("AVT OPERATOR skip %s: %s", candidate.symbol,
                        "; ".join(decision.rationale[:2]))
            return

        # OPTIONS (M3b): the option lifecycle (sizing/booking/exit/accounting) is DELEGATED to the proven options
        # primitives — this branch handles it end-to-end and RETURNS, so the futures/equity block below is untouched.
        from avt.intent import Instrument as _Instr
        if decision.instrument == _Instr.OPTION:
            await self._open_option_position(rc, candidate, sko, version, signal,
                                             entry_price, stop_price, direction, regime_dict)
            return

        # Size off the funding SLEEVE's capital when the account is partitioned (B.1); else the shared portfolio
        # equity (byte-identical to the historical path / futures golden).
        _sleeve = None
        if self._allocator.is_partitioned:
            _sleeve = self._allocator.sleeve_for(decision.instrument.value)
            _size_equity = self._allocator.sleeve_equity(_sleeve)
            _size_dd = self._allocator.sleeve_drawdown_pct(_sleeve)
        else:
            _size_equity, _size_dd = self.portfolio.equity, None

        # Per-sleeve SIZING (Q1): a sleeve may size futures/equity by fixed LOTS or ₹ AMOUNT (the user's model)
        # instead of risk-%. Only when partitioned + the sleeve opts in; an unknown lot size falls back to risk-%.
        sizing = None
        if _sleeve is not None and _sleeve.size_mode in ("lots", "amount"):
            from avt.execution.sizing import sleeve_underlying_qty
            sizing = sleeve_underlying_qty(_sleeve.size_mode, _sleeve.size_value,
                                           candidate.symbol, entry_price, stop_price)
            if sizing is None and _sleeve.size_mode == "lots":
                logger.warning("AVT sleeve '%s' lots-sizing: no F&O lot for %s in the instrument master → "
                               "falling back to risk-%% sizing", _sleeve.name, candidate.symbol)
        if sizing is None:
            plan = for_instrument(decision.instrument).plan(
                intent, equity=_size_equity, config=self.config,
                sizer=self.position_manager, drawdown_pct=_size_dd,
            )
            if plan is None or plan.quantity <= 0:
                return
            # Reconstruct the sizing dict the recorder expects (identical values on the futures path).
            sizing = {
                "shares": int(plan.quantity),
                "risk_amount": float(plan.meta.get("risk_amount", 0.0) or 0.0),
                "risk_pct": float(plan.meta.get("risk_pct", 0.0) or 0.0),
            }

        # Apply lifecycle allocation multiplier (probation=0.25, degraded=0.5, etc.) — but NOT for system cascade
        # templates: they're pre-validated and bypass the taught lifecycle (else an unknown state → 0 allocation).
        if self._lifecycle_manager and not _is_cascade(sko):
            alloc = self._lifecycle_manager.get_allocation_multiplier(candidate.strategy_id)
            if alloc < 1.0:
                sizing["shares"] = int(sizing["shares"] * alloc)
                sizing["risk_amount"] *= alloc

        if sizing["shares"] <= 0:
            return

        # Build feature vector snapshot
        feature_dict = {}
        if signal and hasattr(signal, "individual_scores"):
            for fs in signal.individual_scores:
                feature_dict[fs.feature_name] = fs.raw_value

        virtual_trade = VirtualTrade(
            id=uuid4(),
            trader_id=self.trader_id,   # attribute to this engine's owner
            strategy_id=candidate.strategy_id,
            strategy_version=version.version,
            variant_id=getattr(signal, "matched_variant", None),
            symbol=candidate.symbol,
            direction=direction,
            entry_time=self._now(),
            entry_price=entry_price,
            quantity=sizing["shares"],
            remaining_quantity=sizing["shares"],
            stop_price=stop_price,
            entry_atr=getattr(signal, "atr", None) or None,
            target_prices=target_prices,
            risk_amount=sizing["risk_amount"],
            risk_pct=sizing["risk_pct"],
            conviction_at_entry=getattr(signal, "conviction_score", 0.0),
            feature_vector_at_entry=feature_dict,
            regime_at_entry=regime_dict,
            ranking_position=rc.rank,
            ranking_score=rc.composite_score,
            ranking_explanation=rc.ranking_explanation,
            competing_setups_count=len(self.watchlist),
            active_improvements=[
                str(imp.id) for imp in version.active_improvements
            ],
            instrument=decision.instrument.value,   # 'future' (== default) or 'equity' — for sleeve attribution
        )

        self.portfolio.add_position(virtual_trade)

        logger.info(
            "AVT OPEN: %s %s @ %.2f, stop=%.2f, qty=%d, conv=%.2f, rank=#%d",
            direction.upper(), candidate.symbol, entry_price,
            stop_price, sizing["shares"],
            virtual_trade.conviction_at_entry, rc.rank,
        )

        if self._on_trade_opened:
            self._on_trade_opened(virtual_trade)

    async def _open_option_position(self, rc, candidate, sko, version, signal,
                                    entry_price, stop_price, direction, regime_dict):
        """Book an OPTION expressing the cascade Intent, via the PROVEN options primitives (no option logic lives
        in this glue). Backtest: ride the 1-min (spot, premium) series to exit atomically → reproduces the money
        test. Live: leave open; the manage loop ticks it (A3). Option cash/pnl use the portfolio's P&L-only option
        methods (option slippage + charges already in the net)."""
        if self._option_market is None:
            return
        from avt.live.options_forward import OptionSignal, book_option, manage_to_exit

        now = self._now()
        right = "CE" if direction == "long" else "PE"
        contract = self._option_market.resolve_atm(candidate.symbol, entry_price, right, now)
        if not contract or not contract.get("lot_size"):
            return
        entry_prem = self._option_market.premium(contract, now)
        if entry_prem is None or entry_prem <= 1:
            return
        spot0 = self._option_market.spot(candidate.symbol, now)
        spot0 = float(spot0) if spot0 is not None else float(entry_price)

        # Entry-sanity (master §2p-A): LOG-ONLY here (never skip — keeps the verified engine/A5 option path
        # byte-identical). Records when the execution spot is already through the stop (born-through-stop); the
        # run_options paper runner owns the skip toggle. Telemetry only.
        from avt.execution.guards import entry_still_valid
        _ok_entry, _why_g = entry_still_valid(direction, spot0, float(stop_price))
        if not _ok_entry:
            logger.info("AVT OPTION entry-sanity (log-only): %s %s — %s (signal %.1f, exec_spot %.1f)",
                        direction, candidate.symbol, _why_g, float(entry_price), spot0)

        # SIZE via the operator (money brain), off the OPTIONS SLEEVE's capital when partitioned (B.1) — so a ₹50k
        # options pool sizes independently of the futures pool; else the shared portfolio equity (unchanged path).
        _osleeve = None
        if self._allocator.is_partitioned:
            _osleeve = self._allocator.sleeve_for("option")
            _acct = _osleeve.account
            _cash = self._allocator.sleeve_equity(_osleeve)
            _acct.set_equity(_cash)
        else:
            _acct = self._capital_operator.account
            _cash = self.portfolio.equity
            _acct.set_equity(_cash)
        _buy_fill = entry_prem + self._option_slip
        if _osleeve is not None and _osleeve.size_mode == "lots" and _osleeve.size_value >= 1:
            # Per-sleeve SIZING (Q1): the options sleeve sets a FIXED lot count (bounded by affordability + max_lots).
            from avt.execution.options import MoneyPolicy as _MP, money_lots as _ml
            lots = _ml(_MP(mode="fixed", fixed_lots=int(_osleeve.size_value), max_lots=self._option_money.max_lots),
                       capital=_cash, starting_capital=_osleeve.capital, buy_fill=_buy_fill, lot=int(contract["lot_size"]))
        else:
            lots, _why = self._capital_operator.size_option(
                buy_fill=_buy_fill, lot=int(contract["lot_size"]), account=_acct)
        if lots < 1:
            return
        sig = OptionSignal(candidate.symbol, direction, now, float(entry_price), float(stop_price))
        pos = book_option(sig, spot_at_entry=spot0, contract=contract, entry_prem=entry_prem,
                          cash=_cash, deploy=0.35, slip=self._option_slip,
                          ptrail=self._option_ptrail, lots=lots)
        if pos is None:
            return

        vt = VirtualTrade(
            id=uuid4(), trader_id=self.trader_id, strategy_id=candidate.strategy_id,
            strategy_version=version.version, variant_id=getattr(signal, "matched_variant", None),
            symbol=candidate.symbol, direction=direction, entry_time=now,
            entry_price=float(entry_prem), quantity=int(pos.qty), remaining_quantity=int(pos.qty),
            stop_price=float(stop_price), entry_atr=getattr(signal, "atr", None) or None,
            conviction_at_entry=getattr(signal, "conviction_score", 0.0), regime_at_entry=regime_dict,
            ranking_position=rc.rank, ranking_score=rc.composite_score,
            ranking_explanation=rc.ranking_explanation, competing_setups_count=len(self.watchlist),
            instrument="option", tradeable=str(contract["tradeable"]),
            option_strike=float(contract["strike"]), option_expiry=str(contract["expiry"]),
            option_right=right, option_lots=int(pos.lots), option_lot_size=int(pos.lot_size),
            underlying_stop=float(stop_price),
        )
        self.portfolio.add_option_position(vt)
        self._option_positions[vt.id] = (pos, contract)
        logger.info("AVT OPTION OPEN: %s %s @ %.2f x%dlot (qty %d) stop_spot %.1f",
                    direction.upper(), contract["tradeable"], entry_prem, pos.lots, pos.qty, stop_price)
        if self._on_trade_opened:
            self._on_trade_opened(vt)

        # BACKTEST: ride the day's 1-min series to exit atomically (the validated exit-at-spike + underlying stop).
        ticks = self._option_market.ride_ticks(contract, candidate.symbol, now)
        if ticks is not None:
            exit_prem, reason = manage_to_exit(pos, ticks)
            r = pos.settle(exit_prem)
            self.portfolio.close_option_position(vt, float(exit_prem), float(r["net"]), self._now(), reason)
            self._option_positions.pop(vt.id, None)
            logger.info("AVT OPTION CLOSE: %s @ %.2f [%s] net Rs%.0f",
                        contract["tradeable"], exit_prem, reason, r["net"])
            if self._on_trade_closed:
                self._on_trade_closed(vt)

    async def _manage_option_position(self, position) -> bool:
        """LIVE option management: feed the current (spot, premium) tick to the delegated OpenOptionPosition; exit on
        its 1-min rules (underlying stop / premium-trail) or EOD. Returns True if closed. Backtest options never
        reach here (closed atomically at open)."""
        import pandas as pd
        handle = self._option_positions.get(position.id)
        if handle is None:
            return False
        pos, contract = handle
        now = self._now()
        spot = self._option_market.spot(position.symbol, now)
        prem = self._option_market.premium(contract, now)
        if spot is None or prem is None:
            return False
        reason = pos.on_tick(float(spot), float(prem))
        hm = pd.Timestamp(now).strftime("%H:%M")
        if reason is None and hm >= "15:20":
            reason = "eod"
        position.unrealized_pnl = (pos.last_prem - pos.buy_fill) * pos.qty
        if reason:
            r = pos.settle(pos.last_prem)
            self.portfolio.close_option_position(position, float(pos.last_prem), float(r["net"]), now, reason)
            self._option_positions.pop(position.id, None)
            if self._on_trade_closed:
                self._on_trade_closed(position)
            return True
        return False

    @staticmethod
    def _compute_stop_from_sko(sko: Dict[str, Any], entry_price: float) -> Optional[float]:
        """Compute stop from SKO exit_rules if signal didn't provide one."""
        exit_rules = sko.get("exit_rules", {})
        risk_params = sko.get("risk_parameters", {})

        # Try ATR-based stop
        atr_mult = exit_rules.get("stop_atr_multiple") or risk_params.get("stop_atr_multiple")
        if atr_mult:
            # Estimate ATR as ~1.5% of price as fallback
            est_atr = entry_price * 0.015
            return entry_price - (est_atr * atr_mult)

        # Try fixed percentage stop
        stop_pct = exit_rules.get("stop_pct") or risk_params.get("stop_pct")
        if stop_pct:
            return entry_price * (1.0 - stop_pct / 100.0)

        return None

    # ── Exit Evaluation ────────────────────────────────────────

    async def _evaluate_exit(
        self,
        position: VirtualTrade,
        current_regime: Any,
    ) -> ExitDecision:
        """
        Evaluate all exit conditions for an open position.

        This is where AVT improvements manifest — learned improvements
        are checked as ADDITIONAL exit conditions on top of the base strategy.
        """
        version = self.strategy_versions.get(position.strategy_id)
        if not version:
            return ExitDecision(should_exit=False)

        sko = version.get_active_sko()
        exit_rules = sko.get("exit_rules", {})

        current_price = await self.market.get_current_price(position.symbol)
        if current_price <= 0:
            return ExitDecision(should_exit=False)

        current_r = compute_r_multiple(
            position.entry_price, current_price,
            position.stop_price, position.direction,
        )

        # Update unrealized P&L
        if position.direction == "long":
            position.unrealized_pnl = (current_price - position.entry_price) * position.remaining_quantity
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.remaining_quantity

        # Track maximum favorable excursion
        if current_r > position.max_favorable_r:
            position.max_favorable_r = current_r
            position.max_favorable_price = current_price
            position.max_favorable_time = self._now()

            # Capture features at peak (for R:R analysis)
            if self.features:
                try:
                    candles = await self.market.get_candles(position.symbol, count=50)
                    if candles:
                        position.features_at_peak = self.features.compute_features(
                            candles, position.symbol,
                        )
                except Exception:
                    pass

        # Track maximum adverse excursion
        if current_r < position.max_adverse_r:
            position.max_adverse_r = current_r

        # ── CHECK 0 (REPLAY): OHLC-aware intrabar stop/target fill ──
        # When replaying, fill stops/targets against the bar's low/high (not just
        # close), at the stop/target price. Conservative: if BOTH are touched in
        # one bar, assume the stop filled first. Skip the position's OWN entry bar
        # (entry was at that bar's close, so its earlier range is not tradeable).
        # Opt-out via exit_rules.close_based_stops (some research is close-only — a
        # wick through the stop that closes back inside does NOT stop out); the
        # close-based hard stop (CHECK 1) then governs. Realistic replay leaves this
        # OFF (intrabar wicks DO fill a resting stop).
        close_based = exit_rules.get("close_based_stops")
        bar = self._current_bar(position.symbol)
        if not close_based and bar is not None and position.entry_time != self._now():
            low = float(bar.get("low", current_price))
            high = float(bar.get("high", current_price))
            tgts = position.target_prices or []
            if position.direction == "long":
                stop_hit = low <= position.stop_price
                tgt_idx = next((i for i, t in enumerate(tgts) if high >= t), None)
            else:
                stop_hit = high >= position.stop_price
                tgt_idx = next((i for i, t in enumerate(tgts) if low <= t), None)
            if stop_hit:
                return ExitDecision(
                    should_exit=True, exit_type=ExitType.STOP_LOSS.value,
                    exit_portion=1.0, reason="Stop loss hit (intrabar)",
                    exit_price=position.stop_price,
                )
            if tgt_idx is not None:
                t = tgts[tgt_idx]
                full = tgt_idx >= len(tgts) - 1
                return ExitDecision(
                    should_exit=True, exit_type=ExitType.TARGET_HIT.value,
                    exit_portion=1.0 if full else 0.5,
                    reason=f"{'Final t' if full else 'T'}arget {tgt_idx+1} hit (intrabar) at {t:.2f}",
                    exit_price=t,
                )

        # ── CHECK 1: Hard stop loss ──
        is_stopped = (
            (position.direction == "long" and current_price <= position.stop_price) or
            (position.direction == "short" and current_price >= position.stop_price)
        )
        if is_stopped:
            self._log_exit_slippage(position, position.stop_price, current_price, "hard-stop")
            return ExitDecision(
                should_exit=True,
                exit_type=ExitType.STOP_LOSS.value,
                exit_portion=1.0,
                reason="Stop loss hit",
            )

        # ── CHECK 2: Story-intact evaluation ──
        story_result = await self._evaluate_story_intact(
            position, exit_rules, current_regime
        )
        if story_result.story_broken:
            return ExitDecision(
                should_exit=True,
                exit_type=ExitType.STORY_BROKEN.value,
                exit_portion=1.0,
                reason=story_result.reason,
            )

        # ── CHECK 3: Targets ──
        target_hit = self._check_targets(
            position, exit_rules, current_price, current_r
        )
        if target_hit and target_hit.should_exit:
            return target_hit

        # ── CHECK 4: Time exit ──
        time_exit = self._check_time_exit(position, exit_rules)
        if time_exit and time_exit.should_exit:
            return time_exit

        # ── CHECK 5: LEARNED IMPROVEMENTS ──
        if version.active_improvements:
            for improvement in version.active_improvements:
                imp_exit = await self._check_improvement_exit(
                    position, improvement,
                    current_price, current_r, current_regime,
                )
                if imp_exit and imp_exit.should_exit:
                    return imp_exit

        # ── CHECK 6: ATR TRAILING STOP (opt-in via exit_rules.trail_atr; additive —
        #    SKOs without trail_atr are unaffected). CLOSE-based (exit when the close
        #    retraces past high-water − trail_atr×ATR), matching the frozen CASCADE-V1
        #    2.5-ATR trail (research is close-based). Does NOT mutate position.stop_price,
        #    so the R-multiple stays measured against the ORIGINAL structural risk. ──
        trail_atr = exit_rules.get("trail_atr")
        if trail_atr and position.max_favorable_price:
            # Prefer the FIXED entry-bar ATR the signal supplied (fully warmed, matches the research trail);
            # fall back to a recomputed recent ATR when the signal didn't carry one.
            atr = position.entry_atr if position.entry_atr and position.entry_atr > 0 \
                else await self._recent_atr(position.symbol)
            if atr and atr > 0:
                if position.direction == "long":
                    trail_level = position.max_favorable_price - trail_atr * atr
                    if current_price <= trail_level:
                        self._log_exit_slippage(position, trail_level, current_price, "trail")
                        return ExitDecision(
                            should_exit=True, exit_type=ExitType.TRAILING_STOP.value,
                            exit_portion=1.0, exit_price=current_price,
                            reason=f"Trailing stop {trail_atr}ATR from peak ({current_r:.1f}R)",
                        )
                else:
                    trail_level = position.max_favorable_price + trail_atr * atr
                    if current_price >= trail_level:
                        self._log_exit_slippage(position, trail_level, current_price, "trail")
                        return ExitDecision(
                            should_exit=True, exit_type=ExitType.TRAILING_STOP.value,
                            exit_portion=1.0, exit_price=current_price,
                            reason=f"Trailing stop {trail_atr}ATR from peak ({current_r:.1f}R)",
                        )

        return ExitDecision(should_exit=False)

    async def _recent_atr(self, symbol: str, count: int = 60) -> Optional[float]:
        """Current ATR from recent candles (marketdata.features) — powers the trailing stop.
        Uses the SAME atr the cascade research/generator use, so the replay trail matches research."""
        try:
            candles = await self.market.get_candles(symbol, count=count)
            if not candles or len(candles) < 20:
                return None
            import pandas as pd
            from marketdata import features as _F
            df = pd.DataFrame(candles)
            df["ts"] = pd.to_datetime(df["ts"] if "ts" in df else df["timestamp"])
            feat = _F.compute_features(df, symbol, "5m")
            a = float(feat["atr"].iloc[-1])
            return a if (a == a and a > 0) else None
        except Exception:
            return None

    async def _evaluate_story_intact(
        self,
        position: VirtualTrade,
        exit_rules: Dict[str, Any],
        current_regime: Any,
    ) -> StoryIntactResult:
        """
        Evaluate whether the trade thesis still holds.

        Checks:
          - Key structure levels broken
          - Regime changed unfavorably
          - Volume dried up (no follow-through)
        """
        # Simple implementation: check if regime changed significantly
        if current_regime and position.regime_at_entry:
            entry_regime = position.regime_at_entry
            if hasattr(current_regime, "to_vector"):
                current_vec = current_regime.to_vector()
                entry_vec = [
                    entry_regime.get("index_trend", 0.5),
                    entry_regime.get("volatility_level", 0.5),
                    entry_regime.get("market_breadth", 0.5),
                    entry_regime.get("sector_participation", 0.5),
                    entry_regime.get("fii_dii_flow", 0.5),
                ]
                # Regime shift detection: if trend reversed significantly
                trend_shift = abs(current_vec[0] - entry_vec[0])
                if trend_shift > 0.4:
                    return StoryIntactResult(
                        story_broken=True,
                        reason=f"Regime trend shifted by {trend_shift:.2f} since entry",
                        confidence=min(1.0, trend_shift),
                    )

        # Check story_intact rules from SKO
        story_rules = exit_rules.get("story_intact", {})
        if story_rules.get("check_volume_followthrough"):
            candles = await self.market.get_candles(position.symbol, count=5)
            if candles and len(candles) >= 3:
                recent_vol = sum(c.get("volume", 0) for c in candles[-3:]) / 3
                entry_vol = position.feature_vector_at_entry.get("volume_surge", 1.0)
                if entry_vol > 1.5 and recent_vol < entry_vol * 0.3:
                    return StoryIntactResult(
                        story_broken=True,
                        reason="Volume follow-through collapsed",
                        confidence=0.7,
                    )

        return StoryIntactResult(story_broken=False)

    @staticmethod
    def _check_targets(
        position: VirtualTrade,
        exit_rules: Dict[str, Any],
        current_price: float,
        current_r: float,
    ) -> Optional[ExitDecision]:
        """Check if any target levels have been hit."""
        if not position.target_prices:
            return None

        for i, target in enumerate(position.target_prices):
            hit = (
                (position.direction == "long" and current_price >= target) or
                (position.direction == "short" and current_price <= target)
            )
            if hit:
                # First target → partial exit, last target → full exit
                if i < len(position.target_prices) - 1:
                    return ExitDecision(
                        should_exit=True,
                        exit_type=ExitType.TARGET_HIT.value,
                        exit_portion=0.5,
                        reason=f"Target {i+1} hit at {target:.2f} ({current_r:.1f}R)",
                    )
                else:
                    return ExitDecision(
                        should_exit=True,
                        exit_type=ExitType.TARGET_HIT.value,
                        exit_portion=1.0,
                        reason=f"Final target hit at {target:.2f} ({current_r:.1f}R)",
                    )
        return None

    def _check_time_exit(
        self,
        position: VirtualTrade,
        exit_rules: Dict[str, Any],
    ) -> Optional[ExitDecision]:
        """Check time-based exit conditions (clock-aware: replay uses bar time)."""
        if not position.entry_time:
            return None

        now = self._now()
        max_hold_minutes = exit_rules.get("max_hold_minutes")
        if max_hold_minutes:
            elapsed = (now - position.entry_time).total_seconds() / 60
            if elapsed >= max_hold_minutes:
                return ExitDecision(
                    should_exit=True,
                    exit_type=ExitType.TIME_EXIT.value,
                    exit_portion=1.0,
                    reason=f"Max hold time {max_hold_minutes}min exceeded ({elapsed:.0f}min)",
                )

        # Intraday market-close exit (~3:25 PM IST). Disabled for daily/swing
        # replay (force_intraday_close=False) so multi-day holds aren't force-closed.
        if self.force_intraday_close:
            # Resolve the current IST wall-clock. LIVE feeds a tz-aware UTC `now`
            # (datetime.now(timezone.utc)) → convert +5:30. REPLAY feeds the bar's
            # timestamp, which is already IST wall-clock and tz-NAIVE (NSE bars run
            # 09:15–15:30 IST) → use it directly. (The old code always added 5:30,
            # so replay force-closed at 09:55 IST — flattening every intraday position.)
            if now.tzinfo is not None:
                ist = now + timedelta(hours=5, minutes=30)
                ist_hour, ist_minute = ist.hour, ist.minute
            else:
                ist_hour, ist_minute = now.hour, now.minute
            if ist_hour > 15 or (ist_hour == 15 and ist_minute >= 25):
                return ExitDecision(
                    should_exit=True,
                    exit_type=ExitType.MARKET_CLOSE.value,
                    exit_portion=1.0,
                    reason="Market close approaching — closing intraday position",
                )

        return None

    async def _stop_watch_loop(self, stop_event: Optional[asyncio.Event] = None) -> None:
        """Poll the live LTP of OPEN positions ~every second and fire the HARD structural stop the instant it is
        breached — independent of the 5-min scan. This closes the born-through-stop gap (the realistic backtest
        already fills the hard stop intrabar, so this makes LIVE more faithful, not less). NO strategy change: only
        the fixed stop is watched; the 2.5-ATR trail remains close-based in the scan cycle. Live only."""
        if self._clock is not None:
            return
        interval = float(getattr(self.config, "stop_watch_interval_s", 1.0))
        logger.info("AVT hard-stop watch active (every %.1fs, open positions only)", interval)
        while True:
            if (stop_event and stop_event.is_set()) or self.status == AVTStatus.STOPPED:
                break
            try:
                if await self.market.is_market_open():
                    await self._stop_watch_tick()
            except Exception as e:
                logger.debug("stop-watch tick error: %s", e)
            await asyncio.sleep(interval)
        self._stop_watch_task = None

    async def _stop_watch_tick(self) -> None:
        """One poll: fetch LTP for every symbol we hold (futures/equity) and close any whose hard stop is breached."""
        positions = [p for p in list(self.portfolio.open_positions)
                     if getattr(p, "instrument", "future") != "option" and p.stop_price and p.stop_price > 0]
        if not positions:
            return
        prices: Dict[str, float] = {}

        async def _fetch(sym):
            try:
                prices[sym] = await self.market.get_current_price(sym)
            except Exception:
                pass
        await asyncio.gather(*[_fetch(s) for s in {p.symbol for p in positions}])
        for p in positions:
            if p not in self.portfolio.open_positions:      # already closed by the scan cycle (or a prior tick)
                continue
            px = prices.get(p.symbol)
            if not px or px <= 0:
                continue
            breached = ((p.direction == "long" and px <= p.stop_price)
                        or (p.direction == "short" and px >= p.stop_price))
            if not breached:
                continue
            self._log_exit_slippage(p, p.stop_price, px, "hard-stop(intrabar-watch)")
            # fill AT the breaching LTP (decision.exit_price → _close_virtual_position uses it; no await before the
            # close, so this stays race-free with the scan loop's MANAGE on the single event loop).
            decision = ExitDecision(should_exit=True, exit_type=ExitType.STOP_LOSS.value,
                                    exit_portion=1.0, exit_price=float(px),
                                    reason="Stop loss hit (intrabar watch)")
            await self._close_virtual_position(p, decision)

    async def _check_improvement_exit(
        self,
        position: VirtualTrade,
        improvement: ActiveImprovement,
        current_price: float,
        current_r: float,
        current_regime: Any,
    ) -> Optional[ExitDecision]:
        """
        Check if a learned improvement triggers an exit.

        R:R KILLER: "When trade reaches X:R AND conditions → book partial"
        EARLY EXIT: "When warning pattern detected → exit immediately"
        ENTRY FILTER: not applicable at exit (prevents entry)
        REGIME FILTER: not applicable at exit (prevents entry)
        """
        params = improvement.params

        if improvement.type == "partial_exit_at_rr":
            rr_threshold = params.get("rr_threshold", 2.0)
            if current_r >= rr_threshold:
                # Check additional conditions
                conditions_met = self._check_improvement_conditions(
                    position, improvement, current_regime
                )
                if conditions_met:
                    return ExitDecision(
                        should_exit=True,
                        exit_type=ExitType.IMPROVEMENT_PARTIAL_EXIT.value,
                        exit_portion=params.get("exit_portion", 0.5),
                        reason=f"Learned: {improvement.description}",
                        improvement_id=str(improvement.id),
                    )

        elif improvement.type == "early_exit_pattern":
            # Check if warning pattern is present in recent candles
            candles = await self.market.get_candles(position.symbol, count=10)
            if candles and self._detect_warning_pattern(candles, improvement):
                return ExitDecision(
                    should_exit=True,
                    exit_type=ExitType.IMPROVEMENT_EARLY_EXIT.value,
                    exit_portion=1.0,
                    reason=f"Learned: {improvement.description}",
                    improvement_id=str(improvement.id),
                )

        return None

    @staticmethod
    def _check_improvement_conditions(
        position: VirtualTrade,
        improvement: ActiveImprovement,
        current_regime: Any,
    ) -> bool:
        """Check if improvement's additional conditions are met."""
        params = improvement.params
        conditions = params.get("conditions", {})

        if not conditions:
            return True  # no conditions = always applicable

        # Regime condition
        required_regime = conditions.get("regime")
        if required_regime and current_regime:
            if hasattr(current_regime, "to_vector"):
                # Check if volatility is in expected range
                vol_level = current_regime.volatility_level
                if required_regime == "ranging" and vol_level > 0.6:
                    return False
                if required_regime == "trending" and vol_level < 0.3:
                    return False

        # Feature conditions (from entry features)
        for feat, threshold in conditions.items():
            if feat in ("regime",):
                continue
            entry_val = position.feature_vector_at_entry.get(feat)
            if entry_val is not None and isinstance(threshold, (int, float)):
                if entry_val < threshold:
                    return False

        return True

    @staticmethod
    def _detect_warning_pattern(
        candles: List[Dict[str, Any]],
        improvement: ActiveImprovement,
    ) -> bool:
        """Detect a warning pattern in recent candles."""
        params = improvement.params
        pattern_type = params.get("pattern", "")

        if not candles or len(candles) < 2:
            return False

        last = candles[-1]
        prev = candles[-2]

        if pattern_type == "engulfing_against":
            # Bearish engulfing for long positions
            last_body = last.get("close", 0) - last.get("open", 0)
            prev_body = prev.get("close", 0) - prev.get("open", 0)
            if last_body < 0 and prev_body > 0:
                if abs(last_body) > abs(prev_body) * 1.2:
                    return True

        elif pattern_type == "volume_spike_against":
            vol_avg = sum(c.get("volume", 0) for c in candles[:-1]) / max(len(candles) - 1, 1)
            last_vol = last.get("volume", 0)
            last_body = last.get("close", 0) - last.get("open", 0)
            if last_vol > vol_avg * 2.0 and last_body < 0:
                return True

        elif pattern_type == "momentum_divergence":
            # Price making new highs but RSI not confirming
            closes = [c.get("close", 0) for c in candles]
            if len(closes) >= 5:
                price_trending_up = closes[-1] > closes[-3]
                # Simple momentum check
                recent_momentum = closes[-1] - closes[-3]
                earlier_momentum = closes[-3] - closes[-5] if len(closes) >= 5 else 0
                if price_trending_up and recent_momentum < earlier_momentum * 0.5:
                    return True

        return False

    # ── Close Position ─────────────────────────────────────────

    async def _close_virtual_position(
        self,
        position: VirtualTrade,
        decision: ExitDecision,
    ) -> None:
        """Close (fully or partially) a virtual position."""
        # Prefer an explicit fill price (OHLC-aware intrabar stop/target in replay);
        # otherwise fill at the current (close) price.
        current_price = decision.exit_price if decision.exit_price and decision.exit_price > 0 else 0.0
        if current_price <= 0:
            current_price = await self.market.get_current_price(position.symbol)
        if current_price <= 0:
            current_price = position.entry_price  # fallback

        # Tag improvement-triggered exits
        if decision.improvement_id:
            position.exit_improvement_id = decision.improvement_id

        self.portfolio.close_position(
            trade=position,
            exit_price=current_price,
            exit_time=self._now(),
            exit_type=decision.exit_type,
            portion=decision.exit_portion,
        )

        if decision.exit_portion >= 1.0:
            logger.info(
                "AVT CLOSE: %s %s @ %.2f, R=%.2f, PnL=%.2f, reason=%s",
                position.direction.upper(), position.symbol,
                current_price, position.r_multiple,
                position.realized_pnl, decision.reason,
            )
        else:
            logger.info(
                "AVT PARTIAL: %s %s %.0f%% @ %.2f, reason=%s",
                position.direction.upper(), position.symbol,
                decision.exit_portion * 100, current_price,
                decision.reason,
            )

        if self._on_trade_closed and decision.exit_portion >= 1.0:
            self._on_trade_closed(position)

        # Feed edge monitor with closed trade R-multiple
        if decision.exit_portion >= 1.0 and self._lifecycle_manager:
            r_mult = getattr(position, "r_multiple", None) or getattr(position, "pnl_r", None)
            if r_mult is not None:
                try:
                    self._lifecycle_manager.feed_edge_monitor(
                        position.strategy_id, float(r_mult),
                    )
                except Exception as e:
                    logger.debug("Edge monitor feed failed: %s", e)

    # ── Background Runner ──────────────────────────────────────

    async def run(self, stop_event: Optional[asyncio.Event] = None):
        """
        Run the AVT as a persistent background service.
        Scans every config.scan_interval_seconds.
        """
        self.status = AVTStatus.RUNNING
        logger.info("AVT started with %d strategies, %d watchlist stocks",
                     len(self.strategies), len(self.watchlist))

        # Continuous hard-stop watch for OPEN positions — reacts within ~1s of a breach instead of waiting for the
        # 5-min scan (live only; started once). Cheap: only held symbols, no narration. The 2.5-ATR TRAIL stays
        # close-based (owned by the scan cycle); this watches ONLY the fixed structural stop.
        if self._clock is None and getattr(self, "_stop_watch_task", None) is None:
            self._stop_watch_task = asyncio.create_task(self._stop_watch_loop(stop_event))

        while True:
            if stop_event and stop_event.is_set():
                break

            if not await self.market.is_market_open():
                self.status = AVTStatus.IDLE
                await asyncio.sleep(60)
                continue

            try:
                log = await self.run_scan_cycle()
                if log.errors:
                    logger.warning("Cycle %d had %d errors", self._cycle_count, len(log.errors))
            except Exception as e:
                logger.error("AVT cycle error: %s", e, exc_info=True)

            # bar-close-aligned: sleep until just after the NEXT candle boundary (not a free-running interval
            # after the scan), so confirmations/entries land seconds after the close instead of drifting.
            await asyncio.sleep(self._sleep_to_next_bar())

        self.status = AVTStatus.STOPPED
        logger.info("AVT stopped after %d cycles", self._cycle_count)

    # ── Query Methods ──────────────────────────────────────────

    def get_open_positions(self) -> List[VirtualTrade]:
        return list(self.portfolio.open_positions)

    def get_closed_trades(
        self,
        strategy_id: Optional[str] = None,
        version: Optional[str] = None,
        after_date: Optional[datetime] = None,
    ) -> List[VirtualTrade]:
        """Get closed virtual trades with optional filters."""
        trades = self.portfolio.closed_trades
        if strategy_id:
            trades = [t for t in trades if t.strategy_id == strategy_id]
        if version:
            trades = [t for t in trades if t.strategy_version == version]
        if after_date:
            trades = [t for t in trades if t.entry_time and t.entry_time >= after_date]
        return trades

    def get_trades_where_improvement_applied(
        self,
        improvement_id: str,
    ) -> List[VirtualTrade]:
        """Get trades where a specific improvement was active and triggered an exit."""
        return [
            t for t in self.portfolio.closed_trades
            if t.exit_improvement_id == improvement_id
        ]

    def get_skipped_setups(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[SkippedSetup]:
        """Get setups AVT saw but chose not to trade."""
        setups = self.skipped_setups
        if symbol:
            setups = [s for s in setups if s.symbol == symbol]
        if since:
            setups = [s for s in setups if s.timestamp and s.timestamp >= since]
        return setups

    async def get_health(self) -> Dict[str, Any]:
        """
        User-level health snapshot — answers "is AVT actually working?".

        Distinguishes a healthy-but-quiet engine (market closed / no qualifying
        setups) from a misconfigured one (no tradeable strategies, empty
        watchlist, signal generator missing).
        """
        # "tradeable" = taught strategies (need a compiled scoring_config) OR system cascade templates (driven by
        # CascadeSignalGenerator, which needs no scoring_config). Without this, activated cascades read as "0
        # tradeable → not ready" even though they load + scan.
        from avt.templates import is_cascade_sko
        with_scoring = sum(
            1 for s in self.strategies.values()
            if isinstance(s, dict) and (s.get("scoring_config") or is_cascade_sko(s))
        )
        try:
            market_open = await self.market.is_market_open()
        except Exception:
            market_open = False

        last = self.scan_logs[-1] if self.scan_logs else None
        last_cycle = None
        if last is not None:
            last_cycle = {
                "scanned": last.stocks_scanned,
                "candidates": last.candidates_found,
                "ranked": last.ranked_count,
                "filtered": last.filtered_count,
                "executed": last.executed_count,
                "exits": last.exits_triggered,
                "duration_ms": round(last.duration_ms, 1),
            }

        recent_errors: List[str] = []
        for lg in self.scan_logs[-5:]:
            recent_errors.extend(lg.errors[:3])
        skip_counts: Dict[str, int] = {}
        for sk in self.skipped_setups[-200:]:
            skip_counts[sk.skip_reason] = skip_counts.get(sk.skip_reason, 0) + 1
        dominant_skip = (
            max(skip_counts, key=skip_counts.get) if skip_counts else None
        )

        if with_scoring == 0:
            health, reason = "not_ready", (
                "No tradeable strategies yet — teach or onboard a strategy first."
            )
        elif len(self.watchlist) == 0:
            health, reason = "not_ready", (
                "Watchlist is empty — add symbols to your master watchlist."
            )
        elif self.signal_gen is None:
            health, reason = "not_ready", "Signal generator not attached."
        elif not market_open:
            health, reason = "quiet", "Market closed — AVT scans during NSE hours."
        else:
            recent_exec = sum(lg.executed_count for lg in self.scan_logs[-10:])
            if recent_exec > 0 or self.portfolio.open_positions:
                health, reason = "healthy", "Scanning and trading."
            else:
                health, reason = "quiet", "Scanning — no qualifying setups yet."

        return {
            "health": health,
            "reason": reason,
            "strategies_loaded": len(self.strategies),
            "strategies_with_scoring_config": with_scoring,
            "watchlist_size": len(self.watchlist),
            "signal_gen_attached": self.signal_gen is not None,
            "stock_ranker_attached": self._stock_ranker is not None,
            "market_data_source": type(self.market).__name__,
            "market_open": market_open,
            "status": self.status.value,
            "cycle_count": self._cycle_count,
            "last_scan_at": (
                self._last_scan_at.isoformat() if self._last_scan_at else None
            ),
            "open_positions": len(self.portfolio.open_positions),
            "last_cycle": last_cycle,
            "dominant_skip_reason": dominant_skip,
            "recent_errors": recent_errors[:5],
        }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        p = self.portfolio
        return {
            "equity": p.equity,
            "cash": p.cash,
            "starting_capital": p.starting_capital,
            "total_pnl": p.total_pnl,
            "total_pnl_pct": (p.total_pnl / p.starting_capital * 100) if p.starting_capital else 0,
            "open_positions": len(p.open_positions),
            "total_trades": p.trade_count,
            "closed_trades": len(p.closed_trades),
            "peak_equity": p.peak_equity,
            "current_drawdown_pct": p.current_drawdown_pct * 100,
            "status": self.status.value,
            "cycle_count": self._cycle_count,
            # Capital sleeves (B.1): per-pool capital / equity / free / open slots so the user can SEE how the
            # account is split across futures / options / equity. Single implicit "all" sleeve when unpartitioned.
            "capital_sleeves": self._allocator.summary(),
            "partitioned": self._allocator.is_partitioned,
        }


# ── Per-user engine registry ───────────────────────────────────
# Multi-user: one AutonomousVirtualTrader per trader_id, created lazily when a
# user starts AVT. A sentinel key is used for the legacy single-user/global mode
# (trader_id=None), so existing call sites keep working.

_GLOBAL_KEY = "__default__"
_avt_engines: Dict[str, AutonomousVirtualTrader] = {}


def _key(trader_id: Optional[str]) -> str:
    return str(trader_id) if trader_id is not None else _GLOBAL_KEY


def get_avt(trader_id: Optional[str] = None) -> Optional[AutonomousVirtualTrader]:
    """
    Get a user's AVT engine. With trader_id, returns that user's engine (or None).
    Without it, returns the engine if exactly one exists (back-compat for internal
    callers + single-user mode).
    """
    if trader_id is not None:
        return _avt_engines.get(str(trader_id))
    if len(_avt_engines) == 1:
        return next(iter(_avt_engines.values()))
    return _avt_engines.get(_GLOBAL_KEY)


def all_avt_engines() -> Dict[str, AutonomousVirtualTrader]:
    """All live engines keyed by trader_id (for the shared scheduler/shutdown)."""
    return dict(_avt_engines)


def init_avt(
    strategies: List[Dict[str, Any]],
    config: Optional[AVTConfig] = None,
    trader_id: Optional[str] = None,
    **kwargs,
) -> AutonomousVirtualTrader:
    """Create + register a user's AVT engine (keyed by trader_id)."""
    eng = AutonomousVirtualTrader(
        strategies=strategies, config=config, trader_id=trader_id, **kwargs,
    )
    _avt_engines[_key(trader_id)] = eng
    return eng


def remove_avt(trader_id: Optional[str] = None) -> None:
    """Drop a user's engine from the registry (on stop/teardown)."""
    _avt_engines.pop(_key(trader_id), None)
