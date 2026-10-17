# automation/market_scanner_orchestrator.py
"""
Market Scanner Orchestrator
============================
The autonomous strategy loop that ties everything together:

    Watchlist → Candle Fetch → SKE Signals → Shadow Creation → Journal

Runs as an asyncio background task during market hours. For each scan cycle:

1. Loads watchlist symbols (master + FNO universe)
2. Fetches latest candles for each symbol
3. Runs signals against all active SKOs
4. Creates autonomous shadow trades for qualifying signals
5. Journals every trade for transparency

Configuration:
    - Scan interval: every 60s during market hours
    - Minimum conviction: 0.50 (configurable)
    - Max concurrent shadows: 20
    - Auto-pause outside market hours

Integration:
    Start via lifespan hook in api/main.py:
        from automation.market_scanner_orchestrator import get_orchestrator
        asyncio.create_task(get_orchestrator().run())
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timezone
from typing import Any, Callable, Dict, List, Optional

import pytz

from config.settings import settings
from core.events import Event, EventType, get_event_bus
from automation.autonomous_shadow import AutonomousShadowCreator
from automation.system_journal import SystemJournal
from ske.models import SignalOutput, VariantInfo
from ske.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)

# Type alias for candle fetcher
CandleFetchFn = Callable[[str, int], List[Dict[str, Any]]]

# Defaults
DEFAULT_SCAN_INTERVAL = 60          # seconds between scan cycles
DEFAULT_N_BARS = 100                # candles to fetch per symbol
DEFAULT_MIN_CONVICTION = 0.50       # minimum signal conviction
MARKET_HOURS_BUFFER_MINUTES = 5     # start scanning 5 min after open


def _noop_fetcher(symbol: str, n_bars: int) -> List[Dict[str, Any]]:
    """Default no-op fetcher — returns empty list."""
    return []


class MarketScannerOrchestrator:
    """
    Orchestrates the full autonomous scanning → signaling → shadow creation loop.

    This is the central "brain" that runs during market hours, continuously
    evaluating the trader's watchlist against all active SKOs.

    Usage:
        orchestrator = MarketScannerOrchestrator(db_factory, fetch_candles)
        asyncio.create_task(orchestrator.run())
    """

    def __init__(
        self,
        db_factory: Callable,
        fetch_candles: CandleFetchFn = _noop_fetcher,
        scan_interval: int = DEFAULT_SCAN_INTERVAL,
        min_conviction: float = DEFAULT_MIN_CONVICTION,
        n_bars: int = DEFAULT_N_BARS,
        default_capital: float = 500_000.0,
        max_open_shadows: int = 20,
    ):
        self._db_factory = db_factory
        self._fetch = fetch_candles
        self._interval = scan_interval
        self._min_conviction = min_conviction
        self._n_bars = n_bars
        self._running = False
        self._event_bus = get_event_bus()

        # Subcomponents
        self._signal_generator = SignalGenerator()
        self._shadow_creator = AutonomousShadowCreator(
            default_capital=default_capital,
            max_open_shadows=max_open_shadows,
        )

        # Watchlist
        self._symbols: List[str] = []
        self._sko_cache: List[Dict[str, Any]] = []
        self._variant_cache: Dict[str, List[VariantInfo]] = {}

        # Stats
        self._cycle_count = 0
        self._total_signals = 0
        self._total_shadows = 0

    # ── Configuration ──

    def set_symbols(self, symbols: List[str]) -> None:
        """Set the watchlist symbols to scan."""
        self._symbols = list(set(symbols))
        logger.info("Orchestrator watchlist: %d symbols", len(self._symbols))

    def add_symbols(self, symbols: List[str]) -> None:
        """Add symbols to the watchlist."""
        for s in symbols:
            if s not in self._symbols:
                self._symbols.append(s)

    def set_fetcher(self, fetch_fn: CandleFetchFn) -> None:
        """Inject a candle fetcher (e.g., from Zerodha adapter)."""
        self._fetch = fetch_fn

    def load_skos(self, skos: List[Dict[str, Any]]) -> None:
        """Load active SKOs for signal evaluation."""
        self._sko_cache = skos
        logger.info("Orchestrator loaded %d SKOs", len(skos))

    def load_variants(self, sko_id: str, variants: List[VariantInfo]) -> None:
        """Load variants for a specific SKO."""
        self._variant_cache[sko_id] = variants

    # ── Main Loop ──

    async def run(self) -> None:
        """
        Main async loop — runs during market hours.
        Call once as a background task.
        """
        self._running = True
        logger.info(
            "MarketScannerOrchestrator started: %d symbols, %d SKOs, "
            "interval=%ds, min_conviction=%.2f",
            len(self._symbols), len(self._sko_cache),
            self._interval, self._min_conviction,
        )

        while self._running:
            try:
                if self._is_market_hours():
                    await self._scan_cycle()
                else:
                    logger.debug("Outside market hours — sleeping")
            except Exception as e:
                logger.error("Orchestrator cycle error: %s", e, exc_info=True)

            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        """Stop the orchestrator loop."""
        self._running = False
        logger.info(
            "Orchestrator stopped: %d cycles, %d signals, %d shadows created",
            self._cycle_count, self._total_signals, self._total_shadows,
        )

    async def _scan_cycle(self) -> None:
        """Execute one complete scan cycle."""
        if not self._symbols or not self._sko_cache:
            return

        self._cycle_count += 1
        cycle_start = datetime.now(timezone.utc)

        # 1. Fetch candles concurrently
        loop = asyncio.get_event_loop()
        candles_map: Dict[str, List[Dict[str, Any]]] = {}

        fetch_tasks = {
            sym: loop.run_in_executor(None, self._fetch, sym, self._n_bars)
            for sym in self._symbols
        }

        for sym, coro in fetch_tasks.items():
            try:
                candles = await coro
                if candles and len(candles) >= 5:
                    candles_map[sym] = candles
            except Exception as e:
                logger.warning("Candle fetch failed for %s: %s", sym, e)

        if not candles_map:
            return

        # 2. Generate signals against all SKOs
        all_signals: List[SignalOutput] = []

        for sko in self._sko_cache:
            sko_id = str(sko.get("id", ""))
            variants = self._variant_cache.get(sko_id, [])

            signals = self._signal_generator.evaluate_batch(
                sko, candles_map, variants
            )
            all_signals.extend(signals)

        self._total_signals += len(all_signals)

        if not all_signals:
            return

        # 3. Create autonomous shadows for qualifying signals
        db = self._db_factory()
        try:
            journal = SystemJournal(db)

            shadows = self._shadow_creator.create_batch_from_signals(
                db, all_signals, candles_map
            )

            # 4. Journal each created shadow
            for shadow in shadows:
                journal.record_entry(shadow)

            self._total_shadows += len(shadows)

            elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            logger.info(
                "Orchestrator cycle #%d: %d symbols → %d signals → %d shadows "
                "(%.2fs)",
                self._cycle_count,
                len(candles_map),
                len(all_signals),
                len(shadows),
                elapsed,
            )

            # Publish cycle summary event
            self._event_bus.publish(Event(
                event_type=EventType.SETUP_DETECTED,
                source="market_scanner_orchestrator",
                data={
                    "cycle": self._cycle_count,
                    "symbols_scanned": len(candles_map),
                    "signals_generated": len(all_signals),
                    "shadows_created": len(shadows),
                    "elapsed_seconds": round(elapsed, 2),
                },
            ))
        finally:
            db.close()

    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours (with buffer)."""
        tz = pytz.timezone(settings.MARKET_TIMEZONE)
        now = datetime.now(tz)

        # Weekend check
        if now.weekday() >= 5:
            return False

        open_parts = settings.MARKET_OPEN_TIME.split(":")
        close_parts = settings.MARKET_CLOSE_TIME.split(":")

        market_open = time(int(open_parts[0]), int(open_parts[1]))
        market_close = time(int(close_parts[0]), int(close_parts[1]))

        current_time = now.time()

        # Add buffer after open
        from datetime import timedelta
        open_with_buffer = (
            datetime.combine(now.date(), market_open) +
            timedelta(minutes=MARKET_HOURS_BUFFER_MINUTES)
        ).time()

        return open_with_buffer <= current_time <= market_close

    # ── Synchronous scan (for REST endpoints) ──

    def scan_now(
        self,
        symbols: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Immediate synchronous scan — for REST API use.
        Returns list of signal dicts.
        """
        target_symbols = symbols or self._symbols
        if not target_symbols:
            return []

        candles_map = {}
        for sym in target_symbols:
            try:
                candles = self._fetch(sym, self._n_bars)
                if candles:
                    candles_map[sym] = candles
            except Exception as e:
                logger.warning("Fetch failed for %s: %s", sym, e)

        if not candles_map:
            return []

        all_signals: List[SignalOutput] = []
        for sko in self._sko_cache:
            sko_id = str(sko.get("id", ""))
            variants = self._variant_cache.get(sko_id, [])
            signals = self._signal_generator.evaluate_batch(
                sko, candles_map, variants
            )
            all_signals.extend(signals)

        return [sig.model_dump() for sig in all_signals]

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "running": self._running,
            "symbols_count": len(self._symbols),
            "skos_count": len(self._sko_cache),
            "total_cycles": self._cycle_count,
            "total_signals": self._total_signals,
            "total_shadows_created": self._total_shadows,
            "scan_interval_seconds": self._interval,
            "min_conviction": self._min_conviction,
        }


# ── Module-level singleton ──

_orchestrator: Optional[MarketScannerOrchestrator] = None


def get_orchestrator(
    db_factory: Optional[Callable] = None,
    fetch_candles: CandleFetchFn = _noop_fetcher,
    scan_interval: int = DEFAULT_SCAN_INTERVAL,
) -> MarketScannerOrchestrator:
    """
    Return (or create) the module-level orchestrator singleton.

    First call must pass db_factory to configure it.
    """
    global _orchestrator
    if _orchestrator is None:
        if db_factory is None:
            raise ValueError("db_factory required for first orchestrator creation")
        _orchestrator = MarketScannerOrchestrator(
            db_factory=db_factory,
            fetch_candles=fetch_candles,
            scan_interval=scan_interval,
        )
        logger.info("MarketScannerOrchestrator singleton created")
    return _orchestrator
