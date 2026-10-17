# discovery/orchestrator.py
"""
Pattern Discovery Orchestrator

Runs the full 7-stage pipeline end to end:
  1. Intent Parser        → PatternQuery
  2. Event Extractor      → PatternFingerprint
  3. Similarity Scanner   → List[MatchedEvent]
  4. Statistical Backtester → List[BacktestResult]
  5. Regime Correlator    → CorrelationReport
  6. Strategy Generator   → GeneratedStrategy (YAML)
  7. Validator + Storer   → Accepted/Rejected → stored in library

Supports:
  - Status callbacks for real-time progress updates (WebSocket/SSE)
  - Partial results at each stage
  - Cancellation
  - Persistence of DiscoveryReport for review later
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

import numpy as np

from discovery.backtester import BacktestConfig, StatisticalBacktester
from discovery.event_extractor import CandleFetcher, EventExtractor
from discovery.intent_parser import IntentParser
from discovery.models import (
    DiscoveryReport,
    DiscoveryStatus,
    PatternQuery,
)
from discovery.regime_correlator import RegimeCorrelator
from discovery.similarity_scanner import ScanConfig, SimilarityScanner
from discovery.strategy_generator import StrategyGenerator
from discovery.validator import StrategyValidator

logger = logging.getLogger(__name__)

# Type for progress callback: (discovery_id, status, message, progress_pct)
ProgressCallback = Callable[[str, str, str, float], None]


# ── FNO universe (default scan list) ─────────────────────────────
# Top FNO stocks by liquidity — used when scope = "fno"
FNO_UNIVERSE = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "SBIN", "AXISBANK", "KOTAKBANK", "BAJFINANCE", "BHARTIARTL",
    "LT", "HINDUNILVR", "ITC", "MARUTI", "TATAMOTORS",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "WIPRO", "HCLTECH",
    "TECHM", "ADANIENT", "ADANIPORTS", "SUNPHARMA", "DRREDDY",
    "CIPLA", "DIVISLAB", "TITAN", "NESTLEIND", "ASIANPAINT",
    "ULTRACEMCO", "GRASIM", "M&M", "EICHERMOT", "HEROMOTOCO",
    "BAJAJ-AUTO", "BPCL", "ONGC", "NTPC", "POWERGRID",
    "COALINDIA", "IOC", "GAIL", "VEDL", "JINDALSTEL",
    "DLF", "INDIGO", "SBILIFE", "HDFCLIFE", "ICICIPRULI",
    "TATACONSUM", "BRITANNIA", "DABUR", "GODREJCP", "MARICO",
    "PIDILITIND", "BERGEPAINT", "HAVELLS", "VOLTAS", "SIEMENS",
    "ABB", "HAL", "BEL", "IRCTC", "TRENT",
    "POLYCAB", "PERSISTENT", "COFORGE", "LTIMINDTREE", "LTTS",
    "MPHASIS", "NAUKRI", "INDIAMART", "CHOLAFIN", "SHRIRAMFIN",
    "BAJAJFINSV", "MUTHOOTFIN", "MANAPPURAM", "PFC", "RECLTD",
    "BANKBARODA", "PNB", "FEDERALBNK", "INDUSINDBK", "IDFCFIRSTB",
    "APOLLOHOSP", "LALPATHLAB", "LUPIN", "AUROPHARMA", "BIOCON",
    "TATAPOWER", "TORNTPOWER", "BOSCHLTD", "MRF", "PAGEIND",
    "SRF", "PIIND", "DEEPAKNTR", "UPL", "ESCORTS",
    # Index futures
    "NIFTY", "BANKNIFTY",
]

NIFTY50 = FNO_UNIVERSE[:50]


class PatternDiscoveryOrchestrator:
    """
    End-to-end pattern discovery pipeline.

    Usage:
        orch = PatternDiscoveryOrchestrator(
            candle_fetcher=my_fetcher,
            llm_client=my_llm,           # optional
            on_progress=my_callback,      # optional
        )
        report = await orch.discover("Look at Reliance 15-min breakout at 10:30...")
        # report.status == DiscoveryStatus.REVIEW

        # Human reviews:
        orch.accept(report, notes="Looks good")
        # or
        orch.reject(report, reason="Low win rate")
    """

    def __init__(
        self,
        candle_fetcher: Optional[CandleFetcher] = None,
        llm_client: Any = None,
        on_progress: Optional[ProgressCallback] = None,
    ):
        self._fetcher = candle_fetcher
        self._llm = llm_client
        self._on_progress = on_progress

        # Initialize stages
        self._intent_parser = IntentParser(llm_client=llm_client)
        self._extractor = EventExtractor(fetcher=candle_fetcher)
        self._scanner = SimilarityScanner(fetcher=candle_fetcher)
        self._backtester = StatisticalBacktester(fetcher=candle_fetcher)
        self._correlator = RegimeCorrelator()
        self._generator = StrategyGenerator(llm_client=llm_client)
        self._validator = StrategyValidator()

        # Store active discoveries for review
        self._discoveries: Dict[str, DiscoveryReport] = {}

    def set_fetcher(self, fetcher: CandleFetcher) -> None:
        """Hot-swap candle fetcher (e.g. when broker connection is ready)."""
        self._fetcher = fetcher
        self._extractor.set_fetcher(fetcher)
        self._scanner.set_fetcher(fetcher)
        self._backtester.set_fetcher(fetcher)

    # ── Main pipeline ─────────────────────────────────────────────────

    async def discover(
        self,
        raw_input: str,
        query: Optional[PatternQuery] = None,
        scan_config: Optional[ScanConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
    ) -> DiscoveryReport:
        """
        Run the full 7-stage discovery pipeline.

        Args:
            raw_input: Natural-language description (e.g. "Reliance 15m breakout at 10:30")
            query: Pre-parsed PatternQuery (skips Stage 1 if provided)
            scan_config: Override scan parameters
            backtest_config: Override backtest parameters

        Returns:
            DiscoveryReport with status=REVIEW (awaiting human decision)
        """
        report = DiscoveryReport()
        discovery_id = str(report.discovery_id)

        try:
            # ── Stage 1: Intent Parsing ──
            self._update_progress(discovery_id, "parsing", "Parsing your observation...", 0.05)
            report.status = DiscoveryStatus.PARSING

            if query is None:
                query = await self._intent_parser.parse(raw_input)
            query.raw_input = raw_input
            report.query = query
            logger.info("Stage 1 complete: %s %s %s at %s",
                        query.symbol, query.timeframe, query.pattern_type.value, query.event_time)

            # ── Stage 2: Event Extraction ──
            self._update_progress(discovery_id, "extracting",
                                  f"Extracting pattern fingerprint from {query.symbol} {query.timeframe}...", 0.15)
            report.status = DiscoveryStatus.EXTRACTING

            fingerprint = await self._extractor.extract(query)
            report.fingerprint = fingerprint
            logger.info("Stage 2 complete: fingerprint vector = %s", fingerprint.vector.round(3))

            # ── Stage 3: Similarity Scanning ──
            symbols = self._resolve_scope(query.scope)
            self._update_progress(discovery_id, "scanning",
                                  f"Scanning {len(symbols)} stocks for similar patterns...", 0.25)
            report.status = DiscoveryStatus.SCANNING

            cfg = scan_config or ScanConfig(similarity_threshold=query.similarity_threshold)
            matches = await self._scanner.scan(fingerprint, query, symbols, cfg)
            report.matches = matches
            report.total_matches = len(matches)
            logger.info("Stage 3 complete: %d matches found", len(matches))

            if not matches:
                report.status = DiscoveryStatus.FAILED
                report.error = "No matching patterns found above the similarity threshold"
                self._discoveries[discovery_id] = report
                return report

            # ── Stage 4: Backtesting ──
            self._update_progress(discovery_id, "backtesting",
                                  f"Backtesting {len(matches)} matched events...", 0.45)
            report.status = DiscoveryStatus.BACKTESTING

            bt_cfg = backtest_config or BacktestConfig(
                timeframe_minutes=self._parse_tf_minutes(query.timeframe),
            )
            backtest_results = await self._backtester.backtest_all(matches, bt_cfg)
            report.backtest_results = backtest_results

            # Aggregate stats
            agg = StatisticalBacktester.aggregate(backtest_results)
            report.overall_win_rate = agg["win_rate"]
            report.overall_avg_r = agg["avg_r"]
            report.overall_profit_factor = agg["profit_factor"]
            report.median_hold_minutes = agg["median_hold_minutes"]
            report.avg_mae = agg["avg_mae"]

            # Per-stock stats
            report.top_stocks = StatisticalBacktester.per_stock_stats(backtest_results)

            logger.info("Stage 4 complete: WR=%.1f%%, Avg R=%.2f, PF=%.2f",
                        agg["win_rate"] * 100, agg["avg_r"], agg["profit_factor"])

            # ── Stage 5: Regime Correlation ──
            self._update_progress(discovery_id, "correlating",
                                  "Analyzing regime and time-of-day performance...", 0.65)
            report.status = DiscoveryStatus.CORRELATING

            correlation = self._correlator.analyse(backtest_results)
            report.correlation = correlation
            logger.info("Stage 5 complete: best=%s, worst=%s, time=%s",
                        correlation.best_regime, correlation.worst_regime, correlation.best_time_window)

            # ── Stage 6: Strategy Generation ──
            self._update_progress(discovery_id, "generating",
                                  "Generating strategy DNA...", 0.80)
            report.status = DiscoveryStatus.GENERATING

            strategy = await self._generator.generate(report)
            report.generated_strategy = strategy
            logger.info("Stage 6 complete: %s (confidence=%.0f%%)",
                        strategy.strategy_name, strategy.confidence * 100)

            # ── Stage 7: Validation ──
            self._update_progress(discovery_id, "review",
                                  "Strategy generated — awaiting your review", 0.95)
            report.status = DiscoveryStatus.REVIEW

            issues = self._validator.validate(strategy)
            if issues:
                logger.warning("Validation issues: %s", issues)
                report.error = f"Validation warnings: {'; '.join(issues)}"

            # Store for later review
            self._discoveries[discovery_id] = report

            self._update_progress(discovery_id, "review",
                                  "Discovery complete — ready for review!", 1.0)
            return report

        except Exception as e:
            logger.error("Discovery pipeline failed: %s", e, exc_info=True)
            report.status = DiscoveryStatus.FAILED
            report.error = str(e)
            self._discoveries[str(report.discovery_id)] = report
            self._update_progress(str(report.discovery_id), "failed", f"Error: {e}", 0)
            return report

    # ── Human review actions ──────────────────────────────────────────

    def accept(self, report: DiscoveryReport, notes: str = "") -> str:
        """Accept and store the discovered strategy. Returns file path."""
        return self._validator.accept(report, reviewer_notes=notes)

    def reject(self, report: DiscoveryReport, reason: str = "") -> None:
        """Reject the discovered strategy."""
        self._validator.reject(report, reason)

    async def accept_and_index(self, report: DiscoveryReport, notes: str = "", db: Any = None) -> str:
        """Accept, store, and index in RAG."""
        path = self._validator.accept(report, reviewer_notes=notes)
        await self._validator.index_in_rag(report, db=db)
        return path

    # ── Query existing discoveries ────────────────────────────────────

    def get_discovery(self, discovery_id: str) -> Optional[DiscoveryReport]:
        return self._discoveries.get(discovery_id)

    def list_discoveries(self) -> List[Dict]:
        return [
            {
                "discovery_id": str(r.discovery_id),
                "status": r.status.value,
                "query_symbol": r.query.symbol if r.query else "",
                "query_pattern": r.query.pattern_type.value if r.query else "",
                "total_matches": r.total_matches,
                "win_rate": round(r.overall_win_rate, 3),
                "created_at": r.created_at.isoformat(),
            }
            for r in self._discoveries.values()
        ]

    # ── Helpers ───────────────────────────────────────────────────────

    def _resolve_scope(self, scope: str) -> List[str]:
        """Convert scope string to list of symbols."""
        if scope == "fno":
            return FNO_UNIVERSE
        if scope == "nifty50":
            return NIFTY50
        if scope == "all":
            return FNO_UNIVERSE  # for now, FNO is our max universe
        # Comma-separated custom list
        if "," in scope:
            return [s.strip().upper() for s in scope.split(",")]
        return FNO_UNIVERSE

    def _parse_tf_minutes(self, tf: str) -> int:
        """Parse timeframe string to minutes."""
        tf = tf.lower().strip()
        if tf.endswith("m"):
            return int(tf[:-1])
        if tf.endswith("h"):
            return int(tf[:-1]) * 60
        if tf == "1d":
            return 375  # NSE trading day
        return 15  # default

    def _update_progress(self, discovery_id: str, status: str, message: str, pct: float) -> None:
        if self._on_progress:
            try:
                self._on_progress(discovery_id, status, message, pct)
            except Exception as e:
                logger.debug("Progress callback failed: %s", e)


# ── Module-level singleton ────────────────────────────────────────

_orchestrator: Optional[PatternDiscoveryOrchestrator] = None


def get_discovery_orchestrator() -> PatternDiscoveryOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PatternDiscoveryOrchestrator()
    return _orchestrator
