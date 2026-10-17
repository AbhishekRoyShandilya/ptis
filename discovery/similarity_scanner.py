# discovery/similarity_scanner.py
"""
Stage 3 — Universe Similarity Scanner

Scans historical candle data across the FNO universe to find events
that match the query fingerprint within the similarity threshold.

Architecture:
  - Takes a PatternFingerprint (the "query") and a similarity threshold
  - Iterates over each stock × each date in the lookback window
  - At every bar, computes a fingerprint on-the-fly
  - Keeps matches above threshold, sorted by similarity desc
  - Designed for batch processing (can be parallelised per stock)

Performance:
  - ~200 stocks × 252 days × 26 bars/day ≈ 1.3M bars to scan (3 years)
  - Fingerprint is O(n) per bar (uses pre-computed indicators)
  - Early termination: skip bar if volume_surge alone is too low
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from discovery.event_extractor import EventExtractor
from discovery.models import MatchedEvent, PatternFingerprint, PatternQuery

logger = logging.getLogger(__name__)

# Type for bulk candle fetcher: fetch(symbol, timeframe, start, end) -> DataFrame
BulkCandleFetcher = Callable[[str, str, datetime, datetime], pd.DataFrame]


@dataclass
class ScanConfig:
    """Tuning knobs for the similarity scan."""
    similarity_threshold: float = 0.90
    max_matches_per_stock: int = 100
    max_total_matches: int = 2000
    min_volume_ratio: float = 1.2       # skip bars with vol < 1.2x avg (pre-filter)
    min_atr_move: float = 0.5           # skip bars with move < 0.5 ATR (pre-filter)
    use_weighted_similarity: bool = True
    parallel_workers: int = 4


class SimilarityScanner:
    """
    Scans the FNO universe for historical events matching a fingerprint.

    Usage:
        scanner = SimilarityScanner(fetcher=my_bulk_fetcher)
        matches = await scanner.scan(query_fingerprint, config)
    """

    def __init__(self, fetcher: Optional[BulkCandleFetcher] = None):
        self._fetcher = fetcher
        self._extractor = EventExtractor()  # reuse fingerprint computation

    def set_fetcher(self, fetcher: BulkCandleFetcher) -> None:
        self._fetcher = fetcher

    async def scan(
        self,
        query_fp: PatternFingerprint,
        query: PatternQuery,
        symbols: List[str],
        config: Optional[ScanConfig] = None,
    ) -> List[MatchedEvent]:
        """
        Scan all symbols for events matching the query fingerprint.

        Returns matches sorted by similarity (descending).
        """
        if not self._fetcher:
            raise RuntimeError("No candle fetcher configured — call set_fetcher()")

        cfg = config or ScanConfig(similarity_threshold=query.similarity_threshold)
        all_matches: List[MatchedEvent] = []

        # Date range for historical scan
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365 * query.lookback_years)

        logger.info(
            "Scanning %d symbols, %s to %s, threshold=%.2f",
            len(symbols), start_date.date(), end_date.date(), cfg.similarity_threshold,
        )

        for symbol in symbols:
            try:
                matches = await self._scan_single_stock(
                    symbol, query.timeframe, start_date, end_date,
                    query_fp, cfg,
                )
                all_matches.extend(matches)

                if len(all_matches) >= cfg.max_total_matches:
                    logger.info("Hit max_total_matches (%d), stopping scan", cfg.max_total_matches)
                    break

            except Exception as e:
                logger.warning("Scan failed for %s: %s", symbol, e)
                continue

        # Sort by similarity descending
        all_matches.sort(key=lambda m: m.similarity, reverse=True)
        return all_matches[:cfg.max_total_matches]

    async def _scan_single_stock(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        query_fp: PatternFingerprint,
        cfg: ScanConfig,
    ) -> List[MatchedEvent]:
        """Scan a single stock's history for matching events."""
        df = self._fetcher(symbol, timeframe, start, end)
        if df is None or len(df) < EventExtractor.MIN_BARS:
            return []

        matches: List[MatchedEvent] = []

        # Pre-compute indicators for the whole series (amortise cost)
        atr = self._safe_indicator(lambda: IndicatorEngine.atr(df, period=14), len(df))
        vol_sma = self._safe_indicator(lambda: IndicatorEngine.sma(df["volume"], 20), len(df))

        # Scan every bar (skip first 50 for warm-up)
        for i in range(EventExtractor.MIN_BARS, len(df)):
            # ── Pre-filter: skip low-activity bars ──
            if not self._passes_prefilter(df, i, atr, vol_sma, cfg):
                continue

            # Compute fingerprint for this bar
            try:
                fp = self._extractor.compute_fingerprint(df, i, symbol, timeframe)
            except Exception:
                continue

            # Compute similarity
            if cfg.use_weighted_similarity:
                sim = query_fp.weighted_similarity(fp)
            else:
                sim = query_fp.cosine_similarity(fp)

            if sim >= cfg.similarity_threshold:
                ts = None
                time_str = None
                if "datetime" in df.columns:
                    ts = pd.Timestamp(df["datetime"].iloc[i]).to_pydatetime()
                    time_str = ts.strftime("%H:%M")

                match = MatchedEvent(
                    fingerprint=fp,
                    similarity=sim,
                    symbol=symbol,
                    date=ts,
                    time_of_day=time_str,
                    bar_index=i,
                )
                matches.append(match)

                if len(matches) >= cfg.max_matches_per_stock:
                    break

        return matches

    def _passes_prefilter(
        self, df: pd.DataFrame, idx: int,
        atr: pd.Series, vol_sma: pd.Series, cfg: ScanConfig,
    ) -> bool:
        """Quick check to skip obviously non-matching bars."""
        vol = df["volume"].iloc[idx]
        avg_vol = vol_sma.iloc[idx] if idx < len(vol_sma) else vol
        if avg_vol > 0 and vol / avg_vol < cfg.min_volume_ratio:
            return False

        move = abs(df["close"].iloc[idx] - df["close"].iloc[idx - 1])
        atr_val = atr.iloc[idx] if idx < len(atr) else move
        if atr_val > 0 and move / atr_val < cfg.min_atr_move:
            return False

        return True

    def _safe_indicator(self, fn, length: int) -> pd.Series:
        """Compute indicator, return zeros on failure."""
        try:
            return fn()
        except Exception:
            return pd.Series(np.zeros(length))


# Import here to avoid circular import at module level
from strategy_engine.indicators import IndicatorEngine
