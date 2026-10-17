# ske/regime_fingerprint.py
"""
Multi-dimensional market regime fingerprint for the Strategy Knowledge Engine.

Extends the existing single-enum RegimeDetector with a 5-dimensional continuous
fingerprint: index_trend, volatility_level, market_breadth, sector_participation,
fii_dii_flow — each in [0.0, 1.0].
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from core.enums import MarketRegime
from ske.models import RegimeFingerprint

logger = logging.getLogger(__name__)


class RegimeFingerprintComputer:
    """
    Computes a multi-dimensional regime fingerprint from candle data.

    Uses existing indicator computations (ATR, RSI, ADX, SMA) to derive
    continuous regime dimensions instead of discrete enum values.
    """

    def compute(
        self,
        candles: List[Dict[str, Any]],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> RegimeFingerprint:
        """
        Compute regime fingerprint from OHLCV candles + optional external data.

        Args:
            candles: OHLCV bars (oldest first), minimum 50 for meaningful results.
            market_data: Optional dict with external data:
                - advances, declines: breadth numbers
                - fii_net, dii_net: institutional flow in crores
                - sector_advances, sector_total: sector participation
        """
        if len(candles) < 20:
            return RegimeFingerprint()

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]

        ext = market_data or {}

        return RegimeFingerprint(
            index_trend=self._compute_trend(closes),
            volatility_level=self._compute_volatility(highs, lows, closes),
            market_breadth=self._compute_breadth(ext),
            sector_participation=self._compute_sector(ext),
            fii_dii_flow=self._compute_flow(ext),
        )

    def match_score(
        self,
        current: RegimeFingerprint,
        preferred: List[Dict[str, Any]],
    ) -> float:
        """
        Compute similarity of current regime to preferred regime fingerprints.

        Args:
            current: Current market regime fingerprint.
            preferred: List of dicts with fingerprint + confidence:
                [{index_trend: 0.8, volatility_level: 0.3, ..., confidence: 0.9}, ...]

        Returns:
            Weighted cosine similarity in [0.0, 1.0].
        """
        if not preferred:
            return 0.5  # neutral if no preference

        curr_vec = current.to_vector()
        best_score = 0.0

        for pref in preferred:
            confidence = pref.get("confidence", 1.0)
            pref_vec = [
                pref.get("index_trend", 0.5),
                pref.get("volatility_level", 0.5),
                pref.get("market_breadth", 0.5),
                pref.get("sector_participation", 0.5),
                pref.get("fii_dii_flow", 0.5),
            ]

            sim = self._cosine_similarity(curr_vec, pref_vec)
            weighted = sim * confidence
            best_score = max(best_score, weighted)

        return max(0.0, min(1.0, best_score))

    # ──────────────────────────────────────────
    #  Dimension computations
    # ──────────────────────────────────────────

    @staticmethod
    def _compute_trend(closes: List[float]) -> float:
        """
        Map trend to 0–1: 0 = strong bear, 0.5 = neutral, 1 = strong bull.
        Uses SMA crossover + momentum.
        """
        n = len(closes)
        if n < 50:
            # Fallback: just short-term momentum
            if n < 5:
                return 0.5
            mom = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0.0
            return max(0.0, min(1.0, 0.5 + mom * 10))

        sma20 = sum(closes[-20:]) / 20
        sma50 = sum(closes[-50:]) / 50

        if sma50 <= 0:
            return 0.5

        # SMA ratio: >1 bullish, <1 bearish
        ratio = sma20 / sma50
        # Map ratio to 0–1: ratio of 0.95 → ~0.25, 1.0 → 0.5, 1.05 → ~0.75
        trend = 0.5 + (ratio - 1.0) * 10.0
        return max(0.0, min(1.0, trend))

    @staticmethod
    def _compute_volatility(
        highs: List[float], lows: List[float], closes: List[float],
    ) -> float:
        """ATR percentile → 0–1 (0 = low vol, 1 = high vol)."""
        n = len(closes)
        period = 14
        if n < period + 1:
            return 0.5

        trs = []
        for i in range(1, n):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)

        if len(trs) < period:
            return 0.5

        alpha = 1.0 / period
        atr = sum(trs[:period]) / period
        atr_values = [atr]
        for tr in trs[period:]:
            atr = atr * (1 - alpha) + tr * alpha
            atr_values.append(atr)

        current = atr_values[-1]
        lookback = min(252, len(atr_values))
        historical = sorted(atr_values[-lookback:])
        rank = sum(1 for v in historical if v <= current)
        return rank / len(historical)

    @staticmethod
    def _compute_breadth(ext: Dict[str, Any]) -> float:
        """Market breadth: advances / (advances + declines). Placeholder if unavailable."""
        advances = ext.get("advances")
        declines = ext.get("declines")
        if advances is not None and declines is not None:
            total = advances + declines
            if total > 0:
                return advances / total
        return 0.5  # neutral default

    @staticmethod
    def _compute_sector(ext: Dict[str, Any]) -> float:
        """Sector participation rate. Placeholder if unavailable."""
        sector_advances = ext.get("sector_advances")
        sector_total = ext.get("sector_total")
        if sector_advances is not None and sector_total is not None:
            if sector_total > 0:
                return sector_advances / sector_total
        return 0.5

    @staticmethod
    def _compute_flow(ext: Dict[str, Any]) -> float:
        """
        FII/DII flow mapped to 0–1.
        Positive net flow → >0.5, negative → <0.5.
        """
        fii_net = ext.get("fii_net")
        dii_net = ext.get("dii_net")
        if fii_net is not None and dii_net is not None:
            combined = fii_net + dii_net
            # Normalize: assume ±5000 crores is the typical range
            norm = combined / 5000.0
            return max(0.0, min(1.0, 0.5 + norm * 0.5))
        return 0.5

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        return dot / (norm_a * norm_b)
