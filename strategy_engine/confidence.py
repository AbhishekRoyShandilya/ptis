"""Scoring utilities used by strategies for signal confidence."""

from __future__ import annotations

from typing import Tuple
from core.enums import TradeDirection
from core.enums import MarketRegime, VolatilityRegime


class ConfidenceScorer:
    """Scores trade signal quality 0–100"""

    def score(
        self,
        regime_state: any,
        vol_ratio: float,
        body_ratio: float,
        consol_candles: int,
        direction: TradeDirection,
    ) -> float:

        score = 0.0

        # Regime alignment (40 pts)
        if direction == TradeDirection.LONG:
            score += 10 if regime_state.daily_trend_up else 0
            score += 10 if regime_state.weekly_trend_up else 0
            score += 10 if regime_state.h1_supertrend_up else 0
            score += 10 if regime_state.m15_supertrend_up else 0

        # H1 + M15 DMA (20 pts)
        if regime_state.h1_above_50dma:
            score += 10
        if regime_state.m15_above_50dma:
            score += 10

        # Volume strength (20 pts)
        score += min(20, (vol_ratio - 1.0) * 20)

        # Breakout candle body (10 pts)
        score += min(10, (body_ratio - 1.0) * 10)

        # Consolidation quality (10 pts)
        if 5 <= consol_candles <= 36:
            score += 10 * (1 - abs(consol_candles - 18) / 18)

        return round(min(100, max(0, score)), 2)
