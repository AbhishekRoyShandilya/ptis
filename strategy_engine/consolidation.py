"""Detect consolidation zones on intraday price data."""

from __future__ import annotations

import pandas as pd
from typing import Tuple


class ConsolidationDetector:
    """
    Detects price consolidation zones on 5‑min chart.
    Consolidation = 25 min to 3 hrs (5 to 36 candles on 5‑min)
    """

    def __init__(
        self,
        min_candles: int = 5,  # 25 min
        max_candles: int = 36,  # 3 hrs
        range_pct: float = 0.003,  # 0.3% max range = tight consolidation
    ):
        self.min_candles = min_candles
        self.max_candles = max_candles
        self.range_pct = range_pct

    def find_consolidation(self, df_m5: pd.DataFrame) -> Tuple[bool, float, float, int]:
        """
        Scans backward to find consolidation zone.
        Returns (found, resistance_level, support_level, candle_count)
        """
        for lookback in range(self.max_candles, self.min_candles - 1, -1):
            if lookback > len(df_m5) - 1:
                continue

            segment = df_m5.iloc[-(lookback + 1) : -1]  # exclude current candle
            high = segment["high"].max()
            low = segment["low"].min()
            mid = (high + low) / 2

            price_range_pct = (high - low) / mid

            if price_range_pct <= self.range_pct:
                return True, high, low, lookback

        return False, 0.0, 0.0, 0
