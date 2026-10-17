"""Breakout validation utilities."""

from __future__ import annotations

import pandas as pd
from typing import Tuple


class BreakoutValidator:
    """Validates breakout candle quality:
    - Higher volume than avg of last 4-5 candles
    - Bigger body than avg of last 4-5 candles
    - More body above resistance than below
    """

    def __init__(
        self,
        volume_multiplier: float = 1.2,
        body_multiplier: float = 1.2,
        body_lookback: int = 5,
    ):
        self.volume_mult = volume_multiplier
        self.body_mult = body_multiplier
        self.body_lookback = body_lookback

    def validate(
        self,
        df_m5: pd.DataFrame,
        resistance: float,
    ) -> Tuple[bool, float, float, str]:
        """
        Returns (is_valid, volume_ratio, body_ratio, rejection_reason)
        """
        candle = df_m5.iloc[-1]
        prev_candles = df_m5.iloc[-(self.body_lookback + 1) : -1]

        # --- Volume Check ---
        avg_vol = prev_candles["volume"].mean()
        vol_ratio = candle["volume"] / avg_vol if avg_vol > 0 else 0
        if vol_ratio < self.volume_mult:
            return False, vol_ratio, 0.0, f"Low volume ratio: {vol_ratio:.2f}"

        # --- Body Size Check ---
        candle_body = abs(candle["close"] - candle["open"])
        avg_body = (
            prev_candles["close"] - prev_candles["open"]
        ).abs().mean()
        body_ratio = candle_body / avg_body if avg_body > 0 else 0

        if body_ratio < self.body_mult:
            return False, vol_ratio, body_ratio, f"Small body ratio: {body_ratio:.2f}"

        # --- Body Above Resistance Check ---
        upper_ref = max(candle["open"], candle["close"])
        lower_ref = min(candle["open"], candle["close"])

        body_above = max(0.0, upper_ref - resistance)
        body_below = max(0.0, resistance - lower_ref)

        if body_above <= body_below:
            return (
                False,
                vol_ratio,
                body_ratio,
                f"Body mostly below resistance. Above={body_above:.2f}, Below={body_below:.2f}",
            )

        return True, vol_ratio, body_ratio, "VALID"
