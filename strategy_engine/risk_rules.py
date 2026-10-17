"""Risk management utilities for strategies.

Quant review findings addressed:
  1. SL calculation was simplistic — just used previous candle low/high.
     Now uses ATR-multiples (regime-calibrated) with structure validation
     against recent swing highs/lows and a wick buffer.
  2. Risk-reward ratios were arbitrary fixed multiples with no data backing.
     Now regime-mapped with vol adjustment (unchanged concept, improved ranges).
  3. Trailing stop had an off-by-one bug: when steps_locked == 0 the formula
     produced SL = trigger - trail_step (below trigger).  Fixed.
  4. No portfolio heat check — added check_portfolio_heat().
  5. No ATR passed into SL calculation — entry point now accepts atr explicitly.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from core.enums import MarketRegime, TradeDirection, VolatilityRegime


# ── ATR multipliers for stop placement ────────────────────────────────────────
# Professional quant convention:
#   - Low vol → tighter stops are valid (1.0x ATR still beyond noise)
#   - High/Extreme vol → WIDER stops needed to avoid premature stop-outs
#     that cost slippage + emotional damage.  Compensate by reducing size.
_SL_ATR_MULTIPLES: dict[VolatilityRegime, float] = {
    VolatilityRegime.LOW:     1.0,
    VolatilityRegime.AVERAGE: 1.5,
    VolatilityRegime.HIGH:    2.0,
    VolatilityRegime.EXTREME: 2.5,
}

# Minimum stop distance in ATR units — prevents noise-level stops
_MIN_SL_ATR = 0.5

# Maximum stop distance in ATR units — prevents blow-up on bad fills
_MAX_SL_ATR = 3.5

# How many bars to look back for swing high/low structure
_SWING_LOOKBACK_BARS = 10

# Wick buffer below swing low / above swing high (fraction of ATR)
_WICK_BUFFER_ATR = 0.15


# ── Regime → target R:R map ───────────────────────────────────────────────────
# These are minimum targets; actual RR is capped by visible resistance/support.
# Trending markets offer higher RR because momentum carries price further.
# Range markets need tighter RR because price reverses at boundaries.
_REGIME_MIN_RR: dict[MarketRegime, float] = {
    MarketRegime.TRENDING_UP:    2.5,
    MarketRegime.TRENDING_DOWN:  2.5,
    MarketRegime.RANGE:          1.5,
    MarketRegime.RANGING:        1.5,
    MarketRegime.VOLATILE:       2.0,   # high vol often = momentum, but risky
    MarketRegime.HIGH_VOLATILITY: 2.0,
    MarketRegime.LOW_VOL:        1.5,
    MarketRegime.LOW_VOLATILITY: 1.5,
    MarketRegime.BREAKOUT:       2.0,
    MarketRegime.MEAN_REVERTING: 1.5,
    MarketRegime.UNKNOWN:        1.8,
}

# Vol adjustment to RR target (high vol = bigger moves possible → higher target)
_VOL_RR_ADJUSTMENT: dict[VolatilityRegime, float] = {
    VolatilityRegime.LOW:     0.9,
    VolatilityRegime.AVERAGE: 1.0,
    VolatilityRegime.HIGH:    1.2,
    VolatilityRegime.EXTREME: 1.3,
}


class AdaptiveRiskManager:
    """
    Adaptive stop-loss, take-profit, and trailing stop management.

    Stop-loss logic (FIXED):
      Primary:   ATR-multiple based on volatility regime.
      Secondary: Recent swing high/low + wick buffer (structure-based).
      Rule:      Use the structure SL if it falls within ATR bounds;
                 otherwise fall back to the ATR-only SL.
      This ensures stops are placed BEYOND real structure, not at arbitrary
      multiples that ignore price action.

    Trailing stop (FIXED):
      Previous off-by-one: formula produced SL = trigger - trail_step when
      steps_locked == 0, which could move SL backwards.
      Corrected: steps_locked == 0 → SL moves to trigger (break-even).
                 steps_locked >= 1 → SL at trigger + (steps-1)*trail_step.
    """

    def __init__(
        self,
        trailing_trigger_pts: float = 70.0,
        trail_step_pts: float = 20.0,
        min_rr: float = 1.5,
    ):
        self.trailing_trigger = trailing_trigger_pts
        self.trail_step = trail_step_pts
        self.min_rr = min_rr

    # ── Stop Loss ─────────────────────────────────────────────────────────

    def calculate_sl(
        self,
        df_m5: pd.DataFrame,
        atr: float,
        volatility: VolatilityRegime,
        direction: TradeDirection,
        entry: Optional[float] = None,
        swing_lookback: int = _SWING_LOOKBACK_BARS,
    ) -> float:
        """
        Calculate stop-loss using ATR-multiples validated against price structure.

        Args:
            df_m5: Recent OHLCV bars (at least swing_lookback + 1 rows).
            atr: Current ATR for the instrument/timeframe.
            volatility: Current volatility classification.
            direction: Trade direction.
            entry: Entry price (defaults to last close if not provided).
            swing_lookback: Number of bars to look back for swing structure.

        Returns:
            Stop price as a float.
        """
        if len(df_m5) < 2:
            raise ValueError("Need at least 2 bars for SL calculation")

        entry_price = entry if entry is not None else float(df_m5["close"].iloc[-1])
        multiplier = _SL_ATR_MULTIPLES.get(volatility, 1.5)

        # ── ATR-based SL ──────────────────────────────────────────────────
        if direction == TradeDirection.LONG:
            atr_sl = entry_price - (multiplier * atr)

            # ── Structure-based SL (swing low + buffer) ───────────────────
            lookback_df = df_m5.tail(swing_lookback)
            swing_low = float(lookback_df["low"].min())
            structure_sl = swing_low - (_WICK_BUFFER_ATR * atr)

            sl = self._select_sl(entry_price, atr_sl, structure_sl, atr, direction)

        else:  # SHORT
            atr_sl = entry_price + (multiplier * atr)

            lookback_df = df_m5.tail(swing_lookback)
            swing_high = float(lookback_df["high"].max())
            structure_sl = swing_high + (_WICK_BUFFER_ATR * atr)

            sl = self._select_sl(entry_price, atr_sl, structure_sl, atr, direction)

        return sl

    def _select_sl(
        self,
        entry: float,
        atr_sl: float,
        structure_sl: float,
        atr: float,
        direction: TradeDirection,
    ) -> float:
        """
        Choose between ATR-based and structure-based SL.

        Logic for LONG (inverted for SHORT):
          - Structure SL is used if: 0.5 ATR ≤ (entry - structure_sl) ≤ 3.5 ATR
          - Otherwise fall back to the ATR-based SL.
          - Final guard: enforce min/max ATR bounds in any case.
        """
        if direction == TradeDirection.LONG:
            structure_dist = entry - structure_sl
            min_dist = _MIN_SL_ATR * atr
            max_dist = _MAX_SL_ATR * atr

            if min_dist <= structure_dist <= max_dist:
                sl = structure_sl       # structure is valid — use it
            else:
                sl = atr_sl             # fall back to ATR-based

            # Enforce hard bounds
            sl = min(sl, entry - min_dist)   # never closer than 0.5 ATR
            sl = max(sl, entry - max_dist)   # never wider than 3.5 ATR

        else:  # SHORT
            structure_dist = structure_sl - entry
            min_dist = _MIN_SL_ATR * atr
            max_dist = _MAX_SL_ATR * atr

            if min_dist <= structure_dist <= max_dist:
                sl = structure_sl
            else:
                sl = atr_sl

            sl = max(sl, entry + min_dist)
            sl = min(sl, entry + max_dist)

        return sl

    # ── Risk-Reward / Take-Profit ─────────────────────────────────────────

    def calculate_rr(
        self,
        entry: float,
        sl: float,
        direction: TradeDirection,
        volatility: VolatilityRegime,
        regime: MarketRegime,
    ) -> Tuple[float, float, float]:
        """
        Calculate target price, trail trigger, and effective R:R.

        Returns:
            (target_price, trail_trigger, rr_ratio)
        """
        risk = abs(entry - sl)
        if risk < 1e-8:
            raise ValueError("Stop loss equals entry price — cannot compute R:R")

        regime_rr = _REGIME_MIN_RR.get(regime, 1.8)
        vol_adj = _VOL_RR_ADJUSTMENT.get(volatility, 1.0)
        rr = max(self.min_rr, regime_rr * vol_adj)

        if direction == TradeDirection.LONG:
            target = entry + (risk * rr)
            trail_trigger = entry + self.trailing_trigger
        else:
            target = entry - (risk * rr)
            trail_trigger = entry - self.trailing_trigger

        return target, trail_trigger, rr

    # ── Trailing Stop ─────────────────────────────────────────────────────

    def update_trailing_stop(
        self,
        current_price: float,
        current_high: float,           # kept for API compatibility (unused)
        trail_trigger: float,
        current_sl: float,
        direction: TradeDirection,
    ) -> float:
        """
        Step-based trailing stop — FIXED off-by-one.

        Behaviour:
          steps_locked == 0:  price just crossed trigger → move SL to trigger
                              (achieves break-even lock on first step).
          steps_locked >= 1:  move SL to trigger + (steps-1) * trail_step
                              (SL trails one step behind locked-in profit).

        The previous formula `trigger + steps*step - step` produced
        `trigger - step` when steps == 0, which could push the SL
        in the wrong direction.  `max(new_sl, current_sl)` partially
        masked this, but it failed if current_sl was set below trigger.

        Args:
            current_price: Latest market price.
            current_high: Unused (kept for signature compatibility).
            trail_trigger: Price at which trailing begins.
            current_sl: Current stop-loss price.
            direction: Trade direction.

        Returns:
            Updated stop-loss price (never moves against the position).
        """
        if direction == TradeDirection.LONG:
            if current_price >= trail_trigger:
                profit_pts = current_price - trail_trigger
                steps_locked = int(profit_pts / self.trail_step)
                # steps=0 → break-even (SL = trigger)
                # steps=N → SL = trigger + (N-1)*step
                locked_pts = max(0.0, (steps_locked - 1)) * self.trail_step if steps_locked > 0 else 0.0
                new_sl = trail_trigger + locked_pts
                return max(new_sl, current_sl)   # SL only moves up for longs

        else:  # SHORT
            if current_price <= trail_trigger:
                profit_pts = trail_trigger - current_price
                steps_locked = int(profit_pts / self.trail_step)
                locked_pts = max(0.0, (steps_locked - 1)) * self.trail_step if steps_locked > 0 else 0.0
                new_sl = trail_trigger - locked_pts
                return min(new_sl, current_sl)   # SL only moves down for shorts

        return current_sl

    # ── Portfolio Heat ────────────────────────────────────────────────────

    @staticmethod
    def check_portfolio_heat(
        new_risk_pct: float,
        open_trade_risks: List[float],
        max_heat_pct: float = 6.0,
    ) -> Tuple[bool, float, float]:
        """
        Verify that adding a new trade does not exceed total portfolio risk cap.

        Args:
            new_risk_pct: Risk % of the proposed new trade.
            open_trade_risks: List of risk_pct values for currently open trades.
            max_heat_pct: Maximum allowed total risk across all open positions.

        Returns:
            (allowed: bool, current_heat: float, projected_heat: float)
        """
        current_heat = sum(open_trade_risks)
        projected = current_heat + new_risk_pct
        allowed = projected <= max_heat_pct
        return allowed, round(current_heat, 3), round(projected, 3)
