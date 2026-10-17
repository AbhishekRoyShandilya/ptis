"""Drawdown analysis — production implementation.

Quant review findings addressed:
  1. File was an empty placeholder — the most critical missing analytics module.
  2. No peak-to-trough tracking at account level.
  3. No Max Adverse Excursion (MAE) per trade.
  4. No Calmar ratio.
  5. No drawdown-triggered risk scaling recommendation.
  6. No recovery time analysis.

This module is intentionally self-contained: it takes raw equity / P&L
data and returns structured analysis without any database dependency,
so it can be called from both the API layer and live-monitoring hooks.

Typical usage:
    analyzer = DrawdownAnalyzer()
    state = analyzer.current_state(equity_series)
    # Feed state into PositionSizer.compute(drawdown_pct=state.current_drawdown_pct)

    report = analyzer.full_report(equity_series, trades_df)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DrawdownState:
    """Current drawdown snapshot — fed into PositionSizer for dynamic sizing."""
    peak_equity: float
    current_equity: float
    current_drawdown_pct: float       # 0.0 = at peak, 0.15 = 15% below peak
    current_drawdown_abs: float       # $ value of drawdown
    max_drawdown_pct: float           # worst peak-to-trough over entire history
    max_drawdown_abs: float           # $ value of max drawdown
    drawdown_start: Optional[datetime]
    bars_in_drawdown: int             # duration of CURRENT drawdown period
    recovery_required_pct: float      # % gain needed just to get back to peak
    in_drawdown: bool

    @property
    def size_multiplier(self) -> float:
        """
        Linear drawdown scaling used by PositionSizer as a quick override.
        Returns 1.0 at peak, scales to 0 at 20% drawdown.
        Not a substitute for PositionSizer.compute() — use that for full logic.
        """
        dd = self.current_drawdown_pct
        if dd < 0.05:
            return 1.00
        elif dd < 0.10:
            return 0.75
        elif dd < 0.15:
            return 0.50
        elif dd < 0.20:
            return 0.25
        return 0.00


@dataclass
class DrawdownPeriod:
    """A single contiguous drawdown event."""
    start_idx: int
    end_idx: Optional[int]            # None if still in drawdown
    peak_equity: float
    trough_equity: float
    drawdown_pct: float
    drawdown_abs: float
    duration_bars: int
    recovery_bars: Optional[int]      # bars to recover to new peak (None if unrecovered)


@dataclass
class MAEAnalysis:
    """Max Adverse Excursion analysis across a trade set."""
    avg_mae_r: float                  # average MAE in R-multiples
    avg_mfe_r: float                  # average Max Favourable Excursion in R
    mae_at_stop: float                # % of trades where MAE == stop distance (hit stop)
    efficiency: float                 # MFE / (MFE + MAE) — how efficiently we captured moves
    stop_too_tight_pct: float         # % trades stopped before favourable move (stop < MAE optimal)
    stop_too_wide_pct: float          # % trades where stop was wider than necessary


# ──────────────────────────────────────────────────────────────────────────────
#  DRAWDOWN ANALYZER
# ──────────────────────────────────────────────────────────────────────────────

class DrawdownAnalyzer:
    """
    Comprehensive drawdown and adverse excursion analysis.

    Accepts:
      - equity_series: pd.Series of account balance values over time
        (index can be datetime or integer)
      - trades_df: pd.DataFrame with columns:
          entry_price, exit_price, stop_loss, pnl, direction, risk_amount
          Optional: low_during_trade, high_during_trade (for MAE/MFE)
    """

    # ── Equity curve analysis ─────────────────────────────────────────────

    def compute_series(self, equity: pd.Series) -> pd.DataFrame:
        """
        Compute rolling drawdown at every point in the equity curve.

        Returns a DataFrame with columns:
          equity, peak, drawdown_abs, drawdown_pct, in_drawdown
        """
        if equity.empty:
            return pd.DataFrame()

        df = pd.DataFrame({"equity": equity})
        df["peak"] = equity.cummax()
        df["drawdown_abs"] = df["equity"] - df["peak"]
        df["drawdown_pct"] = df["drawdown_abs"] / df["peak"].replace(0, np.nan)
        df["in_drawdown"] = df["drawdown_pct"] < 0
        return df

    def current_state(self, equity: pd.Series) -> DrawdownState:
        """
        Extract the current drawdown state from an equity series.

        Pass into PositionSizer.compute(drawdown_pct=state.current_drawdown_pct)
        to get automatically scaled position sizes.
        """
        if equity.empty:
            return DrawdownState(
                peak_equity=0, current_equity=0,
                current_drawdown_pct=0, current_drawdown_abs=0,
                max_drawdown_pct=0, max_drawdown_abs=0,
                drawdown_start=None, bars_in_drawdown=0,
                recovery_required_pct=0, in_drawdown=False,
            )

        df = self.compute_series(equity)
        peak = df["peak"].iloc[-1]
        current = df["equity"].iloc[-1]
        current_dd_abs = df["drawdown_abs"].iloc[-1]
        current_dd_pct = df["drawdown_pct"].iloc[-1]

        max_dd_pct = df["drawdown_pct"].min()          # most negative value
        max_dd_abs = df["drawdown_abs"].min()

        # Duration of current drawdown streak
        bars_in_dd = 0
        if current_dd_pct < 0:
            # Walk backwards to find when the current drawdown started
            for i in range(len(df) - 1, -1, -1):
                if df["drawdown_pct"].iloc[i] < 0:
                    bars_in_dd += 1
                else:
                    break

        # Recovery required: if account is at 85% of peak, need +17.6% to recover
        recovery_pct = (peak / current - 1) if current > 0 and current < peak else 0.0

        # Drawdown start timestamp (if index is datetime)
        dd_start = None
        if bars_in_dd > 0 and isinstance(equity.index, pd.DatetimeIndex):
            start_pos = len(df) - bars_in_dd
            dd_start = equity.index[start_pos]

        return DrawdownState(
            peak_equity=float(peak),
            current_equity=float(current),
            current_drawdown_pct=float(abs(current_dd_pct)),  # return as positive
            current_drawdown_abs=float(abs(current_dd_abs)),
            max_drawdown_pct=float(abs(max_dd_pct)),
            max_drawdown_abs=float(abs(max_dd_abs)),
            drawdown_start=dd_start,
            bars_in_drawdown=bars_in_dd,
            recovery_required_pct=float(recovery_pct),
            in_drawdown=current_dd_pct < 0,
        )

    def max_drawdown(self, equity: pd.Series) -> Tuple[float, float]:
        """Return (max_drawdown_pct, max_drawdown_abs) — the all-time worst."""
        if equity.empty:
            return 0.0, 0.0
        df = self.compute_series(equity)
        return float(abs(df["drawdown_pct"].min())), float(abs(df["drawdown_abs"].min()))

    def drawdown_periods(self, equity: pd.Series) -> List[DrawdownPeriod]:
        """
        Identify all distinct drawdown events (peak → trough → recovery).

        Returns a list of DrawdownPeriod objects sorted by drawdown_pct desc.
        """
        df = self.compute_series(equity)
        periods: List[DrawdownPeriod] = []

        i = 0
        n = len(df)
        while i < n:
            if df["in_drawdown"].iloc[i]:
                # Found start of a drawdown
                start_i = i
                peak_eq = df["peak"].iloc[i]
                trough_eq = df["equity"].iloc[i]
                j = i
                while j < n and df["in_drawdown"].iloc[j]:
                    trough_eq = min(trough_eq, df["equity"].iloc[j])
                    j += 1
                end_i = j - 1

                # Find recovery (next time equity exceeds the peak before this DD)
                recovery_bars = None
                k = j
                while k < n:
                    if df["equity"].iloc[k] >= peak_eq:
                        recovery_bars = k - end_i
                        break
                    k += 1

                dd_abs = peak_eq - trough_eq
                dd_pct = dd_abs / peak_eq if peak_eq > 0 else 0

                periods.append(DrawdownPeriod(
                    start_idx=start_i,
                    end_idx=end_i if j < n else None,
                    peak_equity=float(peak_eq),
                    trough_equity=float(trough_eq),
                    drawdown_pct=float(dd_pct),
                    drawdown_abs=float(dd_abs),
                    duration_bars=end_i - start_i + 1,
                    recovery_bars=recovery_bars,
                ))
                i = j
            else:
                i += 1

        periods.sort(key=lambda p: p.drawdown_pct, reverse=True)
        return periods

    # ── Calmar ratio ──────────────────────────────────────────────────────

    def calmar_ratio(
        self,
        equity: pd.Series,
        annual_return: Optional[float] = None,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calmar = annualised_return / max_drawdown_pct.

        If annual_return is not provided, it is estimated from the equity curve
        assuming periods_per_year trading days/bars in the series.

        A Calmar > 2 is considered good; < 1 is poor.
        """
        max_dd, _ = self.max_drawdown(equity)
        if max_dd < 1e-8:
            return float("inf")

        if annual_return is None and len(equity) > 1:
            total_return = equity.iloc[-1] / equity.iloc[0] - 1
            n_years = len(equity) / periods_per_year
            annual_return = (1 + total_return) ** (1 / max(n_years, 1e-8)) - 1

        if annual_return is None or annual_return <= 0:
            return 0.0

        return float(annual_return / max_dd)

    # ── MAE / MFE analysis ────────────────────────────────────────────────

    def mae_analysis(self, trades_df: pd.DataFrame) -> MAEAnalysis:
        """
        Max Adverse Excursion / Max Favourable Excursion analysis.

        Required columns in trades_df:
          entry_price, stop_loss, direction ('long'/'short'), risk_amount
        Optional (for proper MAE/MFE — use 0 if unavailable):
          low_during_trade, high_during_trade, exit_price

        Without intraday data, MAE is estimated as: stop distance (i.e. the
        trade was stopped out) or 0 (trade was not stopped out, winner).
        """
        required = {"entry_price", "stop_loss", "direction", "risk_amount"}
        if trades_df.empty or not required.issubset(trades_df.columns):
            return MAEAnalysis(
                avg_mae_r=0, avg_mfe_r=0, mae_at_stop=0,
                efficiency=0, stop_too_tight_pct=0, stop_too_wide_pct=0,
            )

        maes, mfes = [], []
        stop_hits, too_tight, too_wide = 0, 0, 0

        for _, row in trades_df.iterrows():
            entry = row["entry_price"]
            stop = row["stop_loss"]
            direction = str(row.get("direction", "long")).lower()
            risk = abs(entry - stop)
            if risk <= 0:
                continue

            # MAE in price terms
            if direction == "long":
                low = row.get("low_during_trade", stop)          # worst case = hit stop
                high = row.get("high_during_trade", row.get("exit_price", entry))
                mae = max(0, entry - low)
                mfe = max(0, high - entry)
            else:
                high = row.get("high_during_trade", stop)
                low = row.get("low_during_trade", row.get("exit_price", entry))
                mae = max(0, high - entry)
                mfe = max(0, entry - low)

            mae_r = mae / risk
            mfe_r = mfe / risk
            maes.append(mae_r)
            mfes.append(mfe_r)

            if abs(mae - risk) < 0.01 * risk:   # MAE ≈ stop distance → likely stopped out
                stop_hits += 1
            if mae_r > 0.8 and mfe_r > 1.5:     # got stopped but price later went our way
                too_tight += 1
            if mae_r < 0.3 and mfe_r < 0.5:     # stop was wide but didn't need it
                too_wide += 1

        n = len(maes)
        if n == 0:
            return MAEAnalysis(0, 0, 0, 0, 0, 0)

        avg_mae = sum(maes) / n
        avg_mfe = sum(mfes) / n
        eff = avg_mfe / (avg_mfe + avg_mae) if (avg_mfe + avg_mae) > 0 else 0

        return MAEAnalysis(
            avg_mae_r=round(avg_mae, 3),
            avg_mfe_r=round(avg_mfe, 3),
            mae_at_stop=round(stop_hits / n, 3),
            efficiency=round(eff, 3),
            stop_too_tight_pct=round(too_tight / n, 3),
            stop_too_wide_pct=round(too_wide / n, 3),
        )

    # ── Risk reduction recommendation ─────────────────────────────────────

    def should_reduce_risk(
        self,
        state: DrawdownState,
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.10,
        halt_threshold: float = 0.20,
    ) -> Tuple[bool, float, str]:
        """
        Returns (should_reduce: bool, recommended_multiplier: float, reason: str).

        Used by live trading hooks to enforce drawdown-based size reduction.
        """
        dd = state.current_drawdown_pct
        if dd >= halt_threshold:
            return True, 0.0, (
                f"HALT: drawdown {dd*100:.1f}% exceeds halt threshold {halt_threshold*100:.0f}%. "
                "Stop trading until equity recovers."
            )
        elif dd >= critical_threshold:
            mult = 0.25 + (halt_threshold - dd) / (halt_threshold - critical_threshold) * 0.25
            return True, round(mult, 2), (
                f"CRITICAL: drawdown {dd*100:.1f}%: reduce size to {mult*100:.0f}%"
            )
        elif dd >= warning_threshold:
            mult = 0.75 - (dd - warning_threshold) / (critical_threshold - warning_threshold) * 0.25
            return True, round(mult, 2), (
                f"WARNING: drawdown {dd*100:.1f}%: reduce size to {mult*100:.0f}%"
            )
        return False, 1.0, "No drawdown concern"

    # ── Full report ────────────────────────────────────────────────────────

    def full_report(
        self,
        equity: pd.Series,
        trades_df: Optional[pd.DataFrame] = None,
        periods_per_year: int = 252,
    ) -> Dict:
        """
        Produce a complete drawdown + MAE report dict suitable for the API.
        """
        state = self.current_state(equity)
        max_dd_pct, max_dd_abs = self.max_drawdown(equity)
        calmar = self.calmar_ratio(equity, periods_per_year=periods_per_year)
        periods = self.drawdown_periods(equity)
        _, mult, reason = self.should_reduce_risk(state)

        top_periods = [
            {
                "drawdown_pct": round(p.drawdown_pct * 100, 2),
                "drawdown_abs": round(p.drawdown_abs, 2),
                "duration_bars": p.duration_bars,
                "recovery_bars": p.recovery_bars,
            }
            for p in periods[:5]  # top 5 worst
        ]

        mae = self.mae_analysis(trades_df) if trades_df is not None else None

        report: Dict = {
            "current": {
                "drawdown_pct": round(state.current_drawdown_pct * 100, 2),
                "drawdown_abs": round(state.current_drawdown_abs, 2),
                "bars_in_drawdown": state.bars_in_drawdown,
                "recovery_required_pct": round(state.recovery_required_pct * 100, 2),
                "peak_equity": round(state.peak_equity, 2),
                "current_equity": round(state.current_equity, 2),
            },
            "historical": {
                "max_drawdown_pct": round(max_dd_pct * 100, 2),
                "max_drawdown_abs": round(max_dd_abs, 2),
                "calmar_ratio": round(calmar, 3) if math.isfinite(calmar) else None,
                "total_drawdown_periods": len(periods),
                "top_5_drawdowns": top_periods,
            },
            "risk_scaling": {
                "recommended_multiplier": mult,
                "reason": reason,
            },
        }

        if mae:
            report["mae_analysis"] = {
                "avg_mae_r": mae.avg_mae_r,
                "avg_mfe_r": mae.avg_mfe_r,
                "stop_hit_rate": mae.mae_at_stop,
                "trade_efficiency": mae.efficiency,
                "stop_too_tight_pct": mae.stop_too_tight_pct,
                "stop_too_wide_pct": mae.stop_too_wide_pct,
            }

        return report


# ──────────────────────────────────────────────────────────────────────────────
#  EQUITY CURVE BUILDER (from trade P&L list)
# ──────────────────────────────────────────────────────────────────────────────

def build_equity_curve(
    pnl_list: List[float],
    starting_balance: float = 100_000.0,
) -> pd.Series:
    """
    Construct an equity curve from a list of trade P&L values.

    Args:
        pnl_list: Ordered list of trade P&L in $.
        starting_balance: Account starting balance.

    Returns:
        pd.Series of cumulative equity values (length = len(pnl_list) + 1,
        including the starting value as index 0).
    """
    cumulative = [starting_balance]
    balance = starting_balance
    for pnl in pnl_list:
        balance += pnl
        cumulative.append(balance)
    return pd.Series(cumulative, dtype=float)
