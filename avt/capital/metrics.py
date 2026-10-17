"""
avt.capital.metrics — HONEST account metrics from an equity curve. R-per-trade is never reported as "return";
the account curve only exists once the operator (sizing + compounding + cost) is specified. CAGR is reported
next to max drawdown and MAR/Calmar so a single benign-window number can't masquerade as an edge — and it is
flagged untrustworthy the moment the curve goes non-positive (the equity-scaling fantasy shows up here as a
ruined/`nan` curve rather than a fake giant number).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class AccountMetrics:
    trades: int
    years: float
    win_pct: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    max_dd_pct: float
    mar: float                    # CAGR / |maxDD|  (Calmar) — return per unit of pain
    avg_utilization_pct: float    # avg fraction of equity deployed as margin while in a position
    ruined: bool                  # equity hit <= 0 at any point (or ended non-positive) → CAGR untrustworthy
    note: str = ""

    def row(self) -> str:
        cg = "RUIN" if self.ruined else f"{self.cagr_pct:+.0f}%"
        return (f"{self.trades:>5}  win {self.win_pct:>3.0f}%  "
                f"CAGR {cg:>7}  maxDD {self.max_dd_pct:>5.0f}%  MAR {self.mar:>5.2f}  "
                f"util {self.avg_utilization_pct:>3.0f}%  final Rs{self.final_equity:>14,.0f}")


def compute(curve: List[Tuple[pd.Timestamp, float]], capital: float, wins: int, trades: int,
            utilizations: List[float]) -> AccountMetrics:
    """curve = [(ts, equity_after_trade), ...] starting AFTER the first trade; capital = starting equity."""
    if not curve or trades == 0:
        return AccountMetrics(0, 0, 0, capital, 0, 0, 0, 0, 0, False, "no trades")
    ts = [c[0] for c in curve]
    eq = np.array([capital] + [c[1] for c in curve], dtype=float)
    days = max((pd.Timestamp(ts[-1]) - pd.Timestamp(ts[0])).days, 1)
    yrs = days / 365.0

    ruined = bool(np.any(eq <= 0) or eq[-1] <= 0 or not np.all(np.isfinite(eq)))
    final = float(eq[-1]) if np.isfinite(eq[-1]) else float("nan")

    # drawdown on the finite prefix
    peak = np.maximum.accumulate(eq)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.min((eq - peak) / peak) if np.all(peak > 0) else -1.0
    dd_pct = float(dd * 100) if np.isfinite(dd) else -100.0

    if ruined or final <= 0:
        cagr = float("nan")
    else:
        cagr = ((final / capital) ** (1.0 / yrs) - 1.0) * 100.0
    mar = (cagr / abs(dd_pct)) if (not ruined and dd_pct < 0 and np.isfinite(cagr)) else 0.0

    return AccountMetrics(
        trades=trades, years=yrs, win_pct=100.0 * wins / trades,
        final_equity=final, total_return_pct=(final / capital - 1.0) * 100.0 if np.isfinite(final) else float("nan"),
        cagr_pct=cagr, max_dd_pct=dd_pct, mar=mar,
        avg_utilization_pct=float(np.mean(utilizations) * 100) if utilizations else 0.0,
        ruined=ruined, note=("equity went non-positive — CAGR untrustworthy" if ruined else ""),
    )
