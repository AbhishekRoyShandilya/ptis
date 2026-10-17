"""
avt.execution.guards — pre-trade EXECUTION SANITY checks (master §2p). Additive, live-safe: they only ever
REFUSE a trade that is already invalid at the actual execution price — never change which good trade is taken or
how it exits. The frozen detection/strategy and the raw golden are untouched.
"""
from __future__ import annotations

from typing import Tuple


def entry_still_valid(direction: str, exec_price: float, stop: float,
                      min_risk_frac: float = 0.0) -> Tuple[bool, str]:
    """Is the trade still valid at the moment/price of EXECUTION?

    The signal's stop is computed at DETECTION (a 5m close); the fill happens a little later at the actual live
    price. If, by then, price is already AT or THROUGH the stop, the thesis is already invalidated — booking it is
    an instant loss (the "born through the stop" class: 24% of option trades in backtest, largely a two-feed data
    artifact; in LIVE one feed makes this a genuine 'already dead' check). Also rejects a degenerate near-zero
    risk (stop essentially at price). Returns (ok, reason)."""
    if exec_price is None or exec_price <= 0 or stop is None or stop <= 0:
        return False, "missing/invalid price or stop"
    long = str(direction).lower() == "long"
    if long and exec_price <= stop:
        return False, f"born through stop: exec {exec_price:.2f} <= stop {stop:.2f}"
    if (not long) and exec_price >= stop:
        return False, f"born through stop: exec {exec_price:.2f} >= stop {stop:.2f}"
    if min_risk_frac > 0:
        risk_frac = abs(exec_price - stop) / exec_price
        if risk_frac < min_risk_frac:
            return False, f"risk too small: {risk_frac:.4f} < {min_risk_frac:.4f}"
    return True, "ok"
