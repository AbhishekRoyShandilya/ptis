"""
avt.execution.sizing — per-SLEEVE trade sizing (the user's model: LOTS for F&O, ₹ AMOUNT for equity).

A capital sleeve (avt.capital.allocator.Sleeve) may specify `size_mode`:
  · "risk_pct" — the engine's default risk-% sizing (unchanged; this module isn't consulted).
  · "lots"     — a fixed number of lots. Futures/equity underlying qty = lots × the symbol's F&O lot size.
  · "amount"   — deploy a fixed ₹ per trade → qty ≈ amount / price.
This module resolves the underlying (future/equity) share count for "lots"/"amount". Options are sized in lots by
the Capital Operator (MoneyPolicy), so they only need the lot COUNT (below).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional


@lru_cache(maxsize=512)
def futures_lot_size(symbol: str) -> Optional[int]:
    """The F&O lot size for `symbol` from the real instrument master (a FUT contract's lot), or None if not
    found (caller falls back). Cached — the master is large."""
    try:
        import marketdata.instruments as I
        df = I._load()
        if df is None or len(df) == 0:
            return None
        it = df["instrument_type"].astype(str).str.upper()
        us = df["underlying_symbol"].astype(str).str.upper()
        fut = df[it.str.contains("FUT") & (us == symbol.upper())]
        if fut.empty:
            return None
        ls = fut["lot_size"].dropna()
        return int(ls.iloc[0]) if len(ls) else None
    except Exception:
        return None


def sleeve_underlying_qty(size_mode: str, size_value: float, symbol: str,
                          entry_price: float, stop_price: float) -> Optional[Dict[str, Any]]:
    """Resolve the underlying (future/equity) sizing for a sleeve. Returns {shares, risk_amount, risk_pct} or None
    (→ the caller falls back to the default risk-% sizing). `risk_pct` is 0 here (sizing is lots/amount-driven, not
    risk-driven); `risk_amount` = shares × stop distance for reporting/R."""
    if entry_price is None or entry_price <= 0:
        return None
    stop_dist = abs(float(entry_price) - float(stop_price)) if stop_price else 0.0

    if size_mode == "lots":
        lots = int(size_value)
        if lots < 1:
            return None
        lot = futures_lot_size(symbol)
        if not lot or lot < 1:
            return None                                  # unknown lot → fall back to risk-% (safe)
        shares = lots * lot
    elif size_mode == "amount":
        amount = float(size_value)
        if amount <= 0:
            return None
        shares = int(amount // float(entry_price))
        if shares < 1:
            return None
    else:
        return None                                      # not a lots/amount sleeve → caller uses risk-%

    return {"shares": int(shares), "risk_amount": float(shares * stop_dist), "risk_pct": 0.0}
