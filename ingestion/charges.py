"""
Brokerage + statutory charge estimator for Indian F&O / equity trades.

WHY THIS EXISTS: Groww's API does NOT return per-trade brokerage or taxes — they
only appear on the contract note later. To show a realistic NET P&L (not just the
raw price difference), we estimate charges from the standard NSE/BSE fee
structure. These are ESTIMATES; the exact figures on Groww's contract note may
differ by a rupee or two. All rates are centralised here so they're trivial to
update when SEBI/exchange rates change.

Net P&L = gross P&L (price difference) − estimated round-trip charges.
"""

from __future__ import annotations

from typing import Dict

# ── Brokerage ──
# Groww (discount broker): Rs 20 per executed order for F&O / intraday.
BROKERAGE_PER_ORDER = 20.0

# ── Statutory rates (review when regulations change) ──
STT_OPTIONS_SELL = 0.0010        # 0.10% of premium, SELL side only
STT_FUTURES_SELL = 0.0002        # 0.02% of turnover, SELL side
STT_EQUITY_DELIVERY = 0.0010     # 0.10% both sides (not used for F&O)

# Exchange transaction charge on turnover (premium for options)
TXN_CHARGE = {
    "NSE_OPT": 0.00035,
    "BSE_OPT": 0.000325,
    "NSE_FUT": 0.0000173,
    "BSE_FUT": 0.0000173,
}
SEBI_CHARGE = 0.000001           # Rs 10 per crore
STAMP_OPTIONS_BUY = 0.00003      # 0.003% on BUY side
STAMP_FUTURES_BUY = 0.00002      # 0.002% on BUY side
GST = 0.18                       # 18% on (brokerage + exchange txn + SEBI)


def infer_instrument(symbol: str) -> str:
    """option | future | equity — inferred from the trading symbol."""
    s = (symbol or "").upper()
    if s.endswith("CE") or s.endswith("PE"):
        return "option"
    if "FUT" in s:
        return "future"
    return "equity"


def infer_exchange(symbol: str) -> str:
    """BSE for Sensex/Bankex derivatives, else NSE."""
    s = (symbol or "").upper()
    if s.startswith("SENSEX") or s.startswith("BANKEX"):
        return "BSE"
    return "NSE"


def estimate_charges(
    entry_price: float,
    exit_price: float,
    quantity: float,
    *,
    exchange: str = "NSE",
    instrument: str = "option",
) -> Dict[str, float]:
    """
    Estimate round-trip (buy + sell) charges. Returns a breakdown dict with
    'total'. Returns {'total': 0.0} when turnover is non-positive.
    """
    q = abs(quantity or 0)
    buy_val = (entry_price or 0.0) * q
    sell_val = (exit_price or 0.0) * q
    turnover = buy_val + sell_val
    if turnover <= 0:
        return {"total": 0.0}

    ex = (exchange or "NSE").upper()

    brokerage = BROKERAGE_PER_ORDER * 2  # buy + sell legs
    if instrument == "option":
        stt = STT_OPTIONS_SELL * sell_val
        txn = TXN_CHARGE.get(f"{ex}_OPT", TXN_CHARGE["NSE_OPT"]) * turnover
        stamp = STAMP_OPTIONS_BUY * buy_val
    elif instrument == "future":
        stt = STT_FUTURES_SELL * sell_val
        txn = TXN_CHARGE.get(f"{ex}_FUT", TXN_CHARGE["NSE_FUT"]) * turnover
        stamp = STAMP_FUTURES_BUY * buy_val
    else:  # equity (delivery approximation)
        stt = STT_EQUITY_DELIVERY * turnover
        txn = 0.0000297 * turnover
        stamp = 0.00015 * buy_val

    sebi = SEBI_CHARGE * turnover
    gst = GST * (brokerage + txn + sebi)
    total = brokerage + stt + txn + sebi + stamp + gst

    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_txn": round(txn, 2),
        "sebi": round(sebi, 4),
        "stamp_duty": round(stamp, 2),
        "gst": round(gst, 2),
        "total": round(total, 2),
    }


def estimate_charges_for_symbol(
    symbol: str, entry_price: float, exit_price: float, quantity: float,
    exchange: str = "",
) -> Dict[str, float]:
    """Convenience: infer instrument/exchange from the symbol when not given."""
    return estimate_charges(
        entry_price, exit_price, quantity,
        exchange=exchange or infer_exchange(symbol),
        instrument=infer_instrument(symbol),
    )
