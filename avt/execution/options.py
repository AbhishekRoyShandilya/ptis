"""
avt.execution.options — OptionsExecution: express a cascade Intent as a nearest-weekly ATM index option, using
the REAL broker instrument master (`marketdata/instruments.py`) for the contract + lot size (never hardcoded).
Sizing is whole REAL lots, bounded; entry books on the option premium; the exit is premium exit-at-spike +
underlying structural stop + session close; charges are the REAL option F&O charges. This is the M3 execution for
the `cascade_options_v1` template (SPEC frozen). `resolve_atm_weekly` is pure + testable; premium-dependent sizing
+ position lifecycle are wired into the engine open-path in the M3 integration step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from avt.intent import Instrument, Intent
from avt.execution.base import ExecutionEngine, ExecutionPlan

# NIFTY options list under a few underlying spellings across broker masters — normalise.
_UNDERLYING_ALIASES = {"NIFTY": {"NIFTY", "NIFTY 50", "NIFTY50"},
                       "BANKNIFTY": {"BANKNIFTY", "NIFTY BANK", "BANK NIFTY"},
                       "SENSEX": {"SENSEX"}, "FINNIFTY": {"FINNIFTY", "NIFTY FIN SERVICE"},
                       "MIDCPNIFTY": {"MIDCPNIFTY", "NIFTY MID SELECT"}, "BANKEX": {"BANKEX"}}


def _aliases(underlying: str) -> set[str]:
    u = underlying.upper()
    return {a.upper() for a in _UNDERLYING_ALIASES.get(u, {u})}


def resolve_atm_weekly(underlying: str, spot: float, right: str, as_of) -> Optional[dict]:
    """Nearest-expiry (weekly) ATM option for `underlying` at `spot`, from the REAL instrument master. Returns
    {symbol, instrument_key, strike, expiry, lot_size, right} or None if unavailable."""
    import marketdata.instruments as I
    try:
        df = I._load()
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    it = df["instrument_type"].astype(str).str.upper()
    us = df["underlying_symbol"].astype(str).str.upper()
    ot = df["option_type"].astype(str).str.upper()
    opt = df[it.str.contains("OPT") & us.isin(_aliases(underlying)) & (ot == right.upper())].copy()
    if opt.empty:
        return None
    opt["_exp"] = pd.to_datetime(opt["expiry"], errors="coerce")
    if "active" in opt.columns:
        opt = opt[opt["active"].fillna(True).astype(bool)]
    as_of = pd.Timestamp(as_of).normalize()
    fut = opt[opt["_exp"] >= as_of]
    if fut.empty:
        return None
    exp = fut["_exp"].min()                              # nearest weekly expiry
    chain = fut[fut["_exp"] == exp].copy()
    chain["_dist"] = (chain["strike"].astype(float) - float(spot)).abs()
    row = chain.sort_values("_dist").iloc[0]
    lot = int(row["lot_size"]) if pd.notna(row.get("lot_size")) and row.get("lot_size") else None
    return {"symbol": str(row["symbol"]), "instrument_key": row.get("instrument_key"),
            "strike": float(row["strike"]), "expiry": str(exp.date()), "lot_size": lot, "right": right.upper()}


# ── The validated option execution PRIMITIVES (single source) ──────────────────────────────────────────────
# These are the option mechanics of the private money test (replayed on real 1-min NIFTY premium). They live
# here so backtest AND the live engine call ONE implementation — the same
# discipline as avt/cascade_detect.py for detection. The money test is refactored to call these (its own numbers
# are the golden gate proving they're bit-identical).

def option_charges(buy_prem: float, sell_prem: float, qty: int, stt: float = 0.0015) -> float:
    """Real F&O option round-trip charges: STT on sell premium (user 0.15%) + brokerage ₹20/leg + NSE txn
    ~0.03503% + GST 18% + SEBI + stamp (buy side). Mirrors cascade_account_sim.charges exactly."""
    tb, ts = buy_prem * qty, sell_prem * qty
    brokerage = 20 * 2
    stt_c = stt * ts
    txn = 0.0003503 * (tb + ts)
    gst = 0.18 * (brokerage + txn)
    sebi = 0.000001 * (tb + ts)
    stamp = 0.00003 * tb
    return brokerage + stt_c + txn + gst + sebi + stamp


def size_option_lots(capital: float, buy_fill: float, lot: int, deploy: float,
                     fixed_lots: int = 0, max_deploy: float = 0.9) -> int:
    """Whole bounded lots: never deploy more than `max_deploy` of capital into one position; else `deploy` fraction
    (or a fixed lot count if given). Mirrors cascade_account_sim.simulate's sizing exactly. NOTE: on its own this
    still COMPOUNDS off `capital` (the naive fantasy → 12k lots). Real money management goes through MoneyPolicy /
    money_lots below, which caps compounding + lots."""
    if buy_fill <= 0 or lot <= 0:
        return 0
    afford = int((max_deploy * capital) // (buy_fill * lot))
    want = fixed_lots if fixed_lots else int((deploy * capital) // (buy_fill * lot))
    return min(want, afford)


@dataclass
class MoneyPolicy:
    """DISCIPLINED position sizing for lot-based instruments — the fix for the 'CAGR fantasy' (naive compounding
    ballooned NIFTY option lots 5 → 12,092, which is not executable). Modes:
      · 'fixed'          — always `fixed_lots` lots (honest floor; no compounding).
      · 'off'            — size off the STARTING capital, never off grown equity (disciplined, flat sizing).
      · 'capped_compound'— compound the sizing base only UP TO `grow_to_x` × starting capital, then FREEZE. Profits
                           beyond that accumulate as locked cash; size stops growing. Raising the base past the cap
                           is a deliberate USER action (user permission), not automatic. This is the default.
      · 'naive'          — compound off full equity (the fantasy; kept only for contrast).
    ALL modes are then bounded by whole lots, `max_lots` (liquidity), and `max_deploy` (≤ this frac of cash/pos)."""
    mode: str = "capped_compound"
    fixed_lots: int = 1
    deploy: float = 0.35
    grow_to_x: float = 2.0
    max_lots: int = 50
    max_deploy: float = 0.9


def money_lots(policy: MoneyPolicy, *, capital: float, starting_capital: float,
               buy_fill: float, lot: int) -> int:
    """Whole lots under a MoneyPolicy: disciplined sizing base → deploy fraction → bounded by affordability +
    max_lots. This is what the LIVE runner and any honest sim use (never raw size_option_lots' naive compounding)."""
    if buy_fill <= 0 or lot <= 0 or capital <= 0:
        return 0
    if policy.mode == "fixed":
        want = int(policy.fixed_lots)
    else:
        if policy.mode == "off":
            base = starting_capital
        elif policy.mode == "naive":
            base = capital
        else:  # capped_compound (default): grow only up to grow_to_x × start, then freeze
            base = min(capital, policy.grow_to_x * starting_capital)
        want = int((policy.deploy * base) // (buy_fill * lot))
    afford = int((policy.max_deploy * capital) // (buy_fill * lot))
    return max(0, min(want, afford, int(policy.max_lots)))


def hold_exit(opt, spot, mins, entry_prem: float, stop: float, long: bool, ptrail: float):
    """The exit-at-spike hold loop: minute-by-minute, exit on UNDERLYING stop (thesis invalidation) → else a
    premium trailing stop (fall `ptrail` from running peak) → else EOD (last minute). Returns (exit_prem, reason).
    `opt`/`spot` are per-minute ('%H:%M'-indexed) premium/spot series; `mins` the ordered minutes in the hold
    window. Mirrors cascade_account_sim.simulate's exit loop exactly."""
    import numpy as np
    peak = entry_prem
    exit_prem = float(opt.loc[mins[-1]]); reason = "eod"
    for t in mins[1:]:
        sp = spot.get(t, np.nan); p = float(opt.loc[t])
        if sp == sp and ((long and sp <= stop) or (not long and sp >= stop)):
            exit_prem = p; reason = "underlying_SL"; break
        peak = max(peak, p)
        if p <= peak * (1 - ptrail):
            exit_prem = p; reason = "prem_trail"; break
    return exit_prem, reason


class OpenOptionPosition:
    """The INCREMENTAL (live-shaped) form of `hold_exit`: an open option position that is fed ticks one at a time
    and decides, per tick, whether to exit. Polled every few seconds on a live feed, this is the seconds-resolution
    intrabar-safe exit the user asked for — frequent sampling catches an underlying-stop touch or a premium spike
    the moment it happens, not on the 1-min close. Driven minute-by-minute over stored 1-min data it is IDENTICAL
    to the batch `hold_exit` (same order: underlying-stop → peak update → premium-trail), so the validated money
    test reproduces; `verify_options_live_loop` is the gate."""

    def __init__(self, *, entry_prem: float, stop: float, long: bool, ptrail: float,
                 qty: int, lots: int, lot_size: int, strike: float, expiry: str, right: str,
                 tradeable: str, underlying: str, buy_fill: float, slip: float):
        self.entry_prem = entry_prem; self.stop = stop; self.long = long; self.ptrail = ptrail
        self.qty = qty; self.lots = lots; self.lot_size = lot_size
        self.strike = strike; self.expiry = expiry; self.right = right
        self.tradeable = tradeable; self.underlying = underlying
        self.buy_fill = buy_fill; self.slip = slip
        self.peak = entry_prem
        self.last_prem = entry_prem

    def on_tick(self, spot: float, prem: float) -> Optional[str]:
        """Feed one (spot, premium) tick. Returns an exit reason ('underlying_SL' | 'prem_trail') the instant a
        rule fires, else None. Order matches hold_exit exactly. Frequent polling = intrabar-safe."""
        self.last_prem = prem
        if spot == spot and ((self.long and spot <= self.stop) or (not self.long and spot >= self.stop)):
            return "underlying_SL"
        self.peak = max(self.peak, prem)
        if prem <= self.peak * (1 - self.ptrail):
            return "prem_trail"
        return None

    def settle(self, exit_prem: float, stt: float = 0.0015) -> dict:
        """Close the position at `exit_prem`: sell fill (slippage), gross, real charges, net. Mirrors the money
        test's fill+charge arithmetic exactly."""
        sell_fill = max(0.05, exit_prem - self.slip)
        gross = (sell_fill - self.buy_fill) * self.qty
        cost = option_charges(self.buy_fill, sell_fill, self.qty, stt)
        return {"exit_prem": exit_prem, "sell_fill": sell_fill, "gross": gross,
                "cost": cost, "net": gross - cost, "prem_mult": exit_prem / self.entry_prem if self.entry_prem else 0.0}


class OptionsExecution(ExecutionEngine):
    instrument = Instrument.OPTION

    def plan(self, intent: Intent, *, equity: float, config: Any, sizer: Any,
             drawdown_pct: Optional[float] = None) -> Optional[ExecutionPlan]:
        """Resolve the real ATM weekly contract + lot. Premium-dependent lot sizing + booking are completed in the
        engine open-path (it has the live option quote); here we resolve the tradeable + real lot deterministically."""
        right = "CE" if intent.is_long else "PE"
        c = resolve_atm_weekly(intent.symbol, intent.entry_ref, right, intent.entry_ts or pd.Timestamp.now())
        if c is None or not c.get("lot_size"):
            return None
        return ExecutionPlan(
            instrument=Instrument.OPTION, tradeable=c["symbol"], direction=intent.direction,
            quantity=0,                                   # filled once the premium + bounded lots are known (engine)
            entry_ref=intent.entry_ref, stop=intent.structural_stop,
            lots=None, lot_size=c["lot_size"],
            meta={"strike": c["strike"], "expiry": c["expiry"], "right": right,
                  "instrument_key": c.get("instrument_key"), "underlying": intent.symbol,
                  "exit": "premium exit-at-spike + underlying stop + EOD"},
        )
