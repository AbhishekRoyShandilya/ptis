"""
avt.capital.allocator — CAPITAL SLEEVES (B.1): partition the AVT account into INDEPENDENT capital pools so
strategies/instruments run CONCURRENTLY off SEPARATE capital.

The user's model: someone with ₹5L wants ₹2L for futures, ₹50k for options, and the rest for equity intraday —
each pool sized, slotted and drawn-down on its OWN, so a full futures book never starves the options path (and
options never starve futures). Before this, the AVT shared ONE `VirtualPortfolio.cash`/`equity` + ONE global
`max_open_positions`, so whichever strategy filled the slots/capital first BLOCKED the others. Sleeves remove that.

Design (golden-safe):
  · Membership is by INSTRUMENT (future / equity / option) — matches the user's mental model exactly.
  · Trades stay in the ONE portfolio ledger (single DB/reporting path, no double-bookkeeping); per-sleeve
    equity/available/open-count are computed LIVE from that ledger, filtered by instrument.
  · Each sleeve is a sub-account (own capital / equity / peak / drawdown + own position cap).
  · The DEFAULT is a single implicit WILDCARD sleeve holding ALL capital → `is_partitioned` is False and the
    engine keeps using the historical shared-pool path untouched (the futures golden 391/68.8%/+1.16R never moves).
    The allocator is only *consulted* when the user actually configures ≥1 explicit sleeve.

The engine (not this module) owns booking; the allocator only answers: which sleeve funds this instrument, and how
much equity / free capital / open slots does that sleeve have right now.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from avt.capital.operator import AccountState

logger = logging.getLogger(__name__)

WILDCARD = "*"


def instrument_of(trade: Any) -> str:
    """The instrument a booked trade belongs to (defaults to 'future' — the historical underlying trade)."""
    return str(getattr(trade, "instrument", "future") or "future")


@dataclass
class Sleeve:
    """One independent capital pool. `instruments` is the set of instrument strings it funds; {'*'} = all.

    `size_mode` / `size_value` set HOW trades in this sleeve are sized (the user's model — lots for F&O, ₹ amount
    for equity):
      · "risk_pct" — the engine's default risk-% sizing (backward-compatible; size_value ignored).
      · "lots"     — a FIXED number of lots (`size_value` lots). For options → that many option lots; for futures
                     → size_value × the symbol's F&O lot size.
      · "amount"   — deploy `size_value` rupees per trade (buy ≈ amount/price shares). Intended for intraday
                     equity."""
    name: str
    capital: float
    instruments: frozenset
    max_positions: int
    size_mode: str = "risk_pct"          # "risk_pct" | "lots" | "amount"
    size_value: float = 0.0              # lots count (lots) OR rupees per trade (amount)
    account: AccountState = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.account is None:
            self.account = AccountState(starting_capital=self.capital)

    def covers(self, instrument: str) -> bool:
        return WILDCARD in self.instruments or instrument in self.instruments


class CapitalAllocator:
    """Partitions the account into independent capital sleeves. See module docstring.

    Consult order: the engine calls `sleeve_for(instrument)` to find the funding pool, then `sleeve_equity` /
    `sleeve_available` / `sleeve_open_count` / `sleeve_drawdown_pct` for the sleeve-scoped money math. In the
    default single-wildcard configuration `is_partitioned` is False and the engine bypasses all of this."""

    def __init__(self, portfolio: Any, sleeve_specs: Optional[List[Dict[str, Any]]],
                 default_max_positions: int):
        self.portfolio = portfolio
        self._default_max = int(default_max_positions)
        self.sleeves: List[Sleeve] = []

        specs = sleeve_specs or []
        if not specs:
            # implicit single wildcard sleeve = the whole capital (historical shared-pool behaviour)
            self.sleeves.append(Sleeve(
                name="all",
                capital=float(getattr(portfolio, "starting_capital", 0.0)),
                instruments=frozenset({WILDCARD}),
                max_positions=self._default_max,
            ))
        else:
            for s in specs:
                instruments = frozenset(str(i).lower() for i in s.get("instruments", []))
                if not instruments:
                    instruments = frozenset({WILDCARD})
                self.sleeves.append(Sleeve(
                    name=str(s.get("name", "sleeve")),
                    capital=float(s["capital"]),
                    instruments=instruments,
                    max_positions=int(s.get("max_positions") or self._default_max),
                    size_mode=str(s.get("size_mode", "risk_pct") or "risk_pct"),
                    size_value=float(s.get("size_value", 0.0) or 0.0),
                ))

        self._wildcard: Optional[Sleeve] = next(
            (sl for sl in self.sleeves if WILDCARD in sl.instruments), None)

        total = sum(sl.capital for sl in self.sleeves)
        start = float(getattr(portfolio, "starting_capital", 0.0))
        if self.is_partitioned and start > 0 and total > start + 1e-6:
            logger.warning("AVT sleeves over-allocate capital: sleeves sum Rs%s > account Rs%s",
                           f"{total:,.0f}", f"{start:,.0f}")

    # ── configuration ──────────────────────────────────────────────
    @property
    def is_partitioned(self) -> bool:
        """True once the account is really split — i.e. NOT the single implicit wildcard sleeve. When False the
        engine keeps its byte-identical shared-pool path (golden-safe)."""
        if len(self.sleeves) == 1 and WILDCARD in self.sleeves[0].instruments:
            return False
        return True

    def sleeve_for(self, instrument: str) -> Sleeve:
        """The sleeve funding `instrument` — an explicit match wins over the wildcard; wildcard is the fallback."""
        inst = str(instrument).lower()
        for sl in self.sleeves:
            if inst in sl.instruments:
                return sl
        if self._wildcard is not None:
            return self._wildcard
        return self.sleeves[0]

    # ── per-sleeve money math (live from the one ledger) ───────────
    def _closed_in(self, sleeve: Sleeve) -> List[Any]:
        return [t for t in self.portfolio.closed_trades if sleeve.covers(instrument_of(t))]

    def _open_in(self, sleeve: Sleeve) -> List[Any]:
        return [p for p in self.portfolio.open_positions if sleeve.covers(instrument_of(p))]

    def sleeve_cash(self, sleeve: Sleeve) -> float:
        """Sleeve capital + realized P&L booked to it (correct capital-plus-pnl model, per-sleeve)."""
        realized = sum(float(getattr(t, "realized_pnl", 0.0) or 0.0) for t in self._closed_in(sleeve))
        return sleeve.capital + realized

    def sleeve_equity(self, sleeve: Sleeve) -> float:
        """Sleeve cash + open unrealized — the sizing base for this pool."""
        unrealized = sum(float(getattr(p, "unrealized_pnl", 0.0) or 0.0) for p in self._open_in(sleeve))
        return self.sleeve_cash(sleeve) + unrealized

    def sleeve_deployed(self, sleeve: Sleeve) -> float:
        """Capital tied up in this sleeve's open positions (entry notional / option premium outlay)."""
        return sum(float(getattr(p, "entry_price", 0.0) or 0.0) * int(getattr(p, "remaining_quantity", 0) or 0)
                   for p in self._open_in(sleeve))

    def sleeve_available(self, sleeve: Sleeve) -> float:
        """Free capital in this sleeve = sleeve cash − deployed notional."""
        return self.sleeve_cash(sleeve) - self.sleeve_deployed(sleeve)

    def sleeve_open_count(self, sleeve: Sleeve) -> int:
        return len(self._open_in(sleeve))

    def sleeve_drawdown_pct(self, sleeve: Sleeve) -> float:
        """Sleeve drawdown, tracking the sleeve's OWN peak (updated via `mark`)."""
        sleeve.account.set_equity(self.sleeve_equity(sleeve))
        return sleeve.account.drawdown_pct

    def mark(self) -> None:
        """Refresh every sleeve's account equity/peak from the ledger (call once per scan cycle)."""
        for sl in self.sleeves:
            sl.account.set_equity(self.sleeve_equity(sl))

    # ── dashboard / telemetry ──────────────────────────────────────
    def summary(self) -> List[Dict[str, Any]]:
        """Per-sleeve snapshot for the API/frontend: capital, equity, free, deployed, open slots, drawdown."""
        out: List[Dict[str, Any]] = []
        for sl in self.sleeves:
            out.append({
                "name": sl.name,
                "instruments": sorted(sl.instruments),
                "capital": round(sl.capital, 2),
                "equity": round(self.sleeve_equity(sl), 2),
                "available": round(self.sleeve_available(sl), 2),
                "deployed": round(self.sleeve_deployed(sl), 2),
                "open_positions": self.sleeve_open_count(sl),
                "max_positions": sl.max_positions,
                "drawdown_pct": round(self.sleeve_drawdown_pct(sl) * 100.0, 2),
                "size_mode": sl.size_mode,
                "size_value": sl.size_value,
            })
        return out
