"""
avt.capital.sim_config — the CONFIGURABLE settings surface for the Capital Operator bench (the "test by setting
various capital, charges, #trades, compounding, etc." the user asked for). Backend/CLI first; the frontend
"Capital Lab" panel (Phase 4) will bind to exactly these fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SimConfig:
    # ── Capital ──────────────────────────────────────────────
    capital: float = 200_000.0            # starting account (Rs)

    # ── Sizing policy ────────────────────────────────────────
    # fixed_lot | fixed_fractional | vol_target | kelly_frac | naive_compound (the fantasy, for contrast)
    sizing: str = "fixed_lot"
    fixed_lots: int = 1                   # fixed_lot: how many lots, always
    risk_frac: float = 0.01               # fixed_fractional / vol_target: risk this fraction of equity per trade
    kelly_fraction: float = 0.25          # kelly_frac: fraction of full-Kelly to use (safety)
    kelly_window: int = 40                # kelly_frac: rolling # of PAST trades to estimate edge (point-in-time)
    vol_target_frac: float = 0.01         # vol_target: target per-trade risk as a fraction of equity

    # ── Compounding ──────────────────────────────────────────
    # off = size off the STARTING capital (disciplined floor); bounded = off current equity but lot-granular +
    # leverage/affordability capped; naive = off current equity, continuous, uncapped (the fantasy — for contrast).
    compounding: str = "bounded"          # off | bounded | naive
    max_lots: int = 20                    # hard cap on lots per trade (capacity/liquidity ceiling)
    max_leverage: float = 5.0             # total notional <= max_leverage x equity (intraday MIS-like)
    margin_pct: float = 0.20              # span+exposure margin as a fraction of contract value (~5x)

    # ── Costs ────────────────────────────────────────────────
    charge_model: str = "futures"         # futures (FuturesCostModel) | none
    slippage_bps: float = 3.0             # per-side slippage (bps of price)

    # ── Portfolio / concurrency (the bench trade stream is already one-at-a-time; these bound future streams) ─
    max_concurrency: int = 1              # max simultaneous positions (bench = 1)
    heat_cap_pct: float = 6.0             # max total open risk as % of equity (reserved for multi-position streams)

    # ── Scope ────────────────────────────────────────────────
    symbols: Optional[List[str]] = None   # None = all in the bench
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    max_trades: Optional[int] = None      # cap # of trades (None = all)

    def label(self) -> str:
        c = self.sizing + ("" if self.sizing == "fixed_lot" else f"@{self.risk_frac:.1%}")
        return f"{c}/{self.compounding}"
