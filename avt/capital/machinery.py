"""
avt.capital.machinery — the Capital Operator's VALIDATED MACHINERY: the deterministic primitives it acts
*through*. These are the reflex baseline AND the reasoner's tools — pure, deterministic, unit-testable. Nothing
here reasons; it sizes, margins, costs, and bounds. (Design: docs/avt_adaptive_operator.md §3, §12.)

Sizing is separated into a POLICY (how much risk to want) and BOUNDS (lot granularity + margin affordability +
leverage cap + capacity cap) — so compounding is *bounded* by real-world constraints, which is exactly the fix
for the equity-scaling "CAGR fantasy".
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

from analytics.futures_charges import FuturesCostModel
from avt.capital.sim_config import SimConfig

# Current NSE F&O lot sizes (from marketdata/instruments FUT rows; revised over 2024-25, so approximate for the
# historical window). Shared with scripts/research/cascade_futures.py.
LOTS = {"HDFCBANK": 650, "AXISBANK": 625, "BAJFINANCE": 750, "HINDALCO": 700,
        "JSWSTEEL": 675, "HCLTECH": 400, "EICHERMOT": 100, "DLF": 950}


def lot_size(symbol: str) -> Optional[int]:
    return LOTS.get(symbol)


def cost_model_for(cfg: SimConfig) -> Optional[FuturesCostModel]:
    return FuturesCostModel(slippage_bps=cfg.slippage_bps) if cfg.charge_model == "futures" else None


def _kelly_fraction(recent_r: Sequence[float], kelly_fraction: float) -> Optional[float]:
    """Full-Kelly fraction of bankroll to risk, estimated from PAST realized R (point-in-time), then scaled by
    `kelly_fraction` for safety. Returns None if there isn't enough history (→ caller falls back to risk_frac)."""
    r = [x for x in recent_r if x == x]
    if len(r) < 10:
        return None
    wins = [x for x in r if x > 0]
    losses = [-x for x in r if x <= 0]
    if not wins or not losses:
        return None
    p = len(wins) / len(r)
    b = (sum(wins) / len(wins)) / (sum(losses) / len(losses))   # avg win / avg loss, in R
    if b <= 0:
        return None
    f_star = p - (1.0 - p) / b                                   # Kelly on the win/loss odds
    f_star = max(0.0, min(1.0, f_star))
    return kelly_fraction * f_star


def desired_risk_fraction(cfg: SimConfig, recent_r: Sequence[float]) -> float:
    """Fraction of the sizing-equity to RISK on this trade (before lot/margin/leverage bounds)."""
    if cfg.sizing == "fixed_fractional":
        return cfg.risk_frac
    if cfg.sizing == "vol_target":
        return cfg.vol_target_frac
    if cfg.sizing == "kelly_frac":
        f = _kelly_fraction(recent_r, cfg.kelly_fraction)
        return f if f is not None else cfg.risk_frac       # cold-start → fixed_fractional floor
    return cfg.risk_frac                                    # fixed_lot handled separately in num_lots()


def num_lots(cfg: SimConfig, equity: float, entry_px: float, stop_px: float,
             lot: int, recent_r: Sequence[float]) -> float:
    """How many lots to trade under the config. Returns an int for real modes; a float only for `naive`
    compounding (the uncapped fantasy, kept for contrast). Applies: sizing policy → lot granularity →
    margin affordability → leverage cap → capacity cap."""
    stop_dist = abs(entry_px - stop_px)
    if stop_dist <= 0 or entry_px <= 0 or equity <= 0:
        return 0.0

    # sizing-equity: OFF compounding sizes off the STARTING capital (disciplined floor).
    eq_for_sizing = cfg.capital if cfg.compounding == "off" else equity

    if cfg.sizing == "fixed_lot":
        raw = float(cfg.fixed_lots)
    else:
        risk_rs = desired_risk_fraction(cfg, recent_r) * eq_for_sizing
        raw = risk_rs / (stop_dist * lot)

    if cfg.compounding == "naive":
        return max(0.0, raw)                                # continuous, uncapped → the fantasy

    lots = math.floor(raw)
    # bounds (real world): affordability, leverage, capacity
    margin_per_lot = cfg.margin_pct * entry_px * lot
    if margin_per_lot > 0:
        lots = min(lots, math.floor(equity / margin_per_lot))
    notional_per_lot = entry_px * lot
    if notional_per_lot > 0:
        lots = min(lots, math.floor(equity * cfg.max_leverage / notional_per_lot))
    lots = min(lots, cfg.max_lots)
    return float(max(0, lots))


def trade_pnl(cfg: SimConfig, cost_model: Optional[FuturesCostModel], direction: str,
              entry_px: float, exit_px: float, lot: int, n_lots: float) -> tuple[float, float]:
    """Net Rs P&L for `n_lots` of a futures round-trip, incl. slippage + real F&O charges. Returns (net, charges).
    STT is sell-side: long sells at exit, short sells at entry."""
    qty = lot * n_lots
    if qty <= 0:
        return 0.0, 0.0
    s = cfg.slippage_bps / 1e4
    up = direction == "long"
    eff_entry = entry_px * (1 + s) if up else entry_px * (1 - s)
    eff_exit = exit_px * (1 - s) if up else exit_px * (1 + s)
    gross = (eff_exit - eff_entry) * qty if up else (eff_entry - eff_exit) * qty
    if cost_model is None:
        return gross, 0.0
    buy_notional = eff_entry * qty if up else eff_exit * qty
    sell_notional = eff_exit * qty if up else eff_entry * qty
    ch = cost_model.round_trip(buy_notional, sell_notional)
    return gross - ch, ch
