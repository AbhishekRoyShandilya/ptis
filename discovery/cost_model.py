"""
Transaction Cost Model — Indian FNO market
============================================

Without cost adjustment, backtest R-multiples are gross of:
  - Brokerage
  - Slippage (market impact)
  - Bid-ask spread
  - STT (Securities Transaction Tax)
  - Exchange charges
  - GST on brokerage + exchange charges
  - Stamp duty (buy side)

For scalping strategies (target < 1R), gross-vs-net can flip a perceived
edge into a loss. For Phase 2's target optimizer, "optimal target = 1.2R"
might really be 1.0R after costs — the difference matters.

This module:
  1. Computes realistic costs for each trade (in ₹ and in R-multiples)
  2. Bulk-adjusts a list of BacktestResult-like dicts
  3. Stores both gross and net R on every adjusted trade

Defaults are tuned for Indian FNO (Zerodha-style discount broker), but
all knobs are configurable via CostModelConfig for other brokers/instruments.

Real-money safety: this only adjusts BACKTEST reports — it never changes
live trading behavior. The trader sees both gross and net so they can
verify the cost model is reasonable for their broker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Output dataclasses ──────────────────────────────────────────────


@dataclass
class TransactionCosts:
    """Round-trip transaction costs for a single trade, in ₹ unless noted."""
    brokerage: float = 0.0
    slippage: float = 0.0
    spread_cost: float = 0.0
    stt: float = 0.0
    exchange_charges: float = 0.0
    gst: float = 0.0
    stamp_duty: float = 0.0

    # cost_r is set by apply_costs() once we know the trade's 1R-in-rupees
    cost_r: float = 0.0

    @property
    def total(self) -> float:
        """Total round-trip cost in ₹."""
        return (
            self.brokerage + self.slippage + self.spread_cost
            + self.stt + self.exchange_charges + self.gst + self.stamp_duty
        )

    def to_dict(self) -> dict:
        return {
            "brokerage": round(self.brokerage, 2),
            "slippage": round(self.slippage, 2),
            "spread_cost": round(self.spread_cost, 2),
            "stt": round(self.stt, 2),
            "exchange_charges": round(self.exchange_charges, 2),
            "gst": round(self.gst, 2),
            "stamp_duty": round(self.stamp_duty, 2),
            "total": round(self.total, 2),
            "cost_r": round(self.cost_r, 4),
        }


@dataclass
class CostModelConfig:
    """All cost knobs in one place — defaults tuned for Indian FNO + discount broker."""

    # ── Brokerage ──
    # "flat":  brokerage_per_order × 2 (buy + sell), independent of trade value
    # "percentage": trade_value × brokerage_pct × 2
    # "hybrid": min of the two (matches Zerodha-style "₹20 or 0.03%, whichever lower")
    brokerage_per_order: float = 20.0
    brokerage_pct: float = 0.0003           # 0.03%
    brokerage_model: str = "hybrid"

    # ── Slippage ──
    # "fixed_pct":    flat slippage_pct
    # "volume_based": multiplied when volume_ratio < 0.5 (low liquidity)
    # "volatility_based": multiplied in volatile regimes
    slippage_pct: float = 0.0005            # 5 bps per side (round trip = 0.10%)
    slippage_model: str = "volatility_based"
    slippage_volatile_multiplier: float = 2.0

    # ── Spread ──
    spread_pct: float = 0.0003              # 3 bps per side for liquid FNO
    spread_illiquid_multiplier: float = 3.0

    # ── India regulatory charges ──
    # STT rates per current schedule (verify against your broker statements):
    #   futures (sell side):       0.05%   (recently increased; was 0.0125% pre-2024)
    #   options (sell side, prem): 0.10%   (recently increased; was 0.0625%)
    #   equity intraday (sell):    0.025%
    #   equity delivery (both):    0.10% each side
    stt_pct: float = 0.0005                 # 0.05% — futures sell side default
    stt_pct_options: float = 0.001          # 0.10% — options sell side on premium
    stt_pct_equity_intraday: float = 0.00025  # 0.025% — equity intraday sell side
    stt_pct_equity_delivery: float = 0.001    # 0.10% — equity delivery, BOTH sides
    exchange_charges_pct: float = 0.0000190 # NSE charges (per side)
    gst_pct: float = 0.18                   # 18% GST on (brokerage + exchange charges)
    stamp_duty_pct: float = 0.00002         # 0.002% on buy side

    # ── Defaults for backtest adjustment ──
    default_quantity: int = 250             # ~1 NIFTY lot
    instrument_type: str = "futures"        # "futures" | "options" | "equity_intraday" | "equity_delivery"


# ── Main class ─────────────────────────────────────────────────────


class CostModel:
    """
    Computes realistic transaction costs and applies them to backtest R-multiples.

    Usage:
        model = CostModel()  # default Indian FNO config
        # Per-trade
        net_r, costs = model.apply_costs(
            r_multiple=1.5, entry_price=2500, stop_distance=25,
            quantity=250, regime="normal",
        )
        # Bulk
        adjusted_trades = model.adjust_backtest_results(
            trades, default_entry_price=2500, default_stop_distance=25,
        )
    """

    def __init__(self, config: Optional[CostModelConfig] = None):
        self.config = config or CostModelConfig()

    # ── Cost computation ──

    def compute_costs(
        self,
        entry_price: float,
        quantity: int = 1,
        regime: str = "normal",
        volume_ratio: float = 1.0,
        is_liquid: bool = True,
    ) -> TransactionCosts:
        """
        Compute round-trip costs in ₹ for a trade with the given parameters.
        Does NOT compute cost_r — call apply_costs() for that.
        """
        cfg = self.config
        trade_value = entry_price * quantity

        # ── Brokerage (round trip = buy + sell) ──
        flat_total = cfg.brokerage_per_order * 2
        pct_total = trade_value * cfg.brokerage_pct * 2
        if cfg.brokerage_model == "flat":
            brokerage = flat_total
        elif cfg.brokerage_model == "percentage":
            brokerage = pct_total
        else:  # hybrid — min of flat and percentage
            brokerage = min(flat_total, pct_total)

        # ── Slippage ──
        slippage_pct = cfg.slippage_pct
        if cfg.slippage_model == "volatility_based" and "volatile" in (regime or "").lower():
            slippage_pct *= cfg.slippage_volatile_multiplier
        elif cfg.slippage_model == "volume_based" and volume_ratio < 0.5:
            slippage_pct *= 1.5
        slippage = trade_value * slippage_pct * 2  # entry + exit

        # ── Spread ──
        spread_pct = cfg.spread_pct
        if not is_liquid:
            spread_pct *= cfg.spread_illiquid_multiplier
        spread_cost = trade_value * spread_pct * 2

        # ── India regulatory ──
        # STT structure varies by instrument type (rates set in CostModelConfig)
        instrument = (cfg.instrument_type or "").lower()
        if instrument == "options":
            stt = trade_value * cfg.stt_pct_options              # sell side, on premium
        elif instrument == "equity_intraday":
            stt = trade_value * cfg.stt_pct_equity_intraday      # sell side only
        elif instrument == "equity_delivery" or instrument == "equity":
            stt = trade_value * cfg.stt_pct_equity_delivery * 2  # both sides
        else:  # futures (default)
            stt = trade_value * cfg.stt_pct                      # sell side only

        exchange_charges = trade_value * cfg.exchange_charges_pct * 2
        gst = (brokerage + exchange_charges) * cfg.gst_pct
        stamp_duty = trade_value * cfg.stamp_duty_pct  # buy side

        return TransactionCosts(
            brokerage=brokerage,
            slippage=slippage,
            spread_cost=spread_cost,
            stt=stt,
            exchange_charges=exchange_charges,
            gst=gst,
            stamp_duty=stamp_duty,
        )

    # ── Apply to a single trade (returns net R + cost breakdown) ──

    def apply_costs(
        self,
        r_multiple: float,
        entry_price: float,
        stop_distance: float,
        quantity: int = 1,
        regime: str = "normal",
        volume_ratio: float = 1.0,
        is_liquid: bool = True,
    ) -> Tuple[float, TransactionCosts]:
        """
        Convert gross R to net R after costs.

        Args:
            r_multiple:    gross R from backtest (signed: + for win, - for loss)
            entry_price:   entry price in ₹
            stop_distance: distance from entry to stop in ₹ per share (defines 1R)
            quantity:      number of units traded
            regime:        regime label (e.g. "volatile") for slippage adjustment
            volume_ratio:  current volume / average (< 1 = low liquidity)
            is_liquid:     whether this stock has tight spreads

        Returns:
            (net_r_multiple, TransactionCosts) — costs.cost_r is populated
        """
        costs = self.compute_costs(
            entry_price=entry_price,
            quantity=quantity,
            regime=regime,
            volume_ratio=volume_ratio,
            is_liquid=is_liquid,
        )

        # Convert ₹ cost to R-units. 1R in rupees = stop_distance × quantity
        one_r_in_rupees = stop_distance * quantity
        if one_r_in_rupees <= 0:
            logger.warning(
                "apply_costs: stop_distance × quantity = %.4f — cannot convert to R",
                one_r_in_rupees,
            )
            costs.cost_r = 0.0
            return r_multiple, costs

        cost_r = costs.total / one_r_in_rupees
        costs.cost_r = round(cost_r, 4)

        net_r = r_multiple - cost_r
        return round(net_r, 4), costs

    # ── Bulk adjust a list of trades ──

    def adjust_backtest_results(
        self,
        trades: List[dict],
        default_entry_price: Optional[float] = None,
        default_stop_distance: Optional[float] = None,
        default_quantity: Optional[int] = None,
    ) -> List[dict]:
        """
        Bulk-apply cost adjustment to a list of trade dicts.

        Each trade gets these new keys added (existing keys preserved):
            r_multiple_gross  — original r_multiple (pre-cost)
            r_multiple        — overwritten with NET r (post-cost)
            r_multiple_net    — same as r_multiple (explicit name)
            cost_r            — cost expressed in R-multiples
            cost_breakdown    — dict with full ₹ breakdown

        Args:
            trades: list of dicts with at least 'r_multiple'.
                    Optionally 'entry_price', 'stop_distance', 'quantity',
                    'regime_at_entry', 'volume_ratio'.
            default_entry_price: fallback if trade dict doesn't have entry_price
            default_stop_distance: fallback if trade dict doesn't have stop_distance
            default_quantity: fallback (else uses config.default_quantity)

        Returns:
            New list of dicts (input not mutated).
        """
        adjusted: List[dict] = []
        cfg = self.config
        dq = default_quantity or cfg.default_quantity

        for t in trades:
            entry_price = t.get("entry_price", default_entry_price)
            stop_dist = t.get("stop_distance", default_stop_distance)
            qty = t.get("quantity", dq)
            regime = t.get("regime_at_entry", "normal") or "normal"
            vol_ratio = t.get("volume_ratio", 1.0) or 1.0
            is_liquid = t.get("is_liquid", True)

            raw_r = float(t.get("r_multiple", 0.0) or 0.0)

            if entry_price is None or stop_dist is None or stop_dist <= 0:
                # Cannot compute — pass through unchanged with cost_r=0
                logger.debug(
                    "Skipping cost adjustment for trade (missing entry_price/stop_distance)"
                )
                copy = dict(t)
                copy["r_multiple_gross"] = raw_r
                copy["r_multiple_net"] = raw_r
                copy["r_multiple"] = raw_r
                copy["cost_r"] = 0.0
                copy["cost_breakdown"] = {}
                adjusted.append(copy)
                continue

            net_r, costs = self.apply_costs(
                r_multiple=raw_r,
                entry_price=float(entry_price),
                stop_distance=float(stop_dist),
                quantity=int(qty),
                regime=regime,
                volume_ratio=float(vol_ratio),
                is_liquid=bool(is_liquid),
            )

            copy = dict(t)
            copy["r_multiple_gross"] = raw_r
            copy["r_multiple_net"] = net_r
            copy["r_multiple"] = net_r          # downstream code reads this
            copy["cost_r"] = costs.cost_r
            copy["cost_breakdown"] = costs.to_dict()
            adjusted.append(copy)

        # Summary log
        if adjusted:
            valid = [t for t in adjusted if t.get("cost_r", 0) > 0]
            if valid:
                gross_avg = float(np.mean([t["r_multiple_gross"] for t in valid]))
                net_avg = float(np.mean([t["r_multiple_net"] for t in valid]))
                cost_avg = float(np.mean([t["cost_r"] for t in valid]))
                logger.info(
                    "Cost adjustment applied: %d trades | gross avg R=%.3f, "
                    "net avg R=%.3f, avg cost=%.3fR",
                    len(valid), gross_avg, net_avg, cost_avg,
                )

        return adjusted


# ── Module-level singleton ─────────────────────────────────────────


_singleton: Optional[CostModel] = None


def get_cost_model(config: Optional[CostModelConfig] = None) -> CostModel:
    """Get the module-level cost model singleton."""
    global _singleton
    if _singleton is None or config is not None:
        _singleton = CostModel(config=config)
    return _singleton
