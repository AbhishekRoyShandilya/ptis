"""
analytics.charges — realistic Indian INTRADAY equity cost model, and its conversion to R.

The operating-process backtest (`avt_trader`) measures GROSS R (structural stop + let-run exit) but no
costs. Retail edges die in the costs, so this makes them explicit. The KEY insight: cost-in-R depends on
the STOP DISTANCE — a tight structural stop means charges eat a big fraction of each R; a wide stop dilutes
them. cost_R = round_trip_cost / (risk_per_share * qty) = round_trip_cost% / (stop_distance_as_%_of_entry).

Defaults ~ a discount broker (Zerodha-style). Adjust `slippage_bps` to your fills.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    brokerage_pct: float = 0.0003      # 0.03% per order (intraday), capped at ₹20/order
    brokerage_cap: float = 20.0        # ₹20 max per executed order
    stt_sell_pct: float = 0.00025      # 0.025% on SELL turnover (intraday)
    exch_txn_pct: float = 0.0000297    # NSE ~0.00297% on turnover (both sides)
    sebi_pct: float = 0.000001         # ₹10 / crore
    stamp_buy_pct: float = 0.00003     # 0.003% on BUY turnover
    gst_pct: float = 0.18              # 18% GST on (brokerage + exch + sebi)
    slippage_bps: float = 5.0          # per-side slippage in basis points (5 bps = 0.05%); round-trip = 2x

    def round_trip_cost(self, entry: float, exit: float, qty: float) -> float:
        """Total ₹ cost for one intraday round-trip (buy `entry` + sell `exit`, `qty` shares)."""
        buy_val, sell_val = entry * qty, exit * qty
        brok = min(self.brokerage_pct * buy_val, self.brokerage_cap) + min(self.brokerage_pct * sell_val, self.brokerage_cap)
        stt = self.stt_sell_pct * sell_val
        exch = self.exch_txn_pct * (buy_val + sell_val)
        sebi = self.sebi_pct * (buy_val + sell_val)
        stamp = self.stamp_buy_pct * buy_val
        gst = self.gst_pct * (brok + exch + sebi)
        slip = (self.slippage_bps / 1e4) * (buy_val + sell_val)     # slippage on both fills
        return brok + stt + exch + sebi + stamp + gst + slip

    def cost_in_R(self, entry: float, stop: float, qty: float = 1.0, exit: float | None = None) -> float:
        """The round-trip cost expressed in R (units of the structural risk |entry-stop|). Exit defaults to
        entry (cost is nearly turnover-linear, so exit≈entry is a fine approximation for the R fraction)."""
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return float("nan")
        cost = self.round_trip_cost(entry, exit if exit is not None else entry, qty)
        return cost / (risk_per_share * qty)


if __name__ == "__main__":
    m = CostModel()
    print("Indian intraday round-trip cost in R, by stop distance (entry ₹1000, qty scaled large so ₹20 cap is negligible):")
    print(f"  {'stop dist':<12}{'stop %':<9}{'cost as % turnover':<20}{'COST IN R':<10}")
    entry = 1000.0; qty = 1000.0    # ₹10L position → brokerage cap negligible, costs ~turnover-linear
    for stop_pct in (0.003, 0.005, 0.01, 0.02, 0.03):
        stop = entry * (1 - stop_pct)
        rt = m.round_trip_cost(entry, entry, qty)
        cr = m.cost_in_R(entry, stop, qty)
        print(f"  ₹{entry-stop:<11.1f}{stop_pct*100:<9.1f}{rt/(entry*qty)*100:<20.3f}{cr:<10.2f}")
    print("\n  → tight stops (0.3-0.5%) cost 0.3-0.5R PER TRADE in charges — that's the retail killer.")
    print("  → a setup needs avg gross R meaningfully above its cost_R to make real money net.")
