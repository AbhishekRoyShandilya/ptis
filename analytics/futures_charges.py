"""
analytics.futures_charges — the REAL Indian equity-FUTURES (NSE F&O) round-trip cost, per the user
(2026-07-05). Distinct from the equity intraday model in `charges.py`. The dominant cost is STT: it hits
only the SELL side but on the FULL contract value, so it dwarfs brokerage/exchange fees. Charges are
~turnover-linear, so cost-as-%-of-notional is roughly flat; brokerage (flat Rs20/order) is the only piece
that shrinks as contract size grows.

Verified against the user's worked example (stock Rs1000, lot 500, contract Rs5L): round-trip ~= Rs330.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FuturesCostModel:
    brokerage_per_order: float = 20.0     # Rs20 flat per executed order (discount broker); buy + sell = 2x
    stt_sell_pct: float = 0.0005          # 0.05% on SELL notional (full contract value) — the dominant cost.
    #                                       CURRENT rate: revised to 0.05% effective 1 April 2026 (per user,
    #                                       2026-07-05). Prior rate was ~0.02% (Oct-2024). This is the live rate.
    stamp_buy_pct: float = 0.00002        # 0.002% on BUY notional
    exch_txn_pct: float = 0.0000173       # 0.00173% both sides (NSE F&O transaction charge)
    sebi_pct: float = 0.000001            # 0.0001% both sides (Rs10 / crore)
    ipft_pct: float = 0.000001            # 0.0001% both sides (NSE Investor Protection Fund Trust)
    gst_pct: float = 0.18                 # 18% GST on (brokerage + exchange txn charge) ONLY — not STT/stamp
    slippage_bps: float = 3.0             # per-side slippage in bps (execution)

    def round_trip(self, buy_notional: float, sell_notional: float) -> float:
        """Total Rs cost for a one-lot(or n-lot) round trip: buy `buy_notional` + sell `sell_notional`."""
        brok = 2 * self.brokerage_per_order
        stt = self.stt_sell_pct * sell_notional
        stamp = self.stamp_buy_pct * buy_notional
        exch = self.exch_txn_pct * (buy_notional + sell_notional)
        sebi = self.sebi_pct * (buy_notional + sell_notional)
        ipft = self.ipft_pct * (buy_notional + sell_notional)
        gst = self.gst_pct * (brok + exch)
        return brok + stt + stamp + exch + sebi + ipft + gst

    def cost_pct(self, notional: float) -> float:
        """Round-trip cost as a fraction of contract value (entry approx = exit)."""
        return self.round_trip(notional, notional) / notional if notional > 0 else float("nan")


if __name__ == "__main__":
    m = FuturesCostModel()
    print("NSE futures round-trip cost, by contract value:")
    print(f"  {'contract':<12}{'round-trip Rs':<15}{'% of value':<12}{'STT share'}")
    for cv in (500000, 700000, 1000000, 1500000):
        rt = m.round_trip(cv, cv); stt = m.stt_sell_pct * cv
        print(f"  Rs{cv:<10,.0f}Rs{rt:<13.0f}{rt/cv*100:<12.3f}{stt/rt*100:.0f}%")
    print("\n  check user example (Rs5L contract): "
          f"Rs{m.round_trip(500000, 500000):.0f}  (expected ~Rs330)")
