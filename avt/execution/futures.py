"""
avt.execution.futures — FuturesExecution (also serves EQUITY: same underlying sizing/booking). Thin wrapper of
the engine's EXISTING risk-based sizing (`VirtualPositionManager.compute_size`) so wiring it into the engine
does NOT change behaviour — the M2 swap must reproduce the frozen golden cascade numbers exactly.

Real lots/charges: futures book in the underlying at risk-% shares (today's realization); the engine's
`AVTConfig.cost_model` (FuturesCostModel) already deducts real F&O charges on close. Lot rounding for true
futures contracts is a later refinement (the frozen cascade tester sizes by risk-% shares, per verify_avt_cascade).
"""
from __future__ import annotations

from typing import Any, Optional

from avt.intent import Instrument, Intent
from avt.execution.base import ExecutionEngine, ExecutionPlan


class FuturesExecution(ExecutionEngine):
    def __init__(self, instrument: Instrument = Instrument.FUTURE):
        self.instrument = instrument

    def plan(self, intent: Intent, *, equity: float, config: Any, sizer: Any,
             drawdown_pct: Optional[float] = None) -> Optional[ExecutionPlan]:
        # EXACT existing sizing path: risk-based shares on the underlying via VirtualPositionManager.compute_size.
        # `equity`/`drawdown_pct` are forwarded so a partitioned account sizes off the funding sleeve (B.1); when
        # they equal the portfolio values (default/single-sleeve) the result is byte-identical to before.
        sizing = sizer.compute_size(
            conviction=intent.confidence,
            entry_price=intent.entry_ref,
            stop_price=intent.structural_stop,
            config=config,
            equity=equity,
            drawdown_pct=drawdown_pct,
        )
        if not sizing or sizing.get("shares", 0) <= 0:
            return None
        return ExecutionPlan(
            instrument=self.instrument,
            tradeable=intent.symbol,
            direction=intent.direction,
            quantity=int(sizing["shares"]),
            entry_ref=float(intent.entry_ref),
            stop=float(intent.structural_stop),
            meta={"risk_amount": sizing.get("risk_amount"), "risk_pct": sizing.get("risk_pct")},
        )
