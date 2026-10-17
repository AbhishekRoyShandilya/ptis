"""
avt.execution.base — the ExecutionEngine contract. An engine turns an Intent (+ the operator's sizing) into a
concrete `ExecutionPlan` for ONE instrument type, and knows how to manage/close that instrument. The AVT engine
dispatches to the right one; it never special-cases instruments inline.

M1 defines the seam + `plan()` (the sizing/resolution the engine will call instead of inline `compute_size`).
`manage()`/`close()` are filled per-instrument as the spine is wired (futures = 2.5-ATR trail; options =
premium exit-at-spike). Kept abstract-optional so a thin engine (FuturesExecution) can ship first.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from avt.intent import Instrument, Intent


@dataclass
class ExecutionPlan:
    """The concrete order derived from an Intent for a specific instrument."""
    instrument: Instrument
    tradeable: str            # what actually trades: underlying symbol (future/equity) OR option tradingsymbol
    direction: str            # "long" | "short"
    quantity: int             # shares (future/equity) OR total option qty (= lots × lot_size)
    entry_ref: float          # price booked at entry: underlying (future/equity) OR option premium (option)
    stop: float               # underlying thesis stop (drives the exit for all instruments)
    lots: Optional[int] = None
    lot_size: Optional[int] = None
    meta: dict = field(default_factory=dict)   # risk_amount/risk_pct, option strike/expiry, cost model, …


class ExecutionEngine(ABC):
    """One per instrument. `plan()` = Intent → ExecutionPlan (sizing + resolve the tradeable + real lots/charges)."""
    instrument: Instrument

    @abstractmethod
    def plan(self, intent: Intent, *, equity: float, config: Any, sizer: Any,
             drawdown_pct: Optional[float] = None) -> Optional[ExecutionPlan]:
        """Turn an Intent into a concrete order for this instrument, or None if unsizable/unavailable.
        `sizer` = the engine's VirtualPositionManager (reused so behaviour is identical); `config` = AVTConfig.
        `equity`/`drawdown_pct` size off a specific capital SLEEVE when the account is partitioned (B.1); default
        (None drawdown) = the shared portfolio values."""
        raise NotImplementedError

    # manage()/close() are added per-instrument when the exit path is wired (M2 futures trail, M3 option spike).
