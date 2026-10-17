"""
avt.execution — instrument-specific Execution Engines. A strategy emits an Intent; the Capital Operator decides
take/wait/instrument/size; the matching ExecutionEngine turns that into a concrete order. The AVT engine
DISPATCHES on the chosen instrument — so `engine.py` never grows an `if option: … elif future: …` and adding a
new instrument (crypto) is a new file, not an engine edit.

  Intent → Capital Operator (chooses instrument) → ExecutionEngine.for_instrument(x) → ExecutionPlan → book
"""
from __future__ import annotations

from avt.intent import Instrument
from avt.execution.base import ExecutionEngine, ExecutionPlan
from avt.execution.futures import FuturesExecution
from avt.execution.options import OptionsExecution

# registry — one instance per instrument; extend by adding a file + registering here (never by editing engine.py)
_REGISTRY = {
    Instrument.FUTURE: FuturesExecution(),
    Instrument.EQUITY: FuturesExecution(instrument=Instrument.EQUITY),   # same underlying sizing/booking as futures
    Instrument.OPTION: OptionsExecution(),
}


def for_instrument(instrument: Instrument) -> ExecutionEngine:
    eng = _REGISTRY.get(instrument)
    if eng is None:
        raise KeyError(f"no ExecutionEngine registered for {instrument} (add avt/execution/<x>.py)")
    return eng


__all__ = ["ExecutionEngine", "ExecutionPlan", "FuturesExecution", "for_instrument"]
