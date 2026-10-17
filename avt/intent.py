"""
avt.intent — the INTENT contract: the seam between a Strategy and the Capital Operator.

A strategy does NOT place trades. It emits an **Intent** — a proposal ("I see a strong continuation, 91%
confidence, expected to run, I'd prefer a future"). The Capital Operator (`avt.capital`) then decides take / wait /
skip, **which instrument** (it may override the preference — "buy options instead"), and size; an instrument-
specific `ExecutionEngine` (`avt.execution`) executes. This keeps the spine intact:

    Eye → Understanding → Strategy(Template→Instance) → INTENT → Capital Operator → Execution Engine

Templates are frozen; per-user instances carry the strategy_id/template. The Intent is the ONLY thing a template
is allowed to produce — never an order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd


class Instrument(str, Enum):
    FUTURE = "future"
    OPTION = "option"
    EQUITY = "equity"
    CRYPTO = "crypto"


class Hold(str, Enum):
    """How long the move is expected to work — the operator uses this to pick the instrument (a fast intraday run
    suits a near-expiry weekly option; a multi-day move suits a future / farther option)."""
    INTRADAY = "intraday"      # flat by session close (the cascade strategy)
    MULTIDAY = "multiday"
    SWING = "swing"


@dataclass
class Intent:
    """A strategy's PROPOSAL — not an order. Carries the eye's understanding + the structural facts the Capital
    Operator and Execution Engines need. `preferred_instrument` is a hint; the operator may override it."""
    symbol: str
    direction: str                       # "long" | "short"
    confidence: float                    # 0..1 — the strategy's conviction
    entry_ref: float                     # reference UNDERLYING entry price (spot/future)
    structural_stop: float               # UNDERLYING thesis-invalidation level
    atr: float                           # entry-bar ATR (sizing / trailing)
    entry_ts: Optional[pd.Timestamp] = None
    preferred_instrument: Instrument = Instrument.FUTURE
    expected_hold: Hold = Hold.INTRADAY
    break_level: Optional[float] = None
    understanding: str = ""              # the eye's WHY (continuation strength, room, regime) — enriched over time
    invalidation: str = ""               # what breaks the thesis (usually the structural_stop rationale)
    strategy_id: Optional[str] = None
    template: Optional[str] = None
    evidence: dict = field(default_factory=dict)   # raw signal detail (depth, tags, …)

    @property
    def is_long(self) -> bool:
        return self.direction == "long"

    def risk_per_unit(self) -> float:
        """Underlying risk distance (entry → structural stop) — the sizing input for FUTURE/EQUITY execution."""
        return abs(self.entry_ref - self.structural_stop)
