"""
intelligence.scientist — the SCIENTIST + revisable PRINCIPLE LIBRARY (the "learning from reasoning" layer).

The eye PERCEIVES, the reasoner EXPLAINS one situation, the failure_miner mines episode tag-combos for AVOID
patterns. What was missing (user, 2026-07-21) is the layer ABOVE reasoning: a component that takes a MARKET
PRINCIPLE stated as a measurable hypothesis ("fresh breakouts continue more than exhausted ones"), TESTS it against
history under the project's strict discipline, RECORDS the verdict with its evidence, and UPDATES that belief as new
evidence arrives — so the system's understanding of the market changes from evidence, not from a fresh fit each time.

This is a GENERALISATION of `failure_miner` (which mines categorical tag-combos through `situation_validator`) to any
CONTINUOUS state variable + direction + outcome, over any event universe — plus the persistent, revisable library the
miner lacks. It does NOT replace them: the miner still finds combos; the scientist tests principles and remembers.

Discipline (same spirit as situation_validator: sealed OOS, cross-symbol, robustness, significance):
  · SEALED SPLIT   train = early dates (+ 2/3 symbols when >=3); test = later dates AND held-out symbols
  · EFFECT         favorable-decile minus unfavorable-decile outcome rate on the TEST set (per hypothesised direction)
  · CI             bootstrap 95% CI on that effect (must exclude 0 for HELD)
  · ROBUST         same sign in BOTH test date-halves, and after dropping the biggest-contributing symbol
  · VERDICT        HELD / REJECTED / INCONCLUSIVE — never "found a cluster"; always with n, effect, CI, robustness

An honest negative is a first-class result: REJECTED principles are stored too (so the loop doesn't re-litigate them).

  from intelligence.scientist import Hypothesis, test_hypothesis, PrincipleLibrary
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_LIB = Path("data/scientist/principles.json")
MIN_EFFECT = 0.03          # a proportion lift below this is "no practical effect" regardless of significance
N_BOOT = 2000


@dataclass
class Hypothesis:
    """A market principle stated so it can be measured and falsified."""
    id: str
    statement: str                       # human-readable claim
    variable: str                        # the measurable state column
    direction: str                       # "higher" (higher var -> more outcome) | "lower"
    outcome: str                         # the outcome column (binary 0/1, or continuous)
    universe: str                        # which event set, e.g. "index_break"
    condition: Optional[dict] = None     # {col: [op, thr]} gate — the lineage of a refined hypothesis
    parent_id: Optional[str] = None      # set when this refines a REJECTED parent (reflection)


def _apply_condition(ev: pd.DataFrame, condition: Optional[dict]) -> pd.DataFrame:
    if not condition:
        return ev
    m = pd.Series(True, index=ev.index)
    ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
           "<": lambda a, b: a < b, "==": lambda a, b: a == b}
    for col, (op, thr) in condition.items():
        m &= ops[op](ev[col], thr)
    return ev[m]


def _signed_lift(d: pd.DataFrame, var: str, outcome: str, direction: str) -> float:
    """favorable-extreme minus unfavorable-extreme outcome (signed so + = supports the hypothesis)."""
    if d[var].nunique() <= 2:                                   # binary state variable -> group compare
        hi = d[d[var] == d[var].max()][outcome].mean()
        lo = d[d[var] == d[var].min()][outcome].mean()
    else:
        s = d.sort_values(var); q = max(1, len(s) // 10)
        lo = s.iloc[:q][outcome].mean(); hi = s.iloc[-q:][outcome].mean()
    return (hi - lo) if direction == "higher" else (lo - hi)


def test_hypothesis(events: pd.DataFrame, h: Hypothesis, min_effect: float = MIN_EFFECT) -> "Principle":
    ev = _apply_condition(events, h.condition).dropna(subset=[h.variable, h.outcome]).copy()
    prov = f"n_after_gate={len(ev)}"
    if len(ev) < 200:
        return Principle.from_hypothesis(h, verdict="INCONCLUSIVE", n=len(ev), lift=float("nan"),
                                         ci=(float("nan"), float("nan")), baseline=float("nan"),
                                         split_half_ok=False, cross_symbol_ok=False,
                                         detail=f"too few events ({len(ev)}<200); {prov}", data_window="")
    ev["_d"] = pd.to_datetime(ev["date"]) if "date" in ev else pd.NaT
    has_sym = "sym" in ev and ev["sym"].nunique() >= 3
    cut = ev["_d"].quantile(0.6)
    if has_sym:
        hold = set(sorted(ev["sym"].unique())[::3])
        te = ev[(ev["_d"] >= cut) & (ev["sym"].isin(hold))]
    else:
        te = ev[ev["_d"] >= cut]
    if len(te) < 100:
        te = ev[ev["_d"] >= cut]                                # relax cross-symbol if it starves the test
    baseline = te[h.outcome].mean()
    lift = _signed_lift(te, h.variable, h.outcome, h.direction)
    # bootstrap CI on the signed lift
    boots = np.empty(N_BOOT); idx = np.arange(len(te)); rng = np.random.default_rng(11)
    for b in range(N_BOOT):
        boots[b] = _signed_lift(te.iloc[rng.choice(idx, len(idx), replace=True)], h.variable, h.outcome, h.direction)
    ci = (float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5)))
    # robustness: both test date-halves keep the sign; drop the biggest symbol keeps the sign
    mid = te["_d"].median()
    lh = _signed_lift(te[te["_d"] < mid], h.variable, h.outcome, h.direction)
    rh = _signed_lift(te[te["_d"] >= mid], h.variable, h.outcome, h.direction)
    split_half_ok = bool(lh > 0 and rh > 0) if lift > 0 else False
    if "sym" in te and te["sym"].nunique() > 1:
        big = te["sym"].value_counts().index[0]
        cross_symbol_ok = bool(_signed_lift(te[te["sym"] != big], h.variable, h.outcome, h.direction) > 0) if lift > 0 else False
    else:
        cross_symbol_ok = True                                 # single-symbol universe: N/A
    # verdict
    if lift >= min_effect and ci[0] > 0 and split_half_ok and cross_symbol_ok:
        verdict = "HELD"
    elif lift < min_effect or lift <= 0:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"                                # real point-effect but not robust / CI spans 0
    detail = (f"favorable-vs-unfavorable {h.outcome} lift {lift:+.3f} (base {baseline:.3f}); "
              f"CI[{ci[0]:+.3f},{ci[1]:+.3f}]; halves {lh:+.3f}/{rh:+.3f}; {prov}")
    dw = f"{ev['_d'].min().date()} -> {ev['_d'].max().date()}" if ev["_d"].notna().any() else ""
    return Principle.from_hypothesis(h, verdict=verdict, n=len(te), lift=float(lift), ci=ci, baseline=float(baseline),
                                     split_half_ok=split_half_ok, cross_symbol_ok=cross_symbol_ok,
                                     detail=detail, data_window=dw)


@dataclass
class Principle:
    id: str
    statement: str
    variable: str
    direction: str
    outcome: str
    universe: str
    verdict: str
    n: int
    lift: float
    ci: tuple
    baseline: float
    split_half_ok: bool
    cross_symbol_ok: bool
    detail: str
    data_window: str
    condition: Optional[dict] = None
    parent_id: Optional[str] = None
    tested_at: str = ""
    history: list = field(default_factory=list)

    @classmethod
    def from_hypothesis(cls, h: Hypothesis, **kw) -> "Principle":
        return cls(id=h.id, statement=h.statement, variable=h.variable, direction=h.direction,
                   outcome=h.outcome, universe=h.universe, condition=h.condition, parent_id=h.parent_id,
                   tested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), **kw)

    def line(self) -> str:
        mark = {"HELD": "[HELD]", "REJECTED": "[rej ]", "INCONCLUSIVE": "[?   ]"}[self.verdict]
        return f"{mark} {self.statement}  ->  lift {self.lift:+.3f} CI[{self.ci[0]:+.2f},{self.ci[1]:+.2f}] n={self.n}"


class PrincipleLibrary:
    """Persistent, revisable store of tested market principles — the belief the system updates from evidence."""

    def __init__(self, path: Path = DEFAULT_LIB):
        self.path = Path(path)
        self._items: dict[str, Principle] = {}
        if self.path.exists():
            for d in json.loads(self.path.read_text()):
                d["ci"] = tuple(d.get("ci", (float("nan"), float("nan"))))
                self._items[d["id"]] = Principle(**d)

    def record(self, p: Principle) -> Principle:
        """Add or UPDATE a principle. Re-testing an existing id appends the prior verdict to its history
        (the belief-update trail) rather than silently overwriting it."""
        old = self._items.get(p.id)
        if old is not None:
            p.history = list(old.history) + [{"tested_at": old.tested_at, "verdict": old.verdict,
                                              "lift": old.lift, "n": old.n}]
        self._items[p.id] = p
        return p

    def get(self, pid: str) -> Optional[Principle]:
        return self._items.get(pid)

    def query(self, universe: str = None, verdict: str = None) -> list[Principle]:
        return [p for p in self._items.values()
                if (universe is None or p.universe == universe) and (verdict is None or p.verdict == verdict)]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(p) for p in self._items.values()], indent=2, default=str))

    def summary(self) -> str:
        by = {v: len(self.query(verdict=v)) for v in ("HELD", "REJECTED", "INCONCLUSIVE")}
        return f"library: {len(self._items)} principles  (HELD {by['HELD']} / REJECTED {by['REJECTED']} / ? {by['INCONCLUSIVE']})"
