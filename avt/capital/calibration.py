"""
avt.capital.calibration — the Simulation World's OWN validation (docs/avt_adaptive_operator.md §4.ii). An
uncalibrated simulator is WORSE than a fixed rule because it launders a guess as evidence. So before the operator
is allowed to act on a simulation, we check: **do its predictions match reality?**

v0 checks the `simulate_take` estimator on the bench, walk-forward + point-in-time: at each trade, form the TAKE
estimate from ONLY prior analogues, then compare its `prob_profit` to the realized win. Reports a reliability
curve (predicted-bin vs realized-hit-rate) and a single calibration error — plus how often the estimator honestly
said "use_reflex" (thin evidence). This is what `is_trustworthy()` gates on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from avt.capital.simworld import simulate_take


@dataclass
class CalibrationReport:
    n_estimates: int              # decisions where the sim gave an actual estimate (not use_reflex)
    n_reflex: int                 # decisions it honestly deferred to the reflex (thin evidence)
    calib_error: float            # mean |predicted P(profit) − realized hit-rate| across bins (lower = better)
    bins: List[Tuple[float, float, int]]   # (predicted_mid, realized_hit, n) per populated decile
    mean_pred: float
    mean_realized: float

    @property
    def trustworthy(self) -> bool:
        # trustworthy when it has enough estimates and its probabilities track reality within 15pp
        return self.n_estimates >= 30 and self.calib_error <= 0.15

    def summary(self) -> str:
        head = (f"sim calibration — {self.n_estimates} estimates / {self.n_reflex} deferred-to-reflex | "
                f"calib-error {self.calib_error:.1%} | pred {self.mean_pred:.0%} vs realized {self.mean_realized:.0%} "
                f"| trustworthy={self.trustworthy}")
        rows = "\n".join(f"    pred~{m:.0%}  realized {h:.0%}  (n={n})" for m, h, n in self.bins)
        return head + ("\n" + rows if rows else "")


def evaluate(bench: pd.DataFrame) -> CalibrationReport:
    """Walk-forward, point-in-time reliability of simulate_take's prob_profit on the bench."""
    df = bench.sort_values("entry_ts").reset_index(drop=True)
    preds: List[float] = []; hits: List[int] = []; n_reflex = 0
    for _, t in df.iterrows():
        est = simulate_take(df, t["sym"], t["direction"], t["entry_ts"])
        if est.verdict != "estimate":
            n_reflex += 1
            continue
        preds.append(est.prob_profit); hits.append(int(float(t["r_multiple"]) > 0))
    if not preds:
        return CalibrationReport(0, n_reflex, 1.0, [], 0.0, 0.0)
    preds_a = np.array(preds); hits_a = np.array(hits)
    # reliability by decile of predicted P(profit)
    bins: List[Tuple[float, float, int]] = []; errs: List[float] = []
    for lo in np.arange(0.0, 1.0, 0.1):
        m = (preds_a >= lo) & (preds_a < lo + 0.1)
        if m.sum() == 0:
            continue
        pred_mid = float(preds_a[m].mean()); realized = float(hits_a[m].mean())
        bins.append((pred_mid, realized, int(m.sum()))); errs.append(abs(pred_mid - realized))
    return CalibrationReport(
        n_estimates=len(preds), n_reflex=n_reflex,
        calib_error=float(np.mean(errs)) if errs else 1.0, bins=bins,
        mean_pred=float(preds_a.mean()), mean_realized=float(hits_a.mean()),
    )


def is_trustworthy(bench: pd.DataFrame) -> bool:
    """The gate §4.iii consults: if the simulator isn't calibrated on this tape, the operator uses the reflex."""
    return evaluate(bench).trustworthy
