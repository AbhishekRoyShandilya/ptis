"""
avt.capital.simworld — the SIMULATION WORLD (v0). The operator's imagination: "what will happen if I do this?"

THE LOAD-BEARING PROPERTY (docs/avt_adaptive_operator.md §4): this is a *calibrated belief, not an oracle*.
Claude Code's tests ARE reality; this is a MODEL of the future built from historical analogues. So every estimate
carries **confidence + analogue sample size**, and when the evidence is thin the verdict is **use_reflex** — defer
to the deterministic baseline policy, NEVER invent a number. That single rule is what separates a Capital Operator
from a leveraged Martingale.

v0 SCOPE (honest): it estimates the outcome distribution of TAKING a candidate from point-in-time realized
analogues (the bench trades before `as_of`). Managing an OPEN position (hold / increase / reduce / close) needs a
forward-PATH model (MAE/MFE + DayStory intrabar paths) — that is Phase 2 — so `simulate_manage` honestly returns
`use_reflex` today rather than faking it. Later this grows to consume `intelligence.analog_engine.forecast`
(market-conditioned forward distributions) and the counterfactual store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

MIN_ANALOGUES = 20          # below this, the estimate is not trustworthy → use the reflex
_CONF_FULL_N = 60           # sample size at which the size-driven confidence saturates


@dataclass
class OutcomeDistribution:
    action: str                       # take | wait | skip | hold | increase | reduce | close
    n_analogues: int
    mean_r: float
    median_r: float
    p10_r: float
    p90_r: float
    prob_profit: float
    confidence: float                 # 0..1 — from sample size, discounted by dispersion
    verdict: str                      # 'estimate' | 'use_reflex'
    evidence: List[str] = field(default_factory=list)

    def line(self) -> str:
        v = "USE-REFLEX" if self.verdict == "use_reflex" else "estimate"
        return (f"{self.action:<9} n={self.n_analogues:>3}  meanR {self.mean_r:+.2f}  "
                f"P(profit) {self.prob_profit:.0%}  p10..p90 [{self.p10_r:+.1f},{self.p90_r:+.1f}]  "
                f"conf {self.confidence:.2f}  → {v}")


def _confidence(r: np.ndarray) -> float:
    n = len(r)
    if n == 0:
        return 0.0
    size_term = min(1.0, n / _CONF_FULL_N)
    disp = float(np.std(r))
    disp_term = 1.0 / (1.0 + disp)            # wide outcome spread → less confident
    return round(size_term * disp_term, 3)


def _dist(action: str, r: np.ndarray, evidence: List[str]) -> OutcomeDistribution:
    n = len(r)
    if n == 0:
        return OutcomeDistribution(action, 0, 0, 0, 0, 0, 0, 0.0, "use_reflex",
                                   evidence + ["no analogues yet (cold start) → reflex"])
    conf = _confidence(r)
    verdict = "estimate" if n >= MIN_ANALOGUES else "use_reflex"
    if verdict == "use_reflex":
        evidence = evidence + [f"only {n} analogues (<{MIN_ANALOGUES}) → reflex"]
    return OutcomeDistribution(
        action=action, n_analogues=n,
        mean_r=float(np.mean(r)), median_r=float(np.median(r)),
        p10_r=float(np.percentile(r, 10)), p90_r=float(np.percentile(r, 90)),
        prob_profit=float(np.mean(r > 0)), confidence=conf, verdict=verdict, evidence=evidence,
    )


def simulate_take(bench: pd.DataFrame, symbol: str, direction: str,
                  as_of: pd.Timestamp) -> OutcomeDistribution:
    """Outcome distribution of TAKING this candidate, from POINT-IN-TIME analogues (bench trades that had already
    closed before `as_of`). Prefers same symbol+direction; falls back to all-symbol same-direction if thin —
    reporting which, so the reasoner knows the specificity."""
    past = bench[pd.to_datetime(bench["exit_ts"]) < pd.Timestamp(as_of)]
    same = past[(past["sym"] == symbol) & (past["direction"] == direction)]["r_multiple"].to_numpy(float)
    if len(same) >= MIN_ANALOGUES:
        return _dist("take", same, [f"{len(same)} prior {symbol} {direction} analogues (point-in-time)"])
    # broaden to all-symbol same-direction (lower specificity, honestly flagged)
    broad = past[past["direction"] == direction]["r_multiple"].to_numpy(float)
    ev = [f"only {len(same)} {symbol} {direction} analogues → broadened to {len(broad)} all-symbol {direction}"]
    return _dist("take", broad, ev)


def simulate_manage(action: str) -> OutcomeDistribution:
    """Managing an OPEN position (hold/increase/reduce/close). v0 has no forward-PATH model (Phase 2: MAE/MFE +
    DayStory paths + analog_engine.forecast), so it honestly defers to the reflex rather than fabricate."""
    return OutcomeDistribution(action, 0, 0, 0, 0, 0, 0, 0.0, "use_reflex",
                               ["forward-path model is Phase 2 → manage decisions use the reflex (2.5-ATR trail)"])


def decide_candidate(bench: pd.DataFrame, symbol: str, direction: str, as_of) -> dict:
    """The candidate action set {take, wait, skip} for one decision point. TAKE is estimated; WAIT needs the
    opportunity-cost base-rate (Phase 2) and SKIP is the null — both return use_reflex in v0."""
    as_of = pd.Timestamp(as_of)
    take = simulate_take(bench, symbol, direction, as_of)
    wait = OutcomeDistribution("wait", 0, 0, 0, 0, 0, 0, 0.0, "use_reflex",
                               ["opportunity-cost base-rate is Phase 2 → wait uses the reflex"])
    skip = OutcomeDistribution("skip", 0, 0, 0, 0, 0, 0, 0.0, "estimate", ["null action"])
    return {"take": take, "wait": wait, "skip": skip}
