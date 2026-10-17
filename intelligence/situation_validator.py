"""
intelligence.situation_validator — the STRICT gate (the partner's critical part).

A discovered situation is only TRUSTED if its outcome edge is REAL, not a "Cluster
17" coincidence. Three out-of-sample gates (mirrors improvement_lab/edge_amplifier's
split-half + avt/self_improver's significance):

  • SPLIT-HALF   : the edge holds in BOTH time-halves (not a one-period fluke)
  • CROSS-SYMBOL : it holds across the symbol cross-section (not one stock's quirk)
  • SIGNIFICANCE : the outcome rate beats baseline significantly (binomial, p<0.05)
  • SAMPLE       : enough occurrences

Only situations passing all four are stored. PTIS then says "Situation [X]
repeatedly precedes Y — validated (split-half Δ=…, in N/M symbols, p=…)", never
"Cluster 17 found." The target outcome is the situation's highest-LIFT outcome.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    target_outcome: str
    n: int
    situation_rate: float
    baseline_rate: float
    lift: float
    split_half_ok: bool
    split_half_rates: tuple
    cross_symbol_ok: bool
    cross_symbol_frac: float
    significant: bool
    p_value: float
    temporal_ok: bool = False
    year_lifts: dict = None            # {year: lift} — survives regime change?
    validated: bool = False
    note: str = ""


def baseline_rates(samples: pd.DataFrame) -> dict:
    return samples["outcome"].value_counts(normalize=True).to_dict()


def validate_situation(labeled: pd.DataFrame, situation, base: dict, *,
                       min_n: int = 200, split_half_max_rel_delta: float = 0.35,
                       cross_symbol_min_frac: float = 0.6, min_lift: float = 1.25,
                       alpha: float = 0.05, target_outcome: str = None) -> ValidationResult:
    """Validate ONE situation on `target_outcome` (default: its highest-lift outcome).
    Out-of-sample by construction. The supervised rare-miner pins its intended target."""
    bars = labeled[labeled["_cluster"] == situation.cluster_id]
    n = len(bars)
    # target = the override, else the outcome this situation most ELEVATES vs baseline
    target = target_outcome or max(
        situation.outcome_profile,
        key=lambda o: situation.outcome_profile[o] / max(base.get(o, 1e-9), 1e-9))
    b = float(base.get(target, 0.0))
    rate = float((bars["outcome"] == target).mean())
    lift = rate / max(b, 1e-9)

    # split-half by TIME
    bs = bars.sort_values("ts"); h = len(bs) // 2
    r1 = float((bs.iloc[:h]["outcome"] == target).mean()) if h else 0.0
    r2 = float((bs.iloc[h:]["outcome"] == target).mean()) if h else 0.0
    sh_delta = abs(r1 - r2) / max((r1 + r2) / 2, 1e-9)
    split_half_ok = (sh_delta <= split_half_max_rel_delta) and (r1 > b) and (r2 > b)

    # cross-symbol: rate > baseline in >= frac of symbols
    persym = bars.groupby("symbol").apply(lambda g: float((g["outcome"] == target).mean()),
                                          include_groups=False)
    cross_frac = float((persym > b).mean()) if len(persym) else 0.0
    cross_ok = cross_frac >= cross_symbol_min_frac

    # significance: binomial (one-sided, greater) vs baseline
    try:
        from scipy.stats import binomtest
        k = int((bars["outcome"] == target).sum())
        p = float(binomtest(k, n, max(b, 1e-9), alternative="greater").pvalue) if n > 0 else 1.0
    except Exception:
        p = 1.0
    significant = p < alpha

    # TEMPORAL STABILITY (partner's gold-standard): the lift must hold in EVERY year
    # with enough data — a situation that only worked in one regime is overfit.
    ly = labeled.copy(); ly["_yr"] = pd.to_datetime(ly["ts"]).dt.year
    base_yr = ly.groupby("_yr").apply(lambda g: float((g["outcome"] == target).mean()),
                                      include_groups=False)
    by = bars.copy(); by["_yr"] = pd.to_datetime(by["ts"]).dt.year
    year_lifts = {}
    for yr, gy in by.groupby("_yr"):
        if len(gy) < 50 or yr not in base_yr.index or base_yr[yr] <= 0:
            continue
        year_lifts[int(yr)] = round(float((gy["outcome"] == target).mean()) / base_yr[yr], 2)
    temporal_ok = len(year_lifts) >= 3 and min(year_lifts.values()) >= min_lift

    validated = (n >= min_n and split_half_ok and cross_ok and significant
                 and temporal_ok and lift >= min_lift)
    note = (f"[{situation.name}] {target}: {rate:.1%} vs base {b:.1%} (lift {lift:.2f}x) | "
            f"split-half {r1:.1%}/{r2:.1%} {'OK' if split_half_ok else 'FAIL'} | "
            f"cross-symbol {cross_frac:.0%} {'OK' if cross_ok else 'FAIL'} | "
            f"p={p:.4f} {'OK' if significant else 'FAIL'} | "
            f"yearly-lift {list(year_lifts.values())} {'OK' if temporal_ok else 'FAIL'} | n={n} -> "
            f"{'VALIDATED' if validated else 'rejected'}")
    return ValidationResult(target, n, round(rate, 4), round(b, 4), round(lift, 2),
                            split_half_ok, (round(r1, 4), round(r2, 4)), cross_ok,
                            round(cross_frac, 3), significant, round(p, 6),
                            temporal_ok, year_lifts, validated, note)


def validate_all(labeled: pd.DataFrame, situations, base: dict = None, **kw):
    base = base or baseline_rates(labeled)
    return [(s, validate_situation(labeled, s, base, **kw)) for s in situations]
