"""
intelligence.failure_miner — mine the FAILED episodes for the tag-combinations that OVER-REPRESENT
failure, and surface only the ones whose failure-edge is REAL out-of-sample.

The honest gap this closes ([[episodic]] records failures but nothing actively mines them): the episode
library labels every recognised birth RAN/MODEST/FAILED by realised extension. 66% fail — the trap is the
default. So a bare "this shape failed" is worthless; the question is whether a *specific shape fails MORE
than the 66% floor, reliably*. This mines candidate tag-combos and passes each through the SAME strict
gate as situations (`situation_validator`): split-half (both time-halves), cross-symbol (not one stock),
significance (binomial p<0.05 vs the failure baseline). Only survivors become validated AVOID-patterns the
eye can consult — "this setup matches COUNTER_TREND + ROOM_WALL + WEAK_EFFORT → 84% fail (n=112, p<0.001),
avoid." NOT a rule; a validated bias the reasoner weighs, honestly labelled with its evidence.

Direction is folded in as a DIR_UP/DIR_DOWN tag so the miner discovers whether direction matters.
Temporal (3-year) stability is REPORTED as extra confidence but NOT required — the library is ~1yr today;
it becomes a hard gate for free once coverage/history grows (the running rich harvest widens it).

  PTIS_ENV=test python -m intelligence.failure_miner            # mine FAILED, print + save
  PTIS_ENV=test python -m intelligence.failure_miner --target RAN   # inverse: the reliably-tradeable shapes
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from intelligence.episodic import EpisodeStore, FAILED, RAN
from intelligence.situation_validator import validate_situation, baseline_rates

AVOID_PATH = Path("data/lake/_episodes/avoid_patterns.json")


def load_episode_frame(store: EpisodeStore | None = None) -> pd.DataFrame:
    """The episode library as a validation frame: one row/birth with its outcome + the plot tags
    (direction folded in as a DIR_ tag). `ts` is date-level (imported/harvested births are day-stamped)."""
    eps = (store or EpisodeStore()).load()
    rows = []
    for e in eps:
        if not e.direction:
            continue
        tags = sorted(set(e.tags) | {f"DIR_{e.direction.upper()}"})
        rows.append({"symbol": e.symbol, "ts": pd.Timestamp(e.date), "date": e.date,
                     "direction": e.direction, "outcome": e.outcome, "tags": tags})
    return pd.DataFrame(rows)


def _frequent_combos(df: pd.DataFrame, min_n: int, max_combo: int) -> list[tuple]:
    """Tag-combos (size 1..max_combo) occurring in >= min_n episodes — the only ones worth validating."""
    c: Counter = Counter()
    for tags in df["tags"]:
        ts = sorted(set(tags))
        for r in range(1, max_combo + 1):
            for combo in combinations(ts, r):
                c[combo] += 1
    return [combo for combo, n in c.items() if n >= min_n]


def mine_patterns(df: pd.DataFrame | None = None, *, target: str = FAILED, min_n: int = 60,
                  max_combo: int = 3, min_lift: float = 1.25, alpha: float = 0.05,
                  cross_symbol_min_frac: float = 0.5,
                  split_half_max_rel_delta: float = 0.35) -> list[dict]:
    """Mine + VALIDATE tag-combos on `target` (FAILED = avoid-patterns; RAN = tradeable-patterns).
    Verdict = the CORE out-of-sample gate (sample + split-half + cross-symbol + significance + lift);
    temporal stability is reported as extra confidence, not required (honest given ~1yr of data)."""
    df = load_episode_frame() if df is None else df
    if df.empty or len(df) < min_n:
        return []
    base = baseline_rates(df)
    combos = _frequent_combos(df, min_n, max_combo)
    out: list[dict] = []
    for combo in combos:
        cset = set(combo)
        mask = df["tags"].apply(lambda t: cset <= set(t))
        if int(mask.sum()) < min_n:
            continue
        labeled = df.copy()
        labeled["_cluster"] = 0
        labeled.loc[mask, "_cluster"] = 1
        sit = SimpleNamespace(cluster_id=1, name=" + ".join(combo), outcome_profile={})
        vr = validate_situation(labeled, sit, base, target_outcome=target, min_n=min_n,
                                min_lift=min_lift, alpha=alpha,
                                cross_symbol_min_frac=cross_symbol_min_frac,
                                split_half_max_rel_delta=split_half_max_rel_delta)
        core_ok = (vr.n >= min_n and vr.split_half_ok and vr.cross_symbol_ok
                   and vr.significant and vr.lift >= min_lift)
        if core_ok:
            out.append({
                "combo": list(combo), "target": target, "n": vr.n,
                "rate": vr.situation_rate, "base_rate": vr.baseline_rate, "lift": vr.lift,
                "split_half": list(vr.split_half_rates), "cross_symbol_frac": vr.cross_symbol_frac,
                "p_value": vr.p_value, "temporal_ok": bool(vr.temporal_ok),
                "year_lifts": vr.year_lifts or {},
            })
    # keep only DISTINCT patterns: a longer combo survives only if it adds real edge over its BEST
    # subset (else it's just the simpler pattern with noise bolted on — e.g. PHASE_LATE echoed 15 ways).
    lift_of = {frozenset(r["combo"]): r["lift"] for r in out}
    out.sort(key=lambda r: (r["lift"], r["n"]), reverse=True)
    kept: list[dict] = []
    for r in out:
        rc = frozenset(r["combo"])
        best_sub = max((l for c, l in lift_of.items() if c < rc), default=None)
        if best_sub is not None and r["lift"] <= best_sub + 0.05:
            continue                                   # a simpler subset already captures ~this edge
        kept.append(r)
    return kept


def save_patterns(patterns: list[dict], path: Path = AVOID_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(patterns, indent=2), encoding="utf-8")


_CACHE: dict = {"mtime": None, "patterns": []}


def load_avoid_patterns(path: Path = AVOID_PATH) -> list[dict]:
    """Cached by file mtime — the live eye calls check() every read; don't re-parse JSON each time."""
    if not path.exists():
        return []
    mt = path.stat().st_mtime
    if _CACHE["mtime"] != mt:
        _CACHE["patterns"] = json.loads(path.read_text(encoding="utf-8"))
        _CACHE["mtime"] = mt
    return _CACHE["patterns"]


def check(tags: list[str], direction: str, patterns: list[dict] | None = None) -> list[dict]:
    """Live use: does this birth's signature match any VALIDATED avoid-pattern? Returns the matches
    (worst-first) so the eye/reasoner can WARN — grounded ('84% fail, n=112'), not a veto."""
    patterns = load_avoid_patterns() if patterns is None else patterns
    have = set(tags) | ({f"DIR_{direction.upper()}"} if direction else set())
    hits = [p for p in patterns if set(p["combo"]) <= have]
    hits.sort(key=lambda p: p["lift"], reverse=True)
    return hits


def render(patterns: list[dict], target: str = FAILED) -> str:
    if not patterns:
        return f"No {target} pattern cleared the strict gate (need more coverage/history)."
    verb = "FAIL" if target == FAILED else "RUN"
    lines = [f"{len(patterns)} VALIDATED {target}-patterns (rate vs {'66%-ish' if target==FAILED else 'base'} floor):"]
    for p in patterns:
        star = " ★temporal" if p["temporal_ok"] else ""
        lines.append(
            f"  {' + '.join(p['combo'])}\n"
            f"      {verb} {p['rate']*100:.0f}% vs base {p['base_rate']*100:.0f}% "
            f"(lift {p['lift']}x, n={p['n']}) | split-half {p['split_half'][0]*100:.0f}/{p['split_half'][1]*100:.0f}% "
            f"| cross-sym {p['cross_symbol_frac']*100:.0f}% | p={p['p_value']}{star}")
    return "\n".join(lines)


def main() -> None:
    args = sys.argv[1:]
    target = args[args.index("--target") + 1] if "--target" in args else FAILED
    min_n = int(args[args.index("--min-n") + 1]) if "--min-n" in args else 60
    df = load_episode_frame()
    print(f"episode library: {len(df)} births, {df['symbol'].nunique() if not df.empty else 0} symbols")
    pats = mine_patterns(df, target=target, min_n=min_n)
    print(render(pats, target))
    if target == FAILED:
        save_patterns(pats)
        print(f"saved {len(pats)} avoid-patterns → {AVOID_PATH}")


if __name__ == "__main__":
    main()
