"""
ml.big_move_experiments — pre-packaged Scientist experiments for big-move research.

Provides:
  1. A library of INTUITION-DRIVEN hypotheses (from the user's trading knowledge)
  2. An experiment runner that tests hypotheses against the Research Warehouse
  3. A bridge from ML Miner discoveries → Scientist verification

Every hypothesis is stated so it can be measured and falsified:
  - variable: the column in the warehouse to test
  - direction: "higher" or "lower" (higher var → more RAN)
  - outcome: the outcome column (ran_binary)
  - condition: optional gate (e.g., only births with rvol > 1.5)

Philosophy: ML-discovered hypotheses are ALWAYS tested through the Scientist. Only HELD
principles enter the Brain Library. REJECTED ones are stored honestly (no re-litigation).

Usage:
    from ml.big_move_experiments import run_all_experiments, INTUITION_HYPOTHESES

    # Test all intuition-driven hypotheses
    results = run_all_experiments()

    # Test ML-discovered hypotheses
    results = run_experiments_from_miner()

    # CLI
    python -m ml.big_move_experiments --test-all
    python -m ml.big_move_experiments --from-miner
    python -m ml.big_move_experiments --library-summary
"""
from __future__ import annotations

import logging
from pathlib import Path

from intelligence.scientist import Hypothesis, test_hypothesis, PrincipleLibrary

logger = logging.getLogger(__name__)


# ── INTUITION-DRIVEN HYPOTHESES (from trading experience) ─────────────────────
# These are the user's beliefs stated as falsifiable hypotheses. The Scientist
# will tell us which ones hold and which don't — honestly.

INTUITION_HYPOTHESES = [
    # ── Structure before the break ──
    Hypothesis(
        id="bbw_squeeze_runs",
        statement="Births from tighter Bollinger Bands (compression) produce larger forward extension",
        variable="bbw_at_birth",
        direction="lower",  # lower BBW = more compressed = bigger move
        outcome="ran_binary",
        universe="all_births",
    ),
    Hypothesis(
        id="bbw_rank_squeeze_runs",
        statement="Births when BBW is at its lowest percentile (squeeze) run more",
        variable="bbw_rank",
        direction="lower",  # lower rank = rarer compression = bigger coil release
        outcome="ran_binary",
        universe="all_births",
    ),

    # ── Energy at the break ──
    Hypothesis(
        id="rvol_ignition_runs",
        statement="Births with higher relative volume on the ignition bar run more",
        variable="rvol_at_birth",
        direction="higher",
        outcome="ran_binary",
        universe="all_births",
    ),
    Hypothesis(
        id="body_atr_runs",
        statement="Births with larger ignition bar body (relative to ATR) run more",
        variable="ignition_body_atr",
        direction="higher",
        outcome="ran_binary",
        universe="all_births",
    ),
    Hypothesis(
        id="strong_effort_runs",
        statement="Births tagged as STRONG_EFFORT produce more RAN outcomes",
        variable="strong_effort",
        direction="higher",
        outcome="ran_binary",
        universe="all_births",
    ),

    # ── Room after the break ──
    Hypothesis(
        id="room_clear_runs",
        statement="Births with more room to the next opposing level (higher room_atr) run more",
        variable="room_atr",
        direction="higher",
        outcome="ran_binary",
        universe="all_births",
    ),
    Hypothesis(
        id="n_near_kills",
        statement="Births with more structural levels within 2 ATR ahead (n_near) run less",
        variable="n_near",
        direction="lower",  # fewer obstacles = better
        outcome="ran_binary",
        universe="all_births",
    ),

    # ── Regime alignment ──
    Hypothesis(
        id="with_trend_runs",
        statement="With-trend births run more than counter-trend births",
        variable="with_trend",
        direction="higher",  # 1 = with trend
        outcome="ran_binary",
        universe="all_births",
    ),
    Hypothesis(
        id="adx_trend_strength_runs",
        statement="Births in stronger trending conditions (higher ADX) run more",
        variable="adx_at_birth",
        direction="higher",
        outcome="ran_binary",
        universe="all_births",
    ),

    # ── Session timing ──
    Hypothesis(
        id="morning_runs_more",
        statement="Morning births (bars_since_open < 30) run more than afternoon births",
        variable="bars_since_open",
        direction="lower",  # fewer bars = earlier in session
        outcome="ran_binary",
        universe="all_births",
    ),

    # ── VWAP positioning ──
    Hypothesis(
        id="vwap_confirmation_runs",
        statement="Births with price further above VWAP (for up) produce more RAN outcomes",
        variable="vwap_dist_atr",
        direction="higher",
        outcome="ran_binary",
        universe="all_births",
    ),

    # ── Interaction hypotheses (multi-condition) ──
    Hypothesis(
        id="squeeze_plus_vol",
        statement="Compression (BBW rank < 0.3) + volume surge (rvol > 1.5) → more RAN",
        variable="rvol_at_birth",
        direction="higher",
        outcome="ran_binary",
        universe="all_births",
        condition={"bbw_rank": ["<", 0.3]},
    ),
    Hypothesis(
        id="with_trend_clear_room",
        statement="With-trend births with clear room (room_atr > 3) run significantly more",
        variable="room_atr",
        direction="higher",
        outcome="ran_binary",
        universe="all_births",
        condition={"with_trend": [">=", 1]},
    ),
    Hypothesis(
        id="strong_break_into_wall_fails",
        statement="Even strong-effort births into ROOM_WALL (room_atr < 0.5) fail more",
        variable="room_atr",
        direction="higher",
        outcome="ran_binary",
        universe="all_births",
        condition={"strong_effort": [">=", 1]},
    ),

    # ── MFE as continuous outcome (regression-style) ──
    Hypothesis(
        id="rvol_drives_extension",
        statement="Higher rvol at birth leads to larger peak extension in ATR",
        variable="rvol_at_birth",
        direction="higher",
        outcome="peak_ext_atr",
        universe="all_births",
    ),
    Hypothesis(
        id="compression_drives_extension",
        statement="Lower BBW at birth (tighter compression) leads to larger peak extension",
        variable="bbw_at_birth",
        direction="lower",
        outcome="peak_ext_atr",
        universe="all_births",
    ),
]


# ── EXPERIMENT RUNNER ─────────────────────────────────────────────────────────
def _prepare_warehouse_for_scientist(df) -> dict:
    """Prepare warehouse DataFrame for test_hypothesis — needs 'date' and 'sym' columns."""
    df = df.copy()
    if "date" in df and "sym" not in df:
        df["sym"] = df.get("symbol", "")
    return df


def run_all_experiments(hypotheses: list[Hypothesis] | None = None,
                        library: PrincipleLibrary | None = None,
                        save: bool = True) -> list:
    """Test all hypotheses against the Research Warehouse.

    Returns list of Principle objects with verdicts (HELD / REJECTED / INCONCLUSIVE).
    """
    from ml.research_warehouse import load_warehouse

    if hypotheses is None:
        hypotheses = INTUITION_HYPOTHESES
    if library is None:
        library = PrincipleLibrary()

    df = load_warehouse()
    df = _prepare_warehouse_for_scientist(df)

    results = []
    for h in hypotheses:
        logger.info("Testing: [%s] %s", h.id, h.statement)
        try:
            p = test_hypothesis(df, h)
            library.record(p)
            results.append(p)
            logger.info("  → %s  lift=%+.3f  CI=[%+.3f, %+.3f]  n=%d",
                         p.verdict, p.lift, p.ci[0], p.ci[1], p.n)
        except Exception as e:
            logger.warning("  → FAILED: %s", e)

    if save:
        library.save()
        logger.info("Library saved: %s", library.summary())

    return results


def run_experiments_from_miner(top_n: int = 10, library: PrincipleLibrary | None = None,
                                save: bool = True) -> list:
    """Run the ML Miner, discover hypotheses, and test them through the Scientist."""
    from ml.research_warehouse import load_warehouse
    from ml.ml_miner import Miner

    df = load_warehouse()
    miner = Miner(df)
    hypotheses = miner.discover_hypotheses(top_n=top_n)

    logger.info("ML Miner discovered %d hypotheses — sending to Scientist", len(hypotheses))
    return run_all_experiments(hypotheses=hypotheses, library=library, save=save)


def experiment_report(results: list | None = None) -> str:
    """Generate a human-readable report of experiment results."""
    lines = ["═" * 70, "BIG MOVE EXPERIMENT RESULTS", "═" * 70]

    if results is None:
        lib = PrincipleLibrary()
        results = list(lib._items.values())

    held = [p for p in results if p.verdict == "HELD"]
    rejected = [p for p in results if p.verdict == "REJECTED"]
    inconclusive = [p for p in results if p.verdict == "INCONCLUSIVE"]

    lines.append(f"\nTotal: {len(results)} tested  "
                 f"| HELD: {len(held)}  | REJECTED: {len(rejected)}  | INCONCLUSIVE: {len(inconclusive)}")
    lines.append("")

    if held:
        lines.append("─ HELD (verified principles — enter Brain Library) ─")
        for p in sorted(held, key=lambda p: -p.lift):
            lines.append(f"  ✓ {p.statement}")
            lines.append(f"    lift {p.lift:+.3f}  CI[{p.ci[0]:+.3f},{p.ci[1]:+.3f}]  n={p.n}")
        lines.append("")

    if rejected:
        lines.append("─ REJECTED (honest negatives — stored, not re-litigated) ─")
        for p in sorted(rejected, key=lambda p: p.lift):
            lines.append(f"  ✗ {p.statement}")
            lines.append(f"    lift {p.lift:+.3f}  CI[{p.ci[0]:+.3f},{p.ci[1]:+.3f}]  n={p.n}")
        lines.append("")

    if inconclusive:
        lines.append("─ INCONCLUSIVE (need more data or further investigation) ─")
        for p in inconclusive:
            lines.append(f"  ? {p.statement}  (n={p.n})")

    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Big Move Experiments — Scientist hypothesis testing")
    ap.add_argument("--test-all", action="store_true", help="Test all intuition-driven hypotheses")
    ap.add_argument("--from-miner", action="store_true", help="Run ML Miner → discover → test via Scientist")
    ap.add_argument("--report", action="store_true", help="Print experiment results report")
    ap.add_argument("--library-summary", action="store_true", help="Print Brain Library summary")
    ap.add_argument("--top", type=int, default=10, help="Top N ML-discovered hypotheses")
    args = ap.parse_args()

    if args.test_all:
        results = run_all_experiments()
        print("\n" + experiment_report(results))
    elif args.from_miner:
        results = run_experiments_from_miner(top_n=args.top)
        print("\n" + experiment_report(results))
    elif args.report:
        print(experiment_report())
    elif args.library_summary:
        lib = PrincipleLibrary()
        print(lib.summary())
        for p in lib.query():
            print(p.line())
    else:
        ap.print_help()
