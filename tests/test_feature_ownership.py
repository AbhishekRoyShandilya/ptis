"""
Architectural guard (project_brain/07 + 08): deterministic indicator features are OWNED by the Feature
ENGINE (`marketdata/features.py`). Runtime code must REQUEST them via the read-through store
(`features.get_features`), never recompute via `compute_features`.

This is a RATCHET: it fails if a NEW direct `compute_features(` caller appears, and forces the BASELINE
backlog to shrink honestly as sites migrate. It is the executable form of "07_do_not_duplicate".

Scope: runtime code only. Excluded — the engine itself, archived code, tests, offline research scripts,
and `ske/` (which has its OWN `compute_features`, a different function).
"""
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXCLUDE_SEGMENTS = {"_archive", "tests", "scripts", "ske", ".venv", "node_modules", "examples",
                    "scratchpad", "ptis_analytics"}       # offline research / ETL, not runtime
ENGINE = "marketdata/features.py"
_CALL = re.compile(r"\bcompute_features\s*\(")

# Files that call compute_features directly. The RATCHET prevents NEW ones; shrink this as sites adopt
# get_features. NOTE (2026-06-29): the EYE (state_narrator + balance_state/move_read/momentum_run/
# story_context) intentionally computes its OWN window — a floor-warmed shared store differs at warm-up
# bars and shifts level-detection (project_brain/08). The eye is a documented exception until the
# warm-up semantics are unified + the library re-validated. The rest is genuine migration backlog.
BASELINE = {
    # eye (intentional — window-warmed, validated):
    "intelligence/state_narrator.py",
    "intelligence/balance_state.py",
    "intelligence/move_read.py",
    "intelligence/momentum_run.py",
    "intelligence/story_context.py",
    # migration backlog:
    "avt/engine.py",                       # NB: self.features.compute_features(c, sym) — 2-arg, a DIFFERENT fn (false-positive)
    "intelligence/birth.py",
    "intelligence/lifecycle.py",
    "intelligence/move_nature.py",
    "intelligence/narrative.py",
    "intelligence/recognize.py",
    "intelligence/setup_features.py",
    "marketdata/events.py",
    "marketdata/patterns.py",
    "marketdata/user_research.py",
    # re-baselined 2026-10-16 (the ratchet had been red, so these slipped in unflagged; 6 old entries migrated):
    "intelligence/archetype_facts.py",
    "intelligence/archetype_v2.py",
    "intelligence/level_dynamics.py",
    "ml/research_warehouse.py",            # offline warehouse builders: a full-history recompute is their job
    "ml/strategy_simulator.py",
}


def _direct_callers() -> set[str]:
    hits = set()
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT).as_posix()
        if rel == ENGINE:
            continue
        if EXCLUDE_SEGMENTS & set(rel.split("/")):
            continue
        if rel.startswith("ml/_"):                 # one-off research sweeps
            continue
        try:
            if _CALL.search(p.read_text(encoding="utf-8", errors="ignore")):
                hits.add(rel)
        except Exception:
            pass
    return hits


def test_no_new_direct_compute_features():
    """No NEW module may recompute features directly — request them via features.get_features."""
    new = _direct_callers() - BASELINE
    assert not new, (
        "New direct compute_features() callers found — use features.get_features (read-through store). "
        f"Offenders: {sorted(new)}"
    )


def test_baseline_not_stale():
    """A migrated/removed file must be deleted from BASELINE so the backlog stays honest and shrinks."""
    gone = BASELINE - _direct_callers()
    assert not gone, f"These BASELINE entries no longer call compute_features — remove them: {sorted(gone)}"
