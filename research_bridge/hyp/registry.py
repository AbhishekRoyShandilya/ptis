"""REGISTRATION BEFORE TESTING, enforced (Phase A3, 2026-10-03).

A belief is a spec (YAML in research_bridge/hyp/specs/). Its key is `<id>@<sha8>`, the hash of its canonical content.
  register(spec)          appends ONE row to data/exports/edge_trial_registry.csv with outcome "PENDING(auto:<key>)"
                          (and refreshes docs/research/edge_trial_registry_snapshot.csv). Hand-written rows are never
                          touched; the machine only appends, and only edits its OWN rows.
  assert_registered(spec) raises unless that exact key was registered: an edited spec is a NEW variant, registered again,
                          and the family's variant count goes up (it is visible in the registry).
  open_window(key, name)  the holdout ledger: each confirmation / sealed window of a key can be opened ONCE.
  sync_outcome(key, text) writes the verdict into the key's own row: "AUTO(<key>) <text>".
  log_trial(entry)        every untracked exploratory query (EKB / forward returns without a registered key) is logged.
"""
from __future__ import annotations

import os, json, hashlib, datetime as dt
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPECS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs")
REG = os.path.join(ROOT, "data", "exports", "edge_trial_registry.csv")
SNAP = os.path.join(ROOT, "docs", "research", "edge_trial_registry_snapshot.csv")
VERDICTS = os.path.join(ROOT, "data", "exports", "verdicts")
LEDGER = os.path.join(VERDICTS, "holdout_ledger.json")
TRIALS = os.path.join(VERDICTS, "trial_log.jsonl")
RECORD = os.path.join(ROOT, "docs", "research", "market_mind", "verdicts")   # tracked copy of every verdict card
COLS = ["date", "hypothesis", "mechanism", "falsifier", "thresholds", "variants_planned", "variants_run", "outcome"]


class NotRegistered(RuntimeError):
    pass


class WindowAlreadyOpened(RuntimeError):
    pass


def load_spec(spec):
    if isinstance(spec, dict):
        return spec
    p = spec if os.path.exists(spec) else os.path.join(SPECS, f"{spec}.yaml")
    return yaml.safe_load(open(p, encoding="utf-8"))


def key_of(spec) -> str:
    s = load_spec(spec)
    body = json.dumps({k: v for k, v in s.items() if k not in ("notes",)}, sort_keys=True, default=str)
    return f"{s['id']}@{hashlib.sha256(body.encode()).hexdigest()[:8]}"


def _read(path=None):
    p = path or REG
    return pd.read_csv(p, dtype=str, keep_default_na=False) if os.path.exists(p) else pd.DataFrame(columns=COLS)


def _write(R, path=None):
    R.to_csv(path or REG, index=False)
    if path is None and os.path.isdir(os.path.dirname(SNAP)):
        R.to_csv(SNAP, index=False)


def _rows_for(R, key):
    return R.outcome.str.contains(f"auto:{key}", regex=False) | R.outcome.str.startswith(f"AUTO({key})")


class KilledBeliefRevival(RuntimeError):
    pass


def _idea(spec):
    """what a spec is ABOUT, independent of its parameters: the id, the event generator + event kind, or the
    signals / drivers it tests. Two specs that share any of these test the same idea."""
    s = load_spec(spec); keys = {f"id:{s['id']}"}
    if s.get("generator"):
        kinds = {v.get("kind") for v in (s.get("variants") or [s.get("params") or {}])} - {None}
        keys |= {f"gen:{s['generator']}:{k}" for k in kinds} or {f"gen:{s['generator']}"}
    keys |= {f"sig:{x}" for x in s.get("signals", [])} | {f"drv:{x}" for x in s.get("drivers", [])}
    return keys


def killed_ideas(verdict_dir=None):
    """ideas whose LATEST verdict is killed (nothing survived), from the verdict cards"""
    import glob
    latest = {}
    for f in sorted(glob.glob(os.path.join(verdict_dir or VERDICTS, "*.json"))):
        if f.endswith("holdout_ledger.json"): continue
        v = json.load(open(f)); k = v.get("key")
        order = {"discovery": 0, "confirmation": 1, "sealed": 2}.get(v.get("phase"), 0)
        if k and (k not in latest or order >= latest[k][0]):
            latest[k] = (order, v)
    out = {}
    for k, (_, v) in latest.items():
        sts = [c.get("status") for c in v.get("cards", [])] + [x.get("status") for x in v.get("pairs", []) + v.get("tests", [])]
        if v.get("type") == "descriptive" or not sts or any(x in ("survived_discovery", "confirmed") for x in sts):
            continue
        spec_p = os.path.join(SPECS, f"{v['id']}.yaml")
        ideas = _idea(spec_p) if os.path.exists(spec_p) else {f"id:{v['id']}"}
        for i in ideas: out.setdefault(i, k)
    return out


def register(spec, path=None, verdict_dir=None) -> str:
    """RULE (2026-10-03): a killed belief cannot quietly return under a slightly different parameterisation. A spec
    that tests a killed idea (same id, same event generator / kind, same signal or driver) is refused unless it carries
    `revival: {killed_key, what_is_new}` naming a genuinely NEW mechanism or NEW information - and the registry row
    then says so in plain words."""
    s = load_spec(spec); key = key_of(s); R = _read(path)
    if _rows_for(R, key).any():
        return key
    hit = {i: k for i, k in killed_ideas(verdict_dir).items() if i in _idea(s)}
    rev = s.get("revival") or {}
    if hit and not (rev.get("killed_key") and len(str(rev.get("what_is_new", "")).strip()) >= 40):
        raise KilledBeliefRevival(f"{s['id']} tests an idea already KILLED ({hit}). Register it only with "
                                  "revival: {killed_key, what_is_new} describing a genuinely new mechanism or new information.")
    fam = R.outcome.str.contains(f"auto:{s['id']}@", regex=False) | R.outcome.str.startswith(f"AUTO({s['id']}@")
    variant = int(fam.sum()) + 1
    row = {"date": dt.date.today().isoformat(),
           "hypothesis": (f"REVIVAL of {rev.get('killed_key')} - new: {rev.get('what_is_new')} | " if hit else "")
                         + f"{s['id']} (variant {variant}): {s['hypothesis']}",
           "mechanism": s.get("mechanism", ""), "falsifier": s.get("falsifier", ""),
           "thresholds": json.dumps(s.get("gates", {}), sort_keys=True),
           "variants_planned": s.get("variants_planned", ""), "variants_run": "",
           "outcome": f"PENDING(auto:{key}) registered {dt.datetime.now().isoformat(timespec='seconds')} before any run"}
    R = pd.concat([R, pd.DataFrame([row])], ignore_index=True)
    _write(R, path)
    return key


def assert_registered(spec, path=None) -> str:
    key = key_of(spec)
    if not _rows_for(_read(path), key).any():
        raise NotRegistered(f"{key} is not registered - register the spec first (an edited spec is a new variant)")
    return key


def record_card(fp):
    """keep a TRACKED copy of a verdict card (data/exports is not in git): research results are never lost"""
    import shutil
    os.makedirs(RECORD, exist_ok=True); shutil.copy2(fp, os.path.join(RECORD, os.path.basename(fp)))


def sync_outcome(key, text, variants_run="", path=None):
    R = _read(path); m = _rows_for(R, key)
    if not m.any():
        raise NotRegistered(key)
    prev = R.loc[m, "outcome"].iloc[0]
    hist = prev[len(f"AUTO({key}) "):] if prev.startswith(f"AUTO({key}) ") else ""
    R.loc[m, "outcome"] = f"AUTO({key}) {text}" + (f" || earlier: {hist}" if hist else "")
    if variants_run:
        R.loc[m, "variants_run"] = variants_run
    _write(R, path)


def _ledger(path=None):
    p = path or LEDGER
    return json.load(open(p)) if os.path.exists(p) else {}


def open_window(key, window, path=None):
    """record that `window` of `key` is being opened; refuse a second opening"""
    p = path or LEDGER; L = _ledger(p); k = f"{key}|{window}"
    if k in L:
        raise WindowAlreadyOpened(f"{window} of {key} was already opened at {L[k]}")
    L[k] = dt.datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(L, open(p, "w"), indent=1)


def window_opened(key, window, path=None) -> bool:
    return f"{key}|{window}" in _ledger(path)


def log_trial(entry: dict, path=None):
    p = path or TRIALS
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), **entry}, default=str) + "\n")
