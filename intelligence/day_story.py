"""
intelligence.day_story — Layer 2 of the lakehouse: the EYE's per-day narration, PERSISTED ONCE.

WHY (the shared spine): `narrate_session` (the eye's day-story) is the expensive, SHARED computation BOTH
the strategy backtester (strategy-vs-day) and the behavioral replay (user-vs-day) need. It was recomputed
every time (the teaching backtest caps at 60 sessions because `simulate_session` re-narrates ~10s/session),
and the overnight harvest ALREADY narrates every session but keeps only the *episodes*. This layer persists
the compact per-bar `StateStep` story so both readers reuse it — the 60-cap dies, and no rebuild.

Sibling of `marketdata.features` (Layer-1 input): lake/daystory/type=.../symbol=.../interval=.../{year}.parquet,
one row per bar. The heavy per-bar `memory` dict is dropped (recognition-carrying fields tags/state/move_ext
are kept; session-level facts are re-derivable). The reasoner verdict is layered ON TOP on demand (verdict
cache) — Layer 2 is FACTS. This is also the practical form of the "Layer-2 incremental eye": narrate once,
never recompute; the resumable/streaming narrator (live) writes into the SAME store one bar at a time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from marketdata import lake, instruments

logger = logging.getLogger(__name__)
DAY_STORY_VERSION = 1

_COLS = ["ts", "date", "state", "substate", "changed", "birth", "price", "atr", "move_ext", "tags", "why",
         # ── RICH numeric state (NaN/"" unless narrated with rich=True) ────────────────────────────────
         # These were computed by the eye and DISCARDED here — which is why every downstream study
         # (move_funnel.WHY_PHRASES, expansion_phase1's 32 features) substring-matched English prose to
         # recover facts that existed as floats one layer up. Carrying them through is the unlock.
         "room_atr", "n_near", "energy", "fuel", "loc", "leg_ext_atr", "leg_bars", "in_run", "rsi",
         "d_res_atr", "d_sup_atr", "prev_high", "prev_low", "daily_ctx", "vwap_dist", "rvol_now", "bbw_now"]

_RICH_NUM = ("room_atr", "leg_ext_atr", "leg_bars", "in_run", "rsi", "d_res_atr", "d_sup_atr",
             "prev_high", "prev_low", "vwap_dist", "rvol_now", "bbw_now")
_RICH_STR = ("energy", "fuel", "loc", "daily_ctx")


def steps_to_frame(steps) -> pd.DataFrame:
    """Compact one-row-per-bar frame from narrate_session `steps`."""
    rows = []
    for s in steps or []:
        ts = pd.Timestamp(s.ts)
        rows.append({
            "ts": ts,
            "date": ts.normalize().date().isoformat(),
            "state": s.state or "",
            "substate": getattr(s, "substate", "") or "",
            "changed": getattr(s, "changed", "") or "",
            "birth": getattr(s, "birth", "") or "",
            "price": float(s.price) if s.price == s.price else None,
            "atr": float(s.atr) if s.atr == s.atr else None,
            "move_ext": float(s.move_ext) if getattr(s, "move_ext", float("nan")) == getattr(s, "move_ext", float("nan")) else None,
            "tags": json.dumps(list(getattr(s, "tags", []) or [])),
            "why": json.dumps(list(getattr(s, "why", []) or [])),
        })
        r = rows[-1]
        for k in _RICH_NUM:
            v = getattr(s, k, float("nan"))
            r[k] = float(v) if v == v else None
        r["n_near"] = int(getattr(s, "n_near", 0) or 0)
        for k in _RICH_STR:
            r[k] = getattr(s, k, "") or ""
    return pd.DataFrame(rows, columns=_COLS)


def _dir(inst_type: str, symbol: str, interval: str) -> Path:
    return (lake._lake_root() / "daystory"
            / f"type={lake._sanitize(inst_type)}"
            / f"symbol={lake._sanitize(symbol)}"
            / f"interval={interval}")


def _files(symbol: str, interval: str) -> list[str]:
    root = lake._lake_root() / "daystory"
    if not root.exists():
        return []
    return [str(p) for p in root.glob(f"type=*/symbol={lake._sanitize(symbol)}/interval={interval}/*.parquet")]


def write_frame(symbol: str, interval: str, df: pd.DataFrame) -> int:
    """Idempotent year-partitioned write of a (possibly multi-session) frame (merge on ts, keep last).
    The harvest builds one frame for ALL a symbol's sessions and writes ONCE (avoids O(sessions²) I/O)."""
    if df is None or df.empty:
        return 0
    inst_type, _ = instruments.classify(symbol)
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    total = 0
    for year, grp in df.groupby(df["ts"].dt.year):
        path = _dir(inst_type, symbol, interval) / f"{int(year)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            grp = pd.concat([pd.read_parquet(path), grp], ignore_index=True)
        grp = grp.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)
        grp.to_parquet(path, index=False)
        total += len(grp)
    return total


def write_day_story(symbol: str, interval: str, steps) -> int:
    """Persist one session's steps (safe to re-harvest)."""
    return write_frame(symbol, interval, steps_to_frame(steps))


def read_day_story(symbol: str, interval: str, date) -> pd.DataFrame:
    """The persisted per-bar story for ONE session (symbol, date). Empty if not harvested yet."""
    files = _files(symbol, interval)
    if not files:
        return pd.DataFrame(columns=_COLS)
    d = pd.Timestamp(date).normalize().date().isoformat()
    out = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            sel = df[df["date"] == d]
            if not sel.empty:
                out.append(sel)
        except Exception:
            continue
    if not out:
        return pd.DataFrame(columns=_COLS)
    return pd.concat(out, ignore_index=True).sort_values("ts").reset_index(drop=True)


def has_day_story(symbol: str, interval: str, date) -> bool:
    return not read_day_story(symbol, interval, date).empty
