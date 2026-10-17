"""SESSION SEAL — make completed trading sessions IMMUTABLE (2026-09-25).

WHY (proven, not suspected): `CandleStore._ensure` re-fetches any range the coverage ledger does not mark covered, and
a live read's range spans TODAY *and prior days*, so refreshing today's forming bars re-fetches and overwrites earlier
sessions. Demonstrated on 2026-09-25: reading 2026-09-24 alone moved the partition's mtime, and a repair written from
Upstox was silently replaced by Angel rows minutes later.
The damage that caused: replays of the same session returned different answers on different days, and 2026-09-22
changed DECISION between replays (picks ['COALINDIA'] -> picks []). Thresholds, score history, the reconciliation gate
and any future track record are all built on that history, so none of them meant what they appeared to mean.

THE RULE: a session at or before `sealed_through` is FROZEN. Fetched rows for those dates are DROPPED before the
write. Today and any unsealed session still refresh normally, so live behaviour is unchanged.
Repair tools that genuinely need to rewrite history set PTIS_ALLOW_REWRITE=1 - deliberately, loudly, and then they
must re-seal (which records a new fingerprint, so the change is visible rather than silent).

State: data/lake/session_seal.json  {"sealed_through": "YYYY-MM-DD", "sealed_at": ..., "fingerprints": {date: hash}}
"""
from __future__ import annotations
import os, json, hashlib, datetime as dt
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
SEAL_FILE = _ROOT / "data" / "lake" / "session_seal.json"
_CACHE: dict | None = None


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = json.loads(SEAL_FILE.read_text(encoding="utf-8"))
        except Exception:
            _CACHE = {"sealed_through": None, "fingerprints": {}}
    return _CACHE


def invalidate() -> None:
    global _CACHE
    _CACHE = None


def sealed_through():
    """The last date that is frozen, or None when nothing is sealed (the pre-2026-09-25 behaviour)."""
    v = _load().get("sealed_through")
    if not v:
        return None
    try:
        return dt.date.fromisoformat(v)
    except Exception:
        return None


def rewrite_allowed() -> bool:
    return os.environ.get("PTIS_ALLOW_REWRITE") == "1"


def is_sealed(d) -> bool:
    st = sealed_through()
    if st is None or rewrite_allowed():
        return False
    d = d.date() if hasattr(d, "date") else d
    return d <= st


def drop_sealed(df, ts_col: str = "ts"):
    """Remove rows belonging to frozen sessions. Returns (kept_df, n_dropped)."""
    st = sealed_through()
    if st is None or rewrite_allowed() or df is None or len(df) == 0 or ts_col not in df.columns:
        return df, 0
    import pandas as pd
    d = pd.to_datetime(df[ts_col]).dt.date
    keep = d > st
    n = int((~keep).sum())
    return (df[keep] if n else df), n


def fingerprint(symbols, date, interval: str = "1m") -> str:
    """A content hash of one session across the given symbols: the evidence that history has not moved."""
    import pandas as pd
    from marketdata import lake
    h = hashlib.sha256()
    D = pd.Timestamp(date).normalize()
    for s in sorted(symbols):
        d = lake.read_candles(s, interval, D, D + pd.Timedelta(days=1))
        if d is None or not len(d):
            h.update(f"{s}:none".encode()); continue
        x = d.copy()
        x["ts"] = pd.to_datetime(x.ts)
        x = x[x.ts.dt.normalize() == D].sort_values("ts")
        h.update(f"{s}:{len(x)}:".encode())
        for col in ("open", "high", "low", "close", "volume"):
            if col in x.columns:
                h.update(x[col].round(4).to_numpy().tobytes())
    return h.hexdigest()[:16]


def seal(date, symbols, interval: str = "1m") -> dict:
    """Freeze every session up to `date` and record this one's fingerprint."""
    import pandas as pd
    D = pd.Timestamp(date).normalize().date()
    st = _load()
    fp = fingerprint(symbols, date, interval)
    st.setdefault("fingerprints", {})[str(D)] = fp
    prev = st.get("sealed_through")
    st["sealed_through"] = str(max(D, dt.date.fromisoformat(prev))) if prev else str(D)
    st["sealed_at"] = dt.datetime.now().isoformat(timespec="seconds")
    SEAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEAL_FILE.write_text(json.dumps(st, indent=2), encoding="utf-8")
    invalidate()
    return {"date": str(D), "fingerprint": fp, "sealed_through": st["sealed_through"]}


def verify(symbols, dates=None, interval: str = "1m") -> list[dict]:
    """Recompute fingerprints and report any session whose content changed since it was sealed."""
    st = _load(); fps = st.get("fingerprints", {})
    out = []
    for d in sorted(dates or fps.keys()):
        want = fps.get(str(d))
        if not want:
            continue
        got = fingerprint(symbols, d, interval)
        out.append({"date": str(d), "sealed": want, "now": got, "changed": got != want})
    return out
