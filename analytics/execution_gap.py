"""
analytics.execution_gap — AVT (clean strategy edge) vs USER (actual results).

This is the diagnosis the three-way separation makes possible. For each
(trader, strategy, context):

    AVT  ORB · trending  →  +0.8R   (what the strategy does, perfect discipline)
    User ORB · trending  →  −0.2R   (what the user actually got)
    ───────────────────────────────
    execution_gap        =  −1.0R   (this is the TRADER, not the strategy)

A large negative gap with a healthy AVT edge means the strategy still works and
the user is leaking R to execution (late entries, moved stops, oversizing,
revenge). A negative AVT edge means the strategy itself is decaying — a different
conversation. Neither can be told from user trades alone, which is the whole point
of keeping the two streams separate:

    strategy edge ← AVT   (marketdata.strategy_research → analytics.edge_matrix)
    actual result ← user  (marketdata.user_research)

Joined here on a SHARED context_key (regime / vol / session / dir — NO streak,
since AVT has no behavioural state), so the comparison is apples-to-apples.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from analytics.edge_matrix import _vol_bucket, _session_bucket

logger = logging.getLogger(__name__)

_LEAK_DELTA = 0.30   # gap (R) below which we flag an execution leak
_WEAK_EDGE = 0.05    # AVT avg_r at/below which the strategy itself looks weak


def _shared_contexts(s: pd.Series) -> list[str]:
    """Context dims common to BOTH AVT and user samples (no behavioural streak)."""
    ctx: list[str] = []
    regime = s.get("regime_composite")
    if regime and regime != "unknown":
        ctx.append(f"regime:{regime}")
    vb = _vol_bucket(s.get("mkt_atr_pct"))
    if vb:
        ctx.append(f"vol:{vb}")
    sb = _session_bucket(s.get("ts"))
    if sb:
        ctx.append(f"session:{sb}")
    d = s.get("direction")
    if d:
        ctx.append(f"dir:{d}")
    return ctx


def _user_strategy(row: pd.Series) -> Optional[str]:
    """User-side strategy identity: resolved attribution > declared (object-safe)."""
    rv = row.get("resolved_strategy_id")
    if rv is not None and not pd.isna(rv):
        return str(rv)
    dv = row.get("declared_strategy_id")
    return str(dv) if (dv is not None and not pd.isna(dv)) else None


def _explode(df: pd.DataFrame, strat_fn) -> pd.DataFrame:
    """Long form: one row per (trader, strategy, shared-context) with r + win."""
    rows = []
    for _, s in df.iterrows():
        strat = strat_fn(s)
        tid = s.get("trader_id")
        if not strat or tid is None or pd.isna(tid):
            continue
        r = s.get("r_multiple")
        if r is None or pd.isna(r):
            continue
        win = 1 if str(s.get("outcome")) == "win" else 0
        for ctx in _shared_contexts(s):
            rows.append({
                "trader_id": str(tid), "strategy_code": str(strat),
                "context_key": ctx, "r": float(r), "win": win,
            })
    return pd.DataFrame(rows)


def _agg(long: pd.DataFrame, label: str) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame(columns=[
            "trader_id", "strategy_code", "context_key",
            f"{label}_n", f"{label}_avg_r", f"{label}_win",
        ])
    g = long.groupby(["trader_id", "strategy_code", "context_key"])
    out = g.agg(n=("r", "size"), avg_r=("r", "mean"), win=("win", "mean")).reset_index()
    return out.rename(columns={"n": f"{label}_n", "avg_r": f"{label}_avg_r", "win": f"{label}_win"})


def _verdict(row: pd.Series) -> str:
    avt_r, gap = row["avt_avg_r"], row["gap_r"]
    if avt_r <= _WEAK_EDGE:
        return "strategy_decay"          # the edge itself is weak/gone
    if gap <= -_LEAK_DELTA:
        return "execution_leak"          # strategy works, user leaks R
    if gap >= _LEAK_DELTA:
        return "outperforming"           # user beats clean baseline (rare; check sizing/luck)
    return "tracking"                    # user roughly matches the strategy


def compute_execution_gap(start: datetime, end: datetime, min_n: int = 5) -> pd.DataFrame:
    """
    Per (trader, strategy, context): AVT edge vs user result + gap + verdict.
    Only contexts with >= min_n samples on BOTH sides are reported (the gap is
    meaningless otherwise).
    """
    from marketdata import strategy_research as S
    from marketdata import user_research as U

    avt = _agg(_explode(S.read_samples(start, end), lambda s: s.get("strategy_code")), "avt")
    usr = _agg(_explode(U.read_samples(start, end), _user_strategy), "user")
    if avt.empty or usr.empty:
        return pd.DataFrame()

    m = avt.merge(usr, on=["trader_id", "strategy_code", "context_key"], how="inner")
    m = m[(m["avt_n"] >= min_n) & (m["user_n"] >= min_n)].copy()
    if m.empty:
        return pd.DataFrame()

    m["gap_r"] = (m["user_avg_r"] - m["avt_avg_r"]).round(4)
    m["gap_win"] = (m["user_win"] - m["avt_win"]).round(4)
    for c in ("avt_avg_r", "user_avg_r", "avt_win", "user_win"):
        m[c] = m[c].round(4)
    m["verdict"] = m.apply(_verdict, axis=1)
    return m.sort_values("gap_r").reset_index(drop=True)


def execution_gap_report(db_factory=None, lookback_days: int = 365, min_n: int = 5) -> list[dict]:
    """Convenience wrapper → list of dict rows (for API/coaching surfaces)."""
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = compute_execution_gap(start, end, min_n=min_n)
    return df.to_dict("records") if not df.empty else []


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = execution_gap_report()
    if not rows:
        print("execution_gap: no overlapping (AVT, user) contexts yet — "
              "need AVT trades (strategy_research) AND user trades (user_research).")
    else:
        print(f"execution_gap: {len(rows)} (trader, strategy, context) comparisons")
        for r in rows[:20]:
            print(f"  {r['strategy_code'][:8]} {r['context_key']:<22} "
                  f"AVT={r['avt_avg_r']:+.2f}R  user={r['user_avg_r']:+.2f}R  "
                  f"gap={r['gap_r']:+.2f}R  [{r['verdict']}]")
