"""
analytics.edge_matrix — strategy × context performance (edge-decay detection).

EDGE COMES FROM AVT, NEVER FROM RAW USER TRADES. A user may take most of their
trades with no strategy at all (revenge/FOMO/overtrading), so a user-trade
aggregate measures TRADER NOISE, not the strategy's edge — it cannot tell "the
edge decayed" from "the user didn't follow it". AVT executes the user's taught
strategies with perfect discipline, so its outcomes are the clean signal.

Input is therefore marketdata.strategy_research.read_samples() (frozen AVT
trades: market state at entry + strategy + AVT outcome, NO behaviour column).
This module only groups + computes stats; output upserts to
StrategyContextPerformance. To compare this clean edge against the user's ACTUAL
results (the execution gap), see analytics.execution_gap.

context_key dimensions derived from each frozen sample:
    regime:<composite>, vol:<low|normal|high>, session:<morning|midday|afternoon>,
    dir:<long|short>
(no streak dimension here — AVT has no emotional state; streak is a user-behaviour
variable that lives in user_research, not in the strategy-edge signal.)

Run EOD. edge_trend = recent-N avg-R vs lifetime avg-R.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_EDGE_DELTA = 0.10  # |last_n_avg_r - avg_r| threshold for strengthening/weakening


def _vol_bucket(atr_pct) -> Optional[str]:
    try:
        v = float(atr_pct)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return "low" if v < 0.012 else ("high" if v > 0.022 else "normal")


def _session_bucket(ts) -> Optional[str]:
    try:
        h = pd.Timestamp(ts).hour + pd.Timestamp(ts).minute / 60.0
    except Exception:
        return None
    if h == 0:
        return None  # daily/unknown intraday time
    if h < 11:
        return "morning"
    if h < 13.5:
        return "midday"
    return "afternoon"


def _contexts(s: pd.Series) -> list[str]:
    ctx: list[str] = []
    regime = s.get("regime_composite")
    if regime and regime != "unknown":
        ctx.append(f"regime:{regime}")
    # Multi-TF regime signature (alignment / counter-trend across 5m/15m/1h/1d) —
    # so edge is learned per "1h-bearish + 5m-bullish" type pattern, not just the
    # single composite regime.
    from marketdata.regime_signature import signature_from_row
    sig = signature_from_row(s)
    if sig:
        ctx.append(f"mtf:{sig}")
    vb = _vol_bucket(s.get("mkt_atr_pct"))
    if vb:
        ctx.append(f"vol:{vb}")
    sb = _session_bucket(s.get("ts"))
    if sb:
        ctx.append(f"session:{sb}")
    streak = s.get("beh_streak")
    if streak is not None and not pd.isna(streak):
        ctx.append(f"streak:{'positive' if streak > 0 else ('negative' if streak < 0 else 'flat')}")
    d = s.get("direction")
    if d:
        ctx.append(f"dir:{d}")
    return ctx


def compute_edge_matrix(start: datetime, end: datetime, last_n: int = 20) -> pd.DataFrame:
    """Aggregate clean AVT strategy-research samples → per (trader, strategy, context) stats."""
    from marketdata.strategy_research import read_samples

    df = read_samples(start, end)
    if df is None or df.empty or "r_multiple" not in df.columns:
        return pd.DataFrame()
    df = df[df["r_multiple"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    # strategy_research already carries the resolved strategy_code (AVT trades the
    # taught strategy explicitly — no attribution guessing needed).
    df = df[df["strategy_code"].notna() & (df["trader_id"].notna())]
    if df.empty:
        return pd.DataFrame()

    # explode each sample across its context dimensions
    long_rows = []
    for _, s in df.iterrows():
        win = 1 if str(s.get("outcome")) == "win" else 0
        for ctx in _contexts(s):
            long_rows.append({
                "trader_id": str(s["trader_id"]),
                "strategy_code": str(s["strategy_code"]),
                "context_key": ctx,
                "ts": pd.Timestamp(s["ts"]),
                "r": float(s["r_multiple"]),
                "win": win,
            })
    if not long_rows:
        return pd.DataFrame()
    long = pd.DataFrame(long_rows)

    out = []
    for (tid, strat, ctx), g in long.groupby(["trader_id", "strategy_code", "context_key"]):
        g = g.sort_values("ts")
        n = len(g)
        avg_r = float(g["r"].mean())
        win_rate = float(g["win"].mean())
        recent = g.tail(last_n)
        ln_avg_r = float(recent["r"].mean())
        ln_win = float(recent["win"].mean())
        diff = ln_avg_r - avg_r
        if n < max(5, last_n // 2):
            trend = "stable"  # too few samples to call a trend
        else:
            trend = "strengthening" if diff > _EDGE_DELTA else ("weakening" if diff < -_EDGE_DELTA else "stable")
        out.append({
            "trader_id": tid, "strategy_code": strat, "context_key": ctx,
            "trade_count": n, "win_rate": round(win_rate, 4),
            "avg_r": round(avg_r, 4), "expectancy": round(avg_r, 4),
            "total_r": round(float(g["r"].sum()), 4),
            "last_n": min(last_n, n),
            "last_n_win_rate": round(ln_win, 4), "last_n_avg_r": round(ln_avg_r, 4),
            "edge_trend": trend,
            "sample_window_start": g["ts"].min(), "sample_window_end": g["ts"].max(),
        })
    return pd.DataFrame(out)


def upsert_matrix(db: Any, matrix: pd.DataFrame) -> int:
    """Upsert rows into strategy_context_performance."""
    if matrix is None or matrix.empty:
        return 0
    from core.models import StrategyContextPerformance
    n = 0
    for _, r in matrix.iterrows():
        row = (
            db.query(StrategyContextPerformance)
            .filter(
                StrategyContextPerformance.trader_id == r["trader_id"],
                StrategyContextPerformance.strategy_code == r["strategy_code"],
                StrategyContextPerformance.context_key == r["context_key"],
            )
            .first()
        )
        if row is None:
            row = StrategyContextPerformance(
                trader_id=r["trader_id"], strategy_code=r["strategy_code"],
                context_key=r["context_key"],
            )
            db.add(row)
        for f in ("trade_count", "win_rate", "avg_r", "expectancy", "total_r",
                  "last_n", "last_n_win_rate", "last_n_avg_r", "edge_trend",
                  "sample_window_start", "sample_window_end"):
            setattr(row, f, r[f])
        n += 1
    db.commit()
    logger.info("edge_matrix: upserted %d (trader,strategy,context) rows", n)
    return n


def recompute(db_factory, lookback_days: int = 365, last_n: int = 20) -> int:
    """EOD entry point: aggregate the AVT strategy-research record and upsert the matrix."""
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    matrix = compute_edge_matrix(start, end, last_n=last_n)
    db = db_factory()
    if hasattr(db, "__next__"):
        db = next(db)
    try:
        return upsert_matrix(db, matrix)
    finally:
        db.close()


if __name__ == "__main__":
    # EOD runner: python -m analytics.edge_matrix  (wire to a daily scheduled task,
    # like the market-data update). Recomputes the edge matrix from the clean AVT
    # strategy-research record. Populate it first via: python -m marketdata.strategy_research
    import logging as _l
    _l.basicConfig(level=_l.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from core.database import SessionLocal
    count = recompute(SessionLocal)
    print(f"edge_matrix: upserted {count} (trader, strategy, context) rows")
