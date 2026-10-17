"""
analytics.strategy_symbol_perf — per-(strategy, stock[, regime]) ranking baseline.

The cold-start fix: a freshly-taught strategy has no history, so the live
opportunity ranker can't rank it. This aggregates the benchmark replay's clean
record into per-(strategy, symbol) and per-(strategy, symbol, regime) statistics —
"which stocks does ORB work best on, in which regime?" — to SEED that ranking.

SOURCE INVARIANT — BENCHMARK REPLAY ONLY: input is marketdata.strategy_research
(frozen-v1.0 AVT executions), the SAME clean source as analytics.edge_matrix.
NEVER user trades — user mistakes must not pollute strategy ranking.

Confidence layer: a 3-trade/100% cell must not outrank a 150-trade/58% cell, so
each row carries effective_sample_size + confidence_score = n/(n+K). Consumers
rank by confidence-weighted expectancy (expectancy × confidence_score), which
shrinks thin-sample cells toward zero.

Grain: for each strategy_research trade we emit an OVERALL row (regime='*') and a
per-regime row. market_state is reserved (column exists, populated later); for now
it is '' (not yet computed). strategy_version is carried through ('1.0' for
benchmark) for future per-version intelligence.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_CONF_K = 20.0   # smoothing constant: confidence = n / (n + K)
_OVERALL = "*"   # regime sentinel for the all-regime (symbol-level) row


def _confidence(n: int, k: float = _CONF_K) -> float:
    return round(n / (n + k), 4) if n > 0 else 0.0


def compute_symbol_matrix(start: datetime, end: datetime, last_n: int = 20) -> pd.DataFrame:
    """Aggregate strategy_research → per (trader, strategy, symbol, regime) stats."""
    from marketdata.strategy_research import read_samples

    df = read_samples(start, end)
    if df is None or df.empty or "r_multiple" not in df.columns:
        return pd.DataFrame()
    df = df[df["r_multiple"].notna()].copy()
    df = df[df["strategy_code"].notna() & df["trader_id"].notna() & df["symbol"].notna()]
    if df.empty:
        return pd.DataFrame()

    # explode each trade into an OVERALL row + a per-regime row + a per-MTF-signature row
    from marketdata.regime_signature import signature_from_row
    rows = []
    for _, s in df.iterrows():
        regime = s.get("regime_composite")
        version = s.get("strategy_version") or "1.0"
        buckets = [_OVERALL]
        if regime and not pd.isna(regime) and regime != "unknown":
            buckets.append(str(regime))
        # multi-TF regime signature bucket (stored as a context value in `regime`)
        sig = signature_from_row(s)
        if sig:
            buckets.append(f"mtf:{sig}")
        win = 1 if str(s.get("outcome")) == "win" else 0
        for rg in buckets:
            rows.append({
                "trader_id": str(s["trader_id"]),
                "strategy_code": str(s["strategy_code"]),
                "symbol": str(s["symbol"]),
                "regime": rg,
                "strategy_version": str(version),
                "ts": pd.Timestamp(s["ts"]),
                "r": float(s["r_multiple"]),
                "win": win,
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame()

    out = []
    for (tid, strat, sym, rg, ver), g in long.groupby(
        ["trader_id", "strategy_code", "symbol", "regime", "strategy_version"]
    ):
        g = g.sort_values("ts")
        n = len(g)
        avg_r = float(g["r"].mean())
        recent = g.tail(last_n)
        out.append({
            "trader_id": tid, "strategy_code": strat, "symbol": sym,
            "regime": rg, "market_state": "", "strategy_version": ver,
            "trade_count": n,
            "win_rate": round(float(g["win"].mean()), 4),
            "avg_r": round(avg_r, 4), "expectancy": round(avg_r, 4),
            "total_r": round(float(g["r"].sum()), 4),
            "last_n": min(last_n, n),
            "last_n_avg_r": round(float(recent["r"].mean()), 4),
            "effective_sample_size": float(n),
            "confidence_score": _confidence(n),
            "sample_window_start": g["ts"].min(), "sample_window_end": g["ts"].max(),
        })
    return pd.DataFrame(out)


def upsert_matrix(db: Any, matrix: pd.DataFrame) -> int:
    """Upsert rows into strategy_symbol_performance."""
    if matrix is None or matrix.empty:
        return 0
    from core.models import StrategySymbolPerformance
    n = 0
    for _, r in matrix.iterrows():
        row = (
            db.query(StrategySymbolPerformance)
            .filter(
                StrategySymbolPerformance.trader_id == r["trader_id"],
                StrategySymbolPerformance.strategy_code == r["strategy_code"],
                StrategySymbolPerformance.symbol == r["symbol"],
                StrategySymbolPerformance.regime == r["regime"],
                StrategySymbolPerformance.market_state == r["market_state"],
                StrategySymbolPerformance.strategy_version == r["strategy_version"],
            )
            .first()
        )
        if row is None:
            row = StrategySymbolPerformance(
                trader_id=r["trader_id"], strategy_code=r["strategy_code"],
                symbol=r["symbol"], regime=r["regime"], market_state=r["market_state"],
                strategy_version=r["strategy_version"],
            )
            db.add(row)
        for f in ("trade_count", "win_rate", "avg_r", "expectancy", "total_r",
                  "last_n", "last_n_avg_r", "effective_sample_size", "confidence_score",
                  "sample_window_start", "sample_window_end"):
            setattr(row, f, r[f])
        n += 1
    db.commit()
    logger.info("strategy_symbol_perf: upserted %d (trader,strategy,symbol,regime) rows", n)
    return n


def recompute(db_factory, lookback_days: int = 1825, last_n: int = 20) -> int:
    """EOD entry point: aggregate the symbol matrix and upsert."""
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    matrix = compute_symbol_matrix(start, end, last_n=last_n)
    db = db_factory()
    if hasattr(db, "__next__"):
        db = next(db)
    try:
        return upsert_matrix(db, matrix)
    finally:
        db.close()


if __name__ == "__main__":
    # Runner: python -m analytics.strategy_symbol_perf  (chained by avt.replay after
    # the benchmark replay populates strategy_research). Builds the per-(strategy,
    # stock) ranking baseline from the clean AVT record.
    import logging as _l
    _l.basicConfig(level=_l.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from core.database import SessionLocal
    count = recompute(SessionLocal)
    print(f"strategy_symbol_perf: upserted {count} (trader, strategy, symbol, …) rows")
