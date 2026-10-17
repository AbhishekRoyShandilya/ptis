"""
marketdata.strategy_research — the Strategy Research record (the CLEAN edge signal).

Strategy edge MUST NOT be measured from raw user trades. A user may take 80 of
100 trades with no strategy at all — revenge, FOMO, overtrading, "experiments" —
so the aggregate outcome of their trades measures TRADER NOISE, not the strategy's
edge. If ORB shows −0.3R across a user's trades we cannot tell whether the edge
decayed or the user simply didn't follow it.

AVT removes that ambiguity: it executes the user's TAUGHT strategies with perfect
discipline and zero emotion (no fear / greed / revenge), so AVT's closed trades
are the noise-free record of what the STRATEGY actually does in each market
context. That is the only honest substrate for edge-decay detection.

    STATE(market) + STRATEGY   →   AVT OUTCOME        (NO behavior column)

This feeds analytics.edge_matrix. User trades feed marketdata.user_research
instead (behavioural analysis only). The two streams NEVER mix:

    Market Intelligence  ← candles            (marketdata.patterns / regime / events)
    Strategy Intelligence← AVT virtual trades (THIS module → edge matrix)
    User Intelligence    ← user trades        (user_research → behaviour/coaching)

REUSE: the market state is frozen with the SAME point-in-time enrichment used by
user_research (regime_store + features), so the context_key is identical on both
sides — which is what makes the execution-gap join (AVT expectancy vs user
expectancy per strategy×context) apples-to-apples.

Note: AVT virtual trades are already per-user (AVT trades that user's strategies),
so this is clean PER-USER STRATEGY intelligence — distinct from global, candle-only
market intelligence. It stays on the user side of the wall.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from marketdata import lake

logger = logging.getLogger(__name__)

STRATEGY_RESEARCH_VERSION = "1.0"


def _outcome(pnl_r: Optional[float]) -> str:
    r = float(pnl_r or 0)
    return "win" if r > 0 else ("loss" if r < 0 else "breakeven")


def capture_avt_trade(trade: Any, db: Any) -> Optional[dict]:
    """
    Freeze one CLOSED AVT virtual trade into a clean strategy-research sample:
    market state (point-in-time as-of entry) + strategy + AVT outcome. No behaviour.

    `trade` is an avt.models.VirtualTrade (or any object exposing id / trader_id /
    strategy_id / symbol / direction / entry_time / pnl_r).
    """
    entry_time = getattr(trade, "entry_time", None)
    if entry_time is None or getattr(trade, "pnl_r", None) is None:
        return None  # only closed, resolved trades carry an edge signal

    # SAME market-state enrichment as user_research → identical context_key space.
    from marketdata.user_research import _market_state

    sample = {
        "sample_id": str(trade.id),
        "trader_id": str(trade.trader_id) if getattr(trade, "trader_id", None) else None,
        "symbol": trade.symbol,
        "ts": pd.Timestamp(entry_time),
        "direction": (
            trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction)
        ),
        "strategy_code": str(trade.strategy_id),
        "strategy_version": getattr(trade, "strategy_version", None),
        # AVT outcome — this IS the clean edge signal (no human noise)
        "r_multiple": float(trade.pnl_r),
        "pnl_pct": float(getattr(trade, "pnl_pct", 0) or 0),
        "outcome": _outcome(trade.pnl_r),
        "exit_type": getattr(trade, "exit_type", None),
        "strategy_research_version": STRATEGY_RESEARCH_VERSION,
        "captured_at": pd.Timestamp.utcnow(),
    }
    sample.update(_market_state(trade.symbol, entry_time))
    _append(pd.DataFrame([sample]), "strategy_research", key="sample_id")
    return sample


# ── Lake persistence (date-partitioned, immutable; mirrors user_research) ──

def _dir(name: str) -> Path:
    return lake._lake_root() / name


def _append(df: pd.DataFrame, name: str, key: str) -> None:
    df = df.copy()
    df["_year"] = pd.to_datetime(df["ts"]).dt.year
    for year, grp in df.groupby("_year"):
        grp = grp.drop(columns=["_year"])
        path = _dir(name) / f"{int(year)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            grp = pd.concat([pd.read_parquet(path), grp], ignore_index=True)
        grp = grp.drop_duplicates(subset=[key], keep="last")
        grp.to_parquet(path, index=False)


def read_samples(start: datetime, end: datetime) -> pd.DataFrame:
    """All clean strategy-research samples in [start, end] (entry+outcome in one row)."""
    sdir = _dir("strategy_research")
    files = [str(p) for p in sdir.glob("*.parquet")] if sdir.exists() else []
    if not files:
        return pd.DataFrame()
    return lake._con().execute(
        "SELECT * FROM read_parquet(?) WHERE ts >= ? AND ts <= ? ORDER BY ts",
        [files, start, end],
    ).fetchdf()


# ── AVT → strategy-research backfill (the clean edge substrate) ──

def build_from_avt(db, trader_id: Optional[str] = None, since_days: int = 1825) -> dict:
    """
    Replay AVT's CLOSED virtual trades → strategy_research. Idempotent (dedupes on
    sample_id). This is what populates the edge-decay matrix with a NOISE-FREE
    signal. Requires AVT to have run the user's strategies over history first.
    """
    from avt.repository import AVTTradeRepository

    since = datetime.utcnow() - timedelta(days=since_days)
    repo = AVTTradeRepository(db)
    trades = repo.get_closed_trades(since=since, limit=100000, trader_id=trader_id)

    captured = errors = 0
    for t in trades:
        try:
            if capture_avt_trade(t, db) is not None:
                captured += 1
        except Exception as e:
            errors += 1
            logger.warning("strategy_research capture failed for AVT trade %s: %s", t.id, e)

    logger.info("strategy_research: %d AVT samples captured (%d errors)", captured, errors)
    return {"avt_trades": len(trades), "captured": captured, "errors": errors}


def main(argv=None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Build the strategy-research record from AVT virtual trades (clean edge)"
    )
    p.add_argument("--trader", default=None, help="restrict to one trader_id (default: all)")
    p.add_argument("--since-days", type=int, default=1825, help="lookback window (default 5y)")
    p.add_argument("--no-edge-matrix", action="store_true", help="skip the edge-matrix recompute")
    args = p.parse_args(argv)

    from core.database import SessionLocal
    db = SessionLocal()
    try:
        stats = build_from_avt(db, trader_id=args.trader, since_days=args.since_days)
    finally:
        db.close()
    print(f"strategy_research: {stats['captured']} AVT samples captured "
          f"({stats['errors']} errors) from {stats['avt_trades']} AVT trades")

    if not args.no_edge_matrix:
        from analytics.edge_matrix import recompute
        n = recompute(SessionLocal)
        print(f"edge_matrix: upserted {n} (trader, strategy, context) rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
