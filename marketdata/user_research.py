"""
marketdata.user_research — the USER behavioural record (decision quality, NOT edge).

Every USER trade becomes ONE frozen, point-in-time sample:

    STATE(market) + STRATEGY + USER-BEHAVIOR   →   OUTCOME

Its purpose is BEHAVIOURAL: did the user follow their strategy, or was this a
revenge / FOMO / overtrading / oversized trade? It powers coaching, discipline
scoring and the execution-gap diagnosis — NOT strategy-edge measurement.

WHY THIS IS NOT THE EDGE SIGNAL: a user may take most of their trades with no
strategy at all, so the aggregate OUTCOME here is contaminated by human noise. We
deliberately keep it OUT of analytics.edge_matrix. Strategy edge is measured from
AVT (marketdata.strategy_research) — perfect-discipline executions of the taught
strategy. The two streams never mix:

    Market Intelligence  ← candles            (marketdata.patterns / regime / events)
    Strategy Intelligence← AVT virtual trades (strategy_research → edge matrix)
    User Intelligence    ← user trades        (THIS module → behaviour/coaching)

This is the user-initiated Replay substrate: it only becomes meaningful once the
user has taught their strategies (so adherence can be judged). It is captured live
via the event bus, and backfilled on demand via marketdata.user_research_backfill.

The behaviour column refers to USER behaviour (streak / trades-today / daily-pnl);
market behaviour is a separate, global concept and never stored here as a variable.

REUSE, not rebuild:
  • market state  → marketdata.regime_store (multi-TF regime) + marketdata.features
  • strategy state→ Trade.declared_strategy_id + strategy_attributions
  • behavior state→ computed from the trader's prior trades + Trade.emotion_pre
  • outcome       → Trade (pnl, r_multiple)

Integration is via the EVENT BUS only (TRADE_ENTERED / TRADE_CLOSED already
published by core/trade_pipeline.py) — no pipeline edits. Best-effort: capture
never breaks trading.

QUANT-GRADE: point-in-time (entry features as-of entry_time), immutable/append-only
(entry frozen; outcome separate), versioned (USER_RESEARCH_VERSION + feature/regime
versions), survivorship-safe (keyed by trade/instrument, keeps everything).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from marketdata import lake

logger = logging.getLogger(__name__)

USER_RESEARCH_VERSION = "1.0"
_MKT_INTERVAL = "1d"   # daily feature snapshot; multi-TF regime covers intraday


def _behavior_state(db: Any, trader_id: str, entry_time: datetime) -> dict:
    """Streak / trades-today / daily-pnl-before from the trader's prior trades."""
    out = {"beh_streak": 0, "beh_trades_today": 0, "beh_daily_pnl_before": 0.0}
    try:
        from core.models import Trade
        from core.enums import TradeStatus
        day_start = datetime(entry_time.year, entry_time.month, entry_time.day)
        today = (
            db.query(Trade)
            .filter(Trade.trader_id == trader_id, Trade.entry_time >= day_start,
                    Trade.entry_time < entry_time)
            .all()
        )
        out["beh_trades_today"] = len(today)
        out["beh_daily_pnl_before"] = float(sum(t.pnl or 0 for t in today))
        recent = (
            db.query(Trade)
            .filter(Trade.trader_id == trader_id, Trade.status == TradeStatus.CLOSED,
                    Trade.entry_time < entry_time)
            .order_by(Trade.entry_time.desc()).limit(10).all()
        )
        streak = 0
        for t in recent:
            if t.pnl is None or t.pnl == 0:
                break
            sign = 1 if t.pnl > 0 else -1
            if streak == 0 or (streak > 0) == (sign > 0):
                streak += sign
            else:
                break
        out["beh_streak"] = streak
    except Exception as e:
        logger.debug("behavior state failed: %s", e)
    return out


def _market_state(symbol: str, entry_time: datetime) -> dict:
    """Multi-TF regime + daily feature snapshot, point-in-time as-of entry_time.

    Shared with marketdata.strategy_research so the context_key space is identical
    for user trades and AVT trades (required for the execution-gap join).
    """
    md: dict[str, Any] = {}
    try:
        from marketdata import regime_store as R
        snap = R.compute_regime(symbol, entry_time)
        md.update({
            "regime_composite": snap.get("composite_regime"),
            "regime_alignment": snap.get("alignment"),
            "regime_model_version": snap.get("regime_model_version"),
            **{f"regime_{tf}": snap.get(f"regime_{tf}") for tf in ["5m", "15m", "1h", "1d"]},
        })
    except Exception as e:
        logger.debug("market regime state failed: %s", e)
    try:
        from marketdata import features as F
        from marketdata.candle_store import get_candle_store
        candles = get_candle_store().get(symbol, _MKT_INTERVAL,
                                         entry_time - timedelta(days=400), entry_time)
        if candles is not None and not candles.empty:
            feat = F.compute_features(candles, symbol, _MKT_INTERVAL)
            row = feat.iloc[-1]
            md.update({
                "feature_version": row.get("feature_version"),
                "mkt_atr_pct": float(row.get("atr_pct") or 0),
                "mkt_rsi": float(row.get("rsi") or 0),
                "mkt_adx": float(row.get("adx") or 0),
                "mkt_rvol": float(row.get("rvol") or 0),
                "mkt_bb_width": float(row.get("bb_width_pct") or 0),
                "mkt_vwap_dist_atr": float(row.get("vwap_dist_atr") or 0),
            })
    except Exception as e:
        logger.debug("market feature state failed: %s", e)
    return md


def _strategy_state(db: Any, trade: Any) -> dict:
    out = {
        "declared_strategy_id": getattr(trade, "declared_strategy_id", None),
        "conviction": float(getattr(trade, "conviction_score", 0) or 0),
    }
    try:
        attr = getattr(trade, "attribution", None)
        if attr is not None:
            out["resolved_strategy_id"] = attr.resolved_strategy_id
            out["confidence_score"] = float(attr.confidence_score or 0)
            out["discipline_score"] = float(attr.discipline_score or 0)
    except Exception:
        pass
    return out


# ── Lake persistence (date-partitioned, immutable) ──

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


def capture_entry(trade: Any, db: Any) -> dict:
    """Freeze the entry sample (market+strategy+user-behavior) to lake/user_research/."""
    entry_time = trade.entry_time or datetime.utcnow()
    sample = {
        "sample_id": str(trade.id),
        "trader_id": str(trade.trader_id) if trade.trader_id else None,
        "symbol": trade.symbol,
        "ts": pd.Timestamp(entry_time),
        "direction": trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction),
        "entry_price": float(trade.entry_price or 0),
        "emotion_pre": getattr(trade, "emotion_pre", None).value if getattr(trade, "emotion_pre", None) else None,
        "user_research_version": USER_RESEARCH_VERSION,
        "captured_at": pd.Timestamp.utcnow(),
    }
    sample.update(_strategy_state(db, trade))
    sample.update(_behavior_state(db, str(trade.trader_id), entry_time))
    sample.update(_market_state(trade.symbol, entry_time))
    _append(pd.DataFrame([sample]), "user_research", key="sample_id")
    logger.info("user_research: captured entry sample %s (%s)", sample["sample_id"][:8], trade.symbol)
    return sample


def fill_outcome(trade: Any, db: Any) -> None:
    """Write the outcome (separate dataset, keyed by sample_id) — no leak into entry."""
    out = {
        "sample_id": str(trade.id),
        "trader_id": str(trade.trader_id) if trade.trader_id else None,
        "ts": pd.Timestamp(trade.exit_time or datetime.utcnow()),
        "pnl": float(trade.pnl or 0),
        "pnl_pct": float(getattr(trade, "pnl_pct", 0) or 0),
        "r_multiple": float(trade.r_multiple or 0),
        "outcome": "win" if (trade.pnl or 0) > 0 else ("loss" if (trade.pnl or 0) < 0 else "breakeven"),
        "holding_seconds": (
            (trade.exit_time - trade.entry_time).total_seconds()
            if trade.exit_time and trade.entry_time else None
        ),
        "filled_at": pd.Timestamp.utcnow(),
    }
    _append(pd.DataFrame([out]), "user_research_outcomes", key="sample_id")
    logger.info("user_research: filled outcome %s (r=%.2f)", out["sample_id"][:8], out["r_multiple"])


def read_samples(start: datetime, end: datetime) -> pd.DataFrame:
    """Joined behavioural view: entry sample LEFT JOIN outcome on sample_id."""
    wdir, odir = _dir("user_research"), _dir("user_research_outcomes")
    wfiles = [str(p) for p in wdir.glob("*.parquet")] if wdir.exists() else []
    if not wfiles:
        return pd.DataFrame()
    ofiles = [str(p) for p in odir.glob("*.parquet")] if odir.exists() else []
    con = lake._con()
    if ofiles:
        return con.execute(
            """
            SELECT w.*, o.pnl, o.r_multiple, o.outcome, o.holding_seconds
            FROM read_parquet(?) w
            LEFT JOIN read_parquet(?) o USING (sample_id)
            WHERE w.ts >= ? AND w.ts <= ? ORDER BY w.ts
            """, [wfiles, ofiles, start, end],
        ).fetchdf()
    return con.execute(
        "SELECT * FROM read_parquet(?) WHERE ts >= ? AND ts <= ? ORDER BY ts",
        [wfiles, start, end],
    ).fetchdf()


# ── Event-bus integration (additive — no pipeline edits) ──

def register_user_research_capture(db_factory) -> None:
    """Subscribe to TRADE_ENTERED/TRADE_CLOSED to capture/fill samples. Call at startup."""
    from core.events import EventType, get_event_bus
    from core.models import Trade

    def _load(trade_id):
        db = db_factory()
        if hasattr(db, "__next__"):
            db = next(db)
        return db

    def on_entered(event):
        try:
            db = _load(event.trade_id)
            try:
                trade = db.get(Trade, event.trade_id)
                if trade is not None:
                    capture_entry(trade, db)
            finally:
                db.close()
        except Exception as e:
            logger.debug("user_research capture_entry failed: %s", e)

    def on_closed(event):
        try:
            db = _load(event.trade_id)
            try:
                trade = db.get(Trade, event.trade_id)
                if trade is not None:
                    fill_outcome(trade, db)
            finally:
                db.close()
        except Exception as e:
            logger.debug("user_research fill_outcome failed: %s", e)

    bus = get_event_bus()
    bus.subscribe(EventType.TRADE_ENTERED, on_entered)
    bus.subscribe(EventType.TRADE_CLOSED, on_closed)
    logger.info("User-research record subscribed to TRADE_ENTERED/TRADE_CLOSED")
