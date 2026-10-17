"""
marketdata.regime_store — multi-timeframe regime state (Layer 2 of the lakehouse).

Computes the market regime across 5m/15m/1h/1d for a symbol, plus a composite,
and stores it point-in-time + immutable in the lake. Records regime TRANSITIONS
(the research goldmine: "what happens after ranging→trending?").

REUSE, not rebuild: regime detection is `strategy_engine.regime_detector.
MarketRegimeDetector.detect()` called once per timeframe. This module only adds
the multi-TF composition + persistence.

QUANT-GRADE:
  • Point-in-time: each timeframe uses only candles up to `as_of` (the detector
    is causal); the snapshot is what was knowable then.
  • Versioned: REGIME_MODEL_VERSION stamped on every row.
  • Immutable/append-only lake history (lake/regime/, lake/regime_transitions/).
  • Serving: feeds the composite into the existing MarketRegimeLog (reused) so
    the dashboard RegimeCard + /regime/current keep working — no new serving table.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from marketdata import lake
from marketdata.candle_store import get_candle_store

logger = logging.getLogger(__name__)

REGIME_MODEL_VERSION = "1.0"

# Timeframe → bars needed (detector needs ≥50) → calendar days to fetch.
_TF_DAYS = {"5m": 7, "15m": 12, "1h": 25, "1d": 200}
_TF_ORDER = ["1d", "1h", "15m", "5m"]   # higher TF wins ties


def _regime_for_tf(symbol: str, tf: str, as_of: datetime,
                   live: bool = False) -> tuple[Optional[str], float]:
    """Run the existing detector on one timeframe. Returns (regime, confidence).
    `live=True` appends today's forming bars so the intraday regime is current."""
    try:
        from strategy_engine.regime_detector import MarketRegimeDetector
        start = as_of - timedelta(days=_TF_DAYS.get(tf, 60))
        df = get_candle_store().get(symbol, tf, start, as_of, live=live)
        if df is None or len(df) < 50:
            return None, 0.0
        analysis = MarketRegimeDetector().detect(df.rename(columns={"ts": "datetime"}))
        regime = analysis.regime.value if hasattr(analysis.regime, "value") else str(analysis.regime)
        return regime, float(analysis.confidence or 0.0)
    except Exception as e:
        logger.debug("regime detect failed %s/%s: %s", symbol, tf, e)
        return None, 0.0


def compute_regime(symbol: str, as_of: Optional[datetime] = None) -> dict:
    """Multi-TF regime snapshot for a symbol, point-in-time as of `as_of`.

    When `as_of` is the present (None/today), each timeframe is computed with
    today's LIVE forming bars so the intraday regime is current; a historical
    `as_of` stays lake-only (no lookahead)."""
    as_of = pd.Timestamp(as_of) if as_of else pd.Timestamp.utcnow()
    use_live = as_of.date() >= date.today()
    snap: dict[str, Any] = {
        "symbol": symbol,
        "ts": as_of,
        "regime_model_version": REGIME_MODEL_VERSION,
    }
    valid: dict[str, tuple[str, float]] = {}
    for tf in ["5m", "15m", "1h", "1d"]:
        regime, conf = _regime_for_tf(symbol, tf, as_of, live=use_live)
        snap[f"regime_{tf}"] = regime
        snap[f"conf_{tf}"] = conf
        if regime and regime != "unknown":
            valid[tf] = (regime, conf)

    # Composite: most common regime across TFs; ties → higher timeframe.
    if valid:
        counts = Counter(r for r, _ in valid.values())
        top = max(counts.values())
        tied = [r for r, c in counts.items() if c == top]
        if len(tied) == 1:
            composite = tied[0]
        else:
            composite = next((valid[tf][0] for tf in _TF_ORDER if tf in valid and valid[tf][0] in tied), tied[0])
        agree = [c for (r, c) in valid.values() if r == composite]
        snap["composite_regime"] = composite
        snap["composite_confidence"] = round(sum(agree) / len(agree), 4) if agree else 0.0
        snap["alignment"] = round(len(agree) / len(valid), 4)
    else:
        snap["composite_regime"] = "unknown"
        snap["composite_confidence"] = 0.0
        snap["alignment"] = 0.0
    return snap


# ── Lake persistence ────────────────────────────────────────────────────

def _regime_dir(symbol: str) -> Path:
    return lake._lake_root() / "regime" / f"symbol={lake._sanitize(symbol)}"


def _transitions_dir() -> Path:
    return lake._lake_root() / "regime_transitions"


def _write_parquet_year(df: pd.DataFrame, directory: Path, key_cols: list[str]) -> None:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    for year, grp in df.groupby(df["ts"].dt.year):
        path = directory / f"{int(year)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            grp = pd.concat([pd.read_parquet(path), grp], ignore_index=True)
        grp = grp.drop_duplicates(subset=key_cols, keep="last").sort_values("ts").reset_index(drop=True)
        grp.to_parquet(path, index=False)


def get_current(symbol: str) -> Optional[dict]:
    """Latest stored regime snapshot for a symbol (serving)."""
    d = _regime_dir(symbol)
    if not d.exists():
        return None
    files = [str(p) for p in d.glob("*.parquet")]
    if not files:
        return None
    row = lake._con().execute(
        "SELECT * FROM read_parquet(?) ORDER BY ts DESC LIMIT 1", [files]
    ).fetchdf()
    return None if row.empty else row.iloc[0].to_dict()


def detect_and_store(symbol: str, as_of: Optional[datetime] = None, db: Any = None) -> dict:
    """
    Compute the multi-TF regime, store it (lake), record a transition if the
    composite changed, and feed the composite to MarketRegimeLog for serving.
    """
    prev = get_current(symbol)
    snap = compute_regime(symbol, as_of)

    # 1. Append to immutable lake history
    _write_parquet_year(pd.DataFrame([snap]), _regime_dir(symbol), key_cols=["ts"])

    # 2. Transition (research goldmine) — only when composite changes
    prev_comp = (prev or {}).get("composite_regime")
    if prev_comp and prev_comp != "unknown" and prev_comp != snap["composite_regime"]:
        trans = {
            "symbol": symbol,
            "ts": snap["ts"],
            "from_regime": prev_comp,
            "to_regime": snap["composite_regime"],
            "from_ts": prev.get("ts"),
            "regime_model_version": REGIME_MODEL_VERSION,
        }
        _write_parquet_year(pd.DataFrame([trans]), _transitions_dir(), key_cols=["symbol", "ts"])
        logger.info("regime transition %s: %s → %s", symbol, prev_comp, snap["composite_regime"])

    # 3. Serving: reuse existing MarketRegimeLog so RegimeCard/regime endpoints show the composite
    if db is not None and snap["composite_regime"] != "unknown":
        try:
            from core.models import MarketRegimeLog
            from core.enums import MarketRegime
            db.add(MarketRegimeLog(
                timestamp=datetime.utcnow(),
                regime=MarketRegime(snap["composite_regime"]),
                confidence=snap["composite_confidence"],
                indicators={
                    "source": "multi_tf",
                    "symbol": symbol,
                    "per_tf": {tf: snap.get(f"regime_{tf}") for tf in ["5m", "15m", "1h", "1d"]},
                    "alignment": snap["alignment"],
                    "model_version": REGIME_MODEL_VERSION,
                },
            ))
            db.commit()
        except Exception as e:
            logger.debug("MarketRegimeLog serving upsert failed: %s", e)
            try:
                db.rollback()
            except Exception:
                pass
    return snap
