"""
marketdata.patterns — the analog corpus (Layer 1-2 of the forward intelligence engine).

For every bar across lake history we store an immutable row capturing the STRUCTURE
and what it BECAME + what HAPPENED:

    fingerprint(12d) · regime · market_state · next_market_state ·
    forward_return · mae · mfe · resolution_time

So the engine can answer "this structure is forming → historically it transitioned
to X and produced Y" — evidence-based, no training. Built from the 40M-bar lake;
needs no user trades.

QUANT-GRADE:
  • Point-in-time: fingerprint / regime / market_state at bar i use ONLY data ≤ i
    (derived from the causal feature store). next_market_state + forward outcomes
    use only post-i candles (the outcome IS the future — intentional, isolated).
  • Versioned (feature_version), immutable append-only lake (lake/patterns/).
  • ONE fingerprint function (fingerprint_from_features) shared by the corpus AND
    the live query — so analog similarity is apples-to-apples.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata import lake, features as F

logger = logging.getLogger(__name__)

# Structural state taxonomy (composed from features — NOT a parallel system).
MARKET_STATES = ["compression", "expansion", "trend_up", "trend_down",
                 "exhaustion", "reversal", "range"]

# Corpus columns. The 12 fingerprint dims are fp_<name> (F.FINGERPRINT_DIMS).
FP_COLS = [f"fp_{d}" for d in F.FINGERPRINT_DIMS]
CORPUS_COLS = (
    ["symbol", "interval", "ts"] + FP_COLS +
    ["regime", "market_state", "next_market_state",
     "forward_return", "mae", "mfe", "resolution_time", "feature_version"]
)


def fingerprint_from_features(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized 12-dim canonical fingerprint derived from the enriched feature
    store (F.compute_features output). Same dim names/order as the discovery
    fingerprint; used by BOTH the corpus and the live query for consistency.
    Each dim is normalized to ~[0,1].
    """
    def clip01(s):
        return s.clip(0, 1).fillna(0.5)

    atr_mean = feat["atr_pct"].rolling(20, min_periods=5).mean()
    fp = pd.DataFrame(index=feat.index)
    fp["fp_body_ratio"] = clip01(feat["body_ratio"])
    fp["fp_upper_wick_ratio"] = clip01(feat["upper_wick_ratio"])
    fp["fp_lower_wick_ratio"] = clip01(feat["lower_wick_ratio"])
    fp["fp_volume_surge"] = clip01(feat["rvol"] / 5.0)
    fp["fp_atr_expansion"] = clip01((feat["atr_pct"] / atr_mean.replace(0, np.nan)) / 3.0)
    fp["fp_momentum_score"] = clip01((feat["mom_5"].abs() / feat["atr_pct"].replace(0, np.nan)) / 4.0)
    # consolidation / range_compression: low BB width => high (use rolling percentile)
    bb_rank = feat["bb_width_pct"].rolling(60, min_periods=20).rank(pct=True)
    fp["fp_consolidation"] = clip01(1.0 - bb_rank)
    fp["fp_range_compression"] = clip01(1.0 - bb_rank)
    # trend alignment: +1 aligned / 0 counter / 0.5 flat (from supertrend + EMA stack)
    up = (feat["supertrend_dir"] == 1) & (feat["ema20"] > feat["ema50"])
    dn = (feat["supertrend_dir"] == -1) & (feat["ema20"] < feat["ema50"])
    fp["fp_trend_alignment"] = np.where(up, 1.0, np.where(dn, 0.0, 0.5))
    fp["fp_rsi_norm"] = clip01(feat["rsi"] / 100.0)
    fp["fp_vwap_norm"] = clip01((feat["vwap_dist_atr"] + 3.0) / 6.0)
    fp["fp_rel_volume"] = clip01(feat["rvol"] / 5.0)
    return fp[FP_COLS]


def classify_market_state(feat: pd.DataFrame) -> pd.Series:
    """Vectorized structural label per bar, composed from the feature columns."""
    atr_mean = feat["atr_pct"].rolling(20, min_periods=5).mean()
    bb_rank = feat["bb_width_pct"].rolling(60, min_periods=20).rank(pct=True)
    adx = feat["adx"]
    st = feat["supertrend_dir"]
    rsi = feat["rsi"]
    ext = feat["vwap_dist_atr"].abs()

    state = pd.Series("range", index=feat.index, dtype=object)
    # precedence: exhaustion/reversal > expansion > trend > compression > range
    state[bb_rank <= 0.20] = "compression"
    state[feat["atr_pct"] >= 1.6 * atr_mean] = "expansion"
    state[(adx >= 25) & (st == 1) & (feat["ema20"] > feat["ema50"])] = "trend_up"
    state[(adx >= 25) & (st == -1) & (feat["ema20"] < feat["ema50"])] = "trend_down"
    state[(ext >= 2.5) | (rsi >= 78) | (rsi <= 22)] = "exhaustion"
    flip = st != st.shift(1)
    big_wick = (feat["upper_wick_ratio"] >= 0.6) | (feat["lower_wick_ratio"] >= 0.6)
    state[flip | big_wick] = "reversal"
    return state


def _regime_from_state(state: pd.Series) -> pd.Series:
    m = {
        "trend_up": "trending", "trend_down": "trending",
        "compression": "low_volatility", "expansion": "volatile",
        "exhaustion": "volatile", "reversal": "volatile", "range": "ranging",
    }
    return state.map(m).fillna("ranging")


def _forward_outcomes(candles: pd.DataFrame, atr: pd.Series, n: int,
                      target_atr: float = 2.0, stop_atr: float = 1.0):
    """Real forward stats over the next n bars (the outcome IS the future)."""
    c = candles["close"].values; h = candles["high"].values; l = candles["low"].values
    a = atr.values
    N = len(c)
    fwd = np.full(N, np.nan); mfe = np.full(N, np.nan); mae = np.full(N, np.nan); res = np.full(N, np.nan)
    for i in range(N - 1):
        end = min(i + 1 + n, N)
        wh, wl, wc = h[i + 1:end], l[i + 1:end], c[i + 1:end]
        if len(wh) == 0:
            continue
        base = c[i]
        atr_i = a[i] if (a[i] and a[i] > 0) else max(h[i] - l[i], 1e-9)
        mfe[i] = (wh.max() - base) / atr_i
        mae[i] = (wl.min() - base) / atr_i
        fwd[i] = (wc[-1] - base) / base
        # bars-to-resolution: first bar that hits +target_atr or -stop_atr
        hit = np.where((wh - base) / atr_i >= target_atr)[0]
        stop = np.where((wl - base) / atr_i <= -stop_atr)[0]
        first_t = hit[0] if len(hit) else n + 1
        first_s = stop[0] if len(stop) else n + 1
        res[i] = min(first_t, first_s) + 1
    return fwd, mae, mfe, res


def build_corpus(symbol: str, interval: str, start, end, forward_n: int = 10,
                 write_from=None) -> pd.DataFrame:
    """Build the analog corpus rows for a symbol/timeframe window.

    `write_from`: if set, features/outcomes are computed over the FULL [start, end]
    window (so indicators get proper warm-up), but only rows with ts >= write_from
    are returned — lets a nightly refresh recompute just the recent tail without
    corrupting it with poorly-warmed-up features.
    """
    from marketdata.candle_store import get_candle_store
    candles = get_candle_store().get(symbol, interval, start, end)
    if candles is None or len(candles) < 60:
        return pd.DataFrame(columns=CORPUS_COLS)
    candles = candles.sort_values("ts").reset_index(drop=True)

    feat = F.compute_features(candles, symbol, interval)
    fp = fingerprint_from_features(feat)
    state = classify_market_state(feat)
    regime = _regime_from_state(state)
    atr = feat["atr"].reset_index(drop=True)
    fwd, mae, mfe, res = _forward_outcomes(candles, atr, forward_n)

    # next_market_state = the structural state when the move resolved
    res_idx = (np.arange(len(state)) + np.nan_to_num(res, nan=forward_n)).astype(int)
    res_idx = np.clip(res_idx, 0, len(state) - 1)
    next_state = state.values[res_idx]

    out = pd.DataFrame({"symbol": symbol, "interval": interval, "ts": pd.to_datetime(candles["ts"])})
    for col in FP_COLS:
        out[col] = fp[col].values
    out["regime"] = regime.values
    out["market_state"] = state.values
    out["next_market_state"] = next_state
    out["forward_return"] = fwd
    out["mae"] = mae
    out["mfe"] = mfe
    out["resolution_time"] = res
    out["feature_version"] = F.FEATURE_VERSION
    # drop warm-up / tail rows without a usable fingerprint or forward outcome
    out = out[out["forward_return"].notna() & out["fp_rsi_norm"].notna()]
    if write_from is not None:
        out = out[out["ts"] >= pd.Timestamp(write_from)]
    return out[CORPUS_COLS]


# ── Lake persistence (lake/patterns/symbol=.../interval=.../{year}.parquet) ──

def _patterns_dir(symbol: str, interval: str) -> Path:
    return (lake._lake_root() / "patterns"
            / f"symbol={lake._sanitize(symbol)}" / f"interval={interval}")


def write_corpus(df: pd.DataFrame, symbol: str, interval: str) -> int:
    if df is None or df.empty:
        return 0
    df = df.copy(); df["ts"] = pd.to_datetime(df["ts"])
    total = 0
    for year, grp in df.groupby(df["ts"].dt.year):
        path = _patterns_dir(symbol, interval) / f"{int(year)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            grp = pd.concat([pd.read_parquet(path), grp], ignore_index=True)
        grp = grp.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
        grp.to_parquet(path, index=False)
        total += len(grp)
    logger.info("patterns: wrote %s %s → %d rows", symbol, interval, total)
    return total


def read_corpus(symbols=None, interval: str = "1d") -> pd.DataFrame:
    """Read the analog corpus (all symbols, or a subset) for a timeframe."""
    root = lake._lake_root() / "patterns"
    if not root.exists():
        return pd.DataFrame(columns=CORPUS_COLS)
    if symbols:
        files = []
        for s in symbols:
            files += [str(p) for p in (_patterns_dir(s, interval)).glob("*.parquet")]
    else:
        files = [str(p) for p in root.glob(f"symbol=*/interval={interval}/*.parquet")]
    if not files:
        return pd.DataFrame(columns=CORPUS_COLS)
    return lake._con().execute("SELECT * FROM read_parquet(?)", [files]).fetchdf()


def build_and_store(symbol: str, interval: str, start, end, forward_n: int = 10,
                    write_from=None) -> int:
    return write_corpus(
        build_corpus(symbol, interval, start, end, forward_n, write_from), symbol, interval)


if __name__ == "__main__":
    # Build the analog corpus across the universe:
    #   python -m marketdata.patterns [--interval 1d] [--from 2022-01-01] [--limit N]
    import argparse
    from datetime import date
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build the analog/pattern corpus from the lake")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--from", dest="frm", default="2022-01-01")
    ap.add_argument("--to", dest="to", default=date.today().isoformat())
    ap.add_argument("--forward-n", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--symbol", default=None)
    a = ap.parse_args()
    frm, to = date.fromisoformat(a.frm), date.fromisoformat(a.to)
    if a.symbol:
        syms = [a.symbol]
    else:
        from marketdata.backfill import universe
        syms = universe()
        if a.limit:
            syms = syms[: a.limit]
    total = 0
    for i, s in enumerate(syms, 1):
        try:
            total += build_and_store(s, a.interval, frm, to, a.forward_n)
        except Exception as e:
            logger.warning("corpus build failed for %s: %s", s, e)
        if i % 25 == 0:
            logger.info("… %d/%d symbols", i, len(syms))
    print(f"patterns corpus: {total} rows across {len(syms)} symbols ({a.interval})")
