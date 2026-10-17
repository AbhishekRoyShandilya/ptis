"""
ml.research_warehouse — the RESEARCH WAREHOUSE: every recognised birth + its full observation DNA + forward outcome.

The single most valuable dataset in the system: walks historical sessions, detects births via the existing
state_narrator(lite=True, rich=True) pipeline, snapshots the complete observation state at each birth moment
(point-in-time, no lookahead), and labels with forward outcome (MFE in 60/120 min, peak ext ATR, RAN/MODEST/FAILED).

Philosophy (user directive, 2026-07-21):
  ML is a RESEARCH ASSISTANT, not the decision maker. The center of PTIS is:
      Observation → Scientist → Verified Knowledge
  This warehouse feeds BOTH the Scientist (causal tests on any variable) AND the ML Miner (hypothesis generation),
  but neither writes directly to the Brain Library. Only HELD principles enter there.

Usage:
    from ml.research_warehouse import build_warehouse, load_warehouse, available_symbols

    # Build (slow — replays sessions bar-by-bar)
    df = build_warehouse(symbols=["HAL", "RELIANCE"], start="2023-01-01", end="2024-12-31")

    # Load (fast — reads pre-built parquet)
    df = load_warehouse()

    # CLI
    python -m ml.research_warehouse --build --symbols HAL,RELIANCE --start 2023-01-01
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
WAREHOUSE_DIR = Path("data/lake/research_warehouse")
WAREHOUSE_FILE = WAREHOUSE_DIR / "labeled_births.parquet"
VERSION = "1.0"


# ── helpers: list symbols available in the lake ────────────────────────────────
def available_symbols() -> list[str]:
    """Return symbols that have 1m candle data in the lake (the base resolution)."""
    from config.settings import get_settings
    lake_root = Path(get_settings().LAKE_PATH)
    candle_root = lake_root / "candles" / "type=EQ"
    if not candle_root.exists():
        return []
    syms = sorted(
        d.name.replace("symbol=", "")
        for d in candle_root.iterdir()
        if d.is_dir() and d.name.startswith("symbol=")
    )
    return syms


def _trading_dates(symbol: str, start: date, end: date) -> list[date]:
    """Dates that have at least 50 bars of 5m data in the lake (a real trading day)."""
    from marketdata.candle_store import get_candle_store
    store = get_candle_store()
    df = store.get(symbol, "5m", datetime.combine(start, datetime.min.time()),
                   datetime.combine(end, datetime.max.time()))
    if df is None or df.empty:
        return []
    dates = pd.to_datetime(df["ts"]).dt.date
    counts = dates.value_counts()
    return sorted(d for d, n in counts.items() if n >= 50 and start <= d <= end)


# ── FORWARD OUTCOME: measure MFE from a birth bar, point-in-time ──────────────
def _forward_mfe(candles: pd.DataFrame, birth_bar_idx: int, birth_price: float,
                 direction: str, atr_at_birth: float) -> dict:
    """Measure the Maximum Favorable Excursion (MFE) in windows after birth.

    Returns dict with:
      mfe_30min_pct, mfe_60min_pct, mfe_120min_pct — % move from birth price
      mfe_30min_atr, mfe_60min_atr, mfe_120min_atr — ATR-normalised
      mae_120min_pct — Maximum Adverse Excursion (worst drawdown against you)
      peak_ext_atr — the overall peak extension from birth
      time_to_peak_bars — bars until peak
    """
    out = {}
    ts = pd.to_datetime(candles["ts"])
    h = candles["high"].to_numpy(float)
    l = candles["low"].to_numpy(float)
    cl = candles["close"].to_numpy(float)
    n = len(candles)
    birth_ts = ts.iloc[birth_bar_idx]
    up = direction == "up"
    d = 1 if up else -1

    for label, minutes in [("30min", 30), ("60min", 60), ("120min", 120)]:
        cutoff = birth_ts + pd.Timedelta(minutes=minutes)
        window = candles[(ts > birth_ts) & (ts <= cutoff)]
        if window.empty:
            out[f"mfe_{label}_pct"] = np.nan
            out[f"mfe_{label}_atr"] = np.nan
            if label == "120min":
                out["mae_120min_pct"] = np.nan
            continue
        wi = window.index
        fav = h[wi].max() if up else l[wi].min()
        adv = l[wi].min() if up else h[wi].max()
        mfe_pct = d * (fav - birth_price) / birth_price * 100
        mae_pct = -d * (adv - birth_price) / birth_price * 100  # adverse = negative direction
        out[f"mfe_{label}_pct"] = round(float(mfe_pct), 4)
        out[f"mfe_{label}_atr"] = round(float(d * (fav - birth_price) / atr_at_birth), 2) \
            if np.isfinite(atr_at_birth) and atr_at_birth > 0 else np.nan
        if label == "120min":
            out["mae_120min_pct"] = round(float(mae_pct), 4)

    # overall peak extension until end of session
    rest = candles.iloc[birth_bar_idx + 1:]
    if rest.empty:
        out["peak_ext_atr"] = np.nan
        out["time_to_peak_bars"] = 0
    else:
        ri = rest.index
        if up:
            peak_price = h[ri].max()
            peak_idx = ri[np.argmax(h[ri])]
        else:
            peak_price = l[ri].min()
            peak_idx = ri[np.argmin(l[ri])]
        ext = d * (peak_price - birth_price)
        out["peak_ext_atr"] = round(float(ext / atr_at_birth), 2) \
            if np.isfinite(atr_at_birth) and atr_at_birth > 0 else np.nan
        out["time_to_peak_bars"] = int(peak_idx - birth_bar_idx)

    return out


# ── EXTRACT FEATURES at a birth moment from the StateStep + tags ──────────────
def _extract_birth_features(step, candles: pd.DataFrame, feat: pd.DataFrame,
                            birth_bar_idx: int) -> dict:
    """Extract the full observation DNA at a birth moment.

    Sources: StateStep rich fields (already computed by narrate_session), the tags
    (episodic recognition signature), and raw candle/feature data at the birth bar.
    Every value is POINT-IN-TIME (no lookahead). Returns a flat dict of features."""
    row = {}
    ts = pd.to_datetime(candles["ts"])
    h = candles["high"].to_numpy(float)
    l = candles["low"].to_numpy(float)
    o = candles["open"].to_numpy(float)
    cl = candles["close"].to_numpy(float)
    vol = candles["volume"].to_numpy(float)
    i = birth_bar_idx

    # ── FROM TAGS (episodic recognition signature) ─────────────────────────
    tags = set(step.tags or [])
    row["with_trend"] = 1 if "WITH_TREND" in tags else (0 if "COUNTER_TREND" in tags else -1)
    row["counter_trend"] = 1 if "COUNTER_TREND" in tags else 0
    row["room_tag"] = (
        "WALL" if "ROOM_WALL" in tags
        else "TIGHT" if "ROOM_TIGHT" in tags
        else "SOME" if "ROOM_SOME" in tags
        else "CLEAR" if "ROOM_CLEAR" in tags
        else "unknown"
    )
    row["regime_bull"] = 1 if "REGIME_BULL" in tags else 0
    row["regime_bear"] = 1 if "REGIME_BEAR" in tags else 0
    row["gap_up"] = 1 if "GAP_UP" in tags else 0
    row["gap_down"] = 1 if "GAP_DOWN" in tags else 0
    row["at_open"] = 1 if "AT_OPEN" in tags else 0
    row["above_vwap"] = 1 if "ABOVE_VWAP" in tags else 0
    row["below_vwap"] = 1 if "BELOW_VWAP" in tags else 0
    row["strong_effort"] = 1 if "STRONG_EFFORT" in tags else 0
    row["weak_effort"] = 1 if "WEAK_EFFORT" in tags else 0
    row["struct_hh_hl"] = 1 if "STRUCT_HH_HL" in tags else 0
    row["struct_lh_ll"] = 1 if "STRUCT_LH_LL" in tags else 0
    row["struct_range"] = 1 if "STRUCT_RANGE" in tags else 0
    # session phase from tags
    for ph in ("PHASE_OPEN", "PHASE_MORNING", "PHASE_MIDDAY", "PHASE_AFTERNOON", "PHASE_LATE"):
        if ph in tags:
            row["session_phase"] = ph.replace("PHASE_", "").lower()
            break
    else:
        row["session_phase"] = "unknown"
    row["n_tags"] = len(tags)

    # ── FROM RICH StateStep fields ─────────────────────────────────────────
    row["room_atr"] = step.room_atr if step.room_atr == step.room_atr else np.nan
    row["n_near"] = step.n_near
    row["energy"] = step.energy or ""
    row["fuel"] = step.fuel or ""
    row["loc"] = step.loc or ""
    row["leg_ext_atr"] = step.leg_ext_atr if step.leg_ext_atr == step.leg_ext_atr else np.nan
    row["leg_bars"] = step.leg_bars if step.leg_bars == step.leg_bars else np.nan
    row["in_run"] = step.in_run if step.in_run == step.in_run else np.nan
    row["rsi_at_birth"] = step.rsi if step.rsi == step.rsi else np.nan
    row["d_res_atr"] = step.d_res_atr if step.d_res_atr == step.d_res_atr else np.nan
    row["d_sup_atr"] = step.d_sup_atr if step.d_sup_atr == step.d_sup_atr else np.nan
    row["prev_high"] = step.prev_high if step.prev_high == step.prev_high else np.nan
    row["prev_low"] = step.prev_low if step.prev_low == step.prev_low else np.nan
    row["daily_ctx"] = step.daily_ctx or ""
    row["vwap_dist_atr"] = step.vwap_dist if step.vwap_dist == step.vwap_dist else np.nan
    row["rvol_at_birth"] = step.rvol_now if step.rvol_now == step.rvol_now else np.nan
    row["bbw_at_birth"] = step.bbw_now if step.bbw_now == step.bbw_now else np.nan

    # ── FROM RAW CANDLE / FEATURE DATA at birth bar ────────────────────────
    atr_arr = feat["atr"].to_numpy(float) if "atr" in feat else np.full(len(candles), np.nan)
    a = atr_arr[i] if i < len(atr_arr) and np.isfinite(atr_arr[i]) and atr_arr[i] > 0 else np.nan

    # ignition bar shape
    body = abs(cl[i] - o[i])
    bar_range = h[i] - l[i]
    row["ignition_body_atr"] = round(float(body / a), 3) if np.isfinite(a) else np.nan
    row["ignition_range_atr"] = round(float(bar_range / a), 3) if np.isfinite(a) else np.nan
    row["body_ratio"] = round(float(body / bar_range), 3) if bar_range > 0 else 0.0

    # BBW rank (percentile of current BBW in recent 50 bars)
    if "bb_width_pct" in feat and i >= 20:
        bbw_arr = feat["bb_width_pct"].to_numpy(float)
        window = bbw_arr[max(0, i - 50):i]
        valid = window[np.isfinite(window)]
        if len(valid) >= 10 and np.isfinite(bbw_arr[i]):
            row["bbw_rank"] = round(float(np.mean(valid <= bbw_arr[i])), 3)
        else:
            row["bbw_rank"] = np.nan
    else:
        row["bbw_rank"] = np.nan

    # ADX at birth
    if "adx" in feat:
        adx = feat["adx"].to_numpy(float)
        row["adx_at_birth"] = round(float(adx[i]), 2) if i < len(adx) and np.isfinite(adx[i]) else np.nan
    else:
        row["adx_at_birth"] = np.nan

    # SuperTrend direction alignment (if available)
    if "supertrend_dir" in feat:
        stdir = feat["supertrend_dir"].to_numpy(float)
        row["supertrend_dir"] = int(stdir[i]) if i < len(stdir) and np.isfinite(stdir[i]) else 0
    else:
        row["supertrend_dir"] = 0

    # Volume context: 3-bar rvol average around birth
    if "rvol" in feat:
        rvol_arr = feat["rvol"].to_numpy(float)
        start = max(0, i - 2)
        rvol_slice = rvol_arr[start:i + 1]
        valid_rvol = rvol_slice[np.isfinite(rvol_slice)]
        row["rvol_3bar_avg"] = round(float(np.mean(valid_rvol)), 3) if len(valid_rvol) > 0 else np.nan
    else:
        row["rvol_3bar_avg"] = np.nan

    # Position vs prior day high / low
    if np.isfinite(step.prev_high) and np.isfinite(a) and a > 0:
        row["pos_vs_pdh_atr"] = round(float((cl[i] - step.prev_high) / a), 2)
        row["pos_vs_pdl_atr"] = round(float((cl[i] - step.prev_low) / a), 2)
    else:
        row["pos_vs_pdh_atr"] = np.nan
        row["pos_vs_pdl_atr"] = np.nan

    # bars since open
    day_ts = ts.dt.date
    today = day_ts.iloc[i]
    sess_mask = day_ts == today
    sess_start = np.where(sess_mask.to_numpy())[0][0]
    row["bars_since_open"] = i - sess_start

    return row


# ── PROCESS ONE SESSION ──────────────────────────────────────────────────────
def _process_session(symbol: str, interval: str, session_date: date,
                     candles: pd.DataFrame | None = None) -> list[dict]:
    """Replay one session and extract birth feature rows with forward outcomes."""
    from intelligence.state_narrator import narrate_session
    from intelligence.episodic import classify_outcome
    from marketdata import features as F

    try:
        steps = narrate_session(symbol, interval, session_date, step=1,
                                candles=candles, with_recall=False,  # no recall when building warehouse
                                lite=True, rich=True)
    except Exception as e:
        logger.warning("narrate_session failed for %s %s: %s", symbol, session_date, e)
        return []

    if not steps:
        return []

    # Load candles + features for forward outcome measurement
    from marketdata.candle_store import get_candle_store
    store = get_candle_store()
    day = pd.Timestamp(session_date).normalize()
    c = store.get(symbol, interval, day - pd.Timedelta(days=10), day + pd.Timedelta(days=1))
    if c is None or c.empty:
        return []
    c = c.sort_values("ts").reset_index(drop=True)
    feat = F.compute_features(c, symbol, interval)
    ts_arr = pd.to_datetime(c["ts"])
    sess_mask = ts_arr.dt.date == session_date
    cl = c["close"].to_numpy(float)
    atr = feat["atr"].to_numpy(float)

    rows = []
    for step in steps:
        if not step.tags:   # only birth steps have tags
            continue
        # Parse direction from birth string
        birth_str = step.birth or ""
        parts = birth_str.split()
        if len(parts) < 2 or parts[1] not in ("up", "down"):
            continue
        kind = parts[0]  # BIRTH / REBIRTH / CONTINUATION
        direction = parts[1]

        # Find the bar index in candles matching this step's timestamp
        diffs = (ts_arr - step.ts).abs()
        birth_bar_idx = int(diffs.idxmin())

        # Sanity: must be in session
        if not sess_mask.iloc[birth_bar_idx]:
            continue

        birth_price = cl[birth_bar_idx]
        atr_at_birth = atr[birth_bar_idx]

        # EXTRACT FEATURES (point-in-time)
        features = _extract_birth_features(step, c, feat, birth_bar_idx)

        # FORWARD OUTCOME (the label)
        mfe = _forward_mfe(c, birth_bar_idx, birth_price, direction, atr_at_birth)

        # META
        meta = {
            "symbol": symbol,
            "date": str(session_date),
            "ts": str(step.ts),
            "direction": direction,
            "kind": kind,  # BIRTH / REBIRTH / CONTINUATION
            "birth_price": round(float(birth_price), 2),
            "atr_at_birth": round(float(atr_at_birth), 2) if np.isfinite(atr_at_birth) else np.nan,
            "warehouse_version": VERSION,
        }

        # OUTCOME CLASSIFICATION (consistent with episodic.py)
        peak_ext = mfe.get("peak_ext_atr", np.nan)
        outcome = classify_outcome(peak_ext) if np.isfinite(peak_ext) else "FAILED"

        row = {**meta, **features, **mfe, "outcome": outcome,
               "ran_binary": 1 if outcome == "RAN" else 0,
               "not_failed_binary": 1 if outcome in ("RAN", "MODEST") else 0}
        rows.append(row)

    return rows


# ── BUILD THE WAREHOUSE ──────────────────────────────────────────────────────
def build_warehouse(symbols: list[str] | None = None,
                    start: str | date = "2023-01-01",
                    end: str | date | None = None,
                    interval: str = "5m",
                    progress: int = 10) -> pd.DataFrame:
    """Build the complete Research Warehouse from the lake.

    Walks every (symbol, date) and replays the session to extract birth features + forward outcomes.
    Returns the DataFrame AND persists to parquet.

    Args:
        symbols:  list of stock symbols (default: all available in lake)
        start:    start date
        end:      end date (default: yesterday)
        interval: candle interval (default: 5m — the eye's native resolution)
        progress: log progress every N sessions
    """
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if end is None:
        end = date.today() - timedelta(days=1)
    elif isinstance(end, str):
        end = date.fromisoformat(end)

    if symbols is None:
        symbols = available_symbols()
    if not symbols:
        logger.error("No symbols found in lake")
        return pd.DataFrame()

    logger.info("Building Research Warehouse: %d symbols, %s → %s", len(symbols), start, end)
    t0 = time.time()
    all_rows: list[dict] = []
    total_sessions = 0
    skipped = 0

    for sym in symbols:
        try:
            dates = _trading_dates(sym, start, end)
        except Exception as e:
            logger.warning("Skipping %s — cannot enumerate dates: %s", sym, e)
            continue
        logger.info("  %s: %d trading days", sym, len(dates))
        for di, d in enumerate(dates):
            try:
                rows = _process_session(sym, interval, d)
                all_rows.extend(rows)
            except Exception as e:
                logger.warning("  %s %s FAILED: %s", sym, d, e)
                skipped += 1
            total_sessions += 1
            if total_sessions % progress == 0:
                logger.info("    ... %d sessions processed (%d births so far, %d skipped)",
                            total_sessions, len(all_rows), skipped)

    elapsed = time.time() - t0
    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.warning("Warehouse is empty — no births found")
        return df

    # Persist
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WAREHOUSE_FILE, index=False)
    logger.info("Research Warehouse built: %d births across %d sessions in %.1fs → %s",
                len(df), total_sessions, elapsed, WAREHOUSE_FILE)

    # Summary
    if "outcome" in df:
        oc = df["outcome"].value_counts()
        logger.info("  Outcomes: %s", dict(oc))
    return df


def load_warehouse(path: str | Path | None = None) -> pd.DataFrame:
    """Load the pre-built Research Warehouse from parquet."""
    p = Path(path) if path else WAREHOUSE_FILE
    if not p.exists():
        raise FileNotFoundError(f"Warehouse not found at {p}. Run build_warehouse() first.")
    return pd.read_parquet(p)


def warehouse_summary(df: pd.DataFrame | None = None) -> str:
    """Print a summary of the warehouse contents."""
    if df is None:
        df = load_warehouse()
    lines = [f"Research Warehouse: {len(df)} births"]
    lines.append(f"  Symbols: {df['symbol'].nunique()} ({', '.join(sorted(df['symbol'].unique())[:10])}...)")
    lines.append(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    lines.append(f"  Kinds: {dict(df['kind'].value_counts())}")
    lines.append(f"  Outcomes: {dict(df['outcome'].value_counts())}")
    if "mfe_120min_pct" in df and df["mfe_120min_pct"].notna().any():
        ran = df[df["outcome"] == "RAN"]
        lines.append(f"  RAN births: median MFE-120min {ran['mfe_120min_pct'].median():.2f}%")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Research Warehouse — birth dataset builder")
    ap.add_argument("--build", action="store_true", help="Build the warehouse from lake data")
    ap.add_argument("--summary", action="store_true", help="Print warehouse summary")
    ap.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (default: all)")
    ap.add_argument("--start", type=str, default="2023-01-01", help="Start date")
    ap.add_argument("--end", type=str, default=None, help="End date (default: yesterday)")
    ap.add_argument("--interval", type=str, default="5m", help="Candle interval")
    ap.add_argument("--verify", action="store_true", help="Verify mode — process 1 day and print details")
    args = ap.parse_args()

    if args.build or args.verify:
        syms = args.symbols.split(",") if args.symbols else None
        if args.verify:
            syms = syms or ["HAL"]
            start = args.start
            # Just do 2 days for verification
            end = str(date.fromisoformat(start) + timedelta(days=5))
            df = build_warehouse(symbols=syms, start=start, end=end, interval=args.interval, progress=1)
            if not df.empty:
                print("\n" + warehouse_summary(df))
                print("\n── Sample births ──")
                for _, r in df.head(5).iterrows():
                    print(f"  {r['symbol']} {r['date']} {r['ts']} {r['direction']} {r['kind']}: "
                          f"MFE-120m {r.get('mfe_120min_pct', '?')}%, peak {r.get('peak_ext_atr', '?')} ATR → {r['outcome']}")
                    print(f"    room={r.get('room_tag','?')} rvol={r.get('rvol_at_birth','?')} "
                          f"bbw={r.get('bbw_at_birth','?')} effort={'strong' if r.get('strong_effort') else 'weak' if r.get('weak_effort') else '?'}")
        else:
            build_warehouse(symbols=syms, start=args.start, end=args.end, interval=args.interval)

    elif args.summary:
        print(warehouse_summary())
    else:
        ap.print_help()
