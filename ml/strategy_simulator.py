"""
ml.strategy_simulator — convert births into SIMULATED TRADES and find optimal filter combinations.

The missing bridge: the Research Warehouse labels births by peak extension (RAN/MODEST/FAILED), but TRADING is
different. A RAN move can stop you out before it runs; a MODEST move with a tight entry can be a clean winner.

This module:
  1. Enhances each birth with HIGHER-TIMEFRAME context (15m/1h MA50, trend, structure)
  2. Simulates a trade for EVERY birth (entry/SL/target/exit, bar-by-bar)
  3. Applies Indian transaction costs (STT, brokerage, slippage)
  4. Tests filter combinations against a target trade frequency and win rate

Usage:
    python -m ml.strategy_simulator --simulate          # simulate all births
    python -m ml.strategy_simulator --optimize          # find best filter combos
    python -m ml.strategy_simulator --report            # print strategy report

No existing files modified.
"""
from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/lake/research_warehouse")
TRADES_FILE = RESULTS_DIR / "simulated_trades.parquet"


# ── TRADE SIMULATION CONFIG ──────────────────────────────────────────────────
@dataclass
class TradeConfig:
    """Configuration for trade simulation."""
    stop_atr_multiple: float = 1.0      # SL = 1 ATR below birth bar low (longs) / above high (shorts)
    target_pct: float = 1.5             # target = 1.5% from entry
    target_r: float = 0.0              # alternative: target expressed as R-multiple (0 = use target_pct)
    max_hold_bars: int = 24             # max bars to hold (24 × 5min = 2 hours)
    eod_exit: bool = True               # exit at 15:15 IST regardless
    eod_time: str = "15:15"
    trail_after_1r: bool = False        # trail stop to entry after 1R (optional)
    # Indian FNO cost defaults
    instrument: str = "equity_intraday"
    quantity: int = 1                   # per-share simulation (scaled later)


# ── HTF FEATURE EXTRACTION ──────────────────────────────────────────────────
def _add_htf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add higher-timeframe context features to each birth row.

    For each birth, queries the 15m and 1h candle + feature data to determine:
    - 15m/1h MA50 position (price above/below)
    - 15m/1h trend direction (last 3 bars making HH/HL or LH/LL)
    - 15m/1h RSI (overbought/oversold context)
    - Distance from 15m/1h MA50 in ATR (stretched or near)
    """
    from marketdata.candle_store import get_candle_store
    from marketdata import features as F

    store = get_candle_store()
    new_cols = {
        "htf_15m_above_ma50": [], "htf_15m_ma50_dist_atr": [], "htf_15m_rsi": [],
        "htf_15m_trend": [],  # 1=bullish, -1=bearish, 0=neutral
        "htf_1h_above_ma50": [], "htf_1h_ma50_dist_atr": [], "htf_1h_rsi": [],
        "htf_1h_trend": [],
        "htf_aligned_with_birth": [],  # 1=HTF agrees with birth direction, 0=against
    }

    # Cache candles per (symbol, date) to avoid repeated fetches
    _cache_15m: dict = {}
    _cache_1h: dict = {}

    for _, row in df.iterrows():
        sym = row["symbol"]
        d = pd.Timestamp(row["date"]).normalize()
        birth_ts = pd.Timestamp(row["ts"])
        direction = row["direction"]

        # --- 15m context ---
        key_15 = f"{sym}|{d.date()}"
        if key_15 not in _cache_15m:
            try:
                c15 = store.get(sym, "15m", d - pd.Timedelta(days=15), d + pd.Timedelta(days=1))
                f15 = F.compute_features(c15.sort_values("ts").reset_index(drop=True), sym, "15m")
                _cache_15m[key_15] = (c15.sort_values("ts").reset_index(drop=True), f15)
            except Exception:
                _cache_15m[key_15] = (None, None)

        c15, f15 = _cache_15m[key_15]
        if c15 is not None and not c15.empty and f15 is not None:
            ts15 = pd.to_datetime(c15["ts"])
            # Find the last 15m bar AT or BEFORE the birth timestamp
            mask = ts15 <= birth_ts
            if mask.any():
                idx = mask.values.nonzero()[0][-1]
                cl15 = c15["close"].to_numpy(float)
                ma50_15 = f15["ema50"].to_numpy(float) if "ema50" in f15 else np.full(len(c15), np.nan)
                atr15 = f15["atr"].to_numpy(float)
                rsi15 = f15["rsi"].to_numpy(float) if "rsi" in f15 else np.full(len(c15), np.nan)

                above50_15 = 1 if (np.isfinite(ma50_15[idx]) and cl15[idx] > ma50_15[idx]) else 0
                dist50_15 = round(float((cl15[idx] - ma50_15[idx]) / atr15[idx]), 2) \
                    if (np.isfinite(ma50_15[idx]) and np.isfinite(atr15[idx]) and atr15[idx] > 0) else np.nan
                rsi_val_15 = round(float(rsi15[idx]), 1) if np.isfinite(rsi15[idx]) else np.nan

                # 15m trend (last 3 closes: HH pattern = bullish, LL = bearish)
                if idx >= 2:
                    c3 = cl15[idx-2:idx+1]
                    if c3[2] > c3[1] > c3[0]:
                        trend15 = 1
                    elif c3[2] < c3[1] < c3[0]:
                        trend15 = -1
                    else:
                        trend15 = 0
                else:
                    trend15 = 0

                new_cols["htf_15m_above_ma50"].append(above50_15)
                new_cols["htf_15m_ma50_dist_atr"].append(dist50_15)
                new_cols["htf_15m_rsi"].append(rsi_val_15)
                new_cols["htf_15m_trend"].append(trend15)
            else:
                for k in ["htf_15m_above_ma50", "htf_15m_ma50_dist_atr", "htf_15m_rsi", "htf_15m_trend"]:
                    new_cols[k].append(np.nan)
        else:
            for k in ["htf_15m_above_ma50", "htf_15m_ma50_dist_atr", "htf_15m_rsi", "htf_15m_trend"]:
                new_cols[k].append(np.nan)

        # --- 1h context ---
        key_1h = f"{sym}|{d.date()}|1h"
        if key_1h not in _cache_1h:
            try:
                c1h = store.get(sym, "1h", d - pd.Timedelta(days=30), d + pd.Timedelta(days=1))
                f1h = F.compute_features(c1h.sort_values("ts").reset_index(drop=True), sym, "1h")
                _cache_1h[key_1h] = (c1h.sort_values("ts").reset_index(drop=True), f1h)
            except Exception:
                _cache_1h[key_1h] = (None, None)

        c1h, f1h = _cache_1h[key_1h]
        if c1h is not None and not c1h.empty and f1h is not None:
            ts1h = pd.to_datetime(c1h["ts"])
            mask = ts1h <= birth_ts
            if mask.any():
                idx = mask.values.nonzero()[0][-1]
                cl1h = c1h["close"].to_numpy(float)
                ma50_1h = f1h["ema50"].to_numpy(float) if "ema50" in f1h else np.full(len(c1h), np.nan)
                atr1h = f1h["atr"].to_numpy(float)
                rsi1h = f1h["rsi"].to_numpy(float) if "rsi" in f1h else np.full(len(c1h), np.nan)

                above50_1h = 1 if (np.isfinite(ma50_1h[idx]) and cl1h[idx] > ma50_1h[idx]) else 0
                dist50_1h = round(float((cl1h[idx] - ma50_1h[idx]) / atr1h[idx]), 2) \
                    if (np.isfinite(ma50_1h[idx]) and np.isfinite(atr1h[idx]) and atr1h[idx] > 0) else np.nan
                rsi_val_1h = round(float(rsi1h[idx]), 1) if np.isfinite(rsi1h[idx]) else np.nan

                if idx >= 2:
                    c3 = cl1h[idx-2:idx+1]
                    trend1h = 1 if c3[2] > c3[1] > c3[0] else (-1 if c3[2] < c3[1] < c3[0] else 0)
                else:
                    trend1h = 0

                new_cols["htf_1h_above_ma50"].append(above50_1h)
                new_cols["htf_1h_ma50_dist_atr"].append(dist50_1h)
                new_cols["htf_1h_rsi"].append(rsi_val_1h)
                new_cols["htf_1h_trend"].append(trend1h)
            else:
                for k in ["htf_1h_above_ma50", "htf_1h_ma50_dist_atr", "htf_1h_rsi", "htf_1h_trend"]:
                    new_cols[k].append(np.nan)
        else:
            for k in ["htf_1h_above_ma50", "htf_1h_ma50_dist_atr", "htf_1h_rsi", "htf_1h_trend"]:
                new_cols[k].append(np.nan)

        # --- HTF alignment ---
        t15 = new_cols["htf_15m_trend"][-1]
        t1h = new_cols["htf_1h_trend"][-1]
        up = direction == "up"
        if np.isfinite(t15) and np.isfinite(t1h):
            aligned = 1 if ((up and t15 > 0 and t1h > 0) or (not up and t15 < 0 and t1h < 0)) else 0
        elif np.isfinite(t15):
            aligned = 1 if ((up and t15 > 0) or (not up and t15 < 0)) else 0
        else:
            aligned = 0
        new_cols["htf_aligned_with_birth"].append(aligned)

    for col, vals in new_cols.items():
        df[col] = vals
    return df


# ── TRADE SIMULATOR ──────────────────────────────────────────────────────────
@dataclass
class TradeResult:
    """Result of simulating one trade."""
    symbol: str
    date: str
    ts: str
    direction: str
    entry: float
    stop: float
    target: float
    exit_price: float
    exit_reason: str       # "TARGET" | "STOP" | "EOD" | "MAX_HOLD"
    exit_ts: str
    pnl_pct: float         # gross P&L as % of entry
    pnl_r: float           # gross P&L in R-multiples
    cost_r: float          # transaction cost in R
    net_r: float           # net P&L after costs
    net_pnl_pct: float     # net P&L as %
    win: bool              # net_r > 0
    hold_bars: int
    mae_pct: float         # max adverse excursion during hold


def simulate_trade(candles: pd.DataFrame, birth_bar_idx: int, direction: str,
                   entry_price: float, atr: float, cfg: TradeConfig) -> Optional[TradeResult]:
    """Simulate one trade forward from a birth bar.

    Entry: close of the birth bar.
    SL: for longs, birth_bar_low - stop_atr_multiple × ATR. For shorts, birth_bar_high + stop_atr_multiple × ATR.
    Target: entry ± target_pct%. Or entry ± target_r × (entry - stop) if target_r > 0.
    Exit: target hit, stop hit, EOD, or max_hold_bars — whichever comes first.
    """
    ts_arr = pd.to_datetime(candles["ts"])
    h = candles["high"].to_numpy(float)
    l = candles["low"].to_numpy(float)
    cl = candles["close"].to_numpy(float)
    n = len(candles)
    up = direction == "up"
    d = 1 if up else -1

    entry = entry_price

    # Stop loss
    if up:
        stop = l[birth_bar_idx] - cfg.stop_atr_multiple * atr
    else:
        stop = h[birth_bar_idx] + cfg.stop_atr_multiple * atr
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return None

    # Target
    if cfg.target_r > 0:
        target = entry + d * cfg.target_r * stop_dist
    else:
        target = entry * (1 + d * cfg.target_pct / 100)

    # Parse EOD time
    eod_h, eod_m = (int(x) for x in cfg.eod_time.split(":"))

    # Walk forward bar-by-bar
    exit_price = entry
    exit_reason = "MAX_HOLD"
    exit_ts = str(ts_arr.iloc[birth_bar_idx])
    hold_bars = 0
    mae = 0.0  # max adverse excursion

    for j in range(birth_bar_idx + 1, min(n, birth_bar_idx + 1 + cfg.max_hold_bars)):
        bar_ts = ts_arr.iloc[j]
        hold_bars += 1

        # Check if different trading day (shouldn't hold overnight)
        if bar_ts.date() != ts_arr.iloc[birth_bar_idx].date():
            exit_price = cl[j-1] if j > birth_bar_idx + 1 else entry
            exit_reason = "NEW_DAY"
            exit_ts = str(ts_arr.iloc[j-1])
            break

        # EOD exit check
        if cfg.eod_exit and bar_ts.hour >= eod_h and bar_ts.minute >= eod_m:
            exit_price = cl[j]
            exit_reason = "EOD"
            exit_ts = str(bar_ts)
            break

        # Check stop hit (assume worst-case: stop checked on adverse side first)
        if up:
            if l[j] <= stop:
                exit_price = stop  # filled at stop level
                exit_reason = "STOP"
                exit_ts = str(bar_ts)
                break
            if h[j] >= target:
                exit_price = target
                exit_reason = "TARGET"
                exit_ts = str(bar_ts)
                break
            adverse = (entry - l[j]) / entry * 100
        else:
            if h[j] >= stop:
                exit_price = stop
                exit_reason = "STOP"
                exit_ts = str(bar_ts)
                break
            if l[j] <= target:
                exit_price = target
                exit_reason = "TARGET"
                exit_ts = str(bar_ts)
                break
            adverse = (h[j] - entry) / entry * 100

        mae = max(mae, adverse)

        # Trail stop to entry after 1R (optional)
        if cfg.trail_after_1r and hold_bars >= 2:
            if up and h[j] >= entry + stop_dist:
                stop = max(stop, entry)
            elif not up and l[j] <= entry - stop_dist:
                stop = min(stop, entry)
    else:
        # Max hold reached — exit at close of last bar
        last_j = min(n - 1, birth_bar_idx + cfg.max_hold_bars)
        exit_price = cl[last_j]
        exit_ts = str(ts_arr.iloc[last_j])

    # P&L
    pnl_pct = d * (exit_price - entry) / entry * 100
    pnl_r = d * (exit_price - entry) / stop_dist if stop_dist > 0 else 0

    # Apply costs
    from discovery.cost_model import get_cost_model
    cost_model = get_cost_model()
    net_r, costs = cost_model.apply_costs(
        r_multiple=pnl_r, entry_price=entry, stop_distance=stop_dist,
        quantity=cfg.quantity, regime="normal",
    )
    cost_r = costs.cost_r
    net_pnl_pct = pnl_pct - (cost_r * stop_dist / entry * 100)

    return TradeResult(
        symbol="", date="", ts="", direction=direction,
        entry=round(entry, 2), stop=round(stop, 2), target=round(target, 2),
        exit_price=round(exit_price, 2), exit_reason=exit_reason,
        exit_ts=exit_ts, pnl_pct=round(pnl_pct, 4), pnl_r=round(pnl_r, 4),
        cost_r=round(cost_r, 4), net_r=round(net_r, 4),
        net_pnl_pct=round(net_pnl_pct, 4), win=net_r > 0,
        hold_bars=hold_bars, mae_pct=round(mae, 4),
    )


# ── SIMULATE ALL BIRTHS ─────────────────────────────────────────────────────
def simulate_all(df: pd.DataFrame, cfg: TradeConfig = None) -> pd.DataFrame:
    """Simulate trades for every birth in the warehouse DataFrame.

    Returns the input DataFrame augmented with trade result columns.
    """
    if cfg is None:
        cfg = TradeConfig()

    from marketdata.candle_store import get_candle_store
    store = get_candle_store()

    trade_cols = ["entry", "stop", "target_price", "exit_price", "exit_reason", "exit_ts",
                  "pnl_pct", "pnl_r", "cost_r", "net_r", "net_pnl_pct", "trade_win", "hold_bars", "trade_mae_pct"]
    for col in trade_cols:
        df[col] = np.nan
    df["exit_reason"] = ""
    df["exit_ts"] = ""
    df["trade_win"] = False

    _candle_cache: dict = {}
    simulated = 0
    failed = 0

    for idx, row in df.iterrows():
        sym = row["symbol"]
        d = pd.Timestamp(row["date"]).normalize()
        birth_ts = pd.Timestamp(row["ts"])

        # Get 5m candles for this session
        key = f"{sym}|{d.date()}"
        if key not in _candle_cache:
            try:
                c = store.get(sym, "5m", d - pd.Timedelta(days=2), d + pd.Timedelta(days=1))
                _candle_cache[key] = c.sort_values("ts").reset_index(drop=True) if c is not None else None
            except Exception:
                _candle_cache[key] = None

        c = _candle_cache[key]
        if c is None or c.empty:
            failed += 1
            continue

        ts_arr = pd.to_datetime(c["ts"])
        diffs = (ts_arr - birth_ts).abs()
        birth_bar_idx = int(diffs.idxmin())

        entry_price = row["birth_price"]
        atr_val = row.get("atr_at_birth", np.nan)
        if not np.isfinite(atr_val) or atr_val <= 0:
            failed += 1
            continue

        result = simulate_trade(c, birth_bar_idx, row["direction"], entry_price, atr_val, cfg)
        if result is None:
            failed += 1
            continue

        df.at[idx, "entry"] = result.entry
        df.at[idx, "stop"] = result.stop
        df.at[idx, "target_price"] = result.target
        df.at[idx, "exit_price"] = result.exit_price
        df.at[idx, "exit_reason"] = result.exit_reason
        df.at[idx, "exit_ts"] = result.exit_ts
        df.at[idx, "pnl_pct"] = result.pnl_pct
        df.at[idx, "pnl_r"] = result.pnl_r
        df.at[idx, "cost_r"] = result.cost_r
        df.at[idx, "net_r"] = result.net_r
        df.at[idx, "net_pnl_pct"] = result.net_pnl_pct
        df.at[idx, "trade_win"] = result.win
        df.at[idx, "hold_bars"] = result.hold_bars
        df.at[idx, "trade_mae_pct"] = result.mae_pct
        simulated += 1

    logger.info("Simulated %d trades (%d failed)", simulated, failed)
    return df


# ── FILTER DEFINITIONS ───────────────────────────────────────────────────────
FILTERS = {
    # Session timing
    "morning_only": lambda df: df[df["bars_since_open"] <= 30],
    "first_hour": lambda df: df[df["bars_since_open"] <= 12],
    "not_late": lambda df: df[df["session_phase"] != "late"],

    # Ignition quality
    "strong_body": lambda df: df[df["ignition_body_atr"] >= 0.8],
    "very_strong_body": lambda df: df[df["ignition_body_atr"] >= 1.2],
    "strong_effort_tag": lambda df: df[df["strong_effort"] == 1],

    # HTF alignment
    "htf_15m_aligned": lambda df: df.apply(
        lambda r: (r["direction"] == "up" and r.get("htf_15m_trend", 0) >= 0) or
                  (r["direction"] == "down" and r.get("htf_15m_trend", 0) <= 0), axis=1).pipe(lambda m: df[m]),
    "htf_1h_aligned": lambda df: df.apply(
        lambda r: (r["direction"] == "up" and r.get("htf_1h_trend", 0) >= 0) or
                  (r["direction"] == "down" and r.get("htf_1h_trend", 0) <= 0), axis=1).pipe(lambda m: df[m]),
    "htf_both_aligned": lambda df: df[df.get("htf_aligned_with_birth", pd.Series(0, index=df.index)) == 1],

    # MA50 context
    "15m_above_ma50_for_long": lambda df: df[
        ((df["direction"] == "up") & (df.get("htf_15m_above_ma50", pd.Series(0, index=df.index)) == 1)) |
        ((df["direction"] == "down") & (df.get("htf_15m_above_ma50", pd.Series(1, index=df.index)) == 0))
    ],

    # Stretched from MA50
    "not_stretched_15m": lambda df: df[df["htf_15m_ma50_dist_atr"].abs().fillna(0) <= 3.0],

    # RVOL
    "rvol_above_1": lambda df: df[df["rvol_at_birth"].fillna(0) >= 1.0],
    "rvol_above_1_5": lambda df: df[df["rvol_at_birth"].fillna(0) >= 1.5],

    # Birth type
    "birth_only": lambda df: df[df["kind"].isin(["BIRTH", "REBIRTH"])],
    "birth_no_continuation": lambda df: df[df["kind"] != "CONTINUATION"],
}


def _strategy_stats(df: pd.DataFrame, n_years: float = 2.0, n_symbols: int = 15) -> dict:
    """Compute strategy statistics for a filtered subset of trades."""
    valid = df[df["net_r"].notna()]
    if valid.empty:
        return {"n": 0}

    wins = valid[valid["trade_win"]]
    losses = valid[~valid["trade_win"]]
    n = len(valid)
    wr = len(wins) / n if n > 0 else 0
    avg_win_r = wins["net_r"].mean() if len(wins) > 0 else 0
    avg_loss_r = losses["net_r"].mean() if len(losses) > 0 else 0
    expectancy = wr * avg_win_r + (1 - wr) * avg_loss_r
    per_year_per_sym = n / n_years / n_symbols if n_years > 0 and n_symbols > 0 else n
    pf = abs(wins["net_r"].sum() / losses["net_r"].sum()) if losses["net_r"].sum() != 0 else float("inf")

    return {
        "n": n, "win_rate": round(wr, 3), "per_year_per_sym": round(per_year_per_sym, 1),
        "avg_win_r": round(float(avg_win_r), 3), "avg_loss_r": round(float(avg_loss_r), 3),
        "expectancy_r": round(float(expectancy), 4), "profit_factor": round(float(pf), 2),
        "total_r": round(float(valid["net_r"].sum()), 1),
        "avg_cost_r": round(float(valid["cost_r"].mean()), 4),
        "exit_target_pct": round(float((valid["exit_reason"] == "TARGET").mean()), 2),
        "exit_stop_pct": round(float((valid["exit_reason"] == "STOP").mean()), 2),
        "exit_eod_pct": round(float((valid["exit_reason"] == "EOD").mean()), 2),
        "avg_hold_bars": round(float(valid["hold_bars"].mean()), 1),
        "med_pnl_pct": round(float(valid["net_pnl_pct"].median()), 4),
    }


def optimize_filters(df: pd.DataFrame, target_trades_per_year: tuple = (30, 80),
                     min_win_rate: float = 0.55, max_combos: int = 3,
                     n_years: float = 2.0, n_symbols: int = 15) -> list[dict]:
    """Test filter combinations and find those that meet the trade count and win rate targets.

    Returns list of strategy configs sorted by expectancy, each with:
    - filters applied, trade count, win rate, expectancy, per-year-per-symbol count
    """
    results = []

    # Get available filter names (only those whose columns exist in df)
    available = []
    for name, fn in FILTERS.items():
        try:
            test = fn(df.head(100))
            available.append(name)
        except Exception:
            pass

    logger.info("Testing %d available filters in combos of 1-%d", len(available), max_combos)

    # Test single filters and combinations
    for r in range(1, min(max_combos + 1, len(available) + 1)):
        for combo in itertools.combinations(available, r):
            filtered = df.copy()
            for fname in combo:
                try:
                    filtered = FILTERS[fname](filtered)
                except Exception:
                    break
            if filtered.empty:
                continue

            stats = _strategy_stats(filtered, n_years, n_symbols)
            if stats["n"] < 20:
                continue

            per_yr = stats["per_year_per_sym"]
            wr = stats["win_rate"]

            # Check if in target range
            in_range = target_trades_per_year[0] <= per_yr <= target_trades_per_year[1]

            results.append({
                "filters": list(combo),
                "n_filters": len(combo),
                **stats,
                "in_target_range": in_range,
                "meets_wr": wr >= min_win_rate,
                "score": stats["expectancy_r"] * np.sqrt(stats["n"]),  # reward expectancy × sample size
            })

    # Sort by score (expectancy × sqrt(n))
    results.sort(key=lambda x: -x["score"])
    return results


def strategy_report(results: list[dict], top_n: int = 20) -> str:
    """Generate readable report from optimization results."""
    lines = ["═" * 90, "STRATEGY FILTER OPTIMIZATION REPORT", "═" * 90]
    lines.append(f"  Tested {len(results)} filter combinations\n")

    # Best by expectancy
    best = [r for r in results if r["in_target_range"] and r["n"] >= 50]
    if not best:
        best = results[:top_n]

    lines.append("── TOP STRATEGIES (sorted by score = expectancy × √n) ──\n")
    lines.append(f"{'Rank':>4} {'Filters':<50} {'N':>5} {'Tr/Yr/Sym':>9} {'WR':>6} {'E[R]':>7} {'PF':>5} {'TotR':>7} {'Tgt%':>5} {'Stp%':>5}")
    lines.append("─" * 110)

    for i, r in enumerate(best[:top_n]):
        filt_str = " + ".join(r["filters"])[:48]
        lines.append(
            f"  {i+1:2d}. {filt_str:<50} {r['n']:5d} {r['per_year_per_sym']:9.1f} "
            f"{r['win_rate']:6.1%} {r['expectancy_r']:7.4f} {r['profit_factor']:5.2f} "
            f"{r['total_r']:7.1f} {r['exit_target_pct']:5.0%} {r['exit_stop_pct']:5.0%}"
        )

    # Summary of high WR strategies
    high_wr = [r for r in results if r["win_rate"] >= 0.65 and r["n"] >= 30]
    if high_wr:
        lines.append(f"\n── HIGH WIN-RATE STRATEGIES (≥65%, n≥30) ──\n")
        high_wr.sort(key=lambda x: -x["win_rate"])
        for r in high_wr[:10]:
            filt_str = " + ".join(r["filters"])[:60]
            lines.append(f"  {filt_str:<60} WR={r['win_rate']:.1%}  n={r['n']}  E[R]={r['expectancy_r']:.4f}  "
                         f"trades/yr/sym={r['per_year_per_sym']:.0f}")

    return "\n".join(lines)


# ── MAIN PIPELINE ───────────────────────────────────────────────────────────
def run_full_pipeline(cfg: TradeConfig = None):
    """Complete pipeline: load warehouse → add HTF → simulate → optimize → report."""
    from ml.research_warehouse import load_warehouse
    t0 = time.time()

    # 1. Load warehouse
    logger.info("Loading Research Warehouse...")
    df = load_warehouse()
    logger.info("Loaded %d births", len(df))

    # 2. Add HTF features
    logger.info("Adding HTF features (15m/1h)...")
    df = _add_htf_features(df)
    logger.info("HTF features added")

    # 3. Simulate trades
    logger.info("Simulating trades...")
    if cfg is None:
        cfg = TradeConfig()
    df = simulate_all(df, cfg)
    valid = df[df["net_r"].notna()]
    logger.info("Simulated %d valid trades", len(valid))

    # 4. Save enhanced dataset
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TRADES_FILE, index=False)
    logger.info("Trades saved to %s", TRADES_FILE)

    # 5. Baseline stats (no filters)
    n_syms = df["symbol"].nunique()
    dates = pd.to_datetime(df["date"])
    n_years = (dates.max() - dates.min()).days / 365.25
    baseline = _strategy_stats(df, n_years, n_syms)
    logger.info("Baseline: %s", baseline)

    # 6. Optimize filters
    logger.info("Optimizing filter combinations...")
    results = optimize_filters(df, n_years=n_years, n_symbols=n_syms)
    logger.info("Found %d valid combos", len(results))

    # 7. Report
    report = strategy_report(results)
    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.0fs", elapsed)
    return df, results, report


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Strategy Simulator — birth trades with HTF filters")
    ap.add_argument("--simulate", action="store_true", help="Run full simulation pipeline")
    ap.add_argument("--optimize", action="store_true", help="Optimize filter combinations only (requires prior --simulate)")
    ap.add_argument("--report", action="store_true", help="Print report from saved trades")
    ap.add_argument("--stop-atr", type=float, default=1.0, help="Stop loss ATR multiple")
    ap.add_argument("--target-pct", type=float, default=1.5, help="Target in %%")
    ap.add_argument("--max-hold", type=int, default=24, help="Max hold bars")
    args = ap.parse_args()

    cfg = TradeConfig(stop_atr_multiple=args.stop_atr, target_pct=args.target_pct,
                      max_hold_bars=args.max_hold)

    if args.simulate:
        df, results, report = run_full_pipeline(cfg)
        print("\n" + report)
    elif args.optimize or args.report:
        df = pd.read_parquet(TRADES_FILE)
        n_syms = df["symbol"].nunique()
        dates = pd.to_datetime(df["date"])
        n_years = (dates.max() - dates.min()).days / 365.25
        results = optimize_filters(df, n_years=n_years, n_symbols=n_syms)
        print(strategy_report(results))
    else:
        ap.print_help()
