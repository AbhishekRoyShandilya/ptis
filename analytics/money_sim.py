"""
analytics.money_sim — money read from the CACHED DayStory (low-CPU, no re-narration).

Simulates the operating-process trade with REALISTIC entry + a TRAILING exit, net of Indian charges:
  · ENTRY at the confirmation bar (move held + extended >= confirm_ext ATR from birth), stop = BIRTH LEVEL.
  · ENTRY-REALISM: SKIP moves already extended > max_entry_risk_atr at confirmation — by then price has run
    too far to fill (chasing a spike); a fill there is impossible or a terrible price (the user's "violent
    opening spikes you can't grab").
  · TRAILING EXIT: ratchet the stop to (running peak − trail_atr·ATR); exit when price hits it, or the move
    releases (round-trips birth), or session close. This HOLDS the fat tail instead of exiting on first pullback.
  · Costs via [[analytics.charges]]; results broken down BY TIME-OF-DAY (not just opening).

FIRST/in-sample-ish (H2-2025); rigorous = walk-forward OOS after the deep harvest. Reads only cached DayStory.
  PTIS_ENV=test python -m analytics.money_sim [--symbols A,B] [--trail 1.5] [--max-risk 4] [--require-tags T1,T2]
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from analytics.charges import CostModel
from intelligence import day_story as _ds


def _birth(text: str):
    p = (text or "").split()
    if len(p) >= 4 and "broke" in p:
        try:
            return (p[1] if p[1] in ("up", "down") else None), float(p[p.index("broke") + 1])
        except Exception:
            return None, None
    return None, None


def _phase(ts) -> str:
    m = ts.hour * 60 + ts.minute
    if m <= 585: return "open(9:15-9:45)"
    if m <= 690: return "morning(9:45-11:30)"
    if m <= 795: return "midday(11:30-13:15)"
    if m <= 885: return "afternoon(13:15-14:45)"
    return "late(14:45+)"


def simulate_symbol(symbol, interval, confirm_ext, cm, trail_atr=1.5, max_risk_atr=4.0,
                    require_tags=None, exit_mode="trail", stop_buffer=0.5) -> list[dict]:
    """exit_mode: 'trail' = mechanical trailing stop; 'observer' = the eye's MOVE-OBSERVER — hold while the
    move reads ALIVE, exit when state → EXHAUSTED (weakening after a run) or the move releases. The observer
    exit is the 'exit-when-story-changes' design; needs the RICH DayStory (state populated)."""
    files = _ds._files(symbol, interval)
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if df.empty:
        return []
    df["ts"] = pd.to_datetime(df["ts"])
    return _sim_frame(df, symbol, cm, confirm_ext, trail_atr, max_risk_atr, require_tags, exit_mode, stop_buffer)


def simulate_narrate(symbol, interval, n_sessions, confirm_ext, cm, trail_atr=1.5, max_risk_atr=4.0,
                     require_tags=None, exit_mode="trail", stop_buffer=0.5) -> list[dict]:
    """HIGHER-TIMEFRAME path: narrate `n_sessions` at `interval` (lite, cheap — few bars/session) → build
    the same frame → simulate. Not cached (DayStory is 5m only), but higher TFs have far fewer bars."""
    from intelligence.state_narrator import narrate_session
    from marketdata.candle_store import get_candle_store
    cs = get_candle_store()
    before = pd.Timestamp("2026-01-01")
    try:
        c_all = cs.get(symbol, interval, before - pd.Timedelta(days=n_sessions * 3 + 60), before)
    except Exception:
        return []
    if c_all is None or len(c_all) == 0:
        return []
    c_all = c_all.sort_values("ts").reset_index(drop=True)
    ts_all = pd.to_datetime(c_all["ts"])
    ds = ts_all[ts_all < before].dt.date
    dates = [str(d) for d in sorted(d for d, cnt in ds.value_counts().items() if cnt >= 15)][-n_sessions:]
    frames = []
    for d in dates:
        day = pd.Timestamp(d)
        win = c_all[(ts_all >= day - pd.Timedelta(days=10)) & (ts_all < day + pd.Timedelta(days=1))]
        try:
            steps = narrate_session(symbol, interval, d, candles=win, with_recall=False, lite=True)
            fr = _ds.steps_to_frame(steps)
            if not fr.empty:
                frames.append(fr)
        except Exception:
            continue
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True); df["ts"] = pd.to_datetime(df["ts"])
    return _sim_frame(df, symbol, cm, confirm_ext, trail_atr, max_risk_atr, require_tags, exit_mode, stop_buffer)


def _sim_frame(df, symbol, cm, confirm_ext, trail_atr, max_risk_atr, require_tags, exit_mode, stop_buffer=0.5) -> list[dict]:
    out = []
    for _, day in df.groupby("date"):
        day = day.sort_values("ts").reset_index(drop=True)
        bl = bd = None; bt = set()
        t = None                                        # active trade dict or None
        for _, r in day.iterrows():
            d, lvl = _birth(r.get("birth"))
            if d and lvl is not None:
                bd, bl = d, lvl
                try: bt = set(json.loads(r.get("tags") or "[]"))
                except Exception: bt = set()
            me = r.get("move_ext"); px = r.get("price"); atr = r.get("atr")
            live = me == me
            if not (px == px):
                continue
            if t is None:
                ok = (bl is not None and live and me >= confirm_ext
                      and (require_tags is None or require_tags <= bt) and atr == atr and atr > 0)
                if ok:
                    # STRUCTURAL stop: a tight buffer BELOW/ABOVE the break (birth level), NOT the wide
                    # entry-to-birth distance. With an early entry this makes risk small → big R on runs.
                    stop = (bl - stop_buffer * atr) if bd == "up" else (bl + stop_buffer * atr)
                    risk_atr = abs(px - stop) / atr
                    if risk_atr > max_risk_atr or risk_atr < 0.3:   # too wide (chasing) OR too tight (whipsaw)
                        continue
                    t = {"entry": float(px), "stop": float(stop), "dir": bd, "atr": float(atr),
                         "peak": float(px), "eff": float(stop), "time": _phase(r["ts"])}
            else:
                up = t["dir"] == "up"
                t["peak"] = max(t["peak"], px) if up else min(t["peak"], px)
                if exit_mode == "stall":
                    # MOVE-OBSERVER (robust): exit when the move STOPPED making new ground for K bars (it
                    # stalled) — holds through brief pauses, unlike the churny EXHAUSTED flag. K = trail_atr arg.
                    me_now = me if me == me else t.get("last_ext", 0.0)
                    if me_now > t.get("last_ext", -1) + 0.05:
                        t["last_ext"] = me_now; t["stall"] = 0
                    else:
                        t["stall"] = t.get("stall", 0) + 1
                    if t["stall"] >= int(trail_atr) or (not live):
                        out.append(_close(symbol, t, float(px), cm)); t = None; bl = None
                elif exit_mode == "observer":
                    # the MOVE-OBSERVER exit: hold while ALIVE, exit when it reads EXHAUSTED (run + weakening)
                    # or the story flips to BALANCE, or the move releases. A trail_atr backstop caps giveback.
                    st = str(r.get("state") or "")
                    trail = (t["peak"] - trail_atr * t["atr"]) if up else (t["peak"] + trail_atr * t["atr"])
                    t["eff"] = max(t["eff"], trail) if up else min(t["eff"], trail)
                    backstop = (px <= t["eff"]) if up else (px >= t["eff"])
                    if st in ("EXHAUSTED", "BALANCE") or (not live) or backstop:
                        exit_px = t["eff"] if backstop else float(px)
                        out.append(_close(symbol, t, exit_px, cm)); t = None; bl = None
                else:
                    trail = (t["peak"] - trail_atr * t["atr"]) if up else (t["peak"] + trail_atr * t["atr"])
                    t["eff"] = max(t["eff"], trail) if up else min(t["eff"], trail)
                    hit = (px <= t["eff"]) if up else (px >= t["eff"])
                    if hit or (not live):
                        exit_px = t["eff"] if hit else float(px)
                        out.append(_close(symbol, t, exit_px, cm)); t = None; bl = None
        if t is not None:                               # EOD
            out.append(_close(symbol, t, float(day["price"].iloc[-1]), cm))
    return [o for o in out if o]


def _close(symbol, t, exit_px, cm):
    entry, stop, tdir = t["entry"], t["stop"], t["dir"]
    risk = abs(entry - stop)
    if entry <= 0 or risk / entry < 0.002:
        return None
    profit = (exit_px - entry) if tdir == "up" else (entry - exit_px)
    gross_R = profit / risk
    cost_R = cm.cost_in_R(entry, stop, qty=1.0, exit=exit_px)
    return {"symbol": symbol, "dir": tdir, "time": t["time"], "gross_R": gross_R,
            "net_R": gross_R - cost_R, "cost_R": cost_R, "stop_pct": risk / entry * 100}


def report(trades) -> str:
    if not trades:
        return "no trades"
    df = pd.DataFrame(trades)
    def stat(d):
        return (f"n={len(d):<5} net {d['net_R'].mean():+.2f}R/trade  win {(d['net_R']>0).mean()*100:.0f}%  "
                f"total {d['net_R'].sum():+.0f}R  (gross {d['gross_R'].mean():+.2f}R, cost {d['cost_R'].mean():.2f}R)")
    L = [f"MONEY SIM (trailing exit, no-chase entry) — {len(df)} trades, {df['symbol'].nunique()} symbols",
         f"  ALL: {stat(df)}",
         f"  {'PROFITABLE net' if df['net_R'].mean()>0.03 else 'scratch' if df['net_R'].mean()>-0.03 else 'LOSES net'}",
         "  by TIME-OF-DAY:"]
    order = ["open(9:15-9:45)", "morning(9:45-11:30)", "midday(11:30-13:15)", "afternoon(13:15-14:45)", "late(14:45+)"]
    for ph in order:
        d = df[df["time"] == ph]
        if len(d) >= 20:
            L.append(f"    {ph:<22} {stat(d)}")
    w, l = df[df.net_R > 0]["net_R"], df[df.net_R <= 0]["net_R"]
    if len(w) and len(l):
        L.append(f"  asymmetry: avg WIN {w.mean():+.2f}R vs avg LOSS {l.mean():+.2f}R | best {df.net_R.max():+.1f}R")
    return "\n".join(L)


def _symbols():
    from pathlib import Path
    root = Path("data/lake/daystory")
    return sorted({p.parent.parent.name.split("=", 1)[1] for p in root.glob("**/interval=*/*.parquet")}) if root.exists() else []


if __name__ == "__main__":
    a = sys.argv[1:]
    def opt(f, d): return type(d)(a[a.index(f) + 1]) if f in a else d
    cm = CostModel(slippage_bps=opt("--slip-bps", 5.0))
    req = set(a[a.index("--require-tags") + 1].split(",")) if "--require-tags" in a else None
    syms = (a[a.index("--symbols") + 1].split(",") if "--symbols" in a else _symbols())
    print(f"money sim: {len(syms)} sym | trail {opt('--trail',1.5)}ATR | max-risk {opt('--max-risk',4.0)}ATR | "
          f"selection={sorted(req) if req else 'ALL'}...", flush=True)
    exit_mode = a[a.index("--exit") + 1] if "--exit" in a else "trail"
    narr_tf = a[a.index("--narrate") + 1] if "--narrate" in a else None   # higher-TF: narrate at this interval
    last = int(opt("--last", 120))
    trades = []
    for s in syms:
        try:
            if narr_tf:
                trades += simulate_narrate(s, narr_tf, last, opt("--confirm", 2.5), cm, opt("--trail", 1.5),
                                           opt("--max-risk", 4.0), req, exit_mode=exit_mode, stop_buffer=opt("--stop-buffer", 0.5))
            else:
                trades += simulate_symbol(s, "5m", opt("--confirm", 2.5), cm, opt("--trail", 1.5),
                                          opt("--max-risk", 4.0), req, exit_mode=exit_mode, stop_buffer=opt("--stop-buffer", 0.5))
        except Exception:
            continue
    print(report(trades))
