"""
intelligence.trajectory -- Layer 3: per-birth TRAJECTORIES (multi-timeframe stage sequences), not
single-row snapshots. Every earlier single-parameter pilot this session (trendline alone, compression
alone, HTF-flip alone) landed at breakeven because a "birth" was treated as one independent row with a
scalar outcome; this reframes the unit of research as the story leading into a birth, its stage-by-stage
evolution, and how it resolves.

Reuses `scripts.research.full_stack_model.iter_day_births` as the seed-event list (birth detection is
unchanged) and extends each seed into a bounded pre/post window, run-length-encoded per timeframe.

NOTE: day_story's persisted rich fields dropped `structure` (state_narrator._fill_rich's own docstring:
"only `structure` survived (as a tag)" -- energy/fuel/loc/leg_ext_atr/... were carried, structure was
not). Own-TF structure here is therefore recomputed fresh via `lifecycle_frame` on raw candles, exactly
the same way the 15m/1h/1d HTF stacks are (never persisted at those TFs either) -- callers merge it in
via `own_structure`; this module does not read candles itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.charges import CostModel
from analytics.money_sim import _birth as _parse_birth
from analytics.money_sim import _close

_HTF_PREFIXES = ("htf15", "htf1h", "htf1d")

RLE_CAP_SEGMENTS = 8
_STORE_DIR = Path("data/lake/_trajectories")


def _f(v) -> float:
    try:
        fv = float(v)
        return fv if fv == fv else float("nan")
    except Exception:
        return float("nan")


def relation(birth_str) -> str:
    """WITH/COUNTER-trend read off the birth prose (mirrors full_stack_model.relation -- duplicated
    rather than imported so intelligence/ doesn't depend on scripts/research/)."""
    s = str(birth_str)
    if "COUNTER-TREND" in s:
        return "COUNTER_TREND"
    if "WITH the" in s:
        return "WITH_TREND"
    return "NO_TREND_READ"


def rle_encode(labels, cap_segments: int | None = RLE_CAP_SEGMENTS) -> list:
    """Run-length-encode a categorical sequence into [[label, n_bars], ...]. NaN/None -> "". If
    cap_segments given, keep only the most recent N segments, folding older ones into a leading
    ["...", total_bars] overflow bucket -- bounds record size for symbols/days with very long
    unbroken runs, without discarding the information that a long run happened."""
    out: list = []
    for lab in labels:
        lab = "" if lab is None or (isinstance(lab, float) and lab != lab) else str(lab)
        if out and out[-1][0] == lab:
            out[-1][1] += 1
        else:
            out.append([lab, 1])
    if cap_segments is not None and len(out) > cap_segments:
        overflow = sum(n for _, n in out[:-cap_segments])
        out = [["...", overflow]] + out[-cap_segments:]
    return out


def find_origin(win: pd.DataFrame, energy_col: str = "energy", structure_col: str | None = "structure",
                htf_col: str | None = None, gap_col: str | None = None):
    """The hybrid lookback horizon's 'semantic' half: `win` is an already bar-count-CAPPED,
    chronological (oldest->newest) per-bar frame ending at the birth bar -- this function does not
    itself cap anything, the caller's window slice is the safety valve. Scans backward from the birth
    bar for the MOST RECENT of: an energy exhaustion-reset, a structure-segment change, an HTF regime
    flip, or a gap bar. Returns (origin_ts, origin_reason, origin_freshness_bars). If none of those
    occurred inside `win`, reason='cap_reached' (informative on its own -- the current chapter outlives
    what's visible) and origin = the window's first bar."""
    if win.empty:
        return None, "cap_reached", 0
    n = len(win)
    candidates = []  # (bars_back_from_last_bar, reason) -- smallest bars_back wins (most recent)
    if energy_col in win.columns:
        e = win[energy_col].astype(str).to_numpy()
        for k in range(n - 1, 0, -1):
            if e[k - 1] != e[k]:
                reason = "exhaustion_reset" if (e[k] == "EXHAUSTING" and e[k - 1] in ("DORMANT", "BUILDING")) \
                    else "energy_change"
                candidates.append((n - 1 - k, reason))
                break
    if structure_col and structure_col in win.columns:
        s = win[structure_col].astype(str).to_numpy()
        for k in range(n - 1, 0, -1):
            if s[k - 1] != s[k]:
                candidates.append((n - 1 - k, "structural_break"))
                break
    if htf_col and htf_col in win.columns:
        h = win[htf_col].astype(str).to_numpy()
        for k in range(n - 1, 0, -1):
            if h[k - 1] != h[k]:
                candidates.append((n - 1 - k, "regime_change"))
                break
    if gap_col and gap_col in win.columns:
        g = win[gap_col].to_numpy()
        for k in range(n - 1, 0, -1):
            if bool(g[k]):
                candidates.append((n - 1 - k, "gap"))
                break
    if not candidates:
        return win["ts"].iloc[0], "cap_reached", n - 1
    bars_back, reason = min(candidates, key=lambda c: c[0])
    idx = n - 1 - bars_back
    return win["ts"].iloc[idx], reason, bars_back


def stack_htf(spine_ts: pd.Series, htf_frame: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time join of a full lifecycle_frame output onto a lower-TF spine of timestamps --
    generalizes story_context.htf_trend_series's merge_asof(direction='backward') pattern from a single
    column to the whole HTF frame (energy/structure/fuel/loc/leg_ext_atr/...). No lookahead: each spine
    bar sees only the last COMPLETED higher-TF bar as of its own timestamp."""
    left = pd.DataFrame({"ts": pd.to_datetime(spine_ts).astype("datetime64[ns]")})
    right = htf_frame.copy()
    right["ts"] = pd.to_datetime(right["ts"]).astype("datetime64[ns]")
    return pd.merge_asof(left.sort_values("ts"), right.sort_values("ts"), on="ts", direction="backward")


@dataclass
class TrajectorySummary:
    trajectory_id: str
    sym: str
    date: str
    ts: pd.Timestamp
    dir: str
    birth_type: str
    rel: str

    # ── outcome (unchanged semantics from full_stack_model) ──
    peak: float = float("nan")
    outcome: str = ""
    bars_to_peak: int = 0
    bars_alive: int = 0
    mae_atr: float = float("nan")
    end_energy: str = ""

    # ── own-TF (5m) AT BIRTH ──
    room_atr: float = float("nan")
    n_near: float = float("nan")
    energy: str = ""
    fuel: str = ""
    loc: str = ""
    leg_ext_atr: float = float("nan")
    leg_bars: float = float("nan")
    in_run: float = float("nan")
    rsi: float = float("nan")
    d_res_atr: float = float("nan")
    d_sup_atr: float = float("nan")
    vwap_dist: float = float("nan")
    rvol_now: float = float("nan")
    bbw_now: float = float("nan")
    with_trend: int = 0
    room_clear: int = 0
    room_wall: int = 0
    at_open: int = 0
    strong_effort: int = 0
    gap: int = 0
    above_vwap_tag: int = 0

    # ── own-TF (5m) PRE/POST stage sequence ──
    pre_energy_rle: str = "[]"
    pre_structure_rle: str = "[]"
    pre_fuel_rle: str = "[]"
    post_energy_rle: str = "[]"
    post_structure_rle: str = "[]"
    post_fuel_rle: str = "[]"
    pre_window_bars: int = 0
    post_window_bars: int = 0

    # ── own-TF (5m) origin (hybrid horizon) ──
    origin_ts: object = None
    origin_reason: str = ""
    origin_freshness_bars: int = 0

    # ── HTF stacks (filled by the extended build_trajectory, build-order step 5) ──
    htf15_energy: str = ""; htf15_structure: str = ""; htf15_fuel: str = ""; htf15_loc: str = ""
    htf15_d_res_atr: float = float("nan"); htf15_d_sup_atr: float = float("nan")
    htf15_pre_rle: str = "[]"; htf15_post_rle: str = "[]"
    htf15_origin_reason: str = ""; htf15_origin_freshness_bars: int = 0

    htf1h_energy: str = ""; htf1h_structure: str = ""; htf1h_fuel: str = ""; htf1h_loc: str = ""
    htf1h_d_res_atr: float = float("nan"); htf1h_d_sup_atr: float = float("nan")
    htf1h_pre_rle: str = "[]"; htf1h_post_rle: str = "[]"
    htf1h_origin_reason: str = ""; htf1h_origin_freshness_bars: int = 0

    htf1d_energy: str = ""; htf1d_structure: str = ""; htf1d_fuel: str = ""; htf1d_loc: str = ""
    htf1d_d_res_atr: float = float("nan"); htf1d_d_sup_atr: float = float("nan")
    htf1d_pre_rle: str = "[]"; htf1d_post_rle: str = "[]"
    htf1d_origin_reason: str = ""; htf1d_origin_freshness_bars: int = 0

    # ── micro-structure (filled build-order step 5; the quantitative substrate for "behavior") ──
    n_failed_attempts: int = 0
    attempt_span_bars: int = 0
    rvol_ratio_attempts_vs_break: float = float("nan")

    # ── trade-quality / opportunity-cost (filled build-order step 5, via analytics.money_sim) ──
    mfe_atr: float = float("nan")
    bars_to_mfe: int = 0
    confirm_entry_ext_atr: float = float("nan")
    logical_stop_atr: float = float("nan")
    realized_R: float = float("nan")

    # ── archetype (filled by trajectory_cluster.py; label stays empty until Scientist-validated) ──
    cluster_id: object = None
    cluster_label: str = ""

    def key(self) -> str:
        return self.trajectory_id


def _trade_quality(post: pd.DataFrame, birth_row, trail_atr: float = 1.5, max_risk_atr: float = 4.0,
                   stop_buffer: float = 0.5, confirm_ext: float = 2.5, cost_model=None) -> dict:
    """Trade-quality / opportunity-cost fields (plan review addition #4): reuses
    analytics.money_sim's exact entry/stop/trailing-exit/cost logic, scoped to one already-identified
    birth's post-window (direction/level are already known here -- no birth-scanning needed, unlike
    money_sim's whole-day loop). Answers 'was this a good TRADE', not just 'did the move run'."""
    out = dict(mfe_atr=float("nan"), bars_to_mfe=0, confirm_entry_ext_atr=float("nan"),
               logical_stop_atr=float("nan"), realized_R=float("nan"))
    if post.empty:
        return out
    me = post["move_ext"].to_numpy(float)
    finite = np.isfinite(me)
    if finite.any():
        idx = int(np.nanargmax(np.where(finite, me, -np.inf)))
        out["mfe_atr"] = float(me[idx]); out["bars_to_mfe"] = idx

    cm = cost_model or CostModel()
    bd, bl = _parse_birth(birth_row.get("birth"))
    if bd is None or bl is None:
        return out

    px_arr = post["price"].to_numpy(float)
    atr_arr = post["atr"].to_numpy(float)
    mv_arr = me   # already extracted above as post["move_ext"].to_numpy(float)

    t = None
    for k in range(len(post)):
        px, atr, mv = float(px_arr[k]), float(atr_arr[k]), float(mv_arr[k])
        live = mv == mv
        if t is None:
            if mv == mv and mv >= confirm_ext and atr == atr and atr > 0:
                stop = (bl - stop_buffer * atr) if bd == "up" else (bl + stop_buffer * atr)
                risk_atr = abs(px - stop) / atr
                if risk_atr > max_risk_atr or risk_atr < 0.3:      # too wide (chasing) OR too tight (whipsaw)
                    continue
                t = {"entry": px, "stop": stop, "dir": bd, "atr": atr, "peak": px, "eff": stop, "time": ""}
                out["confirm_entry_ext_atr"] = mv
                out["logical_stop_atr"] = risk_atr
        else:
            up = t["dir"] == "up"
            t["peak"] = max(t["peak"], px) if up else min(t["peak"], px)
            trail = (t["peak"] - trail_atr * t["atr"]) if up else (t["peak"] + trail_atr * t["atr"])
            t["eff"] = max(t["eff"], trail) if up else min(t["eff"], trail)
            hit = (px <= t["eff"]) if up else (px >= t["eff"])
            if hit or not live:
                exit_px = t["eff"] if hit else px
                closed = _close("", t, exit_px, cm)
                out["realized_R"] = closed["net_R"] if closed else float("nan")
                return out
    if t is not None:                                              # EOD/window-end, still open
        closed = _close("", t, _f(post.iloc[-1].get("price")), cm)
        out["realized_R"] = closed["net_R"] if closed else float("nan")
    return out


DEFAULT_HTF_CAPS = {"htf15": 150, "htf1h": 80, "htf1d": 40}   # native-bar caps per TF (plan defaults)


def build_trajectory(seed: dict, day_g: pd.DataFrame, sym: str, date, pre_bars: int = 60,
                     post_bars: int = 120, htf_frames: dict | None = None, htf_caps: dict | None = None,
                     trail_atr: float = 1.5, max_risk_atr: float = 4.0,
                     stop_buffer: float = 0.5, confirm_ext: float = 2.5, cost_model=None) -> TrajectorySummary:
    """One trajectory per seed birth. `seed` = one entry from full_stack_model.iter_day_births
    (i, bt, d, peak, outcome, j_end). `day_g` = the day's day_story frame, already sorted by ts --
    OPTIONALLY enriched by the caller with a `structure` column (own-TF, day_story doesn't persist it --
    see module docstring). `htf_frames` = optional {"htf15"/"htf1h"/"htf1d": lifecycle_frame output}, in
    each TF's OWN native bars (build-order step 5) -- computed once per symbol by the caller and reused
    across every birth. `htf_caps` overrides DEFAULT_HTF_CAPS (native-bar pre/post window per TF -- e.g.
    40 daily bars is a very different calendar span than 40 5m bars, deliberately). Every HTF/structure
    block below degrades gracefully to the dataclass defaults if not supplied -- this is what lets Pass A
    (build-order step 3) and Pass B (step 6) call the SAME function with different arguments."""
    i, j_end = seed["i"], seed["j_end"]
    n = len(day_g)
    r = day_g.iloc[i]
    lo = max(0, i - pre_bars)
    hi = min(n, i + post_bars, j_end if j_end > i else i + post_bars)
    pre = day_g.iloc[lo:i]
    post = day_g.iloc[i:hi]
    try:
        tags = set(json.loads(r.get("tags") or "[]"))
    except Exception:
        tags = set()
    ts = pd.Timestamp(r["ts"])
    trajectory_id = f"{sym}|{date}|{ts:%H:%M}|{seed['d']}"

    fields: dict = dict(
        trajectory_id=trajectory_id, sym=sym, date=str(date), ts=ts, dir=seed["d"],
        birth_type=seed["bt"], rel=relation(r.get("birth")),
        peak=seed["peak"], outcome=seed["outcome"], bars_to_peak=int(hi - i), bars_alive=int(j_end - i),
        room_atr=_f(r.get("room_atr")), n_near=_f(r.get("n_near")),
        energy=r.get("energy") or "", fuel=r.get("fuel") or "", loc=r.get("loc") or "",
        leg_ext_atr=_f(r.get("leg_ext_atr")), leg_bars=_f(r.get("leg_bars")), in_run=_f(r.get("in_run")),
        rsi=_f(r.get("rsi")), d_res_atr=_f(r.get("d_res_atr")), d_sup_atr=_f(r.get("d_sup_atr")),
        vwap_dist=_f(r.get("vwap_dist")), rvol_now=_f(r.get("rvol_now")), bbw_now=_f(r.get("bbw_now")),
        with_trend=int("WITH_TREND" in tags), room_clear=int("ROOM_CLEAR" in tags),
        room_wall=int("ROOM_WALL" in tags), at_open=int("AT_OPEN" in tags),
        strong_effort=int("STRONG_EFFORT" in tags), gap=int(any(t in tags for t in ("GAP_UP", "GAP_DOWN"))),
        above_vwap_tag=int("ABOVE_VWAP" in tags),
        pre_energy_rle=json.dumps(rle_encode(pre["energy"].tolist())),
        pre_fuel_rle=json.dumps(rle_encode(pre["fuel"].tolist())),
        post_energy_rle=json.dumps(rle_encode(post["energy"].tolist())),
        post_fuel_rle=json.dumps(rle_encode(post["fuel"].tolist())),
        pre_window_bars=int(i - lo), post_window_bars=int(hi - i),
    )

    if "structure" in day_g.columns:
        fields["pre_structure_rle"] = json.dumps(rle_encode(pre["structure"].tolist()))
        fields["post_structure_rle"] = json.dumps(rle_encode(post["structure"].tolist()))
        if not pre.empty:
            o_ts, o_reason, o_fresh = find_origin(pre, energy_col="energy", structure_col="structure")
            fields["origin_ts"], fields["origin_reason"], fields["origin_freshness_bars"] = o_ts, o_reason, o_fresh
    elif not pre.empty:
        o_ts, o_reason, o_fresh = find_origin(pre, energy_col="energy", structure_col=None)
        fields["origin_ts"], fields["origin_reason"], fields["origin_freshness_bars"] = o_ts, o_reason, o_fresh

    # HTF stacks -- present only if the caller supplied htf_frames (build-order step 5). Each HTF layer
    # is sliced from its OWN native bars (NOT the 5m spine): a 60-bar 5m window is ~5 hours, which would
    # make htf1d's RLE degenerate (barely more than one daily bar). The window CAP is per-TF (htf_caps),
    # so "150 bars of 15m context" and "40 bars of 1d context" mean genuinely different, TF-appropriate
    # calendar spans -- this is the hybrid lookback horizon's safety-valve half; find_origin below is the
    # semantic half, reading off where the current segment in THAT window began.
    for pfx in _HTF_PREFIXES:
        hf = (htf_frames or {}).get(pfx)
        if hf is None or hf.empty:
            continue
        # Self-caching index: boolean-masking the WHOLE htf frame (hf[hf_ts <= ts]) on every trajectory
        # made the first version of this O(n_htf_bars) per birth -- for 4k+ births/symbol that measured
        # ~892s for ONE symbol (would be ~10hr for 40). `_ts_ns` is computed once (mutates `hf` in place,
        # which is the SAME object reused across every birth of this symbol -- see trajectory_model.py)
        # and every subsequent birth does an O(log n) searchsorted instead.
        if "_ts_ns" not in hf.columns:
            hf["_ts_ns"] = pd.to_datetime(hf["ts"]).astype("int64")
        cap = (htf_caps or DEFAULT_HTF_CAPS).get(pfx, 150)
        ts_arr = hf["_ts_ns"].to_numpy()
        idx_birth = int(np.searchsorted(ts_arr, np.int64(pd.Timestamp(ts).value), side="right"))
        if idx_birth == 0:
            continue
        rb = hf.iloc[idx_birth - 1]
        fields[f"{pfx}_energy"] = rb.get("energy") or ""
        fields[f"{pfx}_structure"] = rb.get("structure") or ""
        fields[f"{pfx}_fuel"] = rb.get("fuel") or ""
        fields[f"{pfx}_loc"] = rb.get("loc") or ""
        fields[f"{pfx}_d_res_atr"] = _f(rb.get("d_res_atr"))
        fields[f"{pfx}_d_sup_atr"] = _f(rb.get("d_sup_atr"))

        hf_pre = hf.iloc[max(0, idx_birth - cap):idx_birth]
        post_end_ts = post["ts"].iloc[-1] if not post.empty else ts
        idx_post_end = int(np.searchsorted(ts_arr, np.int64(pd.Timestamp(post_end_ts).value), side="right"))
        hf_post = hf.iloc[idx_birth:min(idx_post_end, idx_birth + cap)]
        fields[f"{pfx}_pre_rle"] = json.dumps(rle_encode(hf_pre["energy"].tolist()))
        fields[f"{pfx}_post_rle"] = json.dumps(rle_encode(hf_post["energy"].tolist()))
        if not hf_pre.empty:
            _, o_reason, o_fresh = find_origin(hf_pre, energy_col="energy",
                                               structure_col="structure" if "structure" in hf_pre.columns else None)
            fields[f"{pfx}_origin_reason"] = o_reason
            fields[f"{pfx}_origin_freshness_bars"] = o_fresh

    # micro-structure (plan review addition #1's mechanism): day_story doesn't persist raw leg_dir, but
    # leg_ext_atr's SIGN already encodes "extended vs retraced through the last confirmed pivot" (see
    # lifecycle.py: leg_ext[i] = (close-pivot_price)/atr * leg_dir[i], normally >=0; goes negative only
    # when price undercuts/exceeds the last swing pivot) -- a negative-sign run is a failed-attempt proxy,
    # no new detector needed. This is the quantitative substrate clustering can later group into
    # "behaviors" -- see the plan's explicit rejection of hand-authoring a behavior vocabulary up front.
    if not pre.empty and "leg_ext_atr" in pre.columns:
        le = pre["leg_ext_atr"].to_numpy(float)
        neg = le < 0
        fields["n_failed_attempts"] = int(sum(1 for k in range(1, len(le)) if neg[k] and not neg[k - 1]))
        if neg.any():
            first_neg = int(np.argmax(neg))
            fields["attempt_span_bars"] = int(len(le) - first_neg)
            if "rvol_now" in pre.columns:
                rv_pre = pre["rvol_now"].to_numpy(float)
                attempts_rvol = np.nanmean(rv_pre[neg])
                break_rvol = _f(r.get("rvol_now"))
                if break_rvol == break_rvol and break_rvol > 0:
                    fields["rvol_ratio_attempts_vs_break"] = float(attempts_rvol / break_rvol)

    fields.update(_trade_quality(post, r, trail_atr=trail_atr, max_risk_atr=max_risk_atr,
                                 stop_buffer=stop_buffer, confirm_ext=confirm_ext, cost_model=cost_model))

    return TrajectorySummary(**fields)


class TrajectoryStore:
    """Parquet, year-partitioned, dedup on trajectory_id keep-last -- mirrors day_story.write_frame's
    merge pattern. Volume (per-bar RLE sequences, not just scalars) rules out jsonl at this scale."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else _STORE_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def save_batch(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.copy()
        df["_year"] = pd.to_datetime(df["date"]).dt.year
        for year, g in df.groupby("_year"):
            g = g.drop(columns=["_year"])
            path = self.root / f"{year}.parquet"
            if path.exists():
                g = pd.concat([pd.read_parquet(path), g], ignore_index=True)
            g = g.drop_duplicates(subset=["trajectory_id"], keep="last")
            g.to_parquet(path, index=False)

    def load(self, symbols: list | None = None) -> pd.DataFrame:
        files = sorted(self.root.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if symbols:
            df = df[df["sym"].isin(symbols)]
        return df


def spine_for(trajectory_id: str) -> pd.DataFrame:
    """Tier-2 on-demand rebuild: re-read the symbol's persisted 5m day_story for the trajectory's own
    day, for human inspection of one example (daily-lifecycle audit, cluster description). Callers that
    need the HTF spine too already have candle_store access and can call lifecycle_frame themselves for
    the narrow span -- not duplicated here to keep this module free of a marketdata dependency."""
    from intelligence import day_story as ds
    sym, date, hhmm, d = trajectory_id.split("|")
    files = ds._files(sym, "5m")
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"])
    return df[df["ts"].dt.date.astype(str) == date].sort_values("ts").reset_index(drop=True)
