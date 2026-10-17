"""
intelligence.narrative — remember and compare SITUATIONS AS STORIES, not feature vectors.

The move upward from the 7-dim fingerprint (which flattens a situation into a point and throws
away the sequence). A trader doesn't see feature_1..feature_7 — he sees a STORY unfolding:
"gap up → pullback → held structure → compression → afternoon breakout → volume expansion →
resistance test." Two situations with the same fingerprint can be opposite stories; only the
ORDER tells them apart.

So a situation here is an ORDERED SEQUENCE of chapters (the same chapters move_read already
names), and we recall by NARRATIVE similarity — "what past stories does this rhyme with" —
returning the analogous past sessions themselves, not a probability. Humans reason by analogy.

Honest rails:
  • Goal is ANALOGY / understanding, NOT prediction. (Sequence-mining for an EDGE already
    capped out — archived story_* engines; this is a different, viable goal.)
  • Market-grounded: real past sessions + their real outcomes; the system stays objective.
  • Situation TYPES are meant to EMERGE and be named by the human — not pre-declared here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NARRATIVE_VERSION = "1.0"

# the chapter vocabulary (the story's grammar) — readable, ordered as they occur in a session
OPEN_TOK = {"up": "GAP_UP", "down": "GAP_DOWN", "flat": "FLAT_OPEN"}


def day_narrative(day: pd.DataFrame, feat: pd.DataFrame, prev_close: float,
                  pdh: float, atr0: float) -> list[str]:
    """Ordered chapter-tokens for ONE session (the day's story). Reuses the detectors the
    move-observer already uses; collapses consecutive duplicates so it reads like a story."""
    from intelligence.move_nature import atr_zigzag

    o = day["open"].to_numpy(float); h = day["high"].to_numpy(float)
    l = day["low"].to_numpy(float); cl = day["close"].to_numpy(float)
    vol = day["volume"].to_numpy(float) if "volume" in day else np.zeros(len(day))
    atr = feat["atr"].to_numpy(float); vwap = feat["vwap"].to_numpy(float)
    rvol = feat["rvol"].to_numpy(float)
    n = len(day)
    if n < 8 or not np.isfinite(atr0) or atr0 <= 0:
        return []
    toks: list[str] = []

    def push(t):
        if not toks or toks[-1] != t:
            toks.append(t)

    # ── opening chapter ──
    gap = (o[0] - prev_close) / atr0 if np.isfinite(prev_close) else 0.0
    push(OPEN_TOK["up"] if gap > 0.4 else OPEN_TOK["down"] if gap < -0.4 else OPEN_TOK["flat"])
    k = min(6, n - 1)
    drive = (cl[k] - o[0]) / atr0
    if drive >= 1.5:
        push("OPEN_DRIVE_UP")
    elif drive <= -1.5:
        push("OPEN_DRIVE_DOWN")

    # ── swing structure through the day ──
    piv = atr_zigzag(h, l, atr)
    highs = [p for p in piv if p.kind == "H"]
    lows = [p for p in piv if p.kind == "L"]

    run_hi = h[0]; run_lo = l[0]
    comp_win = 8
    for i in range(1, n):
        a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else atr0
        # compression: a tight window relative to ATR
        if i >= comp_win:
            wr = (np.max(h[i - comp_win:i]) - np.min(l[i - comp_win:i])) / a
            if wr <= 1.2:
                push("COMPRESSION")
        # pullback inside an up-context (retrace from running high)
        if h[i - 1] >= run_hi and l[i] < run_hi - 0.8 * a and cl[i] < o[i]:
            push("PULLBACK")
        # VWAP cross
        if np.isfinite(vwap[i]) and np.isfinite(vwap[i - 1]):
            if cl[i - 1] < vwap[i - 1] and cl[i] > vwap[i]:
                push("VWAP_RECLAIM")
            elif cl[i - 1] > vwap[i - 1] and cl[i] < vwap[i]:
                push("VWAP_LOSS")
        # breakout to a new intraday high (with volume → expansion)
        if h[i] > run_hi + 1e-9 and i > comp_win:
            if cl[i] > o[i] and np.isfinite(rvol[i]) and rvol[i] >= 1.5:
                push("BREAKOUT_VOL")
            elif cl[i] > run_hi:
                push("BREAKOUT_UP")
        # breakdown to a new intraday low
        if l[i] < run_lo - 1e-9 and i > comp_win and cl[i] < o[i]:
            push("BREAKDOWN")
        # prior-day-high interaction
        if np.isfinite(pdh):
            if cl[i - 1] < pdh <= h[i] and cl[i] < pdh:
                push("PDH_REJECT")
            elif cl[i - 1] < pdh < cl[i]:
                push("PDH_BREAK")
        run_hi = max(run_hi, h[i]); run_lo = min(run_lo, l[i])

    # ── structure verdict from the last confirmed swings ──
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price:
            push("HH_HL_RECOVERY")
        elif highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price:
            push("LH_LL_ROLLOVER")
    return _collapse(toks)


def _collapse(toks: list[str]) -> list[str]:
    """A trader narrates oscillation as ONE chapter ('battled at the high'), not eight toggles.
    Collapse a maximal A-B-A-B alternation into a single CHOP/PDH_BATTLE chapter."""
    out: list[str] = []
    i = 0
    while i < len(toks):
        if i + 3 < len(toks) and toks[i] == toks[i + 2] and toks[i + 1] == toks[i + 3] \
                and toks[i] != toks[i + 1]:
            pair = {toks[i], toks[i + 1]}
            j = i
            while j + 2 < len(toks) and toks[j] == toks[j + 2]:
                j += 1
            out.append("PDH_BATTLE" if pair <= {"PDH_BREAK", "PDH_REJECT"} else "CHOP")
            i = j + 2
        else:
            out.append(toks[i]); i += 1
    return out


def build_day_narratives(symbols, interval: str = "5m", save: bool = True) -> pd.DataFrame:
    """Compute the story for every session across the universe, with the day's net move for
    grounding. Saved so live recall is fast."""
    from pathlib import Path
    from marketdata.candle_store import get_candle_store
    from marketdata import features as F
    rows = []
    for sym in symbols:
        c = get_candle_store().get(sym, interval, pd.Timestamp("2021-10-01"), pd.Timestamp("2026-01-01"))
        if c is None or len(c) < 1000:
            continue
        c = c.sort_values("ts").reset_index(drop=True)
        feat = F.compute_features(c, sym, interval)
        date = pd.to_datetime(c["ts"]).dt.date.to_numpy()
        days = list(dict.fromkeys(date.tolist()))
        cl = c["close"].to_numpy(float); hi = c["high"].to_numpy(float); op = c["open"].to_numpy(float)
        atr = feat["atr"].to_numpy(float)
        for kk, d in enumerate(days):
            if kk < 1:
                continue
            idx = np.where(date == d)[0]
            pidx = np.where(date == days[kk - 1])[0]
            if len(idx) < 8 or len(pidx) == 0:
                continue
            a0 = atr[idx[0]]
            toks = day_narrative(c.iloc[idx].reset_index(drop=True), feat.iloc[idx].reset_index(drop=True),
                                 float(cl[pidx[-1]]), float(hi[pidx].max()), float(a0))
            if len(toks) < 2:
                continue
            day_dir = float((cl[idx[-1]] - op[idx[0]]) / a0) if np.isfinite(a0) and a0 > 0 else np.nan
            rows.append({"sym": sym, "date": pd.Timestamp(d), "year": pd.Timestamp(d).year,
                         "tokens": toks, "story": " -> ".join(toks), "day_dir": round(day_dir, 1)})
    df = pd.DataFrame(rows)
    if save and not df.empty:
        out = Path("data/lake/narrative"); out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "day_narratives.parquet")
    return df


def recall_narratives(query_tokens: list[str], as_of, pool: pd.DataFrame | None = None,
                      k: int = 6) -> tuple[pd.DataFrame, list[str]]:
    """Point-in-time: the past sessions whose STORY most rhymes with this one (2026 sealed).
    Returns the analogous days + the shared motif (chapters common to most of them)."""
    from pathlib import Path
    if pool is None:
        p = Path("data/lake/narrative/day_narratives.parquet")
        pool = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    as_of = pd.Timestamp(as_of)
    past = pool[(pd.to_datetime(pool["date"]) < as_of) & (pool["year"] <= 2025)].copy()
    if past.empty:
        return past, []
    past["sim"] = past["tokens"].apply(lambda t: narrative_similarity(query_tokens, list(t)))
    top = past.sort_values("sim", ascending=False).head(k)
    # shared motif: chapters present in >=60% of the matches, ordered by their average position
    from collections import defaultdict
    cnt = defaultdict(int); pos = defaultdict(list)
    for t in top["tokens"]:
        seen = set()
        for j, tok in enumerate(t):
            if tok not in seen:
                cnt[tok] += 1; pos[tok].append(j / max(1, len(t) - 1)); seen.add(tok)
    motif = [tok for tok, c2 in cnt.items() if c2 >= 0.6 * len(top)]
    motif.sort(key=lambda tk: np.mean(pos[tk]))
    return top, motif


def _lcs_ratio(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n] / max(m, n)


def narrative_similarity(a: list[str], b: list[str]) -> float:
    """Order-aware story similarity: blend of shared-chapter overlap (set) and sequence
    alignment (LCS) — so the ORDER of the story matters, not just the chapters present."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    jacc = len(sa & sb) / len(sa | sb)
    return 0.5 * jacc + 0.5 * _lcs_ratio(a, b)
