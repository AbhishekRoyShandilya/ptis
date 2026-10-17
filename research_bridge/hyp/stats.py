"""Statistics of the kill machine (Phase A3, 2026-10-03). One implementation, used by every belief test and by the
Event Knowledge Base, so a number in a verdict card and a number the Finmedia agent quotes come from the same code.

  cluster_t       t on CLUSTER MEANS (events on the same date / week / quarter are not independent)
  clustered_ols   OLS with cluster-robust (sandwich) standard errors - the INCREMENTAL test: does the new information
                  still matter after the controls PTIS already knows (momentum, volatility, liquidity, sector, regime)?
  bh_fdr          Benjamini-Hochberg false-discovery control for a FAMILY of tests (e.g. sectors x macro drivers)
  years_won, top_share, placebo_percentile
Ported from scripts/research/ear_drift_test.cluster_t and scripts/research/sector_dir_layer1.clustered_t.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _norm_p(t: float) -> float:
    """two-sided p for a t treated as normal (cluster counts here are >= 20 by gate)"""
    if t is None or not np.isfinite(t):
        return 1.0
    return math.erfc(abs(t) / math.sqrt(2))


def cluster_t(x, groups) -> dict:
    """mean of the event values, and t computed on the means of each cluster (date, week or quarter)"""
    x = pd.Series(np.asarray(x, float)); g = pd.Series(np.asarray(groups))
    ok = x.notna(); x, g = x[ok], g[ok]
    if not len(x):
        return {"n": 0, "clusters": 0, "mean": None, "median": None, "t": None, "p": 1.0}
    cm = x.groupby(g.values).mean()
    t = float(cm.mean() / (cm.std() / np.sqrt(len(cm)))) if len(cm) > 2 and cm.std() > 0 else None
    return {"n": int(len(x)), "clusters": int(len(cm)), "mean": float(x.mean()), "median": float(x.median()),
            "q25": float(x.quantile(.25)), "q75": float(x.quantile(.75)), "hit_rate_pct": float((x > 0).mean() * 100),
            "cluster_mean": float(cm.mean()), "t": t, "p": _norm_p(t)}


def clustered_ols(y, X: pd.DataFrame, groups, names=None) -> dict:
    """OLS y ~ 1 + X with cluster-robust SEs. Returns coefficient, t and p per column of X."""
    df = pd.concat([pd.Series(np.asarray(y, float), name="__y"), X.reset_index(drop=True),
                    pd.Series(np.asarray(groups), name="__g")], axis=1).dropna()
    if len(df) < X.shape[1] + 3:
        return {"n": int(len(df)), "coef": {}, "t": {}, "p": {}}
    A = np.c_[np.ones(len(df)), df[list(X.columns)].to_numpy(float)]; yy = df["__y"].to_numpy(float)
    b, *_ = np.linalg.lstsq(A, yy, rcond=None); e = yy - A @ b
    AtAi = np.linalg.pinv(A.T @ A); meat = np.zeros((A.shape[1], A.shape[1]))
    for _, idx in df.groupby("__g").indices.items():
        s = A[idx].T @ e[idx]; meat += np.outer(s, s)
    G = df["__g"].nunique(); k = A.shape[1]; n = len(df)
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))                  # small-sample cluster correction
    V = adj * AtAi @ meat @ AtAi
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    cols = ["const"] + list(X.columns)
    tt = {c: (float(b[i] / se[i]) if se[i] > 0 else None) for i, c in enumerate(cols)}
    return {"n": int(n), "clusters": int(G), "coef": {c: float(b[i]) for i, c in enumerate(cols)}, "t": tt,
            "p": {c: _norm_p(tt[c]) for c in cols}}


def bh_fdr(pvals, q: float = 0.10):
    """Benjamini-Hochberg: returns (reject flags, adjusted q-values) in the input order"""
    p = np.asarray(pvals, float); m = len(p)
    if not m:
        return np.array([], bool), np.array([])
    o = np.argsort(p); ranked = p[o] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]; adj = np.clip(adj, 0, 1)
    out = np.empty(m); out[o] = adj
    return out <= q, out


def years_won(x, dates, min_n: int = 3) -> dict:
    """share of calendar years (with >= min_n events) whose mean is > 0"""
    s = pd.Series(np.asarray(x, float), index=pd.to_datetime(pd.Series(dates)).dt.year.values).dropna()
    by = s.groupby(level=0).agg(["mean", "size"]); by = by[by["size"] >= min_n]
    return {"years": int(len(by)), "won": int((by["mean"] > 0).sum()),
            "share": float((by["mean"] > 0).mean()) if len(by) else None,
            "by_year": {int(k): round(float(v), 4) for k, v in by["mean"].items()}}


def top_share(x, k: int = 10):
    """share of the TOTAL result produced by the k best events (> 1 means the rest lose money together).
    None when the total is not positive (the gate then fails on its own)."""
    s = pd.Series(np.asarray(x, float)).dropna()
    tot = s.sum()
    if not len(s) or tot <= 0:
        return None
    return float(s.nlargest(k).sum() / tot)


def placebo_percentile(effect: float, null_draws) -> float | None:
    d = np.asarray([v for v in null_draws if v is not None and np.isfinite(v)], float)
    if not len(d) or effect is None:
        return None
    return float((d < effect).mean() * 100)
