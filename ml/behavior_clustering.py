"""
Behavior Clustering
====================
Groups trades into behavioral archetypes using unsupervised clustering.
Reveals hidden patterns like "patient disciplined trades" vs "reactive overrides".
Uses K-Means with a rule-based fallback for environments without scikit-learn.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BehaviorCluster:
    cluster_id: int
    label: str                  # human-readable archetype name
    trade_count: int
    win_rate: float
    avg_pnl: float
    avg_discipline: float
    avg_hold_minutes: float
    avg_override_count: float
    dominant_emotion: Optional[str]
    trade_ids: list = field(default_factory=list)   # which trades belong here


@dataclass
class ClusteringResult:
    clusters: list              # list[BehaviorCluster]
    n_clusters: int
    method: str                 # "kmeans" | "rule_based"
    silhouette_score: float     # quality metric -1 to 1 (0 if rule-based)
    feature_names: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "method": self.method,
            "silhouette_score": round(self.silhouette_score, 3),
            "clusters": [
                {"cluster_id": c.cluster_id, "label": c.label,
                 "trade_count": c.trade_count, "win_rate": round(c.win_rate, 4),
                 "avg_pnl": round(c.avg_pnl, 2),
                 "avg_discipline": round(c.avg_discipline, 2),
                 "avg_hold_minutes": round(c.avg_hold_minutes, 1),
                 "avg_override_count": round(c.avg_override_count, 2),
                 "dominant_emotion": c.dominant_emotion}
                for c in self.clusters
            ],
        }


# ---------------------------------------------------------------------------
# Feature extraction (standalone, no ML dependency)
# ---------------------------------------------------------------------------

def _extract_features(trade) -> Optional[dict]:
    """Extract numeric behavioral features from a single trade."""
    pnl = getattr(trade, "pnl", None)
    if pnl is None:
        return None
    disc = getattr(trade, "discipline_score", None) or 50.0
    ovr  = len(getattr(trade, "overrides", []))
    em   = getattr(trade, "emotion_pre", None)
    em_str = str(em.value if hasattr(em, "value") else em) if em else "neutral"
    # Risky emotions get value 1, calm = 0
    risky = {"revenge", "fomo", "frustrated", "euphoric", "anxious", "fearful"}
    em_risk = 1.0 if em_str.lower() in risky else 0.0

    hold_min = 0.0
    et = getattr(trade, "entry_time", None)
    xt = getattr(trade, "exit_time", None)
    if et and xt:
        hold_min = max((xt - et).total_seconds() / 60, 0.0)

    risk_pct = getattr(trade, "risk_pct", None) or 1.0

    return {
        "discipline": disc,
        "override_count": float(ovr),
        "emotion_risk": em_risk,
        "hold_min": hold_min,
        "risk_pct": float(risk_pct),
        "pnl_sign": 1.0 if pnl > 0 else 0.0,
    }


FEATURE_ORDER = ["discipline", "override_count", "emotion_risk", "hold_min", "risk_pct", "pnl_sign"]


def _normalize(rows: list[list[float]]) -> list[list[float]]:
    if not rows:
        return rows
    n_feat = len(rows[0])
    means = [statistics.mean(r[i] for r in rows) for i in range(n_feat)]
    stds  = [max(statistics.stdev(r[i] for r in rows), 1e-9) for i in range(n_feat)]
    return [[(v - means[i]) / stds[i] for i, v in enumerate(row)] for row in rows]


def _archetype_label(avg_disc: float, avg_ovr: float, win_rate: float,
                     avg_emotion_risk: float) -> str:
    """Assign a descriptive label to a cluster centroid."""
    if avg_disc >= 75 and avg_ovr < 0.5:
        return "Disciplined Executor"
    if avg_disc < 50 and avg_ovr >= 1.5:
        return "Reactive Overrider"
    if win_rate >= 0.60 and avg_disc >= 60:
        return "Consistent Winner"
    if win_rate < 0.35:
        return "Struggling / Overfit"
    if avg_emotion_risk > 0.5:
        return "Emotionally Driven"
    return "Balanced Trader"


# ---------------------------------------------------------------------------
# Rule-based clustering (no scikit-learn required)
# ---------------------------------------------------------------------------

class RuleBasedBehaviorClusterer:
    """Splits trades into 3 archetypes using thresholds — no ML needed."""

    def cluster(self, trades: list) -> ClusteringResult:
        buckets: dict[str, list] = {"disciplined": [], "reactive": [], "mixed": []}
        for t in trades:
            f = _extract_features(t)
            if f is None:
                continue
            if f["discipline"] >= 70 and f["override_count"] < 1:
                buckets["disciplined"].append(t)
            elif f["discipline"] < 50 or f["override_count"] >= 2:
                buckets["reactive"].append(t)
            else:
                buckets["mixed"].append(t)

        clusters = []
        for cid, (name, ts) in enumerate(buckets.items()):
            if not ts:
                continue
            pnls  = [t.pnl for t in ts if getattr(t, "pnl", None) is not None]
            discs = [getattr(t, "discipline_score", 50) or 50 for t in ts]
            ovrs  = [len(getattr(t, "overrides", [])) for t in ts]
            wins  = sum(1 for p in pnls if p > 0)
            wr    = wins / len(pnls) if pnls else 0.0
            holds = []
            for t in ts:
                et = getattr(t, "entry_time", None)
                xt = getattr(t, "exit_time", None)
                if et and xt:
                    holds.append((xt - et).total_seconds() / 60)
            em_counts: dict[str, int] = {}
            for t in ts:
                em = getattr(t, "emotion_pre", None)
                if em:
                    key = str(em.value if hasattr(em, "value") else em)
                    em_counts[key] = em_counts.get(key, 0) + 1
            dom_em = max(em_counts, key=em_counts.get) if em_counts else None
            avg_disc = statistics.mean(discs) if discs else 0.0
            avg_ovr  = statistics.mean(ovrs) if ovrs else 0.0
            label = _archetype_label(avg_disc, avg_ovr, wr, 0.0)
            clusters.append(BehaviorCluster(
                cluster_id=cid, label=label, trade_count=len(ts),
                win_rate=wr,
                avg_pnl=statistics.mean(pnls) if pnls else 0.0,
                avg_discipline=avg_disc,
                avg_hold_minutes=statistics.mean(holds) if holds else 0.0,
                avg_override_count=avg_ovr,
                dominant_emotion=dom_em,
                trade_ids=[str(getattr(t, "id", "")) for t in ts],
            ))

        return ClusteringResult(clusters=clusters, n_clusters=len(clusters),
                                method="rule_based", silhouette_score=0.0,
                                feature_names=FEATURE_ORDER)


# ---------------------------------------------------------------------------
# K-Means clustering (requires scikit-learn)
# ---------------------------------------------------------------------------

class KMeansBehaviorClusterer:

    def __init__(self, n_clusters: int = 4):
        self._k = n_clusters

    def cluster(self, trades: list) -> ClusteringResult:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            logger.warning("scikit-learn not installed; falling back to rule-based clustering")
            return RuleBasedBehaviorClusterer().cluster(trades)

        data = []  # (trade, feature_dict)
        for t in trades:
            f = _extract_features(t)
            if f:
                data.append((t, f))
        if len(data) < self._k * 2:
            return RuleBasedBehaviorClusterer().cluster(trades)

        valid_trades = [d[0] for d in data]
        rows = [[f[k] for k in FEATURE_ORDER] for _, f in data]
        normed = _normalize(rows)

        km = KMeans(n_clusters=min(self._k, len(normed)), random_state=42, n_init=10)
        labels = km.fit_predict(normed)
        sil = silhouette_score(normed, labels) if len(set(labels)) > 1 else 0.0

        # Build clusters
        bucket: dict[int, list] = {}
        for t, lbl in zip(valid_trades, labels):
            bucket.setdefault(int(lbl), []).append(t)

        clusters = []
        for cid, ts in sorted(bucket.items()):
            pnls  = [t.pnl for t in ts if getattr(t, "pnl", None) is not None]
            discs = [getattr(t, "discipline_score", 50) or 50 for t in ts]
            ovrs  = [len(getattr(t, "overrides", [])) for t in ts]
            wins  = sum(1 for p in pnls if p > 0)
            wr    = wins / len(pnls) if pnls else 0.0
            holds = []
            for t in ts:
                et, xt = getattr(t, "entry_time", None), getattr(t, "exit_time", None)
                if et and xt:
                    holds.append((xt - et).total_seconds() / 60)
            em_counts: dict[str, int] = {}
            for t in ts:
                em = getattr(t, "emotion_pre", None)
                if em:
                    key = str(em.value if hasattr(em, "value") else em)
                    em_counts[key] = em_counts.get(key, 0) + 1
            avg_disc = statistics.mean(discs) if discs else 0.0
            avg_ovr  = statistics.mean(ovrs) if ovrs else 0.0
            avg_em_r = statistics.mean(
                1.0 if str(getattr(t, "emotion_pre", "") or "").lower() in
                {"revenge", "fomo", "frustrated", "euphoric", "anxious"} else 0.0
                for t in ts
            )
            label = _archetype_label(avg_disc, avg_ovr, wr, avg_em_r)
            clusters.append(BehaviorCluster(
                cluster_id=cid, label=label, trade_count=len(ts), win_rate=wr,
                avg_pnl=statistics.mean(pnls) if pnls else 0.0,
                avg_discipline=avg_disc,
                avg_hold_minutes=statistics.mean(holds) if holds else 0.0,
                avg_override_count=avg_ovr,
                dominant_emotion=max(em_counts, key=em_counts.get) if em_counts else None,
                trade_ids=[str(getattr(t, "id", "")) for t in ts],
            ))

        return ClusteringResult(clusters=clusters, n_clusters=len(clusters),
                                method="kmeans", silhouette_score=round(float(sil), 3),
                                feature_names=FEATURE_ORDER)


def get_behavior_clusterer(use_ml: bool = True, n_clusters: int = 4):
    return KMeansBehaviorClusterer(n_clusters) if use_ml else RuleBasedBehaviorClusterer()
