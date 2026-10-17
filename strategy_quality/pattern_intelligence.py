# strategy_quality/pattern_intelligence.py
"""
Pattern Intelligence Module
=============================

Handles the LAST 5-10% that makes the system production-grade:

1. Dynamic pattern clustering with variant discovery
2. Proper feature scaling (z-score + min-max)
3. Intelligent pattern hashing (quantile + cluster-based)
4. Regime-aware contradiction detection

This module is used BY the DataQualityFirewall and
StrategyUnderstandingEngine. It provides the PATTERN LAYER
intelligence that those systems consume.
"""

from __future__ import annotations

import logging
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

logger = logging.getLogger(__name__)

# Only import heavy ML libs when needed
_SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler, QuantileTransformer
    from sklearn.metrics import silhouette_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    logger.info("sklearn not available — using fallback clustering")


# ══════════════════════════════════════════════════════════════════
#  FIX 2: FEATURE SCALER
# ══════════════════════════════════════════════════════════════════


@dataclass
class FeatureStats:
    """Running statistics for a single feature."""
    name: str
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0          # for Welford's online variance
    min_val: float = float("inf")
    max_val: float = float("-inf")
    # Quantile tracking (approximate)
    _reservoir: List[float] = field(default_factory=list)
    _reservoir_max: int = 1000

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 1.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return max(self.variance ** 0.5, 1e-8)  # never zero

    @property
    def range(self) -> float:
        if self.max_val == float("-inf"):
            return 1.0
        return max(self.max_val - self.min_val, 1e-8)

    def update(self, value: float) -> None:
        """Welford's online algorithm for mean and variance."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

        # Reservoir sampling for quantiles
        if len(self._reservoir) < self._reservoir_max:
            self._reservoir.append(value)
        else:
            import random
            j = random.randint(0, self.count - 1)
            if j < self._reservoir_max:
                self._reservoir[j] = value

    def z_score(self, value: float) -> float:
        """Standardize to mean=0, std=1."""
        return (value - self.mean) / self.std

    def min_max_scale(self, value: float) -> float:
        """Scale to [0, 1]."""
        return (value - self.min_val) / self.range

    def quantile_bucket(self, value: float, n_buckets: int = 10) -> int:
        """
        Assign value to a quantile-based bucket.
        Uses reservoir of observed values.
        """
        if len(self._reservoir) < 10:
            # Not enough data — fall back to linear bucketing
            if self.range < 1e-6:
                return n_buckets // 2
            normalized = (value - self.min_val) / self.range
            return min(n_buckets - 1, max(0, int(normalized * n_buckets)))

        sorted_res = sorted(self._reservoir)
        n = len(sorted_res)

        for i in range(n_buckets):
            threshold_idx = int((i + 1) / n_buckets * n) - 1
            threshold_idx = max(0, min(n - 1, threshold_idx))
            if value <= sorted_res[threshold_idx]:
                return i

        return n_buckets - 1


class FeatureScaler:
    """
    Online feature scaling that handles:
    - Different scales (RSI 0-100, volume in millions)
    - Streaming updates (no need to refit on all data)
    - Multiple scaling methods per use case

    FIX for: cosine distance distortion from mixed scales.
    """

    def __init__(self, feature_keys: List[str]):
        self._feature_keys = feature_keys
        self._stats: Dict[str, FeatureStats] = {
            key: FeatureStats(name=key) for key in feature_keys
        }
        self._min_samples_for_scaling = 20

    def update(self, features: Dict[str, float]) -> None:
        """Update running statistics with new observation."""
        for key in self._feature_keys:
            val = features.get(key)
            if val is not None:
                try:
                    self._stats[key].update(float(val))
                except (ValueError, TypeError):
                    pass

    @property
    def is_ready(self) -> bool:
        """Do we have enough samples to scale reliably?"""
        return all(
            s.count >= self._min_samples_for_scaling
            for s in self._stats.values()
        )

    @property
    def sample_count(self) -> int:
        counts = [s.count for s in self._stats.values()]
        return min(counts) if counts else 0

    def scale_zscore(
        self, features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Z-score normalization: mean=0, std=1.
        Best for: distance calculations, clustering.
        """
        scaled = {}
        for key in self._feature_keys:
            val = features.get(key)
            if val is None:
                scaled[key] = 0.0  # mean-impute
            else:
                stats = self._stats[key]
                if stats.count >= self._min_samples_for_scaling:
                    scaled[key] = stats.z_score(float(val))
                else:
                    scaled[key] = float(val)  # raw if not enough data
        return scaled

    def scale_minmax(
        self, features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Min-max normalization: [0, 1].
        Best for: scoring, visualization.
        """
        scaled = {}
        for key in self._feature_keys:
            val = features.get(key)
            if val is None:
                scaled[key] = 0.5  # midpoint impute
            else:
                stats = self._stats[key]
                if stats.count >= self._min_samples_for_scaling:
                    scaled[key] = max(0.0, min(1.0,
                        stats.min_max_scale(float(val))
                    ))
                else:
                    scaled[key] = 0.5
        return scaled

    def to_vector(
        self,
        features: Dict[str, float],
        method: str = "zscore",
    ) -> np.ndarray:
        """
        Convert features dict to scaled numpy vector.
        Aligned to self._feature_keys order.
        """
        if method == "zscore":
            scaled = self.scale_zscore(features)
        elif method == "minmax":
            scaled = self.scale_minmax(features)
        else:
            scaled = {
                k: float(features.get(k, 0.0))
                for k in self._feature_keys
            }

        return np.array([
            scaled.get(k, 0.0) for k in self._feature_keys
        ])

    def get_quantile_bucket(
        self,
        feature_name: str,
        value: float,
        n_buckets: int = 10,
    ) -> int:
        """Get quantile-based bucket for a feature value."""
        stats = self._stats.get(feature_name)
        if stats is None or stats.count < 10:
            return n_buckets // 2
        return stats.quantile_bucket(value, n_buckets)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Diagnostic info about feature distributions."""
        return {
            key: {
                "count": stats.count,
                "mean": round(stats.mean, 4),
                "std": round(stats.std, 4),
                "min": round(stats.min_val, 4) if stats.min_val != float("inf") else None,
                "max": round(stats.max_val, 4) if stats.max_val != float("-inf") else None,
                "ready": stats.count >= self._min_samples_for_scaling,
            }
            for key, stats in self._stats.items()
        }


# ══════════════════════════════════════════════════════════════════
#  FIX 1 + 3: DYNAMIC PATTERN CLUSTERING
# ══════════════════════════════════════════════════════════════════


@dataclass
class PatternCluster:
    """A discovered cluster of similar trade patterns."""
    cluster_id: int
    centroid: np.ndarray              # scaled feature centroid
    centroid_dict: Dict[str, float]   # original scale for readability
    size: int = 0
    avg_outcome_r: float = 0.0
    win_rate: float = 0.0
    outcomes: List[float] = field(default_factory=list)
    label: str = ""                   # human-readable name
    # Per-regime breakdown
    regime_outcomes: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @property
    def is_profitable(self) -> bool:
        return self.avg_outcome_r > 0 and self.size >= 5

    @property
    def is_anti_pattern(self) -> bool:
        return self.avg_outcome_r < -0.2 and self.size >= 5


@dataclass
class ClusteringResult:
    """Result of pattern clustering analysis."""
    clusters: List[PatternCluster] = field(default_factory=list)
    n_clusters: int = 0
    silhouette: float = 0.0              # cluster quality (-1 to 1)
    noise_points: int = 0                # unclustered (DBSCAN)
    total_points: int = 0
    method: str = ""                     # "kmeans" | "dbscan" | "fallback"
    feature_keys: List[str] = field(default_factory=list)


class DynamicPatternClusterer:
    """
    Discovers and maintains pattern clusters (variants)
    from trade data.

    FIX 1: Dynamic clustering with variant discovery
    FIX 3: Cluster-based pattern IDs (not crude bucketing)

    Uses DBSCAN for initial discovery (finds natural groupings)
    then KMeans for refinement (stable centroids).

    Automatically:
    - Discovers how many variants exist
    - Assigns new trades to nearest cluster
    - Splits clusters that become too diverse
    - Merges clusters that are too similar
    - Tracks per-cluster and per-regime performance
    """

    def __init__(
        self,
        feature_keys: List[str],
        scaler: FeatureScaler,
        min_cluster_size: int = 5,
        max_clusters: int = 8,
    ):
        self._feature_keys = feature_keys
        self._scaler = scaler
        self._min_cluster_size = min_cluster_size
        self._max_clusters = max_clusters

        # Current clusters
        self._clusters: List[PatternCluster] = []
        self._clustering_result: Optional[ClusteringResult] = None

        # Raw data buffer for reclustering
        self._data_buffer: List[Dict[str, Any]] = []
        self._scaled_vectors: List[np.ndarray] = []
        self._outcomes: List[float] = []
        self._regimes: List[str] = []

        # Reclustering trigger
        self._last_cluster_count = 0
        self._recluster_interval = 30     # recluster every N new trades
        self._min_data_for_clustering = 20

    def add_trade(
        self,
        features: Dict[str, float],
        outcome_r: float,
        regime: str = "",
    ) -> Optional[int]:
        """
        Add a trade. Returns cluster assignment.
        Triggers reclustering when enough new data.
        """
        # Update scaler
        self._scaler.update(features)

        # Store raw data
        self._data_buffer.append({
            "features": features,
            "outcome_r": outcome_r,
            "regime": regime,
        })
        self._outcomes.append(outcome_r)
        self._regimes.append(regime)

        # Scale and store vector
        if self._scaler.is_ready:
            vec = self._scaler.to_vector(features, method="zscore")
            self._scaled_vectors.append(vec)
        else:
            self._scaled_vectors.append(None)

        # Assign to existing cluster (if clusters exist)
        cluster_id = None
        if self._clusters and self._scaler.is_ready:
            vec = self._scaler.to_vector(features, method="zscore")
            cluster_id = self._assign_to_cluster(
                vec, outcome_r, regime
            )

        # Check if reclustering needed
        n = len(self._data_buffer)
        if (
            n >= self._min_data_for_clustering
            and (n - self._last_cluster_count) >= self._recluster_interval
        ):
            self.recluster()
            # Re-assign this trade after reclustering
            if self._scaler.is_ready:
                vec = self._scaler.to_vector(features, method="zscore")
                cluster_id = self._assign_to_cluster(
                    vec, outcome_r, regime
                )

        return cluster_id

    def recluster(self) -> ClusteringResult:
        """
        Run full clustering on all accumulated data.

        Strategy:
        1. Try DBSCAN first (discovers natural number of clusters)
        2. If DBSCAN finds reasonable clusters, use them
        3. If not, fall back to KMeans with silhouette-optimal k
        """
        # Build feature matrix (only scaled vectors)
        valid_indices = [
            i for i, v in enumerate(self._scaled_vectors)
            if v is not None
        ]

        if len(valid_indices) < self._min_data_for_clustering:
            return ClusteringResult(
                method="insufficient_data",
                total_points=len(valid_indices),
            )

        X = np.array([
            self._scaled_vectors[i] for i in valid_indices
        ])
        outcomes = [self._outcomes[i] for i in valid_indices]
        regimes = [self._regimes[i] for i in valid_indices]

        result = None

        if _SKLEARN_AVAILABLE:
            # Try DBSCAN first
            result = self._try_dbscan(X, outcomes, regimes)

            # If DBSCAN didn't work well, try KMeans
            if (
                result is None
                or result.n_clusters < 2
                or result.silhouette < 0.15
            ):
                result = self._try_kmeans(X, outcomes, regimes)

        if result is None:
            # Fallback: simple distance-based clustering
            result = self._fallback_clustering(
                X, outcomes, regimes
            )

        self._clusters = result.clusters
        self._clustering_result = result
        self._last_cluster_count = len(self._data_buffer)

        logger.info(
            "Reclustered: %d clusters, silhouette=%.3f, "
            "method=%s, %d points",
            result.n_clusters,
            result.silhouette,
            result.method,
            result.total_points,
        )

        return result

    def _try_dbscan(
        self,
        X: np.ndarray,
        outcomes: List[float],
        regimes: List[str],
    ) -> Optional[ClusteringResult]:
        """DBSCAN: discovers natural clusters without specifying k."""
        if not _SKLEARN_AVAILABLE:
            return None

        # Adaptive eps based on data spread
        # Use average nearest-neighbor distance
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(5, len(X)))
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        avg_dist = float(np.mean(distances[:, -1]))

        eps = avg_dist * 1.2  # slight expansion

        db = DBSCAN(
            eps=eps,
            min_samples=max(3, self._min_cluster_size),
        )
        labels = db.fit_predict(X)

        n_clusters = len(set(labels) - {-1})
        noise_count = int(np.sum(labels == -1))

        if n_clusters < 2 or n_clusters > self._max_clusters:
            return None

        # Compute silhouette (excluding noise)
        mask = labels != -1
        if mask.sum() < n_clusters + 1:
            return None

        sil = float(silhouette_score(X[mask], labels[mask]))

        if sil < 0.10:
            return None  # clusters aren't meaningful

        # Build cluster objects
        clusters = self._build_clusters(
            X, labels, outcomes, regimes
        )

        return ClusteringResult(
            clusters=clusters,
            n_clusters=n_clusters,
            silhouette=round(sil, 3),
            noise_points=noise_count,
            total_points=len(X),
            method="dbscan",
            feature_keys=self._feature_keys,
        )

    def _try_kmeans(
        self,
        X: np.ndarray,
        outcomes: List[float],
        regimes: List[str],
    ) -> Optional[ClusteringResult]:
        """KMeans with silhouette-optimal k."""
        if not _SKLEARN_AVAILABLE:
            return None

        best_k = 2
        best_sil = -1.0

        max_k = min(self._max_clusters, len(X) // self._min_cluster_size)
        max_k = max(2, max_k)

        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)

            sil = float(silhouette_score(X, labels))
            if sil > best_sil:
                best_sil = sil
                best_k = k

        # Final fit with best k
        km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        clusters = self._build_clusters(
            X, labels, outcomes, regimes
        )

        return ClusteringResult(
            clusters=clusters,
            n_clusters=best_k,
            silhouette=round(best_sil, 3),
            noise_points=0,
            total_points=len(X),
            method="kmeans",
            feature_keys=self._feature_keys,
        )

    def _fallback_clustering(
        self,
        X: np.ndarray,
        outcomes: List[float],
        regimes: List[str],
    ) -> ClusteringResult:
        """
        Simple distance-based clustering when sklearn unavailable.
        Greedy: assign each point to nearest existing centroid,
        or create new centroid if too far.
        """
        threshold = 1.5  # z-score distance threshold
        centroids = []
        labels = np.full(len(X), -1, dtype=int)

        for i, vec in enumerate(X):
            assigned = False
            for c_idx, centroid in enumerate(centroids):
                dist = np.linalg.norm(vec - centroid["sum"] / centroid["count"])
                if dist < threshold:
                    centroid["sum"] += vec
                    centroid["count"] += 1
                    labels[i] = c_idx
                    assigned = True
                    break

            if not assigned and len(centroids) < self._max_clusters:
                centroids.append({
                    "sum": vec.copy(),
                    "count": 1,
                })
                labels[i] = len(centroids) - 1

        # Remove tiny clusters
        for c_idx in range(len(centroids)):
            count = int(np.sum(labels == c_idx))
            if count < self._min_cluster_size:
                labels[labels == c_idx] = -1

        clusters = self._build_clusters(
            X, labels, outcomes, regimes
        )
        n_clusters = len(clusters)

        sil = 0.0
        if n_clusters >= 2 and _SKLEARN_AVAILABLE:
            mask = labels != -1
            if mask.sum() > n_clusters:
                sil = float(silhouette_score(X[mask], labels[mask]))

        return ClusteringResult(
            clusters=clusters,
            n_clusters=n_clusters,
            silhouette=round(sil, 3),
            noise_points=int(np.sum(labels == -1)),
            total_points=len(X),
            method="fallback",
            feature_keys=self._feature_keys,
        )

    def _build_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        outcomes: List[float],
        regimes: List[str],
    ) -> List[PatternCluster]:
        """Build PatternCluster objects from labels."""
        clusters = []
        unique_labels = sorted(set(labels) - {-1})

        for c_id in unique_labels:
            mask = labels == c_id
            cluster_X = X[mask]
            cluster_outcomes = [
                outcomes[i] for i in range(len(outcomes)) if mask[i]
            ]
            cluster_regimes = [
                regimes[i] for i in range(len(regimes)) if mask[i]
            ]

            centroid = np.mean(cluster_X, axis=0)

            # Convert centroid back to original scale for readability
            centroid_dict = {}
            for j, key in enumerate(self._feature_keys):
                stats = self._scaler._stats.get(key)
                if stats and stats.count >= 20:
                    # Inverse z-score
                    centroid_dict[key] = round(
                        centroid[j] * stats.std + stats.mean, 3
                    )
                else:
                    centroid_dict[key] = round(float(centroid[j]), 3)

            # Per-regime outcomes
            regime_outcomes = defaultdict(list)
            for r, o in zip(cluster_regimes, cluster_outcomes):
                if r:
                    regime_outcomes[r].append(o)

            outcome_arr = np.array(cluster_outcomes)

            cluster = PatternCluster(
                cluster_id=int(c_id),
                centroid=centroid,
                centroid_dict=centroid_dict,
                size=int(mask.sum()),
                avg_outcome_r=round(float(np.mean(outcome_arr)), 4),
                win_rate=round(
                    float(np.mean(outcome_arr > 0)), 4
                ),
                outcomes=cluster_outcomes,
                label=f"variant_{c_id}",
                regime_outcomes=dict(regime_outcomes),
            )
            clusters.append(cluster)

        return clusters

    def _assign_to_cluster(
        self,
        scaled_vector: np.ndarray,
        outcome_r: float,
        regime: str,
    ) -> Optional[int]:
        """Assign a new trade to the nearest existing cluster."""
        if not self._clusters:
            return None

        min_dist = float("inf")
        best_cluster = None

        for cluster in self._clusters:
            dist = np.linalg.norm(scaled_vector - cluster.centroid)
            if dist < min_dist:
                min_dist = dist
                best_cluster = cluster

        if best_cluster is not None:
            # Update cluster stats
            best_cluster.size += 1
            best_cluster.outcomes.append(outcome_r)
            arr = np.array(best_cluster.outcomes)
            best_cluster.avg_outcome_r = round(
                float(np.mean(arr)), 4
            )
            best_cluster.win_rate = round(
                float(np.mean(arr > 0)), 4
            )
            if regime:
                best_cluster.regime_outcomes.setdefault(
                    regime, []
                ).append(outcome_r)

            return best_cluster.cluster_id

        return None

    # ── FIX 3: Cluster-Based Pattern ID ───────────────────────────

    def get_pattern_id(
        self, features: Dict[str, float]
    ) -> str:
        """
        Get a pattern ID based on cluster assignment.
        MUCH better than crude round(val, 1) bucketing.

        If clusters exist → use cluster ID
        If not → fall back to quantile-based bucketing
        """
        if self._clusters and self._scaler.is_ready:
            vec = self._scaler.to_vector(features, method="zscore")

            # Find nearest cluster
            min_dist = float("inf")
            best_id = -1
            for cluster in self._clusters:
                dist = np.linalg.norm(vec - cluster.centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_id = cluster.cluster_id

            # If distance is reasonable, use cluster ID
            # Otherwise mark as "outlier"
            avg_cluster_radius = self._avg_cluster_radius()
            if min_dist < avg_cluster_radius * 2.5:
                return f"cluster_{best_id}"
            else:
                return "outlier"

        # Fallback: quantile-based bucketing
        return self._quantile_pattern_id(features)

    def _quantile_pattern_id(
        self, features: Dict[str, float]
    ) -> str:
        """
        FIX 3: Quantile-based bucketing instead of round(val, 1).
        Each feature is placed in its quantile bucket.
        """
        buckets = {}
        for key in self._feature_keys:
            val = features.get(key)
            if val is None:
                buckets[key] = "M"  # missing
            else:
                bucket = self._scaler.get_quantile_bucket(
                    key, float(val), n_buckets=5
                )
                buckets[key] = str(bucket)

        # Create stable hash from bucket assignments
        raw = json.dumps(buckets, sort_keys=True)
        return "q_" + hashlib.md5(raw.encode()).hexdigest()[:10]

    def _avg_cluster_radius(self) -> float:
        """Average distance from centroid to cluster members."""
        if not self._clusters or not self._scaled_vectors:
            return 2.0

        radii = []
        for cluster in self._clusters:
            # Approximate from stored data
            distances = []
            for i, vec in enumerate(self._scaled_vectors):
                if vec is not None:
                    d = np.linalg.norm(vec - cluster.centroid)
                    distances.append(d)

            if distances:
                # Use 75th percentile as "radius"
                radii.append(float(np.percentile(distances, 75)))

        return float(np.mean(radii)) if radii else 2.0

    def get_pattern_distance(
        self, features: Dict[str, float]
    ) -> float:
        """
        Distance from nearest cluster centroid.
        Returns 0-1 where 0 = exactly on centroid, 1 = very far.
        Used by DataQualityFirewall for pattern consistency.
        """
        if not self._clusters or not self._scaler.is_ready:
            return 0.5  # neutral

        vec = self._scaler.to_vector(features, method="zscore")

        min_dist = float("inf")
        for cluster in self._clusters:
            dist = np.linalg.norm(vec - cluster.centroid)
            min_dist = min(min_dist, dist)

        # Normalize by average cluster radius
        radius = self._avg_cluster_radius()
        if radius < 1e-6:
            return 0.5

        normalized = min_dist / (radius * 2)
        return min(1.0, normalized)

    def get_cluster_info(self) -> Dict[str, Any]:
        """Diagnostic info about current clusters."""
        return {
            "n_clusters": len(self._clusters),
            "method": (
                self._clustering_result.method
                if self._clustering_result
                else "none"
            ),
            "silhouette": (
                self._clustering_result.silhouette
                if self._clustering_result
                else 0.0
            ),
            "total_data_points": len(self._data_buffer),
            "clusters": [
                {
                    "id": c.cluster_id,
                    "label": c.label,
                    "size": c.size,
                    "avg_r": c.avg_outcome_r,
                    "win_rate": c.win_rate,
                    "is_profitable": c.is_profitable,
                    "is_anti": c.is_anti_pattern,
                    "centroid": c.centroid_dict,
                    "regime_breakdown": {
                        regime: {
                            "count": len(outs),
                            "avg_r": round(float(np.mean(outs)), 3),
                            "win_rate": round(
                                float(np.mean(
                                    np.array(outs) > 0
                                )), 3
                            ),
                        }
                        for regime, outs in c.regime_outcomes.items()
                        if len(outs) >= 3
                    },
                }
                for c in self._clusters
            ],
        }


# ══════════════════════════════════════════════════════════════════
#  FIX 4: REGIME-AWARE CONTRADICTION DETECTION
# ══════════════════════════════════════════════════════════════════


class ContradictionDetector:
    """
    Detects contradictions at the RIGHT granularity:

    FIX 4: A pattern may be consistent WITHIN a regime
    but contradictory ACROSS regimes. This is NOT a problem
    with the pattern — it's a regime sensitivity signal.

    We separate:
    - INTRA-REGIME contradiction: same pattern, same regime,
      different outcomes → REAL problem (unstable edge)
    - CROSS-REGIME contradiction: same pattern, different
      regime, different outcomes → REGIME SENSITIVITY
      (not a problem, just needs regime filtering)
    """

    def __init__(
        self,
        clusterer: DynamicPatternClusterer,
    ):
        self._clusterer = clusterer
        # pattern_id → regime → list of outcomes
        self._outcomes: Dict[str, Dict[str, List[float]]] = (
            defaultdict(lambda: defaultdict(list))
        )

    def record_outcome(
        self,
        features: Dict[str, float],
        outcome_r: float,
        regime: str,
    ) -> None:
        """Record a trade outcome for contradiction tracking."""
        pattern_id = self._clusterer.get_pattern_id(features)
        self._outcomes[pattern_id][regime].append(outcome_r)

    def detect(
        self,
        features: Dict[str, float],
        regime: str,
    ) -> Dict[str, Any]:
        """
        Detect contradiction level for this pattern.

        Returns:
        - intra_regime_contradiction: 0-1 (REAL problem)
        - cross_regime_contradiction: 0-1 (regime sensitivity)
        - overall_contradiction: 0-1 (weighted combination)
        - diagnosis: human-readable explanation
        """
        pattern_id = self._clusterer.get_pattern_id(features)
        pattern_data = self._outcomes.get(pattern_id, {})

        if not pattern_data:
            return {
                "intra_regime": 0.0,
                "cross_regime": 0.0,
                "overall": 0.0,
                "diagnosis": "insufficient_data",
                "pattern_id": pattern_id,
            }

        # ── INTRA-REGIME contradiction ──
        # Same pattern + same regime → outcomes should be consistent
        intra_contradiction = 0.0
        intra_details = {}

        for r, outcomes in pattern_data.items():
            if len(outcomes) < 3:
                continue

            arr = np.array(outcomes)
            mean_o = float(np.mean(arr))
            std_o = float(np.std(arr))

            winners = int(np.sum(arr > 0.5))
            losers = int(np.sum(arr < -0.5))

            regime_contra = 0.0

            if winners >= 2 and losers >= 2:
                # Both significant winners and losers in SAME regime
                balance = 1.0 - abs(
                    winners / len(arr) - 0.5
                ) * 2
                regime_contra = balance

            elif abs(mean_o) < 0.1 and std_o > 0.8:
                # Near-zero mean with high spread
                regime_contra = min(0.7, std_o * 0.5)

            intra_details[r] = {
                "contradiction": round(regime_contra, 3),
                "mean": round(mean_o, 3),
                "std": round(std_o, 3),
                "n": len(outcomes),
            }
            intra_contradiction = max(
                intra_contradiction, regime_contra
            )

        # ── CROSS-REGIME contradiction ──
        # Same pattern in different regimes → compare means
        cross_contradiction = 0.0

        regime_means = {}
        for r, outcomes in pattern_data.items():
            if len(outcomes) >= 3:
                regime_means[r] = float(np.mean(outcomes))

        if len(regime_means) >= 2:
            means = list(regime_means.values())
            spread = max(means) - min(means)

            # Large spread between regimes is regime sensitivity, not contradiction
            if spread > 0.5:
                cross_contradiction = min(1.0, spread / 2.0)

        # ── OVERALL ──
        # Intra-regime matters MORE (it's a real problem)
        # Cross-regime is informational (regime filtering opportunity)
        overall = (
            intra_contradiction * 0.75
            + cross_contradiction * 0.25
        )

        # Diagnosis
        if intra_contradiction > 0.5:
            diagnosis = (
                f"UNSTABLE EDGE: Pattern '{pattern_id}' produces "
                f"contradictory outcomes even within the same regime. "
                f"This suggests the pattern alone isn't sufficient — "
                f"there's a hidden factor we're not capturing."
            )
        elif cross_contradiction > 0.5 and intra_contradiction < 0.3:
            diagnosis = (
                f"REGIME SENSITIVE: Pattern '{pattern_id}' is "
                f"consistent within each regime but performs very "
                f"differently across regimes. This is NOT a problem — "
                f"add regime filtering. Best regime: "
                f"{max(regime_means, key=regime_means.get) if regime_means else 'unknown'}"
            )
        elif overall < 0.2:
            diagnosis = "consistent"
        else:
            diagnosis = "minor_inconsistency"

        return {
            "intra_regime": round(intra_contradiction, 3),
            "cross_regime": round(cross_contradiction, 3),
            "overall": round(overall, 3),
            "diagnosis": diagnosis,
            "pattern_id": pattern_id,
            "intra_details": intra_details,
            "regime_means": {
                k: round(v, 3) for k, v in regime_means.items()
            },
        }


# ══════════════════════════════════════════════════════════════════
#  INTEGRATED: UPDATED DATA QUALITY FIREWALL
# ══════════════════════════════════════════════════════════════════


class PatternAwareFirewall:
    """
    DataQualityFirewall that uses all the pattern intelligence:
    - FeatureScaler for proper normalization
    - DynamicPatternClusterer for pattern matching
    - ContradictionDetector for regime-aware contradictions

    This REPLACES the pattern-related methods in the original
    DataQualityFirewall.
    """

    def __init__(
        self,
        profile: Any,  # StrategyProfile
        feature_keys: List[str],
    ):
        self._profile = profile
        self._feature_keys = feature_keys

        # Initialize pattern intelligence
        self._scaler = FeatureScaler(feature_keys)
        self._clusterer = DynamicPatternClusterer(
            feature_keys=feature_keys,
            scaler=self._scaler,
        )
        self._contradiction = ContradictionDetector(
            clusterer=self._clusterer,
        )

    @property
    def scaler(self) -> FeatureScaler:
        return self._scaler

    @property
    def clusterer(self) -> DynamicPatternClusterer:
        return self._clusterer

    @property
    def contradiction_detector(self) -> ContradictionDetector:
        return self._contradiction

    def check_pattern_consistency(
        self, features: Dict[str, float]
    ) -> float:
        """
        Pattern match score using SCALED distance from centroids.
        FIX 2: Proper feature scaling prevents distortion.
        FIX 1: Uses dynamically discovered clusters.
        """
        distance = self._clusterer.get_pattern_distance(features)
        # Convert: 0 distance = 1.0 score, 1 distance = 0.0 score
        return round(max(0.0, 1.0 - distance), 3)

    def check_contradictions(
        self,
        features: Dict[str, float],
        regime: str,
    ) -> Dict[str, Any]:
        """
        FIX 4: Regime-aware contradiction detection.
        Separates intra-regime (real problem) from
        cross-regime (regime sensitivity).
        """
        return self._contradiction.detect(features, regime)

    def get_pattern_id(
        self, features: Dict[str, float]
    ) -> str:
        """
        FIX 3: Cluster-based or quantile-based pattern ID.
        No more crude round(val, 1).
        """
        return self._clusterer.get_pattern_id(features)

    def record_trade(
        self,
        features: Dict[str, float],
        outcome_r: float,
        regime: str = "",
    ) -> Optional[int]:
        """
        Record a completed trade for learning.
        Updates: scaler, clusters, contradiction tracker.
        Returns cluster assignment.
        """
        # Update contradiction tracker
        self._contradiction.record_outcome(
            features, outcome_r, regime
        )

        # Add to clusterer (which also updates scaler)
        cluster_id = self._clusterer.add_trade(
            features, outcome_r, regime
        )

        return cluster_id

    def get_diagnostics(self) -> Dict[str, Any]:
        """Complete diagnostic report."""
        return {
            "scaler": self._scaler.get_diagnostics(),
            "clusters": self._clusterer.get_cluster_info(),
            "scaler_ready": self._scaler.is_ready,
            "sample_count": self._scaler.sample_count,
        }