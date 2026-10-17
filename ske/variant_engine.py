# ske/variant_engine.py
"""
Variant Engine — discovers trade clusters (variants & anti-variants)
within an SKO using HDBSCAN or K-Means clustering on feature vectors.

Variants are groups of trades sharing similar feature patterns with
distinct performance. Anti-variants are failure clusters (win_rate < 40%
AND negative expectancy).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ske.models import DistinguishingFeature, VariantInfo

logger = logging.getLogger(__name__)

# Anti-variant thresholds
_ANTI_VARIANT_WIN_RATE = 0.40
_ANTI_VARIANT_EXPECTANCY = 0.0


class VariantEngine:
    """
    Discovers and manages variants within a Strategy Knowledge Object.
    """

    def __init__(self, min_cluster_size: int = 5):
        self._min_cluster_size = min_cluster_size

    def discover_variants(
        self,
        feature_vectors: List[Dict[str, float]],
        outcomes: List[float],
        feature_names: Optional[List[str]] = None,
        learning_weights: Optional[List[float]] = None,
    ) -> List[VariantInfo]:
        """
        Cluster feature vectors into variants.

        Args:
            feature_vectors: List of feature dicts (one per trade).
            outcomes: R-multiples or PnL per trade (aligned with feature_vectors).
            feature_names: Ordered feature names to use. If None, derived from first vector.
            learning_weights: Quality gate weights per trade. Trades with
                weight < 0.5 are excluded from clustering; weights are used
                for weighted centroid computation.

        Returns:
            List of VariantInfo (including anti-variants).
        """
        # Pre-filter: exclude low-weight trades from clustering
        if learning_weights is not None:
            filtered = [
                (fv, o, w) for fv, o, w
                in zip(feature_vectors, outcomes, learning_weights)
                if w >= 0.5 and fv
            ]
            if filtered:
                feature_vectors = [x[0] for x in filtered]
                outcomes = [x[1] for x in filtered]
                learning_weights = [x[2] for x in filtered]
            else:
                logger.info("No trades with weight >= 0.5 for variant discovery")
                return []

        # Adaptive min_cluster_size based on effective N
        effective_min = self._min_cluster_size
        if learning_weights:
            sum_w = sum(learning_weights)
            sum_w2 = sum(w * w for w in learning_weights)
            eff_n = (sum_w * sum_w) / sum_w2 if sum_w2 > 0 else 0
            # Scale min_cluster_size: at least 3, proportional to effective N
            effective_min = max(3, min(self._min_cluster_size, int(eff_n / 5)))

        if len(feature_vectors) < effective_min:
            logger.info("Too few trades (%d) for variant discovery (min=%d)",
                        len(feature_vectors), effective_min)
            return []

        if feature_names is None:
            feature_names = sorted(feature_vectors[0].keys()) if feature_vectors else []

        # Build numpy matrix
        matrix = self._vectors_to_matrix(feature_vectors, feature_names)
        if matrix.shape[0] < effective_min:
            return []

        # Standardize features
        means = np.mean(matrix, axis=0)
        stds = np.std(matrix, axis=0)
        stds[stds == 0] = 1.0
        standardized = (matrix - means) / stds

        # Try HDBSCAN, fall back to K-Means
        old_min = self._min_cluster_size
        self._min_cluster_size = effective_min
        labels = self._cluster_hdbscan(standardized)
        if labels is None:
            labels = self._cluster_kmeans(standardized)
        self._min_cluster_size = old_min

        if labels is None:
            return []

        # Build variant infos
        outcomes_arr = np.array(outcomes[:len(labels)])
        variants = self._build_variant_infos(
            labels, matrix, outcomes_arr, feature_names, means, stds,
        )

        return variants

    def classify_trade(
        self,
        feature_vector: Dict[str, float],
        variants: List[VariantInfo],
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Classify a new trade into the nearest variant.

        Returns:
            (variant_label, similarity_score) or (None, 0.0) if no match.
        """
        if not variants:
            return None, 0.0

        if feature_names is None:
            feature_names = sorted(feature_vector.keys())

        trade_vec = np.array([feature_vector.get(f, 0.0) for f in feature_names])

        best_label = None
        best_sim = -1.0

        for variant in variants:
            centroid_vec = np.array([variant.centroid.get(f, 0.0) for f in feature_names])

            # Cosine similarity
            dot = np.dot(trade_vec, centroid_vec)
            norm_t = np.linalg.norm(trade_vec)
            norm_c = np.linalg.norm(centroid_vec)
            if norm_t > 0 and norm_c > 0:
                sim = float(dot / (norm_t * norm_c))
            else:
                sim = 0.0

            if sim > best_sim:
                best_sim = sim
                best_label = variant.label

        return best_label, max(0.0, best_sim)

    # ──────────────────────────────────────────
    #  Internal clustering
    # ──────────────────────────────────────────

    def _cluster_hdbscan(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Try HDBSCAN clustering. Returns labels or None."""
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self._min_cluster_size,
                min_samples=max(2, self._min_cluster_size // 2),
            )
            labels = clusterer.fit_predict(data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                logger.debug("HDBSCAN found %d clusters, falling back", n_clusters)
                return None
            logger.info("HDBSCAN found %d variant clusters", n_clusters)
            return labels
        except ImportError:
            logger.debug("hdbscan not installed, using K-Means fallback")
            return None
        except Exception as e:
            logger.warning("HDBSCAN failed: %s", e)
            return None

    def _cluster_kmeans(self, data: np.ndarray) -> Optional[np.ndarray]:
        """K-Means fallback clustering."""
        try:
            from sklearn.cluster import KMeans

            n_samples = data.shape[0]
            # Heuristic: sqrt(n/2), clamped to [2, 8]
            k = max(2, min(8, int(math.sqrt(n_samples / 2))))

            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(data)

            # Filter out tiny clusters
            from collections import Counter
            counts = Counter(labels)
            valid_clusters = {c for c, cnt in counts.items() if cnt >= self._min_cluster_size}
            if len(valid_clusters) < 2:
                # Relax: accept clusters with at least 3 trades
                valid_clusters = {c for c, cnt in counts.items() if cnt >= 3}

            if len(valid_clusters) < 2:
                return None

            # Relabel: invalid clusters become -1
            labels = np.array([l if l in valid_clusters else -1 for l in labels])
            logger.info("K-Means found %d variant clusters", len(valid_clusters))
            return labels

        except ImportError:
            logger.warning("scikit-learn not installed, cannot cluster variants")
            return None
        except Exception as e:
            logger.warning("K-Means failed: %s", e)
            return None

    def _build_variant_infos(
        self,
        labels: np.ndarray,
        matrix: np.ndarray,
        outcomes: np.ndarray,
        feature_names: List[str],
        global_means: np.ndarray,
        global_stds: np.ndarray,
    ) -> List[VariantInfo]:
        """Build VariantInfo objects from cluster labels."""
        unique_labels = sorted(set(labels))
        variants: List[VariantInfo] = []

        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue  # noise

            mask = labels == cluster_id
            cluster_data = matrix[mask]
            cluster_outcomes = outcomes[mask] if len(outcomes) >= len(labels) else np.array([])

            trade_count = int(mask.sum())
            if trade_count < 2:
                continue

            # Centroid in original feature space
            centroid = np.mean(cluster_data, axis=0)
            centroid_dict = {f: float(centroid[i]) for i, f in enumerate(feature_names)}

            # Performance stats
            if len(cluster_outcomes) > 0:
                wins = np.sum(cluster_outcomes > 0)
                win_rate = float(wins / len(cluster_outcomes))
                avg_r = float(np.mean(cluster_outcomes))

                avg_win = float(np.mean(cluster_outcomes[cluster_outcomes > 0])) if wins > 0 else 0.0
                losses = cluster_outcomes[cluster_outcomes <= 0]
                avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 1.0
                expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
            else:
                win_rate = 0.0
                avg_r = 0.0
                expectancy = 0.0

            # Distinguishing features: features most different from global mean
            cluster_means = np.mean(cluster_data, axis=0)
            cluster_stds = np.std(cluster_data, axis=0)
            diff_scores = np.abs(cluster_means - global_means) / np.maximum(global_stds, 1e-9)
            top_indices = np.argsort(diff_scores)[-3:][::-1]  # top 3

            dist_features = []
            for idx in top_indices:
                if diff_scores[idx] > 0.3:  # only meaningful differences
                    dist_features.append(DistinguishingFeature(
                        feature=feature_names[idx],
                        mean=float(cluster_means[idx]),
                        std=float(cluster_stds[idx]),
                        importance=float(diff_scores[idx]),
                    ))

            is_anti = (win_rate < _ANTI_VARIANT_WIN_RATE and expectancy < _ANTI_VARIANT_EXPECTANCY)

            label = self._generate_label(cluster_id, dist_features, is_anti)

            # Compute confidence level based on trade count and consistency
            confidence = 0.0
            if trade_count >= 50:
                confidence = 0.9
            elif trade_count >= 20:
                confidence = 0.7
            elif trade_count >= 10:
                confidence = 0.5
            elif trade_count >= 5:
                confidence = 0.3
            else:
                confidence = 0.1

            # Adjust confidence by performance consistency (std dev of outcomes)
            if len(cluster_outcomes) > 2:
                outcome_std = float(np.std(cluster_outcomes))
                if outcome_std < 0.5:
                    confidence = min(1.0, confidence + 0.1)  # very consistent
                elif outcome_std > 2.0:
                    confidence = max(0.0, confidence - 0.1)  # highly variable

            variants.append(VariantInfo(
                label=label,
                centroid=centroid_dict,
                trade_count=trade_count,
                win_rate=round(win_rate, 4),
                avg_r=round(avg_r, 4),
                expectancy=round(expectancy, 4),
                distinguishing_features=dist_features,
                is_anti_variant=is_anti,
                provenance="system",
                confidence_level=round(confidence, 2),
            ))

        return variants

    @staticmethod
    def _generate_label(
        cluster_id: int,
        features: List[DistinguishingFeature],
        is_anti: bool,
    ) -> str:
        """Generate a human-readable label for a cluster."""
        prefix = "anti_" if is_anti else ""
        if features:
            top_feat = features[0].feature
            return f"{prefix}cluster_{cluster_id}_{top_feat}"
        return f"{prefix}cluster_{cluster_id}"

    @staticmethod
    def _vectors_to_matrix(
        vectors: List[Dict[str, float]],
        feature_names: List[str],
    ) -> np.ndarray:
        """Convert list of dicts to numpy matrix."""
        rows = []
        for v in vectors:
            rows.append([v.get(f, 0.0) for f in feature_names])
        return np.array(rows, dtype=np.float64)
