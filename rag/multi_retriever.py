# rag/multi_retriever.py
"""
Multi-Layer Retriever — Cross-layer retrieval with per-query-type weighting.

Architecture Rules:
  - Retrieval uses embeddings, computation uses structured data
  - Unknown pattern safety: similarity < threshold → reduce RAG impact by 50%
  - Consistency check: split-half WR comparison downgrades trust
  - Sample sizes always included for transparency
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rag.knowledge_types import MultiLayerContext, RetrievedChunk
from rag.layered_store import LayeredVectorStore, MemoryLayer, get_layered_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for a multi-layer retrieval query."""
    layers: List[MemoryLayer] = field(default_factory=lambda: list(MemoryLayer))
    weights: Dict[MemoryLayer, float] = field(default_factory=dict)
    top_k_per_layer: int = 5
    final_top_k: int = 10
    score_threshold: float = 0.35
    filter_source: Optional[str] = None   # "user" | "avt"
    filter_strategy: Optional[str] = None

    # Unknown pattern safety (GAP 2)
    unknown_pattern_threshold: float = 0.45  # below this = "unfamiliar"


class MultiLayerRetriever:
    """
    Cross-layer retrieval with weighted scoring and safety checks.
    """

    def __init__(
        self,
        store: Optional[LayeredVectorStore] = None,
        embedder=None,
    ):
        self._store = store
        self._embedder = embedder

    def _get_store(self) -> LayeredVectorStore:
        if self._store is None:
            self._store = get_layered_store()
        return self._store

    def _get_embedder(self):
        if self._embedder is None:
            from rag.embeddings import get_embedding_provider
            self._embedder = get_embedding_provider()
        return self._embedder

    def query(
        self,
        text: str,
        config: Optional[RetrievalConfig] = None,
    ) -> MultiLayerContext:
        """
        Query across multiple layers, return merged results.
        """
        config = config or RetrievalConfig()
        store = self._get_store()
        embedder = self._get_embedder()

        # Embed query
        embedding = embedder.embed(text)

        # Build metadata filter
        filter_meta = {}
        if config.filter_source:
            filter_meta["source"] = config.filter_source
        if config.filter_strategy:
            filter_meta["strategy"] = config.filter_strategy
        if not filter_meta:
            filter_meta = None

        # Query each layer
        chunks_by_layer: Dict[str, List[RetrievedChunk]] = {}
        all_chunks: List[RetrievedChunk] = []

        for layer in config.layers:
            weight = config.weights.get(layer, 1.0)
            results = store.query(
                layer,
                embedding,
                top_k=config.top_k_per_layer,
                filter_metadata=filter_meta,
            )

            # Apply weight and threshold
            layer_chunks = []
            for rc in results:
                if rc.score < config.score_threshold:
                    continue
                # Weight the score
                weighted_rc = RetrievedChunk(
                    chunk=rc.chunk,
                    score=rc.score * weight,
                )
                weighted_rc.chunk.metadata["_memory_layer"] = layer.value
                weighted_rc.chunk.metadata["_raw_score"] = str(rc.score)
                layer_chunks.append(weighted_rc)

            chunks_by_layer[layer.value] = layer_chunks
            all_chunks.extend(layer_chunks)

        # Sort by weighted score
        all_chunks.sort(key=lambda r: -r.score)
        all_chunks = all_chunks[: config.final_top_k]

        # Compute confidence
        if all_chunks:
            raw_scores = [
                float(c.chunk.metadata.get("_raw_score", c.score))
                for c in all_chunks
            ]
            avg_score = sum(raw_scores) / len(raw_scores)
        else:
            avg_score = 0.0

        # Sample sizes per layer
        sample_sizes = {
            layer_name: len(chunks)
            for layer_name, chunks in chunks_by_layer.items()
        }

        # Token estimate (rough: chars / 4)
        total_chars = sum(len(c.text) for c in all_chunks)
        tokens_est = total_chars // 4

        # Unknown pattern safety check (GAP 2)
        best_score = all_chunks[0].score if all_chunks else 0.0
        unknown_warning = best_score < config.unknown_pattern_threshold

        ctx = MultiLayerContext(
            query=text,
            chunks_by_layer=chunks_by_layer,
            all_chunks=all_chunks,
            total_tokens_est=tokens_est,
            confidence_score=avg_score,
            sample_sizes=sample_sizes,
            consistency_level="unknown",  # set by context_builder
        )

        if unknown_warning:
            logger.info(
                "Unknown pattern safety: best similarity %.2f < %.2f threshold",
                best_score, config.unknown_pattern_threshold,
            )
            # Tag the context for downstream consumers
            ctx.sample_sizes["_unknown_pattern"] = 1

        return ctx

    def query_with_legacy_fallback(
        self,
        text: str,
        db=None,
        config: Optional[RetrievalConfig] = None,
    ) -> MultiLayerContext:
        """
        Query multi-layer store. If empty, fall back to legacy ptis_knowledge collection.
        """
        ctx = self.query(text, config)
        if ctx.all_chunks:
            return ctx

        # Fallback to legacy single-collection store
        try:
            from rag.vector_store import get_vector_store
            legacy_store = get_vector_store()
            if legacy_store.count() == 0:
                return ctx

            embedder = self._get_embedder()
            embedding = embedder.embed(text)
            legacy_results = legacy_store.query(embedding, top_k=5)

            ctx.all_chunks = legacy_results
            ctx.chunks_by_layer["legacy"] = legacy_results
            ctx.sample_sizes["legacy"] = len(legacy_results)
            if legacy_results:
                ctx.confidence_score = sum(r.score for r in legacy_results) / len(legacy_results)
            logger.debug("Multi-layer empty, fell back to legacy store (%d results)", len(legacy_results))
        except Exception as e:
            logger.debug("Legacy fallback failed: %s", e)

        return ctx

    # ══════════════════════════════════════════════════════════════
    #  PRE-BUILT CONFIGS for common use cases
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def for_trade_review() -> RetrievalConfig:
        """Config optimized for reviewing a trade."""
        return RetrievalConfig(
            layers=[MemoryLayer.TRADE, MemoryLayer.STRATEGY, MemoryLayer.INSIGHT, MemoryLayer.PATTERN],
            weights={
                MemoryLayer.TRADE: 1.5,
                MemoryLayer.STRATEGY: 1.2,
                MemoryLayer.INSIGHT: 0.8,
                MemoryLayer.PATTERN: 0.5,
            },
            top_k_per_layer=5,
            final_top_k=10,
        )

    @staticmethod
    def for_coaching() -> RetrievalConfig:
        """Config optimized for coaching / behavioral feedback."""
        return RetrievalConfig(
            layers=[MemoryLayer.INSIGHT, MemoryLayer.STRATEGY, MemoryLayer.TRADE, MemoryLayer.PATTERN],
            weights={
                MemoryLayer.INSIGHT: 1.5,
                MemoryLayer.STRATEGY: 1.2,
                MemoryLayer.TRADE: 1.0,
                MemoryLayer.PATTERN: 0.5,
            },
            top_k_per_layer=5,
            final_top_k=10,
        )

    @staticmethod
    def for_signal_eval() -> RetrievalConfig:
        """Config optimized for evaluating a setup/signal."""
        return RetrievalConfig(
            layers=[MemoryLayer.STRATEGY, MemoryLayer.PATTERN, MemoryLayer.TRADE, MemoryLayer.INSIGHT],
            weights={
                MemoryLayer.STRATEGY: 1.5,
                MemoryLayer.PATTERN: 1.5,
                MemoryLayer.TRADE: 0.8,
                MemoryLayer.INSIGHT: 0.3,
            },
            top_k_per_layer=5,
            final_top_k=8,
        )

    @staticmethod
    def for_co_planning() -> RetrievalConfig:
        """Config optimized for co-planner session."""
        return RetrievalConfig(
            layers=[MemoryLayer.STRATEGY, MemoryLayer.TRADE, MemoryLayer.INSIGHT, MemoryLayer.PATTERN],
            weights={
                MemoryLayer.STRATEGY: 1.5,
                MemoryLayer.TRADE: 1.2,
                MemoryLayer.INSIGHT: 1.0,
                MemoryLayer.PATTERN: 0.8,
            },
            top_k_per_layer=6,
            final_top_k=12,
        )

    @staticmethod
    def for_avt_only() -> RetrievalConfig:
        """Config filtered to AVT trades only."""
        return RetrievalConfig(
            layers=[MemoryLayer.TRADE, MemoryLayer.STRATEGY],
            weights={
                MemoryLayer.TRADE: 1.5,
                MemoryLayer.STRATEGY: 1.0,
            },
            filter_source="avt",
            top_k_per_layer=10,
            final_top_k=15,
        )


# ── Singleton ─────────────────────────────────────────────────────

_retriever_instance: Optional[MultiLayerRetriever] = None


def get_multi_retriever() -> MultiLayerRetriever:
    """Get or create the singleton MultiLayerRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = MultiLayerRetriever()
    return _retriever_instance
