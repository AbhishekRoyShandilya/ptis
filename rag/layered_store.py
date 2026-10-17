# rag/layered_store.py
"""
Layered Vector Store — 4-collection ChromaDB wrapper.

4 memory layers, each its own ChromaDB collection:
  STRATEGY — strategy profiles, regime preferences, entry criteria (LOW volume, HIGH importance)
  PATTERN  — feature vectors, clusters, micro-features, pattern fingerprints
  TRADE    — all trades (user + AVT separated by metadata.source)
  INSIGHT  — behavioral observations, coaching, DQ analysis, LLM-distilled

Backward compatible: existing `ptis_knowledge` collection untouched.
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from rag.knowledge_types import KnowledgeChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class MemoryLayer(str, Enum):
    STRATEGY = "strategy_memory"
    PATTERN = "pattern_memory"
    TRADE = "trade_memory"
    INSIGHT = "insight_memory"


class LayeredVectorStore:
    """
    4-collection ChromaDB store with per-layer isolation.

    Falls back to in-memory dicts if ChromaDB is not available.
    All collections use cosine distance (same as existing ptis_knowledge).
    """

    def __init__(self, persist_path: str = "./data/chroma"):
        self._persist_path = persist_path
        self._collections: Dict[MemoryLayer, Any] = {}
        self._backend = "memory"  # updated if chroma loads

        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=persist_path)
            for layer in MemoryLayer:
                self._collections[layer] = self._client.get_or_create_collection(
                    name=layer.value,
                    metadata={"hnsw:space": "cosine"},
                )
            self._backend = "chroma"
            logger.info(
                "LayeredVectorStore: 4 collections at %s (%s)",
                persist_path,
                ", ".join(l.value for l in MemoryLayer),
            )
        except ImportError:
            logger.warning(
                "chromadb not installed — LayeredVectorStore using in-memory fallback"
            )
            self._client = None
            for layer in MemoryLayer:
                self._collections[layer] = {}  # chunk_id → (chunk, embedding)

    # ── Upsert ────────────────────────────────────────────────────

    def upsert(
        self,
        layer: MemoryLayer,
        chunk: KnowledgeChunk,
        embedding: List[float],
    ) -> None:
        """Add or update a single chunk in the specified layer."""
        col = self._collections[layer]

        if self._backend == "chroma":
            meta = self._build_meta(chunk)
            col.upsert(
                ids=[chunk.chunk_id],
                embeddings=[embedding],
                documents=[chunk.text],
                metadatas=[meta],
            )
        else:
            col[chunk.chunk_id] = (chunk, embedding)

    def upsert_batch(
        self,
        layer: MemoryLayer,
        chunks: List[KnowledgeChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Batch upsert into a single layer."""
        if not chunks:
            return

        col = self._collections[layer]

        if self._backend == "chroma":
            # ChromaDB batch limit is ~5000; chunk if needed
            batch_size = 5000
            for i in range(0, len(chunks), batch_size):
                batch_c = chunks[i : i + batch_size]
                batch_e = embeddings[i : i + batch_size]
                ids = [c.chunk_id for c in batch_c]
                docs = [c.text for c in batch_c]
                metas = [self._build_meta(c) for c in batch_c]
                col.upsert(ids=ids, embeddings=batch_e, documents=docs, metadatas=metas)
        else:
            for c, e in zip(chunks, embeddings):
                col[c.chunk_id] = (c, e)

    # ── Query ─────────────────────────────────────────────────────

    def query(
        self,
        layer: MemoryLayer,
        embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, str]] = None,
    ) -> List[RetrievedChunk]:
        """Query a single layer by embedding similarity."""
        col = self._collections[layer]

        if self._backend == "chroma":
            n = min(top_k, col.count() or 1)
            if n == 0:
                return []
            where = filter_metadata if filter_metadata else None
            results = col.query(
                query_embeddings=[embedding],
                n_results=n,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            return self._parse_chroma_results(results)
        else:
            return self._memory_query(col, embedding, top_k, filter_metadata)

    def query_multi(
        self,
        layers: List[MemoryLayer],
        embedding: List[float],
        top_k: int = 10,
        weights: Optional[Dict[MemoryLayer, float]] = None,
        filter_metadata: Optional[Dict[str, str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Cross-layer query: search multiple layers, merge by weighted score.

        weights: per-layer multiplier (default 1.0). Higher = more important.
        Returns top_k overall results sorted by weighted score.
        """
        weights = weights or {}
        all_results: List[RetrievedChunk] = []

        for layer in layers:
            per_layer_k = max(3, top_k)  # fetch enough per layer
            results = self.query(layer, embedding, per_layer_k, filter_metadata)
            w = weights.get(layer, 1.0)
            for rc in results:
                # Apply weight and tag the layer
                weighted = RetrievedChunk(
                    chunk=rc.chunk,
                    score=rc.score * w,
                )
                # Stash layer info in chunk metadata for downstream use
                weighted.chunk.metadata["_memory_layer"] = layer.value
                all_results.append(weighted)

        # Sort by weighted score descending, take top_k
        all_results.sort(key=lambda r: -r.score)
        return all_results[:top_k]

    # ── Delete / Count ────────────────────────────────────────────

    def delete(self, layer: MemoryLayer, chunk_id: str) -> None:
        col = self._collections[layer]
        if self._backend == "chroma":
            col.delete(ids=[chunk_id])
        else:
            col.pop(chunk_id, None)

    def delete_by_entry(self, layer: MemoryLayer, entry_id: str) -> None:
        """Delete all chunks belonging to a given entry_id in a layer."""
        col = self._collections[layer]
        if self._backend == "chroma":
            col.delete(where={"entry_id": entry_id})
        else:
            to_del = [
                cid for cid, (c, _) in col.items() if c.entry_id == entry_id
            ]
            for cid in to_del:
                del col[cid]

    def count(self, layer: MemoryLayer) -> int:
        col = self._collections[layer]
        if self._backend == "chroma":
            return col.count()
        return len(col)

    def count_all(self) -> Dict[str, int]:
        """Return chunk counts for every layer."""
        return {layer.value: self.count(layer) for layer in MemoryLayer}

    # ── Internals ─────────────────────────────────────────────────

    @staticmethod
    def _build_meta(chunk: KnowledgeChunk) -> Dict[str, str]:
        """Build ChromaDB-compatible metadata dict (string values only)."""
        meta = {
            "entry_id": chunk.entry_id,
            "category": chunk.category,
            "title": chunk.title,
            "source": chunk.source,
            "tags": json.dumps(chunk.tags),
            "created_at": chunk.created_at,
        }
        # Flatten extra metadata — chromadb requires str/int/float/bool values
        for k, v in chunk.metadata.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = json.dumps(v) if v is not None else ""
        return meta

    @staticmethod
    def _parse_chroma_results(results: dict) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for cid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist  # cosine distance → similarity
            chunk = KnowledgeChunk(
                chunk_id=cid,
                entry_id=meta.get("entry_id", ""),
                category=meta.get("category", ""),
                title=meta.get("title", ""),
                text=doc,
                source=meta.get("source", ""),
                tags=json.loads(meta.get("tags", "[]")),
                created_at=meta.get("created_at", ""),
                metadata={
                    k: v
                    for k, v in meta.items()
                    if k not in ("entry_id", "category", "title", "source", "tags", "created_at")
                },
            )
            out.append(RetrievedChunk(chunk=chunk, score=score))
        return out

    def _memory_query(
        self,
        store: dict,
        embedding: List[float],
        top_k: int,
        filter_metadata: Optional[Dict[str, str]],
    ) -> List[RetrievedChunk]:
        """In-memory cosine similarity search."""
        import math

        scored: List[RetrievedChunk] = []
        for cid, (chunk, emb) in store.items():
            # Apply metadata filter
            if filter_metadata:
                skip = False
                for fk, fv in filter_metadata.items():
                    chunk_val = chunk.metadata.get(fk, getattr(chunk, fk, None))
                    if str(chunk_val) != str(fv):
                        skip = True
                        break
                if skip:
                    continue

            dot = sum(a * b for a, b in zip(embedding, emb))
            mag_a = math.sqrt(sum(x * x for x in embedding)) or 1.0
            mag_b = math.sqrt(sum(x * x for x in emb)) or 1.0
            score = dot / (mag_a * mag_b)
            scored.append(RetrievedChunk(chunk=chunk, score=score))

        scored.sort(key=lambda r: -r.score)
        return scored[:top_k]


# ── Singleton ─────────────────────────────────────────────────────

_store_instance: Optional[LayeredVectorStore] = None


def get_layered_store(persist_path: str = "./data/chroma") -> LayeredVectorStore:
    """Get or create the singleton LayeredVectorStore."""
    global _store_instance
    if _store_instance is None:
        _store_instance = LayeredVectorStore(persist_path)
    return _store_instance
