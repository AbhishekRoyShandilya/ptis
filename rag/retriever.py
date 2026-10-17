"""
Retriever
==========
Semantic search layer that ties embeddings + vector store together.
Used by AI agents to fetch relevant personal knowledge before generating responses.

Main entry point: retriever.query(text, top_k, category) → AgentContext
"""
from __future__ import annotations

import logging
from typing import Optional

from rag.knowledge_types import AgentContext, RetrievedChunk
from rag.embeddings import EmbeddingProvider, get_embedding_provider
from rag.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

# Minimum relevance score to include a chunk (cosine similarity)
DEFAULT_SCORE_THRESHOLD = 0.35
DEFAULT_TOP_K = 5


class KnowledgeRetriever:

    def __init__(self, embedder: Optional[EmbeddingProvider] = None,
                 store: Optional[VectorStore] = None,
                 score_threshold: float = DEFAULT_SCORE_THRESHOLD):
        self._embedder = embedder or get_embedding_provider()
        self._store = store or get_vector_store()
        self._threshold = score_threshold

    # ------------------------------------------------------------------
    # Main retrieval
    # ------------------------------------------------------------------

    def query(self, text: str, top_k: int = DEFAULT_TOP_K,
              category: Optional[str] = None,
              min_score: Optional[float] = None) -> AgentContext:
        """
        Semantic search over personal knowledge base.

        Args:
            text:       Natural language query (e.g. "how do I trade BANKNIFTY breakouts?")
            top_k:      Maximum chunks to return
            category:   Filter by knowledge category ("strategy", "lesson", etc.)
            min_score:  Override default score threshold

        Returns:
            AgentContext with ranked RetrievedChunks, ready for prompt injection.
        """
        threshold = min_score if min_score is not None else self._threshold

        if not text.strip():
            return AgentContext(query=text)

        try:
            embedding = self._embedder.embed(text)
        except Exception as e:
            logger.error(f"Embedding failed for query '{text[:60]}': {e}")
            return AgentContext(query=text)

        try:
            raw = self._store.query(embedding, top_k=top_k * 2,  # over-fetch then filter
                                    filter_category=category)
        except Exception as e:
            logger.error(f"Vector store query failed: {e}")
            return AgentContext(query=text)

        filtered = [rc for rc in raw if rc.score >= threshold]
        # Re-rank: deduplicate by entry_id keeping highest score per entry
        seen_entries: dict[str, RetrievedChunk] = {}
        for rc in filtered:
            eid = rc.chunk.entry_id
            if eid not in seen_entries or rc.score > seen_entries[eid].score:
                seen_entries[eid] = rc
        ranked = sorted(seen_entries.values(), key=lambda x: -x.score)[:top_k]

        total_chars = sum(len(rc.text) for rc in ranked)
        return AgentContext(
            query=text,
            chunks=ranked,
            total_tokens_est=total_chars // 4,
        )

    def query_for_agent(self, strategy_name: Optional[str] = None,
                        situation: Optional[str] = None,
                        top_k: int = DEFAULT_TOP_K) -> AgentContext:
        """
        Higher-level retrieval combining strategy context + current situation.
        Merges results from two separate queries and deduplicates.
        """
        all_chunks: list[RetrievedChunk] = []

        if strategy_name:
            ctx = self.query(f"strategy rules checklist {strategy_name}", top_k=3,
                             category="strategy")
            all_chunks.extend(ctx.chunks)

        if situation:
            ctx = self.query(situation, top_k=top_k)
            all_chunks.extend(ctx.chunks)

        if not (strategy_name or situation):
            # Fallback: fetch recent lessons
            ctx = self.query("recent lesson mistake improvement", top_k=top_k,
                             category="lesson")
            all_chunks.extend(ctx.chunks)

        # Deduplicate by chunk_id, keep highest score
        seen: dict[str, RetrievedChunk] = {}
        for rc in all_chunks:
            if rc.chunk_id not in seen or rc.score > seen[rc.chunk_id].score:
                seen[rc.chunk_id] = rc

        ranked = sorted(seen.values(), key=lambda x: -x.score)[:top_k]
        query_str = " | ".join(filter(None, [strategy_name, situation]))
        total_chars = sum(len(rc.text) for rc in ranked)
        return AgentContext(query=query_str, chunks=ranked,
                            total_tokens_est=total_chars // 4)

    # ------------------------------------------------------------------
    # Convenience: fallback to SQL search if vector store is empty
    # ------------------------------------------------------------------

    def query_with_fallback(self, text: str, db, top_k: int = DEFAULT_TOP_K,
                            category: Optional[str] = None) -> AgentContext:
        """
        Try semantic search first; if vector store is empty or fails,
        fall back to the SQL keyword search in KnowledgeBase.
        """
        if self._store.count() > 0:
            ctx = self.query(text, top_k=top_k, category=category)
            if ctx.chunks:
                return ctx

        # SQL fallback
        logger.info("Vector store empty or no results — using SQL keyword fallback")
        from rag.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(db)
        entries = kb.search(text, category=category, limit=top_k)
        from rag.knowledge_types import KnowledgeChunk, RetrievedChunk
        chunks = []
        for e in entries:
            chunk = KnowledgeChunk(
                chunk_id=f"{e.id}_0",
                entry_id=str(e.id),
                category=e.category,
                title=e.title,
                text=f"{e.title}\n\n{e.content}",
                source=e.source or "manual",
                tags=list(e.tags or []),
            )
            chunks.append(RetrievedChunk(chunk=chunk, score=0.5))   # neutral score for SQL results
        return AgentContext(query=text, chunks=chunks)


# ---------------------------------------------------------------------------
# Module-level singleton (lazy initialisation)
# ---------------------------------------------------------------------------
_retriever: Optional[KnowledgeRetriever] = None


def get_retriever() -> KnowledgeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever


# Alias used by api/dependencies.py
Retriever = KnowledgeRetriever
