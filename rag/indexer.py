"""
Indexer
========
Converts KnowledgeEntry DB rows into KnowledgeChunks, embeds them,
and upserts into the vector store.

Key design choices:
- Chunks entries at paragraph boundaries (≤400 tokens each)
- Idempotent: re-indexing the same entry_id overwrites old chunks
- Called from: API on new/edited entries, and scheduled background job
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from rag.knowledge_types import KnowledgeChunk
from rag.embeddings import EmbeddingProvider, get_embedding_provider
from rag.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

_CHUNK_MAX_CHARS = 1600   # ≈ 400 tokens — safe for all embedding models
_CHUNK_OVERLAP   = 200    # overlap to preserve context across chunk boundaries


def _split_text(text: str, max_chars: int = _CHUNK_MAX_CHARS,
                overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks at paragraph/sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    # Split on paragraph breaks first, then sentences
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # If paragraph itself exceeds max, split on sentences
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_chars:
                        current = (current + " " + sent).strip() if current else sent
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
            else:
                current = para
    if current:
        chunks.append(current)

    # Add overlap: prepend tail of previous chunk to each subsequent chunk
    if overlap and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped

    return chunks


class KnowledgeIndexer:

    def __init__(self, embedder: Optional[EmbeddingProvider] = None,
                 store: Optional[VectorStore] = None):
        self._embedder = embedder or get_embedding_provider()
        self._store = store or get_vector_store()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_entry(self, entry) -> int:
        """
        Index a single KnowledgeEntry (SQLAlchemy model instance).
        Deletes existing chunks for this entry_id first (idempotent).
        Returns number of chunks created.
        """
        entry_id = str(entry.id)
        self._store.delete_by_entry(entry_id)

        chunks = self._entry_to_chunks(entry)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_batch(texts)
        self._store.upsert_batch(chunks, embeddings)
        logger.info(f"Indexed entry {entry_id} → {len(chunks)} chunk(s)")
        return len(chunks)

    def index_batch(self, entries: list) -> int:
        """Index multiple entries. Returns total chunk count."""
        total = 0
        for entry in entries:
            total += self.index_entry(entry)
        return total

    def index_all(self, db) -> int:
        """Re-index every KnowledgeEntry in the database."""
        from core.models import KnowledgeEntry
        entries = db.query(KnowledgeEntry).all()
        total = self.index_batch(entries)
        logger.info(f"Full re-index complete: {len(entries)} entries → {total} chunks")
        return total

    def delete_entry(self, entry_id: str) -> None:
        """Remove all chunks for a deleted knowledge entry."""
        self._store.delete_by_entry(entry_id)
        logger.info(f"Removed chunks for entry {entry_id}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _entry_to_chunks(self, entry) -> list[KnowledgeChunk]:
        entry_id = str(entry.id)
        # Build rich text: title + content for better semantic retrieval
        full_text = f"{entry.title}\n\n{entry.content}"
        raw_chunks = _split_text(full_text)

        tags = list(entry.tags or [])
        chunks: list[KnowledgeChunk] = []
        for i, text in enumerate(raw_chunks):
            chunks.append(KnowledgeChunk(
                chunk_id=f"{entry_id}_{i}",
                entry_id=entry_id,
                category=entry.category,
                title=entry.title,
                text=text,
                source=entry.source or "manual",
                tags=tags,
                created_at=entry.created_at.isoformat() if hasattr(entry.created_at, "isoformat") else str(entry.created_at),
                metadata={"chunk_index": i, "total_chunks": len(raw_chunks)},
            ))
        return chunks
