"""
Vector Store
=============
Persists and queries knowledge chunk embeddings.
Primary backend: ChromaDB (local, no server required).
Falls back to an in-memory numpy store for environments without ChromaDB.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Optional

from rag.knowledge_types import KnowledgeChunk, RetrievedChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ptis_knowledge"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VectorStore:
    def upsert(self, chunk: KnowledgeChunk, embedding: list[float]) -> None:
        raise NotImplementedError

    def upsert_batch(self, chunks: list[KnowledgeChunk],
                     embeddings: list[list[float]]) -> None:
        for c, e in zip(chunks, embeddings):
            self.upsert(c, e)

    def query(self, embedding: list[float], top_k: int = 5,
              filter_category: Optional[str] = None) -> list[RetrievedChunk]:
        raise NotImplementedError

    def delete(self, chunk_id: str) -> None:
        raise NotImplementedError

    def delete_by_entry(self, entry_id: str) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------

class ChromaVectorStore(VectorStore):
    """
    Uses ChromaDB for persistent local vector storage.
    Requires: pip install chromadb
    Data persists in <persist_path> on disk.
    """

    def __init__(self, persist_path: str = "./data/chroma"):
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=persist_path)
            self._col = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB store initialised at {persist_path}")
        except ImportError:
            raise ImportError("chromadb package required: pip install chromadb")

    def upsert(self, chunk: KnowledgeChunk, embedding: list[float]) -> None:
        meta = {
            "entry_id": chunk.entry_id,
            "category": chunk.category,
            "title": chunk.title,
            "source": chunk.source,
            "tags": json.dumps(chunk.tags),
            "created_at": chunk.created_at,
        }
        meta.update({k: str(v) for k, v in chunk.metadata.items()})
        self._col.upsert(
            ids=[chunk.chunk_id],
            embeddings=[embedding],
            documents=[chunk.text],
            metadatas=[meta],
        )

    def upsert_batch(self, chunks: list[KnowledgeChunk],
                     embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = []
        for c in chunks:
            meta = {"entry_id": c.entry_id, "category": c.category,
                    "title": c.title, "source": c.source,
                    "tags": json.dumps(c.tags), "created_at": c.created_at}
            meta.update({k: str(v) for k, v in c.metadata.items()})
            metas.append(meta)
        self._col.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)

    def query(self, embedding: list[float], top_k: int = 5,
              filter_category: Optional[str] = None) -> list[RetrievedChunk]:
        where = {"category": filter_category} if filter_category else None
        results = self._col.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for cid, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0]
        ):
            score = 1.0 - dist   # chroma cosine distance → similarity
            chunk = KnowledgeChunk(
                chunk_id=cid,
                entry_id=meta.get("entry_id", ""),
                category=meta.get("category", ""),
                title=meta.get("title", ""),
                text=doc,
                source=meta.get("source", ""),
                tags=json.loads(meta.get("tags", "[]")),
                created_at=meta.get("created_at", ""),
            )
            out.append(RetrievedChunk(chunk=chunk, score=score))
        return out

    def delete(self, chunk_id: str) -> None:
        self._col.delete(ids=[chunk_id])

    def delete_by_entry(self, entry_id: str) -> None:
        self._col.delete(where={"entry_id": entry_id})

    def count(self) -> int:
        return self._col.count()


# ---------------------------------------------------------------------------
# In-memory fallback (no external dependencies)
# ---------------------------------------------------------------------------

class InMemoryVectorStore(VectorStore):
    """
    Simple numpy-free cosine similarity store.
    NOT persistent — data lost on restart.  Use for dev / tests.
    """

    def __init__(self):
        self._store: dict[str, tuple[KnowledgeChunk, list[float]]] = {}
        logger.warning("Using InMemoryVectorStore — data will NOT persist across restarts")

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a)) or 1.0
        mag_b = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (mag_a * mag_b)

    def upsert(self, chunk: KnowledgeChunk, embedding: list[float]) -> None:
        self._store[chunk.chunk_id] = (chunk, embedding)

    def query(self, embedding: list[float], top_k: int = 5,
              filter_category: Optional[str] = None) -> list[RetrievedChunk]:
        scored: list[tuple[float, KnowledgeChunk]] = []
        for cid, (chunk, emb) in self._store.items():
            if filter_category and chunk.category != filter_category:
                continue
            score = self._cosine(embedding, emb)
            scored.append((score, chunk))
        scored.sort(key=lambda x: -x[0])
        return [RetrievedChunk(chunk=c, score=s) for s, c in scored[:top_k]]

    def delete(self, chunk_id: str) -> None:
        self._store.pop(chunk_id, None)

    def delete_by_entry(self, entry_id: str) -> None:
        to_del = [cid for cid, (c, _) in self._store.items() if c.entry_id == entry_id]
        for cid in to_del:
            del self._store[cid]

    def count(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vector_store(backend: str = "auto", persist_path: str = "./data/chroma") -> VectorStore:
    """
    backend: "chroma" | "memory" | "auto"
    "auto" tries ChromaDB, falls back to in-memory.
    """
    if backend == "memory":
        return InMemoryVectorStore()

    if backend == "chroma":
        return ChromaVectorStore(persist_path)

    # auto
    try:
        return ChromaVectorStore(persist_path)
    except ImportError:
        logger.warning("ChromaDB not installed, falling back to in-memory vector store")
        return InMemoryVectorStore()
