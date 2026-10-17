"""
Knowledge Types
================
Shared type definitions for the RAG (Retrieval-Augmented Generation) system.
Decoupled from SQLAlchemy models so the RAG layer can work independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Knowledge categories (mirrors KnowledgeEntry.category in the DB model)
# ---------------------------------------------------------------------------
KNOWLEDGE_CATEGORIES = [
    "strategy",   # strategy definitions, rules, checklists
    "lesson",     # lessons from individual trades
    "pattern",    # recurring behavioral or market patterns
    "insight",    # AI-generated insights
    "rule",       # personal trading rules
    "review",     # weekly / monthly review summaries
]


@dataclass
class KnowledgeChunk:
    """
    A single chunk of text ready for embedding and vector storage.
    May be a full entry or a paragraph-level split of a longer entry.
    """
    chunk_id: str               # "<entry_id>_<chunk_index>"
    entry_id: str               # original KnowledgeEntry.id (UUID string)
    category: str
    title: str
    text: str                   # the actual text to embed
    source: str                 # "manual" | "trade_analysis" | "ai_<name>"
    tags: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)  # arbitrary extra metadata

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "entry_id": self.entry_id,
            "category": self.category,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "tags": self.tags,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class RetrievedChunk:
    """A knowledge chunk returned from a similarity search, with its score."""
    chunk: KnowledgeChunk
    score: float        # cosine similarity 0-1; higher = more relevant

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def category(self) -> str:
        return self.chunk.category

    @property
    def title(self) -> str:
        return self.chunk.title


@dataclass
class MultiLayerContext:
    """
    Cross-layer retrieval result — chunks grouped by memory layer.
    Used by ContextBuilder and GroundedLLM.
    """
    query: str
    chunks_by_layer: dict = field(default_factory=dict)  # {layer_name: [RetrievedChunk]}
    all_chunks: list = field(default_factory=list)        # flat sorted list
    total_tokens_est: int = 0
    confidence_score: float = 0.0       # avg retrieval similarity
    sample_sizes: dict = field(default_factory=dict)      # {layer: count}
    consistency_level: str = "unknown"  # "high" | "medium" | "low" | "unknown"


@dataclass
class AgentContext:
    """
    Assembled context block ready to inject into an LLM prompt.
    Includes retrieved chunks and a formatted string representation.
    """
    query: str
    chunks: list = field(default_factory=list)   # list[RetrievedChunk]
    total_tokens_est: int = 0       # rough estimate (chars / 4)

    def format_for_prompt(self, max_chunks: int = 5) -> str:
        """Return a formatted string suitable for LLM system prompt injection."""
        if not self.chunks:
            return "No relevant personal knowledge found."
        parts = ["=== PERSONAL KNOWLEDGE BASE (most relevant) ==="]
        for rc in self.chunks[:max_chunks]:
            parts.append(f"\n[{rc.category.upper()}] {rc.title}  (relevance: {rc.score:.2f})")
            parts.append(rc.text[:500])
        parts.append("=== END KNOWLEDGE ===")
        return "\n".join(parts)
