# rag/__init__.py
"""
RAG (Retrieval-Augmented Generation) system for PTIS.

4-layer memory architecture:
  STRATEGY — strategy profiles, regime preferences, pattern library
  PATTERN  — feature vectors, clusters, micro-features
  TRADE    — closed trades (user + AVT separated)
  INSIGHT  — behavioral observations, coaching, DQ analysis

Core components:
  LayeredVectorStore  — 4-collection ChromaDB wrapper
  SemanticConverter   — features/trades → natural language + structured data
  RAGIngestor         — event-driven auto-ingestion pipeline
  MultiLayerRetriever — cross-layer retrieval with weighted scoring
  ContextBuilder      — compression before LLM injection
  GroundedLLM         — anti-hallucination wrapper with RAG clamping
"""

from rag.knowledge_types import (
    KnowledgeChunk,
    RetrievedChunk,
    MultiLayerContext,
    AgentContext,
)
from rag.layered_store import MemoryLayer, LayeredVectorStore, get_layered_store
from rag.semantic_converter import SemanticConverter, DualOutput
from rag.multi_retriever import MultiLayerRetriever, RetrievalConfig, get_multi_retriever
from rag.context_builder import ContextBuilder, StructuredContext
from rag.grounded_llm import GroundedLLM, GroundedResponse, DecisionTrace, get_grounded_llm
from rag.ingestor import RAGIngestor, init_rag_ingestor, get_rag_ingestor, cap_confidence

__all__ = [
    # Types
    "KnowledgeChunk", "RetrievedChunk", "MultiLayerContext", "AgentContext",
    "DualOutput", "StructuredContext", "GroundedResponse", "DecisionTrace",
    "RetrievalConfig",
    # Enums
    "MemoryLayer",
    # Core classes
    "LayeredVectorStore", "SemanticConverter", "RAGIngestor",
    "MultiLayerRetriever", "ContextBuilder", "GroundedLLM",
    # Factories / singletons
    "get_layered_store", "get_multi_retriever", "get_grounded_llm",
    "init_rag_ingestor", "get_rag_ingestor",
    # Utilities
    "cap_confidence",
]
