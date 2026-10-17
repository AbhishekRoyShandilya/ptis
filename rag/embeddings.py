"""
Embeddings
===========
Generates vector embeddings for knowledge chunks.
Supports OpenAI embeddings (primary) and a local sentence-transformers fallback.
Configuration via core.config: OPENAI_API_KEY, EMBEDDING_MODEL.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default embedding dimensions
OPENAI_DIM = 1536    # text-embedding-3-small / text-embedding-ada-002
LOCAL_DIM  = 384     # all-MiniLM-L6-v2


class EmbeddingProvider:
    """Base class; subclass per provider."""

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Uses OpenAI embeddings API.
    Requires: pip install openai
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        try:
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self._model = model

    def embed(self, text: str) -> list[float]:
        text = text.replace("\n", " ").strip()
        resp = self._client.embeddings.create(input=[text], model=self._model)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        cleaned = [t.replace("\n", " ").strip() for t in texts]
        resp = self._client.embeddings.create(input=cleaned, model=self._model)
        return [item.embedding for item in resp.data]

    @property
    def dimension(self) -> int:
        return OPENAI_DIM


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Uses sentence-transformers for fully local, offline embeddings.
    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers package required: pip install sentence-transformers"
            )

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()

    @property
    def dimension(self) -> int:
        return LOCAL_DIM


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic mock for tests — produces consistent (but meaningless) embeddings.
    Uses SHA-256 hash of text to seed a reproducible float vector.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        # Repeat hash bytes to fill dimension
        repeats = (self._dim * 4 // len(h)) + 1
        raw = (h * repeats)[: self._dim * 4]
        import struct
        floats = [struct.unpack_from("f", raw, i * 4)[0] for i in range(self._dim)]
        # Normalize
        mag = sum(x * x for x in floats) ** 0.5 or 1.0
        return [x / mag for x in floats]

    @property
    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedding_provider(provider: str = "auto",
                            model: Optional[str] = None,
                            api_key: Optional[str] = None) -> EmbeddingProvider:
    """
    provider: "openai" | "local" | "mock" | "auto"
    "auto" tries openai first, falls back to local, then mock.
    """
    if provider == "mock":
        return MockEmbeddingProvider()

    if provider == "local":
        return LocalEmbeddingProvider(model or "all-MiniLM-L6-v2")

    if provider == "openai":
        return OpenAIEmbeddingProvider(model or "text-embedding-3-small", api_key=api_key)

    # "auto" — try in order
    try:
        _key = api_key
        if not _key:
            from core.config import settings
            _key = getattr(settings, "OPENAI_API_KEY", None)
        if _key:
            return OpenAIEmbeddingProvider(model or "text-embedding-3-small", api_key=_key)
    except Exception as e:
        logger.debug(f"OpenAI embedding init failed: {e}")

    try:
        return LocalEmbeddingProvider(model or "all-MiniLM-L6-v2")
    except Exception as e:
        logger.debug(f"Local embedding init failed: {e}")

    logger.warning("Using mock embedding provider — embeddings are non-semantic")
    return MockEmbeddingProvider()
