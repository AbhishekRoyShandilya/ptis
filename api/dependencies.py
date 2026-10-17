"""
API Dependencies — FastAPI dependency injection.

DB Session Pattern (two factories, used intentionally):

  get_db()             — ASYNC session (AsyncSession via asyncpg).
                         Used by WebSocket routes and any async endpoint.
                         Import: from api.dependencies import get_db

  get_db_dependency()  — SYNC session (Session via psycopg2).
                         Used by all standard REST endpoints.
                         Import: from core.database import get_db_dependency

Both return a scoped session per-request. The codebase intentionally mixes
sync and async DB access because the REST layer was built first (sync) and
WebSocket routes require async. Full async migration is planned post-beta.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from agents.llm_client import LLMClient
from core.database import async_session_factory, get_db_dependency
from api.auth import get_optional_user
from config.settings import settings
from rag.retriever import Retriever


async def get_db() -> AsyncGenerator:
    """
    Async database session dependency.
    Falls back to a sync session wrapped in a compatible interface when asyncpg
    is not available, so endpoints degrade gracefully instead of dropping the
    connection (which causes "Failed to fetch" on the frontend).
    """
    if async_session_factory is not None:
        async with async_session_factory() as session:
            try:
                yield session
            finally:
                await session.close()
    else:
        # Sync fallback — yields a regular SQLAlchemy Session.
        # Async methods (await session.execute) won't work but simple
        # attribute access and ORM queries will. Routes that only read
        # context (co-plan, chat) work fine with this.
        db = next(get_db_dependency())
        try:
            yield db
        finally:
            db.close()


def get_llm_client() -> LLMClient:
    """LLM client dependency."""
    return LLMClient(
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        fallback_provider=settings.LLM_FALLBACK_PROVIDER,
        fallback_model=settings.LLM_FALLBACK_MODEL,
        max_retries=settings.LLM_MAX_RETRIES,
        timeout=settings.LLM_TIMEOUT,
    )


def get_rag_retriever() -> Optional[Retriever]:
    """RAG retriever dependency. Returns None if not configured."""
    try:
        return Retriever()
    except Exception:
        return None


def get_dna_loader():
    """Strategy DNA loader — used by strategies routes."""
    from strategy_engine.strategy_dna.loader import StrategyDNALoader
    loader = StrategyDNALoader()
    loader.load_all()
    return loader


def get_query_engine():
    """Strategy DNA query engine — used by regime and strategies routes."""
    from strategy_engine.strategy_dna.query_engine import StrategyQueryEngine
    return StrategyQueryEngine(get_dna_loader())


def get_orchestrator(db=Depends(get_db_dependency)):
    """Agent orchestrator dependency — used by dashboard routes."""
    from agents.orchestrator import AgentOrchestrator
    return AgentOrchestrator(db=db, dna_loader=get_dna_loader())


def get_vision_analyzer():
    """Chart vision analyzer dependency — for strategy learning.
    No client passed: the analyzer builds its OWN vision client (VISION_MODEL, a multimodal model) —
    the text LLM (llama-3.3-70b) is text-only and would fail on images."""
    from strategy_learning.vision_analyzer import ChartVisionAnalyzer
    return ChartVisionAnalyzer()


def get_screenshot_store():
    """Screenshot file storage dependency."""
    from strategy_learning.screenshot_store import ScreenshotStore
    return ScreenshotStore()


def get_signal_generator():
    """SKE signal generator dependency."""
    from ske.signal_generator import SignalGenerator
    return SignalGenerator()


def get_sko_builder():
    """SKE builder dependency."""
    from ske.builder import SKOBuilder
    return SKOBuilder()


def get_dqal_validator():
    """DQAL validator dependency."""
    from ske.dqal import DQALValidator
    return DQALValidator()


def get_variant_engine():
    """Variant engine dependency."""
    from ske.variant_engine import VariantEngine
    return VariantEngine()


def get_bootstrap_orchestrator():
    """Bootstrap orchestrator dependency."""
    from bootstrap.orchestrator import BootstrapOrchestrator
    return BootstrapOrchestrator()


def get_multi_retriever():
    """Multi-layer RAG retriever dependency."""
    from rag.multi_retriever import get_multi_retriever as _get
    return _get()


def get_context_builder():
    """RAG context builder dependency."""
    from rag.context_builder import ContextBuilder
    from core.database import SessionLocal
    return ContextBuilder(db_factory=SessionLocal)


def get_grounded_llm():
    """Grounded LLM dependency (anti-hallucination wrapper)."""
    from rag.grounded_llm import get_grounded_llm as _get
    from core.database import SessionLocal
    return _get(db_factory=SessionLocal)


async def get_current_trader_id(
    x_trader_id: Optional[str] = Header(None, alias="X-Trader-ID"),
    auth_user=Depends(get_optional_user),
    db=Depends(get_db_dependency),
) -> str:
    """
    Resolve the trader ID for the request, consistent with the JWT-authenticated user.

    Order: explicit X-Trader-ID header → the JWT-AUTHENTICATED user (so co-plan / behavior read the SAME
    trades the journal does — this was the "0 trades" bug: the logged-in user's trades sit under their UUID
    but this used to fall straight to 'default-trader') → dev-bypass user → env/literal default.
    """
    if x_trader_id:
        return x_trader_id

    # THE FIX: use the real authenticated user (matches get_current_user, which the journal/trades use).
    if auth_user is not None:
        return str(auth_user.id)

    if settings.DEV_AUTH_BYPASS:
        from core.models import User
        dev_user = db.query(User).filter(User.username == "dev").first()
        if dev_user:
            return str(dev_user.id)

    default_id = os.getenv("DEFAULT_TRADER_ID", "default-trader")
    return default_id