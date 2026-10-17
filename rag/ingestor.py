# rag/ingestor.py
"""
RAG Ingestor — Event-driven auto-ingestion pipeline.

Subscribes to event bus → converts via SemanticConverter → embeds → stores DUAL:
  Vector DB (layered_store) + Postgres (ingestion_store)

Architecture Rules:
  - Confidence auto-capped by sample size (<5 = max 30%)
  - User strategy → ONLY user writes, AVT → ONLY AVT writes
  - Failures logged but never crash the pipeline
  - Ingestion is idempotent: re-ingesting deletes old chunks first
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, Optional

from rag.layered_store import LayeredVectorStore, MemoryLayer, get_layered_store
from rag.semantic_converter import DualOutput, SemanticConverter
from rag.knowledge_types import KnowledgeChunk

logger = logging.getLogger(__name__)


# ── Confidence auto-capping (production plan Rule 21) ─────────

def cap_confidence(raw_confidence: float, sample_size: int) -> float:
    """Cap confidence based on sample size. Prevents overconfident answers from 3 trades."""
    if sample_size < 5:
        return min(raw_confidence, 0.30)
    if sample_size < 15:
        return min(raw_confidence, 0.60)
    if sample_size < 30:
        return min(raw_confidence, 0.80)
    return raw_confidence


class RAGIngestor:
    """
    Event-driven ingestion pipeline.

    Subscribes to PTIS events → converts to semantic text → embeds → stores.
    """

    def __init__(
        self,
        converter: Optional[SemanticConverter] = None,
        embedder=None,
        store: Optional[LayeredVectorStore] = None,
        db_factory: Optional[Callable] = None,
    ):
        self._converter = converter or SemanticConverter()
        self._embedder = embedder
        self._store = store
        self._db_factory = db_factory
        self._stats = {layer.value: 0 for layer in MemoryLayer}

    def _get_embedder(self):
        """Lazy-load embedder."""
        if self._embedder is None:
            from rag.embeddings import get_embedding_provider
            self._embedder = get_embedding_provider()
        return self._embedder

    def _get_store(self) -> LayeredVectorStore:
        """Lazy-load store."""
        if self._store is None:
            self._store = get_layered_store()
        return self._store

    def _get_db(self):
        """Get a DB session from factory."""
        if self._db_factory is None:
            from core.database import SessionLocal
            return SessionLocal()
        db_or_gen = self._db_factory()
        # Handle generator-based dependency
        if hasattr(db_or_gen, "__next__"):
            return next(db_or_gen)
        return db_or_gen

    # ══════════════════════════════════════════════════════════════
    #  CORE INGEST — takes DualOutput → embeds → stores
    # ══════════════════════════════════════════════════════════════

    def _ingest_dual(self, dual: DualOutput) -> bool:
        """Embed text, store in vector DB + Postgres. Returns True on success."""
        if not dual.text or dual.quality_score <= 0:
            return False

        try:
            store = self._get_store()
            embedder = self._get_embedder()

            # Delete old chunks for this entry (idempotent)
            if dual.entry_id:
                store.delete_by_entry(dual.layer, dual.entry_id)

            # Create chunk
            chunk_id = f"{dual.layer.value}_{dual.entry_id}_0"
            chunk = KnowledgeChunk(
                chunk_id=chunk_id,
                entry_id=dual.entry_id,
                category=dual.layer.value,
                title=dual.title,
                text=dual.text,
                source=dual.source,
                tags=dual.tags,
                metadata={
                    "quality_score": str(dual.quality_score),
                    "source": dual.source,
                },
            )

            # Embed
            embedding = embedder.embed(dual.text)

            # Store in vector DB
            store.upsert(dual.layer, chunk, embedding)

            # Store structured data in Postgres
            self._log_ingestion(dual)

            self._stats[dual.layer.value] = self._stats.get(dual.layer.value, 0) + 1
            return True

        except Exception as e:
            logger.warning("Ingest failed for %s/%s: %s", dual.layer.value, dual.entry_id, e)
            return False

    def _log_ingestion(self, dual: DualOutput) -> None:
        """Write structured half to Postgres ingestion log."""
        try:
            db = self._get_db()
            try:
                from rag.ingestion_store import IngestionRecord
                record = IngestionRecord(
                    layer=dual.layer.value,
                    source=dual.source,
                    source_id=dual.entry_id,
                    structured_data=dual.structured,
                    quality_score=dual.quality_score,
                    chunk_count=1,
                    title=dual.title,
                )
                # Upsert: delete old record for same source_id + layer
                existing = (
                    db.query(IngestionRecord)
                    .filter_by(source_id=dual.entry_id, layer=dual.layer.value)
                    .first()
                )
                if existing:
                    db.delete(existing)
                db.add(record)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.debug("Ingestion log write failed (non-critical): %s", e)

    # ══════════════════════════════════════════════════════════════
    #  EVENT HANDLERS — subscribe to event bus
    # ══════════════════════════════════════════════════════════════

    def subscribe(self) -> None:
        """Subscribe to all relevant events on the event bus."""
        from core.events import EventType, get_event_bus

        bus = get_event_bus()
        bus.subscribe(EventType.TRADE_CLOSED, self._on_trade_closed)
        bus.subscribe(EventType.AVT_TRADE_CLOSED, self._on_avt_trade_closed)
        bus.subscribe(EventType.STRATEGY_PROFILE_UPDATED, self._on_strategy_updated)
        bus.subscribe(EventType.LEARNING_SESSION_COMPLETED, self._on_learning_completed)
        bus.subscribe(EventType.STRATEGY_DISCOVERED, self._on_strategy_discovered)
        bus.subscribe(EventType.BOOTSTRAP_COMPLETED, self._on_bootstrap_completed)
        bus.subscribe(EventType.BEHAVIORAL_EVENT, self._on_behavioral_event)
        bus.subscribe(EventType.DQAL_COMPLETED, self._on_dqal_completed)
        bus.subscribe(EventType.TRADE_INSIGHTS_READY, self._on_trade_insights)
        logger.info("RAG ingestor subscribed to 9 event types")

    def _on_trade_closed(self, event) -> None:
        """Ingest closed trade into TRADE layer."""
        try:
            trade_id = event.data.get("trade_id")
            if not trade_id:
                return
            db = self._get_db()
            try:
                from core.models import Trade
                trade = db.query(Trade).filter_by(id=trade_id).first()
                if not trade:
                    return
                attribution = getattr(trade, "attribution", None)
                journal = getattr(trade, "journal_entry", None)
                dq = getattr(trade, "decision_quality", None)
                dual = self._converter.trade_to_dual(trade, attribution, journal, dq)
                self._ingest_dual(dual)
            finally:
                db.close()
        except Exception as e:
            logger.warning("RAG ingest trade_closed failed: %s", e)

    def _on_avt_trade_closed(self, event) -> None:
        """Ingest AVT virtual trade into TRADE layer with source='avt'."""
        try:
            dual = self._converter.avt_trade_to_dual(event.data)
            self._ingest_dual(dual)
        except Exception as e:
            logger.warning("RAG ingest avt_trade_closed failed: %s", e)

    def _on_strategy_updated(self, event) -> None:
        """Re-ingest updated strategy profile into STRATEGY layer."""
        try:
            profile_id = event.data.get("profile_id")
            if not profile_id:
                return
            db = self._get_db()
            try:
                from core.models import StrategyProfile
                profile = db.query(StrategyProfile).filter_by(id=profile_id).first()
                if profile:
                    dual = self._converter.strategy_profile_to_dual(profile)
                    self._ingest_dual(dual)
            finally:
                db.close()
        except Exception as e:
            logger.warning("RAG ingest strategy_updated failed: %s", e)

    def _on_learning_completed(self, event) -> None:
        """Re-embed strategy profile after learning session."""
        self._on_strategy_updated(event)

    def _on_strategy_discovered(self, event) -> None:
        """Ingest discovered pattern fingerprint into PATTERN layer."""
        try:
            fingerprint_data = event.data.get("fingerprint")
            if fingerprint_data and hasattr(fingerprint_data, "vector"):
                dual = self._converter.fingerprint_to_dual(fingerprint_data)
                self._ingest_dual(dual)
        except Exception as e:
            logger.warning("RAG ingest strategy_discovered failed: %s", e)

    def _on_bootstrap_completed(self, event) -> None:
        """Ingest bootstrap patterns (bulk) into PATTERN layer."""
        try:
            patterns = event.data.get("patterns", [])
            for fp in patterns:
                if hasattr(fp, "vector"):
                    dual = self._converter.fingerprint_to_dual(fp)
                    self._ingest_dual(dual)
        except Exception as e:
            logger.warning("RAG ingest bootstrap_completed failed: %s", e)

    def _on_behavioral_event(self, event) -> None:
        """Ingest behavioral observation into INSIGHT layer."""
        try:
            dual = self._converter.behavioral_event_to_dual(event.data)
            self._ingest_dual(dual)
        except Exception as e:
            logger.warning("RAG ingest behavioral_event failed: %s", e)

    def _on_dqal_completed(self, event) -> None:
        """Ingest decision quality analysis into INSIGHT layer."""
        try:
            eval_id = event.data.get("evaluation_id")
            if not eval_id:
                return
            db = self._get_db()
            try:
                from core.models import DecisionQualityRecord
                dq = db.query(DecisionQualityRecord).filter_by(evaluation_id=eval_id).first()
                if dq:
                    dual = self._converter.decision_quality_to_dual(dq)
                    self._ingest_dual(dual)
            finally:
                db.close()
        except Exception as e:
            logger.warning("RAG ingest dqal_completed failed: %s", e)

    def _on_trade_insights(self, event) -> None:
        """Ingest continuous learner insights into INSIGHT layer."""
        try:
            insights = event.data.get("insights", [])
            if isinstance(insights, list):
                for ins in insights:
                    if isinstance(ins, dict):
                        dual = self._converter.coaching_insight_to_dual(ins)
                        self._ingest_dual(dual)
        except Exception as e:
            logger.warning("RAG ingest trade_insights failed: %s", e)

    # ══════════════════════════════════════════════════════════════
    #  MANUAL / BULK INGESTION
    # ══════════════════════════════════════════════════════════════

    def ingest_trade(self, db, trade_id: str) -> bool:
        """Manually ingest a single trade by ID."""
        from core.models import Trade
        trade = db.query(Trade).filter_by(id=trade_id).first()
        if not trade:
            return False
        dual = self._converter.trade_to_dual(
            trade,
            getattr(trade, "attribution", None),
            getattr(trade, "journal_entry", None),
            getattr(trade, "decision_quality", None),
        )
        return self._ingest_dual(dual)

    def ingest_strategy(self, db, profile_id: str) -> bool:
        """Manually ingest a single strategy profile."""
        from core.models import StrategyProfile
        profile = db.query(StrategyProfile).filter_by(id=profile_id).first()
        if not profile:
            return False
        dual = self._converter.strategy_profile_to_dual(profile)
        return self._ingest_dual(dual)

    def ingest_all_trades(self, db, limit: int = 500) -> int:
        """Bulk ingest all closed trades."""
        from core.models import Trade
        from core.enums import TradeStatus
        trades = (
            db.query(Trade)
            .filter(Trade.status == TradeStatus.CLOSED)
            .order_by(Trade.exit_time.desc())
            .limit(limit)
            .all()
        )
        count = 0
        for trade in trades:
            dual = self._converter.trade_to_dual(
                trade,
                getattr(trade, "attribution", None),
                getattr(trade, "journal_entry", None),
                getattr(trade, "decision_quality", None),
            )
            if self._ingest_dual(dual):
                count += 1
        return count

    def ingest_all_strategies(self, db) -> int:
        """Bulk ingest all active strategy profiles."""
        from core.models import StrategyProfile
        profiles = (
            db.query(StrategyProfile)
            .filter(StrategyProfile.status.in_(["active", "learning"]))
            .all()
        )
        count = 0
        for profile in profiles:
            dual = self._converter.strategy_profile_to_dual(profile)
            if self._ingest_dual(dual):
                count += 1
        return count

    def ingest_pattern_library(self) -> int:
        """Seed pattern library templates into STRATEGY layer."""
        try:
            from bootstrap.pattern_library import get_pattern_library
            library = get_pattern_library()
            templates = library.list_templates() if hasattr(library, "list_templates") else []

            # Fallback: try to access templates directly
            if not templates and hasattr(library, "_templates"):
                templates = list(library._templates.values())
            if not templates and hasattr(library, "templates"):
                templates = list(library.templates.values()) if isinstance(library.templates, dict) else library.templates

            count = 0
            for template in templates:
                dual = self._converter.pattern_template_to_dual(template)
                if self._ingest_dual(dual):
                    count += 1
            logger.info("Seeded %d pattern library templates into STRATEGY layer", count)
            return count
        except Exception as e:
            logger.warning("Pattern library seeding failed: %s", e)
            return 0

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Return ingestion stats: counts per layer + quality summary."""
        store = self._get_store()
        counts = store.count_all()
        return {
            "vector_counts": counts,
            "session_ingested": dict(self._stats),
            "total_vectors": sum(counts.values()),
        }


# ── Singleton + init ──────────────────────────────────────────────

_ingestor_instance: Optional[RAGIngestor] = None


def get_rag_ingestor() -> Optional[RAGIngestor]:
    """Get the singleton RAG ingestor (None if not initialized)."""
    return _ingestor_instance


def init_rag_ingestor(db_factory=None) -> RAGIngestor:
    """Initialize and subscribe the RAG ingestor. Called at startup."""
    global _ingestor_instance
    if _ingestor_instance is not None:
        return _ingestor_instance

    ingestor = RAGIngestor(db_factory=db_factory)
    ingestor.subscribe()
    _ingestor_instance = ingestor
    return ingestor
