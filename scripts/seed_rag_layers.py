#!/usr/bin/env python3
"""
Seed RAG Layers — One-time bootstrap for existing data.

Run once after deployment to populate the 4-layer RAG memory
from existing trades, strategies, and pattern library.

Usage:
    python scripts/seed_rag_layers.py
"""
from __future__ import annotations

import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import SessionLocal
from rag.ingestor import RAGIngestor
from rag.semantic_converter import SemanticConverter
from rag.layered_store import get_layered_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("═══ RAG Layer Seeding ═══")

    ingestor = RAGIngestor(db_factory=SessionLocal)
    db = SessionLocal()

    try:
        # 1. Seed pattern library (8 curated templates → STRATEGY layer)
        logger.info("Seeding pattern library...")
        pattern_count = ingestor.ingest_pattern_library()
        logger.info("  → %d pattern templates seeded", pattern_count)

        # 2. Seed strategy profiles → STRATEGY layer
        logger.info("Seeding strategy profiles...")
        strategy_count = ingestor.ingest_all_strategies(db)
        logger.info("  → %d strategy profiles seeded", strategy_count)

        # 3. Seed closed trades → TRADE layer
        logger.info("Seeding closed trades...")
        trade_count = ingestor.ingest_all_trades(db, limit=500)
        logger.info("  → %d trades seeded", trade_count)

        # 4. Seed decision quality records → INSIGHT layer
        logger.info("Seeding decision quality records...")
        dq_count = _seed_decision_quality(ingestor, db)
        logger.info("  → %d DQ records seeded", dq_count)

        # 5. Seed journal entries → INSIGHT layer
        logger.info("Seeding journal entries...")
        journal_count = _seed_journals(ingestor, db)
        logger.info("  → %d journal entries seeded", journal_count)

        # Summary
        stats = ingestor.get_ingestion_stats()
        logger.info("═══ Seeding Complete ═══")
        logger.info("Vector counts: %s", stats["vector_counts"])
        logger.info("Total vectors: %d", stats["total_vectors"])

    finally:
        db.close()


def _seed_decision_quality(ingestor: RAGIngestor, db) -> int:
    """Seed all DQ records into INSIGHT layer."""
    try:
        from core.models import DecisionQualityRecord
        records = db.query(DecisionQualityRecord).all()
        count = 0
        for dq in records:
            dual = ingestor._converter.decision_quality_to_dual(dq)
            if ingestor._ingest_dual(dual):
                count += 1
        return count
    except Exception as e:
        logger.warning("DQ seeding failed: %s", e)
        return 0


def _seed_journals(ingestor: RAGIngestor, db) -> int:
    """Seed journal entries with lessons into INSIGHT layer."""
    try:
        from core.models import JournalEntry, Trade
        journals = (
            db.query(JournalEntry)
            .filter(JournalEntry.lesson.isnot(None))
            .all()
        )
        count = 0
        for journal in journals:
            trade = db.query(Trade).filter_by(id=journal.trade_id).first()
            dual = ingestor._converter.journal_to_dual(journal, trade)
            if ingestor._ingest_dual(dual):
                count += 1
        return count
    except Exception as e:
        logger.warning("Journal seeding failed: %s", e)
        return 0


if __name__ == "__main__":
    main()
