#!/usr/bin/env python
"""
scripts/seed_knowledge.py
Index strategy YAML files and documentation into the RAG vector store.
"""
import logging
import sys

sys.path.insert(0, ".")

from core.database import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Seeding RAG knowledge base...")
    db = SessionLocal()
    try:
        from rag.indexer import KnowledgeIndexer
        indexer = KnowledgeIndexer()
        count = indexer.index_all(db)
        db.commit()
        logger.info("Indexed %s knowledge entries", count)
    except Exception as e:
        logger.error("Knowledge seeding failed: %s", e)
        db.rollback()
        raise
    finally:
        db.close()

    logger.info("Knowledge seeding complete!")


if __name__ == "__main__":
    main()
