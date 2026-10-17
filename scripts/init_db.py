# scripts/init_db.py
"""
Initialize database and verify strategy data loads.
Run once on first setup, safe to re-run (idempotent).

Strategies are loaded from YAML files at runtime by StrategyRegistry —
they don't need a DB table. This script just ensures:
  1. All DB tables exist (via SQLAlchemy metadata)
  2. Strategy YAML files can be parsed
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing PTIS database...")

    # Create tables (safe to re-run — CREATE IF NOT EXISTS)
    from core.database import init_db
    init_db()
    logger.info("Database tables verified.")

    # Verify strategy YAML files can be loaded
    try:
        from strategy_engine.strategy_registry import get_registry
        registry = get_registry()
        count = registry.initialize()
        logger.info("Strategy registry: %d strategies loaded from YAML", count)
    except Exception as e:
        logger.warning("Strategy registry init skipped: %s", e)

    logger.info("Database initialization complete!")


if __name__ == "__main__":
    main()
