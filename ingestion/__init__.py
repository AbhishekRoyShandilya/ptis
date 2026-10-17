# ingestion/__init__.py
"""
Trade data ingestion — CSV import, broker API adapters, deduplication.
"""

from ingestion.csv_importer import CSVTradeImporter
from ingestion.deduplicator import TradeDeduplicator

__all__ = [
    "CSVTradeImporter",
    "TradeDeduplicator",
]
