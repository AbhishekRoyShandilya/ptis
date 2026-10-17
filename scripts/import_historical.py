# import_historical.py
# scripts/import_historical.py
"""
Bulk import historical trades from CSV.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_db
from ingestion.csv_importer import CSVTradeImporter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def import_trades(file_path: str, broker: str = "generic"):
    logger.info(f"Importing from {file_path} (broker: {broker})")

    with get_db() as db:
        importer = CSVTradeImporter(db, broker=broker)
        trades = importer.import_file(file_path)
        logger.info(f"Imported {len(trades)} trades")

    logger.info("Import complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to CSV file")
    parser.add_argument("--broker", default="generic", help="Broker format")
    args = parser.parse_args()
    import_trades(args.file, args.broker)