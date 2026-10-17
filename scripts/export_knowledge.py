# export_knowledge.py
# scripts/export_knowledge.py
"""
Export knowledge base and strategy DNA to portable format.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from core.database import get_db
from core.models import KnowledgeEntry
from strategy_engine.strategy_dna.loader import StrategyDNALoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_all(output_dir: str = "data/exports"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export knowledge base
    with get_db() as db:
        entries = db.query(KnowledgeEntry).all()
        knowledge = [
            {
                "id": e.id,
                "category": e.category,
                "title": e.title,
                "content": e.content,
                "tags": e.tags,
                "source": e.source,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in entries
        ]

        path = os.path.join(output_dir, f"knowledge_{timestamp}.json")
        with open(path, "w") as f:
            json.dump(knowledge, f, indent=2)
        logger.info(f"Exported {len(knowledge)} knowledge entries to {path}")

    # Export strategy DNA summaries
    loader = StrategyDNALoader()
    strategies = loader.load_all()

    summaries = {}
    for name, dna in strategies.items():
        summaries[name] = {
            "display_name": dna.display_name,
            "category": dna.category,
            "description": dna.description,
            "best_regimes": dna.best_regimes,
            "worst_regimes": dna.worst_regimes,
            "learned_statistics": dna.personal_insights.learned_statistics,
            "weaknesses": dna.personal_insights.weaknesses,
            "strengths": dna.personal_insights.strengths,
        }

    path = os.path.join(output_dir, f"strategy_dna_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info(f"Exported {len(summaries)} strategy DNA summaries to {path}")


if __name__ == "__main__":
    export_all()