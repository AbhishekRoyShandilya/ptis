# seed_strategies.py
# scripts/seed_strategies.py
"""
Load/reload all strategy DNA YAML files into the database.
Safe to re-run — updates existing, creates new.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_db
from core.models import Strategy
from strategy_engine.strategy_dna.loader import StrategyDNALoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed():
    loader = StrategyDNALoader()
    strategies = loader.load_all()
    logger.info(f"Found {len(strategies)} strategy DNA files")

    with get_db() as db:
        for name, dna in strategies.items():
            existing = db.query(Strategy).filter(Strategy.name == name).first()

            data = {
                "description": dna.description,
                "is_active": True,
                "entry_rules": {
                    "narrative": dna.entry_narrative,
                    "conditions": [
                        {"name": c.name, "importance": c.importance, "description": c.description}
                        for c in dna.setup_conditions
                    ],
                    "a_plus": dna.a_plus_conditions,
                    "skip_conditions": dna.skip_conditions,
                },
                "exit_rules": {
                    "primary_target": dna.target_primary,
                    "secondary_target": dna.target_secondary,
                    "profit_taking": dna.profit_taking_rules,
                    "time_exits": dna.time_exit_rules,
                },
                "risk_rules": {
                    "stop_method": dna.stop_method,
                    "stop_description": dna.stop_description,
                    "risk_per_trade": dna.risk_per_trade,
                    "max_risk": dna.max_risk,
                    "sizing_method": dna.sizing_method,
                    "size_adjustments": dna.size_adjustments,
                },
                "filter_rules": dna.skip_conditions,
                "best_regime": [r.regime for r in dna.market_conditions.works_best_regimes],
                "best_session": [
                    s.get("name", "") for s in dna.market_conditions.works_best_sessions
                ],
                "best_volatility": dna.market_conditions.regime_sizing.get("high_volatility", "standard"),
                "best_timeframes": [dna.timeframe_primary],
                "default_risk_percent": 1.0,
            }

            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
                logger.info(f"  Updated: {dna.display_name}")
            else:
                strategy = Strategy(name=name, **data)
                db.add(strategy)
                logger.info(f"  Created: {dna.display_name}")

    logger.info("Strategy seeding complete!")


if __name__ == "__main__":
    seed()