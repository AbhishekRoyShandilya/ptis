# bootstrap/repository.py
"""
Bootstrap persistence layer.

JudgmentLearnerStateRepository persists the user's learned judgment models — the
raw accept/reject decision history per (strategy_id, judgment_name) — to the
``judgment_learner_state`` table (migration 032) so they survive server restarts.

We store the RAW decisions as JSONB rather than a pickled model. The Bayesian
tables and the GBM are rebuilt deterministically on load (see
JudgmentLearner.load_decisions), which keeps the blob small and immune to
sklearn-version / feature-schema drift. The user's reverse-engineered judgment
is the most expensive data in the system — it is built from scarce human clicks —
so it must be durable AND reconstructable.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class JudgmentLearnerStateRepository:
    """Upsert / load / delete judgment learner state."""

    def __init__(self, db: Session):
        self.db = db

    def save(
        self,
        strategy_id: str,
        judgment_name: str,
        micro_features: Any,
        decisions: Dict[str, Any],
        total_samples: int,
        model_score: float,
    ) -> None:
        """Upsert one judgment factor's decision history."""
        now = datetime.now(timezone.utc)
        self.db.execute(
            text("""
                INSERT INTO judgment_learner_state
                    (strategy_id, judgment_name, micro_features, decisions,
                     total_samples, model_score, updated_at)
                VALUES
                    (:sid, :jname, :feats::jsonb, :dec::jsonb, :total, :score, :now)
                ON CONFLICT (strategy_id, judgment_name) DO UPDATE SET
                    micro_features = EXCLUDED.micro_features,
                    decisions      = EXCLUDED.decisions,
                    total_samples  = EXCLUDED.total_samples,
                    model_score    = EXCLUDED.model_score,
                    updated_at     = EXCLUDED.updated_at
            """),
            {
                "sid": str(strategy_id),
                "jname": judgment_name,
                "feats": json.dumps(micro_features or []),
                "dec": json.dumps(decisions or {}),
                "total": int(total_samples),
                "score": float(model_score or 0.0),
                "now": now,
            },
        )
        self.db.flush()

    def load_all(self, strategy_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Return all persisted judgment factors for a strategy, shaped for
        JudgmentManager.load_state(): {judgment_name: {micro_features, X, y, ...}}.
        """
        rows = self.db.execute(
            text("""
                SELECT judgment_name, micro_features, decisions,
                       total_samples, model_score
                FROM judgment_learner_state
                WHERE strategy_id = :sid
            """),
            {"sid": str(strategy_id)},
        ).fetchall()

        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            dec = row.decisions or {}
            result[row.judgment_name] = {
                "micro_features": row.micro_features or [],
                "X": dec.get("X", []),
                "y": dec.get("y", []),
                "total_samples": row.total_samples,
                "model_score": row.model_score,
            }
        return result

    def delete(
        self, strategy_id: str, judgment_name: Optional[str] = None
    ) -> None:
        """Remove persisted state for one factor, or all factors of a strategy."""
        if judgment_name:
            self.db.execute(
                text("""
                    DELETE FROM judgment_learner_state
                    WHERE strategy_id = :sid AND judgment_name = :jname
                """),
                {"sid": str(strategy_id), "jname": judgment_name},
            )
        else:
            self.db.execute(
                text("""
                    DELETE FROM judgment_learner_state WHERE strategy_id = :sid
                """),
                {"sid": str(strategy_id)},
            )
        self.db.flush()
