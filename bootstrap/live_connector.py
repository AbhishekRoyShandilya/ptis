# bootstrap/live_connector.py
"""
Live Connector
===============

THE missing piece between bootstrap and live trading.

Manages JudgmentManager instances per strategy and provides
hooks for the Strategy Quality Engine to use learned scores.

Lifecycle:
1. BOOTSTRAP_COMPLETED event → create JudgmentManager for strategy
2. Live setup evaluation → SoftScoringLayer asks for learned scores
3. User YES/NO on setup → record_decision() updates the model
4. Every N decisions → uncertainty reductions fed back to UncertaintyLayer

This module is a SINGLETON service. All strategy engines
reference the same instance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LiveJudgmentConnector:
    """
    Singleton service managing JudgmentManagers for all strategies.

    Strategy Quality Engine calls into this to:
    - Get learned soft scores (replacing template defaults)
    - Get uncertainty reductions (reducing penalties over time)
    - Record user decisions (feeding the learning loop)
    """

    def __init__(self, db_factory: Any = None):
        self._managers: Dict[str, Any] = {}  # strategy_id → JudgmentManager
        self._initialized = False
        self._db_factory = db_factory

    # ── DB persistence plumbing ─────────────────────────────────

    def set_db_factory(self, db_factory: Any) -> None:
        """Inject the DB session factory used for judgment persistence."""
        self._db_factory = db_factory

    def _resolve_db_factory(self):
        """Lazily fall back to the app's SessionLocal if none was injected."""
        if self._db_factory is None:
            try:
                from core.database import SessionLocal
                self._db_factory = SessionLocal
            except Exception as e:
                logger.debug(
                    "No DB factory for judgment persistence: %s", e
                )
        return self._db_factory

    def _open_db(self):
        factory = self._resolve_db_factory()
        if factory is None:
            return None
        db = factory()
        if hasattr(db, "__next__"):  # generator-style (get_db)
            db = next(db)
        return db

    def _persist_learner(
        self, strategy_id: str, manager: Any, judgment_name: str
    ) -> None:
        """Persist one judgment factor's decision history to DB."""
        learner = (
            manager._learners.get(judgment_name)
            if hasattr(manager, "_learners") else None
        )
        if learner is None or learner.total_samples == 0:
            return
        db = self._open_db()
        if db is None:
            return
        try:
            from bootstrap.repository import JudgmentLearnerStateRepository
            exp = learner.export_decisions()
            JudgmentLearnerStateRepository(db).save(
                strategy_id=strategy_id,
                judgment_name=judgment_name,
                micro_features=exp["micro_features"],
                decisions={"X": exp["X"], "y": exp["y"]},
                total_samples=len(exp["y"]),
                model_score=exp["model_score"],
            )
            db.commit()
        except Exception as e:
            logger.warning(
                "Failed to persist judgment %s/%s: %s",
                strategy_id, judgment_name, e,
            )
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    def _restore_from_db(self, strategy_id: str, manager: Any) -> int:
        """Restore a strategy's judgment learners from DB. Returns factors restored."""
        db = self._open_db()
        if db is None:
            return 0
        try:
            from bootstrap.repository import JudgmentLearnerStateRepository
            state = JudgmentLearnerStateRepository(db).load_all(strategy_id)
            if not state:
                return 0
            restored = manager.load_state(state)
            if restored:
                logger.info(
                    "Restored %d judgment factor(s) for %s from DB",
                    restored, strategy_id,
                )
            return restored
        except Exception as e:
            logger.warning(
                "Failed to restore judgment state for %s: %s",
                strategy_id, e,
            )
            return 0
        finally:
            db.close()

    def initialize_for_strategy(
        self,
        strategy_id: str,
        uncertainty_factors: List[Dict[str, Any]],
    ) -> None:
        """
        Create a JudgmentManager for a strategy.

        Called when:
        - BOOTSTRAP_COMPLETED fires
        - Strategy loaded from DB with learnable factors
        """
        learnable = [
            f for f in uncertainty_factors if f.get("can_learn", False)
        ]

        if not learnable:
            logger.info(
                "Strategy %s has no learnable factors, skipping",
                strategy_id,
            )
            return

        if strategy_id in self._managers:
            logger.info(
                "JudgmentManager already exists for %s, updating",
                strategy_id,
            )

        try:
            from bootstrap.judgment_learner import (
                JudgmentManager,
                MicroFeatureComputer,
            )

            manager = JudgmentManager(
                strategy_id=strategy_id,
                uncertainty_factors=learnable,
                micro_feature_computer=MicroFeatureComputer(),
            )
            self._managers[strategy_id] = manager
            self._initialized = True

            # Restore any previously learned judgment from DB so the user's
            # accumulated decisions survive restarts.
            self._restore_from_db(strategy_id, manager)

            logger.info(
                "JudgmentManager initialized for '%s' with %d learnable factors: %s",
                strategy_id,
                len(learnable),
                [f["name"] for f in learnable],
            )

        except Exception as e:
            logger.error(
                "Failed to create JudgmentManager for %s: %s",
                strategy_id, e,
            )

    def get_manager(self, strategy_id: str) -> Optional[Any]:
        """Get JudgmentManager for a strategy."""
        return self._managers.get(strategy_id)

    def has_manager(self, strategy_id: str) -> bool:
        return strategy_id in self._managers

    # ── Soft Score Hook ─────────────────────────────────────────

    def get_learned_soft_score(
        self,
        strategy_id: str,
        scorer_name: str,
        features: Dict[str, Any],
        candles: Optional[List[Dict]] = None,
    ) -> Optional[float]:
        """
        Get learned soft score for a specific scorer.

        Called by SoftScoringLayer BEFORE using template compute_fn.
        Returns None if no learned model exists (fall back to template).

        Args:
            strategy_id: Which strategy to look up
            scorer_name: Name of the soft scorer (must match uncertainty_factor name)
            features: Current signal features
            candles: Raw candle data for micro-feature computation
        """
        manager = self._managers.get(strategy_id)
        if manager is None:
            return None

        try:
            score = manager.get_soft_score(scorer_name, features)
            if score is not None:
                logger.debug(
                    "Learned score for %s.%s: %.3f",
                    strategy_id, scorer_name, score,
                )
            return score
        except Exception as e:
            logger.warning(
                "get_soft_score failed for %s.%s: %s",
                strategy_id, scorer_name, e,
            )
            return None

    # ── Uncertainty Reduction Hook ──────────────────────────────

    def get_uncertainty_reductions(
        self, strategy_id: str
    ) -> Dict[str, float]:
        """
        Get all uncertainty reductions for a strategy.

        Called by UncertaintyLayer to reduce penalties as the
        judgment model improves.

        Returns: {factor_name: reduction_0_to_1}
        """
        manager = self._managers.get(strategy_id)
        if manager is None:
            return {}

        try:
            return manager.get_all_uncertainty_reductions()
        except Exception as e:
            logger.warning(
                "get_uncertainty_reductions failed for %s: %s",
                strategy_id, e,
            )
            return {}

    # ── Decision Recording Hook ─────────────────────────────────

    def record_decision(
        self,
        strategy_id: str,
        judgment_name: str,
        candles: List[Dict[str, Any]],
        user_accepted: bool,
        compression_start: Optional[int] = None,
        compression_end: Optional[int] = None,
    ) -> bool:
        """
        Record a user's accept/reject decision.

        Called when user says YES/NO to a presented setup.
        THIS is the most valuable data in the system.

        Args:
            strategy_id: Which strategy
            judgment_name: Which judgment factor (e.g., "compression_quality_judgment")
            candles: Raw OHLCV candle data around the setup
            user_accepted: True = user would trade this, False = user rejected
            compression_start: Start index in candles (optional, for micro-feature computation)
            compression_end: End index in candles (optional)

        Returns: True if recorded successfully
        """
        manager = self._managers.get(strategy_id)
        if manager is None:
            logger.warning(
                "No JudgmentManager for strategy %s, decision not recorded",
                strategy_id,
            )
            return False

        try:
            manager.record_decision(
                judgment_name=judgment_name,
                candles=candles,
                user_accepted=user_accepted,
                compression_start=compression_start,
                compression_end=compression_end,
            )
            # Persist immediately — each decision is scarce, hard-won data.
            self._persist_learner(strategy_id, manager, judgment_name)
            logger.info(
                "Decision recorded for %s.%s: %s (total decisions: %d)",
                strategy_id,
                judgment_name,
                "ACCEPT" if user_accepted else "REJECT",
                manager.total_decisions(judgment_name),
            )
            return True
        except Exception as e:
            logger.error(
                "record_decision failed for %s.%s: %s",
                strategy_id, judgment_name, e,
            )
            return False

    # ── Sync Uncertainty Back ───────────────────────────────────

    def sync_uncertainty_to_engine(
        self,
        strategy_id: str,
        uncertainty_layer: Any,  # UncertaintyLayer
    ) -> int:
        """
        Push learned uncertainty reductions into the UncertaintyLayer.

        Called periodically (e.g., after every 10 decisions)
        or before each evaluation.

        Returns: number of factors updated
        """
        reductions = self.get_uncertainty_reductions(strategy_id)
        updated = 0

        for factor_name, reduction in reductions.items():
            if reduction > 0.01:
                uncertainty_layer.update_learning(factor_name, reduction)
                updated += 1

        if updated:
            logger.info(
                "Synced %d uncertainty reductions for %s",
                updated, strategy_id,
            )

        return updated

    # ── Explanations ────────────────────────────────────────────

    def get_judgment_explanations(
        self, strategy_id: str
    ) -> Dict[str, str]:
        """
        Get plain-English explanations of what the system has learned.

        E.g., "Here's what 'ugly compression' means to you:
               - High wick ratios (>0.6)
               - Low range smoothness
               - Volume spikes during compression"
        """
        manager = self._managers.get(strategy_id)
        if manager is None:
            return {}

        try:
            return manager.get_all_explanations()
        except Exception:
            return {}

    # ── Status ──────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Status summary for API/debugging."""
        return {
            "initialized": self._initialized,
            "active_strategies": len(self._managers),
            "strategies": {
                sid: {
                    "learners": list(mgr._learners.keys()) if hasattr(mgr, "_learners") else [],
                    "total_decisions": mgr.total_decisions_all()
                    if hasattr(mgr, "total_decisions_all") else 0,
                }
                for sid, mgr in self._managers.items()
            },
        }


# ── Module-level singleton ──────────────────────────────────

_connector: Optional[LiveJudgmentConnector] = None


def get_judgment_connector(db_factory: Any = None) -> LiveJudgmentConnector:
    """Get the global LiveJudgmentConnector singleton.

    Pass db_factory once at startup to enable judgment persistence; if omitted,
    the connector lazily falls back to the app's SessionLocal.
    """
    global _connector
    if _connector is None:
        _connector = LiveJudgmentConnector(db_factory=db_factory)
    elif db_factory is not None:
        _connector.set_db_factory(db_factory)
    return _connector
