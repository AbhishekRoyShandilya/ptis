# coplanner/engine.py
"""
Co-Planner Engine — The Orchestrator.

UPGRADED from LLM chatbot → engine-driven intelligence layer.

RULE: Engine computes → LLM explains. NEVER the reverse.

7-Step Flow:
  1. Behavioral guard (should I even trade right now?)
  2. Score all setups (conviction engine)
  3. Check regime alignment (strategy-regime matrix)
  4. Enforce RR (risk-reward gate)
  5. Cross-stock comparison (head-to-head)
  6. Multi-factor ranking (7-dimensional)
  7. Generate recommendation + LLM explanation

What Co-Planner borrows READ-ONLY from AVT:
  - avt_bridge clean trade data for matrix building
  - variant performance cache for enrichment
  Never writes to AVT. Never mutates strategy definitions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from coplanner.behavioral_guard import BehavioralPlannerGuard
from coplanner.conviction_engine import ConvictionEngine
from coplanner.cross_stock_comparator import CrossStockComparator
from coplanner.data_trust_layer import DataTrustLayer
from coplanner.db_interface import CoPlannerDB
from coplanner.models import (
    BehavioralCheckResult,
    PlanSessionResult,
    RankedOpportunity,
    SetupData,
)
from coplanner.opportunity_ranker import OpportunityRanker
from coplanner.rr_enforcer import RiskRewardEnforcer
from coplanner.strategy_regime_matrix import StrategyRegimeMatrix
from coplanner.trade_quality_classifier import TradeQualityClassifier

logger = logging.getLogger(__name__)


class CoPlannerEngine:
    """
    Main entry point for co-planner intelligence.

    Old: LLM chatbot that reads one chart.
    New: Engine-driven intelligence + LLM explanation.

    Usage:
        engine = CoPlannerEngine(db_session)
        result = engine.plan_session(setups)
        # result.ranked_opportunities, result.recommendation, ...
    """

    def __init__(
        self,
        db: CoPlannerDB,
        pattern_learner: Any = None,
        regime_builder: Any = None,
        llm_client: Any = None,
        user_profile: Optional[Dict[str, Any]] = None,
        trader_id: Optional[str] = None,
        avt_data_source: Any = None,
    ):
        self._db = db
        self._llm = llm_client
        self._trader_id = trader_id

        # Strategy profiles loaded from DB (user's learned strategies)
        self._learned_strategies: Dict[str, Dict[str, Any]] = {}

        # Data Trust Layer — cleans noisy user data
        self._classifier = TradeQualityClassifier(db=db)
        self._trust_layer = DataTrustLayer(
            db=db,
            avt_data_source=avt_data_source,
            quality_classifier=self._classifier,
        )

        # Phase 1: Foundation — matrix now uses trust layer
        self._regime_matrix = StrategyRegimeMatrix(
            db, regime_builder, data_trust_layer=self._trust_layer,
        )
        self._rr_enforcer = RiskRewardEnforcer(user_profile)
        # Edge amplifier: detects A+ setups within strategies for size-up
        edge_amp = None
        try:
            from improvement_lab.edge_amplifier import get_edge_amplifier
            edge_amp = get_edge_amplifier()
        except Exception:
            pass

        self._conviction = ConvictionEngine(
            pattern_learner=pattern_learner,
            regime_matrix=self._regime_matrix,
            db=db,
            rr_enforcer=self._rr_enforcer,
            edge_amplifier=edge_amp,
        )

        # Phase 2: Comparison
        self._comparator = CrossStockComparator(
            conviction_engine=self._conviction,
            regime_matrix=self._regime_matrix,
            rr_enforcer=self._rr_enforcer,
        )
        self._ranker = OpportunityRanker(self._regime_matrix)

        # Phase 3: Behavioral
        self._guard = BehavioralPlannerGuard(db, trader_id)

        # Auto-load learned strategies on init
        self._load_learned_strategies()

    # ─────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────

    def plan_session(
        self,
        setups: List[SetupData],
        open_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> PlanSessionResult:
        """
        MAIN FLOW — called when user opens planning session.

        1. Behavioral guard
        2. Cross-stock comparison (includes conviction + regime + RR)
        3. Multi-factor ranking
        4. Regime warnings
        5. Recommendation
        6. LLM explanation (optional)
        """
        session_id = str(uuid4())

        # Step 1: Behavioral guard
        behavioral = self._guard.check(
            current_hour=datetime.now(timezone.utc).hour,
        )

        if not behavioral.proceed:
            result = PlanSessionResult(
                behavioral_check=behavioral,
                recommendation={
                    "action": "DO_NOT_TRADE",
                    "reason": behavioral.verdict,
                },
                total_scanned=len(setups),
            )
            if self._llm:
                result.explanation = self._llm_explain(result)
            self._log_decision(session_id, result)
            return result

        # Step 2: Cross-stock comparison
        comparison = self._comparator.compare(setups)

        # Step 3: Multi-factor ranking
        ranked = self._ranker.rank(
            comparison["ranked_setups"],
            open_positions=open_positions,
        )

        # Step 4: Regime warnings
        regime_warnings = self._collect_regime_warnings(setups)

        # Step 5: Recommendation
        recommendation = comparison["recommendation"]

        # Build result
        result = PlanSessionResult(
            behavioral_check=behavioral,
            ranked_opportunities=ranked,
            regime_warnings=regime_warnings,
            recommendation=recommendation,
            total_scanned=len(setups),
            actionable_count=comparison["actionable"],
            computed_at=datetime.now(timezone.utc),
        )

        # Step 6: LLM explanation (optional)
        if self._llm:
            result.explanation = self._llm_explain(result)

        # Log decision
        self._log_decision(session_id, result)

        return result

    # ─────────────────────────────────────────────
    # SINGLE SETUP EVALUATION
    # ─────────────────────────────────────────────

    def evaluate_single(self, setup: SetupData) -> Dict[str, Any]:
        """Quick evaluation of a single setup without full ranking."""
        conv = self._conviction.compute(setup)
        avoidance = self._regime_matrix.should_avoid(
            setup.strategy_name, setup.current_regime
        )
        regime_stats = self._regime_matrix.query(
            setup.strategy_name, setup.current_regime
        )

        return {
            "conviction": {
                "score": conv.conviction_score,
                "level": conv.conviction_level.value,
                "components": conv.components,
                "warnings": conv.warnings,
                "sizing": conv.sizing_modifier,
                "verdict": conv.verdict,
            },
            "regime": {
                "win_rate": regime_stats.win_rate,
                "avg_rr": regime_stats.avg_rr,
                "trades": regime_stats.total_trades,
                "confidence": regime_stats.sample_confidence.value,
            },
            "avoidance": {
                "avoid": avoidance.avoid,
                "reason": avoidance.reason,
                "alternatives": avoidance.alternatives,
            },
            "rr": (
                {
                    "expected_rr": conv.rr_assessment.expected_rr,
                    "grade": conv.rr_assessment.grade.value,
                    "acceptable": conv.rr_assessment.acceptable,
                    "message": conv.rr_assessment.message,
                }
                if conv.rr_assessment
                else None
            ),
        }

    # ─────────────────────────────────────────────
    # REGIME MATRIX ACCESS
    # ─────────────────────────────────────────────

    def get_regime_matrix(self) -> Dict[str, Any]:
        """Full matrix display for UI."""
        return self._regime_matrix.get_full_matrix_display()

    def get_strategy_profile(self, strategy_name: str) -> Dict[str, Any]:
        """Cross-regime profile for a strategy."""
        return self._regime_matrix.get_strategy_profile(strategy_name)

    def rebuild_matrix(self) -> Dict[str, Any]:
        """Force rebuild of regime matrix."""
        return self._regime_matrix.build(force=True)

    # ─────────────────────────────────────────────
    # RR TOOLS
    # ─────────────────────────────────────────────

    def suggest_targets(
        self, entry: float, stop_loss: float,
    ) -> Dict[str, Any]:
        """Multi-target exit suggestions."""
        return self._rr_enforcer.suggest_multiple_targets(entry, stop_loss)

    def get_rr_config(self) -> Dict[str, Any]:
        return self._rr_enforcer.get_current_config()

    # ─────────────────────────────────────────────
    # DATA TRUST & QUALITY
    # ─────────────────────────────────────────────

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Data trust layer quality report — execution grade, clean ratio, etc."""
        return self._trust_layer.get_data_quality_report()

    def get_data_availability(self) -> Dict[str, Any]:
        """Cold start / warming up / AVT ready status."""
        return self._trust_layer.get_data_availability()

    def get_execution_profile(self) -> Dict[str, Any]:
        """Full execution quality profile — strengths, weaknesses, growth."""
        return self._classifier.get_execution_profile()

    def get_evolution_timeline(self, window_size: int = 20) -> List[Dict[str, Any]]:
        """Quality evolution over time."""
        return self._classifier.get_evolution_timeline(window_size)

    def detect_behavioral_patterns(self) -> Dict[str, Any]:
        """Recurring behavioral patterns in trading."""
        return self._classifier.detect_behavioral_patterns()

    def refresh_trust_weights(self) -> None:
        """Recompute adaptive trust weights based on execution quality."""
        self._trust_layer.refresh_weights()
        self._trust_layer.invalidate_cache()

    # ─────────────────────────────────────────────
    # LEARNED STRATEGIES (from StrategyProfile)
    # ─────────────────────────────────────────────

    def _load_learned_strategies(self) -> None:
        """
        Load active StrategyProfile records from DB so co-planner
        knows which strategies the user has, including:
        - regime_preferences (preferred/avoid regimes)
        - risk_parameters (typical RR, risk %)
        - entry_criteria (what patterns to look for)
        - confidence_scores (how well the system understands each strategy)

        This is critical for cold-start: when there's no trade history yet,
        the co-planner can still recommend strategies based on the user's
        learned regime preferences.
        """
        try:
            from core.models import StrategyProfile
            raw_db = self._db._db  # underlying SQLAlchemy session

            q = raw_db.query(StrategyProfile).filter(
                StrategyProfile.status.in_(["active", "learning"])
            )
            # Scope to the caller's own strategies — co-planner must plan around
            # THIS user's strategies, not every user's.
            if self._trader_id:
                q = q.filter(StrategyProfile.trader_id == self._trader_id)
            profiles = q.all()

            self._learned_strategies = {}
            for p in profiles:
                self._learned_strategies[p.short_code] = {
                    "id": str(p.id),
                    "name": p.name,
                    "short_code": p.short_code,
                    "status": p.status,
                    "regime_preferences": p.regime_preferences or {},
                    "risk_parameters": p.risk_parameters or {},
                    "entry_criteria": p.entry_criteria or [],
                    "confidence_scores": p.confidence_scores or {},
                    "examples_count": p.examples_count or 0,
                }

            if self._learned_strategies:
                logger.info(
                    "Co-planner loaded %d learned strategies: %s",
                    len(self._learned_strategies),
                    list(self._learned_strategies.keys()),
                )
                # Seed regime matrix with profile regime preferences (cold-start support)
                self._seed_matrix_from_profiles()
        except Exception as e:
            logger.debug("Could not load learned strategies: %s", e)

    def _seed_matrix_from_profiles(self) -> None:
        """
        For strategies with no trade history yet, seed the regime matrix
        with the user's stated regime preferences from StrategyProfile.

        This ensures the co-planner can recommend "use breakout in trending"
        even before the user has traded with that strategy.

        Seeds are marked with low confidence (INSUFFICIENT) so real trade
        data overrides them as soon as it arrives.
        """
        from coplanner.models import ConfidenceLevel, RegimePerformanceStats

        for code, profile in self._learned_strategies.items():
            prefs = profile.get("regime_preferences", {})
            preferred = prefs.get("preferred", [])
            avoid = prefs.get("avoid", [])

            # Only seed if matrix doesn't already have data for this strategy
            normalized = self._regime_matrix._normalize(code)
            if normalized in self._regime_matrix.matrix:
                continue  # real data exists, don't override

            if not preferred and not avoid:
                continue

            self._regime_matrix.matrix[normalized] = {}
            if normalized not in self._regime_matrix._all_strategies:
                self._regime_matrix._all_strategies.append(normalized)

            # Preferred regimes → optimistic seed (60% WR, 1.5 RR)
            for pref in preferred:
                regime = pref if isinstance(pref, str) else pref.get("regime", "")
                confidence_val = pref.get("confidence", 0.5) if isinstance(pref, dict) else 0.5
                if not regime:
                    continue
                r = self._regime_matrix._normalize(regime)
                if r not in self._regime_matrix._all_regimes:
                    self._regime_matrix._all_regimes.append(r)
                self._regime_matrix.matrix[normalized][r] = RegimePerformanceStats(
                    strategy_name=code,
                    regime=regime,
                    win_rate=0.55,
                    avg_rr=1.3,
                    total_trades=0,  # 0 trades = clearly a seed
                    sample_confidence=ConfidenceLevel.INSUFFICIENT,
                )

            # Avoided regimes → pessimistic seed (35% WR)
            for av in avoid:
                regime = av if isinstance(av, str) else av.get("regime", "")
                if not regime:
                    continue
                r = self._regime_matrix._normalize(regime)
                if r not in self._regime_matrix._all_regimes:
                    self._regime_matrix._all_regimes.append(r)
                self._regime_matrix.matrix[normalized][r] = RegimePerformanceStats(
                    strategy_name=code,
                    regime=regime,
                    win_rate=0.35,
                    avg_rr=0.8,
                    total_trades=0,
                    sample_confidence=ConfidenceLevel.INSUFFICIENT,
                )

            logger.debug(
                "Seeded matrix for '%s': %d preferred, %d avoid regimes",
                code, len(preferred), len(avoid),
            )

    def get_learned_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Return all loaded strategy profiles for display/debugging."""
        return self._learned_strategies

    def get_strategies_for_regime(self, regime: str) -> List[Dict[str, Any]]:
        """
        Get user's strategies ranked by suitability for the given regime.
        Combines trade-history matrix data with learned regime preferences.
        """
        # Start with matrix-based ranking
        matrix_ranked = self._regime_matrix.get_best_strategies_for_regime(regime)

        # Add strategies that have profile regime preferences but no trade data
        matrix_strategies = {r["strategy"] for r in matrix_ranked}
        r_norm = self._regime_matrix._normalize(regime)

        for code, profile in self._learned_strategies.items():
            code_norm = self._regime_matrix._normalize(code)
            if code_norm in matrix_strategies:
                continue

            prefs = profile.get("regime_preferences", {})
            preferred = prefs.get("preferred", [])
            avoid = prefs.get("avoid", [])

            # Check if this regime is in preferred list
            is_preferred = any(
                (p if isinstance(p, str) else p.get("regime", "")).lower().replace(" ", "_") == r_norm
                for p in preferred
            )
            is_avoided = any(
                (a if isinstance(a, str) else a.get("regime", "")).lower().replace(" ", "_") == r_norm
                for a in avoid
            )

            if is_preferred:
                matrix_ranked.append({
                    "strategy": code,
                    "win_rate": 0.55,
                    "avg_rr": 1.3,
                    "trades": 0,
                    "confidence": "insufficient",
                    "expectancy_score": 0.4,
                    "holding_hours": 0,
                    "source": "learned_preference",
                    "note": f"User's strategy prefers {regime} (no trade history yet)",
                })
            elif not is_avoided:
                # Neutral — include with low score
                matrix_ranked.append({
                    "strategy": code,
                    "win_rate": 0.5,
                    "avg_rr": 1.0,
                    "trades": 0,
                    "confidence": "insufficient",
                    "expectancy_score": 0.3,
                    "holding_hours": 0,
                    "source": "learned_neutral",
                    "note": f"Strategy '{code}' has no data for {regime}",
                })

        # Re-sort by score
        matrix_ranked.sort(key=lambda x: x.get("expectancy_score", 0), reverse=True)
        return matrix_ranked

    # ─────────────────────────────────────────────
    # INTERNALS
    # ─────────────────────────────────────────────

    def _collect_regime_warnings(
        self, setups: List[SetupData],
    ) -> List[Dict[str, Any]]:
        warnings = []
        for setup in setups:
            advice = self._regime_matrix.should_avoid(
                setup.strategy_name, setup.current_regime
            )
            if advice.avoid:
                warnings.append({
                    "symbol": setup.symbol,
                    "strategy": setup.strategy_name,
                    "regime": setup.current_regime,
                    "warning": advice.reason,
                    "alternatives": advice.alternatives,
                })
        return warnings

    def _llm_explain(self, result: PlanSessionResult) -> str:
        """LLM narrates engine output. Never computes."""
        if not self._llm:
            return ""
        import asyncio

        # Build structured data for LLM
        engine_data = {
            "behavioral_verdict": result.behavioral_check.verdict,
            "behavioral_proceed": result.behavioral_check.proceed,
            "total_scanned": result.total_scanned,
            "actionable": result.actionable_count,
            "recommendation": result.recommendation,
            "regime_warnings": result.regime_warnings,
            "top_opportunities": [
                {
                    "rank": r.rank,
                    "symbol": r.symbol,
                    "strategy": r.strategy_name,
                    "composite": r.composite_score,
                    "conviction": r.conviction.conviction_score,
                    "action": r.action,
                    "sizing": r.sizing,
                    "explanation": r.explanation,
                }
                for r in result.ranked_opportunities[:5]
            ],
        }

        prompt = (
            "You are a trading co-planner. The engine has computed the "
            "following analysis. Explain it clearly to the trader.\n\n"
            "Be direct. Be specific. Use the numbers provided.\n"
            "Do NOT add your own analysis — only explain what the engine found.\n\n"
            "Format:\n"
            "1. Overall situation (1-2 lines)\n"
            "2. Best opportunity (if any)\n"
            "3. What to avoid and why\n"
            "4. Behavioral notes (if any)\n"
            "5. Final recommendation\n\n"
            f"Engine Result:\n{json.dumps(engine_data, indent=2, default=str)}"
        )

        try:
            from agents.llm_client import LLMClient, LLMMessage
            from agents.prompts import EXPLANATION_ONLY

            async def _call():
                return await self._llm.chat(
                    messages=[
                        LLMMessage(role="system", content=EXPLANATION_ONLY),
                        LLMMessage(role="user", content=prompt),
                    ],
                    temperature=0.3,
                    max_tokens=800,
                    caller="coplanner_explain",
                )

            response = asyncio.run(_call())
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning("LLM explanation failed: %s", e)
            return ""

    def _log_decision(self, session_id: str, result: PlanSessionResult) -> None:
        try:
            top_conv = 0.0
            if result.ranked_opportunities:
                top_conv = result.ranked_opportunities[0].conviction.conviction_score
            self._db.log_decision(session_id, {
                "total_scanned": result.total_scanned,
                "actionable": result.actionable_count,
                "recommendation": result.recommendation,
                "top_conviction": top_conv,
            })
        except Exception:
            logger.debug("Decision logging failed", exc_info=True)
