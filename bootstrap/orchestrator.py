# bootstrap/orchestrator.py
"""
Bootstrap Orchestrator
=======================

End-to-end pipeline coordinator for the Bootstrap Understanding System.

Flow:
1. User describes strategy (NL text + optional pattern selection)
2. NL Interpreter → StrategyDraft
3. Fast Backtester → BacktestResult (synthetic trades)
4. Warm Starter → initializes all downstream systems
5. User reviews → Correction Loop refines understanding
6. Finalize → StrategyProfile (DB) + StrategyDNA (YAML)

Sessions are in-memory (bootstrap is a single-sitting operation).

┌─────────────────────────────────────────────────────────────┐
│  ARCHITECTURAL RULES — LEARNING DOMAIN SEPARATION           │
│                                                             │
│  Rule 1: Onboarding learns WHAT user wants (intent)         │
│  Rule 2: AVT learns WHAT works (edge)                       │
│  Rule 3: AVT NEVER changes user strategy directly           │
│  Rule 4: User system uses scoring, not hard filters         │
│  Rule 5: All improvements = suggestions, not auto-applied   │
│                                                             │
│  StrategyProfile = USER MIND (never overwritten by outcomes)│
│  AVT = MARKET TRUTH (never overwrites user profile)         │
│  They must NEVER overwrite each other.                      │
│                                                             │
│  This means: onboarding NEVER rejects a strategy based on   │
│  backtest outcomes. It only INFORMS the user. Backtest data  │
│  goes to AVT and warm-start, NOT into StrategyProfile.      │
└─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  SESSION MODEL
# ══════════════════════════════════════════════════════════════


@dataclass
class BootstrapSession:
    """Full state of a bootstrap session."""
    session_id: str = ""
    status: str = "created"
    # created → interpreting → backtesting → reviewing → correcting → finalized

    # Input
    user_description: str = ""
    selected_pattern_ids: List[str] = field(default_factory=list)
    timeframe: str = "5min"
    direction: str = "long"
    instruments: List[str] = field(default_factory=list)
    user_claimed_wr: Optional[float] = None

    # Pipeline outputs
    draft: Any = None                    # StrategyDraft
    backtest_result: Any = None          # BacktestResult
    warm_start_report: Any = None        # WarmStartReport

    # Corrections
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    convergence_score: float = 0.0

    # Final output
    profile_id: Optional[str] = None
    yaml_path: Optional[str] = None
    sko_id: Optional[str] = None
    sko_status: Optional[str] = None

    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session for API responses."""
        result = {
            "session_id": self.session_id,
            "status": self.status,
            "iteration": self.iteration,
            "convergence_score": round(self.convergence_score, 3),
            "warnings": self.warnings,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

        if self.draft:
            result["draft"] = {
                "strategy_name": self.draft.strategy_name,
                "understanding_confidence": self.draft.understanding_confidence,
                "hard_rules_count": len(self.draft.hard_rules),
                "soft_scorers_count": len(self.draft.soft_scorers),
                "feature_keys": self.draft.feature_keys,
                "gaps": self.draft.gaps,
                "questions": self.draft.questions_to_ask,
                "warnings": self.draft.warnings,
            }

        if self.backtest_result:
            bt = self.backtest_result
            result["backtest"] = {
                "total_trades": bt.total_trades,
                "win_rate": round(bt.win_rate, 3),
                "avg_r": round(bt.avg_r, 3),
                "total_r": round(bt.total_r, 3),
                "profit_factor": round(bt.profit_factor, 3),
                "max_drawdown_r": round(bt.max_drawdown_r, 3),
                "avg_synthetic_confidence": round(bt.avg_synthetic_confidence, 3),
                "regime_distribution": bt.regime_distribution,
                "warnings": bt.warnings,
            }

        if self.warm_start_report:
            ws = self.warm_start_report
            result["warm_start"] = {
                "scaler_initialized": ws.scaler_initialized,
                "clusterer_initialized": ws.clusterer_initialized,
                "clusters_found": ws.clusters_found,
                "regime_checker_initialized": ws.regime_checker_initialized,
                "performance_reasonable": ws.performance_reasonable,
                "recommended_avt_mode": ws.recommended_avt_mode,
                "warnings": ws.warnings,
            }

        if self.profile_id:
            result["profile_id"] = self.profile_id
        if self.yaml_path:
            result["yaml_path"] = self.yaml_path
        if self.sko_id:
            result["sko_id"] = self.sko_id
            result["sko_status"] = self.sko_status

        return result


# ══════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════


class BootstrapOrchestrator:
    """
    Coordinates the full bootstrap pipeline.

    Usage:
        orchestrator = BootstrapOrchestrator(db=db_session)
        session = await orchestrator.start(description="...", patterns=["compression_breakout"])
        session = await orchestrator.correct(session.session_id, corrections=[...])
        profile, yaml_path = await orchestrator.finalize(session.session_id)
    """

    MAX_ITERATIONS = 5

    def __init__(
        self,
        db: Any = None,
        llm: Any = None,
        broker_facade: Any = None,
        broker_name: str = "",
    ):
        self._db = db
        self._llm = llm
        self._sessions: Dict[str, BootstrapSession] = {}

        # Build adapters (lazy — only when actually running bootstrap, not just listing patterns)
        self._broker_facade = broker_facade
        self._broker_name = broker_name
        self._adapters = None  # initialized on first use

        # Initialize components
        from bootstrap.pattern_library import PatternLibrary
        self._pattern_library = PatternLibrary()

        from bootstrap.nl_interpreter import LLMStrategyInterpreter
        self._interpreter = LLMStrategyInterpreter(
            library=self._pattern_library,
            llm_client=llm,
        )

    async def start(
        self,
        description: str,
        selected_pattern_ids: Optional[List[str]] = None,
        timeframe: str = "5min",
        direction: str = "long",
        instruments: Optional[List[str]] = None,
        user_claimed_wr: Optional[float] = None,
    ) -> BootstrapSession:
        """
        Start a bootstrap session.

        1. Interpret NL description → StrategyDraft
        2. Run fast backtest → synthetic trades
        3. Warm-start downstream systems
        4. Return session ready for review
        """
        session = BootstrapSession(
            session_id=str(uuid.uuid4()),
            user_description=description,
            selected_pattern_ids=selected_pattern_ids or [],
            timeframe=timeframe,
            direction=direction,
            instruments=instruments or [],
            user_claimed_wr=user_claimed_wr,
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[session.session_id] = session

        # Step 1: Interpret
        session.status = "interpreting"
        session.updated_at = datetime.now(timezone.utc)
        try:
            draft = await self._interpreter.interpret(
                user_text=description,
                selected_template_ids=selected_pattern_ids or [],
            )
            session.draft = draft
        except Exception as e:
            session.warnings.append(f"Interpretation failed: {e}")
            logger.error("Bootstrap interpretation failed: %s", e)
            session.status = "reviewing"
            return session

        # Step 2: Backtest
        session.status = "backtesting"
        session.updated_at = datetime.now(timezone.utc)
        try:
            symbols = instruments or self._get_default_symbols()
            backtest_result = await self._run_backtest(draft, symbols)
            session.backtest_result = backtest_result
            session.warnings.extend(backtest_result.warnings)
        except Exception as e:
            session.warnings.append(f"Backtest failed: {e}")
            logger.error("Bootstrap backtest failed: %s", e)

        # Step 3: Warm-start (only if we have backtest results)
        if session.backtest_result and session.backtest_result.trades:
            try:
                warm_report = self._run_warm_start(
                    session.backtest_result, draft, user_claimed_wr
                )
                session.warm_start_report = warm_report
                session.warnings.extend(warm_report.warnings)
            except Exception as e:
                session.warnings.append(f"Warm-start failed: {e}")
                logger.error("Bootstrap warm-start failed: %s", e)

        session.status = "reviewing"
        session.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Bootstrap session %s started: %d trades, confidence %.2f",
            session.session_id[:8],
            session.backtest_result.total_trades if session.backtest_result else 0,
            session.draft.understanding_confidence if session.draft else 0.0,
        )

        return session

    async def correct(
        self,
        session_id: str,
        corrections: List[Dict[str, Any]],
    ) -> BootstrapSession:
        """
        Apply user corrections and re-run pipeline.

        Corrections can be:
        - {"type": "modify_rule", "rule_index": 0, "field": "value", "new_value": 50}
        - {"type": "reject_trade", "trade_id": "syn_RELIANCE_42", "reason": "..."}
        - {"type": "modify_narrative", "text": "I actually mean..."}
        - {"type": "answer_question", "question_index": 0, "answer": "..."}
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        if session.iteration >= self.MAX_ITERATIONS:
            session.warnings.append("Max iterations reached")
            return session

        session.status = "correcting"
        session.corrections.extend(corrections)
        session.iteration += 1
        session.updated_at = datetime.now(timezone.utc)

        # Apply corrections to draft
        if session.draft:
            old_draft = session.draft
            session.draft = await self._apply_corrections(
                session.draft, corrections
            )

            # Compute convergence (how much changed)
            session.convergence_score = self._compute_convergence(
                old_draft, session.draft
            )

        # Re-backtest with updated draft
        if session.draft:
            session.status = "backtesting"
            try:
                symbols = session.instruments or self._get_default_symbols()
                session.backtest_result = await self._run_backtest(
                    session.draft, symbols
                )
                session.warnings = [
                    w for w in session.warnings
                    if "Backtest" not in w
                ]
                session.warnings.extend(session.backtest_result.warnings)
            except Exception as e:
                session.warnings.append(f"Re-backtest failed: {e}")

        session.status = "reviewing"
        session.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Bootstrap session %s corrected (iteration %d, convergence %.2f)",
            session_id[:8], session.iteration, session.convergence_score,
        )

        return session

    async def finalize(
        self, session_id: str, trader_id: Any = None
    ) -> Dict[str, Any]:
        """
        Finalize bootstrap session:
        1. Convert StrategyDraft → StrategyProfile (DB record)
        2. Generate StrategyDNA YAML
        3. Publish BOOTSTRAP_COMPLETED event

        Returns: {"profile_id": str, "yaml_path": str, "short_code": str}
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        if not session.draft:
            raise ValueError("No draft to finalize")

        session.status = "finalizing"
        session.updated_at = datetime.now(timezone.utc)

        draft = session.draft

        # Build StrategyProfile from draft
        profile = self._draft_to_profile(draft, session, trader_id)

        # Save to database
        if self._db is not None:
            try:
                self._db.add(profile)
                self._db.commit()
                self._db.refresh(profile)
                session.profile_id = str(profile.id)
            except Exception as e:
                self._db.rollback()
                logger.error("Failed to save profile: %s", e)
                raise

        # Compile into a tradeable SKO (scoring_config) so AVT can trade it.
        # Never fatal — onboarding still succeeds even if compile is skipped.
        try:
            self._compile_and_persist_sko(draft, session, profile)
        except Exception as e:
            logger.warning("SKO compile step failed (non-fatal): %s", e)
            session.warnings.append(f"Strategy compile skipped: {e}")

        # Generate DNA YAML (name:-format → StrategyDNALoader / guidance)
        try:
            from strategy_learning.dna_generator import save_dna_yaml
            yaml_path = save_dna_yaml(profile)
            session.yaml_path = yaml_path
        except Exception as e:
            logger.warning("DNA YAML generation failed: %s", e)
            session.warnings.append(f"YAML generation failed: {e}")

        # Generate registry template (strategy:-format) so the intelligence
        # scanner, co-planner, behavioral matcher and setup scanner become aware
        # of THIS strategy. Without it, teaching would never reach those engines.
        try:
            from strategy_engine.templates.profile_to_template import save_template_yaml
            save_template_yaml(
                profile,
                direction=getattr(session, "direction", None),
                timeframe=getattr(draft, "timeframe", None),
            )
        except Exception as e:
            logger.warning("Registry template generation failed: %s", e)
            session.warnings.append(f"Strategy not wired into intelligence: {e}")

        # Publish event
        try:
            from core.events import get_event_bus, Event, EventType
            get_event_bus().publish(Event(
                event_type=EventType.BOOTSTRAP_COMPLETED,
                source="bootstrap.orchestrator",
                strategy_id=draft.strategy_name,
                data={
                    "session_id": session.session_id,
                    "profile_id": session.profile_id,
                    "strategy_name": draft.strategy_name,
                    "yaml_path": session.yaml_path,
                    "understanding_confidence": draft.understanding_confidence,
                    "uncertainty_factors": [
                        f for f in (draft.uncertainty_factors or [])
                        if f.get("can_learn")
                    ],
                    "synthetic_trades": (
                        session.backtest_result.total_trades
                        if session.backtest_result else 0
                    ),
                },
            ))
        except Exception as e:
            logger.warning("Event publish failed: %s", e)

        session.status = "finalized"
        session.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Bootstrap session %s finalized → profile %s, YAML %s",
            session_id[:8],
            session.profile_id or "not saved",
            session.yaml_path or "not generated",
        )

        return {
            "profile_id": session.profile_id,
            "yaml_path": session.yaml_path,
            "short_code": profile.short_code,
            "strategy_name": profile.name,
        }

    def get_session(self, session_id: str) -> Optional[BootstrapSession]:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)

    def list_patterns(self) -> List[Dict[str, Any]]:
        """List all available pattern templates."""
        return [
            {
                "pattern_id": pt.pattern_id,
                "name": pt.name,
                "category": pt.category,
                "description": pt.description,
                "hard_rules_count": len(pt.hard_rules),
                "variants_count": len(pt.variants),
                "regime_performance": pt.regime_performance,
            }
            for pid in self._pattern_library.get_all_ids()
            for pt in [self._pattern_library.get_template(pid)]
            if pt
        ]

    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get full details of a pattern template."""
        pt = self._pattern_library.get_template(pattern_id)
        if pt is None:
            return None
        return {
            "pattern_id": pt.pattern_id,
            "name": pt.name,
            "category": pt.category,
            "description": pt.description,
            "hard_rules": pt.hard_rules,
            "soft_scorers": pt.soft_scorers,
            "context_factors": pt.context_factors,
            "feature_keys": pt.feature_keys,
            "variants": [
                {"name": v.name, "description": v.description}
                for v in pt.variants
            ],
            "regime_performance": pt.regime_performance,
            "failure_modes": [
                {"name": fm.name, "description": fm.description}
                for fm in pt.failure_modes
            ],
            "uncertainty_factors": pt.uncertainty_factors,
            "teaching_questions": pt.teaching_questions,
        }

    def search_patterns(self, query: str) -> List[Dict[str, Any]]:
        """Search patterns by keyword."""
        matches = self._pattern_library.search(query)
        return [
            {
                "pattern_id": pt.pattern_id,
                "name": pt.name,
                "category": pt.category,
                "description": pt.description,
                "match_score": score,
            }
            for pt, score in matches
        ]

    # ── Private Methods ─────────────────────────────────────

    async def _run_backtest(
        self, draft: Any, symbols: List[str]
    ) -> Any:
        """Run fast backtest with adapter-wrapped services."""
        from bootstrap.fast_backtester import FastBacktestEngine
        if self._adapters is None:
            from bootstrap.adapters import build_bootstrap_adapters
            self._adapters = build_bootstrap_adapters(
                broker_facade=self._broker_facade,
                broker_name=self._broker_name,
            )

        engine = FastBacktestEngine(
            market_data=self._adapters["market_data"],
            feature_computer=self._adapters["feature_computer"],
            regime_engine=self._adapters["regime_engine"],
        )
        return await engine.run(
            draft=draft,
            symbols=symbols,
            lookback_days=180,
            max_trades=200,
        )

    def _run_warm_start(
        self, backtest_result: Any, draft: Any, user_claimed_wr: Optional[float]
    ) -> Any:
        """Run warm-start with adapter-wrapped services."""
        from bootstrap.warm_starter import WarmStarter

        # Build pattern intelligence from draft's feature keys
        feature_keys = draft.feature_keys if draft.feature_keys else []
        from bootstrap.adapters import PatternIntelFacade, StrategyEngineFacade

        starter = WarmStarter(
            strategy_engine=StrategyEngineFacade(),
            pattern_intel=PatternIntelFacade(feature_keys=feature_keys),
        )
        return starter.warm_start(
            backtest_result=backtest_result,
            draft=draft,
            user_claimed_wr=user_claimed_wr,
        )

    async def _apply_corrections(
        self, draft: Any, corrections: List[Dict[str, Any]]
    ) -> Any:
        """Apply user corrections to the draft."""
        for correction in corrections:
            ctype = correction.get("type", "")

            if ctype == "modify_rule":
                idx = correction.get("rule_index", -1)
                if 0 <= idx < len(draft.hard_rules):
                    field_name = correction.get("field", "")
                    new_value = correction.get("new_value")
                    if field_name and new_value is not None:
                        draft.hard_rules[idx][field_name] = new_value

            elif ctype == "add_rule":
                draft.hard_rules.append(correction.get("rule", {}))

            elif ctype == "remove_rule":
                idx = correction.get("rule_index", -1)
                if 0 <= idx < len(draft.hard_rules):
                    draft.hard_rules.pop(idx)

            elif ctype == "modify_narrative":
                # Re-interpret with additional context
                text = correction.get("text", "")
                if text:
                    draft = await self._interpreter.refine(
                        draft=draft,
                        additional_input=text,
                    )

            elif ctype == "answer_question":
                answer = correction.get("answer", "")
                if answer:
                    draft = await self._interpreter.refine(
                        draft=draft,
                        additional_input=answer,
                    )

            elif ctype == "modify_exit":
                field_name = correction.get("field", "")
                new_value = correction.get("new_value")
                if field_name:
                    for exit_rule in draft.exit_logic:
                        if field_name in str(exit_rule):
                            exit_rule["params"] = exit_rule.get("params", {})
                            exit_rule["params"][field_name] = new_value
                            break

        return draft

    def _compute_convergence(self, old_draft: Any, new_draft: Any) -> float:
        """
        Compute convergence score between old and new drafts.
        1.0 = identical (fully converged), 0.0 = completely different.
        """
        score = 0.0
        weights = {
            "hard_rules": 0.35,
            "soft_scorers": 0.20,
            "exit_logic": 0.20,
            "feature_keys": 0.15,
            "confidence": 0.10,
        }

        # Hard rules similarity
        old_rules = str(sorted(str(r) for r in old_draft.hard_rules))
        new_rules = str(sorted(str(r) for r in new_draft.hard_rules))
        score += weights["hard_rules"] * (1.0 if old_rules == new_rules else 0.3)

        # Soft scorers similarity
        old_soft = str(sorted(str(s) for s in old_draft.soft_scorers))
        new_soft = str(sorted(str(s) for s in new_draft.soft_scorers))
        score += weights["soft_scorers"] * (1.0 if old_soft == new_soft else 0.3)

        # Exit logic similarity
        old_exit = str(old_draft.exit_logic)
        new_exit = str(new_draft.exit_logic)
        score += weights["exit_logic"] * (1.0 if old_exit == new_exit else 0.3)

        # Feature keys overlap
        old_keys = set(old_draft.feature_keys)
        new_keys = set(new_draft.feature_keys)
        if old_keys or new_keys:
            overlap = len(old_keys & new_keys) / max(1, len(old_keys | new_keys))
            score += weights["feature_keys"] * overlap
        else:
            score += weights["feature_keys"]

        # Confidence change
        old_conf = old_draft.understanding_confidence or 0.0
        new_conf = new_draft.understanding_confidence or 0.0
        conf_diff = abs(old_conf - new_conf)
        score += weights["confidence"] * max(0, 1.0 - conf_diff * 5)

        return round(score, 3)

    def _draft_to_profile(
        self, draft: Any, session: BootstrapSession, trader_id: Any = None
    ) -> Any:
        """Convert StrategyDraft → StrategyProfile ORM model."""
        from core.models import StrategyProfile

        # Generate short code from strategy name
        name = draft.strategy_name or "bootstrap_strategy"
        short_code = self._generate_short_code(name)

        # Build confidence scores
        confidence = draft.understanding_confidence or 0.5
        conf_scores = {
            "entry": confidence,
            "exit": confidence * 0.8,  # exits typically less certain
            "stop": confidence * 0.9,
            "overall": confidence,
            "source": "bootstrap",
            "iterations": session.iteration,
        }

        # Build entry criteria from hard rules + soft scorers
        entry_criteria = []
        for rule in draft.hard_rules:
            entry_criteria.append({
                "criterion": f"{rule.get('field', '')} {rule.get('operator', '')} {rule.get('value', '')}",
                "description": rule.get("description", ""),
                "confidence": rule.get("confidence", confidence),
                "source": rule.get("source", "bootstrap"),
            })
        for scorer in draft.soft_scorers:
            entry_criteria.append({
                "criterion": scorer.get("name", ""),
                "description": scorer.get("description", ""),
                "confidence": scorer.get("weight", 0.5),
                "source": "bootstrap_soft",
            })

        # Build exit rules from draft exit logic
        exit_rules = {"stop_method": "atr", "targets": []}
        for exit_rule in (draft.exit_logic or []):
            if exit_rule.get("type") == "stop_loss":
                exit_rules["stop_method"] = exit_rule.get("method", "atr")
                exit_rules["typical_stop_atr"] = (
                    exit_rule.get("params", {}).get("atr_multiple", 1.5)
                )
            elif exit_rule.get("type") == "target":
                exit_rules["targets"].append(exit_rule.get("params", {}))

        # Build regime preferences from matched template data
        regime_prefs = {"preferred": [], "avoid": []}
        for template in (draft.matched_templates or []):
            perf = getattr(template, "regime_performance", {})
            if isinstance(perf, dict):
                for regime, rating in perf.items():
                    if rating in ("excellent", "good"):
                        regime_prefs["preferred"].append(
                            {"regime": regime, "confidence": confidence}
                        )
                    elif rating in ("poor", "avoid"):
                        regime_prefs["avoid"].append(
                            {"regime": regime, "confidence": confidence}
                        )

        profile = StrategyProfile(
            name=name,
            short_code=short_code,
            trader_id=trader_id,
            status="active",
            narrative=self._build_narrative(draft, session),
            entry_criteria=entry_criteria,
            exit_rules=exit_rules,
            risk_parameters={
                "source": "bootstrap",
                # NOTE: Synthetic trade count/WR intentionally excluded.
                # StrategyProfile stores INTENT, not outcome metrics.
            },
            regime_preferences=regime_prefs,
            time_preferences={},
            confidence_scores=conf_scores,
            examples_count=0,
            sessions_count=0,
            tags=["bootstrap", f"iter_{session.iteration}"],
            profile_metadata={
                "bootstrap_session_id": session.session_id,
                "bootstrap_convergence": session.convergence_score,
                "matched_patterns": [
                    t.pattern_id for t in (draft.matched_templates or [])
                    if hasattr(t, "pattern_id")
                ],
                # Timeframe the user trades on — drives scanner candle interval
                "preferred_timeframes": [draft.timeframe] if draft.timeframe else ["15m"],
            },
        )

        return profile

    def _compile_and_persist_sko(
        self, draft: Any, session: BootstrapSession, profile: Any
    ) -> None:
        """
        Compile the onboarded strategy into a tradeable SKO (with scoring_config)
        and persist it to strategy_knowledge_objects so AVT can load and trade it.

        Reuses SKOBuilder.build_from_trades on the synthetic backtest trades.
        The real entry-candle windows (SyntheticTrade.entry_candles) are passed
        so SKE's FeatureComputer derives a scoring_config keyed on the SAME
        features AVT computes at runtime — otherwise signals never fire.
        """
        if self._db is None:
            return

        bt = session.backtest_result
        trades = list(getattr(bt, "trades", []) or []) if bt else []
        if len(trades) < 10:
            session.warnings.append(
                "Not enough signal to compile a tradeable strategy yet "
                f"({len(trades)} synthetic trades; need 10+). Teach more "
                "examples or widen the strategy, then re-finalize."
            )
            return

        # Map synthetic trades → builder trade dicts (+ candle windows)
        trade_dicts: List[Dict[str, Any]] = []
        candles_by_trade: Dict[str, List[Dict[str, Any]]] = {}
        for t in trades:
            tid = str(getattr(t, "id", "") or "")
            trade_dicts.append({
                "id": tid,
                "symbol": getattr(t, "symbol", ""),
                "entry_price": getattr(t, "entry_price", 0.0),
                "exit_price": getattr(t, "exit_price", 0.0),
                "stop_loss": getattr(t, "stop_price", 0.0),  # builder reads stop_loss
                "direction": getattr(t, "direction", "long"),
                "r_multiple": getattr(t, "r_multiple", 0.0),
                "conviction_score": getattr(t, "conviction_score", 0.0),
            })
            ec = getattr(t, "entry_candles", None)
            if ec:
                candles_by_trade[tid] = ec

        try:
            from ske.builder import SKOBuilder
            built = SKOBuilder().build_from_trades(
                trade_dicts,
                name=profile.name,
                short_code=profile.short_code,
                description=self._build_narrative(draft, session),
                candles_by_trade=candles_by_trade or None,
                apply_quality_gate=False,  # synthetic trades lack discipline/journal scores
            )
        except Exception as e:
            logger.error("SKO compile failed: %s", e)
            session.warnings.append(f"Strategy compile failed: {e}")
            return

        if not built.get("scoring_config"):
            session.warnings.append(
                "Could not derive a scoring config from the examples — "
                "strategy not yet tradeable by AVT."
            )
            return

        # Persist (upsert on short_code) — mirrors api/routes/ske.py build, sync session
        from core.models import StrategyKnowledgeObject, SKOVariant
        from core.enums import SKOStatus

        sko = (
            self._db.query(StrategyKnowledgeObject)
            .filter(StrategyKnowledgeObject.short_code == profile.short_code)
            .first()
        )
        is_new = sko is None
        if is_new:
            sko = StrategyKnowledgeObject(
                id=uuid.uuid4(),
                name=profile.name[:100],
                short_code=profile.short_code[:30],
            )

        sko.name = profile.name[:100]
        sko.narrative = getattr(profile, "narrative", None)
        sko.description = built.get("description", "")
        sko.trader_id = getattr(profile, "trader_id", None)
        try:
            sko.status = SKOStatus(built.get("status", "hypothesis"))
        except ValueError:
            sko.status = SKOStatus.HYPOTHESIS
        for fld in [
            "tier", "total_trades", "win_rate", "expectancy", "profit_factor",
            "scoring_config", "conviction_map", "entry_zones", "exit_rules",
            "risk_parameters", "regime_fingerprint", "time_preferences",
            "dqal_score", "dqal_grade", "tags",
        ]:
            if built.get(fld) is not None:
                setattr(sko, fld, built[fld])
        sko.updated_at = datetime.now(timezone.utc)

        if is_new:
            self._db.add(sko)
            self._db.flush()  # assign sko.id
        else:
            self._db.query(SKOVariant).filter(
                SKOVariant.sko_id == sko.id
            ).delete()

        for v in built.get("variants", []):
            self._db.add(SKOVariant(
                id=uuid.uuid4(),
                sko_id=sko.id,
                variant_type="anti_variant" if v.get("is_anti_variant") else "variant",
                label=v.get("label", "unknown"),
                centroid=v.get("centroid", {}),
                trade_count=v.get("trade_count", 0),
                win_rate=v.get("win_rate"),
                avg_r=v.get("avg_r"),
                expectancy=v.get("expectancy"),
                distinguishing_features=v.get("distinguishing_features", []),
            ))

        self._db.commit()
        self._db.refresh(sko)
        session.sko_id = str(sko.id)
        session.sko_status = (
            sko.status.value if hasattr(sko.status, "value") else str(sko.status)
        )

        logger.info(
            "Compiled tradeable SKO %s (%s) status=%s, %d scoring features, %d trades",
            profile.short_code, session.sko_id, session.sko_status,
            len(built.get("scoring_config", {})), len(trade_dicts),
        )

        # Notify AVT to hot-reload the new strategy
        try:
            from core.events import get_event_bus, Event, EventType
            get_event_bus().publish(Event(
                event_type=EventType.SKO_CREATED,
                source="bootstrap.orchestrator",
                data={
                    "sko_id": session.sko_id,
                    "short_code": profile.short_code,
                    "name": profile.name,
                    "status": session.sko_status,
                },
            ))
        except Exception as e:
            logger.debug("SKO_CREATED publish failed: %s", e)

    def _build_narrative(self, draft: Any, session: BootstrapSession) -> str:
        """Build human-readable narrative from draft."""
        parts = []

        name = draft.strategy_name or "Strategy"
        parts.append(f"**{name}** (bootstrapped)")
        parts.append("")

        if draft.matched_templates:
            template_names = [
                getattr(t, "name", str(t))
                for t in draft.matched_templates
            ]
            parts.append(f"Based on: {', '.join(template_names)}")

        if draft.hard_rules:
            parts.append(f"\nEntry requires {len(draft.hard_rules)} hard rule(s):")
            for rule in draft.hard_rules[:5]:
                field = rule.get("field", "")
                op = rule.get("operator", "")
                val = rule.get("value", "")
                parts.append(f"  - {field} {op} {val}")

        # NOTE: Backtest outcome data (WR, Avg R) is intentionally EXCLUDED
        # from the profile narrative. StrategyProfile = USER INTENT (what user
        # wants). Backtest outcomes = MARKET EDGE (what works). Mixing them
        # causes intent drift. Backtest results are stored separately in
        # session.backtest_result and warm_start_report for AVT to use.

        conf = draft.understanding_confidence or 0.0
        parts.append(f"\nUnderstanding confidence: {conf:.0%}")

        if draft.gaps:
            parts.append(f"\nKnown gaps: {', '.join(draft.gaps[:3])}")

        return "\n".join(parts)

    @staticmethod
    def _generate_short_code(name: str) -> str:
        """Generate a short code from strategy name."""
        # "compression_breakout" → "COMP_BO"
        # "my pullback strategy" → "MY_PB"
        words = name.lower().replace("-", "_").split("_")
        if len(words) == 1:
            words = name.lower().split()

        if len(words) >= 2:
            code = words[0][:4].upper() + "_" + words[1][:2].upper()
        else:
            code = words[0][:6].upper()

        # Add random suffix for uniqueness
        suffix = str(uuid.uuid4())[:4].upper()
        return f"{code}_{suffix}"

    @staticmethod
    def _get_default_symbols() -> List[str]:
        """Default FNO symbols for backtesting."""
        return [
            "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK",
            "SBIN", "BAJFINANCE", "BHARTIARTL", "ITC", "KOTAKBANK",
            "LT", "AXISBANK", "HCLTECH", "WIPRO", "MARUTI",
        ]
