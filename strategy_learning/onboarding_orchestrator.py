"""
============================================================
STRATEGY ONBOARDING ENGINE — MAIN ORCHESTRATOR
============================================================
File: soe/orchestrator.py
============================================================

This is the TOP-LEVEL controller that orchestrates the
entire onboarding flow from start to finish.

User interacts with THIS. Everything else is internal.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger("SOE.Orchestrator")


class OnboardingState(Enum):
    """Current state of the onboarding process"""
    NOT_STARTED = "not_started"
    NARRATIVE_RECEIVED = "narrative_received"
    COLLECTING_EXAMPLES = "collecting_examples"
    READY_FOR_LEARNING = "ready_for_learning"
    LEARNING_COMPLETE = "learning_complete"
    VALIDATING = "validating"
    QUIZ_IN_PROGRESS = "quiz_in_progress"
    AWAITING_CORRECTIONS = "awaiting_corrections"
    COLLECTING_MORE_EXAMPLES = "collecting_more_examples"
    COMPLETE = "complete"
    FAILED = "failed"


class StrategyOnboardingEngine:
    """
    MAIN ORCHESTRATOR for the entire onboarding process.
    
    This is the single entry point that:
    1. Receives user's strategy description
    2. Guides example collection
    3. Triggers learning
    4. Validates understanding
    5. Produces final SKO
    6. Connects to AVT scoring
    
    ═══════════════════════════════════════════
    USAGE:
    ═══════════════════════════════════════════
    
    # Initialize
    engine = StrategyOnboardingEngine(data_provider=my_data_source)
    
    # Step 1: User describes strategy
    result = engine.receive_strategy_description(
        strategy_id="compression_breakout",
        description="I trade compression breakouts near support...",
        intake_results={...}  # From IntakeAssessor
    )
    
    # Step 2: User provides examples
    for example in user_examples:
        result = engine.add_example(
            strategy_id="compression_breakout",
            symbol="NIFTY",
            timeframe="15min",
            date="2024-01-10",
            label="good",
            explanation="Clean compression, volume dying..."
        )
    
    # Step 3: Trigger learning
    result = engine.run_learning("compression_breakout")
    
    # Step 4: Get understanding report
    report = engine.get_understanding_report("compression_breakout")
    
    # Step 5: Quiz
    quiz = engine.start_quiz("compression_breakout", candidate_setups)
    
    # Step 6: User answers quiz
    for question in quiz:
        engine.answer_quiz(question.id, user_answer)
    
    # Step 7: Finalize
    sko = engine.finalize_onboarding("compression_breakout")
    
    # Step 8: Get AVT config
    avt_config = engine.get_avt_scoring_config("compression_breakout")
    """

    def __init__(
        self,
        data_provider: 'DataProvider',
        intake_assessor: Optional[Any] = None,  # Existing IntakeAssessor
    ):
        # Core components
        self.data_provider = data_provider
        self.intake_assessor = intake_assessor

        # Internal modules
        self.example_collector = ExampleCollector(data_provider)
        self.feature_extractor = OnboardingFeatureExtractor()
        self.regime_builder = RegimeDefinitionBuilder()
        self.domain_separator = LearningDomainSeparator()

        # Per-strategy state
        self.strategies: Dict[str, Dict[str, Any]] = {}

        # Results storage
        self.pattern_learners: Dict[str, OnboardingPatternLearner] = {}
        self.pattern_discoveries: Dict[str, PatternDiscovery] = {}
        self.regime_definitions: Dict[str, RegimeDefinition] = {}
        self.confidence_decompositions: Dict[str, ConfidenceDecomposition] = {}
        self.understanding_reports: Dict[str, UnderstandingReport] = {}
        self.validators: Dict[str, UnderstandingValidator] = {}
        self.final_skos: Dict[str, OnboardingSKO] = {}
        self.avt_configs: Dict[str, AVTScoringConfig] = {}

    # ═══════════════════════════════════════════════════════
    # STEP 1: RECEIVE STRATEGY DESCRIPTION
    # ═══════════════════════════════════════════════════════

    def receive_strategy_description(
        self,
        strategy_id: str,
        description: str,
        intake_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        First step: User describes their strategy in natural language.
        
        If IntakeAssessor has already processed it, pass intake_results.
        Otherwise, this will do basic parsing.
        """
        logger.info(f"Receiving strategy description for {strategy_id}")

        # Initialize strategy state
        self.strategies[strategy_id] = {
            "state": OnboardingState.NARRATIVE_RECEIVED,
            "description": description,
            "intake_results": intake_results,
            "created_at": datetime.now().isoformat(),
            "history": []
        }

        # Parse what needs examples
        needs_examples = self._identify_example_needs(
            description, intake_results
        )

        # Extract regime mentions for RegimeDefinitionBuilder
        regime_mentions = self._extract_regime_mentions(description)
        if regime_mentions:
            self.regime_definitions[strategy_id] = (
                self.regime_builder.build_from_description(
                    strategy_id, description, regime_mentions
                )
            )

        self._add_history(strategy_id, "narrative_received")

        response = {
            "status": "success",
            "state": OnboardingState.NARRATIVE_RECEIVED.value,
            "message": self._generate_example_request_message(
                needs_examples, intake_results
            ),
            "needs_examples_for": needs_examples,
            "hard_rules_extracted": (
                len(intake_results.get("components", []))
                if intake_results else 0
            ),
            "regime_detected": regime_mentions,
            "next_step": "add_example",
            "minimum_examples": 15,
            "recommended_examples": 25,
            "guidelines": {
                "mix_of_labels": "Include both TAKE and SKIP examples",
                "diverse_contexts": "Different days, symbols, market conditions",
                "explain_why": "For each example, explain WHY you would take or skip",
                "include_edge_cases": "Include borderline decisions too"
            }
        }

        self.strategies[strategy_id]["state"] = OnboardingState.COLLECTING_EXAMPLES

        return response

    # ═══════════════════════════════════════════════════════
    # STEP 2: ADD EXAMPLES
    # ═══════════════════════════════════════════════════════

    def add_example(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        date: str,
        label: str,
        explanation: str,
        screenshot_path: Optional[str] = None,
        entry_time: Optional[str] = None,
        outcome: Optional[str] = None,
        outcome_r: Optional[float] = None,
        label_confidence: float = 0.8
    ) -> Dict[str, Any]:
        """
        Add a single training example.
        
        This is called 20-30 times during onboarding.
        Each call collects, validates, and processes one example.
        """
        if strategy_id not in self.strategies:
            return {
                "status": "error",
                "message": f"Strategy {strategy_id} not found. "
                          f"Call receive_strategy_description first."
            }

        # ── Collect example ──
        collection_result = self.example_collector.collect_example(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            date=date,
            label=label,
            explanation=explanation,
            screenshot_path=screenshot_path,
            entry_time=entry_time,
            outcome=outcome,
            outcome_r=outcome_r,
            label_confidence=label_confidence
        )

        example = collection_result["example"]

        # ── Extract features ──
        if example.candle_data is not None:
            features = self.feature_extractor.extract_features(example)
            example.features = features

        # ── Route through domain separator ──
        signals = self.domain_separator.route_onboarding_example(example)

        # Check if outcome data was provided (warn about intent vs edge)
        outcome_warning = None
        if outcome_r is not None:
            outcome_warning = (
                "📝 Note: Outcome data stored for AVT reference, but "
                "NOT used for strategy understanding. Onboarding learns "
                "your INTENT (what you look for), not EDGE (what wins)."
            )

        # ── Check readiness ──
        collection_status = self.example_collector.get_collection_status(
            strategy_id
        )

        # ── Auto-trigger learning prompt ──
        auto_learn = False
        if (collection_status["ready_for_learning"] and
            collection_status["balanced"]):
            auto_learn = True

        self._add_history(strategy_id, "example_added", {
            "example_id": example.example_id,
            "label": label,
            "features_extracted": len(example.features)
        })

        return {
            "status": collection_result["status"],
            "example_id": example.example_id,
            "features_extracted": len(example.features),
            "message": collection_result["message"],
            "warnings": collection_result["warnings"],
            "outcome_warning": outcome_warning,
            "collection_status": collection_status,
            "ready_for_learning": auto_learn,
            "signals_routed": len(signals),
            "parsed_explanation": collection_result.get("parsed_explanation", {}),
            "next_step": (
                "run_learning" if auto_learn
                else "add_example"
            )
        }

    def add_examples_batch(
        self,
        strategy_id: str,
        examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Add multiple examples at once (batch mode).
        
        Args:
            examples: List of dicts with keys matching add_example params
        """
        results = []
        for ex_data in examples:
            result = self.add_example(
                strategy_id=strategy_id,
                **ex_data
            )
            results.append(result)

        # Summary
        accepted = sum(1 for r in results if r["status"] == "accepted")
        partial = sum(1 for r in results if r["status"] == "partial")
        warnings = [
            w for r in results for w in r.get("warnings", [])
        ]

        collection_status = self.example_collector.get_collection_status(
            strategy_id
        )

        return {
            "status": "batch_complete",
            "total_submitted": len(examples),
            "accepted": accepted,
            "partial": partial,
            "rejected": len(examples) - accepted - partial,
            "warnings": warnings[:10],  # Top 10 warnings
            "collection_status": collection_status,
            "ready_for_learning": collection_status["ready_for_learning"],
            "individual_results": results
        }

    # ═══════════════════════════════════════════════════════
    # STEP 3: RUN LEARNING
    # ═══════════════════════════════════════════════════════

    def run_learning(self, strategy_id: str) -> Dict[str, Any]:
        """
        Trigger the pattern learning process.
        
        This is where the ML magic happens:
        - Analyzes all examples
        - Discovers patterns
        - Trains model
        - Identifies gaps
        """
        if strategy_id not in self.strategies:
            return {"status": "error", "message": "Strategy not found"}

        examples = self.example_collector.examples.get(strategy_id, [])
        if len(examples) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Only {len(examples)} examples. Need at least 10.",
                "current": len(examples),
                "minimum": 10
            }

        logger.info(f"Running learning for {strategy_id} "
                    f"with {len(examples)} examples")

        self.strategies[strategy_id]["state"] = OnboardingState.READY_FOR_LEARNING

        # ── Run Pattern Learner ──
        learner = OnboardingPatternLearner()
        discovery = learner.learn(examples, strategy_id)

        self.pattern_learners[strategy_id] = learner
        self.pattern_discoveries[strategy_id] = discovery

        # ── Refine Regime Definition from examples ──
        regime_examples = []
        for ex in examples:
            if ex.features:
                regime_features = {
                    k: v for k, v in ex.features.items()
                    if k in [
                        "trend_direction_score", "adx", "price_vs_ema50",
                        "price_vs_ema20", "ema_20_slope", "trend_strength",
                        "atr_percentile", "higher_lows_count", "lower_highs_count"
                    ]
                }
                intent = "would_take" if (
                    hasattr(ex.label, 'is_positive') and ex.label.is_positive
                ) else "would_skip"

                regime_examples.append({
                    "intent": intent,
                    "regime_features": regime_features
                })

        if regime_examples:
            refined_regime = self.regime_builder.refine_from_examples(
                strategy_id, regime_examples
            )
            self.regime_definitions[strategy_id] = refined_regime

        # ── Compute Confidence Decomposition ──
        confidence_computer = ConfidenceComputer()
        confidence = confidence_computer.compute(
            pattern_discovery=discovery,
            regime_definition=self.regime_definitions.get(strategy_id),
            n_examples=len(examples)
        )
        self.confidence_decompositions[strategy_id] = confidence

        self.strategies[strategy_id]["state"] = OnboardingState.LEARNING_COMPLETE
        self._add_history(strategy_id, "learning_complete", {
            "model_accuracy": discovery.model_accuracy,
            "n_features": len(discovery.feature_discriminators),
            "confidence": confidence.overall_confidence
        })

        return {
            "status": "learning_complete",
            "model_accuracy": discovery.model_accuracy,
            "model_accuracy_std": discovery.model_accuracy_std,
            "n_examples_used": discovery.n_examples,
            "n_discriminating_features": len([
                d for d in discovery.feature_discriminators.values()
                if d.strength != FeatureDiscriminatorStrength.NONE
            ]),
            "top_features": discovery.top_discriminating_features[:5],
            "hidden_features": discovery.hidden_features,
            "context_dependencies_found": len([
                d for d in discovery.context_dependencies
                if d.is_significant
            ]),
            "interaction_effects_found": len(discovery.interaction_effects),
            "confidence_decomposition": confidence.to_display(),
            "overall_confidence": confidence.overall_confidence,
            "weakest_component": confidence.weakest_component,
            "next_step": "get_understanding_report"
        }

    # ═══════════════════════════════════════════════════════
    # STEP 4: UNDERSTANDING REPORT
    # ═══════════════════════════════════════════════════════

    def get_understanding_report(
        self, strategy_id: str
    ) -> Dict[str, Any]:
        """
        Generate and return the understanding report.
        User reviews this to validate system's understanding.
        """
        discovery = self.pattern_discoveries.get(strategy_id)
        learner = self.pattern_learners.get(strategy_id)

        if discovery is None or learner is None:
            return {
                "status": "error",
                "message": "Run learning first"
            }

        # Create validator
        validator = UnderstandingValidator(learner, discovery)
        examples = self.example_collector.examples.get(strategy_id, [])
        validator.set_examples(examples)
        self.validators[strategy_id] = validator

        # Generate report
        regime_def = self.regime_definitions.get(strategy_id)
        report = validator.generate_understanding_report(regime_def)
        self.understanding_reports[strategy_id] = report

        self.strategies[strategy_id]["state"] = OnboardingState.VALIDATING

        return {
            "status": "report_generated",
            "report": report.to_display(),
            "report_data": {
                "good_profile": report.good_setup_profile,
                "bad_profile": report.bad_setup_profile,
                "context_insights": report.context_insights,
                "gaps": [
                    {
                        "type": g.gap_type,
                        "description": g.description,
                        "severity": g.severity,
                        "action": g.suggested_action
                    }
                    for g in report.understanding_gaps
                ],
                "confidence": report.overall_confidence,
                "additional_examples_needed": report.additional_examples_needed
            },
            "has_gaps": len(report.understanding_gaps) > 0,
            "high_severity_gaps": len([
                g for g in report.understanding_gaps
                if g.severity == "high"
            ]),
            "next_step": "start_quiz" if report.overall_confidence > 0.5
                        else "add_more_examples",
            "user_action_required": (
                "Please review this report. Is it accurate? "
                "What did I get wrong? What did I miss?"
            )
        }

    # ═══════════════════════════════════════════════════════
    # STEP 5: USER CORRECTIONS
    # ═══════════════════════════════════════════════════════

    def submit_corrections(
        self,
        strategy_id: str,
        corrections: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        User corrects the understanding report.
        
        corrections format:
        [
            {
                "feature": "wick_ratio",
                "correction": "I actually don't care about wick ratio as much. 
                              It's more about overall candle 'feel'.",
                "type": "overweighted"
            },
            {
                "feature": "breakout_style",
                "correction": "You missed that I prefer controlled breakouts, 
                              not explosive ones.",
                "type": "missing"
            }
        ]
        """
        self.strategies[strategy_id]["state"] = OnboardingState.AWAITING_CORRECTIONS

        # Store corrections
        if "corrections" not in self.strategies[strategy_id]:
            self.strategies[strategy_id]["corrections"] = []
        self.strategies[strategy_id]["corrections"].extend(corrections)

        # Analyze corrections
        needs_more = []
        for correction in corrections:
            if correction.get("type") == "missing":
                needs_more.append(
                    f"Show me examples where '{correction['feature']}' "
                    f"was the key factor: {correction['correction']}"
                )
            elif correction.get("type") == "overweighted":
                logger.info(
                    f"User says {correction['feature']} is overweighted. "
                    f"Will adjust in next learning round."
                )
            elif correction.get("type") == "wrong":
                needs_more.append(
                    f"Show me examples that demonstrate: {correction['correction']}"
                )

        self._add_history(strategy_id, "corrections_received", {
            "n_corrections": len(corrections)
        })

        return {
            "status": "corrections_received",
            "message": f"Received {len(corrections)} corrections. Thank you!",
            "needs_more_examples": needs_more,
            "should_relearn": len(needs_more) > 0,
            "next_step": (
                "add_more_examples" if needs_more
                else "start_quiz"
            )
        }

    # ═══════════════════════════════════════════════════════
    # STEP 6: QUIZ
    # ═══════════════════════════════════════════════════════

    def start_quiz(
        self,
        strategy_id: str,
        candidate_setups: List[Dict[str, float]],
        n_questions: int = 5
    ) -> Dict[str, Any]:
        """
        Start quiz mode to validate understanding.
        
        Args:
            candidate_setups: List of feature dicts from market scan
                            or historical setups NOT in training set
        """
        validator = self.validators.get(strategy_id)
        if validator is None:
            return {"status": "error", "message": "Run understanding report first"}

        self.strategies[strategy_id]["state"] = OnboardingState.QUIZ_IN_PROGRESS

        questions = validator.generate_quiz(candidate_setups, n_questions)

        return {
            "status": "quiz_started",
            "n_questions": len(questions),
            "questions": [
                {
                    "question_id": q.question_id,
                    "system_prediction": q.system_prediction,
                    "system_confidence": q.system_confidence,
                    "system_reasoning": q.system_reasoning,
                    "key_features": q.key_features_shown,
                    "prompt": (
                        f"Setup #{i+1}: Would you TAKE or SKIP this setup?\n"
                        f"My prediction: {q.system_prediction.upper()} "
                        f"(confidence: {q.system_confidence:.0%})\n"
                        f"My reasoning: {q.system_reasoning}\n"
                        f"Key features: {q.key_features_shown}"
                    )
                }
                for i, q in enumerate(questions)
            ]
        }

    def answer_quiz_question(
        self,
        strategy_id: str,
        question_id: str,
        user_answer: str,
        user_explanation: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process user's answer to a quiz question"""
        validator = self.validators.get(strategy_id)
        if validator is None:
            return {"status": "error", "message": "No active quiz"}

        feedback = validator.submit_quiz_answer(
            question_id, user_answer, user_explanation
        )

        # Check if quiz is complete
        quiz_results = validator.get_quiz_results()
        all_answered = quiz_results.get("total", 0) >= len(validator.quiz_results)

        result = {
            "status": "answer_recorded",
            "feedback": feedback,
            "quiz_complete": all_answered
        }

        if all_answered:
            result["quiz_results"] = quiz_results
            result["next_step"] = (
                "finalize_onboarding" if quiz_results["accuracy"] >= 0.6
                else "add_more_examples"
            )

        return result

    # ═══════════════════════════════════════════════════════
    # STEP 7: FINALIZE
    # ═══════════════════════════════════════════════════════

    def finalize_onboarding(
        self, strategy_id: str
    ) -> Dict[str, Any]:
        """
        Finalize the onboarding process and produce the final SKO.
        
        This is the LAST step. After this, the system is ready
        for live operation.
        """
        logger.info(f"Finalizing onboarding for {strategy_id}")

        discovery = self.pattern_discoveries.get(strategy_id)
        learner = self.pattern_learners.get(strategy_id)
        regime_def = self.regime_definitions.get(strategy_id)
        confidence = self.confidence_decompositions.get(strategy_id)
        report = self.understanding_reports.get(strategy_id)

        if discovery is None or learner is None:
            return {"status": "error", "message": "Learning not complete"}

        # Get quiz results if available
        validator = self.validators.get(strategy_id)
        quiz_results = None
        if validator:
            quiz_results = validator.get_quiz_results()
            if quiz_results.get("status") == "complete":
                # Boost confidence with quiz results
                if confidence and quiz_results.get("accuracy"):
                    confidence_computer = ConfidenceComputer()
                    confidence = confidence_computer.compute(
                        pattern_discovery=discovery,
                        regime_definition=regime_def,
                        quiz_results=validator.quiz_results,
                        n_examples=discovery.n_examples
                    )
                    self.confidence_decompositions[strategy_id] = confidence

        # ── Build Final SKO ──
        sko_builder = OnboardingSKOBuilder()
        sko = sko_builder.build(
            strategy_id=strategy_id,
            intake_results=self.strategies[strategy_id].get("intake_results"),
            pattern_discovery=discovery,
            pattern_learner=learner,
            regime_definition=regime_def,
            confidence_decomposition=confidence,
            understanding_report=report,
            quiz_results=quiz_results
        )
        self.final_skos[strategy_id] = sko

        # ── Build AVT Scoring Config ──
        avt_bridge = OnboardingToAVTBridge()
        avt_config = avt_bridge.build_avt_config(
            onboarding_sko=sko,
            confidence_decomposition=confidence,
            pattern_discovery=discovery,
            regime_definition=regime_def
        )
        self.avt_configs[strategy_id] = avt_config

        self.strategies[strategy_id]["state"] = OnboardingState.COMPLETE
        self._add_history(strategy_id, "onboarding_complete")

        quality = sko.training_metadata.get("sko_quality", {})

        return {
            "status": "onboarding_complete",
            "sko_id": strategy_id,
            "quality_tier": quality.get("tier", "unknown"),
            "quality_score": quality.get("overall_quality", 0),
            "model_accuracy": discovery.model_accuracy,
            "overall_confidence": confidence.overall_confidence if confidence else 0,
            "confidence_breakdown": confidence.to_display() if confidence else "",
            "quiz_accuracy": quiz_results.get("accuracy") if quiz_results else None,
            "avt_initial_mode": self._determine_initial_avt_mode(
                confidence, discovery, quiz_results
            ),
            "summary": self._generate_completion_summary(
                strategy_id, sko, confidence, quiz_results
            ),
            "what_happens_next": {
                "avt_mode": self._determine_initial_avt_mode(
                    confidence, discovery, quiz_results
                ),
                "explanation": self._explain_avt_mode(
                    confidence, discovery, quiz_results
                ),
                "continuous_learning": (
                    "System will keep learning from your accept/reject "
                    "decisions on presented setups. Accuracy will improve "
                    "over time."
                )
            }
        }

    # ═══════════════════════════════════════════════════════
    # STEP 8: GET AVT CONFIG (for downstream systems)
    # ═══════════════════════════════════════════════════════

    def get_avt_scoring_config(
        self, strategy_id: str
    ) -> Optional[AVTScoringConfig]:
        """Get the AVT scoring configuration for a strategy"""
        return self.avt_configs.get(strategy_id)

    def get_final_sko(
        self, strategy_id: str
    ) -> Optional[OnboardingSKO]:
        """Get the final SKO for a strategy"""
        return self.final_skos.get(strategy_id)

    def get_bayesian_priors(
        self, strategy_id: str
    ) -> Dict[str, Dict]:
        """
        Get Bayesian priors for JudgmentLearner initialization.
        
        This is how onboarding connects to the existing
        CalibrationTracker / JudgmentLearner system.
        """
        sko = self.final_skos.get(strategy_id)
        if sko:
            return sko.bayesian_priors
        return {}

    # ═══════════════════════════════════════════════════════
    # RELEARNING (when user adds more examples later)
    # ═══════════════════════════════════════════════════════

    def add_more_examples_and_relearn(
        self,
        strategy_id: str,
        new_examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Add more examples and retrain.
        Used when:
        - System detected gaps
        - User wants to refine understanding
        - Quiz accuracy was low
        """
        self.strategies[strategy_id]["state"] = (
            OnboardingState.COLLECTING_MORE_EXAMPLES
        )

        # Add examples
        batch_result = self.add_examples_batch(strategy_id, new_examples)

        # Rerun learning
        learn_result = self.run_learning(strategy_id)

        return {
            "status": "relearning_complete",
            "examples_added": batch_result["accepted"],
            "new_accuracy": learn_result.get("model_accuracy", 0),
            "new_confidence": learn_result.get("overall_confidence", 0),
            "improvement": self._compute_improvement(strategy_id),
            "next_step": "get_understanding_report"
        }

    # ═══════════════════════════════════════════════════════
    # STATUS & MONITORING
    # ═══════════════════════════════════════════════════════

    def get_onboarding_status(
        self, strategy_id: str
    ) -> Dict[str, Any]:
        """Get complete status of onboarding process"""
        if strategy_id not in self.strategies:
            return {"status": "not_found"}

        state = self.strategies[strategy_id]
        collection = self.example_collector.get_collection_status(strategy_id)
        discovery = self.pattern_discoveries.get(strategy_id)
        confidence = self.confidence_decompositions.get(strategy_id)

        return {
            "strategy_id": strategy_id,
            "state": state["state"].value,
            "created_at": state.get("created_at"),
            "collection": collection,
            "learning": {
                "model_accuracy": discovery.model_accuracy if discovery else None,
                "n_features": len(discovery.feature_discriminators) if discovery else 0,
                "confidence": confidence.overall_confidence if confidence else 0
            },
            "history": state.get("history", [])[-10:],  # Last 10 events
            "next_recommended_action": self._recommend_next_action(
                strategy_id
            )
        }

    # ═══════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════

    def _identify_example_needs(
        self, description: str, intake_results: Optional[Dict]
    ) -> List[str]:
        """
        Identify what aspects of the strategy need examples.
        
        Rules can be extracted from text.
        Judgment, context, preferences NEED examples.
        """
        needs = []

        if intake_results:
            components = intake_results.get("components", [])
            for comp in components:
                comp_type = comp.get("type", "")
                comp_desc = comp.get("description", "")

                if comp_type == "score":
                    needs.append(
                        f"What '{comp_desc}' looks like in practice "
                        f"(show examples of good and bad)"
                    )
                elif comp_type == "judgment":
                    needs.append(
                        f"Your subjective judgment on '{comp_desc}' "
                        f"(show what you consider good vs bad)"
                    )
                elif comp_type == "uncapturable":
                    needs.append(
                        f"Examples of '{comp_desc}' so I can try to "
                        f"find measurable proxies"
                    )

            # Check for gaps identified by IntakeAssessor
            capture_gaps = intake_results.get("capture_gaps", [])
            for gap in capture_gaps:
                needs.append(f"Examples showing: {gap}")
        else:
            # Without intake results, identify from description
            judgment_keywords = [
                "clean", "ugly", "feel", "looks", "sense",
                "quality", "smooth", "good", "nice", "perfect"
            ]
            desc_lower = description.lower()
            for keyword in judgment_keywords:
                if keyword in desc_lower:
                    needs.append(
                        f"What '{keyword}' means to you specifically "
                        f"(show examples)"
                    )

        # Always need these
        if not needs:
            needs = [
                "What a GOOD setup looks like to you",
                "What makes you SKIP a setup",
                "Different market conditions where you trade"
            ]

        return needs

    def _extract_regime_mentions(self, description: str) -> List[str]:
        """Extract regime-related terms from description"""
        regime_terms = [
            "bearish", "bullish", "uptrend", "downtrend",
            "sideways", "ranging", "trending", "volatile",
            "low volatility", "choppy market", "strong trend",
            "weak trend", "consolidating"
        ]

        found = []
        desc_lower = description.lower()
        for term in regime_terms:
            if term in desc_lower:
                found.append(term)

        return found

    def _generate_example_request_message(
        self,
        needs_examples: List[str],
        intake_results: Optional[Dict]
    ) -> str:
        """Generate the message asking user for examples"""
        lines = []
        lines.append("I've understood the core of your strategy.")
        lines.append("")

        if intake_results:
            rules = [
                c for c in intake_results.get("components", [])
                if c.get("type") == "rule"
            ]
            if rules:
                lines.append("✅ Hard rules I extracted:")
                for rule in rules:
                    lines.append(f"   • {rule.get('description', '')}")
                lines.append("")

        lines.append("To truly understand your strategy, I need to SEE it.")
        lines.append("Please show me 20-25 chart examples:")
        lines.append("")
        lines.append("For each example, provide:")
        lines.append("  📊 Symbol + Timeframe + Date")
        lines.append("  🏷️ Label: TAKE (would trade) or SKIP (would avoid)")
        lines.append("  💬 Explanation: WHY this label")
        lines.append("  📸 Screenshot (optional but helpful)")
        lines.append("")
        lines.append("I specifically need examples showing:")
        for need in needs_examples[:5]:
            lines.append(f"  ❓ {need}")
        lines.append("")
        lines.append("Mix of:")
        lines.append("  • 10-15 setups you WOULD take")
        lines.append("  • 8-12 setups you WOULD skip/avoid")
        lines.append("  • 2-3 borderline/edge cases")
        lines.append("  • Different symbols and trading days")

        return "\n".join(lines)

    def _determine_initial_avt_mode(
        self,
        confidence: Optional['ConfidenceDecomposition'],
        discovery: Optional['PatternDiscovery'],
        quiz_results: Optional[Dict]
    ) -> str:
        """
        Determine what mode AVT should start in.
        
        Based on confidence level from onboarding.
        
        HIGH confidence (>80%) → candidate_presenter with auto-filter
        MEDIUM confidence (60-80%) → candidate_presenter
        LOW confidence (<60%) → observer
        """
        if confidence is None:
            return "observer"

        overall = confidence.overall_confidence
        quiz_acc = quiz_results.get("accuracy", 0) if quiz_results else 0

        # Combined score
        if quiz_acc > 0:
            effective_confidence = overall * 0.6 + quiz_acc * 0.4
        else:
            effective_confidence = overall

        if effective_confidence >= 0.80:
            return "candidate_presenter_with_filter"
        elif effective_confidence >= 0.65:
            return "candidate_presenter"
        elif effective_confidence >= 0.50:
            return "cautious_presenter"
        else:
            return "observer"

    def _explain_avt_mode(
        self,
        confidence: Optional['ConfidenceDecomposition'],
        discovery: Optional['PatternDiscovery'],
        quiz_results: Optional[Dict]
    ) -> str:
        """Explain to user why AVT is in this mode"""
        mode = self._determine_initial_avt_mode(
            confidence, discovery, quiz_results
        )

        explanations = {
            "candidate_presenter_with_filter": (
                "HIGH understanding confidence. I'll actively scan for "
                "setups matching your strategy, pre-filter out obvious "
                "mismatches, and present candidates for your confirmation. "
                "I'll also flag when I'm uncertain."
            ),
            "candidate_presenter": (
                "GOOD understanding confidence. I'll present setups that "
                "might match your strategy, but I'll show more candidates "
                "since I'm not fully confident in filtering yet. "
                "Your feedback will help me improve."
            ),
            "cautious_presenter": (
                "MODERATE understanding. I'll show you setups but with "
                "less confidence in my filtering. Expect some false "
                "positives and negatives. Your feedback is critical "
                "at this stage."
            ),
            "observer": (
                "LOW understanding confidence. I'll watch your trading "
                "and learn from your decisions. I won't actively suggest "
                "setups until I understand your strategy better. "
                "Consider providing more examples."
            )
        }

        return explanations.get(mode, "Mode not recognized")

    def _generate_completion_summary(
        self,
        strategy_id: str,
        sko: 'OnboardingSKO',
        confidence: Optional['ConfidenceDecomposition'],
        quiz_results: Optional[Dict]
    ) -> str:
        """Generate final onboarding completion summary"""
        lines = []
        lines.append("=" * 60)
        lines.append("🎯 ONBOARDING COMPLETE")
        lines.append("=" * 60)
        lines.append("")

        # Stats
        meta = sko.training_metadata
        lines.append(f"📊 Examples analyzed: {meta.get('n_examples', 0)}")
        lines.append(f"   TAKE examples: {meta.get('n_positive', 0)}")
        lines.append(f"   SKIP examples: {meta.get('n_negative', 0)}")
        lines.append("")

        # Model quality
        lines.append(f"🤖 Model accuracy: {meta.get('model_accuracy', 0):.0%} "
                     f"± {meta.get('model_accuracy_std', 0):.0%}")
        if quiz_results and quiz_results.get("accuracy"):
            lines.append(f"🎯 Quiz accuracy: {quiz_results['accuracy']:.0%}")
        lines.append("")

        # Confidence
        if confidence:
            lines.append(confidence.to_display())
            lines.append("")

        # What system learned
        lines.append("📝 Key learnings:")
        n_features = len([
            f for f, d in (sko.pattern_discovery.feature_discriminators.items()
                          if sko.pattern_discovery else [])
            if d.strength != FeatureDiscriminatorStrength.NONE
        ])
        lines.append(f"   • {n_features} discriminating features identified")

        if sko.pattern_discovery and sko.pattern_discovery.hidden_features:
            lines.append(
                f"   • {len(sko.pattern_discovery.hidden_features)} hidden "
                f"patterns discovered (you may not consciously notice these)"
            )

        n_context = len(sko.context_factors.get("dependencies", []))
        if n_context:
            lines.append(f"   • {n_context} context dependencies found")

        lines.append("")

        # Quality
        quality = meta.get("sko_quality", {})
        tier = quality.get("tier", "unknown")
        tier_emoji = {
            "production_ready": "🟢",
            "usable_with_caution": "🟡",
            "needs_improvement": "🟠",
            "insufficient": "🔴"
        }
        lines.append(f"{tier_emoji.get(tier, '⚪')} Quality tier: {tier}")
        lines.append("")

        # What happens next
        avt_mode = self._determine_initial_avt_mode(
            confidence, sko.pattern_discovery, quiz_results
        )
        lines.append(f"🚀 AVT starting mode: {avt_mode}")
        lines.append("")
        lines.append("The system will continue learning from your")
        lines.append("accept/reject decisions. Understanding will")
        lines.append("improve with each interaction.")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _compute_improvement(self, strategy_id: str) -> Dict[str, Any]:
        """Compute how much learning improved after adding more examples"""
        history = self.strategies.get(strategy_id, {}).get("history", [])

        # Find previous and current learning events
        learning_events = [
            h for h in history if h["event"] == "learning_complete"
        ]

        if len(learning_events) < 2:
            return {"improved": False, "message": "First learning round"}

        prev = learning_events[-2].get("data", {})
        curr = learning_events[-1].get("data", {})

        prev_acc = prev.get("model_accuracy", 0)
        curr_acc = curr.get("model_accuracy", 0)
        prev_conf = prev.get("confidence", 0)
        curr_conf = curr.get("confidence", 0)

        return {
            "improved": curr_acc > prev_acc,
            "accuracy_change": curr_acc - prev_acc,
            "confidence_change": curr_conf - prev_conf,
            "previous_accuracy": prev_acc,
            "current_accuracy": curr_acc,
            "message": (
                f"Accuracy: {prev_acc:.0%} → {curr_acc:.0%} "
                f"({'↑' if curr_acc > prev_acc else '↓'} "
                f"{abs(curr_acc - prev_acc):.0%})"
            )
        }

    def _recommend_next_action(self, strategy_id: str) -> str:
        """Recommend what user should do next"""
        state = self.strategies.get(strategy_id, {}).get(
            "state", OnboardingState.NOT_STARTED
        )

        recommendations = {
            OnboardingState.NOT_STARTED:
                "Start by describing your strategy",
            OnboardingState.NARRATIVE_RECEIVED:
                "Begin adding labeled examples (aim for 20-25)",
            OnboardingState.COLLECTING_EXAMPLES:
                "Continue adding examples",
            OnboardingState.READY_FOR_LEARNING:
                "Run the learning process",
            OnboardingState.LEARNING_COMPLETE:
                "Review the understanding report",
            OnboardingState.VALIDATING:
                "Review report and provide corrections if needed",
            OnboardingState.QUIZ_IN_PROGRESS:
                "Complete the quiz questions",
            OnboardingState.AWAITING_CORRECTIONS:
                "Provide corrections or additional examples",
            OnboardingState.COLLECTING_MORE_EXAMPLES:
                "Add more examples for the identified gaps",
            OnboardingState.COMPLETE:
                "Onboarding complete! System is learning from live interactions.",
            OnboardingState.FAILED:
                "Something went wrong. Try adding more examples."
        }

        return recommendations.get(state, "Check onboarding status")

    def _add_history(
        self, strategy_id: str, event: str, data: Dict = None
    ):
        """Add event to strategy history"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id].setdefault("history", []).append({
                "event": event,
                "timestamp": datetime.now().isoformat(),
                "data": data or {}
            })
