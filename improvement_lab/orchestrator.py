# improvement_lab/orchestrator.py
"""
Improvement Lab Orchestrator
=============================

Enforces the correct ORDER of analysis:
  1. REGIME CHECK — is the failure concentrated in a regime?
  2. VARIANT CHECK — is it concentrated in a variant?
  3. BEHAVIORAL CHECK — is it trader behavior or market?
  4. PATTERN CHECK — rule-based + ML pattern detection

Each layer REMOVES explained trades before subsequent layers.

Pipeline: DETECTED → SHADOW_TESTING → SHADOW_PASSED →
  INSIGHT_PRESENTED → SUGGESTING → ACCEPTANCE_REVIEW →
  GRADUATED → MONITORING → (DEGRADING → RETIRED)

Two-speed learning:
  Fast layer updates on every trade.
  Deep layer retrains every 20 trades.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

import numpy as np
from scipy import stats as scipy_stats

from improvement_lab.behavioral_separator import BehavioralMarketSeparator
from improvement_lab.entry_filter_learner import EntryFilterLearner
from improvement_lab.early_exit_detector import EarlyExitDetector
from improvement_lab.feature_extractor import FeatureExtractor
from improvement_lab.models import (
    AnalysisLayer,
    BehavioralClassification,
    ExperimentStatus,
    ImprovementHypothesis,
    ImprovementLabReport,
    ImprovementType,
    PatternClassification,
    ValidationProgress,
)
from improvement_lab.rr_killer_detector import RRKillerDetector
from improvement_lab.shadow_validator import ShadowValidator
from improvement_lab.tradeoff_simulator import CompositionTester, TradeoffSimulator
from improvement_lab.two_speed_learner import TwoSpeedLearner
from improvement_lab.variant_analyzer import VariantAnalyzer

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────

# Reanalysis triggers (whichever comes first)
REANALYSIS_TRADE_INTERVAL = 10
REANALYSIS_TIME_INTERVAL = timedelta(days=1)

# Minimum thresholds
MIN_TRADES_RULE = 30
MIN_TRADES_ML = 100
MIN_REGIME_DROP = 0.15          # 15pp below overall win rate
MIN_REGIME_SAMPLE = 5           # minimum trades in a regime to flag
MIN_VARIANT_DROP = 0.10         # 10pp below overall
MIN_VARIANT_SAMPLE = 5
MIN_PATTERN_TRADES = 10         # minimum remaining trades for pattern layer
QUALITY_GATE_MIN_WEIGHT = 0.5   # from quality_gate.py spec
MAX_ACTIVE_EXPERIMENTS = 2      # per strategy
MIN_NET_R_FOR_SHADOW = 0.10     # minimum net R to enter shadow testing


class ImprovementLabOrchestrator:
    """
    Central analysis pipeline.

    Analysis order: Regime → Variant → Behavioral → Pattern
    Each layer removes explained trades from subsequent layers.
    Learning: Two-speed (fast per trade, deep periodic)
    Validation: Shadow A/B testing with full lifecycle
    """

    def __init__(self, db_session: Any = None, quality_gate: Any = None, db_factory: Any = None):
        self._db = db_session
        self._db_factory = db_factory  # callable → fresh session (for model persistence)
        self._quality_gate = quality_gate

        # Analyzers
        self._extractor = FeatureExtractor()
        self._rr_detector = RRKillerDetector()
        self._exit_detector = EarlyExitDetector()
        self._filter_learner = EntryFilterLearner()
        self._variant_analyzer = VariantAnalyzer(db_session)
        self._behavioral_separator = BehavioralMarketSeparator(db_session)
        self._tradeoff_sim = TradeoffSimulator()
        self._composition_tester = CompositionTester()

        # Shadow validator
        self._shadow = ShadowValidator()

        # Two-speed learners per strategy
        self._learners: Dict[str, Dict[str, TwoSpeedLearner]] = {}

        # State tracking
        self._trade_counts: Dict[str, int] = {}
        self._last_analysis_at: Dict[str, int] = {}
        self._last_analysis_time: Dict[str, datetime] = {}
        self._last_reports: Dict[str, ImprovementLabReport] = {}

    # ── Event wiring ──────────────────────────────────────────────

    def wire_events(self, event_bus: Any) -> None:
        """Subscribe to TRADE_CLOSED events."""
        try:
            from core.events import EventType
            event_bus.subscribe(EventType.TRADE_CLOSED, self._on_trade_closed)
            logger.info("ImprovementLab wired to TRADE_CLOSED events")
        except ImportError:
            logger.warning("Could not wire events — EventType not available")

    def _on_trade_closed(self, event: Any) -> None:
        """
        Handle a closed trade.
        1. Fast layer update (milliseconds)
        2. Feed active shadow experiments
        3. Check if deep retrain needed
        4. Check if full reanalysis needed (trade count OR time)
        """
        trade = event.data.get("trade")
        strategy_id = str(
            event.strategy_id or event.data.get("strategy_id", "unknown")
        )
        if not trade:
            return

        # 1. Update fast learners
        self._update_fast_learners(trade, strategy_id)

        # 2. Feed shadow experiments
        self._shadow.process_trade(trade, strategy_id)

        # 3. Check deep retrain
        self._check_deep_retrain(strategy_id)

        # 4. Check reanalysis
        count = self._trade_counts.get(strategy_id, 0) + 1
        self._trade_counts[strategy_id] = count
        self._trigger_reanalysis_if_needed(strategy_id, count)

    def _trigger_reanalysis_if_needed(
        self, strategy_id: str, current_count: int
    ) -> None:
        """Trigger reanalysis based on trade count OR time elapsed."""
        last_count = self._last_analysis_at.get(strategy_id, 0)
        last_time = self._last_analysis_time.get(strategy_id)
        now = datetime.now(timezone.utc)

        trade_trigger = (current_count - last_count) >= REANALYSIS_TRADE_INTERVAL
        time_trigger = (
            last_time is None
            or (now - last_time) >= REANALYSIS_TIME_INTERVAL
        )

        if trade_trigger or time_trigger:
            # Query trades from DB instead of carrying them in events
            trades = self._fetch_strategy_trades(strategy_id)
            if trades and len(trades) >= MIN_TRADES_RULE:
                self.analyze(trades, strategy_id)
                self._last_analysis_at[strategy_id] = current_count
                self._last_analysis_time[strategy_id] = now

    def _fetch_strategy_trades(self, strategy_id: str) -> List[Any]:
        """
        Fetch recent trades for a strategy from DB.
        Override this in subclasses or inject a repository.
        """
        if self._db is None:
            logger.warning(
                "No DB session — cannot fetch trades for %s", strategy_id
            )
            return []
        # Placeholder: implement with your trade repository
        try:
            from core.repositories import TradeRepository
            repo = TradeRepository(self._db)
            return repo.get_recent_trades(
                strategy_id=strategy_id, limit=500
            )
        except ImportError:
            return []

    # ── Two-speed learning ────────────────────────────────────────

    def _get_learner(
        self, strategy_id: str, detector_type: str
    ) -> TwoSpeedLearner:
        """Get or create a two-speed learner for a strategy+detector."""
        if strategy_id not in self._learners:
            self._learners[strategy_id] = {}
        if detector_type not in self._learners[strategy_id]:
            learner = TwoSpeedLearner(
                strategy_id=strategy_id,
                detector_type=detector_type,
            )
            # Restore persisted models on first access (avoids cold-start retrain)
            self._load_learner_models(learner)
            # Wire persistence callback so models are saved after every deep retrain
            if self._db_factory is not None:
                learner._persist_callback = self._make_persist_callback()
            self._learners[strategy_id][detector_type] = learner
        return self._learners[strategy_id][detector_type]

    def _make_persist_callback(self):
        """Return a callback that saves model blobs to avt_learner_state via db_factory."""
        db_factory = self._db_factory

        def _callback(strategy_id: str, detector_type: str, model_blob: bytes) -> None:
            try:
                db = db_factory()
                if hasattr(db, "__next__"):
                    db = next(db)
                try:
                    from avt.repository import AVTLearnerStateRepository
                    AVTLearnerStateRepository(db).save(
                        strategy_id=strategy_id,
                        detector_type=detector_type,
                        bayesian_state=None,  # Bayesian state saved separately via get_state()
                        model_blob=model_blob,
                        trade_count=0,
                        model_score=0.0,
                    )
                    db.commit()
                finally:
                    db.close()
            except Exception as e:
                logger.warning("Model persistence failed for %s/%s: %s", strategy_id, detector_type, e)

        return _callback

    def _load_learner_models(self, learner: TwoSpeedLearner) -> None:
        """On startup, restore deep models from DB if available."""
        if self._db_factory is None:
            return
        try:
            db = self._db_factory()
            if hasattr(db, "__next__"):
                db = next(db)
            try:
                from avt.repository import AVTLearnerStateRepository
                state = AVTLearnerStateRepository(db).load(
                    learner.strategy_id, learner.detector_type
                )
                if state and state.get("model_blob"):
                    loaded = learner.load_deep_models(state["model_blob"])
                    if loaded:
                        logger.info(
                            "Restored deep models for %s/%s from DB",
                            learner.strategy_id, learner.detector_type,
                        )
            finally:
                db.close()
        except Exception as e:
            logger.debug("Model load skipped for %s/%s: %s", learner.strategy_id, learner.detector_type, e)

    def _update_fast_learners(self, trade: Any, strategy_id: str) -> None:
        """Update all fast learners with a new closed trade."""
        ctx_features = self._extractor.extract_trade_context(trade)

        r = _safe_float(_get_field(trade, "r_multiple"), 0.0)
        pnl = _safe_float(_get_field(trade, "pnl"), 0.0)

        # Entry filter learner
        learner = self._get_learner(strategy_id, "entry_filter")
        learner.update_fast(ctx_features, pnl > 0)

        # RR killer learner
        max_r = _safe_float(_get_field(trade, "max_favorable_r"), None)
        if max_r is not None and max_r >= 1.5:
            rr_features = dict(ctx_features)
            rr_features["trigger_r"] = max_r
            reversed_significantly = (max_r - r) > 0.5 * max_r
            learner = self._get_learner(strategy_id, "rr_killer")
            learner.update_fast(rr_features, reversed_significantly)

        # Early exit learner
        hit_sl = r <= -0.9 if r is not None else False
        learner = self._get_learner(strategy_id, "early_exit")
        learner.update_fast(ctx_features, hit_sl)

    def _check_deep_retrain(self, strategy_id: str) -> None:
        """Check if any learner needs deep retraining."""
        for detector_type, learner in self._learners.get(
            strategy_id, {}
        ).items():
            if learner.should_retrain_deep():
                report = learner.retrain_deep()
                if report.get("status") == "trained":
                    logger.info(
                        "Deep retrain for %s/%s: GBM=%.3f RF=%.3f",
                        strategy_id,
                        detector_type,
                        report.get("gbm_score", 0),
                        report.get("rf_score", 0),
                    )

    # ── Full analysis pipeline (correct order) ────────────────────

    def analyze(
        self,
        trades: List[Any],
        strategy_id: str,
        min_trades: int = MIN_TRADES_RULE,
    ) -> ImprovementLabReport:
        """
        Run the full analysis pipeline in strict order:
          1. Quality gate filter
          2. Regime segmentation
          3. Variant concentration
          4. Behavioral separation
          5. Pattern detection

        Each layer removes explained trades from subsequent layers.
        This prevents complex explanations for simple problems.
        """
        # ── STEP 0: Quality gate filter ──
        # CRITICAL: Only learn from disciplined trades
        filtered_trades = self._apply_quality_gate(trades)

        if len(filtered_trades) < min_trades:
            logger.info(
                "Strategy %s: only %d quality trades (need %d). "
                "Skipping analysis.",
                strategy_id,
                len(filtered_trades),
                min_trades,
            )
            return ImprovementLabReport(
                strategy_id=strategy_id,
                total_trades=len(trades),
                quality_filtered_trades=len(filtered_trades),
            )

        baseline_wr = _compute_win_rate(filtered_trades)
        baseline_exp = _compute_expectancy(filtered_trades)

        # Build trade ID index for O(1) removal
        trade_id_map: Dict[str, Any] = {}
        for i, t in enumerate(filtered_trades):
            tid = str(_get_field(t, "id", f"_idx_{i}"))
            trade_id_map[tid] = t

        remaining_ids: Set[str] = set(trade_id_map.keys())
        hypotheses: List[ImprovementHypothesis] = []

        # ── LAYER 1: Regime Check ──
        regime_hypotheses = self._regime_analysis(
            list(trade_id_map.values()),
            strategy_id,
            baseline_wr,
        )
        for h in regime_hypotheses:
            hypotheses.append(h)
            remaining_ids -= set(h.explained_trade_ids)
        remaining_trades = [
            trade_id_map[tid] for tid in remaining_ids
            if tid in trade_id_map
        ]

        # ── LAYER 2: Variant Check ──
        variant_hypotheses = self._variant_analysis(
            remaining_trades,
            strategy_id,
            baseline_wr,
        )
        for h in variant_hypotheses:
            hypotheses.append(h)
            remaining_ids -= set(h.explained_trade_ids)
        remaining_trades = [
            trade_id_map[tid] for tid in remaining_ids
            if tid in trade_id_map
        ]

        # ── LAYER 3: Behavioral Check ──
        # Runs as independent analysis on remaining trades.
        # Produces coaching recommendations (not strategy mods).
        behavioral_hypotheses = self._behavioral_analysis(
            remaining_trades,
            strategy_id,
        )
        for h in behavioral_hypotheses:
            hypotheses.append(h)
            # Behavioral patterns also remove explained trades
            remaining_ids -= set(h.explained_trade_ids)
        remaining_trades = [
            trade_id_map[tid] for tid in remaining_ids
            if tid in trade_id_map
        ]

        # ── LAYER 4: Pattern Detection ──
        if len(remaining_trades) >= MIN_PATTERN_TRADES:
            pattern_hypotheses = self._pattern_analysis(
                remaining_trades,
                strategy_id,
                baseline_wr,
                filtered_trades,
            )
            for h in pattern_hypotheses:
                hypotheses.append(h)

        # ── Tradeoff analysis for each hypothesis ──
        for h in hypotheses:
            if h.improvement_type == "coaching_recommendation":
                continue
            h.tradeoff = self._tradeoff_sim.simulate(h, filtered_trades)
            if h.tradeoff and h.tradeoff.net_expectancy_change_r <= 0:
                h.tradeoff.recommendation = (
                    "NOT RECOMMENDED: " + (h.tradeoff.recommendation or "")
                )

        # ── Composition testing ──
        positive_hypotheses = [
            h for h in hypotheses
            if h.tradeoff
            and h.tradeoff.net_expectancy_change_r > 0
            and h.improvement_type != "coaching_recommendation"
        ]
        composition = None
        if len(positive_hypotheses) >= 2:
            composition = self._composition_tester.test_composition(
                positive_hypotheses, filtered_trades
            )

        # ── Create shadow experiments (with limits) ──
        self._create_shadow_experiments(
            positive_hypotheses, strategy_id
        )

        # ── Build report ──
        progress = self._shadow.get_all_progress(strategy_id)
        explained_count = len(trade_id_map) - len(remaining_ids)

        report = ImprovementLabReport(
            strategy_id=strategy_id,
            total_trades=len(trades),
            quality_filtered_trades=len(filtered_trades),
            analysis_mode=(
                "hybrid" if len(filtered_trades) >= MIN_TRADES_ML
                else "rule_based"
            ),
            baseline_win_rate=round(baseline_wr, 4),
            baseline_expectancy=round(baseline_exp, 4),
            regime_findings=[
                h for h in hypotheses
                if h.analysis_layer == AnalysisLayer.REGIME.value
            ],
            variant_findings=[
                h for h in hypotheses
                if h.analysis_layer == AnalysisLayer.VARIANT.value
            ],
            behavioral_findings=[
                h for h in hypotheses
                if h.analysis_layer == AnalysisLayer.BEHAVIORAL.value
            ],
            pattern_findings=[
                h for h in hypotheses
                if h.analysis_layer == AnalysisLayer.PATTERN.value
            ],
            candidates=hypotheses,
            composition=composition,
            total_explained_trades=explained_count,
            total_unexplained_trades=len(remaining_ids),
            total_estimated_impact_r=round(
                sum(
                    h.tradeoff.net_expectancy_change_r
                    for h in hypotheses
                    if h.tradeoff
                ),
                4,
            ),
            validation_progress=progress,
        )

        self._last_reports[strategy_id] = report

        logger.info(
            "ImprovementLab[%s]: %d total trades, %d quality-filtered, "
            "%d hypotheses (regime=%d, variant=%d, behavioral=%d, "
            "pattern=%d), %d explained, %d unexplained",
            strategy_id,
            len(trades),
            len(filtered_trades),
            len(hypotheses),
            len(report.regime_findings),
            len(report.variant_findings),
            len(report.behavioral_findings),
            len(report.pattern_findings),
            explained_count,
            len(remaining_ids),
        )

        return report

    # ── AVT-specific analysis (called by self_improver) ───────────

    def analyze_avt_trades(
        self,
        trades: List[Any],
        strategy_id: str,
        strategy_version: str = "",
    ) -> ImprovementLabReport:
        """
        Run analysis on AVT's own clean virtual trades.

        Differences from analyze():
          - NO quality gate (AVT trades are perfectly disciplined)
          - NO behavioral layer (AVT has no behavioral noise)
          - Source set to AVT
          - Version tracked for proof lifecycle

        Called by AVTSelfImprover.daily_analysis().
        """
        from improvement_lab.models import HypothesisSource

        if len(trades) < MIN_TRADES_RULE:
            return ImprovementLabReport(
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                source=HypothesisSource.AVT.value,
                total_trades=len(trades),
                sufficient_data=False,
            )

        baseline_wr = _compute_win_rate(trades)
        baseline_exp = _compute_expectancy(trades)

        trade_id_map: Dict[str, Any] = {}
        for i, t in enumerate(trades):
            tid = str(_get_field(t, "id", f"_idx_{i}"))
            trade_id_map[tid] = t

        remaining_ids: Set[str] = set(trade_id_map.keys())
        hypotheses: List[ImprovementHypothesis] = []

        # LAYER 1: Regime Check
        regime_hypotheses = self._regime_analysis(
            list(trade_id_map.values()), strategy_id, baseline_wr,
        )
        for h in regime_hypotheses:
            h.source = HypothesisSource.AVT.value
            hypotheses.append(h)
            remaining_ids -= set(h.explained_trade_ids)
        remaining_trades = [
            trade_id_map[tid] for tid in remaining_ids
            if tid in trade_id_map
        ]

        # LAYER 2: Variant Check
        variant_hypotheses = self._variant_analysis(
            remaining_trades, strategy_id, baseline_wr,
        )
        for h in variant_hypotheses:
            h.source = HypothesisSource.AVT.value
            hypotheses.append(h)
            remaining_ids -= set(h.explained_trade_ids)
        remaining_trades = [
            trade_id_map[tid] for tid in remaining_ids
            if tid in trade_id_map
        ]

        # SKIP LAYER 3 (Behavioral) — AVT has no behavioral noise

        # LAYER 4: Pattern Detection
        if len(remaining_trades) >= MIN_PATTERN_TRADES:
            pattern_hypotheses = self._pattern_analysis(
                remaining_trades, strategy_id, baseline_wr, trades,
            )
            for h in pattern_hypotheses:
                h.source = HypothesisSource.AVT.value
                hypotheses.append(h)

        # Tradeoff analysis
        for h in hypotheses:
            h.tradeoff = self._tradeoff_sim.simulate(h, trades)

        # Filter actionable: net positive tradeoff
        actionable = [
            h for h in hypotheses
            if h.tradeoff and h.tradeoff.net_expectancy_change_r > 0
        ]
        rejected = [
            h for h in hypotheses
            if h.tradeoff and h.tradeoff.net_expectancy_change_r <= 0
        ]

        explained_count = len(trade_id_map) - len(remaining_ids)

        report = ImprovementLabReport(
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            source=HypothesisSource.AVT.value,
            total_trades=len(trades),
            sufficient_data=True,
            analysis_mode=(
                "hybrid" if len(trades) >= MIN_TRADES_ML
                else "rule_based"
            ),
            baseline_win_rate=round(baseline_wr, 4),
            baseline_expectancy=round(baseline_exp, 4),
            regime_findings=[
                h for h in hypotheses
                if h.analysis_layer == AnalysisLayer.REGIME.value
            ],
            variant_findings=[
                h for h in hypotheses
                if h.analysis_layer == AnalysisLayer.VARIANT.value
            ],
            behavioral_findings=[],  # empty — AVT has no behavioral noise
            pattern_findings=[
                h for h in hypotheses
                if h.analysis_layer == AnalysisLayer.PATTERN.value
            ],
            actionable_hypotheses=actionable,
            rejected_hypotheses=rejected,
            total_explained_trades=explained_count,
            total_unexplained_trades=len(remaining_ids),
            total_estimated_impact_r=round(
                sum(
                    h.tradeoff.net_expectancy_change_r
                    for h in actionable if h.tradeoff
                ), 4,
            ),
        )

        logger.info(
            "ImprovementLab[AVT/%s v%s]: %d trades, "
            "%d hypotheses (%d actionable, %d rejected)",
            strategy_id, strategy_version, len(trades),
            len(hypotheses), len(actionable), len(rejected),
        )

        return report

    # ── Quality gate ──────────────────────────────────────────────

    def _apply_quality_gate(self, trades: List[Any]) -> List[Any]:
        """
        Filter trades through quality gate.
        Only learn improvement patterns from disciplined trades.

        Trades with learning_weight < 0.5 are excluded.
        This removes revenge trades, FOMO entries, tilt trades.
        """
        if self._quality_gate is not None:
            try:
                return [
                    t for t in trades
                    if self._quality_gate.compute_learning_weight(t)
                    >= QUALITY_GATE_MIN_WEIGHT
                ]
            except Exception as e:
                logger.warning(
                    "Quality gate failed, falling back to direct filter: %s",
                    e,
                )

        # Fallback: use learning_weight field if present
        filtered = []
        for t in trades:
            weight = _safe_float(_get_field(t, "learning_weight"), 1.0)
            if weight >= QUALITY_GATE_MIN_WEIGHT:
                filtered.append(t)
        return filtered

    # ── Analysis layers ───────────────────────────────────────────

    def _regime_analysis(
        self,
        trades: List[Any],
        strategy_id: str,
        baseline_wr: float,
    ) -> List[ImprovementHypothesis]:
        """
        Layer 1: Check if failures are concentrated in specific regimes.
        Pure pandas groupby logic. No ML.

        If any regime has win rate 15+ points below average
        with 5+ trades → flag as regime filter opportunity.
        """
        hypotheses = []

        regime_groups: Dict[str, List[Any]] = {}
        for trade in trades:
            regime = _get_field(trade, "market_regime", "unknown")
            if hasattr(regime, "value"):
                regime = regime.value
            regime = str(regime)
            regime_groups.setdefault(regime, []).append(trade)

        for regime, regime_trades in regime_groups.items():
            if regime == "unknown" or len(regime_trades) < MIN_REGIME_SAMPLE:
                continue

            wr = _compute_win_rate(regime_trades)
            drop = baseline_wr - wr

            if drop < MIN_REGIME_DROP:
                continue

            trade_ids = [
                str(_get_field(t, "id", "")) for t in regime_trades
            ]
            # Remove empty IDs
            trade_ids = [tid for tid in trade_ids if tid]

            exp = _compute_expectancy(regime_trades)

            # Statistical significance: proportions z-test
            n_regime = len(regime_trades)
            wins_regime = sum(
                1 for t in regime_trades
                if _safe_float(_get_field(t, "pnl", 0)) > 0
            )
            wins_total = sum(
                1 for t in trades
                if _safe_float(_get_field(t, "pnl", 0)) > 0
            )
            n_total = len(trades)

            # Two-proportion z-test
            p_value = _proportions_z_test(
                wins_regime, n_regime, wins_total, n_total
            )

            # Confidence based on statistical test + sample size
            confidence = _compute_regime_confidence(
                drop, n_regime, p_value
            )

            hypotheses.append(
                ImprovementHypothesis(
                    strategy_id=strategy_id,
                    improvement_type=ImprovementType.REGIME_FILTER.value,
                    analysis_layer=AnalysisLayer.REGIME.value,
                    pattern_type="rule_based",
                    title=(
                        f"Skip '{regime}' regime — "
                        f"WR {wr:.0%} vs baseline {baseline_wr:.0%}"
                    ),
                    description=(
                        f"Your strategy has {wr:.0%} win rate in "
                        f"'{regime}' regime vs {baseline_wr:.0%} overall. "
                        f"{len(regime_trades)} trades affected. "
                        f"Expectancy in this regime: {exp:.2f}R. "
                        f"p-value: {p_value:.3f}."
                    ),
                    expected_impact_r=round(
                        abs(exp) * 0.5, 3
                    ) if exp < 0 else 0.0,
                    confidence=round(confidence, 3),
                    evidence={
                        "regime": regime,
                        "regime_win_rate": round(wr, 4),
                        "baseline_win_rate": round(baseline_wr, 4),
                        "win_rate_drop": round(drop, 4),
                        "regime_expectancy_r": round(exp, 4),
                        "sample_size": n_regime,
                        "p_value": round(p_value, 4),
                    },
                    suggested_action=(
                        f"Add entry filter: skip trades when regime is "
                        f"'{regime}'. Would have avoided "
                        f"{len(regime_trades)} trades."
                    ),
                    proposed_rule={
                        "type": "regime_filter",
                        "action": "skip_entry",
                        "condition": {"market_regime": {"not_in": [regime]}},
                    },
                    explained_trade_ids=trade_ids,
                )
            )

        return hypotheses

    def _variant_analysis(
        self,
        trades: List[Any],
        strategy_id: str,
        baseline_wr: float,
    ) -> List[ImprovementHypothesis]:
        """
        Layer 2: Check if failures concentrate in specific variants.
        Cross-references with SKE variant engine anti-variants.
        """
        hypotheses = []

        variant_groups = self._variant_analyzer.group_trades_by_variant(
            trades, strategy_id
        )

        for variant_id, variant_trades in variant_groups.items():
            if variant_id is None or len(variant_trades) < MIN_VARIANT_SAMPLE:
                continue

            wr = _compute_win_rate(variant_trades)
            if wr >= baseline_wr - MIN_VARIANT_DROP:
                continue

            trade_ids = [
                str(_get_field(t, "id", ""))
                for t in variant_trades
            ]
            trade_ids = [tid for tid in trade_ids if tid]

            conc = self._variant_analyzer.check_variant_concentration(
                trade_ids, strategy_id, variant_trades
            )
            exp = _compute_expectancy(variant_trades)

            if conc.variant_type == "anti":
                hypotheses.append(
                    ImprovementHypothesis(
                        strategy_id=strategy_id,
                        improvement_type=ImprovementType.VARIANT_AVOIDANCE.value,
                        analysis_layer=AnalysisLayer.VARIANT.value,
                        pattern_type="rule_based",
                        title=(
                            f"Avoid anti-variant "
                            f"'{conc.primary_variant_label}'"
                        ),
                        description=(
                            f"Failing trades are concentrated in "
                            f"anti-variant '{conc.primary_variant_label}'. "
                            f"WR: {wr:.0%}. Expectancy: {exp:.2f}R. "
                            f"Fix = avoid this variant, not new exit rules."
                        ),
                        expected_impact_r=round(
                            abs(exp) * 0.7, 3
                        ),
                        confidence=round(conc.concentration_pct, 3),
                        variant_id=variant_id,
                        variant_concentration=conc,
                        applies_to="variant_specific",
                        evidence={
                            "variant_id": variant_id,
                            "variant_type": "anti",
                            "variant_label": conc.primary_variant_label,
                            "variant_win_rate": round(wr, 4),
                            "variant_expectancy_r": round(exp, 4),
                            "concentration_pct": round(
                                conc.concentration_pct, 4
                            ),
                            "sample_size": len(variant_trades),
                        },
                        suggested_action=(
                            f"Stop trading variant "
                            f"'{conc.primary_variant_label}'."
                        ),
                        proposed_rule={
                            "type": "variant_avoidance",
                            "action": "skip_entry",
                            "condition": {"variant_id": variant_id},
                        },
                        explained_trade_ids=trade_ids,
                    )
                )
            elif conc.variant_type == "marginal":
                hypotheses.append(
                    ImprovementHypothesis(
                        strategy_id=strategy_id,
                        improvement_type=ImprovementType.VARIANT_REFINEMENT.value,
                        analysis_layer=AnalysisLayer.VARIANT.value,
                        pattern_type="rule_based",
                        title=(
                            f"Refine variant "
                            f"'{conc.primary_variant_label}'"
                        ),
                        description=(
                            f"Variant '{conc.primary_variant_label}' is "
                            f"marginal: WR {wr:.0%}, Exp {exp:.2f}R. "
                            f"Needs variant-specific exit rules or "
                            f"tighter stops."
                        ),
                        confidence=round(conc.concentration_pct * 0.7, 3),
                        variant_id=variant_id,
                        variant_concentration=conc,
                        applies_to="variant_specific",
                        evidence={
                            "variant_id": variant_id,
                            "variant_type": "marginal",
                            "variant_win_rate": round(wr, 4),
                            "variant_expectancy_r": round(exp, 4),
                            "sample_size": len(variant_trades),
                        },
                        suggested_action=(
                            f"Tighter stops or reduced sizing for "
                            f"variant '{conc.primary_variant_label}'."
                        ),
                        proposed_rule={
                            "type": "variant_refinement",
                            "action": "adjust_exits",
                            "condition": {"variant_id": variant_id},
                        },
                        explained_trade_ids=trade_ids,
                    )
                )

        return hypotheses

    def _behavioral_analysis(
        self,
        trades: List[Any],
        strategy_id: str,
    ) -> List[ImprovementHypothesis]:
        """
        Layer 3: Behavioral vs market separation.

        Compare actual trades vs shadow trades to determine:
        - Market-driven failures (shadow also failed) → strategy fix
        - Behavior-driven failures (shadow succeeded) → coaching
        - Mixed → both

        Uses shadow/comparison.py data.
        Emits BEHAVIORAL_PATTERN_DETECTED events for coaching agents.
        """
        hypotheses = []

        # Get losing trades
        losing_trades = [
            t for t in trades
            if _safe_float(_get_field(t, "pnl", 0)) < 0
        ]

        if len(losing_trades) < 5:
            return hypotheses

        # Classify all losing trades
        classification = self._behavioral_separator.classify_batch(
            losing_trades, strategy_id
        )

        if classification is None:
            return hypotheses

        # If significant behavioral component
        if (
            classification.behavioral_pct > 0.3
            and len(classification.behavioral_trade_ids) >= 3
        ):
            hypotheses.append(
                ImprovementHypothesis(
                    strategy_id=strategy_id,
                    improvement_type="coaching_recommendation",
                    analysis_layer=AnalysisLayer.BEHAVIORAL.value,
                    pattern_type="rule_based",
                    title=(
                        f"Behavioral pattern: "
                        f"{classification.behavioral_pct:.0%} of losses "
                        f"are trader-driven"
                    ),
                    description=(
                        f"{len(classification.behavioral_trade_ids)} of "
                        f"{len(losing_trades)} losses show behavioral "
                        f"origin (shadow trades succeeded). "
                        f"{classification.note}"
                    ),
                    confidence=classification.confidence,
                    behavioral_classification=classification,
                    evidence={
                        "behavioral_pct": round(
                            classification.behavioral_pct, 4
                        ),
                        "market_pct": round(
                            classification.market_pct, 4
                        ),
                        "behavioral_trade_count": len(
                            classification.behavioral_trade_ids
                        ),
                        "total_losing_trades": len(losing_trades),
                    },
                    suggested_action=(
                        f"BEHAVIORAL: {classification.note}. "
                        f"Route to coaching agent — this is not a "
                        f"strategy problem."
                    ),
                    explained_trade_ids=classification.behavioral_trade_ids,
                )
            )

        return hypotheses

    def _pattern_analysis(
        self,
        trades: List[Any],
        strategy_id: str,
        baseline_wr: float,
        all_trades: List[Any],
    ) -> List[ImprovementHypothesis]:
        """
        Layer 4: Pattern detection — only runs on trades NOT
        explained by regime, variant, or behavioral layers.

        Three detectors:
        - RR killer: favorable excursion reversals
        - Early exit: warning patterns before SL hit
        - Entry filter: conditions where strategy fails
        """
        hypotheses = []

        # R:R killer detection
        _, rr_candidates = self._rr_detector.detect(trades, strategy_id)
        for c in rr_candidates:
            c.analysis_layer = AnalysisLayer.PATTERN.value
            if c.explained_trade_ids:
                c.variant_concentration = (
                    self._variant_analyzer.check_variant_concentration(
                        c.explained_trade_ids, strategy_id, trades
                    )
                )
            hypotheses.append(c)

        # Early exit detection
        _, exit_candidates = self._exit_detector.detect(
            trades, strategy_id
        )
        for c in exit_candidates:
            c.analysis_layer = AnalysisLayer.PATTERN.value
            hypotheses.append(c)

        # Entry filter learning
        _, filter_candidates = self._filter_learner.learn(
            trades, strategy_id
        )
        for c in filter_candidates:
            c.analysis_layer = AnalysisLayer.PATTERN.value
            hypotheses.append(c)

        return hypotheses

    # ── Shadow experiment management ──────────────────────────────

    def _create_shadow_experiments(
        self,
        positive_hypotheses: List[ImprovementHypothesis],
        strategy_id: str,
    ) -> None:
        """
        Create shadow experiments for validated hypotheses.
        Enforces MAX_ACTIVE_EXPERIMENTS per strategy.
        Deduplicates by proposed_rule hash, not title.
        """
        active_experiments = self._shadow.get_active_experiments(
            strategy_id
        )
        active_count = len(active_experiments)

        if active_count >= MAX_ACTIVE_EXPERIMENTS:
            logger.info(
                "Strategy %s already has %d active experiments "
                "(max %d). Not creating new ones.",
                strategy_id,
                active_count,
                MAX_ACTIVE_EXPERIMENTS,
            )
            return

        # Deduplicate by rule fingerprint
        existing_rules = {
            _rule_fingerprint(e.candidate.proposed_rule)
            for e in active_experiments
            if hasattr(e, "candidate") and e.candidate
        }

        # Sort by expected impact (highest first)
        candidates = sorted(
            positive_hypotheses,
            key=lambda h: (
                h.tradeoff.net_expectancy_change_r if h.tradeoff else 0
            ),
            reverse=True,
        )

        for h in candidates:
            if active_count >= MAX_ACTIVE_EXPERIMENTS:
                break

            if h.tradeoff.net_expectancy_change_r < MIN_NET_R_FOR_SHADOW:
                continue

            fingerprint = _rule_fingerprint(h.proposed_rule)
            if fingerprint in existing_rules:
                continue

            self._shadow.create_experiment(h, strategy_id)
            existing_rules.add(fingerprint)
            active_count += 1

            logger.info(
                "Created shadow experiment for %s: %s (net R: %.3f)",
                strategy_id,
                h.title,
                h.tradeoff.net_expectancy_change_r,
            )

    # ── Real-time predictions ─────────────────────────────────────

    def real_time_check(
        self,
        trade: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Real-time check using two-speed learners.
        Returns ensemble predictions from fast + deep layers.
        """
        strategy_id = str(
            _get_field(trade, "declared_strategy_id")
            or (context or {}).get("strategy_id", "")
        )
        if not strategy_id:
            return {}

        features = self._extractor.extract_trade_context(
            trade, context=context
        )
        result: Dict[str, Any] = {}

        min_fast_trades = 20

        # Entry filter prediction
        learner = self._learners.get(strategy_id, {}).get("entry_filter")
        if learner and learner._fast_trade_count >= min_fast_trades:
            pred = learner.predict(features)
            win_prob = pred["probability"]
            result["win_probability"] = {
                **pred,
                "recommendation": (
                    "skip" if win_prob < 0.25
                    else "reduce_size" if win_prob < 0.35
                    else "proceed" if win_prob < 0.6
                    else "high_conviction"
                ),
            }

        # SL probability
        learner = self._learners.get(strategy_id, {}).get("early_exit")
        if learner and learner._fast_trade_count >= min_fast_trades:
            pred = learner.predict(features)
            sl_prob = pred["probability"]
            result["sl_probability"] = {
                **pred,
                "recommendation": (
                    "exit_immediately" if sl_prob > 0.85
                    else "tighten_stop" if sl_prob > 0.7
                    else "monitor" if sl_prob > 0.5
                    else "hold"
                ),
            }

        # Reversal risk
        learner = self._learners.get(strategy_id, {}).get("rr_killer")
        if learner and learner._fast_trade_count >= min_fast_trades:
            pred = learner.predict(features)
            rev_prob = pred["probability"]
            result["reversal_risk"] = {
                **pred,
                "recommendation": (
                    "take_partial" if rev_prob > 0.7
                    else "tighten_trail" if rev_prob > 0.5
                    else "hold"
                ),
            }

        if result:
            result["overall"] = _synthesize_predictions(result)

        return result

    # ── Experiment management API ─────────────────────────────────

    def get_report(
        self, strategy_id: str
    ) -> Optional[ImprovementLabReport]:
        return self._last_reports.get(strategy_id)

    def get_experiments(
        self, strategy_id: Optional[str] = None
    ) -> List[ValidationProgress]:
        return self._shadow.get_all_progress(strategy_id)

    def graduate_experiment(
        self, experiment_id: str
    ) -> Optional[ValidationProgress]:
        exp = self._shadow.graduate(experiment_id)
        return exp.to_progress() if exp else None

    def reject_experiment(
        self, experiment_id: str, reason: str = ""
    ) -> Optional[ValidationProgress]:
        exp = self._shadow.reject(experiment_id, reason)
        return exp.to_progress() if exp else None

    def shadow_summary(
        self, strategy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._shadow.summary(strategy_id)


# ══════════════════════════════════════════════════════════════════
#  Module-level helper functions
# ══════════════════════════════════════════════════════════════════


def _get_field(obj: Any, field: str, default: Any = None) -> Any:
    """Extract field from dict or object."""
    if isinstance(obj, dict):
        return obj.get(field, default)
    val = getattr(obj, field, default)
    return default if val is None else val


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _compute_win_rate(trades: List[Any]) -> float:
    """Compute win rate from trades."""
    if not trades:
        return 0.0
    wins = sum(
        1 for t in trades
        if _safe_float(_get_field(t, "pnl", 0)) > 0
    )
    return wins / len(trades)


def _compute_expectancy(trades: List[Any]) -> float:
    """Compute average R-multiple."""
    if not trades:
        return 0.0
    r_values = [
        _safe_float(_get_field(t, "r_multiple", 0))
        for t in trades
    ]
    return float(np.mean(r_values)) if r_values else 0.0


def _proportions_z_test(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
) -> float:
    """
    Two-proportion z-test.
    Returns p-value for the hypothesis that p_a < p_b.
    """
    if n_a == 0 or n_b == 0:
        return 1.0

    p_a = successes_a / n_a
    p_b = successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)

    if p_pool == 0 or p_pool == 1:
        return 1.0

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0

    z = (p_a - p_b) / se
    # One-sided: is regime WORSE than baseline?
    return float(scipy_stats.norm.sf(abs(z)))


def _compute_regime_confidence(
    drop: float, sample_size: int, p_value: float
) -> float:
    """
    Compute regime hypothesis confidence from:
    - Size of win rate drop
    - Sample size
    - Statistical significance
    """
    # Effect size component (0-0.4)
    effect_conf = min(0.4, drop * 2)

    # Sample size component (0-0.3)
    size_conf = min(0.3, sample_size / 50)

    # Significance component (0-0.3)
    sig_conf = 0.3 * max(0, 1 - p_value / 0.10) if p_value < 0.10 else 0.0

    return min(0.95, effect_conf + size_conf + sig_conf)


def _rule_fingerprint(proposed_rule: Dict[str, Any]) -> str:
    """
    Create a hashable fingerprint from a proposed rule
    for deduplication. More robust than title matching.
    """
    import json
    try:
        return json.dumps(proposed_rule, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(proposed_rule)


def _synthesize_predictions(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Combine predictions into overall recommendation."""
    flags = []
    risk_score = 0.0

    win = predictions.get("win_probability", {})
    rec = win.get("recommendation", "")
    if rec == "skip":
        flags.append("Low win probability — consider skipping")
        risk_score += 0.4
    elif rec == "reduce_size":
        flags.append("Below-average win probability — reduce size")
        risk_score += 0.2

    sl = predictions.get("sl_probability", {})
    rec = sl.get("recommendation", "")
    if rec == "exit_immediately":
        flags.append("Very high SL probability — exit or tighten stop")
        risk_score += 0.4
    elif rec == "tighten_stop":
        flags.append("Elevated SL risk — tighten stop")
        risk_score += 0.2

    rev = predictions.get("reversal_risk", {})
    if rev.get("recommendation") == "take_partial":
        flags.append("High reversal risk — take partial profit")
        risk_score += 0.2

    return {
        "risk_level": (
            "high_risk" if risk_score >= 0.6
            else "caution" if risk_score >= 0.3
            else "normal"
        ),
        "risk_score": round(min(risk_score, 1.0), 2),
        "flags": flags,
    }