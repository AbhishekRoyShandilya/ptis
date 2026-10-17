# bootstrap/warm_starter.py
"""
Warm Start System
==================

Takes the backtest results and initializes ALL downstream
systems with warm data so they're functional from Day 1.

Also manages the correction loop where user reviews
and corrects the system's understanding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WarmStartReport:
    """Report of what was initialized and how."""
    strategy_id: str = ""

    # What was initialized
    scaler_initialized: bool = False
    scaler_features: int = 0
    scaler_samples: int = 0

    clusterer_initialized: bool = False
    clusters_found: int = 0
    cluster_method: str = ""

    regime_checker_initialized: bool = False
    regimes_with_data: int = 0

    # Quality
    synthetic_trade_count: int = 0
    avg_synthetic_confidence: float = 0.0

    # Validation
    performance_reasonable: bool = False
    warnings: List[str] = field(default_factory=list)

    # AVT mode recommendation
    recommended_avt_mode: str = "observer"


class WarmStarter:
    """
    Initializes all downstream systems from synthetic data.
    """

    def __init__(
        self,
        strategy_engine: Any,    # StrategyUnderstandingEngine
        pattern_intel: Any,      # PatternAwareFirewall
    ):
        self._engine = strategy_engine
        self._intel = pattern_intel

    def warm_start(
        self,
        backtest_result: Any,    # BacktestResult
        draft: Any,              # StrategyDraft
        user_claimed_wr: Optional[float] = None,
    ) -> WarmStartReport:
        """
        Initialize all systems with synthetic trade data.

        IMPORTANT: All synthetic data is weighted at 0.5
        (half the weight of real trades). As real data
        arrives, it gradually replaces synthetic data.
        """
        report = WarmStartReport(
            strategy_id=draft.strategy_name,
        )

        trades = backtest_result.trades
        feature_vectors = backtest_result.feature_vectors

        if not trades:
            report.warnings.append(
                "No synthetic trades generated. "
                "Cannot warm-start systems."
            )
            return report

        report.synthetic_trade_count = len(trades)
        report.avg_synthetic_confidence = (
            backtest_result.avg_synthetic_confidence
        )

        # 1. Initialize feature scaler
        self._init_scaler(feature_vectors, report)

        # 2. Initialize pattern clusterer
        self._init_clusterer(trades, report)

        # 3. Initialize regime checker
        self._init_regime_checker(trades, report)

        # 4. Initialize template centroids if backtest
        #    didn't generate enough for clustering
        if not report.clusterer_initialized and draft.initial_centroids:
            self._init_from_template_centroids(
                draft.initial_centroids, report
            )

        # 5. Validate performance
        self._validate_performance(
            backtest_result, user_claimed_wr, report
        )

        # 6. Determine AVT mode
        report.recommended_avt_mode = self._determine_avt_mode(
            draft, backtest_result, report
        )

        logger.info(
            "Warm start complete for '%s': "
            "%d synthetic trades, %d clusters, "
            "%d regimes, AVT mode: %s",
            draft.strategy_name,
            report.synthetic_trade_count,
            report.clusters_found,
            report.regimes_with_data,
            report.recommended_avt_mode,
        )

        return report

    def _init_scaler(
        self,
        feature_vectors: List[Dict[str, float]],
        report: WarmStartReport,
    ) -> None:
        """Initialize feature scaler with synthetic data."""
        scaler = self._intel.scaler

        for fv in feature_vectors:
            scaler.update(fv)

        report.scaler_initialized = scaler.is_ready
        report.scaler_features = len(scaler._feature_keys)
        report.scaler_samples = scaler.sample_count

        if not report.scaler_initialized:
            report.warnings.append(
                f"Scaler not fully ready: "
                f"{scaler.sample_count} samples "
                f"(need 20)"
            )

    def _init_clusterer(
        self,
        trades: List[Any],
        report: WarmStartReport,
    ) -> None:
        """Initialize clusterer with synthetic trades."""
        clusterer = self._intel.clusterer

        for trade in trades:
            features = trade.features_at_entry
            if not features:
                continue
            cluster_id = clusterer.add_trade(
                features=features,
                outcome_r=trade.r_multiple,
                regime=trade.market_regime or "",
            )

        # Force recluster
        if len(clusterer._data_buffer) >= 20:
            result = clusterer.recluster()
            report.clusterer_initialized = result.n_clusters >= 2
            report.clusters_found = result.n_clusters
            report.cluster_method = result.method
        else:
            report.warnings.append(
                "Not enough trades for clustering"
            )

    def _init_regime_checker(
        self,
        trades: List[Any],
        report: WarmStartReport,
    ) -> None:
        """Initialize regime stability checker."""
        regime_checker = self._engine._regime

        for trade in trades:
            regime = trade.market_regime or "unknown"
            regime_checker.record_outcome(
                regime=regime,
                match_score=trade.conviction_score,
                actual_outcome_r=trade.r_multiple,
                timestamp=trade.entry_time,
            )

        stability = regime_checker.compute_stability()
        report.regime_checker_initialized = stability.get(
            "sufficient_data", False
        )
        report.regimes_with_data = len(
            stability.get("regime_scores", {})
        )

    def _init_from_template_centroids(
        self,
        centroids: List[Dict[str, float]],
        report: WarmStartReport,
    ) -> None:
        """
        If clustering didn't work, use template centroids
        as initial pattern definitions.
        """
        # Manually create clusters from template centroids
        from strategy_quality.pattern_intelligence import PatternCluster

        clusterer = self._intel.clusterer
        scaler = self._intel.scaler

        for i, centroid_dict in enumerate(centroids):
            # Update scaler with centroid values
            scaler.update(centroid_dict)

        if scaler.is_ready:
            for i, centroid_dict in enumerate(centroids):
                vec = scaler.to_vector(centroid_dict, method="zscore")
                cluster = PatternCluster(
                    cluster_id=i,
                    centroid=vec,
                    centroid_dict=centroid_dict,
                    size=1,
                    label=f"template_variant_{i}",
                )
                clusterer._clusters.append(cluster)

            report.clusterer_initialized = True
            report.clusters_found = len(centroids)
            report.cluster_method = "template_centroids"

    def _validate_performance(
        self,
        result: Any,
        user_claimed_wr: Optional[float],
        report: WarmStartReport,
    ) -> None:
        """
        Check if synthetic performance is reasonable.

        KEY INSIGHT: Synthetic performance should UNDERPERFORM
        user's real performance because we're missing the
        judgment layer. If synthetic OUTPERFORMS user's claim,
        something is wrong.
        """
        wr = result.win_rate
        avg_r = result.avg_r

        # Basic sanity
        if wr > 0.80:
            report.warnings.append(
                f"Synthetic WR {wr:.0%} is suspiciously high. "
                f"Rules may be overfitted."
            )
        if wr < 0.20:
            report.warnings.append(
                f"Synthetic WR {wr:.0%} is very low. "
                f"Rules may be too strict or wrong."
            )

        # Compare with user's claim
        if user_claimed_wr is not None:
            deviation = abs(wr - user_claimed_wr)
            result.deviation_from_claim = deviation

            if wr > user_claimed_wr + 0.10:
                report.warnings.append(
                    f"Synthetic WR ({wr:.0%}) EXCEEDS user's "
                    f"claim ({user_claimed_wr:.0%}). This suggests "
                    f"overfitting or the rules are too selective "
                    f"(taking only easy setups)."
                )
            elif wr < user_claimed_wr - 0.20:
                report.warnings.append(
                    f"Synthetic WR ({wr:.0%}) is much BELOW "
                    f"user's claim ({user_claimed_wr:.0%}). "
                    f"Expected (judgment gap), but if gap is "
                    f"too large our rules may be wrong."
                )
            elif wr < user_claimed_wr:
                # This is EXPECTED and GOOD
                report.performance_reasonable = True
            else:
                report.performance_reasonable = True

        elif 0.35 <= wr <= 0.70 and avg_r > -0.5:
            report.performance_reasonable = True

    def _determine_avt_mode(
        self,
        draft: Any,
        result: Any,
        report: WarmStartReport,
    ) -> str:
        """Determine initial AVT mode."""
        confidence = draft.understanding_confidence or 0.0

        if (
            confidence >= 0.75
            and report.performance_reasonable
            and report.clusterer_initialized
            and report.avg_synthetic_confidence >= 0.6
        ):
            return "independent"
        elif (
            confidence >= 0.45
            and report.performance_reasonable
        ):
            return "candidate_presenter"
        else:
            return "observer"


class CorrectionLoop:
    """
    Manages user review and correction of system understanding.

    Presents signals to user. User says "yes I'd trade this"
    or "no, that's not my strategy because..."

    Each correction updates:
    - Hard rules (if user identifies wrong conditions)
    - Soft scores (recalibrates what "quality" means)
    - Feature scaler (new data points)
    - Clusters (new pattern assignments)
    - Understanding confidence (improves with corrections)
    """

    def __init__(
        self,
        engine: Any,    # StrategyUnderstandingEngine
        interpreter: Any,  # LLMStrategyInterpreter
    ):
        self._engine = engine
        self._interpreter = interpreter
        self._corrections: List[Dict[str, Any]] = []

    async def present_for_review(
        self,
        signal_features: Dict[str, float],
        signal_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Present a signal to user for review.
        Returns the signal with system's assessment.
        """
        output = self._engine.evaluate(signal_features)

        return {
            "signal": signal_context,
            "system_assessment": {
                "match_score": output.overall_match_score,
                "confidence": output.strategy_confidence,
                "decision": output.decision,
                "soft_scores": output.soft_score_components,
                "context": output.context_components,
            },
            "question": (
                "Would you trade this setup? "
                "If no, what's wrong with it?"
            ),
        }

    async def process_correction(
        self,
        signal_features: Dict[str, float],
        user_response: str,      # "yes" | "no"
        user_reason: str = "",   # why they'd skip it
        regime: str = "",
    ) -> Dict[str, Any]:
        """
        Process user's correction.

        This is THE most valuable data in the system.
        Each correction directly improves understanding.
        """
        correction = {
            "features": signal_features,
            "user_response": user_response,
            "reason": user_reason,
            "regime": regime,
            "timestamp": str(
                __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                )
            ),
        }
        self._corrections.append(correction)

        # Update calibration
        output = self._engine.evaluate(signal_features)

        # TODO: Wire calibration tracking when orchestrator connects
        # from avt.calibration_tracker import SignalCalibrationTracker
        # tracker.record(output, user_response, regime=regime)

        result = {
            "correction_recorded": True,
            "total_corrections": len(self._corrections),
        }

        # If user said no and gave a reason, try to learn from it
        if user_response == "no" and user_reason:
            learning = await self._learn_from_rejection(
                signal_features, user_reason
            )
            result["learning"] = learning

        # Update recommendation
        if len(self._corrections) >= 10:
            result["calibration_update"] = self._recalibrate()

        return result

    async def _learn_from_rejection(
        self,
        features: Dict[str, float],
        reason: str,
    ) -> Dict[str, Any]:
        """
        Use LLM to interpret WHY user rejected and
        map it to system parameters.
        """
        if self._interpreter._llm is None:
            return {"status": "no_llm"}

        try:
            from agents.prompts import SIGNAL_INTERPRETER as _SYS
            response = await self._interpreter._llm.chat([
                {
                    "role": "system",
                    "content": _SYS,
                },
                {
                    "role": "user",
                    "content": (
                        f"Signal features: {features}\n"
                        f"Rejection reason: {reason}\n\n"
                        "What should be changed? Respond as JSON: "
                        '{"adjustment_type": "hard_rule|soft_score|'
                        'uncertainty", "details": {...}}'
                    ),
                },
            ], caller="bootstrap:rejection_interpret")

            import json
            adjustment = json.loads(response.content)
            return {"status": "learned", "adjustment": adjustment}

        except Exception as e:
            return {"status": "learning_failed", "error": str(e)}

    def _recalibrate(self) -> Dict[str, Any]:
        """Recalibrate based on accumulated corrections."""
        total = len(self._corrections)
        yes_count = sum(
            1 for c in self._corrections
            if c["user_response"] == "yes"
        )
        no_count = total - yes_count

        acceptance_rate = yes_count / max(1, total)

        result = {
            "total_corrections": total,
            "acceptance_rate": round(acceptance_rate, 3),
        }

        if acceptance_rate < 0.3:
            result["diagnosis"] = (
                "System generates too many false signals. "
                "Rules are too loose."
            )
        elif acceptance_rate > 0.8:
            result["diagnosis"] = (
                "System is well-calibrated or too conservative."
            )
        else:
            result["diagnosis"] = "Normal calibration range."

        return result

    def get_correction_summary(self) -> Dict[str, Any]:
        """Summary of all corrections for review."""
        if not self._corrections:
            return {"total": 0}

        reasons = [
            c.get("reason", "")
            for c in self._corrections
            if c["user_response"] == "no" and c.get("reason")
        ]

        return {
            "total": len(self._corrections),
            "accepted": sum(
                1 for c in self._corrections
                if c["user_response"] == "yes"
            ),
            "rejected": sum(
                1 for c in self._corrections
                if c["user_response"] == "no"
            ),
            "common_rejection_reasons": reasons[:10],
        }