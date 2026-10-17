# coplanner/conviction_engine.py
"""
Conviction Engine — 5-Component Numerical Scoring.

Computes conviction for a single setup. This is the CORE
scoring that Co-Planner uses for decision making.

RULE: Engine computes conviction. LLM only explains it.

Components & Weights:
  pattern_quality    × 0.30  — from pattern_learner.predict()
  regime_alignment   × 0.25  — from strategy_regime_matrix
  cluster_performance× 0.25  — historical cluster win rate
  volatility_fitness × 0.10  — current vol vs optimal range
  liquidity_score    × 0.10  — absolute + relative volume

Conviction Levels:
  >= 0.75 → HIGH_CONVICTION  (full size)
  >= 0.55 → MODERATE         (reduced size)
  >= 0.40 → LOW_CONVICTION   (skip or minimal)
  <  0.40 → AVOID
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from coplanner.models import (
    ConvictionLevel,
    ConvictionResult,
    RRAssessment,
    SetupData,
    WarningType,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────

HIGH_CONVICTION = 0.75
MODERATE_CONVICTION = 0.55
LOW_CONVICTION = 0.40

# Component weights
WEIGHTS = {
    "pattern_quality": 0.30,
    "regime_alignment": 0.25,
    "cluster_performance": 0.25,
    "volatility_fitness": 0.10,
    "liquidity_score": 0.10,
}


class ConvictionEngine:
    """
    Computes conviction for trade setups.

    Distinct from AVT score (which is for autonomy decisions).
    Co-Planner conviction is for human advisory decisions.

    Usage:
        engine = ConvictionEngine(
            pattern_learner=pattern_learner,
            regime_matrix=strategy_regime_matrix,
            db=coplanner_db,
            rr_enforcer=rr_enforcer,
        )
        result = engine.compute(setup)
        # → ConvictionResult(conviction_score=0.72, ...)
    """

    def __init__(
        self,
        pattern_learner: Any = None,
        regime_matrix: Any = None,
        db: Any = None,
        rr_enforcer: Any = None,
        edge_amplifier: Any = None,
    ):
        self._pattern_learner = pattern_learner
        self._regime_matrix = regime_matrix
        self._db = db
        self._rr_enforcer = rr_enforcer
        self._edge_amplifier = edge_amplifier

    # ─────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────

    def compute(self, setup: SetupData) -> ConvictionResult:
        """Compute conviction for a single setup."""
        components: Dict[str, float] = {}
        warnings: List[Dict[str, Any]] = []

        # 1. Pattern quality (0.30)
        components["pattern_quality"] = self._pattern_quality(setup)

        # 2. Regime alignment (0.25)
        components["regime_alignment"] = self._regime_alignment(
            setup.strategy_name, setup.current_regime
        )

        # 3. Cluster performance (0.25)
        components["cluster_performance"] = self._cluster_performance(
            setup.cluster_id
        )

        # 4. Volatility fitness (0.10)
        components["volatility_fitness"] = self._volatility_fitness(
            setup.symbol, setup.current_volume, setup.avg_volume
        )

        # 5. Liquidity score (0.10)
        components["liquidity_score"] = self._liquidity_score(
            setup.current_volume, setup.avg_volume
        )

        # Weighted composite
        conviction = sum(
            components[k] * WEIGHTS[k] for k in WEIGHTS
        )
        conviction = round(min(1.0, max(0.0, conviction)), 4)

        # Generate warnings
        warnings = self._generate_warnings(components)

        # Determine level + sizing
        level = self._classify(conviction)
        sizing = self._compute_sizing(conviction, warnings)

        # RR assessment (if prices available)
        rr_assessment: Optional[RRAssessment] = None
        if (
            self._rr_enforcer
            and setup.entry_price > 0
            and setup.stop_loss > 0
            and setup.target_price > 0
        ):
            rr_assessment = self._rr_enforcer.assess_with_conviction(
                entry=setup.entry_price,
                stop_loss=setup.stop_loss,
                target=setup.target_price,
                conviction_score=conviction,
            )
            # RR modifies sizing
            if rr_assessment:
                sizing = round(sizing * rr_assessment.sizing_modifier, 2)
                if not rr_assessment.acceptable:
                    warnings.append({
                        "type": WarningType.POOR_RR.value,
                        "severity": "HIGH",
                        "message": rr_assessment.message,
                    })

        # Edge amplifier: detect A+ setups and boost sizing
        edge_match = None
        if self._edge_amplifier and setup.strategy_name:
            try:
                trade_context = {
                    "market_regime": setup.current_regime or "",
                    "direction": "long",  # default; overridden by caller if available
                }
                edge_mult, edge_match = self._edge_amplifier.get_sizing_multiplier(
                    setup.strategy_name, trade_context,
                )
                if edge_mult > 1.0:
                    sizing = round(sizing * edge_mult, 2)
                    logger.info(
                        "Edge amplifier: %s A+ edge matched → sizing %.2fx (grade=%s, WR=%.0f%%)",
                        setup.symbol, edge_mult,
                        edge_match.grade if edge_match else "?",
                        (edge_match.win_rate * 100) if edge_match else 0,
                    )
            except Exception as e:
                logger.debug("Edge amplifier failed: %s", e)

        # Build result components dict with edge info
        result_components = {k: round(v, 4) for k, v in components.items()}
        if edge_match:
            result_components["edge_amplifier"] = round(edge_match.confidence, 4)

        return ConvictionResult(
            symbol=setup.symbol,
            strategy_name=setup.strategy_name,
            conviction_score=conviction,
            conviction_level=level,
            components=result_components,
            warnings=warnings,
            sizing_modifier=sizing,
            verdict=self._verdict(conviction, level),
            rr_assessment=rr_assessment,
            computed_at=datetime.now(),
        )

    # ─────────────────────────────────────────────
    # COMPONENT 1: Pattern Quality
    # ─────────────────────────────────────────────

    def _pattern_quality(self, setup: SetupData) -> float:
        """
        From pattern_learner.predict() or fallback to setup.pattern_score.
        """
        # Try pattern_learner first
        if self._pattern_learner and setup.pattern_data:
            try:
                result = self._pattern_learner.predict(setup.pattern_data)
                if isinstance(result, (int, float)):
                    return float(result)
                if isinstance(result, dict):
                    return float(result.get("probability", result.get("score", 0.5)))
            except Exception:
                logger.debug("pattern_learner.predict() failed", exc_info=True)

        # Fallback: use pre-computed pattern_score
        if setup.pattern_score > 0:
            return min(1.0, setup.pattern_score)

        return 0.5  # neutral

    # ─────────────────────────────────────────────
    # COMPONENT 2: Regime Alignment
    # ─────────────────────────────────────────────

    def _regime_alignment(self, strategy_name: str, current_regime: str) -> float:
        """
        How well does this strategy perform in current regime?
        From strategy_regime_matrix.query_normalized_score().
        """
        if self._regime_matrix:
            try:
                return self._regime_matrix.query_normalized_score(
                    strategy_name, current_regime
                )
            except Exception:
                logger.debug("Regime matrix query failed", exc_info=True)
        return 0.5  # neutral if unavailable

    # ─────────────────────────────────────────────
    # COMPONENT 3: Cluster Performance
    # ─────────────────────────────────────────────

    def _cluster_performance(self, cluster_id: Optional[str]) -> float:
        """
        Historical win rate of similar pattern clusters.
        Confidence-adjusted: pulls toward 0.5 for small samples.
        """
        if not cluster_id or not self._db:
            return 0.5

        try:
            stats = self._db.get_cluster_stats(cluster_id)
            wr = stats.get("win_rate", 0.5)
            n = stats.get("total_trades", 0)

            # Confidence adjustment
            if n >= 30:
                conf = 0.95
            elif n >= 15:
                conf = 0.80
            elif n >= 5:
                conf = 0.60
            else:
                conf = 0.30

            return wr * conf + 0.5 * (1 - conf)
        except Exception:
            return 0.5

    # ─────────────────────────────────────────────
    # COMPONENT 4: Volatility Fitness
    # ─────────────────────────────────────────────

    def _volatility_fitness(
        self,
        symbol: str,
        current_volume: float,
        avg_volume: float,
    ) -> float:
        """
        Is current volatility within the user's optimal range?
        Approximated from volume ratio as a volatility proxy.
        """
        if avg_volume <= 0:
            return 0.5

        ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Optimal range: 0.8x to 2.0x average volume
        if 0.8 <= ratio <= 2.0:
            # Within range — score by distance from sweet spot (1.2x)
            sweet_spot = 1.2
            distance = abs(ratio - sweet_spot) / 1.2
            return round(max(0.5, 1.0 - distance), 4)
        elif ratio < 0.8:
            return round(max(0.1, ratio * 0.6), 4)
        else:
            # Very high volume — may indicate unusual event
            return round(max(0.3, 2.0 / ratio * 0.7), 4)

    # ─────────────────────────────────────────────
    # COMPONENT 5: Liquidity Score
    # ─────────────────────────────────────────────

    def _liquidity_score(
        self,
        current_volume: float,
        avg_volume: float,
    ) -> float:
        """
        Blend of absolute volume + relative volume.
        60% absolute + 40% relative.
        Returns 0.70 (neutral) when no volume data is available.
        """
        # No volume data provided — return neutral score, not a penalty
        if avg_volume <= 0 and current_volume <= 0:
            return 0.70

        # Absolute score
        if avg_volume >= 5_000_000:
            abs_score = 1.0
        elif avg_volume >= 2_000_000:
            abs_score = 0.9
        elif avg_volume >= 1_000_000:
            abs_score = 0.8
        elif avg_volume >= 500_000:
            abs_score = 0.65
        elif avg_volume >= 100_000:
            abs_score = 0.5
        else:
            abs_score = 0.3

        # Relative score
        if avg_volume <= 0:
            rel_score = 0.5
        else:
            ratio = current_volume / avg_volume
            if ratio >= 1.5:
                rel_score = 1.0
            elif ratio >= 1.2:
                rel_score = 0.85
            elif ratio >= 1.0:
                rel_score = 0.75
            elif ratio >= 0.7:
                rel_score = 0.6
            else:
                rel_score = max(0.2, ratio * 0.5)

        return round(abs_score * 0.6 + rel_score * 0.4, 4)

    # ─────────────────────────────────────────────
    # WARNINGS
    # ─────────────────────────────────────────────

    def _generate_warnings(self, components: Dict[str, float]) -> List[Dict[str, Any]]:
        warnings: List[Dict[str, Any]] = []

        if components.get("regime_alignment", 1) < 0.40:
            warnings.append({
                "type": WarningType.REGIME_MISMATCH.value,
                "severity": "HIGH",
                "message": "Strategy underperforms in current regime.",
            })

        if components.get("cluster_performance", 1) < 0.45:
            warnings.append({
                "type": WarningType.LOW_CLUSTER_WR.value,
                "severity": "MEDIUM",
                "message": "Similar patterns historically weak.",
            })

        if components.get("volatility_fitness", 1) < 0.40:
            warnings.append({
                "type": WarningType.VOLATILITY_OUTSIDE.value,
                "severity": "MEDIUM",
                "message": "Volatility outside optimal range.",
            })

        if components.get("liquidity_score", 1) < 0.50:
            warnings.append({
                "type": WarningType.LOW_LIQUIDITY.value,
                "severity": "MEDIUM",
                "message": "Low liquidity — slippage risk.",
            })

        return warnings

    # ─────────────────────────────────────────────
    # CLASSIFICATION
    # ─────────────────────────────────────────────

    @staticmethod
    def _classify(score: float) -> ConvictionLevel:
        if score >= HIGH_CONVICTION:
            return ConvictionLevel.HIGH
        if score >= MODERATE_CONVICTION:
            return ConvictionLevel.MODERATE
        if score >= LOW_CONVICTION:
            return ConvictionLevel.LOW
        return ConvictionLevel.AVOID

    @staticmethod
    def _compute_sizing(conviction: float, warnings: List[Dict]) -> float:
        if conviction >= HIGH_CONVICTION:
            sizing = 1.0
        elif conviction >= MODERATE_CONVICTION:
            # Linear scale from 0.5 to 0.9 within moderate band
            ratio = (conviction - MODERATE_CONVICTION) / (HIGH_CONVICTION - MODERATE_CONVICTION)
            sizing = 0.5 + ratio * 0.4
        elif conviction >= LOW_CONVICTION:
            sizing = 0.3
        else:
            sizing = 0.0

        # Penalty for HIGH severity warnings: -15% each
        high_warnings = sum(1 for w in warnings if w.get("severity") == "HIGH")
        sizing = max(0.0, sizing - high_warnings * 0.15)

        return round(sizing, 2)

    @staticmethod
    def _verdict(score: float, level: ConvictionLevel) -> str:
        if level == ConvictionLevel.HIGH:
            return "STRONG — proceed with full size"
        if level == ConvictionLevel.MODERATE:
            return "MODERATE — proceed with reduced size"
        if level == ConvictionLevel.LOW:
            return "WEAK — consider skipping"
        return "AVOID — conditions unfavorable"
