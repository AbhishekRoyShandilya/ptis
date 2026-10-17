# coplanner/cross_stock_comparator.py
"""
Cross-Stock Comparator — Head-to-Head Setup Comparison.

Takes multiple setups (same or different strategies),
scores each through conviction engine, and outputs:
  - Ranked list with composite scores
  - Clear recommendation (TAKE / CAUTIOUS / SKIP_ALL)
  - Alternative strategy suggestions when current is weak

RULE: Engine ranks. LLM explains "why HDFC > INFY".
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from coplanner.conviction_engine import ConvictionEngine
from coplanner.models import (
    ConvictionResult,
    RankedOpportunity,
    SetupData,
)
from coplanner.rr_enforcer import RiskRewardEnforcer
from coplanner.strategy_regime_matrix import StrategyRegimeMatrix

logger = logging.getLogger(__name__)


class CrossStockComparator:
    """
    Compare multiple setups head-to-head and recommend the best.

    Usage:
        comparator = CrossStockComparator(conviction, matrix, rr_enforcer)
        result = comparator.compare(setups)
        # result['ranked_setups'], result['recommendation']
    """

    def __init__(
        self,
        conviction_engine: ConvictionEngine,
        regime_matrix: StrategyRegimeMatrix,
        rr_enforcer: RiskRewardEnforcer,
    ):
        self._conviction = conviction_engine
        self._matrix = regime_matrix
        self._rr = rr_enforcer

    def compare(self, setups: List[SetupData]) -> Dict[str, Any]:
        """
        Input:  list of SetupData from watchlist scan
        Output: ranked comparison with clear recommendation
        """
        if not setups:
            return {
                "ranked_setups": [],
                "recommendation": {"action": "NO_SETUPS", "message": "No setups to evaluate."},
                "total_scanned": 0,
                "actionable": 0,
            }

        evaluated: List[Dict[str, Any]] = []

        for setup in setups:
            # Score through conviction engine
            conv = self._conviction.compute(setup)

            # Regime stats
            regime_stats = self._matrix.query(
                setup.strategy_name, setup.current_regime
            )

            # Avoidance check
            avoidance = self._matrix.should_avoid(
                setup.strategy_name, setup.current_regime
            )

            # Final rank score
            rr_ok = conv.rr_assessment.acceptable if conv.rr_assessment else True
            expected_rr = conv.rr_assessment.expected_rr if conv.rr_assessment else setup.compute_raw_rr()

            composite = self._final_rank_score(
                conv.conviction_score,
                regime_stats.win_rate,
                expected_rr,
                rr_ok,
            )

            # Strengths / weaknesses
            strengths, weaknesses = self._analyze_dimensions(conv, regime_stats)

            evaluated.append({
                "symbol": setup.symbol,
                "strategy": setup.strategy_name,
                "conviction": conv,
                "regime_wr": regime_stats.win_rate,
                "regime_avg_rr": regime_stats.avg_rr,
                "regime_confidence": regime_stats.sample_confidence.value,
                "expected_rr": expected_rr,
                "rr_acceptable": rr_ok,
                "avoidance": avoidance,
                "composite_score": composite,
                "strengths": strengths,
                "weaknesses": weaknesses,
            })

        # Sort by composite
        evaluated.sort(key=lambda x: x["composite_score"], reverse=True)

        # Assign ranks and generate recommendation
        recommendation = self._generate_recommendation(evaluated)

        return {
            "ranked_setups": evaluated,
            "recommendation": recommendation,
            "total_scanned": len(setups),
            "actionable": sum(
                1 for e in evaluated
                if e["conviction"].conviction_score >= 0.55 and e["rr_acceptable"]
            ),
        }

    def _final_rank_score(
        self,
        conviction: float,
        regime_wr: float,
        expected_rr: float,
        rr_ok: bool,
    ) -> float:
        """Composite rank score blending conviction + regime + RR."""
        if not rr_ok:
            return conviction * 0.3  # heavily penalize bad RR

        rr_normalized = min(expected_rr / 3.0, 1.0)
        return round(
            conviction * 0.40 + regime_wr * 0.35 + rr_normalized * 0.25,
            4,
        )

    def _analyze_dimensions(
        self,
        conv: ConvictionResult,
        regime_stats: Any,
    ) -> tuple[List[str], List[str]]:
        """Identify strengths and weaknesses across dimensions."""
        strengths: List[str] = []
        weaknesses: List[str] = []

        c = conv.components
        if c.get("pattern_quality", 0) >= 0.70:
            strengths.append("Strong pattern quality")
        elif c.get("pattern_quality", 1) < 0.40:
            weaknesses.append("Weak pattern quality")

        if c.get("regime_alignment", 0) >= 0.65:
            strengths.append("Good regime alignment")
        elif c.get("regime_alignment", 1) < 0.40:
            weaknesses.append("Poor regime alignment")

        if c.get("cluster_performance", 0) >= 0.65:
            strengths.append("Strong cluster track record")
        elif c.get("cluster_performance", 1) < 0.45:
            weaknesses.append("Weak cluster history")

        if c.get("liquidity_score", 0) >= 0.80:
            strengths.append("High liquidity")
        elif c.get("liquidity_score", 1) < 0.50:
            weaknesses.append("Low liquidity")

        if regime_stats.win_rate >= 0.55 and regime_stats.total_trades >= 5:
            strengths.append(f"Strategy WR {regime_stats.win_rate:.0%} in this regime")
        elif regime_stats.win_rate < 0.42 and regime_stats.total_trades >= 8:
            weaknesses.append(f"Strategy WR only {regime_stats.win_rate:.0%} in this regime")

        return strengths, weaknesses

    def _generate_recommendation(
        self, ranked: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate clear action recommendation from ranked setups."""
        if not ranked:
            return {"action": "NO_SETUPS", "message": "No setups to evaluate."}

        top = ranked[0]
        top_conv = top["conviction"].conviction_score
        top_rr_ok = top["rr_acceptable"]

        if top_conv >= 0.70 and top_rr_ok:
            return {
                "action": "TAKE",
                "primary": top["symbol"],
                "message": (
                    f"{top['symbol']} is clear best — "
                    f"conviction {top_conv:.2f}, "
                    f"strategy WR {top['regime_wr']:.0%} in this regime."
                ),
                "sizing": "FULL" if top_conv >= 0.78 else "REDUCED",
            }

        if top_conv >= 0.55 and top_rr_ok:
            return {
                "action": "CAUTIOUS",
                "primary": top["symbol"],
                "message": (
                    f"{top['symbol']} is moderate — proceed with reduced size."
                ),
                "sizing": "REDUCED",
            }

        # Check if avoidance has alternatives
        alt_msg = ""
        if top["avoidance"].avoid and top["avoidance"].alternatives:
            best_alt = top["avoidance"].alternatives[0]
            alt_msg = (
                f" Consider {best_alt['strategy']} instead "
                f"(WR {best_alt['win_rate']:.0%} in this regime)."
            )

        return {
            "action": "SKIP_ALL",
            "message": f"No high-conviction setup today. Best to wait.{alt_msg}",
            "sizing": "NONE",
        }

    # ─────────────────────────────────────────────
    # EXTENDED COMPARISONS
    # ─────────────────────────────────────────────

    def compare_same_strategy_different_stocks(
        self, strategy_name: str, symbols: List[str], current_regime: str,
    ) -> Dict[str, Any]:
        """
        Compare same strategy across different stocks.
        E.g., "breakout on RELIANCE vs INFY vs HDFC in trending_up"
        """
        setups = [
            SetupData(
                symbol=sym,
                strategy_name=strategy_name,
                current_regime=current_regime,
            )
            for sym in symbols
        ]
        result = self.compare(setups)
        result["comparison_type"] = "same_strategy_different_stocks"
        result["strategy"] = strategy_name
        result["regime"] = current_regime
        return result

    def compare_different_strategies_same_stock(
        self, symbol: str, strategies: List[str], current_regime: str,
    ) -> Dict[str, Any]:
        """
        Compare different strategies on the same stock.
        E.g., "breakout vs pullback vs reversal on RELIANCE in trending_up"
        """
        setups = [
            SetupData(
                symbol=symbol,
                strategy_name=strat,
                current_regime=current_regime,
            )
            for strat in strategies
        ]
        result = self.compare(setups)
        result["comparison_type"] = "different_strategies_same_stock"
        result["symbol"] = symbol
        result["regime"] = current_regime
        return result

    def suggest_alternative_strategy(
        self, symbol: str, current_strategy: str, current_regime: str,
    ) -> Dict[str, Any]:
        """
        If current strategy is weak for this regime, suggest better ones
        based on regime matrix data.
        """
        avoidance = self._matrix.should_avoid(current_strategy, current_regime)
        best_strategies = self._matrix.get_best_strategies_for_regime(current_regime)

        if not avoidance.avoid:
            current_stats = self._matrix.query(current_strategy, current_regime)
            return {
                "needs_alternative": False,
                "current_strategy": current_strategy,
                "current_wr": current_stats.win_rate,
                "message": f"{current_strategy} is acceptable in {current_regime}.",
            }

        alternatives = [
            s for s in best_strategies
            if s["strategy"] != current_strategy.strip().lower().replace(" ", "_")
        ][:3]

        return {
            "needs_alternative": True,
            "current_strategy": current_strategy,
            "current_wr": avoidance.current_wr,
            "avoid_reason": avoidance.reason,
            "alternatives": alternatives,
            "recommendation": (
                f"Avoid {current_strategy} in {current_regime} "
                f"(WR {avoidance.current_wr:.0%}). "
                + (
                    f"Try {alternatives[0]['strategy']} instead "
                    f"(WR {alternatives[0]['win_rate']:.0%})."
                    if alternatives else "No strong alternatives available."
                )
            ),
        }

    def get_narrative(
        self, comparison_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build LLM-ready structured comparison data.
        Engine computes, LLM narrates.
        """
        ranked = comparison_result.get("ranked_setups", [])
        if not ranked:
            return {"narrative_data": "No setups to compare."}

        lines: List[str] = []
        for i, setup in enumerate(ranked, 1):
            conv = setup.get("conviction", None)
            conv_score = conv.conviction_score if conv else 0
            lines.append(
                f"#{i} {setup['symbol']} ({setup['strategy']}): "
                f"conviction={conv_score:.2f}, "
                f"regime_wr={setup.get('regime_wr', 0):.0%}, "
                f"RR={setup.get('expected_rr', 0):.1f}"
            )
            if setup.get("strengths"):
                lines.append(f"   Strengths: {', '.join(setup['strengths'])}")
            if setup.get("weaknesses"):
                lines.append(f"   Weaknesses: {', '.join(setup['weaknesses'])}")

        rec = comparison_result.get("recommendation", {})
        lines.append(f"\nRecommendation: {rec.get('action', 'N/A')} — {rec.get('message', '')}")

        return {
            "narrative_data": "\n".join(lines),
            "comparison_type": comparison_result.get("comparison_type", "general"),
            "total": comparison_result.get("total_scanned", 0),
            "actionable": comparison_result.get("actionable", 0),
        }
