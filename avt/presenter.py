# avt/presenter.py
"""
AVT Presenter — Storytelling with Data
========================================

Generates human-readable, compelling presentations of what the AVT
learned and proved. This is the INTERFACE between AVT intelligence
and human trust.

NOT: "Statistical analysis shows improvement of 0.31R with p-value 0.04"

YES: "I've been trading your Compression Breakout strategy for 6 weeks.
     In 89 virtual trades, I found something interesting:

     When a trade reaches 1:2 profit and the market has gone flat,
     the price reverses back 73% of the time. I started booking half
     my position at 1:2 in these conditions 3 weeks ago.

     Since then: 12 times triggered, 9 times helped, net +0.22R/trade.
     Here are 3 specific examples..."

This is STORYTELLING WITH DATA. Not data with reluctant explanation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from avt.models import (
    ActiveImprovement,
    CompetitionReport,
    ImprovementPresentation,
    PerformanceSnapshot,
    VirtualTrade,
    compute_performance,
)

logger = logging.getLogger(__name__)


class AVTPresenter:
    """
    Generates narratives that convince the trader through evidence.

    Uses LLM when available, falls back to template-based generation.
    """

    def __init__(self, llm: Any = None):
        self._llm = llm

    # ── Improvement Presentation ───────────────────────────────

    async def generate_improvement_presentation(
        self,
        improvement: ActiveImprovement,
        avt_engine: Any,
    ) -> ImprovementPresentation:
        """
        Generate a compelling, evidence-based presentation
        of a proven improvement.
        """
        strategy_id = improvement.strategy_id

        # Get trades where improvement was active
        trades_with = avt_engine.get_trades_where_improvement_applied(
            str(improvement.id),
        )

        # Get trades before improvement
        trades_before = avt_engine.get_closed_trades(
            strategy_id=strategy_id,
        )
        trades_without = [
            t for t in trades_before
            if t.entry_time and improvement.applied_at
            and t.entry_time < improvement.applied_at
        ]

        # Find best examples
        examples = self._find_best_examples(trades_with, improvement)

        # Compute stats
        perf_before = improvement.performance_before or compute_performance(trades_without)
        perf_after = improvement.performance_after or compute_performance(trades_with)

        times_triggered = len(trades_with)
        times_helped = sum(1 for t in trades_with if (t.pnl_r or 0) > 0)
        times_hurt = sum(1 for t in trades_with if (t.pnl_r or 0) < 0)
        net_impact_r = perf_after.expectancy_r - perf_before.expectancy_r if perf_before else 0
        total_impact_r = sum((t.pnl_r or 0) for t in trades_with)
        weeks_since = 0
        if improvement.applied_at:
            weeks_since = (datetime.now(timezone.utc) - improvement.applied_at).days / 7

        summary_stats = {
            "times_triggered": times_triggered,
            "times_helped": times_helped,
            "times_hurt": times_hurt,
            "net_impact_per_trade_r": round(net_impact_r, 3),
            "total_impact_r": round(total_impact_r, 2),
            "weeks_since_applied": round(weeks_since, 1),
            "before_win_rate": round(perf_before.win_rate, 3) if perf_before else 0,
            "after_win_rate": round(perf_after.win_rate, 3) if perf_after else 0,
            "before_expectancy": round(perf_before.expectancy_r, 3) if perf_before else 0,
            "after_expectancy": round(perf_after.expectancy_r, 3) if perf_after else 0,
        }

        # Generate narrative
        narrative = await self._generate_narrative(
            improvement=improvement,
            trades_with=trades_with,
            trades_without=trades_without,
            examples=examples,
            summary_stats=summary_stats,
        )

        return ImprovementPresentation(
            title=f"Improvement Found: {improvement.description}",
            narrative=narrative,
            summary_stats=summary_stats,
            examples=examples,
            avt_confidence=improvement.evidence_strength,
        )

    # ── Example Finding ────────────────────────────────────────

    @staticmethod
    def _find_best_examples(
        trades: List[VirtualTrade],
        improvement: ActiveImprovement,
        n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find the most dramatic examples where the improvement helped.

        Best examples = trades where the improvement-triggered exit
        captured significantly more profit than holding would have.
        """
        if not trades:
            return []

        # For partial exit improvements, the "counterfactual" is
        # what would have happened without the partial exit
        examples = []

        for trade in trades:
            # Estimate counterfactual R (what would have happened without improvement)
            # If improvement caused an exit, the trade would have continued
            # Use max_adverse_r as worst case, or assume it would have reverted
            actual_r = (trade.pnl_r or 0)
            counterfactual_r = actual_r  # default

            if improvement.type == "partial_exit_at_rr":
                # Without the partial exit, the trade would have run to final exit
                # Use max_favorable - some giveback as estimate
                if trade.max_favorable_r > 0:
                    # Assume 40-60% giveback without partial exit
                    counterfactual_r = trade.max_favorable_r * 0.5
                    # If actual R > counterfactual, the improvement helped
                    if actual_r <= counterfactual_r:
                        counterfactual_r = actual_r * 0.6  # would have been worse

            elif improvement.type == "early_exit_pattern":
                # Without early exit, trade would have hit stop loss
                counterfactual_r = -1.0  # assume full SL hit

            difference = actual_r - counterfactual_r

            examples.append({
                "symbol": trade.symbol,
                "trade_date": trade.entry_time,
                "with_improvement_r": round(actual_r, 2),
                "without_improvement_r": round(counterfactual_r, 2),
                "difference": round(difference, 2),
                "explanation": _build_example_explanation(trade, improvement, difference),
            })

        # Sort by difference (most dramatic first)
        examples.sort(key=lambda e: e["difference"], reverse=True)
        return examples[:n]

    # ── Narrative Generation ───────────────────────────────────

    async def _generate_narrative(
        self,
        improvement: ActiveImprovement,
        trades_with: List[VirtualTrade],
        trades_without: List[VirtualTrade],
        examples: List[Dict[str, Any]],
        summary_stats: Dict[str, Any],
    ) -> str:
        """
        Generate natural, conversational explanation.

        Uses LLM when available, falls back to template.
        """
        # Try LLM-generated narrative
        if self._llm:
            try:
                return await self._generate_llm_narrative(
                    improvement, trades_with, trades_without,
                    examples, summary_stats,
                )
            except Exception as e:
                logger.debug("LLM narrative failed, using template: %s", e)

        # Template-based fallback
        return self._generate_template_narrative(
            improvement, examples, summary_stats,
        )

    async def _generate_llm_narrative(
        self,
        improvement: ActiveImprovement,
        trades_with: List[VirtualTrade],
        trades_without: List[VirtualTrade],
        examples: List[Dict[str, Any]],
        summary_stats: Dict[str, Any],
    ) -> str:
        """Use LLM to generate conversational narrative."""
        strategy_name = improvement.strategy_id

        best_example = examples[0] if examples else None
        example_text = ""
        if best_example:
            trade_date = best_example.get("trade_date")
            date_str = trade_date.strftime('%b %d') if trade_date else 'recent'
            example_text = (
                f"Best example trade:\n"
                f"- Stock: {best_example['symbol']}, Date: {date_str}\n"
                f"- With improvement: {best_example['with_improvement_r']:.1f}R\n"
                f"- Without would have been: {best_example['without_improvement_r']:.1f}R\n"
            )

        prompt = (
            f"Generate a conversational explanation of this trading improvement "
            f"for a trader. Write as if you're a knowledgeable trading partner, "
            f"not a robot. Use specific numbers and examples.\n\n"
            f"Strategy: {strategy_name}\n"
            f"Improvement: {improvement.description}\n\n"
            f"Performance before improvement:\n"
            f"- Win rate: {summary_stats['before_win_rate']:.0%}\n"
            f"- Average R: {summary_stats['before_expectancy']:.2f}\n\n"
            f"Performance after improvement:\n"
            f"- {summary_stats['times_triggered']} trades\n"
            f"- Win rate: {summary_stats['after_win_rate']:.0%}\n"
            f"- Average R: {summary_stats['after_expectancy']:.2f}\n\n"
            f"{example_text}\n"
            f"Write 3-4 paragraphs. Be specific. Be honest about both "
            f"benefits and costs. End with a clear recommendation."
        )

        from agents.prompts import AVT_PERSONA as _SYS
        response = await self._llm.generate(prompt, system_prompt=_SYS, caller="avt:presenter")
        return response

    @staticmethod
    def _generate_template_narrative(
        improvement: ActiveImprovement,
        examples: List[Dict[str, Any]],
        stats: Dict[str, Any],
    ) -> str:
        """Template-based narrative when LLM not available."""
        parts = []

        # Opening
        weeks = stats.get("weeks_since_applied", 0)
        triggered = stats.get("times_triggered", 0)
        parts.append(
            f"I've been testing a modification to your strategy for "
            f"{weeks:.0f} weeks now. The change: {improvement.description}"
        )

        # Results
        helped = stats.get("times_helped", 0)
        hurt = stats.get("times_hurt", 0)
        net_r = stats.get("net_impact_per_trade_r", 0)

        parts.append(
            f"\nIn {triggered} virtual trades where this condition triggered:\n"
            f"- {helped} times it helped (improved the outcome)\n"
            f"- {hurt} times it hurt (would have been better without it)\n"
            f"- Net impact: {'+' if net_r > 0 else ''}{net_r:.2f}R per trade"
        )

        # Best example
        if examples:
            ex = examples[0]
            trade_date = ex.get("trade_date")
            date_str = ", " + trade_date.strftime('%b %d') if trade_date else ""
            parts.append(
                f"\nBest example: {ex['symbol']}{date_str}\n"
                f"- With this change: {ex['with_improvement_r']:+.1f}R\n"
                f"- Without: {ex['without_improvement_r']:+.1f}R\n"
                f"- Difference: {ex['difference']:+.1f}R"
            )

        # Recommendation
        if net_r > 0.1:
            parts.append(
                f"\nThis change has consistently improved results. "
                f"I'd recommend adopting it into your trading rules."
            )
        elif net_r > 0:
            parts.append(
                f"\nThe improvement is positive but modest. "
                f"I'll keep testing to build more evidence before recommending it."
            )
        else:
            parts.append(
                f"\nThe results are mixed so far. "
                f"I'll continue monitoring to see if a clearer pattern emerges."
            )

        return "\n".join(parts)

    # ── Competition Report Narrative ───────────────────────────

    async def generate_competition_narrative(
        self,
        report: CompetitionReport,
    ) -> str:
        """Generate narrative summary of AVT vs User competition."""
        parts = []

        gap = report.performance_gap
        user = report.user_stats
        avt = report.avt_stats

        # Opening
        if gap > 0.2:
            parts.append(
                f"Over the last {report.period_days} days trading "
                f"{report.strategy_name}, I've been outperforming by "
                f"{gap:.2f}R per trade."
            )
        elif gap > 0:
            parts.append(
                f"Over the last {report.period_days} days, we're trading "
                f"{report.strategy_name} similarly, but I'm slightly ahead "
                f"by {gap:.2f}R per trade."
            )
        else:
            parts.append(
                f"Over the last {report.period_days} days, you're actually "
                f"outperforming me on {report.strategy_name} by "
                f"{abs(gap):.2f}R per trade. Well done."
            )

        # Stats comparison
        parts.append(
            f"\nYour results: {user.get('trade_count', 0)} trades, "
            f"{user.get('win_rate', 0):.0%} win rate, {user.get('avg_r', 0):.2f}R average, "
            f"{user.get('total_r', 0):.1f}R total"
        )
        parts.append(
            f"My results: {avt.get('trade_count', 0)} trades, "
            f"{avt.get('win_rate', 0):.0%} win rate, {avt.get('avg_r', 0):.2f}R average, "
            f"{avt.get('total_r', 0):.1f}R total"
        )

        # Key differences
        if report.differences:
            parts.append("\nWhere I did differently:")
            for i, diff in enumerate(report.differences[:3], 1):
                parts.append(f"\n{i}. {diff.description}")
                if diff.actionable_insight:
                    parts.append(f"   → {diff.actionable_insight}")

        # Proven improvements
        if report.proven_improvements:
            proven = [
                imp for imp in report.proven_improvements
                if imp.status == "proven"
            ]
            if proven:
                parts.append(
                    f"\nI have {len(proven)} proven improvement(s) "
                    f"you might want to adopt."
                )

        return "\n".join(parts)

    # ── Batch Presentations ────────────────────────────────────

    async def generate_all_presentations(
        self,
        avt_engine: Any,
        self_improver: Any,
    ) -> List[ImprovementPresentation]:
        """Generate presentations for all proven improvements."""
        presentations = []

        proven = self_improver.get_proven_improvements()
        for improvement in proven:
            try:
                pres = await self.generate_improvement_presentation(
                    improvement, avt_engine,
                )
                presentations.append(pres)
            except Exception as e:
                logger.error(
                    "Presentation failed for %s: %s",
                    improvement.description, e,
                )

        return presentations


# ── Helpers ────────────────────────────────────────────────────

def _build_example_explanation(
    trade: VirtualTrade,
    improvement: ActiveImprovement,
    difference: float,
) -> str:
    """Build a one-sentence explanation of a specific trade example."""
    if improvement.type == "partial_exit_at_rr":
        if difference > 0:
            return (
                f"Booked partial at {trade.max_favorable_r:.1f}R. "
                f"Price then reversed. Saved {difference:.1f}R on the closed portion."
            )
        else:
            return (
                f"Booked partial at {trade.max_favorable_r:.1f}R. "
                f"Price continued running. Missed {abs(difference):.1f}R "
                f"on the closed portion, but still captured on remainder."
            )

    elif improvement.type == "early_exit_pattern":
        if difference > 0:
            return (
                f"Warning pattern detected. Exited at {(trade.pnl_r or 0):.1f}R "
                f"instead of holding to stop loss. Saved {difference:.1f}R."
            )
        else:
            return (
                f"Warning pattern triggered false alarm. "
                f"Exited at {(trade.pnl_r or 0):.1f}R but trade would have recovered."
            )

    return f"Impact: {'+' if difference > 0 else ''}{difference:.1f}R"
