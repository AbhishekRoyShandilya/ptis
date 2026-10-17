"""
Strategy Evolver
=================
Analyses strategy performance over time and proposes rule adjustments.
Uses edge analysis + statistical tests to detect when a strategy is:
  - Working well and could be sized up
  - Showing early decay and needs filter tightening
  - Clearly broken and should be paused

Output is structured recommendations — NOT automatic rule changes.
The trader reviews and accepts/rejects each recommendation.

Integration:
  - Called by the /automation/strategy/evolve API endpoint
  - Can be scheduled as a weekly background job
  - Recommendations stored in the knowledge base for future AI context
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EvolutionRecommendation:
    recommendation_id: str
    strategy_id: str
    type: str            # "filter_tighten" | "filter_relax" | "size_up" | "pause" | "maintain"
    priority: str        # "high" | "medium" | "low" | "info"
    title: str
    rationale: str
    specific_action: str
    data_backing: dict = field(default_factory=dict)   # metrics that support this rec
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.recommendation_id,
            "strategy_id": self.strategy_id,
            "type": self.type,
            "priority": self.priority,
            "title": self.title,
            "rationale": self.rationale,
            "specific_action": self.specific_action,
            "data_backing": self.data_backing,
            "created_at": self.created_at,
        }


@dataclass
class EvolutionReport:
    strategy_id: str
    strategy_name: str
    trade_count: int
    lookback_days: int
    win_rate_30d: float
    win_rate_90d: float
    profit_factor: float
    avg_r: float
    edge_stable: bool
    decay_detected: bool
    recommendations: list = field(default_factory=list)   # list[EvolutionRecommendation]
    overall_verdict: str = "maintain"   # "scale_up" | "maintain" | "tighten" | "pause"
    summary: str = ""
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "trade_count": self.trade_count,
            "lookback_days": self.lookback_days,
            "metrics": {
                "win_rate_30d": round(self.win_rate_30d, 4),
                "win_rate_90d": round(self.win_rate_90d, 4),
                "profit_factor": round(self.profit_factor, 3),
                "avg_r": round(self.avg_r, 3),
            },
            "edge_health": {
                "stable": self.edge_stable,
                "decay_detected": self.decay_detected,
            },
            "overall_verdict": self.overall_verdict,
            "summary": self.summary,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "computed_at": self.computed_at,
        }


# ---------------------------------------------------------------------------
# Evolver
# ---------------------------------------------------------------------------

class StrategyEvolver:

    def analyse(self, strategy_id: str, trades: list,
                lookback_days: int = 90) -> EvolutionReport:
        """
        Analyse a strategy's trades and generate evolution recommendations.

        trades: list of Trade model instances attributed to this strategy.
        """
        import uuid
        from analytics.edge_analysis import EdgeAnalyzer

        strategy_name = str(strategy_id)   # fallback; caller can override

        if not trades:
            return EvolutionReport(
                strategy_id=str(strategy_id),
                strategy_name=strategy_name,
                trade_count=0, lookback_days=lookback_days,
                win_rate_30d=0.0, win_rate_90d=0.0, profit_factor=0.0, avg_r=0.0,
                edge_stable=True, decay_detected=False,
                overall_verdict="maintain",
                summary="No trades found for this strategy in the lookback period.",
            )

        edge = EdgeAnalyzer().compute(trades)

        win_rate_30 = edge.rolling_win_rate_30d
        win_rate_90 = edge.rolling_win_rate_90d
        pf          = edge.profit_factor
        avg_r       = edge.overall_expectancy

        recs: list[EvolutionRecommendation] = []
        verdict = "maintain"

        def _rec(rtype, priority, title, rationale, action, data=None):
            recs.append(EvolutionRecommendation(
                recommendation_id=str(uuid.uuid4()),
                strategy_id=str(strategy_id),
                type=rtype, priority=priority,
                title=title, rationale=rationale,
                specific_action=action,
                data_backing=data or {},
            ))

        # --- Decay detection ---
        wr_drop = win_rate_90 - win_rate_30
        if wr_drop > 0.15:
            verdict = "tighten"
            _rec(
                "filter_tighten", "high",
                "Significant Win Rate Drop",
                f"30-day win rate ({win_rate_30:.0%}) has dropped {wr_drop:.0%} below the 90-day baseline ({win_rate_90:.0%}). This is a statistically meaningful decline.",
                "Review the last 10 losing trades. Identify the most common failing condition and add it as a negative filter.",
                {"win_rate_30d": win_rate_30, "win_rate_90d": win_rate_90, "drop": wr_drop},
            )
        elif wr_drop > 0.08:
            _rec(
                "filter_tighten", "medium",
                "Win Rate Trending Down",
                f"30-day win rate ({win_rate_30:.0%}) is moderately below 90-day average ({win_rate_90:.0%}). Monitor closely.",
                "Audit recent entries for pattern drift — are you taking setups outside the strategy's original conditions?",
                {"win_rate_30d": win_rate_30, "win_rate_90d": win_rate_90},
            )

        # --- Profit factor analysis ---
        if pf < 1.0:
            verdict = "pause"
            _rec(
                "pause", "high",
                "Profit Factor Below 1.0 — Strategy Destroying Capital",
                f"Profit factor {pf:.2f} means losses exceed profits. This strategy is net-negative.",
                "Pause this strategy immediately. Paper trade it for 2 weeks with tightened filters before resuming.",
                {"profit_factor": pf},
            )
        elif pf < 1.3:
            if verdict == "maintain":
                verdict = "tighten"
            _rec(
                "filter_tighten", "medium",
                "Low Profit Factor",
                f"Profit factor {pf:.2f} leaves little margin for error (slippage, commissions).",
                "Raise the minimum conviction score threshold or tighten the entry timing window.",
                {"profit_factor": pf},
            )
        elif pf > 2.5 and win_rate_30 >= win_rate_90 * 0.95:
            verdict = "scale_up"
            _rec(
                "size_up", "low",
                "Strong Edge — Consider Modest Size Increase",
                f"Profit factor {pf:.2f} with stable win rate suggests a robust, consistent edge.",
                "Consider increasing position size by 10-15% on A-grade setups only. Review after 20 more trades.",
                {"profit_factor": pf, "win_rate_90d": win_rate_90},
            )

        # --- R-multiple analysis ---
        if avg_r < 0.5 and verdict == "maintain":
            _rec(
                "filter_relax", "medium",
                "Low Average R-Multiple",
                f"Average expectancy {avg_r:.2f}R is below healthy levels. Targets may be too tight or entries too late.",
                "Review your exit discipline — are you cutting winners short? Backtest a wider target.",
                {"avg_r": avg_r},
            )

        # --- Session / time analysis ---
        # Check if late-session trades are dragging down overall performance
        late_trades = [t for t in trades if getattr(t, "entry_time", None) and t.entry_time.hour >= 14]
        if len(late_trades) >= 5:
            late_pnls = [t.pnl for t in late_trades if getattr(t, "pnl", None) is not None]
            if late_pnls:
                late_avg = statistics.mean(late_pnls)
                all_pnls = [t.pnl for t in trades if getattr(t, "pnl", None) is not None]
                overall_avg = statistics.mean(all_pnls) if all_pnls else 0
                if late_avg < overall_avg * 0.5:
                    _rec(
                        "filter_tighten", "low",
                        "Late-Session Underperformance",
                        f"Trades taken after 2 PM average ₹{late_avg:.0f} vs overall ₹{overall_avg:.0f}.",
                        "Consider adding a time filter: no new entries after 14:00.",
                        {"late_avg_pnl": late_avg, "overall_avg_pnl": overall_avg},
                    )

        # --- No issues found ---
        if not recs:
            _rec(
                "maintain", "info",
                "Strategy Performing Within Normal Parameters",
                f"Win rate {win_rate_90:.0%}, profit factor {pf:.2f}, avg R {avg_r:.2f}. No significant issues detected.",
                "Continue with current rules. Schedule next review in 30 days or after 20 more trades.",
                {},
            )

        summary = (
            f"Analysed {len(trades)} trades over {lookback_days} days. "
            f"Win rate: {win_rate_90:.0%} (90d) → {win_rate_30:.0%} (30d). "
            f"PF: {pf:.2f}. Verdict: {verdict.upper()}."
        )

        return EvolutionReport(
            strategy_id=str(strategy_id),
            strategy_name=strategy_name,
            trade_count=len(trades),
            lookback_days=lookback_days,
            win_rate_30d=win_rate_30,
            win_rate_90d=win_rate_90,
            profit_factor=pf if pf < 999 else 99.0,
            avg_r=avg_r,
            edge_stable=not edge.edge_decay_detected,
            decay_detected=edge.edge_decay_detected,
            recommendations=recs,
            overall_verdict=verdict,
            summary=summary,
        )

    def analyse_from_db(self, db, strategy_id: str,
                        lookback_days: int = 90) -> EvolutionReport:
        from core.models import Trade, StrategyAttribution
        from core.enums import TradeStatus
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        # Get trades attributed to this strategy
        attributed = (
            db.query(Trade)
            .join(StrategyAttribution, Trade.id == StrategyAttribution.trade_id)
            .filter(
                StrategyAttribution.strategy_id == strategy_id,
                Trade.status == TradeStatus.CLOSED,
                Trade.entry_time >= cutoff,
            )
            .all()
        )

        # Resolve strategy name
        strategy_name = strategy_id
        if attributed:
            first_attr = next(
                (a for a in attributed[0].strategy_attributions
                 if str(a.strategy_id) == str(strategy_id)), None
            ) if hasattr(attributed[0], "strategy_attributions") else None
            if first_attr and hasattr(first_attr, "strategy_name"):
                strategy_name = first_attr.strategy_name

        report = self.analyse(strategy_id, attributed, lookback_days)
        report.strategy_name = strategy_name

        # Optionally save recommendations to knowledge base
        if report.recommendations:
            try:
                from rag.knowledge_base import KnowledgeBase
                kb = KnowledgeBase(db)
                content = f"Strategy Evolution Report ({report.computed_at[:10]})\n\n"
                content += report.summary + "\n\nRecommendations:\n"
                for r in report.recommendations:
                    content += f"\n[{r.priority.upper()}] {r.title}\n{r.rationale}\nAction: {r.specific_action}\n"
                kb.add_insight(
                    title=f"Strategy Evolution: {strategy_name} — {report.overall_verdict}",
                    content=content,
                    agent_name="StrategyEvolver",
                    tags=["strategy_evolution", f"strategy:{strategy_name}"],
                )
                db.flush()
            except Exception as e:
                logger.warning(f"Failed to save evolution report to knowledge base: {e}")

        return report


_evolver: Optional[StrategyEvolver] = None


def get_strategy_evolver() -> StrategyEvolver:
    global _evolver
    if _evolver is None:
        _evolver = StrategyEvolver()
    return _evolver
