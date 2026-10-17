"""
Shadow Comparison Agent
========================
Compares real trade execution against shadow (rules-based) outcomes.
Surfaces recurring execution gaps that are costing P&L.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session

from agents.base_agent import BaseAgent, AgentInsight
from core.models import Trade
from core.enums import TradeStatus
from shadow.models import ShadowSession, ShadowTrade, TradeDivergence
from strategy_engine.strategy_dna.query_engine import StrategyQueryEngine
import logging

logger = logging.getLogger(__name__)

LOOKBACK_DAYS          = 60
MIN_TRADES_FOR_PATTERN = 5
OPP_COST_THRESHOLD     = 500.0    # ₹ — flag if cumulative opportunity cost exceeds this


class ShadowComparisonAgent(BaseAgent):
    """
    6️⃣ Shadow Comparison Agent

    Learns from the gap between what you did and what you should have done:
    - Per-trade opportunity cost vs shadow variants
    - Chronic early-exit pattern (strategy shadow beats real)
    - Stop placement divergence patterns
    - Cumulative P&L left on the table
    - "What-if" narrative for the current trade
    """

    def __init__(self, strategy_query: StrategyQueryEngine, db: Session):
        super().__init__("shadow_comparison_agent", strategy_query, db)

    def analyze(self, context: Dict) -> List[AgentInsight]:
        insights = []
        trade_id = context.get("trade_id")

        if trade_id:
            insights.extend(self._analyze_single_trade(str(trade_id)))

        insights.extend(self._detect_chronic_early_exit())
        insights.extend(self._detect_stop_placement_drift())
        insights.extend(self._summarize_cumulative_opp_cost())

        return insights

    # ------------------------------------------------------------------

    def _analyze_single_trade(self, trade_id: str) -> List[AgentInsight]:
        """Immediate shadow comparison for a just-closed trade."""
        insights = []

        trade = self.db.query(Trade).filter(Trade.id == trade_id).first()
        if not trade or trade.pnl is None:
            return insights

        session = self.db.query(ShadowSession).filter(
            ShadowSession.real_trade_id == trade.id
        ).first()
        if not session:
            return insights

        shadows = self.db.query(ShadowTrade).filter(
            ShadowTrade.session_id == session.id
        ).all()
        if not shadows:
            return insights

        real_pnl  = trade.pnl
        best      = max(shadows, key=lambda s: s.realized_pnl or 0)
        best_pnl  = best.realized_pnl or 0
        opp_cost  = max(0.0, best_pnl - real_pnl)

        if opp_cost > 100:
            shadow_label = (best.label or best.shadow_type or "best shadow").replace("_", " ")
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="performance",
                severity="warning" if opp_cost < 1000 else "critical",
                title=f"₹{opp_cost:.0f} left on the table vs {shadow_label}",
                message=(
                    f"Real P&L: ₹{real_pnl:.0f}. "
                    f"Best shadow ({shadow_label}): ₹{best_pnl:.0f}. "
                    f"Opportunity cost: ₹{opp_cost:.0f}. "
                    + self._exit_narrative(trade, best)
                ),
                data={
                    "real_pnl": round(real_pnl, 2),
                    "best_shadow_pnl": round(best_pnl, 2),
                    "opportunity_cost": round(opp_cost, 2),
                    "best_shadow_type": best.shadow_type,
                    "trade_id": trade_id,
                },
                trade_ids=[trade_id],
                action_suggestion=self._exit_suggestion(best),
            ))
        elif real_pnl >= best_pnl:
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="performance",
                severity="positive",
                title="Execution beat all shadow variants",
                message=(
                    f"Your exit at ₹{real_pnl:.0f} outperformed all shadow trades. "
                    f"Best shadow: ₹{best_pnl:.0f}. Excellent discretionary execution."
                ),
                data={"real_pnl": round(real_pnl, 2), "best_shadow_pnl": round(best_pnl, 2)},
                trade_ids=[trade_id],
                actionable=False,
            ))

        return insights

    def _detect_chronic_early_exit(self) -> List[AgentInsight]:
        """
        Find trades where the strategy shadow consistently outperforms —
        signature of chronic early exits.
        """
        insights = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

        closed_trades = self.db.query(Trade).filter(
            Trade.status == TradeStatus.CLOSED,
            Trade.entry_time >= cutoff,
            Trade.pnl.isnot(None),
        ).all()

        early_exit_count = 0
        total_opp_cost   = 0.0
        sample_ids: List[str] = []

        for trade in closed_trades:
            session = self.db.query(ShadowSession).filter(
                ShadowSession.real_trade_id == trade.id
            ).first()
            if not session:
                continue

            strategy_shadow = self.db.query(ShadowTrade).filter(
                ShadowTrade.session_id == session.id,
                ShadowTrade.shadow_type == "strategy",
            ).first()
            if not strategy_shadow:
                continue

            shadow_pnl = strategy_shadow.realized_pnl or 0
            real_pnl   = trade.pnl or 0
            if shadow_pnl > real_pnl + 50:
                early_exit_count += 1
                total_opp_cost   += shadow_pnl - real_pnl
                if len(sample_ids) < 5:
                    sample_ids.append(str(trade.id))

        if early_exit_count >= MIN_TRADES_FOR_PATTERN:
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="behavioral",
                severity="critical",
                title=f"Chronic early exit pattern — {early_exit_count} trades in {LOOKBACK_DAYS}d",
                message=(
                    f"In {early_exit_count} trades, the strategy shadow outperformed your exit. "
                    f"Cumulative opportunity cost: ₹{total_opp_cost:.0f}. "
                    f"You are consistently exiting before the strategy target, "
                    f"which may be driven by fear of giving back profits."
                ),
                data={
                    "early_exit_count": early_exit_count,
                    "cumulative_opportunity_cost": round(total_opp_cost, 2),
                    "period_days": LOOKBACK_DAYS,
                    "sample_trade_ids": sample_ids,
                },
                action_suggestion=(
                    "Commit to letting trades reach the first target (T1) before managing. "
                    "Set a reminder alert at T1 instead of exiting early."
                ),
            ))

        return insights

    def _detect_stop_placement_drift(self) -> List[AgentInsight]:
        """Detect systematic difference between real stops and shadow stops."""
        insights = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

        divergences = self.db.query(TradeDivergence).filter(
            TradeDivergence.divergence_type == "stop_placement",
        ).all()

        # Filter to recent ones via join
        recent_divs = []
        for div in divergences:
            trade = self.db.query(Trade).filter(Trade.id == div.real_trade_id).first()
            if trade and trade.entry_time and trade.entry_time >= cutoff:
                recent_divs.append(div)

        if len(recent_divs) < MIN_TRADES_FOR_PATTERN:
            return insights

        # Positive impact_amount = shadow stop is farther away (real stop too tight)
        avg_impact = sum(d.impact_amount or 0 for d in recent_divs) / len(recent_divs)

        if avg_impact > 0.003:
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="behavioral",
                severity="warning",
                title=f"Real stops consistently tighter than strategy rules ({len(recent_divs)} trades)",
                message=(
                    f"Your stops average {avg_impact:.4f} below the strategy-defined level. "
                    f"Tight stops cause premature stop-outs before the trade can work. "
                    f"Trust the strategy's structural stop placement."
                ),
                data={
                    "avg_stop_difference": round(avg_impact, 4),
                    "divergence_count": len(recent_divs),
                },
                action_suggestion="Place stops at the strategy-defined structural level, not tighter.",
            ))
        elif avg_impact < -0.003:
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="behavioral",
                severity="warning",
                title=f"Real stops wider than strategy rules — extra risk ({len(recent_divs)} trades)",
                message=(
                    f"Your stops are {abs(avg_impact):.4f} wider than the strategy level. "
                    f"This increases risk per trade beyond what the strategy assumes."
                ),
                data={
                    "avg_stop_difference": round(avg_impact, 4),
                    "divergence_count": len(recent_divs),
                },
                action_suggestion="Tighten stops to the strategy's structural level.",
            ))

        return insights

    def _summarize_cumulative_opp_cost(self) -> List[AgentInsight]:
        """Summarise total P&L left on the table in the lookback window."""
        insights = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

        closed_trades = self.db.query(Trade).filter(
            Trade.status == TradeStatus.CLOSED,
            Trade.entry_time >= cutoff,
            Trade.pnl.isnot(None),
        ).all()

        total_opp_cost = 0.0
        compared_count = 0

        for trade in closed_trades:
            session = self.db.query(ShadowSession).filter(
                ShadowSession.real_trade_id == trade.id
            ).first()
            if not session:
                continue

            shadows = self.db.query(ShadowTrade).filter(
                ShadowTrade.session_id == session.id
            ).all()
            if not shadows:
                continue

            best_shadow_pnl = max((s.realized_pnl or 0) for s in shadows)
            gap = max(0.0, best_shadow_pnl - (trade.pnl or 0))
            total_opp_cost += gap
            compared_count += 1

        if compared_count < MIN_TRADES_FOR_PATTERN:
            return insights

        if total_opp_cost > OPP_COST_THRESHOLD:
            daily_cost = total_opp_cost / LOOKBACK_DAYS
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="performance",
                severity="warning",
                title=f"₹{total_opp_cost:.0f} cumulative opportunity cost ({LOOKBACK_DAYS}d)",
                message=(
                    f"Across {compared_count} compared trades you left ₹{total_opp_cost:.0f} "
                    f"on the table vs the best shadow variant — "
                    f"averaging ₹{daily_cost:.0f}/day. "
                    f"This gap represents execution quality you can recover through discipline."
                ),
                data={
                    "total_opportunity_cost": round(total_opp_cost, 2),
                    "compared_trades": compared_count,
                    "avg_daily_cost": round(daily_cost, 2),
                    "period_days": LOOKBACK_DAYS,
                },
                action_suggestion="Focus on following the exit plan — the biggest gain is in exits.",
            ))

        return insights

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _exit_narrative(trade: Trade, best_shadow: ShadowTrade) -> str:
        if best_shadow.exit_reason and best_shadow.exit_reason != "real_trade_closed":
            return (
                f"The shadow exited via '{best_shadow.exit_reason}' "
                f"which captured more of the move."
            )
        return "The shadow held longer and captured more of the move."

    @staticmethod
    def _exit_suggestion(best_shadow: ShadowTrade) -> str:
        shadow_type = best_shadow.shadow_type or "strategy"
        if shadow_type == "variant_trail":
            return "Consider using a trailing stop to lock in gains as the trade extends."
        if shadow_type == "strategy":
            return "Follow the strategy exit rules — let it reach the defined target."
        if shadow_type == "variant_wide":
            return "Wider stop may have prevented the premature stop-out."
        return "Let the trade reach its target before exiting."
