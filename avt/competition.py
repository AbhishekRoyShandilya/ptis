# avt/competition.py
"""
Competition Engine — AVT vs User Performance
==============================================

Tracks and compares AVT performance vs user performance.
Generates evidence-based improvement presentations.

This is the TRUST BUILDER. The user doesn't need to understand
statistics. They see:

  "My system traded the same strategy and made 23% more.
   Here's exactly what it did differently."

CATEGORIES OF DIFFERENCE:
  1. Stock Selection — AVT picked better stocks
  2. Execution Quality — on same stocks, AVT executed better
  3. Learned Improvements — AVT's self-discovered rules
  4. Trade Avoidance — setups user took (and lost on) that AVT skipped
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from avt.models import (
    ActiveImprovement,
    CompetitionReport,
    PerformanceDifference,
    PerformanceSnapshot,
    VirtualTrade,
    compute_performance,
)

logger = logging.getLogger(__name__)


class CompetitionEngine:
    """
    Compares AVT vs User performance and identifies specific
    categories where AVT outperformed.
    """

    def __init__(
        self,
        avt_engine: Any,          # AutonomousVirtualTrader
        self_improver: Any = None, # AVTSelfImprover
        db_session: Any = None,
    ):
        self.avt = avt_engine
        self.self_improver = self_improver
        self._db = db_session

    # ── Main Comparison ────────────────────────────────────────

    async def generate_comparison(
        self,
        strategy_id: str,
        period_days: int = 30,
    ) -> CompetitionReport:
        """
        Generate a full AVT vs User comparison report.

        Fetches user's trades from DB and AVT's virtual trades,
        then analyzes differences across all categories.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)

        # Get user's trades
        user_trades = await self._get_user_trades(strategy_id, cutoff)

        # Get AVT's virtual trades
        avt_trades = self.avt.get_closed_trades(
            strategy_id=strategy_id,
            after_date=cutoff,
        )

        # Compute performance
        user_perf = self._compute_user_performance(user_trades)
        avt_perf = compute_performance(avt_trades)

        # Get strategy name
        strategy_name = self._get_strategy_name(strategy_id)

        # Get version history
        version_history = []
        if self.self_improver:
            version_history = self.self_improver.get_version_history(strategy_id)

        # Get proven improvements
        proven = []
        if self.self_improver:
            proven = self.self_improver.get_proven_improvements(strategy_id)

        # Analyze differences
        differences = await self._analyze_differences(
            user_trades, avt_trades, strategy_id, user_perf, avt_perf,
        )

        return CompetitionReport(
            strategy_name=strategy_name,
            period_days=period_days,
            user_stats={
                "trade_count": user_perf.trade_count,
                "win_rate": user_perf.win_rate,
                "avg_r": user_perf.expectancy_r,
                "total_r": user_perf.total_r,
            },
            avt_stats={
                "trade_count": avt_perf.trade_count,
                "win_rate": avt_perf.win_rate,
                "avg_r": avt_perf.expectancy_r,
                "total_r": avt_perf.total_r,
                "version_history": version_history,
            },
            performance_gap=avt_perf.expectancy_r - user_perf.expectancy_r,
            differences=differences,
            proven_improvements=proven,
        )

    # ── Difference Analysis ────────────────────────────────────

    async def _analyze_differences(
        self,
        user_trades: List[Dict[str, Any]],
        avt_trades: List[VirtualTrade],
        strategy_id: str,
        user_perf: PerformanceSnapshot,
        avt_perf: PerformanceSnapshot,
    ) -> List[PerformanceDifference]:
        """
        Identify specific categories where AVT outperformed.
        Returns differences sorted by impact (largest first).
        """
        differences: List[PerformanceDifference] = []

        stock_diff = self._analyze_stock_selection(user_trades, avt_trades, user_perf)
        if stock_diff:
            differences.append(stock_diff)

        exec_diff = self._analyze_execution_quality(user_trades, avt_trades, user_perf)
        if exec_diff:
            differences.append(exec_diff)

        imp_diffs = self._analyze_improvement_impact(avt_trades, strategy_id)
        differences.extend(imp_diffs)

        avoid_diff = await self._analyze_trade_avoidance(user_trades, strategy_id)
        if avoid_diff:
            differences.append(avoid_diff)

        differences.sort(key=lambda d: abs(d.impact_r), reverse=True)
        return differences

    # ── Category 1: Stock Selection ────────────────────────────

    def _analyze_stock_selection(
        self,
        user_trades: List[Dict[str, Any]],
        avt_trades: List[VirtualTrade],
        user_perf: PerformanceSnapshot,
    ) -> Optional[PerformanceDifference]:
        """Did AVT pick better stocks?"""
        user_symbols = set(t.get("symbol", "") for t in user_trades)
        avt_symbols = set(t.symbol for t in avt_trades)

        avt_only = avt_symbols - user_symbols
        if not avt_only:
            return None

        avt_exclusive = [t for t in avt_trades if t.symbol in avt_only]
        if not avt_exclusive:
            return None

        avt_exclusive_perf = compute_performance(avt_exclusive)
        if avt_exclusive_perf.expectancy_r <= user_perf.expectancy_r:
            return None

        impact = avt_exclusive_perf.expectancy_r - user_perf.expectancy_r

        # Top stocks by avg R
        stock_rs: Dict[str, List[float]] = {}
        for t in avt_exclusive:
            stock_rs.setdefault(t.symbol, []).append(t.pnl_r or 0)
        top_stocks = sorted(
            stock_rs.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True,
        )[:3]
        top_stock_names = [s[0] for s in top_stocks]

        return PerformanceDifference(
            category="stock_selection",
            impact_r=round(impact, 3),
            description=(
                f"AVT traded {len(avt_only)} stocks you didn't see. "
                f"These averaged {avt_exclusive_perf.expectancy_r:.2f}R "
                f"vs your overall {user_perf.expectancy_r:.2f}R. "
                f"Top picks: {', '.join(top_stock_names)}"
            ),
            actionable_insight=(
                "AVT picked stocks with higher RVOL and stronger sector "
                "alignment. Consider adding these screening criteria "
                "to your watchlist."
            ),
        )

    # ── Category 2: Execution Quality ──────────────────────────

    def _analyze_execution_quality(
        self,
        user_trades: List[Dict[str, Any]],
        avt_trades: List[VirtualTrade],
        user_perf: PerformanceSnapshot,
    ) -> Optional[PerformanceDifference]:
        """When both traded the same stock, did AVT execute better?"""
        user_symbols = set(t.get("symbol", "") for t in user_trades)
        avt_symbols = set(t.symbol for t in avt_trades)
        common = user_symbols & avt_symbols

        if not common:
            return None

        user_common = [t for t in user_trades if t.get("symbol", "") in common]
        avt_common = [t for t in avt_trades if t.symbol in common]

        user_common_perf = self._compute_user_performance(user_common)
        avt_common_perf = compute_performance(avt_common)

        gap = avt_common_perf.expectancy_r - user_common_perf.expectancy_r
        if gap <= 0.10:
            return None

        analysis = self._diagnose_execution_gap(user_common, avt_common)

        return PerformanceDifference(
            category="execution_quality",
            impact_r=round(gap, 3),
            description=(
                f"On {len(common)} stocks you both traded, AVT averaged "
                f"{avt_common_perf.expectancy_r:.2f}R vs your "
                f"{user_common_perf.expectancy_r:.2f}R. "
                f"{analysis['description']}"
            ),
            actionable_insight=analysis.get("insight", ""),
        )

    @staticmethod
    def _diagnose_execution_gap(
        user_trades: List[Dict[str, Any]],
        avt_trades: List[VirtualTrade],
    ) -> Dict[str, Any]:
        """Diagnose WHY execution differs on same stocks."""
        reasons = []

        user_rs = [t.get("r_multiple", 0) for t in user_trades if t.get("r_multiple")]
        avt_rs = [t.pnl_r or 0 for t in avt_trades]

        user_sl_rate = sum(1 for r in user_rs if r <= -0.9) / max(len(user_rs), 1)
        avt_sl_rate = sum(1 for r in avt_rs if r <= -0.9) / max(len(avt_rs), 1)
        if user_sl_rate > avt_sl_rate + 0.10:
            reasons.append("more stop losses hit")

        user_winners = [r for r in user_rs if r > 0]
        avt_winners = [r for r in avt_rs if r > 0]
        user_avg_win = sum(user_winners) / max(len(user_winners), 1)
        avt_avg_win = sum(avt_winners) / max(len(avt_winners), 1)
        if avt_avg_win > user_avg_win + 0.2:
            reasons.append("smaller average winners (possible premature exits)")

        if "more stop losses hit" in reasons:
            insight = "Your entries may be less precise. AVT waits for higher conviction signals."
        elif "smaller average winners" in reasons:
            insight = "You may be exiting winners too early. AVT follows strategy exit rules strictly."
        else:
            insight = "AVT's disciplined execution captures more R per trade."

        return {
            "description": f"Key differences: {', '.join(reasons) or 'general execution quality'}.",
            "insight": insight,
        }

    # ── Category 3: Improvement Impact ─────────────────────────

    def _analyze_improvement_impact(
        self,
        avt_trades: List[VirtualTrade],
        strategy_id: str,
    ) -> List[PerformanceDifference]:
        """How much did AVT's learned improvements contribute?"""
        differences: List[PerformanceDifference] = []

        if not self.self_improver:
            return differences

        active_improvements = self.self_improver.get_active_improvements(strategy_id)

        for improvement in active_improvements:
            if improvement.status != "proven":
                continue

            imp_trades = self.avt.get_trades_where_improvement_applied(str(improvement.id))
            if not imp_trades:
                continue

            perf_before = improvement.performance_before
            perf_after = improvement.performance_after
            if not perf_before or not perf_after:
                continue

            before_exp = perf_before.get("expectancy_r", 0) if isinstance(perf_before, dict) \
                else getattr(perf_before, "expectancy_r", 0)
            after_exp = perf_after.get("expectancy_r", 0) if isinstance(perf_after, dict) \
                else getattr(perf_after, "expectancy_r", 0)
            per_trade_r = after_exp - before_exp

            differences.append(PerformanceDifference(
                category="learned_improvement",
                impact_r=round(per_trade_r, 3),
                description=(
                    f"Improvement: {improvement.description}\n"
                    f"Applied {len(imp_trades)} times. "
                    f"Average impact: +{per_trade_r:.2f}R per trade.\n"
                    f"Applied since: {improvement.applied_at.strftime('%b %d') if improvement.applied_at else 'recently'}"
                ),
                actionable_insight=(
                    f"This improvement has been proven over "
                    f"{len(imp_trades)} virtual trades. "
                    f"Would you like to adopt it?"
                ),
                improvement_id=improvement.id,
            ))

        return differences

    # ── Category 4: Trade Avoidance ────────────────────────────

    async def _analyze_trade_avoidance(
        self,
        user_trades: List[Dict[str, Any]],
        strategy_id: str,
    ) -> Optional[PerformanceDifference]:
        """Did AVT skip setups the user took (and lost on)?"""
        user_losers = [t for t in user_trades if t.get("r_multiple", 0) < 0]
        if not user_losers:
            return None

        skip_matches = []
        skipped = self.avt.get_skipped_setups()

        for trade in user_losers:
            symbol = trade.get("symbol", "")
            entry_time = trade.get("entry_time")
            if not symbol or not entry_time:
                continue

            for skip in skipped:
                if skip.symbol != symbol or skip.strategy_id != strategy_id:
                    continue
                if skip.timestamp and entry_time:
                    time_diff = abs((skip.timestamp - entry_time).total_seconds())
                    if time_diff < 1800:
                        skip_matches.append({
                            "user_trade": trade,
                            "skip_reason": skip.skip_reason,
                            "avt_conviction": skip.conviction,
                        })
                        break

        if not skip_matches:
            return None

        total_savings = sum(
            abs(m["user_trade"].get("r_multiple", 0)) for m in skip_matches
        )

        reason_counts: Dict[str, int] = {}
        for m in skip_matches:
            r = m["skip_reason"]
            reason_counts[r] = reason_counts.get(r, 0) + 1
        reason_str = ", ".join(
            f"{r} ({c}x)" for r, c in sorted(reason_counts.items(), key=lambda x: -x[1])
        )

        return PerformanceDifference(
            category="trade_avoidance",
            impact_r=round(total_savings / max(len(user_trades), 1), 3),
            description=(
                f"AVT skipped {len(skip_matches)} setups you took "
                f"(and lost on). Reasons: {reason_str}. "
                f"Savings: {total_savings:.1f}R total."
            ),
            actionable_insight=(
                "AVT uses stricter conviction thresholds and stock "
                "quality filters. Your losses often came from setups "
                "that scored below AVT's minimum threshold."
            ),
        )

    # ── Data Access ────────────────────────────────────────────

    async def _get_user_trades(
        self, strategy_id: str, since: datetime,
    ) -> List[Dict[str, Any]]:
        """Fetch user's actual trades from DB."""
        if not self._db:
            return []

        try:
            from sqlalchemy import select
            from core.models import Trade, StrategyAttribution

            stmt = (
                select(Trade)
                .join(StrategyAttribution, Trade.id == StrategyAttribution.trade_id)
                .where(
                    StrategyAttribution.resolved_strategy_id == strategy_id,
                    Trade.entry_time >= since,
                    Trade.status == "CLOSED",
                )
                .order_by(Trade.entry_time)
            )

            result = self._db.execute(stmt)
            trades = result.scalars().all()

            return [
                {
                    "id": str(t.id),
                    "symbol": t.symbol,
                    "direction": t.direction.value if hasattr(t.direction, "value") else str(t.direction),
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": float(t.entry_price) if t.entry_price else 0,
                    "exit_price": float(t.exit_price) if t.exit_price else 0,
                    "pnl": float(t.pnl) if t.pnl else 0,
                    "r_multiple": float(t.r_multiple) if t.r_multiple else 0,
                    "stop_loss": float(t.stop_loss) if t.stop_loss else 0,
                }
                for t in trades
            ]
        except Exception as e:
            logger.warning("Could not fetch user trades: %s", e)
            return []

    @staticmethod
    def _compute_user_performance(trades: List[Dict[str, Any]]) -> PerformanceSnapshot:
        """Compute performance from user trade dicts."""
        if not trades:
            return PerformanceSnapshot()

        rs = [t.get("r_multiple", 0) for t in trades if t.get("r_multiple") is not None]
        if not rs:
            return PerformanceSnapshot()

        winners = [r for r in rs if r > 0]
        losers = [r for r in rs if r <= 0]
        win_r = sum(winners) if winners else 0.0
        loss_r = abs(sum(losers)) if losers else 0.0

        return PerformanceSnapshot(
            trade_count=len(rs),
            win_count=len(winners),
            loss_count=len(losers),
            win_rate=len(winners) / len(rs) if rs else 0.0,
            expectancy_r=sum(rs) / len(rs) if rs else 0.0,
            total_r=sum(rs),
            avg_winner_r=win_r / len(winners) if winners else 0.0,
            avg_loser_r=loss_r / len(losers) if losers else 0.0,
            profit_factor=win_r / loss_r if loss_r > 0 else (
                float("inf") if win_r > 0 else 0.0
            ),
        )

    def _get_strategy_name(self, strategy_id: str) -> str:
        sko = self.avt.strategies.get(strategy_id, {})
        return sko.get("name", strategy_id)

    # ── Summary ────────────────────────────────────────────────

    async def generate_summary(self, period_days: int = 30) -> Dict[str, Any]:
        """High-level summary across all strategies."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
        all_avt_trades = self.avt.get_closed_trades(after_date=cutoff)
        avt_perf = compute_performance(all_avt_trades)

        strategies_compared = []
        for strategy_id in self.avt.strategies:
            strat_trades = self.avt.get_closed_trades(
                strategy_id=strategy_id, after_date=cutoff,
            )
            if strat_trades:
                sp = compute_performance(strat_trades)
                strategies_compared.append({
                    "strategy_id": strategy_id,
                    "strategy_name": self._get_strategy_name(strategy_id),
                    "avt_trades": sp.trade_count,
                    "avt_win_rate": round(sp.win_rate, 3),
                    "avt_expectancy_r": round(sp.expectancy_r, 3),
                    "avt_total_r": round(sp.total_r, 2),
                })

        proven_count = 0
        if self.self_improver:
            proven_count = len(self.self_improver.get_proven_improvements())

        return {
            "period_days": period_days,
            "avt_total_trades": avt_perf.trade_count,
            "avt_overall_win_rate": round(avt_perf.win_rate, 3),
            "avt_overall_expectancy_r": round(avt_perf.expectancy_r, 3),
            "avt_total_r": round(avt_perf.total_r, 2),
            "strategies_traded": len(strategies_compared),
            "proven_improvements": proven_count,
            "strategies": strategies_compared,
            "portfolio": self.avt.get_portfolio_summary(),
        }
