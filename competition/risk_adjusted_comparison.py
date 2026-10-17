# competition/risk_adjusted_comparison.py
"""
Weekly Risk-Adjusted P&L Comparison
====================================
Goes beyond raw P&L to compare risk-adjusted performance metrics.

Raw P&L can be misleading — a trader taking 2x the risk will naturally
have higher P&L. This module normalizes for risk to show TRUE alpha.

Metrics computed:
- Sharpe Ratio (annualized)
- Sortino Ratio (downside deviation only)
- Calmar Ratio (return / max drawdown)
- Max Drawdown (peak-to-trough)
- Risk-Adjusted Return (P&L / total risk taken)
- Win/Loss Asymmetry (avg win size vs avg loss size)
- Kelly Criterion optimal sizing
- Ulcer Index (drawdown severity measure)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session as DBSession

from core.models import Trade
from shadow.models import ShadowTrade

logger = logging.getLogger(__name__)

# Annualization factor (approx. 252 trading days)
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_DAILY = 0.06 / TRADING_DAYS_PER_YEAR  # 6% annual → daily


@dataclass
class RiskMetrics:
    """Risk-adjusted performance metrics for one participant."""
    participant: str  # "trader" | "system"

    # Returns
    total_return_pct: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    avg_daily_return: float = 0.0

    # Risk measures
    volatility_daily: float = 0.0
    volatility_annual: float = 0.0
    downside_deviation: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0
    drawdown_duration_days: int = 0
    ulcer_index: float = 0.0

    # Risk-adjusted ratios
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Position sizing
    avg_risk_per_trade_pct: float = 0.0
    max_risk_per_trade_pct: float = 0.0
    kelly_criterion: float = 0.0

    # Win/Loss quality
    avg_win_size: float = 0.0
    avg_loss_size: float = 0.0
    win_loss_ratio: float = 0.0  # avg_win / avg_loss
    payoff_ratio: float = 0.0   # same as win_loss_ratio but in R terms
    expectancy_per_trade: float = 0.0

    # Risk-adjusted P&L
    risk_adjusted_return: float = 0.0  # total P&L / total risk taken

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant": self.participant,
            "returns": {
                "total_return_pct": round(self.total_return_pct, 4),
                "avg_daily_return": round(self.avg_daily_return, 6),
            },
            "risk": {
                "volatility_daily": round(self.volatility_daily, 6),
                "volatility_annual": round(self.volatility_annual, 4),
                "downside_deviation": round(self.downside_deviation, 6),
                "max_drawdown_pct": round(self.max_drawdown_pct, 4),
                "max_drawdown_abs": round(self.max_drawdown_abs, 2),
                "drawdown_duration_days": self.drawdown_duration_days,
                "ulcer_index": round(self.ulcer_index, 4),
            },
            "ratios": {
                "sharpe_ratio": round(self.sharpe_ratio, 4),
                "sortino_ratio": round(self.sortino_ratio, 4),
                "calmar_ratio": round(self.calmar_ratio, 4),
            },
            "position_sizing": {
                "avg_risk_per_trade_pct": round(self.avg_risk_per_trade_pct, 4),
                "max_risk_per_trade_pct": round(self.max_risk_per_trade_pct, 4),
                "kelly_criterion": round(self.kelly_criterion, 4),
            },
            "quality": {
                "avg_win_size": round(self.avg_win_size, 2),
                "avg_loss_size": round(self.avg_loss_size, 2),
                "win_loss_ratio": round(self.win_loss_ratio, 4),
                "payoff_ratio": round(self.payoff_ratio, 4),
                "expectancy_per_trade": round(self.expectancy_per_trade, 4),
            },
            "risk_adjusted_return": round(self.risk_adjusted_return, 4),
        }


@dataclass
class RiskComparisonReport:
    """Full risk-adjusted comparison report."""
    period_start: datetime
    period_end: datetime
    capital_base: float  # assumed capital for % calculations

    trader_metrics: Optional[RiskMetrics] = None
    system_metrics: Optional[RiskMetrics] = None

    # Head-to-head on risk-adjusted basis
    sharpe_winner: str = "tie"
    sortino_winner: str = "tie"
    drawdown_winner: str = "tie"  # who had less drawdown
    risk_efficiency_winner: str = "tie"

    # Edge analysis
    trader_edge: List[str] = field(default_factory=list)  # Areas where trader is better
    system_edge: List[str] = field(default_factory=list)  # Areas where system is better

    # Coaching insights
    insights: List[str] = field(default_factory=list)
    risk_grade: str = "C"  # A-F grade for risk management

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
                "capital_base": self.capital_base,
            },
            "trader": self.trader_metrics.to_dict() if self.trader_metrics else {},
            "system": self.system_metrics.to_dict() if self.system_metrics else {},
            "winners": {
                "sharpe": self.sharpe_winner,
                "sortino": self.sortino_winner,
                "drawdown": self.drawdown_winner,
                "risk_efficiency": self.risk_efficiency_winner,
            },
            "edges": {
                "trader_strengths": self.trader_edge,
                "system_strengths": self.system_edge,
            },
            "insights": self.insights,
            "risk_grade": self.risk_grade,
        }


class RiskAdjustedComparator:
    """
    Computes and compares risk-adjusted performance metrics.

    Usage:
        comparator = RiskAdjustedComparator(db, capital=500000)
        report = comparator.weekly_comparison()
    """

    def __init__(self, db: DBSession, capital: float = 500_000.0):
        self._db = db
        self._capital = capital

    def weekly_comparison(self, week_offset: int = 0) -> RiskComparisonReport:
        """Risk-adjusted comparison for a specific week."""
        today = datetime.now(timezone.utc).date()
        start_of_week = today - timedelta(days=today.weekday()) + timedelta(weeks=week_offset)
        end_of_week = start_of_week + timedelta(days=6)

        start = datetime.combine(start_of_week, datetime.min.time()).replace(tzinfo=timezone.utc)
        end = datetime.combine(end_of_week, datetime.max.time()).replace(tzinfo=timezone.utc)

        return self.compare_period(start, end)

    def compare_period(
        self, start: datetime, end: datetime
    ) -> RiskComparisonReport:
        """Full risk-adjusted comparison for any period."""
        # Fetch data
        trader_trades = (
            self._db.query(Trade)
            .filter(Trade.entry_time >= start, Trade.entry_time <= end)
            .filter(Trade.pnl.isnot(None))
            .order_by(Trade.exit_time)
            .all()
        )

        system_shadows = (
            self._db.query(ShadowTrade)
            .filter(
                ShadowTrade.shadow_type == "autonomous",
                ShadowTrade.entry_time >= start,
                ShadowTrade.entry_time <= end,
                ShadowTrade.status.in_(["closed", "stopped", "target_hit"]),
            )
            .order_by(ShadowTrade.exit_time)
            .all()
        )

        report = RiskComparisonReport(
            period_start=start,
            period_end=end,
            capital_base=self._capital,
        )

        # Compute metrics
        trader_pnls = [t.pnl or 0 for t in trader_trades]
        trader_r_mults = [t.r_multiple for t in trader_trades if t.r_multiple is not None]
        trader_risks = [t.risk_amount or 0 for t in trader_trades]

        system_pnls = []
        system_r_mults = []
        system_risks = []
        for s in system_shadows:
            config = s.config or {}
            pnl = config.get("journal_pnl", s.realized_pnl or 0)
            r = config.get("journal_r_multiple", s.r_multiple)
            system_pnls.append(pnl if pnl is not None else 0)
            if r is not None:
                system_r_mults.append(r)
            system_risks.append(s.risk_amount or 0)

        # Build daily P&L series
        trader_daily = self._build_daily_returns(trader_trades, start, end, is_shadow=False)
        system_daily = self._build_daily_returns(system_shadows, start, end, is_shadow=True)

        report.trader_metrics = self._compute_risk_metrics(
            "trader", trader_pnls, trader_r_mults, trader_risks, trader_daily
        )
        report.system_metrics = self._compute_risk_metrics(
            "system", system_pnls, system_r_mults, system_risks, system_daily
        )

        # Determine winners
        tm = report.trader_metrics
        sm = report.system_metrics

        report.sharpe_winner = "trader" if tm.sharpe_ratio > sm.sharpe_ratio else "system" if sm.sharpe_ratio > tm.sharpe_ratio else "tie"
        report.sortino_winner = "trader" if tm.sortino_ratio > sm.sortino_ratio else "system" if sm.sortino_ratio > tm.sortino_ratio else "tie"
        report.drawdown_winner = "trader" if abs(tm.max_drawdown_pct) < abs(sm.max_drawdown_pct) else "system" if abs(sm.max_drawdown_pct) < abs(tm.max_drawdown_pct) else "tie"
        report.risk_efficiency_winner = "trader" if tm.risk_adjusted_return > sm.risk_adjusted_return else "system" if sm.risk_adjusted_return > tm.risk_adjusted_return else "tie"

        # Edge analysis and insights
        self._analyze_edges(report)
        self._generate_risk_insights(report)
        self._compute_risk_grade(report)

        return report

    def _build_daily_returns(
        self, trades, start: datetime, end: datetime, is_shadow: bool = False
    ) -> List[float]:
        """Build daily P&L return series."""
        daily_pnl: Dict[str, float] = defaultdict(float)

        for t in trades:
            exit_time = t.exit_time
            if exit_time:
                day = exit_time.strftime("%Y-%m-%d")
                if is_shadow:
                    config = getattr(t, "config", {}) or {}
                    pnl = config.get("journal_pnl", t.realized_pnl or 0)
                else:
                    pnl = t.pnl or 0
                daily_pnl[day] += pnl if pnl is not None else 0

        # Build ordered return series
        returns = []
        current = start.date()
        end_date = end.date()
        while current <= end_date:
            if current.weekday() < 5:  # Trading days only
                day_str = current.strftime("%Y-%m-%d")
                daily_return = daily_pnl.get(day_str, 0) / self._capital
                returns.append(daily_return)
            current += timedelta(days=1)

        return returns

    def _compute_risk_metrics(
        self,
        participant: str,
        pnls: List[float],
        r_multiples: List[float],
        risk_amounts: List[float],
        daily_returns: List[float],
    ) -> RiskMetrics:
        """Compute comprehensive risk metrics."""
        metrics = RiskMetrics(participant=participant)

        if not pnls:
            return metrics

        total_pnl = sum(pnls)
        metrics.total_return_pct = total_pnl / self._capital
        metrics.daily_returns = daily_returns
        metrics.avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0

        # Volatility
        if len(daily_returns) > 1:
            mean = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            metrics.volatility_daily = math.sqrt(variance)
            metrics.volatility_annual = metrics.volatility_daily * math.sqrt(TRADING_DAYS_PER_YEAR)

            # Downside deviation (only negative returns)
            downside_returns = [r for r in daily_returns if r < 0]
            if downside_returns:
                down_var = sum(r ** 2 for r in downside_returns) / len(daily_returns)
                metrics.downside_deviation = math.sqrt(down_var)

        # Max drawdown
        dd_pct, dd_abs, dd_days = self._compute_drawdown(daily_returns)
        metrics.max_drawdown_pct = dd_pct
        metrics.max_drawdown_abs = dd_abs * self._capital
        metrics.drawdown_duration_days = dd_days

        # Ulcer index
        metrics.ulcer_index = self._compute_ulcer_index(daily_returns)

        # Sharpe Ratio
        if metrics.volatility_daily > 0:
            excess_return = metrics.avg_daily_return - RISK_FREE_RATE_DAILY
            metrics.sharpe_ratio = (excess_return / metrics.volatility_daily) * math.sqrt(TRADING_DAYS_PER_YEAR)

        # Sortino Ratio
        if metrics.downside_deviation > 0:
            excess_return = metrics.avg_daily_return - RISK_FREE_RATE_DAILY
            metrics.sortino_ratio = (excess_return / metrics.downside_deviation) * math.sqrt(TRADING_DAYS_PER_YEAR)

        # Calmar Ratio
        if abs(metrics.max_drawdown_pct) > 0:
            annualized_return = metrics.total_return_pct * (TRADING_DAYS_PER_YEAR / max(1, len(daily_returns)))
            metrics.calmar_ratio = annualized_return / abs(metrics.max_drawdown_pct)

        # Win/Loss analysis
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        metrics.avg_win_size = sum(wins) / len(wins) if wins else 0
        metrics.avg_loss_size = abs(sum(losses) / len(losses)) if losses else 0
        metrics.win_loss_ratio = metrics.avg_win_size / metrics.avg_loss_size if metrics.avg_loss_size > 0 else float("inf")

        # R-multiple analysis
        if r_multiples:
            r_wins = [r for r in r_multiples if r > 0]
            r_losses = [abs(r) for r in r_multiples if r <= 0]
            avg_win_r = sum(r_wins) / len(r_wins) if r_wins else 0
            avg_loss_r = sum(r_losses) / len(r_losses) if r_losses else 1
            metrics.payoff_ratio = avg_win_r / avg_loss_r if avg_loss_r > 0 else float("inf")

            win_rate = len(wins) / len(pnls)
            metrics.expectancy_per_trade = win_rate * avg_win_r - (1 - win_rate) * avg_loss_r

        # Kelly Criterion
        win_rate = len(wins) / len(pnls) if pnls else 0
        if metrics.win_loss_ratio > 0 and win_rate > 0:
            metrics.kelly_criterion = win_rate - ((1 - win_rate) / metrics.win_loss_ratio)
            metrics.kelly_criterion = max(0, metrics.kelly_criterion)  # Floor at 0

        # Risk sizing
        if risk_amounts:
            nonzero_risks = [r for r in risk_amounts if r > 0]
            if nonzero_risks:
                metrics.avg_risk_per_trade_pct = (sum(nonzero_risks) / len(nonzero_risks)) / self._capital
                metrics.max_risk_per_trade_pct = max(nonzero_risks) / self._capital

        # Risk-adjusted return
        total_risk = sum(abs(r) for r in risk_amounts if r > 0)
        if total_risk > 0:
            metrics.risk_adjusted_return = total_pnl / total_risk

        return metrics

    @staticmethod
    def _compute_drawdown(daily_returns: List[float]) -> tuple:
        """
        Compute max drawdown from daily return series.
        Returns: (max_dd_pct, max_dd_abs, max_dd_duration_days)
        """
        if not daily_returns:
            return 0.0, 0.0, 0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        dd_start = 0
        max_dd_duration = 0
        current_dd_start = 0
        in_drawdown = False

        for i, r in enumerate(daily_returns):
            cumulative += r
            if cumulative > peak:
                peak = cumulative
                if in_drawdown:
                    duration = i - current_dd_start
                    max_dd_duration = max(max_dd_duration, duration)
                in_drawdown = False
            else:
                dd = peak - cumulative
                if dd > max_dd:
                    max_dd = dd
                if not in_drawdown:
                    current_dd_start = i
                    in_drawdown = True

        if in_drawdown:
            max_dd_duration = max(max_dd_duration, len(daily_returns) - current_dd_start)

        return max_dd, max_dd, max_dd_duration

    @staticmethod
    def _compute_ulcer_index(daily_returns: List[float]) -> float:
        """Compute Ulcer Index — measures drawdown severity."""
        if not daily_returns:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        dd_squared_sum = 0.0

        for r in daily_returns:
            cumulative += r
            if cumulative > peak:
                peak = cumulative
            dd_pct = ((peak - cumulative) / max(abs(peak), 0.001)) * 100
            dd_squared_sum += dd_pct ** 2

        return math.sqrt(dd_squared_sum / len(daily_returns))

    def _analyze_edges(self, report: RiskComparisonReport) -> None:
        """Identify specific edges for trader vs system."""
        tm = report.trader_metrics
        sm = report.system_metrics

        if not tm or not sm:
            return

        # Trader edges
        if tm.sharpe_ratio > sm.sharpe_ratio + 0.2:
            report.trader_edge.append("Higher risk-adjusted returns (Sharpe)")
        if abs(tm.max_drawdown_pct) < abs(sm.max_drawdown_pct) * 0.8:
            report.trader_edge.append("Tighter risk control (smaller drawdowns)")
        if tm.win_loss_ratio > sm.win_loss_ratio * 1.2:
            report.trader_edge.append("Better payoff asymmetry (bigger wins vs losses)")
        if tm.expectancy_per_trade > sm.expectancy_per_trade:
            report.trader_edge.append("Higher per-trade expectancy")

        # System edges
        if sm.sharpe_ratio > tm.sharpe_ratio + 0.2:
            report.system_edge.append("Higher risk-adjusted returns (Sharpe)")
        if abs(sm.max_drawdown_pct) < abs(tm.max_drawdown_pct) * 0.8:
            report.system_edge.append("Tighter risk control (smaller drawdowns)")
        if sm.win_loss_ratio > tm.win_loss_ratio * 1.2:
            report.system_edge.append("Better payoff asymmetry (bigger wins vs losses)")
        if sm.expectancy_per_trade > tm.expectancy_per_trade:
            report.system_edge.append("Higher per-trade expectancy")

    def _generate_risk_insights(self, report: RiskComparisonReport) -> None:
        """Generate risk-focused coaching insights."""
        tm = report.trader_metrics
        sm = report.system_metrics

        if not tm or not sm:
            return

        insights = []

        # Sharpe comparison
        if tm.sharpe_ratio != 0 or sm.sharpe_ratio != 0:
            insights.append(
                f"📊 Sharpe Ratio: You={tm.sharpe_ratio:.2f} vs System={sm.sharpe_ratio:.2f} — "
                f"{'your risk-adjusted returns are stronger' if tm.sharpe_ratio > sm.sharpe_ratio else 'system generated better risk-adjusted returns'}"
            )

        # Drawdown insight
        if tm.max_drawdown_pct > 0.05:
            insights.append(
                f"📉 Max Drawdown: {tm.max_drawdown_pct:.1%} (₹{tm.max_drawdown_abs:,.0f}) "
                f"lasting {tm.drawdown_duration_days} days — "
                f"{'within acceptable limits' if tm.max_drawdown_pct < 0.10 else 'consider reducing position sizes'}"
            )

        # Kelly insight
        if tm.kelly_criterion > 0:
            half_kelly = tm.kelly_criterion / 2
            insights.append(
                f"🎯 Kelly optimal: {tm.kelly_criterion:.1%} of capital per trade "
                f"(half-Kelly recommendation: {half_kelly:.1%}). "
                f"Current avg risk: {tm.avg_risk_per_trade_pct:.2%}"
            )

        # Win/loss quality
        if tm.win_loss_ratio > 0 and sm.win_loss_ratio > 0:
            insights.append(
                f"⚖️ Payoff ratio: You={tm.win_loss_ratio:.2f}x vs System={sm.win_loss_ratio:.2f}x — "
                f"{'your winners are proportionally bigger' if tm.win_loss_ratio > sm.win_loss_ratio else 'system captures better risk/reward'}"
            )

        # Expectancy
        if tm.expectancy_per_trade != 0:
            insights.append(
                f"💡 Expectancy: {tm.expectancy_per_trade:+.2f}R per trade — "
                f"{'positive edge, keep trading' if tm.expectancy_per_trade > 0 else '⚠️ negative expectancy, review strategy selection'}"
            )

        report.insights = insights

    def _compute_risk_grade(self, report: RiskComparisonReport) -> None:
        """Assign a letter grade for overall risk management."""
        tm = report.trader_metrics
        if not tm:
            report.risk_grade = "N/A"
            return

        score = 0

        # Sharpe contribution (0-25)
        if tm.sharpe_ratio >= 2.0:
            score += 25
        elif tm.sharpe_ratio >= 1.0:
            score += 20
        elif tm.sharpe_ratio >= 0.5:
            score += 15
        elif tm.sharpe_ratio >= 0:
            score += 10

        # Drawdown contribution (0-25)
        if abs(tm.max_drawdown_pct) <= 0.03:
            score += 25
        elif abs(tm.max_drawdown_pct) <= 0.05:
            score += 20
        elif abs(tm.max_drawdown_pct) <= 0.10:
            score += 15
        elif abs(tm.max_drawdown_pct) <= 0.15:
            score += 10

        # Win/loss ratio contribution (0-25)
        if tm.win_loss_ratio >= 2.0:
            score += 25
        elif tm.win_loss_ratio >= 1.5:
            score += 20
        elif tm.win_loss_ratio >= 1.0:
            score += 15
        elif tm.win_loss_ratio >= 0.5:
            score += 10

        # Expectancy contribution (0-25)
        if tm.expectancy_per_trade >= 0.5:
            score += 25
        elif tm.expectancy_per_trade >= 0.2:
            score += 20
        elif tm.expectancy_per_trade >= 0:
            score += 15

        if score >= 85:
            report.risk_grade = "A"
        elif score >= 70:
            report.risk_grade = "B"
        elif score >= 55:
            report.risk_grade = "C"
        elif score >= 40:
            report.risk_grade = "D"
        else:
            report.risk_grade = "F"
