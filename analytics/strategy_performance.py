# analytics/strategy_performance.py
"""
Per-strategy performance tracking and edge monitoring.
Detects when a strategy's edge is degrading, strengthening,
or varies by regime.

Used by:
- Deviation Attribution Engine (is strategy losing edge?)
- Strategy Discovery Engine (which regimes need coverage?)
- Coaching Agent (when to suggest strategy changes)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session as DBSession

from core.enums import MarketRegime, SessionType, TradeStatus
from core.models import StrategyAttribution, Trade

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Performance metrics for a single strategy."""
    strategy_id: str
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0

    # P&L
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0

    # R-multiples
    avg_r: float = 0.0
    median_r: float = 0.0
    max_r: float = 0.0
    min_r: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0

    # Discipline
    avg_discipline: float = 0.0
    clean_execution_rate: float = 0.0

    # Regime breakdown
    regime_metrics: dict[str, dict] = field(default_factory=dict)
    # {regime: {win_rate, avg_r, trade_count, total_pnl}}

    # Session breakdown
    session_metrics: dict[str, dict] = field(default_factory=dict)

    # Streaks
    current_streak: int = 0     # positive = winning, negative = losing
    max_win_streak: int = 0
    max_loss_streak: int = 0

    # Trend
    recent_win_rate: float = 0.0     # last 20 trades
    historical_win_rate: float = 0.0  # all-time
    recent_expectancy: float = 0.0
    recent_trade_count: int = 0
    edge_trend: str = "stable"       # improving, declining, stable

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate, 3),
            "total_pnl": round(self.total_pnl, 2),
            "avg_r": round(self.avg_r, 2),
            "expectancy": round(self.expectancy, 3),
            "profit_factor": round(self.profit_factor, 2),
            "avg_discipline": round(self.avg_discipline, 1),
            "clean_execution_rate": round(self.clean_execution_rate, 3),
            "streaks": {
                "current": self.current_streak,
                "max_win": self.max_win_streak,
                "max_loss": self.max_loss_streak,
            },
            "edge_trend": self.edge_trend,
            "recent_vs_historical": {
                "recent_wr": round(self.recent_win_rate, 3),
                "historical_wr": round(self.historical_win_rate, 3),
                "recent_expectancy": round(self.recent_expectancy, 3),
                "recent_trades": self.recent_trade_count,
            },
            "regime_breakdown": self.regime_metrics,
            "session_breakdown": self.session_metrics,
        }


class StrategyPerformanceTracker:
    """
    Computes and monitors per-strategy performance metrics.
    Provides data for deviation attribution and strategy discovery.
    """

    def __init__(self, recent_window: int = 20):
        self._recent_window = recent_window

    def compute(
        self, db: DBSession, strategy_id: str, lookback_days: int = 180
    ) -> StrategyMetrics:
        """Compute full metrics for a single strategy."""
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        trades = (
            db.query(Trade)
            .filter(
                Trade.declared_strategy_id == strategy_id,
                Trade.status == TradeStatus.CLOSED,
                Trade.entry_time >= cutoff,
            )
            .order_by(Trade.entry_time.asc())
            .all()
        )

        metrics = StrategyMetrics(strategy_id=strategy_id)

        if not trades:
            return metrics

        metrics.trade_count = len(trades)

        # Win/Loss
        winners = [t for t in trades if t.pnl and t.pnl > 0]
        losers = [t for t in trades if t.pnl and t.pnl <= 0]
        metrics.win_count = len(winners)
        metrics.loss_count = len(losers)
        metrics.win_rate = len(winners) / len(trades)

        # P&L
        pnls = [t.pnl for t in trades if t.pnl is not None]
        if pnls:
            metrics.total_pnl = sum(pnls)
            metrics.avg_pnl = sum(pnls) / len(pnls)

        if winners:
            metrics.avg_winner = sum(t.pnl for t in winners) / len(winners)
        if losers:
            metrics.avg_loser = sum(abs(t.pnl) for t in losers) / len(losers)

        # R-multiples
        r_vals = [t.r_multiple for t in trades if t.r_multiple is not None]
        if r_vals:
            import statistics
            metrics.avg_r = statistics.mean(r_vals)
            metrics.median_r = statistics.median(r_vals)
            metrics.max_r = max(r_vals)
            metrics.min_r = min(r_vals)

        # Expectancy
        if metrics.avg_winner > 0 and metrics.avg_loser > 0:
            metrics.expectancy = (
                metrics.win_rate * metrics.avg_winner
                - (1 - metrics.win_rate) * metrics.avg_loser
            )

        # Profit factor
        gross_profit = sum(t.pnl for t in winners if t.pnl)
        gross_loss = sum(abs(t.pnl) for t in losers if t.pnl)
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss

        # Discipline
        trade_ids = [t.id for t in trades]
        attributions = (
            db.query(StrategyAttribution)
            .filter(StrategyAttribution.trade_id.in_(trade_ids))
            .all()
        )
        disc_scores = [a.discipline_score for a in attributions if a.discipline_score is not None]
        if disc_scores:
            metrics.avg_discipline = sum(disc_scores) / len(disc_scores)

        clean = sum(1 for a in attributions if a.discipline_score and a.discipline_score >= 95)
        metrics.clean_execution_rate = clean / len(trades) if trades else 0

        # Streaks
        self._compute_streaks(metrics, trades)

        # Regime breakdown
        self._compute_regime_metrics(metrics, trades)

        # Session breakdown
        self._compute_session_metrics(metrics, trades)

        # Recent vs historical (edge trend)
        self._compute_edge_trend(metrics, trades)

        return metrics

    def compute_all(
        self, db: DBSession, lookback_days: int = 180
    ) -> dict[str, StrategyMetrics]:
        """Compute metrics for all strategies with trades."""
        strategy_ids = (
            db.query(Trade.declared_strategy_id)
            .filter(
                Trade.declared_strategy_id.isnot(None),
                Trade.status == TradeStatus.CLOSED,
            )
            .distinct()
            .all()
        )

        results = {}
        for (sid,) in strategy_ids:
            results[sid] = self.compute(db, sid, lookback_days)

        return results

    def get_recent_performance(
        self, db: DBSession, strategy_id: str
    ) -> dict[str, Any]:
        """Quick recent performance summary for deviation attribution."""
        metrics = self.compute(db, strategy_id, lookback_days=90)
        return {
            "recent_win_rate": metrics.recent_win_rate,
            "historical_win_rate": metrics.historical_win_rate,
            "recent_expectancy": metrics.recent_expectancy,
            "recent_trade_count": metrics.recent_trade_count,
            "current_streak": metrics.current_streak,
            "edge_trend": metrics.edge_trend,
            "avg_discipline": metrics.avg_discipline,
        }

    def _compute_streaks(self, metrics: StrategyMetrics, trades: list[Trade]) -> None:
        current = 0
        max_win = 0
        max_loss = 0
        win_streak = 0
        loss_streak = 0

        for t in trades:
            if t.pnl and t.pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win = max(max_win, win_streak)
            elif t.pnl and t.pnl <= 0:
                loss_streak += 1
                win_streak = 0
                max_loss = max(max_loss, loss_streak)

        # Current streak (from most recent)
        for t in reversed(trades):
            if t.pnl and t.pnl > 0:
                current += 1
            elif t.pnl and t.pnl <= 0:
                current -= 1
            else:
                continue
            # Only count consecutive same-sign
            if current > 0 and t.pnl and t.pnl <= 0:
                break
            if current < 0 and t.pnl and t.pnl > 0:
                break

        metrics.current_streak = current
        metrics.max_win_streak = max_win
        metrics.max_loss_streak = max_loss

    def _compute_regime_metrics(
        self, metrics: StrategyMetrics, trades: list[Trade]
    ) -> None:
        regime_groups: dict[str, list[Trade]] = defaultdict(list)
        for t in trades:
            if t.market_regime:
                regime_groups[t.market_regime.value].append(t)

        for regime, group in regime_groups.items():
            wins = sum(1 for t in group if t.pnl and t.pnl > 0)
            r_vals = [t.r_multiple for t in group if t.r_multiple is not None]
            pnls = [t.pnl for t in group if t.pnl is not None]

            metrics.regime_metrics[regime] = {
                "trade_count": len(group),
                "win_rate": round(wins / len(group), 3) if group else 0,
                "avg_r": round(sum(r_vals) / len(r_vals), 2) if r_vals else 0,
                "total_pnl": round(sum(pnls), 2) if pnls else 0,
            }

    def _compute_session_metrics(
        self, metrics: StrategyMetrics, trades: list[Trade]
    ) -> None:
        session_groups: dict[str, list[Trade]] = defaultdict(list)
        for t in trades:
            if t.session_type:
                session_groups[t.session_type.value].append(t)

        for session, group in session_groups.items():
            wins = sum(1 for t in group if t.pnl and t.pnl > 0)
            r_vals = [t.r_multiple for t in group if t.r_multiple is not None]

            metrics.session_metrics[session] = {
                "trade_count": len(group),
                "win_rate": round(wins / len(group), 3) if group else 0,
                "avg_r": round(sum(r_vals) / len(r_vals), 2) if r_vals else 0,
            }

    def _compute_edge_trend(
        self, metrics: StrategyMetrics, trades: list[Trade]
    ) -> None:
        if len(trades) < self._recent_window:
            metrics.recent_win_rate = metrics.win_rate
            metrics.historical_win_rate = metrics.win_rate
            metrics.recent_trade_count = len(trades)
            return

        recent = trades[-self._recent_window:]
        historical = trades[:-self._recent_window]

        recent_wins = sum(1 for t in recent if t.pnl and t.pnl > 0)
        metrics.recent_win_rate = recent_wins / len(recent)
        metrics.recent_trade_count = len(recent)

        hist_wins = sum(1 for t in historical if t.pnl and t.pnl > 0)
        metrics.historical_win_rate = hist_wins / len(historical) if historical else 0

        # Recent expectancy
        recent_r = [t.r_multiple for t in recent if t.r_multiple is not None]
        if recent_r:
            metrics.recent_expectancy = sum(recent_r) / len(recent_r)

        # Edge trend
        wr_diff = metrics.recent_win_rate - metrics.historical_win_rate
        if wr_diff > 0.05:
            metrics.edge_trend = "improving"
        elif wr_diff < -0.05:
            metrics.edge_trend = "declining"
        else:
            metrics.edge_trend = "stable"