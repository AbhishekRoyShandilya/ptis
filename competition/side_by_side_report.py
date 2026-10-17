# competition/side_by_side_report.py
"""
Side-by-Side Comparison Report
===============================
Generates comprehensive weekly/monthly reports comparing the trader's
actual performance against the system's autonomous virtual trades.

The report answers: "Who did better — you or the system?"

Sections:
1. Headline comparison (P&L, win rate, Sharpe)
2. Per-strategy breakdown
3. Category analysis (by symbol, by time-of-day, by setup type)
4. Streak analysis (winning/losing streaks)
5. AI narrative summary
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func as sa_func
from sqlalchemy.orm import Session as DBSession

from core.models import Trade
from shadow.models import ShadowTrade

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
#  Report Data Structures
# ═══════════════════════════════════════════

@dataclass
class PerformanceSummary:
    """Aggregated performance metrics for one participant (trader or system)."""
    participant: str  # "trader" or "system"
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    total_r: float = 0.0
    avg_r: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    best_trade_r: float = 0.0
    worst_trade_r: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_hold_minutes: float = 0.0
    profit_factor: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant": self.participant,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl": round(self.avg_pnl, 2),
            "total_r": round(self.total_r, 2),
            "avg_r": round(self.avg_r, 4),
            "best_trade_pnl": round(self.best_trade_pnl, 2),
            "worst_trade_pnl": round(self.worst_trade_pnl, 2),
            "best_trade_r": round(self.best_trade_r, 2),
            "worst_trade_r": round(self.worst_trade_r, 2),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_hold_minutes": round(self.avg_hold_minutes, 1),
            "profit_factor": round(self.profit_factor, 2),
        }


@dataclass
class CategoryBreakdown:
    """Performance breakdown by category (symbol, strategy, time-of-day)."""
    category_type: str  # "symbol", "strategy", "time_of_day"
    category_value: str
    trader_pnl: float = 0.0
    trader_trades: int = 0
    trader_win_rate: float = 0.0
    system_pnl: float = 0.0
    system_trades: int = 0
    system_win_rate: float = 0.0
    better_performer: str = "tie"  # "trader" | "system" | "tie"
    gap: float = 0.0  # absolute P&L gap

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_type": self.category_type,
            "category_value": self.category_value,
            "trader": {
                "pnl": round(self.trader_pnl, 2),
                "trades": self.trader_trades,
                "win_rate": round(self.trader_win_rate, 4),
            },
            "system": {
                "pnl": round(self.system_pnl, 2),
                "trades": self.system_trades,
                "win_rate": round(self.system_win_rate, 4),
            },
            "better_performer": self.better_performer,
            "gap": round(self.gap, 2),
        }


@dataclass
class SideBySideReport:
    """Complete side-by-side comparison report."""
    period_start: datetime
    period_end: datetime
    period_label: str  # "Week 12, 2026" or "March 2026"

    trader_summary: Optional[PerformanceSummary] = None
    system_summary: Optional[PerformanceSummary] = None

    # Category breakdowns
    by_symbol: List[CategoryBreakdown] = field(default_factory=list)
    by_strategy: List[CategoryBreakdown] = field(default_factory=list)
    by_time_of_day: List[CategoryBreakdown] = field(default_factory=list)

    # Head-to-head verdict
    overall_winner: str = "tie"  # "trader" | "system" | "tie"
    pnl_gap: float = 0.0
    r_gap: float = 0.0
    win_rate_gap: float = 0.0

    # Daily P&L curves for charting
    trader_daily_pnl: List[Dict[str, Any]] = field(default_factory=list)
    system_daily_pnl: List[Dict[str, Any]] = field(default_factory=list)

    # AI narrative
    headline: str = ""
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
                "label": self.period_label,
            },
            "trader": self.trader_summary.to_dict() if self.trader_summary else {},
            "system": self.system_summary.to_dict() if self.system_summary else {},
            "head_to_head": {
                "winner": self.overall_winner,
                "pnl_gap": round(self.pnl_gap, 2),
                "r_gap": round(self.r_gap, 4),
                "win_rate_gap": round(self.win_rate_gap, 4),
            },
            "breakdowns": {
                "by_symbol": [b.to_dict() for b in self.by_symbol],
                "by_strategy": [b.to_dict() for b in self.by_strategy],
                "by_time_of_day": [b.to_dict() for b in self.by_time_of_day],
            },
            "daily_curves": {
                "trader": self.trader_daily_pnl,
                "system": self.system_daily_pnl,
            },
            "narrative": {
                "headline": self.headline,
                "key_findings": self.key_findings,
                "recommendations": self.recommendations,
            },
        }


# ═══════════════════════════════════════════
#  Report Generator
# ═══════════════════════════════════════════

class SideBySideReportGenerator:
    """
    Generates side-by-side comparison reports between trader and system.

    Usage:
        generator = SideBySideReportGenerator(db)
        report = generator.weekly_report()  # current week
        report = generator.monthly_report() # current month
        report = generator.custom_report(start, end)
    """

    def __init__(self, db: DBSession):
        self._db = db

    def weekly_report(self, week_offset: int = 0) -> SideBySideReport:
        """Generate report for a specific week (0=current, -1=last week, etc.)."""
        today = datetime.now(timezone.utc).date()
        # Monday of the target week
        start_of_week = today - timedelta(days=today.weekday()) + timedelta(weeks=week_offset)
        end_of_week = start_of_week + timedelta(days=6)

        start = datetime.combine(start_of_week, datetime.min.time()).replace(tzinfo=timezone.utc)
        end = datetime.combine(end_of_week, datetime.max.time()).replace(tzinfo=timezone.utc)

        iso_year, iso_week, _ = start_of_week.isocalendar()
        label = f"Week {iso_week}, {iso_year}"

        return self._generate_report(start, end, label)

    def monthly_report(self, month_offset: int = 0) -> SideBySideReport:
        """Generate report for a specific month (0=current, -1=last month, etc.)."""
        today = datetime.now(timezone.utc).date()
        # First day of target month
        month = today.month + month_offset
        year = today.year
        while month <= 0:
            month += 12
            year -= 1
        while month > 12:
            month -= 12
            year += 1

        start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)

        label = start.strftime("%B %Y")
        return self._generate_report(start, end, label)

    def custom_report(self, start: datetime, end: datetime) -> SideBySideReport:
        """Generate report for a custom date range."""
        label = f"{start.strftime('%d %b')} – {end.strftime('%d %b %Y')}"
        return self._generate_report(start, end, label)

    def _generate_report(
        self, start: datetime, end: datetime, label: str
    ) -> SideBySideReport:
        """Core report generation logic."""
        # Fetch trader trades
        trader_trades = (
            self._db.query(Trade)
            .filter(Trade.entry_time >= start, Trade.entry_time <= end)
            .filter(Trade.pnl.isnot(None))
            .all()
        )

        # Fetch system autonomous shadows
        system_trades = (
            self._db.query(ShadowTrade)
            .filter(
                ShadowTrade.shadow_type == "autonomous",
                ShadowTrade.entry_time >= start,
                ShadowTrade.entry_time <= end,
                ShadowTrade.status.in_(["closed", "stopped", "target_hit"]),
            )
            .all()
        )

        report = SideBySideReport(
            period_start=start,
            period_end=end,
            period_label=label,
        )

        # Build summaries
        report.trader_summary = self._build_trader_summary(trader_trades)
        report.system_summary = self._build_system_summary(system_trades)

        # Head-to-head
        tp = report.trader_summary.total_pnl
        sp = report.system_summary.total_pnl
        report.pnl_gap = sp - tp
        report.r_gap = report.system_summary.avg_r - report.trader_summary.avg_r
        report.win_rate_gap = report.system_summary.win_rate - report.trader_summary.win_rate

        if abs(report.pnl_gap) < 100:
            report.overall_winner = "tie"
        elif sp > tp:
            report.overall_winner = "system"
        else:
            report.overall_winner = "trader"

        # Category breakdowns
        report.by_symbol = self._breakdown_by_symbol(trader_trades, system_trades)
        report.by_strategy = self._breakdown_by_strategy(trader_trades, system_trades)
        report.by_time_of_day = self._breakdown_by_time(trader_trades, system_trades)

        # Daily P&L curves
        report.trader_daily_pnl = self._daily_pnl_curve(trader_trades, start, end)
        report.system_daily_pnl = self._daily_pnl_curve_system(system_trades, start, end)

        # Generate narrative
        self._generate_narrative(report)

        return report

    def _build_trader_summary(self, trades: List[Trade]) -> PerformanceSummary:
        """Build performance summary from real trades."""
        summary = PerformanceSummary(participant="trader")
        if not trades:
            return summary

        pnls = [t.pnl or 0 for t in trades]
        r_mults = [t.r_multiple for t in trades if t.r_multiple is not None]

        summary.total_trades = len(trades)
        summary.wins = sum(1 for p in pnls if p > 0)
        summary.losses = sum(1 for p in pnls if p <= 0)
        summary.win_rate = summary.wins / summary.total_trades if summary.total_trades else 0
        summary.total_pnl = sum(pnls)
        summary.avg_pnl = summary.total_pnl / summary.total_trades
        summary.total_r = sum(r_mults)
        summary.avg_r = sum(r_mults) / len(r_mults) if r_mults else 0
        summary.best_trade_pnl = max(pnls) if pnls else 0
        summary.worst_trade_pnl = min(pnls) if pnls else 0
        summary.best_trade_r = max(r_mults) if r_mults else 0
        summary.worst_trade_r = min(r_mults) if r_mults else 0

        # Streaks
        summary.max_consecutive_wins = self._max_streak(pnls, positive=True)
        summary.max_consecutive_losses = self._max_streak(pnls, positive=False)

        # Hold time
        durations = []
        for t in trades:
            if t.entry_time and t.exit_time:
                durations.append((t.exit_time - t.entry_time).total_seconds() / 60)
        summary.avg_hold_minutes = sum(durations) / len(durations) if durations else 0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        summary.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        return summary

    def _build_system_summary(self, shadows: List[ShadowTrade]) -> PerformanceSummary:
        """Build performance summary from autonomous shadow trades."""
        summary = PerformanceSummary(participant="system")
        if not shadows:
            return summary

        pnls = []
        r_mults = []
        for s in shadows:
            config = s.config or {}
            pnl = config.get("journal_pnl", s.realized_pnl or 0)
            r = config.get("journal_r_multiple", s.r_multiple)
            pnls.append(pnl if pnl is not None else 0)
            if r is not None:
                r_mults.append(r)

        summary.total_trades = len(shadows)
        summary.wins = sum(1 for p in pnls if p > 0)
        summary.losses = sum(1 for p in pnls if p <= 0)
        summary.win_rate = summary.wins / summary.total_trades if summary.total_trades else 0
        summary.total_pnl = sum(pnls)
        summary.avg_pnl = summary.total_pnl / summary.total_trades
        summary.total_r = sum(r_mults)
        summary.avg_r = sum(r_mults) / len(r_mults) if r_mults else 0
        summary.best_trade_pnl = max(pnls) if pnls else 0
        summary.worst_trade_pnl = min(pnls) if pnls else 0
        summary.best_trade_r = max(r_mults) if r_mults else 0
        summary.worst_trade_r = min(r_mults) if r_mults else 0

        summary.max_consecutive_wins = self._max_streak(pnls, positive=True)
        summary.max_consecutive_losses = self._max_streak(pnls, positive=False)

        durations = []
        for s in shadows:
            if s.entry_time and s.exit_time:
                durations.append((s.exit_time - s.entry_time).total_seconds() / 60)
        summary.avg_hold_minutes = sum(durations) / len(durations) if durations else 0

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        summary.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        return summary

    def _breakdown_by_symbol(
        self, trader_trades: list, system_trades: list
    ) -> List[CategoryBreakdown]:
        """Break down performance by symbol."""
        symbols = set()
        trader_by_sym: Dict[str, list] = defaultdict(list)
        system_by_sym: Dict[str, list] = defaultdict(list)

        for t in trader_trades:
            symbols.add(t.symbol)
            trader_by_sym[t.symbol].append(t.pnl or 0)

        for s in system_trades:
            symbols.add(s.symbol)
            config = s.config or {}
            pnl = config.get("journal_pnl", s.realized_pnl or 0)
            system_by_sym[s.symbol].append(pnl if pnl is not None else 0)

        breakdowns = []
        for sym in sorted(symbols):
            t_pnls = trader_by_sym.get(sym, [])
            s_pnls = system_by_sym.get(sym, [])
            t_total = sum(t_pnls)
            s_total = sum(s_pnls)

            breakdowns.append(CategoryBreakdown(
                category_type="symbol",
                category_value=sym,
                trader_pnl=t_total,
                trader_trades=len(t_pnls),
                trader_win_rate=sum(1 for p in t_pnls if p > 0) / len(t_pnls) if t_pnls else 0,
                system_pnl=s_total,
                system_trades=len(s_pnls),
                system_win_rate=sum(1 for p in s_pnls if p > 0) / len(s_pnls) if s_pnls else 0,
                better_performer="system" if s_total > t_total else "trader" if t_total > s_total else "tie",
                gap=abs(s_total - t_total),
            ))

        return sorted(breakdowns, key=lambda b: b.gap, reverse=True)

    def _breakdown_by_strategy(
        self, trader_trades: list, system_trades: list
    ) -> List[CategoryBreakdown]:
        """Break down performance by strategy/SKO."""
        strategies = set()
        trader_by_strat: Dict[str, list] = defaultdict(list)
        system_by_strat: Dict[str, list] = defaultdict(list)

        for t in trader_trades:
            strat = t.declared_strategy_id or "unclassified"
            strategies.add(strat)
            trader_by_strat[strat].append(t.pnl or 0)

        for s in system_trades:
            config = s.config or {}
            strat = config.get("sko_name", s.strategy_id or "system_auto")
            strategies.add(strat)
            pnl = config.get("journal_pnl", s.realized_pnl or 0)
            system_by_strat[strat].append(pnl if pnl is not None else 0)

        breakdowns = []
        for strat in sorted(strategies):
            t_pnls = trader_by_strat.get(strat, [])
            s_pnls = system_by_strat.get(strat, [])
            t_total = sum(t_pnls)
            s_total = sum(s_pnls)

            breakdowns.append(CategoryBreakdown(
                category_type="strategy",
                category_value=strat,
                trader_pnl=t_total,
                trader_trades=len(t_pnls),
                trader_win_rate=sum(1 for p in t_pnls if p > 0) / len(t_pnls) if t_pnls else 0,
                system_pnl=s_total,
                system_trades=len(s_pnls),
                system_win_rate=sum(1 for p in s_pnls if p > 0) / len(s_pnls) if s_pnls else 0,
                better_performer="system" if s_total > t_total else "trader" if t_total > s_total else "tie",
                gap=abs(s_total - t_total),
            ))

        return sorted(breakdowns, key=lambda b: b.gap, reverse=True)

    def _breakdown_by_time(
        self, trader_trades: list, system_trades: list
    ) -> List[CategoryBreakdown]:
        """Break down performance by time-of-day buckets."""
        buckets = {
            "09:15-10:00": (9, 15, 10, 0),
            "10:00-11:30": (10, 0, 11, 30),
            "11:30-13:00": (11, 30, 13, 0),
            "13:00-14:30": (13, 0, 14, 30),
            "14:30-15:30": (14, 30, 15, 30),
        }

        breakdowns = []
        for label, (sh, sm, eh, em) in buckets.items():
            t_pnls = [
                t.pnl or 0 for t in trader_trades
                if t.entry_time and sh * 60 + sm <= t.entry_time.hour * 60 + t.entry_time.minute < eh * 60 + em
            ]
            s_pnls = []
            for s in system_trades:
                if s.entry_time and sh * 60 + sm <= s.entry_time.hour * 60 + s.entry_time.minute < eh * 60 + em:
                    config = s.config or {}
                    pnl = config.get("journal_pnl", s.realized_pnl or 0)
                    s_pnls.append(pnl if pnl is not None else 0)

            t_total = sum(t_pnls)
            s_total = sum(s_pnls)

            breakdowns.append(CategoryBreakdown(
                category_type="time_of_day",
                category_value=label,
                trader_pnl=t_total,
                trader_trades=len(t_pnls),
                trader_win_rate=sum(1 for p in t_pnls if p > 0) / len(t_pnls) if t_pnls else 0,
                system_pnl=s_total,
                system_trades=len(s_pnls),
                system_win_rate=sum(1 for p in s_pnls if p > 0) / len(s_pnls) if s_pnls else 0,
                better_performer="system" if s_total > t_total else "trader" if t_total > s_total else "tie",
                gap=abs(s_total - t_total),
            ))

        return breakdowns

    def _daily_pnl_curve(
        self, trades: list, start: datetime, end: datetime
    ) -> List[Dict[str, Any]]:
        """Build daily cumulative P&L curve for trader."""
        daily: Dict[str, float] = defaultdict(float)
        for t in trades:
            if t.exit_time:
                day = t.exit_time.strftime("%Y-%m-%d")
                daily[day] += t.pnl or 0

        cumulative = 0.0
        curve = []
        current = start.date()
        end_date = end.date()
        while current <= end_date:
            day_str = current.strftime("%Y-%m-%d")
            cumulative += daily.get(day_str, 0)
            curve.append({"date": day_str, "cumulative_pnl": round(cumulative, 2)})
            current += timedelta(days=1)

        return curve

    def _daily_pnl_curve_system(
        self, shadows: list, start: datetime, end: datetime
    ) -> List[Dict[str, Any]]:
        """Build daily cumulative P&L curve for system."""
        daily: Dict[str, float] = defaultdict(float)
        for s in shadows:
            exit_time = s.exit_time
            if exit_time:
                day = exit_time.strftime("%Y-%m-%d")
                config = s.config or {}
                pnl = config.get("journal_pnl", s.realized_pnl or 0)
                daily[day] += pnl if pnl is not None else 0

        cumulative = 0.0
        curve = []
        current = start.date()
        end_date = end.date()
        while current <= end_date:
            day_str = current.strftime("%Y-%m-%d")
            cumulative += daily.get(day_str, 0)
            curve.append({"date": day_str, "cumulative_pnl": round(cumulative, 2)})
            current += timedelta(days=1)

        return curve

    def _generate_narrative(self, report: SideBySideReport) -> None:
        """Generate AI-style narrative findings and recommendations."""
        ts = report.trader_summary
        ss = report.system_summary

        if not ts or not ss:
            report.headline = "Insufficient data for comparison"
            return

        # Headline
        if report.overall_winner == "system":
            report.headline = (
                f"System outperformed by ₹{abs(report.pnl_gap):,.0f} this {report.period_label}. "
                f"System: {ss.win_rate:.0%} win rate | You: {ts.win_rate:.0%}"
            )
        elif report.overall_winner == "trader":
            report.headline = (
                f"You beat the system by ₹{abs(report.pnl_gap):,.0f} this {report.period_label}! "
                f"Your edge: {ts.win_rate:.0%} vs {ss.win_rate:.0%}"
            )
        else:
            report.headline = (
                f"Neck and neck this {report.period_label}. "
                f"P&L gap under ₹100 — both running comparable strategies."
            )

        # Key findings
        findings = []

        if ts.total_trades > 0 and ss.total_trades > 0:
            findings.append(
                f"You took {ts.total_trades} trades vs system's {ss.total_trades} "
                f"(system is {'more selective' if ss.total_trades < ts.total_trades else 'more active'})"
            )

        if ts.avg_r != 0 and ss.avg_r != 0:
            if ss.avg_r > ts.avg_r:
                findings.append(
                    f"System's avg R ({ss.avg_r:+.2f}R) beats yours ({ts.avg_r:+.2f}R) — "
                    f"suggests better exit management"
                )
            else:
                findings.append(
                    f"Your avg R ({ts.avg_r:+.2f}R) beats system ({ss.avg_r:+.2f}R) — "
                    f"your exit timing is strong"
                )

        if ts.profit_factor > 0 and ss.profit_factor > 0:
            findings.append(
                f"Profit factor: You={ts.profit_factor:.2f} vs System={ss.profit_factor:.2f}"
            )

        # Worst time-of-day finding
        if report.by_time_of_day:
            worst_bucket = max(
                [b for b in report.by_time_of_day if b.better_performer == "system"],
                key=lambda b: b.gap,
                default=None,
            )
            if worst_bucket:
                findings.append(
                    f"Biggest gap in {worst_bucket.category_value}: system ahead by "
                    f"₹{worst_bucket.gap:,.0f} — consider being more disciplined here"
                )

        report.key_findings = findings

        # Recommendations
        recs = []
        if report.overall_winner == "system":
            if ts.win_rate < ss.win_rate:
                recs.append(
                    "Focus on setup selection — the system is more selective and has "
                    "a higher win rate. Consider waiting for higher conviction setups."
                )
            if ts.avg_r < ss.avg_r:
                recs.append(
                    "Your exits leave R on the table. Study the system's exit points "
                    "this week — it's capturing more of the move."
                )
            if ts.max_consecutive_losses > 3:
                recs.append(
                    f"You had a {ts.max_consecutive_losses}-trade losing streak. "
                    "Consider implementing a pause rule after 3 consecutive losses."
                )
        else:
            recs.append("Great performance this period! Keep doing what you're doing.")
            if ss.total_trades > ts.total_trades:
                recs.append(
                    "Note: the system found more setups than you took. "
                    "Review the missed opportunities report for potential additions."
                )

        report.recommendations = recs

    @staticmethod
    def _max_streak(pnls: list, positive: bool = True) -> int:
        """Calculate maximum consecutive wins or losses."""
        max_streak = 0
        current = 0
        for p in pnls:
            if (positive and p > 0) or (not positive and p <= 0):
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak
