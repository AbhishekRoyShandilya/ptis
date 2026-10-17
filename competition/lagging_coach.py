# competition/lagging_coach.py
"""
"Where Am I Lagging?" Coaching Mode
=====================================
Synthesizes insights from all competition framework modules to answer
the trader's most important question: "Where exactly am I falling short?"

This module is the coaching layer that sits on top of:
- SideBySideReport (raw performance comparison)
- MissedOpportunityDetector (what they didn't take)
- RiskAdjustedComparator (risk-normalized comparison)

It produces structured, actionable coaching recommendations in 5 categories:

1. SETUP SELECTION — Are you picking the right trades?
2. RISK MANAGEMENT — Are you sizing and stopping correctly?
3. EXIT EXECUTION — Are you leaving money on the table?
4. DISCIPLINE — Are you following your own rules?
5. AWARENESS — Are you missing what the system sees?

Output format: Coaching cards that can be rendered in the frontend
as an interactive coaching session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session as DBSession

from competition.side_by_side_report import SideBySideReportGenerator, SideBySideReport
from competition.missed_opportunity_detector import MissedOpportunityDetector, MissedOpportunityReport
from competition.risk_adjusted_comparison import RiskAdjustedComparator, RiskComparisonReport

logger = logging.getLogger(__name__)


@dataclass
class CoachingCard:
    """
    A single coaching insight card — the atomic unit of coaching feedback.

    Cards are designed to be rendered in the frontend as actionable items.
    """
    card_id: str
    category: str        # "setup_selection" | "risk_management" | etc.
    severity: str        # "critical" | "warning" | "info" | "positive"
    title: str
    description: str
    metric_label: str    # e.g., "Win Rate Gap"
    metric_value: str    # e.g., "-12%"
    comparison: str      # "You: 58% vs System: 70%"
    action_items: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)  # supporting data points
    priority: int = 0    # 0=highest priority

    def to_dict(self) -> Dict[str, Any]:
        return {
            "card_id": self.card_id,
            "category": self.category,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "metric": {
                "label": self.metric_label,
                "value": self.metric_value,
                "comparison": self.comparison,
            },
            "action_items": self.action_items,
            "evidence": self.evidence,
            "priority": self.priority,
        }


@dataclass
class CoachingSession:
    """
    Complete coaching session — a collection of coaching cards
    organized by category and priority.
    """
    session_id: str
    generated_at: datetime
    period_label: str

    cards: List[CoachingCard] = field(default_factory=list)
    overall_grade: str = "C"
    headline: str = ""
    quick_wins: List[str] = field(default_factory=list)  # top 3 easiest improvements

    # Category scores
    category_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "generated_at": self.generated_at.isoformat(),
            "period_label": self.period_label,
            "overall_grade": self.overall_grade,
            "headline": self.headline,
            "quick_wins": self.quick_wins,
            "category_scores": {
                k: round(v, 1) for k, v in self.category_scores.items()
            },
            "cards": [c.to_dict() for c in self.cards],
            "summary": {
                "total_cards": len(self.cards),
                "critical": sum(1 for c in self.cards if c.severity == "critical"),
                "warnings": sum(1 for c in self.cards if c.severity == "warning"),
                "positive": sum(1 for c in self.cards if c.severity == "positive"),
            },
        }


class LaggingCoach:
    """
    Produces actionable coaching insights by analyzing where the trader
    is lagging relative to the system.

    This is the central coaching intelligence that combines all competition
    framework outputs into structured, prioritized feedback.

    Usage:
        coach = LaggingCoach(db)
        session = coach.weekly_coaching()
        cards = session.cards  # iterate and render in UI
    """

    def __init__(self, db: DBSession, capital: float = 500_000.0):
        self._db = db
        self._capital = capital
        self._card_counter = 0

    def weekly_coaching(self, week_offset: int = 0) -> CoachingSession:
        """Generate a full coaching session for a specific week."""
        import uuid

        # Gather all analysis
        report_gen = SideBySideReportGenerator(self._db)
        side_by_side = report_gen.weekly_report(week_offset)

        missed_detector = MissedOpportunityDetector(self._db)
        missed = missed_detector.analyze_week(week_offset)

        risk_comp = RiskAdjustedComparator(self._db, self._capital)
        risk_report = risk_comp.weekly_comparison(week_offset)

        # Build coaching session
        session = CoachingSession(
            session_id=str(uuid.uuid4()),
            generated_at=datetime.now(timezone.utc),
            period_label=side_by_side.period_label,
        )

        # Generate cards for each category
        cards: List[CoachingCard] = []
        cards.extend(self._setup_selection_cards(side_by_side, missed))
        cards.extend(self._risk_management_cards(risk_report))
        cards.extend(self._exit_execution_cards(side_by_side, risk_report))
        cards.extend(self._discipline_cards(side_by_side, missed))
        cards.extend(self._awareness_cards(missed))

        # Sort by priority
        cards.sort(key=lambda c: c.priority)

        session.cards = cards

        # Category scores
        session.category_scores = self._compute_category_scores(
            side_by_side, missed, risk_report
        )

        # Overall grade
        session.overall_grade = risk_report.risk_grade

        # Headline
        session.headline = self._generate_headline(side_by_side, missed, risk_report)

        # Quick wins (top 3 most impactful, easiest actions)
        session.quick_wins = self._identify_quick_wins(cards)

        return session

    def _next_card_id(self) -> str:
        self._card_counter += 1
        return f"card_{self._card_counter:03d}"

    # ────────────────────────────────────────────
    #  Setup Selection Cards
    # ────────────────────────────────────────────

    def _setup_selection_cards(
        self, report: SideBySideReport, missed: MissedOpportunityReport
    ) -> List[CoachingCard]:
        """Analyze setup selection quality."""
        cards = []
        ts = report.trader_summary
        ss = report.system_summary

        if not ts or not ss:
            return cards

        # Win rate comparison
        if ts.total_trades > 0 and ss.total_trades > 0:
            wr_gap = ss.win_rate - ts.win_rate
            if wr_gap > 0.05:
                cards.append(CoachingCard(
                    card_id=self._next_card_id(),
                    category="setup_selection",
                    severity="warning" if wr_gap < 0.15 else "critical",
                    title="Setup Selection Needs Work",
                    description=(
                        f"The system has a {wr_gap:.0%} higher win rate. "
                        f"This suggests you're taking lower-quality setups."
                    ),
                    metric_label="Win Rate Gap",
                    metric_value=f"{wr_gap:+.0%}",
                    comparison=f"You: {ts.win_rate:.0%} vs System: {ss.win_rate:.0%}",
                    action_items=[
                        "Review your last 10 losing trades — were any forced entries?",
                        "Compare your entry criteria against the SKO scoring thresholds",
                        "Consider waiting for conviction ≥ 0.65 before entering",
                    ],
                    evidence=[
                        f"Your win rate: {ts.win_rate:.0%} ({ts.wins}W/{ts.losses}L)",
                        f"System win rate: {ss.win_rate:.0%} ({ss.wins}W/{ss.losses}L)",
                    ],
                    priority=10 if wr_gap >= 0.15 else 20,
                ))
            elif wr_gap < -0.05:
                cards.append(CoachingCard(
                    card_id=self._next_card_id(),
                    category="setup_selection",
                    severity="positive",
                    title="Great Setup Selection! 🎯",
                    description=(
                        f"Your win rate ({ts.win_rate:.0%}) beats the system ({ss.win_rate:.0%}). "
                        f"Your discretionary filtering is adding value."
                    ),
                    metric_label="Win Rate Advantage",
                    metric_value=f"{abs(wr_gap):+.0%}",
                    comparison=f"You: {ts.win_rate:.0%} vs System: {ss.win_rate:.0%}",
                    priority=90,
                ))

        # Trade volume comparison
        if ts.total_trades > ss.total_trades * 1.5:
            cards.append(CoachingCard(
                card_id=self._next_card_id(),
                category="setup_selection",
                severity="warning",
                title="Possible Over-Trading",
                description=(
                    f"You took {ts.total_trades} trades vs system's {ss.total_trades}. "
                    f"More trades ≠ better results. The system is more selective."
                ),
                metric_label="Trade Count Ratio",
                metric_value=f"{ts.total_trades}/{ss.total_trades}",
                comparison=f"You: {ts.total_trades} trades vs System: {ss.total_trades}",
                action_items=[
                    "Identify your 'impulse trades' — entries without a full setup",
                    "Set a daily trade limit and stick to it",
                ],
                priority=25,
            ))

        return cards

    # ────────────────────────────────────────────
    #  Risk Management Cards
    # ────────────────────────────────────────────

    def _risk_management_cards(
        self, risk_report: RiskComparisonReport
    ) -> List[CoachingCard]:
        """Analyze risk management quality."""
        cards = []
        tm = risk_report.trader_metrics
        sm = risk_report.system_metrics

        if not tm or not sm:
            return cards

        # Drawdown comparison
        if abs(tm.max_drawdown_pct) > abs(sm.max_drawdown_pct) * 1.5:
            cards.append(CoachingCard(
                card_id=self._next_card_id(),
                category="risk_management",
                severity="critical",
                title="Drawdown Risk is High",
                description=(
                    f"Your max drawdown ({tm.max_drawdown_pct:.1%}) is significantly worse than "
                    f"the system's ({sm.max_drawdown_pct:.1%}). Larger drawdowns are harder to recover from."
                ),
                metric_label="Max Drawdown",
                metric_value=f"{tm.max_drawdown_pct:.1%}",
                comparison=f"You: {tm.max_drawdown_pct:.1%} vs System: {sm.max_drawdown_pct:.1%}",
                action_items=[
                    "Reduce position sizes during losing streaks",
                    f"Consider a daily loss limit of {self._capital * 0.02:,.0f} (2% of capital)",
                    "Implement a 'circuit breaker' — pause after 3 consecutive losses",
                ],
                evidence=[
                    f"Drawdown lasted {tm.drawdown_duration_days} days",
                    f"Drawdown amount: ₹{tm.max_drawdown_abs:,.0f}",
                ],
                priority=5,
            ))

        # Sharpe comparison
        if sm.sharpe_ratio > tm.sharpe_ratio + 0.5:
            cards.append(CoachingCard(
                card_id=self._next_card_id(),
                category="risk_management",
                severity="warning",
                title="Risk-Adjusted Returns Lagging",
                description=(
                    f"System's Sharpe ratio ({sm.sharpe_ratio:.2f}) significantly exceeds yours "
                    f"({tm.sharpe_ratio:.2f}). You're taking more risk per unit of return."
                ),
                metric_label="Sharpe Ratio",
                metric_value=f"{tm.sharpe_ratio:.2f}",
                comparison=f"You: {tm.sharpe_ratio:.2f} vs System: {sm.sharpe_ratio:.2f}",
                action_items=[
                    "Reduce position size on lower-conviction setups",
                    "Focus on higher R:R setups to improve return per unit of risk",
                ],
                priority=15,
            ))

        # Over-risking
        if tm.avg_risk_per_trade_pct > 0.02:
            cards.append(CoachingCard(
                card_id=self._next_card_id(),
                category="risk_management",
                severity="warning",
                title="Position Sizes May Be Too Large",
                description=(
                    f"Your average risk per trade is {tm.avg_risk_per_trade_pct:.1%} of capital. "
                    f"Professional guideline is 1-2%."
                ),
                metric_label="Avg Risk Per Trade",
                metric_value=f"{tm.avg_risk_per_trade_pct:.1%}",
                comparison=f"You: {tm.avg_risk_per_trade_pct:.1%} vs Recommended: 1-2%",
                action_items=[
                    f"Cap risk at ₹{self._capital * 0.01:,.0f} per trade (1%)",
                    "Use the Kelly criterion for optimal sizing",
                ],
                priority=20,
            ))

        return cards

    # ────────────────────────────────────────────
    #  Exit Execution Cards
    # ────────────────────────────────────────────

    def _exit_execution_cards(
        self, report: SideBySideReport, risk_report: RiskComparisonReport
    ) -> List[CoachingCard]:
        """Analyze exit quality — are you leaving R on the table?"""
        cards = []
        ts = report.trader_summary
        ss = report.system_summary
        tm = risk_report.trader_metrics
        sm = risk_report.system_metrics

        if not ts or not ss or not tm or not sm:
            return cards

        # Average R comparison
        r_gap = ss.avg_r - ts.avg_r
        if r_gap > 0.2 and ts.total_trades > 3:
            cards.append(CoachingCard(
                card_id=self._next_card_id(),
                category="exit_execution",
                severity="warning" if r_gap < 0.5 else "critical",
                title="Leaving R on the Table",
                description=(
                    f"The system captures {r_gap:+.2f}R more per trade on average. "
                    f"This suggests premature exits or loose stops."
                ),
                metric_label="Avg R Gap",
                metric_value=f"{r_gap:+.2f}R",
                comparison=f"You: {ts.avg_r:+.2f}R vs System: {ss.avg_r:+.2f}R",
                action_items=[
                    "Review your last 5 winners — did you exit too early?",
                    "Consider using trailing stops instead of fixed targets",
                    "Test partial exits: sell 50% at 1R, let rest run to 2R",
                ],
                evidence=[
                    f"Your avg R: {ts.avg_r:+.2f}",
                    f"System avg R: {ss.avg_r:+.2f}",
                    f"Estimated impact: ₹{r_gap * self._capital * 0.01 * ts.total_trades:,.0f} over {ts.total_trades} trades",
                ],
                priority=12,
            ))

        # Win/loss ratio
        if sm.win_loss_ratio > tm.win_loss_ratio * 1.3 and tm.win_loss_ratio > 0:
            cards.append(CoachingCard(
                card_id=self._next_card_id(),
                category="exit_execution",
                severity="info",
                title="Payoff Asymmetry Can Improve",
                description=(
                    f"System's average winner is {sm.win_loss_ratio:.1f}x its average loser, "
                    f"yours is {tm.win_loss_ratio:.1f}x. Better exits = better asymmetry."
                ),
                metric_label="Payoff Ratio",
                metric_value=f"{tm.win_loss_ratio:.1f}x",
                comparison=f"You: {tm.win_loss_ratio:.1f}x vs System: {sm.win_loss_ratio:.1f}x",
                action_items=[
                    "Cut losses faster — tighten stops on setups that aren't working",
                    "Let winners run — resist the urge to book profits early",
                ],
                priority=30,
            ))

        return cards

    # ────────────────────────────────────────────
    #  Discipline Cards
    # ────────────────────────────────────────────

    def _discipline_cards(
        self, report: SideBySideReport, missed: MissedOpportunityReport
    ) -> List[CoachingCard]:
        """Analyze trading discipline."""
        cards = []
        ts = report.trader_summary

        if not ts:
            return cards

        # Losing streak
        if ts.max_consecutive_losses >= 4:
            cards.append(CoachingCard(
                card_id=self._next_card_id(),
                category="discipline",
                severity="critical",
                title=f"{ts.max_consecutive_losses}-Trade Losing Streak",
                description=(
                    f"You had {ts.max_consecutive_losses} consecutive losses. "
                    f"After 3 losses, the probability of emotional decision-making increases significantly."
                ),
                metric_label="Max Consecutive Losses",
                metric_value=str(ts.max_consecutive_losses),
                comparison="Recommended max: 3 before pause",
                action_items=[
                    "Implement a firm 3-loss pause rule",
                    "After a streak, review if later trades deviated from your plan",
                    "Consider reducing size by 50% after 2 consecutive losses",
                ],
                priority=8,
            ))

        # Time-of-day performance gap
        worst_time = None
        worst_gap = 0
        for b in report.by_time_of_day:
            if b.better_performer == "system" and b.gap > worst_gap:
                worst_time = b
                worst_gap = b.gap

        if worst_time and worst_gap > 500:
            cards.append(CoachingCard(
                card_id=self._next_card_id(),
                category="discipline",
                severity="warning",
                title=f"Weak Performance During {worst_time.category_value}",
                description=(
                    f"Your P&L during {worst_time.category_value} lags the system by "
                    f"₹{worst_gap:,.0f}. This may indicate fatigue, rushed entries, or FOMO."
                ),
                metric_label="Time Bucket Gap",
                metric_value=f"₹{worst_gap:,.0f}",
                comparison=f"You: ₹{worst_time.trader_pnl:,.0f} vs System: ₹{worst_time.system_pnl:,.0f}",
                action_items=[
                    f"Consider skipping or reducing size during {worst_time.category_value}",
                    "Track your energy and focus levels throughout the day",
                ],
                priority=22,
            ))

        return cards

    # ────────────────────────────────────────────
    #  Awareness Cards
    # ────────────────────────────────────────────

    def _awareness_cards(self, missed: MissedOpportunityReport) -> List[CoachingCard]:
        """Analyze market awareness — what are you not seeing?"""
        cards = []

        if missed.critical_misses > 0:
            cards.append(CoachingCard(
                card_id=self._next_card_id(),
                category="awareness",
                severity="critical" if missed.critical_misses >= 3 else "warning",
                title=f"{missed.critical_misses} High-Value Setups Missed",
                description=(
                    f"The system identified {missed.critical_misses} high-conviction setups "
                    f"you didn't take, worth ₹{missed.total_missed_pnl:,.0f} in total profit."
                ),
                metric_label="Missed Profit",
                metric_value=f"₹{missed.total_missed_pnl:,.0f}",
                comparison=f"System signals: {missed.total_system_signals} | You traded: {missed.trader_trades_taken}",
                action_items=[
                    "Enable real-time alerts for signals with conviction ≥ 0.70",
                    f"Add {missed.most_missed_symbol} to your primary watchlist" if missed.most_missed_symbol else "Review your watchlist coverage",
                    "Set up the scanner dashboard as your market open screen",
                ],
                evidence=[
                    f"Biggest miss: {missed.biggest_miss_symbol} (₹{missed.biggest_miss_pnl:,.0f})",
                    f"Most missed time: {missed.most_missed_time_bucket}" if missed.most_missed_time_bucket else "",
                ],
                priority=7 if missed.critical_misses >= 3 else 18,
            ))

        # User-taught variant misses (most painful)
        user_variant_misses = [
            m for m in missed.missed_opportunities
            if m.variant_provenance == "user_taught" and m.severity == "critical"
        ]
        if user_variant_misses:
            variant_pnl = sum(m.shadow_pnl for m in user_variant_misses if m.shadow_pnl)
            cards.append(CoachingCard(
                card_id=self._next_card_id(),
                category="awareness",
                severity="critical",
                title="You Missed YOUR OWN Patterns! 😤",
                description=(
                    f"The system matched {len(user_variant_misses)} setups to variants "
                    f"you personally taught it, but you didn't take the trades. "
                    f"Combined profit: ₹{variant_pnl:,.0f}."
                ),
                metric_label="User-Variant Misses",
                metric_value=str(len(user_variant_misses)),
                comparison="These are YOUR taught patterns!",
                action_items=[
                    "Set up priority alerts for user-taught variant matches",
                    "Review why you hesitated — was the conviction score too low?",
                    "Consider if these variants need refinement or if it was just execution hesitation",
                ],
                priority=3,
            ))

        return cards

    # ────────────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────────────

    def _compute_category_scores(
        self,
        side_by_side: SideBySideReport,
        missed: MissedOpportunityReport,
        risk_report: RiskComparisonReport,
    ) -> Dict[str, float]:
        """Compute 0-100 score for each coaching category."""
        scores: Dict[str, float] = {}
        ts = side_by_side.trader_summary
        ss = side_by_side.system_summary
        tm = risk_report.trader_metrics

        # Setup selection (based on win rate gap)
        if ts and ss:
            wr_gap = ts.win_rate - ss.win_rate
            scores["setup_selection"] = min(100, max(0, 50 + wr_gap * 200))
        else:
            scores["setup_selection"] = 50

        # Risk management (based on risk grade)
        grade_scores = {"A": 90, "B": 75, "C": 60, "D": 40, "F": 20}
        scores["risk_management"] = grade_scores.get(risk_report.risk_grade, 50)

        # Exit execution (based on avg R gap)
        if ts and ss:
            r_gap = ts.avg_r - ss.avg_r
            scores["exit_execution"] = min(100, max(0, 50 + r_gap * 50))
        else:
            scores["exit_execution"] = 50

        # Discipline (based on streak + time consistency)
        disc_score = 80
        if ts and ts.max_consecutive_losses >= 4:
            disc_score -= 20
        if ts and ts.max_consecutive_losses >= 6:
            disc_score -= 20
        scores["discipline"] = max(0, disc_score)

        # Awareness (based on coverage)
        if missed.total_system_signals > 0:
            coverage = 1 - (missed.missed_count / missed.total_system_signals)
            scores["awareness"] = coverage * 100
        else:
            scores["awareness"] = 50

        return scores

    def _generate_headline(
        self,
        side_by_side: SideBySideReport,
        missed: MissedOpportunityReport,
        risk_report: RiskComparisonReport,
    ) -> str:
        """Generate a coaching headline."""
        ts = side_by_side.trader_summary
        grade = risk_report.risk_grade

        if grade in ("A", "B"):
            return (
                f"Strong week! Grade: {grade}. "
                f"{'Minor improvement areas below.' if grade == 'B' else 'Keep sharpening the edge.'}"
            )
        elif grade == "C":
            return (
                f"Acceptable performance (Grade: {grade}), but clear improvement areas. "
                f"Focus on the critical cards below."
            )
        else:
            critical_count = sum(
                1 for m in missed.missed_opportunities if m.severity == "critical"
            )
            return (
                f"Challenging week (Grade: {grade}). "
                f"{critical_count} critical issues identified. "
                f"Start with the quick wins below."
            )

    def _identify_quick_wins(self, cards: List[CoachingCard]) -> List[str]:
        """Identify top 3 easiest improvement actions."""
        quick_actions = []

        for card in cards:
            if card.severity in ("critical", "warning") and card.action_items:
                quick_actions.append(card.action_items[0])

        return quick_actions[:3]
