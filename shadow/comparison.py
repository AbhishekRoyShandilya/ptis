# comparison.py
# shadow/comparison.py
"""
Real vs Shadow comparison analytics.
Computes divergences, decision quality impact, and what-if analysis.
Called when real trade closes — compares against all shadow variants.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session as DBSession

from core.models import Trade
from shadow.models import ShadowSession, ShadowTrade, TradeDivergence

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  COMPARISON RESULT STRUCTURES
# ══════════════════════════════════════════════

@dataclass
class ShadowComparison:
    """Comparison of a single shadow trade against the real trade."""
    shadow_id: str
    shadow_type: str
    label: str

    # P&L comparison
    real_pnl: float = 0.0
    shadow_pnl: float = 0.0
    pnl_difference: float = 0.0
    real_r: Optional[float] = None
    shadow_r: Optional[float] = None
    r_difference: Optional[float] = None

    # Execution comparison
    same_direction: bool = True
    entry_price_diff: float = 0.0
    exit_price_diff: float = 0.0
    stop_diff: Optional[float] = None

    # Excursion comparison
    real_max_favorable: Optional[float] = None
    shadow_max_favorable: Optional[float] = None
    real_max_adverse: Optional[float] = None
    shadow_max_adverse: Optional[float] = None

    # Exit comparison
    real_exit_reason: str = ""
    shadow_exit_reason: str = ""
    real_duration_min: Optional[float] = None
    shadow_duration_min: Optional[float] = None

    # Verdict
    shadow_better: bool = False
    improvement_amount: float = 0.0
    improvement_r: Optional[float] = None

    # Partial exits
    shadow_partial_exits: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shadow_id": self.shadow_id,
            "shadow_type": self.shadow_type,
            "label": self.label,
            "real_pnl": round(self.real_pnl, 2),
            "shadow_pnl": round(self.shadow_pnl, 2),
            "pnl_difference": round(self.pnl_difference, 2),
            "real_r": self.real_r,
            "shadow_r": self.shadow_r,
            "r_difference": round(self.r_difference, 2) if self.r_difference else None,
            "shadow_better": self.shadow_better,
            "improvement_amount": round(self.improvement_amount, 2),
            "improvement_r": round(self.improvement_r, 2) if self.improvement_r else None,
            "real_exit_reason": self.real_exit_reason,
            "shadow_exit_reason": self.shadow_exit_reason,
            "real_duration_min": self.real_duration_min,
            "shadow_duration_min": self.shadow_duration_min,
            "stop_diff": round(self.stop_diff, 4) if self.stop_diff else None,
            "shadow_partial_exits": self.shadow_partial_exits,
        }


@dataclass
class ComparisonReport:
    """Full comparison report for a real trade against all its shadows."""
    real_trade_id: str
    real_pnl: float = 0.0
    real_r: Optional[float] = None

    comparisons: list[ShadowComparison] = field(default_factory=list)
    divergences: list[dict] = field(default_factory=list)

    # Aggregated insights
    best_shadow_id: Optional[str] = None
    best_shadow_pnl: float = 0.0
    best_shadow_type: Optional[str] = None
    worst_shadow_pnl: float = 0.0

    # Opportunity cost
    optimal_pnl: float = 0.0
    opportunity_cost: float = 0.0
    opportunity_cost_r: Optional[float] = None

    # Decision score impact
    execution_quality: float = 0.0  # 0-100: how close to optimal

    # Insights
    insights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "real_trade_id": self.real_trade_id,
            "real_pnl": round(self.real_pnl, 2),
            "real_r": self.real_r,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "divergences": self.divergences,
            "best_shadow": {
                "id": self.best_shadow_id,
                "type": self.best_shadow_type,
                "pnl": round(self.best_shadow_pnl, 2),
            },
            "opportunity_cost": round(self.opportunity_cost, 2),
            "opportunity_cost_r": (
                round(self.opportunity_cost_r, 2)
                if self.opportunity_cost_r else None
            ),
            "execution_quality": round(self.execution_quality, 1),
            "insights": self.insights,
        }


# ══════════════════════════════════════════════
#  COMPARISON ENGINE
# ══════════════════════════════════════════════

class ShadowComparisonEngine:
    """
    Compares real trade execution against shadow trade outcomes.
    Generates divergence records and decision quality metrics.
    """

    def compare(
        self,
        db: DBSession,
        real_trade_id: uuid.UUID,
    ) -> Optional[ComparisonReport]:
        """
        Run full comparison for a closed real trade.
        Returns None if trade not found or no shadows exist.
        """
        trade = db.query(Trade).filter(Trade.id == real_trade_id).first()
        if not trade:
            logger.warning("Trade %s not found for comparison", real_trade_id)
            return None

        if trade.pnl is None:
            logger.warning("Trade %s not closed — cannot compare", real_trade_id)
            return None

        session = (
            db.query(ShadowSession)
            .filter(ShadowSession.real_trade_id == real_trade_id)
            .first()
        )
        if not session:
            logger.info("No shadow session for trade %s", real_trade_id)
            return None

        shadows = (
            db.query(ShadowTrade)
            .filter(ShadowTrade.session_id == session.id)
            .all()
        )

        if not shadows:
            return None

        report = ComparisonReport(
            real_trade_id=str(real_trade_id),
            real_pnl=trade.pnl,
            real_r=trade.r_multiple,
        )

        # Compare each shadow
        for shadow in shadows:
            comparison = self._compare_single(trade, shadow)
            report.comparisons.append(comparison)

            # Record divergences in DB
            divergences = self._detect_divergences(trade, shadow)
            for div in divergences:
                db.add(div)
                report.divergences.append({
                    "shadow_id": str(shadow.id),
                    "type": div.divergence_type,
                    "description": div.description,
                    "impact_amount": div.impact_amount,
                })

        # Aggregate analysis
        self._compute_aggregates(report, trade)
        self._generate_comparison_insights(report, trade)

        db.commit()

        logger.info(
            "Comparison complete for trade %s: real=$%.2f, best_shadow=$%.2f, opp_cost=$%.2f",
            real_trade_id, report.real_pnl,
            report.best_shadow_pnl, report.opportunity_cost,
        )

        return report

    def _compare_single(
        self, trade: Trade, shadow: ShadowTrade
    ) -> ShadowComparison:
        """Compare a single shadow against the real trade."""
        real_pnl = trade.pnl or 0.0
        shadow_pnl = shadow.realized_pnl or shadow.unrealized_pnl or 0.0

        real_r = trade.r_multiple
        shadow_r = shadow.r_multiple

        r_diff = None
        if real_r is not None and shadow_r is not None:
            r_diff = shadow_r - real_r

        pnl_diff = shadow_pnl - real_pnl

        # Duration
        real_dur = None
        shadow_dur = None
        if trade.entry_time and trade.exit_time:
            real_dur = (trade.exit_time - trade.entry_time).total_seconds() / 60
        if shadow.entry_time and shadow.exit_time:
            shadow_dur = (shadow.exit_time - shadow.entry_time).total_seconds() / 60

        # Stop comparison
        stop_diff = None
        if trade.stop_loss is not None and shadow.original_stop is not None:
            stop_diff = shadow.original_stop - trade.stop_loss

        comparison = ShadowComparison(
            shadow_id=str(shadow.id),
            shadow_type=shadow.shadow_type,
            label=shadow.label or shadow.shadow_type,
            real_pnl=real_pnl,
            shadow_pnl=shadow_pnl,
            pnl_difference=pnl_diff,
            real_r=real_r,
            shadow_r=shadow_r,
            r_difference=r_diff,
            entry_price_diff=shadow.entry_price - trade.entry_price,
            exit_price_diff=(
                (shadow.exit_price - trade.exit_price)
                if shadow.exit_price and trade.exit_price else 0.0
            ),
            stop_diff=stop_diff,
            real_max_favorable=None,
            shadow_max_favorable=shadow.max_favorable,
            real_max_adverse=None,
            shadow_max_adverse=shadow.max_adverse,
            real_exit_reason="manual",
            shadow_exit_reason=shadow.exit_reason or "open",
            real_duration_min=real_dur,
            shadow_duration_min=shadow_dur,
            shadow_better=shadow_pnl > real_pnl,
            improvement_amount=max(0, shadow_pnl - real_pnl),
            improvement_r=max(0, r_diff) if r_diff is not None else None,
            shadow_partial_exits=shadow.partial_exits or [],
        )

        return comparison

    def _detect_divergences(
        self, trade: Trade, shadow: ShadowTrade
    ) -> list[TradeDivergence]:
        """Detect and create divergence records between real and shadow."""
        divergences = []
        real_pnl = trade.pnl or 0.0
        shadow_pnl = shadow.realized_pnl or 0.0

        # ── P&L Divergence (always record) ──
        pnl_diff = shadow_pnl - real_pnl
        if abs(pnl_diff) > 1.0:  # \$1 threshold
            divergences.append(TradeDivergence(
                id=uuid.uuid4(),
                shadow_trade_id=shadow.id,
                real_trade_id=trade.id,
                divergence_type="pnl_difference",
                description=(
                    f"P&L divergence: Real=${real_pnl:.2f} vs "
                    f"Shadow=${shadow_pnl:.2f} (diff=${pnl_diff:+.2f})"
                ),
                real_value=f"${real_pnl:.2f}",
                shadow_value=f"${shadow_pnl:.2f}",
                impact_amount=pnl_diff,
                impact_r=(
                    (shadow.r_multiple or 0) - (trade.r_multiple or 0)
                    if shadow.r_multiple is not None and trade.r_multiple is not None
                    else None
                ),
            ))

        # ── Stop Placement Divergence ──
        if (
            trade.stop_loss is not None
            and shadow.original_stop is not None
            and trade.stop_loss != shadow.original_stop
        ):
            diff = shadow.original_stop - trade.stop_loss
            divergences.append(TradeDivergence(
                id=uuid.uuid4(),
                shadow_trade_id=shadow.id,
                real_trade_id=trade.id,
                divergence_type="stop_placement",
                description=(
                    f"Stop divergence: Real={trade.stop_loss:.4f} vs "
                    f"Shadow={shadow.original_stop:.4f}"
                ),
                real_value=f"{trade.stop_loss:.4f}",
                shadow_value=f"{shadow.original_stop:.4f}",
                impact_amount=diff,
            ))

        # ── Exit Timing Divergence ──
        if trade.exit_time and shadow.exit_time:
            time_diff_minutes = abs(
                (trade.exit_time - shadow.exit_time).total_seconds()
            ) / 60
            if time_diff_minutes > 5:
                divergences.append(TradeDivergence(
                    id=uuid.uuid4(),
                    shadow_trade_id=shadow.id,
                    real_trade_id=trade.id,
                    divergence_type="exit_timing",
                    description=(
                        f"Exit timing divergence: {time_diff_minutes:.0f} minutes apart. "
                        f"Real exit: {trade.exit_time.strftime('%H:%M')}, "
                        f"Shadow exit: {shadow.exit_time.strftime('%H:%M')}"
                    ),
                    real_value=trade.exit_time.isoformat(),
                    shadow_value=shadow.exit_time.isoformat(),
                ))

        # ── Exit Reason Divergence ──
        if shadow.exit_reason and shadow.exit_reason != "real_trade_closed":
            divergences.append(TradeDivergence(
                id=uuid.uuid4(),
                shadow_trade_id=shadow.id,
                real_trade_id=trade.id,
                divergence_type="exit_reason",
                description=(
                    f"Shadow would have exited via '{shadow.exit_reason}' "
                    f"{'before' if shadow.exit_time and trade.exit_time and shadow.exit_time < trade.exit_time else 'after'} "
                    f"the real exit"
                ),
                real_value="manual_close",
                shadow_value=shadow.exit_reason,
                impact_amount=pnl_diff,
            ))

        return divergences

    def _compute_aggregates(
        self, report: ComparisonReport, trade: Trade
    ) -> None:
        """Compute aggregate metrics across all shadow comparisons."""
        if not report.comparisons:
            return

        best = max(report.comparisons, key=lambda c: c.shadow_pnl)
        worst = min(report.comparisons, key=lambda c: c.shadow_pnl)

        report.best_shadow_id = best.shadow_id
        report.best_shadow_pnl = best.shadow_pnl
        report.best_shadow_type = best.shadow_type
        report.worst_shadow_pnl = worst.shadow_pnl

        report.optimal_pnl = best.shadow_pnl
        report.opportunity_cost = max(0, best.shadow_pnl - report.real_pnl)

        if trade.risk_amount and trade.risk_amount > 0:
            report.opportunity_cost_r = report.opportunity_cost / trade.risk_amount

        # Execution quality: how close is real P&L to best shadow P&L
        pnl_range = best.shadow_pnl - worst.shadow_pnl
        if pnl_range > 0:
            position = (report.real_pnl - worst.shadow_pnl) / pnl_range
            report.execution_quality = max(0, min(100, position * 100))
        elif report.real_pnl >= best.shadow_pnl:
            report.execution_quality = 100.0
        else:
            report.execution_quality = 50.0

    def _generate_comparison_insights(
        self, report: ComparisonReport, trade: Trade
    ) -> None:
        """Generate human-readable insights from comparison data."""
        if not report.comparisons:
            return

        # Strategy shadow comparison
        strategy_comp = next(
            (c for c in report.comparisons if c.shadow_type == "strategy"), None
        )
        if strategy_comp:
            if strategy_comp.shadow_better:
                report.insights.append(
                    f"📊 Strategy rules would have yielded ${strategy_comp.improvement_amount:.2f} "
                    f"more ({strategy_comp.improvement_r:+.1f}R)" if strategy_comp.improvement_r
                    else f"📊 Strategy rules would have yielded ${strategy_comp.improvement_amount:.2f} more"
                )
            else:
                saved = abs(strategy_comp.pnl_difference)
                report.insights.append(
                    f"✅ Your execution beat the strategy shadow by ${saved:.2f}"
                )

        # Trailing variant insight
        trail_comp = next(
            (c for c in report.comparisons if c.shadow_type == "variant_trail"), None
        )
        if trail_comp and trail_comp.shadow_better:
            report.insights.append(
                f"🔄 A trailing stop would have captured ${trail_comp.shadow_pnl:.2f} "
                f"vs your ${report.real_pnl:.2f}"
            )

        # Tight vs wide stop comparison
        tight_comp = next(
            (c for c in report.comparisons if c.shadow_type == "variant_tight"), None
        )
        wide_comp = next(
            (c for c in report.comparisons if c.shadow_type == "variant_wide"), None
        )

        if tight_comp and wide_comp:
            if tight_comp.shadow_pnl > wide_comp.shadow_pnl and tight_comp.shadow_pnl > report.real_pnl:
                report.insights.append(
                    "📏 Tighter stop was optimal this time — "
                    "trade didn't need the extra room"
                )
            elif wide_comp.shadow_pnl > tight_comp.shadow_pnl and wide_comp.shadow_pnl > report.real_pnl:
                report.insights.append(
                    "📏 Wider stop was optimal — "
                    "price shook you out before reaching target"
                )

        # Opportunity cost
        if report.opportunity_cost > 0:
            report.insights.append(
                f"💰 Opportunity cost: ${report.opportunity_cost:.2f} "
                f"vs best shadow ({report.best_shadow_type})"
            )

        # Execution quality
        if report.execution_quality >= 80:
            report.insights.append(
                f"🎯 Execution quality: {report.execution_quality:.0f}/100 — Excellent"
            )
        elif report.execution_quality >= 50:
            report.insights.append(
                f"🎯 Execution quality: {report.execution_quality:.0f}/100 — Room for improvement"
            )
        else:
            report.insights.append(
                f"🎯 Execution quality: {report.execution_quality:.0f}/100 — "
                "Significant gap between actual and optimal execution"
            )


# ══════════════════════════════════════════════
#  PUBLIC FACADE  (used by api/routes/comparison.py)
# ══════════════════════════════════════════════

@dataclass
class TradeComparisonResult:
    """Flat result structure returned to the API layer."""
    real_trade_id: str
    symbol: str
    direction: str
    real_entry: Optional[float]
    real_exit: Optional[float]
    real_pnl: float
    real_r: Optional[float]
    real_hold_minutes: Optional[float]
    shadows: list[dict]
    divergences: list[dict]
    best_outcome: dict
    pnl_left_on_table: float
    key_insight: str


@dataclass
class SessionComparisonResult:
    """Session-level comparison result."""
    date: Any
    real_summary: dict
    mirror_summary: dict
    strategy_summary: dict
    divergence_summary: dict
    ai_insights: list[str]
    trade_comparisons: list[dict]


class ComparisonEngine:
    """
    High-level facade over ShadowComparisonEngine.
    Provides the interface expected by api/routes/comparison.py.
    """

    def __init__(self, db: DBSession):
        self._db = db
        self._engine = ShadowComparisonEngine()

    def compare_trade(self, trade_id: str) -> TradeComparisonResult:
        """Compare a real trade against all its shadows."""
        import uuid as _uuid
        try:
            tid = _uuid.UUID(trade_id)
        except ValueError:
            raise ValueError(f"Invalid trade_id: {trade_id}")

        trade = self._db.query(Trade).filter(Trade.id == tid).first()
        if not trade:
            raise ValueError(f"Trade {trade_id} not found")

        report = self._engine.compare(self._db, tid)

        hold_min = None
        if trade.entry_time and trade.exit_time:
            hold_min = (trade.exit_time - trade.entry_time).total_seconds() / 60

        shadows = []
        if report:
            for c in report.comparisons:
                shadows.append(c.to_dict())

        divergences = report.divergences if report else []

        best_outcome: dict = {}
        pnl_left = 0.0
        key_insight = "No shadow data available."

        if report and report.comparisons:
            best_outcome = {
                "shadow_type": report.best_shadow_type,
                "pnl": round(report.best_shadow_pnl, 2),
                "execution_quality": round(report.execution_quality, 1),
            }
            pnl_left = round(report.opportunity_cost, 2)
            key_insight = report.insights[0] if report.insights else "Execution matched shadow closely."

        return TradeComparisonResult(
            real_trade_id=trade_id,
            symbol=trade.symbol,
            direction=trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction),
            real_entry=trade.entry_price,
            real_exit=trade.exit_price,
            real_pnl=trade.pnl or 0.0,
            real_r=trade.r_multiple,
            real_hold_minutes=hold_min,
            shadows=shadows,
            divergences=divergences,
            best_outcome=best_outcome,
            pnl_left_on_table=pnl_left,
            key_insight=key_insight,
        )

    def compare_session(self, session_date) -> SessionComparisonResult:
        """Aggregate comparison for all trades on a given date."""
        from datetime import timedelta
        start = datetime.combine(session_date, datetime.min.time())
        end = start + timedelta(days=1)

        trades = self._db.query(Trade).filter(
            Trade.entry_time >= start,
            Trade.entry_time < end,
        ).all()

        real_pnl = sum(t.pnl or 0 for t in trades)
        wins = [t for t in trades if (t.pnl or 0) > 0]
        losses = [t for t in trades if (t.pnl or 0) < 0]

        real_summary = {
            "trade_count": len(trades),
            "total_pnl": round(real_pnl, 2),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": round(len(wins) / len(trades), 3) if trades else 0,
        }

        # Aggregate shadow summaries from individual comparisons
        mirror_pnl = 0.0
        strategy_pnl = 0.0
        all_divergences: list[dict] = []
        trade_comparisons: list[dict] = []

        for trade in trades:
            if trade.pnl is None:
                continue
            report = self._engine.compare(self._db, trade.id)
            if not report:
                continue
            mirror = next((c for c in report.comparisons if c.shadow_type == "mirror"), None)
            strategy = next((c for c in report.comparisons if c.shadow_type == "strategy"), None)
            if mirror:
                mirror_pnl += mirror.shadow_pnl
            if strategy:
                strategy_pnl += strategy.shadow_pnl
            all_divergences.extend(report.divergences)
            trade_comparisons.append({
                "trade_id": str(trade.id),
                "symbol": trade.symbol,
                "pnl": round(trade.pnl, 2),
                "opportunity_cost": round(report.opportunity_cost, 2),
            })

        return SessionComparisonResult(
            date=session_date,
            real_summary=real_summary,
            mirror_summary={"total_pnl": round(mirror_pnl, 2)},
            strategy_summary={"total_pnl": round(strategy_pnl, 2)},
            divergence_summary={"count": len(all_divergences)},
            ai_insights=[
                f"Session P&L: ₹{real_pnl:.0f}",
                f"Strategy shadow P&L: ₹{strategy_pnl:.0f}",
            ],
            trade_comparisons=trade_comparisons,
        )

    def get_cumulative_comparison(self, days: int = 30) -> dict:
        """Cumulative P&L comparison over the last N days."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        trades = self._db.query(Trade).filter(
            Trade.entry_time >= cutoff,
            Trade.pnl.isnot(None),
        ).all()

        real_total = sum(t.pnl for t in trades)
        shadows_total: dict[str, float] = {}
        for trade in trades:
            report = self._engine.compare(self._db, trade.id)
            if not report:
                continue
            for c in report.comparisons:
                shadows_total[c.shadow_type] = shadows_total.get(c.shadow_type, 0) + c.shadow_pnl

        return {
            "period_days": days,
            "trade_count": len(trades),
            "real_total_pnl": round(real_total, 2),
            "shadow_totals": {k: round(v, 2) for k, v in shadows_total.items()},
            "best_shadow_type": max(shadows_total, key=shadows_total.get) if shadows_total else None,
        }

    def get_divergence_patterns(self, days: int = 30) -> dict:
        """Analyze divergence patterns over the last N days."""
        from datetime import timedelta
        from collections import Counter
        cutoff = datetime.utcnow() - timedelta(days=days)
        trades = self._db.query(Trade).filter(
            Trade.entry_time >= cutoff,
            Trade.pnl.isnot(None),
        ).all()

        type_counts: Counter = Counter()
        total_impact = 0.0
        for trade in trades:
            report = self._engine.compare(self._db, trade.id)
            if not report:
                continue
            for div in report.divergences:
                type_counts[div.get("type", "unknown")] += 1
                total_impact += div.get("impact_amount", 0)

        return {
            "period_days": days,
            "divergence_type_counts": dict(type_counts.most_common()),
            "total_pnl_impact": round(total_impact, 2),
            "most_common_divergence": type_counts.most_common(1)[0][0] if type_counts else None,
        }