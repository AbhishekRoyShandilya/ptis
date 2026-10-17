# competition/missed_opportunity_detector.py
"""
Missed Opportunity Detector
============================
Identifies profitable setups that the system's autonomous scanner detected
(generated signals with sufficient conviction) but the trader did NOT take.

This is the most powerful coaching feedback loop. It answers:
"What did you miss, and what would it have been worth?"

Detection logic:
1. Find all system signals generated during a period
2. Find all trader trades during the same period
3. Match by symbol and time window
4. Any signal NOT matched to a trader trade = missed opportunity
5. Score missed opportunities by how profitable the system's shadow was

Categories:
- HIGH_CONVICTION_MISS: conviction >= 0.70, system shadow was profitable
- VARIANT_MATCH_MISS: matched a user-taught variant, user didn't trade it
- REGIME_ALIGNED_MISS: signal was regime-aligned, perfect conditions
- MARGINAL_SKIP: conviction was borderline, reasonable to skip
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session as DBSession

from core.models import Trade
from shadow.models import ShadowTrade

logger = logging.getLogger(__name__)

# Time window for matching: if a trader trade occurs within N minutes
# of a system signal on the same symbol, it's considered "taken"
MATCH_WINDOW_MINUTES = 30


@dataclass
class MissedOpportunity:
    """A single missed trade opportunity."""
    signal_id: str
    symbol: str
    direction: str
    timestamp: datetime

    # Signal details
    conviction_score: float
    conviction_band: str
    matched_variant: Optional[str]
    variant_provenance: Optional[str]
    regime_match_score: float

    # Shadow outcome (what would have happened)
    shadow_id: Optional[str]
    shadow_pnl: Optional[float]
    shadow_r: Optional[float]
    shadow_exit_reason: Optional[str]

    # Classification
    category: str  # "high_conviction_miss" | "variant_match_miss" | etc.
    severity: str  # "critical" | "notable" | "informational"

    # Coaching context
    sko_name: str = ""
    narrative: str = ""  # Why this was a good setup
    what_if_pnl: float = 0.0  # Estimated P&L if trader had taken it

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "conviction_score": round(self.conviction_score, 4),
            "conviction_band": self.conviction_band,
            "matched_variant": self.matched_variant,
            "variant_provenance": self.variant_provenance,
            "regime_match_score": round(self.regime_match_score, 4),
            "shadow_id": self.shadow_id,
            "shadow_pnl": round(self.shadow_pnl, 2) if self.shadow_pnl else None,
            "shadow_r": round(self.shadow_r, 2) if self.shadow_r else None,
            "shadow_exit_reason": self.shadow_exit_reason,
            "category": self.category,
            "severity": self.severity,
            "sko_name": self.sko_name,
            "narrative": self.narrative,
            "what_if_pnl": round(self.what_if_pnl, 2),
        }


@dataclass
class MissedOpportunityReport:
    """Complete missed opportunity analysis for a period."""
    period_start: datetime
    period_end: datetime

    total_system_signals: int = 0
    trader_trades_taken: int = 0
    missed_count: int = 0

    missed_opportunities: List[MissedOpportunity] = field(default_factory=list)

    # Aggregates
    total_missed_pnl: float = 0.0  # sum of all missed shadow P&L (profitable only)
    avg_missed_conviction: float = 0.0
    critical_misses: int = 0
    notable_misses: int = 0

    # Top misses
    biggest_miss_pnl: float = 0.0
    biggest_miss_symbol: str = ""
    biggest_miss_r: float = 0.0

    # Pattern analysis
    most_missed_symbol: str = ""
    most_missed_time_bucket: str = ""
    most_missed_variant: str = ""

    # Coaching insights
    insights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "summary": {
                "total_system_signals": self.total_system_signals,
                "trader_trades_taken": self.trader_trades_taken,
                "missed_count": self.missed_count,
                "total_missed_pnl": round(self.total_missed_pnl, 2),
                "avg_missed_conviction": round(self.avg_missed_conviction, 4),
                "critical_misses": self.critical_misses,
                "notable_misses": self.notable_misses,
            },
            "biggest_miss": {
                "pnl": round(self.biggest_miss_pnl, 2),
                "symbol": self.biggest_miss_symbol,
                "r_multiple": round(self.biggest_miss_r, 2),
            },
            "patterns": {
                "most_missed_symbol": self.most_missed_symbol,
                "most_missed_time_bucket": self.most_missed_time_bucket,
                "most_missed_variant": self.most_missed_variant,
            },
            "insights": self.insights,
            "opportunities": [m.to_dict() for m in self.missed_opportunities],
        }


class MissedOpportunityDetector:
    """
    Detects and analyzes missed trading opportunities.

    Usage:
        detector = MissedOpportunityDetector(db)
        report = detector.analyze_week()
        report = detector.analyze_period(start, end)
    """

    def __init__(self, db: DBSession):
        self._db = db

    def analyze_week(self, week_offset: int = 0) -> MissedOpportunityReport:
        """Analyze missed opportunities for a specific week."""
        today = datetime.now(timezone.utc).date()
        start_of_week = today - timedelta(days=today.weekday()) + timedelta(weeks=week_offset)
        end_of_week = start_of_week + timedelta(days=6)

        start = datetime.combine(start_of_week, datetime.min.time()).replace(tzinfo=timezone.utc)
        end = datetime.combine(end_of_week, datetime.max.time()).replace(tzinfo=timezone.utc)

        return self.analyze_period(start, end)

    def analyze_period(
        self, start: datetime, end: datetime
    ) -> MissedOpportunityReport:
        """
        Analyze missed opportunities for any date range.

        Algorithm:
        1. Get all autonomous system shadows (= signals the system acted on)
        2. Get all trader trades in the same period
        3. For each system shadow, check if trader took a matching trade
        4. Unmatched system shadows with positive P&L = missed opportunities
        """
        report = MissedOpportunityReport(period_start=start, period_end=end)

        # 1. All autonomous system shadows
        system_shadows = (
            self._db.query(ShadowTrade)
            .filter(
                ShadowTrade.shadow_type == "autonomous",
                ShadowTrade.entry_time >= start,
                ShadowTrade.entry_time <= end,
            )
            .all()
        )
        report.total_system_signals = len(system_shadows)

        if not system_shadows:
            report.insights.append("No system signals generated in this period.")
            return report

        # 2. All trader trades
        trader_trades = (
            self._db.query(Trade)
            .filter(Trade.entry_time >= start, Trade.entry_time <= end)
            .all()
        )
        report.trader_trades_taken = len(trader_trades)

        # 3. Build trader trade index by symbol
        trader_index: Dict[str, List[Trade]] = defaultdict(list)
        for t in trader_trades:
            trader_index[t.symbol].append(t)

        # 4. Match and detect misses
        missed: List[MissedOpportunity] = []

        for shadow in system_shadows:
            config = shadow.config or {}

            # Check if trader took a matching trade
            symbol_trades = trader_index.get(shadow.symbol, [])
            was_taken = any(
                self._is_time_match(shadow.entry_time, t.entry_time)
                for t in symbol_trades
            )

            if was_taken:
                continue  # Trader took this one

            # This is a missed opportunity
            shadow_pnl = config.get("journal_pnl", shadow.realized_pnl)
            shadow_r = config.get("journal_r_multiple", shadow.r_multiple)
            conviction = config.get("conviction_score", 0.0)
            conviction_band = config.get("conviction_band", "unknown")
            matched_variant = config.get("matched_variant")
            variant_provenance = config.get("variant_provenance")
            regime_match = config.get("regime_match_score", 0.0)

            # Classify the miss
            category, severity = self._classify_miss(
                conviction, shadow_pnl, matched_variant, variant_provenance, regime_match
            )

            # Build narrative
            narrative = self._build_narrative(
                shadow, config, shadow_pnl, shadow_r, conviction_band
            )

            miss = MissedOpportunity(
                signal_id=str(shadow.id),
                symbol=shadow.symbol,
                direction=shadow.direction if isinstance(shadow.direction, str) else shadow.direction.value,
                timestamp=shadow.entry_time,
                conviction_score=conviction,
                conviction_band=conviction_band,
                matched_variant=matched_variant,
                variant_provenance=variant_provenance,
                regime_match_score=regime_match,
                shadow_id=str(shadow.id),
                shadow_pnl=shadow_pnl,
                shadow_r=shadow_r,
                shadow_exit_reason=config.get("journal_exit_reason", shadow.exit_reason),
                category=category,
                severity=severity,
                sko_name=config.get("sko_name", ""),
                narrative=narrative,
                what_if_pnl=shadow_pnl if shadow_pnl and shadow_pnl > 0 else 0,
            )
            missed.append(miss)

        # Sort by severity then P&L impact
        severity_order = {"critical": 0, "notable": 1, "informational": 2}
        missed.sort(key=lambda m: (severity_order.get(m.severity, 3), -(m.shadow_pnl or 0)))

        report.missed_opportunities = missed
        report.missed_count = len(missed)

        # Aggregate
        self._compute_aggregates(report)
        self._detect_patterns(report)
        self._generate_insights(report)

        return report

    def _is_time_match(self, sys_time: Optional[datetime], trader_time: Optional[datetime]) -> bool:
        """Check if two timestamps are within the match window."""
        if not sys_time or not trader_time:
            return False
        delta = abs((sys_time - trader_time).total_seconds())
        return delta <= MATCH_WINDOW_MINUTES * 60

    def _classify_miss(
        self,
        conviction: float,
        shadow_pnl: Optional[float],
        matched_variant: Optional[str],
        variant_provenance: Optional[str],
        regime_match: float,
    ) -> tuple:
        """Classify the category and severity of a missed opportunity."""
        profitable = shadow_pnl is not None and shadow_pnl > 0

        if conviction >= 0.70 and profitable:
            return "high_conviction_miss", "critical"

        if matched_variant and variant_provenance == "user_taught" and profitable:
            return "variant_match_miss", "critical"

        if regime_match > 0.7 and conviction >= 0.60 and profitable:
            return "regime_aligned_miss", "notable"

        if conviction >= 0.50 and profitable:
            return "moderate_conviction_miss", "notable"

        return "marginal_skip", "informational"

    def _build_narrative(
        self,
        shadow: ShadowTrade,
        config: dict,
        shadow_pnl: Optional[float],
        shadow_r: Optional[float],
        conviction_band: str,
    ) -> str:
        """Build a human-readable narrative for the missed opportunity."""
        parts = []

        sko_name = config.get("sko_name", "")
        variant = config.get("matched_variant")
        direction = shadow.direction if isinstance(shadow.direction, str) else shadow.direction.value

        parts.append(
            f"System identified a {conviction_band.upper()} conviction "
            f"{direction} setup on {shadow.symbol}"
        )

        if sko_name:
            parts.append(f"using strategy '{sko_name}'")

        if variant:
            prov = config.get("variant_provenance", "system")
            variant_label = f"'{variant}'"
            if prov == "user_taught":
                variant_label += " (your own taught variant!)"
            parts.append(f"matching variant {variant_label}")

        if shadow_pnl is not None and shadow_pnl > 0:
            parts.append(
                f"The virtual trade closed at ₹{shadow_pnl:,.0f} profit"
            )
            if shadow_r is not None:
                parts.append(f"({shadow_r:+.1f}R)")

        return ". ".join(parts) + "."

    def _compute_aggregates(self, report: MissedOpportunityReport) -> None:
        """Compute aggregate stats from missed opportunities."""
        if not report.missed_opportunities:
            return

        profitable_misses = [
            m for m in report.missed_opportunities
            if m.shadow_pnl is not None and m.shadow_pnl > 0
        ]

        report.total_missed_pnl = sum(m.shadow_pnl for m in profitable_misses)
        convictions = [m.conviction_score for m in report.missed_opportunities]
        report.avg_missed_conviction = sum(convictions) / len(convictions)

        report.critical_misses = sum(
            1 for m in report.missed_opportunities if m.severity == "critical"
        )
        report.notable_misses = sum(
            1 for m in report.missed_opportunities if m.severity == "notable"
        )

        if profitable_misses:
            biggest = max(profitable_misses, key=lambda m: m.shadow_pnl or 0)
            report.biggest_miss_pnl = biggest.shadow_pnl or 0
            report.biggest_miss_symbol = biggest.symbol
            report.biggest_miss_r = biggest.shadow_r or 0

    def _detect_patterns(self, report: MissedOpportunityReport) -> None:
        """Detect patterns in missed opportunities."""
        if not report.missed_opportunities:
            return

        # Most missed symbol
        sym_counts: Dict[str, int] = defaultdict(int)
        for m in report.missed_opportunities:
            sym_counts[m.symbol] += 1
        if sym_counts:
            report.most_missed_symbol = max(sym_counts, key=sym_counts.get)

        # Most missed time bucket
        time_counts: Dict[str, int] = defaultdict(int)
        for m in report.missed_opportunities:
            if m.timestamp:
                hour = m.timestamp.hour
                if hour < 10:
                    bucket = "09:15-10:00"
                elif hour < 12:
                    bucket = "10:00-12:00"
                elif hour < 14:
                    bucket = "12:00-14:00"
                else:
                    bucket = "14:00-15:30"
                time_counts[bucket] += 1
        if time_counts:
            report.most_missed_time_bucket = max(time_counts, key=time_counts.get)

        # Most missed variant
        variant_counts: Dict[str, int] = defaultdict(int)
        for m in report.missed_opportunities:
            if m.matched_variant:
                variant_counts[m.matched_variant] += 1
        if variant_counts:
            report.most_missed_variant = max(variant_counts, key=variant_counts.get)

    def _generate_insights(self, report: MissedOpportunityReport) -> None:
        """Generate coaching insights from missed opportunity data."""
        insights = []

        if report.missed_count == 0:
            insights.append(
                "🎯 Perfect coverage! You didn't miss any system-identified setups."
            )
        else:
            coverage = 1 - (
                report.missed_count / report.total_system_signals
            ) if report.total_system_signals > 0 else 0

            insights.append(
                f"📊 Coverage: {coverage:.0%} — you took {report.trader_trades_taken} trades, "
                f"missed {report.missed_count} system setups"
            )

        if report.total_missed_pnl > 0:
            insights.append(
                f"💰 Total missed profit: ₹{report.total_missed_pnl:,.0f} from "
                f"{report.critical_misses + report.notable_misses} actionable setups"
            )

        if report.biggest_miss_pnl > 0:
            insights.append(
                f"🏆 Biggest miss: {report.biggest_miss_symbol} "
                f"(₹{report.biggest_miss_pnl:,.0f}, {report.biggest_miss_r:+.1f}R)"
            )

        if report.most_missed_symbol:
            insights.append(
                f"🔍 Most frequently missed: {report.most_missed_symbol} — "
                f"consider adding it to your active watchlist"
            )

        if report.most_missed_time_bucket:
            insights.append(
                f"⏰ Most misses during {report.most_missed_time_bucket} — "
                f"are you less attentive during this window?"
            )

        if report.critical_misses > 3:
            insights.append(
                f"⚠️ {report.critical_misses} critical misses (high conviction + profitable). "
                f"These are your biggest improvement lever."
            )

        # Check for user-taught variant misses (most painful)
        user_variant_misses = [
            m for m in report.missed_opportunities
            if m.variant_provenance == "user_taught" and m.severity == "critical"
        ]
        if user_variant_misses:
            insights.append(
                f"😤 You missed {len(user_variant_misses)} setups matching YOUR OWN taught variants. "
                f"The system learned from you, then you didn't take the trades!"
            )

        report.insights = insights
