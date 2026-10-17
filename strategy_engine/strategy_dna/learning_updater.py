# strategy_engine/strategy_dna/learning_updater.py
"""
Updates Strategy DNA 'learned_statistics' and 'personal_insights'
from actual trade performance data.

This is what makes the system SMARTER over time.
After every N trades, this runs and updates what the system
knows about YOUR execution of each strategy.
"""

from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from core.models import Trade, JournalEntry, DailyPerformanceSummary
from core.enums import TradeStatus, TradeDirection, MarketRegime, SessionType
from shadow.models import ShadowTrade, ShadowType, TradeDivergence
from strategy_engine.strategy_dna.schema import StrategyDNA
from strategy_engine.strategy_dna.loader import StrategyDNALoader
import yaml
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StrategyPerformanceSnapshot:
    """Computed performance metrics for a strategy."""

    def __init__(self):
        self.total_trades: int = 0
        self.winners: int = 0
        self.losers: int = 0
        self.breakeven: int = 0
        self.total_pnl: float = 0
        self.total_r: float = 0
        self.r_values: List[float] = []
        self.pnl_values: List[float] = []
        self.hold_durations: List[float] = []

        # By regime
        self.by_regime: Dict[str, Dict] = defaultdict(
            lambda: {"trades": 0, "winners": 0, "total_r": 0, "r_values": []}
        )

        # By session
        self.by_session: Dict[str, Dict] = defaultdict(
            lambda: {"trades": 0, "winners": 0, "total_r": 0}
        )

        # By emotion
        self.by_pre_emotion: Dict[str, Dict] = defaultdict(
            lambda: {"trades": 0, "winners": 0, "total_r": 0}
        )

        # By time of day
        self.by_hour: Dict[int, Dict] = defaultdict(
            lambda: {"trades": 0, "winners": 0, "total_r": 0}
        )

        # Checklist correlation
        self.checklist_scores: List[float] = []
        self.checklist_vs_outcome: List[Tuple[float, float]] = []

        # Deviation tracking
        self.trades_with_overrides: int = 0
        self.override_outcomes: List[float] = []  # R values when override detected
        self.clean_outcomes: List[float] = []      # R values when no override

        # Shadow comparison
        self.real_vs_shadow_diff: List[float] = []

    @property
    def win_rate(self) -> float:
        return (self.winners / self.total_trades * 100) if self.total_trades > 0 else 0

    @property
    def avg_r(self) -> float:
        return np.mean(self.r_values) if self.r_values else 0

    @property
    def median_r(self) -> float:
        return np.median(self.r_values) if self.r_values else 0

    @property
    def expectancy(self) -> float:
        if not self.r_values:
            return 0
        avg_win = np.mean([r for r in self.r_values if r > 0]) if any(r > 0 for r in self.r_values) else 0
        avg_loss = abs(np.mean([r for r in self.r_values if r < 0])) if any(r < 0 for r in self.r_values) else 0
        wr = self.win_rate / 100
        return (wr * avg_win) - ((1 - wr) * avg_loss)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(r for r in self.r_values if r > 0)
        gross_loss = abs(sum(r for r in self.r_values if r < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    @property
    def best_regime(self) -> Optional[str]:
        if not self.by_regime:
            return None
        return max(
            self.by_regime.keys(),
            key=lambda k: (
                self.by_regime[k]["winners"] / max(self.by_regime[k]["trades"], 1)
                if self.by_regime[k]["trades"] >= 5 else 0
            )
        )

    @property
    def worst_regime(self) -> Optional[str]:
        if not self.by_regime:
            return None
        regimes_with_data = {
            k: v for k, v in self.by_regime.items() if v["trades"] >= 5
        }
        if not regimes_with_data:
            return None
        return min(
            regimes_with_data.keys(),
            key=lambda k: regimes_with_data[k]["winners"] / max(regimes_with_data[k]["trades"], 1)
        )

    @property
    def avg_hold_minutes(self) -> float:
        return np.mean(self.hold_durations) if self.hold_durations else 0

    @property
    def best_hour(self) -> Optional[int]:
        if not self.by_hour:
            return None
        hours_with_data = {k: v for k, v in self.by_hour.items() if v["trades"] >= 3}
        if not hours_with_data:
            return None
        return max(
            hours_with_data.keys(),
            key=lambda k: hours_with_data[k]["winners"] / max(hours_with_data[k]["trades"], 1)
        )

    @property
    def override_impact(self) -> Optional[float]:
        """Difference in avg R between clean trades and override trades."""
        if not self.clean_outcomes or not self.override_outcomes:
            return None
        return np.mean(self.clean_outcomes) - np.mean(self.override_outcomes)


class StrategyDNALearningUpdater:
    """
    Analyzes all trades and updates Strategy DNA with learned knowledge.

    Run this:
    - End of each trading day
    - Weekly for deeper analysis
    - On demand when reviewing performance
    """

    def __init__(self, db: Session, dna_loader: StrategyDNALoader):
        self.db = db
        self.dna_loader = dna_loader
        self.strategies = dna_loader.get_all()

    def update_all_strategies(
        self,
        lookback_days: int = 90
    ) -> Dict[str, StrategyPerformanceSnapshot]:
        """
        Update learned statistics for ALL strategies.
        Returns performance snapshots for each.
        """
        results = {}

        for name, dna in self.strategies.items():
            try:
                snapshot = self._compute_snapshot(name, lookback_days)
                self._update_dna_file(name, dna, snapshot)
                self._update_db_strategy(name, snapshot)
                results[name] = snapshot

                logger.info(
                    f"Updated DNA for {dna.display_name}: "
                    f"{snapshot.total_trades} trades, "
                    f"{snapshot.win_rate:.1f}% WR, "
                    f"{snapshot.avg_r:.2f} avg R, "
                    f"best regime: {snapshot.best_regime}"
                )
            except Exception as e:
                logger.error(f"Failed to update {name}: {e}", exc_info=True)

        return results

    def update_single_strategy(
        self,
        strategy_name: str,
        lookback_days: int = 90
    ) -> Optional[StrategyPerformanceSnapshot]:
        """Update learned statistics for a single strategy."""
        dna = self.strategies.get(strategy_name)
        if not dna:
            return None

        snapshot = self._compute_snapshot(strategy_name, lookback_days)
        self._update_dna_file(strategy_name, dna, snapshot)
        self._update_db_strategy(strategy_name, snapshot)
        return snapshot

    def generate_learning_report(
        self,
        strategy_name: str,
        lookback_days: int = 90
    ) -> Dict:
        """
        Generate a comprehensive learning report for a strategy.
        Used by: Coaching Agent, Weekly Review, Dashboard
        """
        dna = self.strategies.get(strategy_name)
        snapshot = self._compute_snapshot(strategy_name, lookback_days)

        report = {
            "strategy": strategy_name,
            "display_name": dna.display_name if dna else strategy_name,
            "period_days": lookback_days,
            "sample_size": snapshot.total_trades,

            # Core metrics
            "performance": {
                "win_rate": round(snapshot.win_rate, 1),
                "avg_r": round(snapshot.avg_r, 2),
                "median_r": round(snapshot.median_r, 2),
                "expectancy": round(snapshot.expectancy, 3),
                "profit_factor": round(snapshot.profit_factor, 2),
                "total_pnl": round(snapshot.total_pnl, 2),
                "avg_hold_minutes": round(snapshot.avg_hold_minutes, 1),
            },

            # Regime analysis
            "regime_performance": {},

            # Session analysis
            "session_performance": {},

            # Time analysis
            "best_hour": snapshot.best_hour,

            # Behavioral
            "discipline": {
                "trades_with_overrides": snapshot.trades_with_overrides,
                "override_pct": (
                    snapshot.trades_with_overrides / max(snapshot.total_trades, 1) * 100
                ),
                "override_impact_r": snapshot.override_impact,
                "avg_checklist_score": (
                    np.mean(snapshot.checklist_scores)
                    if snapshot.checklist_scores else None
                ),
            },

            # Discoveries
            "discoveries": [],
        }

        # Regime breakdown
        for regime, data in snapshot.by_regime.items():
            if data["trades"] >= 3:
                wr = data["winners"] / data["trades"] * 100
                avg_r = np.mean(data["r_values"]) if data["r_values"] else 0
                report["regime_performance"][regime] = {
                    "trades": data["trades"],
                    "win_rate": round(wr, 1),
                    "avg_r": round(avg_r, 2),
                }

        # Session breakdown
        for session, data in snapshot.by_session.items():
            if data["trades"] >= 3:
                wr = data["winners"] / data["trades"] * 100
                report["session_performance"][session] = {
                    "trades": data["trades"],
                    "win_rate": round(wr, 1),
                }

        # Auto-discover insights
        report["discoveries"] = self._discover_insights(dna, snapshot)

        return report

    def _compute_snapshot(
        self,
        strategy_name: str,
        lookback_days: int
    ) -> StrategyPerformanceSnapshot:
        """Compute full performance snapshot from trade data."""
        snapshot = StrategyPerformanceSnapshot()
        cutoff = datetime.now() - timedelta(days=lookback_days)

        # Query closed trades for this strategy
        trades = self.db.query(Trade).filter(
            and_(
                Trade.strategy_name == strategy_name,
                Trade.status == TradeStatus.CLOSED,
                Trade.entry_time >= cutoff,
            )
        ).all()

        for trade in trades:
            snapshot.total_trades += 1

            # PnL
            pnl = trade.pnl
            if pnl is not None:
                snapshot.pnl_values.append(pnl)
                snapshot.total_pnl += pnl
                if pnl > 0:
                    snapshot.winners += 1
                elif pnl < 0:
                    snapshot.losers += 1
                else:
                    snapshot.breakeven += 1

            # R multiple
            if trade.actual_r_multiple is not None:
                snapshot.r_values.append(trade.actual_r_multiple)
                snapshot.total_r += trade.actual_r_multiple

            # Hold duration
            if trade.hold_duration_minutes:
                snapshot.hold_durations.append(trade.hold_duration_minutes)

            # By regime
            if trade.market_regime:
                regime_key = trade.market_regime.value if hasattr(trade.market_regime, 'value') else str(trade.market_regime)
                regime_data = snapshot.by_regime[regime_key]
                regime_data["trades"] += 1
                if pnl and pnl > 0:
                    regime_data["winners"] += 1
                if trade.actual_r_multiple is not None:
                    regime_data["total_r"] += trade.actual_r_multiple
                    regime_data["r_values"].append(trade.actual_r_multiple)

            # By session
            if trade.session_type:
                session_key = trade.session_type.value if hasattr(trade.session_type, 'value') else str(trade.session_type)
                session_data = snapshot.by_session[session_key]
                session_data["trades"] += 1
                if pnl and pnl > 0:
                    session_data["winners"] += 1
                if trade.actual_r_multiple is not None:
                    session_data["total_r"] += trade.actual_r_multiple

            # By hour
            if trade.entry_time:
                hour = trade.entry_time.hour
                hour_data = snapshot.by_hour[hour]
                hour_data["trades"] += 1
                if pnl and pnl > 0:
                    hour_data["winners"] += 1
                if trade.actual_r_multiple is not None:
                    hour_data["total_r"] += trade.actual_r_multiple

            # By pre-trade emotion
            if trade.journal_entry and trade.journal_entry.pre_trade_emotion:
                emo = trade.journal_entry.pre_trade_emotion.value
                emo_data = snapshot.by_pre_emotion[emo]
                emo_data["trades"] += 1
                if pnl and pnl > 0:
                    emo_data["winners"] += 1
                if trade.actual_r_multiple is not None:
                    emo_data["total_r"] += trade.actual_r_multiple

            # Checklist scores & overrides from tags
            if trade.tags:
                cs = trade.tags.get("checklist_score")
                if cs is not None:
                    snapshot.checklist_scores.append(cs)
                    if trade.actual_r_multiple is not None:
                        snapshot.checklist_vs_outcome.append((cs, trade.actual_r_multiple))

                overrides = trade.tags.get("overrides", [])
                if overrides:
                    snapshot.trades_with_overrides += 1
                    if trade.actual_r_multiple is not None:
                        snapshot.override_outcomes.append(trade.actual_r_multiple)
                else:
                    if trade.actual_r_multiple is not None:
                        snapshot.clean_outcomes.append(trade.actual_r_multiple)

        # Shadow comparison
        for trade in trades:
            mirror = self.db.query(ShadowTrade).filter(
                and_(
                    ShadowTrade.real_trade_id == trade.id,
                    ShadowTrade.shadow_type == ShadowType.MIRROR,
                    ShadowTrade.status == TradeStatus.CLOSED,
                )
            ).first()

            if mirror and trade.pnl is not None and mirror.pnl is not None:
                snapshot.real_vs_shadow_diff.append(trade.pnl - mirror.pnl)

        return snapshot

    def _update_dna_file(
        self,
        strategy_name: str,
        dna: StrategyDNA,
        snapshot: StrategyPerformanceSnapshot
    ):
        """Update the YAML file with learned statistics."""
        file_path = Path(self.dna_loader.strategy_dir) / f"{strategy_name}.yaml"
        if not file_path.exists():
            return

        with open(file_path) as f:
            raw = yaml.safe_load(f)

        # Update learned_statistics section
        if "personal_insights" not in raw:
            raw["personal_insights"] = {}

        raw["personal_insights"]["learned_statistics"] = {
            "total_trades": snapshot.total_trades,
            "win_rate": round(snapshot.win_rate, 1),
            "avg_r": round(snapshot.avg_r, 2),
            "median_r": round(snapshot.median_r, 2),
            "expectancy": round(snapshot.expectancy, 3),
            "profit_factor": round(snapshot.profit_factor, 2),
            "best_regime": snapshot.best_regime,
            "worst_regime": snapshot.worst_regime,
            "best_hour": snapshot.best_hour,
            "avg_hold_minutes": round(snapshot.avg_hold_minutes, 1),
            "override_impact_r": (
                round(snapshot.override_impact, 2) if snapshot.override_impact else None
            ),
            "avg_checklist_score": (
                round(np.mean(snapshot.checklist_scores), 1)
                if snapshot.checklist_scores else None
            ),
            "last_updated": datetime.now().isoformat(),
        }

        # Write back
        with open(file_path, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info(f"Updated DNA file: {file_path}")

    def _update_db_strategy(
        self,
        strategy_name: str,
        snapshot: StrategyPerformanceSnapshot
    ):
        """Update Strategy table in database."""
        from core.models import Strategy

        strategy = self.db.query(Strategy).filter(
            Strategy.name == strategy_name
        ).first()

        if strategy:
            strategy.total_trades = snapshot.total_trades
            strategy.win_rate = snapshot.win_rate
            strategy.avg_r_multiple = snapshot.avg_r
            strategy.expectancy = snapshot.expectancy
            self.db.flush()

    def _discover_insights(
        self,
        dna: Optional[StrategyDNA],
        snapshot: StrategyPerformanceSnapshot
    ) -> List[str]:
        """Auto-discover insights from the data."""
        insights = []

        if snapshot.total_trades < 10:
            insights.append(
                f"⚠️ Only {snapshot.total_trades} trades — need 20+ for reliable statistics."
            )
            return insights

        # ── Win rate insights ──
        if snapshot.win_rate >= 60:
            insights.append(
                f"✅ Strong win rate: {snapshot.win_rate:.1f}%. "
                f"This strategy has a clear edge."
            )
        elif snapshot.win_rate < 40:
            insights.append(
                f"🔴 Low win rate: {snapshot.win_rate:.1f}%. "
                f"Review entry criteria or consider retiring this strategy."
            )

        # ── Regime discovery ──
        if snapshot.best_regime and snapshot.worst_regime:
            best_data = snapshot.by_regime.get(snapshot.best_regime, {})
            worst_data = snapshot.by_regime.get(snapshot.worst_regime, {})

            best_wr = best_data["winners"] / max(best_data["trades"], 1) * 100
            worst_wr = worst_data["winners"] / max(worst_data["trades"], 1) * 100

            if best_wr - worst_wr > 20:
                insights.append(
                    f"📊 REGIME EDGE: Win rate is {best_wr:.0f}% in "
                    f"{snapshot.best_regime} but only {worst_wr:.0f}% in "
                    f"{snapshot.worst_regime}. "
                    f"Consider SKIPPING this strategy in {snapshot.worst_regime}."
                )

        # ── Override impact ──
        if snapshot.override_impact is not None and abs(snapshot.override_impact) > 0.3:
            if snapshot.override_impact > 0:
                insights.append(
                    f"🧠 DISCIPLINE PAYS: Clean trades average "
                    f"{np.mean(snapshot.clean_outcomes):.2f}R vs "
                    f"override trades {np.mean(snapshot.override_outcomes):.2f}R. "
                    f"Following rules gains you {snapshot.override_impact:.2f}R per trade."
                )
            else:
                insights.append(
                    f"🤔 INTERESTING: Your overrides actually perform BETTER "
                    f"({np.mean(snapshot.override_outcomes):.2f}R) than clean trades "
                    f"({np.mean(snapshot.clean_outcomes):.2f}R). "
                    f"Your intuition may be ahead of your rules — review and update strategy."
                )

        # ── Checklist correlation ──
        if len(snapshot.checklist_vs_outcome) >= 10:
            high_score = [r for cs, r in snapshot.checklist_vs_outcome if cs >= 80]
            low_score = [r for cs, r in snapshot.checklist_vs_outcome if cs < 60]

            if high_score and low_score:
                high_avg = np.mean(high_score)
                low_avg = np.mean(low_score)
                if high_avg > low_avg + 0.3:
                    insights.append(
                        f"✅ CHECKLIST WORKS: High checklist trades avg "
                        f"{high_avg:.2f}R vs low checklist {low_avg:.2f}R. "
                        f"Your checklist is effective — keep using it."
                    )

        # ── Emotion impact ──
        risky_emotions = {"revenge", "fomo", "frustrated", "anxious", "euphoric"}
        calm_data = snapshot.by_pre_emotion.get("calm", {"trades": 0, "winners": 0, "total_r": 0})
        risky_trades = 0
        risky_r_total = 0
        for emo in risky_emotions:
            data = snapshot.by_pre_emotion.get(emo, {"trades": 0, "total_r": 0})
            risky_trades += data["trades"]
            risky_r_total += data["total_r"]

        if calm_data["trades"] >= 5 and risky_trades >= 5:
            calm_avg_r = calm_data["total_r"] / calm_data["trades"]
            risky_avg_r = risky_r_total / risky_trades
            if calm_avg_r > risky_avg_r + 0.3:
                insights.append(
                    f"🧘 EMOTIONAL IMPACT: Calm trades average {calm_avg_r:.2f}R "
                    f"vs emotional trades {risky_avg_r:.2f}R. "
                    f"Emotional state costs you {calm_avg_r - risky_avg_r:.2f}R per trade."
                )

        # ── Shadow comparison ──
        if snapshot.real_vs_shadow_diff:
            avg_diff = np.mean(snapshot.real_vs_shadow_diff)
            if avg_diff < -50:  # losing more than \$50 per trade vs rules
                insights.append(
                    f"📉 SHADOW GAP: Your real trades underperform rule-based "
                    f"shadow trades by ${abs(avg_diff):.0f} on average. "
                    f"Following rules more closely would improve results."
                )
            elif avg_diff > 50:
                insights.append(
                    f"📈 OUTPERFORMING RULES: Your real trades beat shadow trades "
                    f"by ${avg_diff:.0f} on average. Your discretion adds value."
                )

        # ── Best time of day ──
        if snapshot.best_hour is not None:
            hour_data = snapshot.by_hour[snapshot.best_hour]
            wr = hour_data["winners"] / max(hour_data["trades"], 1) * 100
            insights.append(
                f"⏰ BEST HOUR: {snapshot.best_hour}:00 has {wr:.0f}% win rate "
                f"({hour_data['trades']} trades). Focus your trading here."
            )

        # ── DNA alignment check ──
        if dna:
            # Check if actual best regime matches DNA expectation
            if snapshot.best_regime and snapshot.best_regime not in dna.best_regimes:
                insights.append(
                    f"🔄 UPDATE DNA: Your actual best regime is {snapshot.best_regime}, "
                    f"but your strategy DNA says best in {dna.best_regimes}. "
                    f"Consider updating your strategy profile."
                )

        return insights