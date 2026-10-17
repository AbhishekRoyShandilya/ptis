# coplanner/strategy_regime_matrix.py
"""
Strategy x Regime Performance Matrix — THE STATISTICAL BACKBONE.

Builds and maintains:

Strategy     | Trending_Up | Trending_Down | Ranging | Volatile
─────────────┼─────────────┼───────────────┼─────────┼─────────
Breakout     | 62% / 2.1RR | 35% / 0.8RR  | 38%     | 55%
Pullback     | 48%         | 52%           | 58%     | 42%
Reversal     | 35%         | 55%           | 52%     | 60%

Built FROM USER'S OWN TRADE HISTORY — not theoretical.

DATA TRUST LAYER:
  Tier 1 (AVT autonomous)   — weight 1.0 (60% of blended score)
  Tier 2 (user clean trades) — weight 0.7 (30%)
  Tier 3 (all user trades)   — weight 0.3 (10%)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from coplanner.db_interface import CoPlannerDB
from coplanner.models import (
    AvoidanceAdvice,
    ConfidenceLevel,
    DataTrustTier,
    RegimePerformanceStats,
    TradeRecord,
)

logger = logging.getLogger(__name__)

# Data trust weights
TRUST_WEIGHTS = {
    DataTrustTier.AVT_AUTONOMOUS: 1.0,
    DataTrustTier.USER_QUALITY: 0.70,
    DataTrustTier.USER_ALL: 0.30,
}


class StrategyRegimeMatrix:
    """
    Core matrix: how does each strategy perform in each market regime?

    Usage:
        matrix = StrategyRegimeMatrix(db)
        matrix.build()
        stats = matrix.query('breakout', 'trending_up')
        advice = matrix.should_avoid('breakout', 'ranging')
        best = matrix.get_best_strategies_for_regime('ranging')

    When a DataTrustLayer is connected, uses weighted/blended data
    so emotional trades don't corrupt strategy evaluation.
    """

    # Configurable thresholds
    AVOID_WR_THRESHOLD = 0.42
    AVOID_MIN_SAMPLE = 8
    MIN_SAMPLE_FOR_RANKING = 5
    STRONG_WR_THRESHOLD = 0.55
    MATRIX_STALENESS_DAYS = 1

    def __init__(
        self,
        db: CoPlannerDB,
        regime_builder: Any = None,
        data_trust_layer: Any = None,
    ):
        self.db = db
        self.regime_builder = regime_builder
        self.trust_layer = data_trust_layer
        self.matrix: Dict[str, Dict[str, RegimePerformanceStats]] = {}
        self._last_build_time: Optional[datetime] = None
        self._all_strategies: List[str] = []
        self._all_regimes: List[str] = []

    # ─────────────────────────────────────────────
    # BUILD
    # ─────────────────────────────────────────────

    def build(self, force: bool = False) -> Dict[str, Any]:
        """
        Build the full strategy x regime matrix from trade history.
        Uses trust layer if available, falls back to raw trades.
        EXPENSIVE — run periodically (daily), not per-request.
        """
        if not force and self._is_cache_fresh():
            return {"status": "cached", "strategies": len(self._all_strategies)}

        if self.trust_layer:
            return self._build_from_trust_layer()
        return self._build_from_raw_trades()

    def _build_from_trust_layer(self) -> Dict[str, Any]:
        """Build matrix using DataTrustLayer — weighted by trust scores."""
        logger.info("Building matrix from TRUST LAYER (clean data)...")

        availability = self.trust_layer.get_data_availability()
        logger.info("Data availability: %s", availability["mode"])

        if availability["mode"] == "COLD_START":
            logger.warning("No data available — matrix will be empty")
            return {"status": "empty", "trades": 0, "data_mode": "COLD_START"}

        tiered_trades = self.trust_layer.get_trusted_trades()

        # Collect unique strategy x regime combinations
        combos = set()
        for tiered in tiered_trades:
            strategy = self._normalize(tiered.trade.strategy_name)
            regime = self._normalize(tiered.trade.regime_at_entry)
            if strategy and regime:
                combos.add((strategy, regime))

        self.matrix = {}
        self._all_strategies = []
        self._all_regimes = []

        for strategy, regime in combos:
            if strategy not in self.matrix:
                self.matrix[strategy] = {}
                self._all_strategies.append(strategy)
            if regime not in self._all_regimes:
                self._all_regimes.append(regime)

            stats = self.trust_layer.get_blended_performance(strategy, regime)
            self.matrix[strategy][regime] = stats

        self.db.save_regime_matrix(self.matrix)
        self._last_build_time = datetime.now(timezone.utc)

        quality = self.trust_layer.get_data_quality_report()

        summary = {
            "status": "built",
            "data_mode": availability["mode"],
            "total_trades": len(tiered_trades),
            "effective_sample": quality.get("effective_sample_size", 0),
            "execution_grade": quality.get("execution_grade", "N/A"),
            "strategies": len(self._all_strategies),
            "regimes": len(self._all_regimes),
            "avt_trades": availability.get("avt_trades", 0),
            "data_quality_message": quality.get("message", ""),
        }
        logger.info("Matrix built from trust layer: %s", summary)
        return summary

    def _build_from_raw_trades(self) -> Dict[str, Any]:
        """Original build — used when trust layer not connected."""
        logger.info("Building matrix from RAW trades (no trust layer)...")

        trades = self.db.get_all_completed_trades()
        if not trades:
            logger.warning("No completed trades found — matrix empty.")
            return {"status": "empty", "trades": 0}

        self.matrix = {}
        self._all_strategies = []
        self._all_regimes = []

        for trade in trades:
            strategy = self._normalize(trade.strategy_name)
            regime = self._normalize(trade.regime_at_entry)
            if not strategy or not regime:
                continue

            if strategy not in self.matrix:
                self.matrix[strategy] = {}
                self._all_strategies.append(strategy)
            if regime not in self.matrix[strategy]:
                self.matrix[strategy][regime] = RegimePerformanceStats(
                    strategy_name=strategy, regime=regime
                )
            if regime not in self._all_regimes:
                self._all_regimes.append(regime)

            stats = self.matrix[strategy][regime]
            stats.total_trades += 1
            stats.total_rr += trade.actual_rr

            if trade.outcome == "win":
                stats.wins += 1
                stats.max_win_rr = max(stats.max_win_rr, trade.actual_rr)
            else:
                stats.losses += 1
                stats.max_loss_rr = min(stats.max_loss_rr, trade.actual_rr)

            if trade.entry_time and trade.exit_time:
                hold_h = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                n = stats.total_trades
                stats.avg_holding_period = (
                    (stats.avg_holding_period * (n - 1) + hold_h) / n
                )

        self._compute_derived()
        self.db.save_regime_matrix(self.matrix)
        self._last_build_time = datetime.now(timezone.utc)

        summary = {
            "status": "built",
            "data_mode": "RAW",
            "total_trades": len(trades),
            "strategies": len(self._all_strategies),
            "regimes": len(self._all_regimes),
            "matrix_cells": sum(len(r) for r in self.matrix.values()),
        }
        logger.info("Matrix built: %s", summary)
        return summary

    def _compute_derived(self) -> None:
        for strategy in self.matrix:
            for regime in self.matrix[strategy]:
                s = self.matrix[strategy][regime]
                if s.total_trades > 0:
                    s.win_rate = round(s.wins / s.total_trades, 4)
                    s.avg_rr = round(s.total_rr / s.total_trades, 4)
                s.sample_confidence = self._confidence(s.total_trades)
                s.last_updated = datetime.now(timezone.utc)

    # ─────────────────────────────────────────────
    # QUERY
    # ─────────────────────────────────────────────

    def query(self, strategy_name: str, regime: str) -> RegimePerformanceStats:
        """Get performance for strategy x regime. Returns neutral if unknown."""
        self._ensure_built()
        s = self._normalize(strategy_name)
        r = self._normalize(regime)
        if s in self.matrix and r in self.matrix[s]:
            return self.matrix[s][r]
        return RegimePerformanceStats(
            strategy_name=strategy_name,
            regime=regime,
            win_rate=0.5,
            avg_rr=1.0,
            sample_confidence=ConfidenceLevel.INSUFFICIENT,
        )

    def query_normalized_score(self, strategy_name: str, regime: str) -> float:
        """0-1 score accounting for WR, expectancy, AND confidence."""
        return self.query(strategy_name, regime).normalized_score()

    # ─────────────────────────────────────────────
    # SHOULD AVOID
    # ─────────────────────────────────────────────

    def should_avoid(self, strategy_name: str, current_regime: str) -> AvoidanceAdvice:
        """
        Should user avoid this strategy in current regime?
        If yes, provides ranked alternatives.
        """
        self._ensure_built()
        stats = self.query(strategy_name, current_regime)

        if (
            stats.win_rate < self.AVOID_WR_THRESHOLD
            and stats.total_trades >= self.AVOID_MIN_SAMPLE
        ):
            alts = self.get_best_strategies_for_regime(current_regime)
            alts = [
                a for a in alts
                if a["strategy"] != self._normalize(strategy_name)
            ]
            return AvoidanceAdvice(
                avoid=True,
                strategy_name=strategy_name,
                current_regime=current_regime,
                reason=(
                    f"Win rate only {stats.win_rate:.0%} in '{current_regime}' "
                    f"({stats.total_trades} trades, avg RR: {stats.avg_rr:.2f}). "
                    f"Below threshold {self.AVOID_WR_THRESHOLD:.0%}."
                ),
                current_wr=stats.win_rate,
                sample_size=stats.total_trades,
                alternatives=alts[:3],
            )

        if (
            stats.win_rate < self.AVOID_WR_THRESHOLD
            and 0 < stats.total_trades < self.AVOID_MIN_SAMPLE
        ):
            return AvoidanceAdvice(
                avoid=False,
                strategy_name=strategy_name,
                current_regime=current_regime,
                reason=(
                    f"WR {stats.win_rate:.0%} but only {stats.total_trades} trades "
                    f"— insufficient data to avoid. Proceed with caution."
                ),
                current_wr=stats.win_rate,
                sample_size=stats.total_trades,
            )

        return AvoidanceAdvice(
            avoid=False,
            strategy_name=strategy_name,
            current_regime=current_regime,
            reason=f"Strategy at {stats.win_rate:.0%} — acceptable.",
            current_wr=stats.win_rate,
            sample_size=stats.total_trades,
        )

    # ─────────────────────────────────────────────
    # RANK — best strategies for a regime
    # ─────────────────────────────────────────────

    def get_best_strategies_for_regime(self, regime: str) -> List[Dict[str, Any]]:
        """Strategies ranked by expectancy score in given regime."""
        self._ensure_built()
        r = self._normalize(regime)
        results = []
        for strategy, regimes in self.matrix.items():
            if r in regimes:
                s = regimes[r]
                if s.total_trades >= self.MIN_SAMPLE_FOR_RANKING:
                    results.append({
                        "strategy": strategy,
                        "win_rate": s.win_rate,
                        "avg_rr": s.avg_rr,
                        "trades": s.total_trades,
                        "confidence": s.sample_confidence.value,
                        "expectancy_score": s.normalized_score(),
                        "holding_hours": round(s.avg_holding_period, 1),
                    })
        results.sort(key=lambda x: x["expectancy_score"], reverse=True)
        return results

    def get_strategy_profile(self, strategy_name: str) -> Dict[str, Any]:
        """Full cross-regime profile for a strategy."""
        self._ensure_built()
        s = self._normalize(strategy_name)
        if s not in self.matrix:
            return {"strategy": strategy_name, "error": "No data", "regime_breakdown": {}}

        breakdown = {}
        total_wins = total_trades = 0
        best_regime = worst_regime = None
        best_wr, worst_wr = 0.0, 1.0

        for regime, stats in self.matrix[s].items():
            breakdown[regime] = {
                "win_rate": stats.win_rate,
                "avg_rr": stats.avg_rr,
                "trades": stats.total_trades,
                "confidence": stats.sample_confidence.value,
                "score": stats.normalized_score(),
            }
            total_wins += stats.wins
            total_trades += stats.total_trades

            if stats.total_trades >= self.MIN_SAMPLE_FOR_RANKING:
                if stats.win_rate > best_wr:
                    best_wr = stats.win_rate
                    best_regime = regime
                if stats.win_rate < worst_wr:
                    worst_wr = stats.win_rate
                    worst_regime = regime

        return {
            "strategy": strategy_name,
            "best_regime": best_regime,
            "best_regime_wr": best_wr,
            "worst_regime": worst_regime,
            "worst_regime_wr": worst_wr,
            "regime_breakdown": breakdown,
            "overall_wr": total_wins / total_trades if total_trades else 0,
            "total_trades": total_trades,
        }

    def compare_strategies_in_regime(
        self, strategies: List[str], regime: str
    ) -> List[Dict[str, Any]]:
        """Head-to-head comparison in a specific regime."""
        self._ensure_built()
        r = self._normalize(regime)
        comparison = []
        for name in strategies:
            stats = self.query(name, r)
            comparison.append({
                "strategy": name,
                "win_rate": stats.win_rate,
                "avg_rr": stats.avg_rr,
                "trades": stats.total_trades,
                "confidence": stats.sample_confidence.value,
                "score": stats.normalized_score(),
                "viable": (
                    stats.win_rate >= self.AVOID_WR_THRESHOLD
                    or stats.total_trades < self.AVOID_MIN_SAMPLE
                ),
            })
        comparison.sort(key=lambda x: x["score"], reverse=True)
        if comparison:
            comparison[0]["is_best"] = True
            for c in comparison[1:]:
                c["is_best"] = False
        return comparison

    def get_full_matrix_display(self) -> Dict[str, Any]:
        """Display-friendly format for UI."""
        self._ensure_built()
        cells: Dict[str, Dict[str, Any]] = {}
        for strategy in self._all_strategies:
            cells[strategy] = {}
            for regime in self._all_regimes:
                if strategy in self.matrix and regime in self.matrix[strategy]:
                    s = self.matrix[strategy][regime]
                    cells[strategy][regime] = {
                        "wr": f"{s.win_rate:.0%}",
                        "rr": f"{s.avg_rr:.1f}",
                        "n": s.total_trades,
                        "conf": s.sample_confidence.value,
                        "color": self._cell_color(s),
                    }
                else:
                    cells[strategy][regime] = {
                        "wr": "—", "rr": "—", "n": 0, "conf": "NONE", "color": "gray"
                    }
        return {
            "strategies": self._all_strategies,
            "regimes": self._all_regimes,
            "cells": cells,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ─────────────────────────────────────────────
    # INCREMENTAL UPDATE
    # ─────────────────────────────────────────────

    def update_with_new_trade(self, trade: TradeRecord) -> None:
        """Incrementally update matrix when a new trade completes."""
        s = self._normalize(trade.strategy_name)
        r = self._normalize(trade.regime_at_entry)
        if not s or not r:
            return

        if s not in self.matrix:
            self.matrix[s] = {}
            self._all_strategies.append(s)
        if r not in self.matrix[s]:
            self.matrix[s][r] = RegimePerformanceStats(strategy_name=s, regime=r)
        if r not in self._all_regimes:
            self._all_regimes.append(r)

        stats = self.matrix[s][r]
        stats.total_trades += 1
        stats.total_rr += trade.actual_rr
        if trade.outcome == "win":
            stats.wins += 1
            stats.max_win_rr = max(stats.max_win_rr, trade.actual_rr)
        else:
            stats.losses += 1
            stats.max_loss_rr = min(stats.max_loss_rr, trade.actual_rr)

        stats.win_rate = round(stats.wins / stats.total_trades, 4)
        stats.avg_rr = round(stats.total_rr / stats.total_trades, 4)
        stats.sample_confidence = self._confidence(stats.total_trades)
        stats.last_updated = datetime.now(timezone.utc)

        self.db.save_regime_matrix(self.matrix)
        logger.info(
            "Matrix updated: %s x %s → WR=%s, n=%d",
            s, r, f"{stats.win_rate:.0%}", stats.total_trades,
        )

    # ─────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────

    def _ensure_built(self) -> None:
        if not self.matrix:
            self.matrix = self.db.load_regime_matrix()
            if self.matrix:
                self._all_strategies = list(self.matrix.keys())
                self._all_regimes = list(
                    {r for regs in self.matrix.values() for r in regs}
                )
                return
            self.build()

    def _is_cache_fresh(self) -> bool:
        if self._last_build_time is None:
            return False
        return (datetime.now(timezone.utc) - self._last_build_time) < timedelta(
            days=self.MATRIX_STALENESS_DAYS
        )

    @staticmethod
    def _normalize(name: str) -> str:
        if not name:
            return ""
        return name.strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _confidence(n: int) -> ConfidenceLevel:
        if n >= 30:
            return ConfidenceLevel.HIGH
        elif n >= 15:
            return ConfidenceLevel.MODERATE
        elif n >= 5:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.INSUFFICIENT

    @staticmethod
    def _cell_color(stats: RegimePerformanceStats) -> str:
        if stats.total_trades < 5:
            return "gray"
        if stats.win_rate >= 0.55:
            return "green"
        if stats.win_rate >= 0.42:
            return "yellow"
        return "red"

    def get_current_regime(self) -> str:
        if self.regime_builder:
            try:
                regime = self.regime_builder.get_current_regime()
                return self._normalize(str(regime))
            except Exception as e:
                logger.warning("Failed to get regime: %s", e)
        return "unknown"
