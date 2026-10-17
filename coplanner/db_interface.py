# coplanner/db_interface.py
"""
Co-Planner Database Interface.

Adapts PTIS's existing SQLAlchemy Trade model into the TradeRecord
dataclass that engine modules consume.  All co-planner-specific
aggregate tables use SQLAlchemy text() with PostgreSQL syntax.

READ-ONLY from user trade data.
WRITE to coplanner_* aggregate tables only.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from coplanner.models import (
    ConfidenceLevel,
    DataTrustTier,
    RegimePerformanceStats,
    TradeRecord,
)

logger = logging.getLogger(__name__)


class CoPlannerDB:
    """
    Interface between co-planner engines and PTIS database.

    Reads from:
      - trades (existing PTIS table)
      - override_events (existing)
      - behavioral_events (existing)
    Writes to:
      - coplanner_regime_matrix (new)
      - coplanner_cluster_stats (new)
      - coplanner_decisions (new)
    """

    def __init__(self, db: Session):
        self._db = db

    # ─────────────────────────────────────────────
    # Trade History (READ from existing trades table)
    # ─────────────────────────────────────────────

    def get_all_completed_trades(
        self,
        trader_id: Optional[str] = None,
    ) -> List[TradeRecord]:
        """Pull completed trades from PTIS trades table."""
        try:
            result = self._db.execute(
                text("""
                    SELECT
                        t.id, t.symbol, t.direction,
                        COALESCE(t.declared_strategy_id, sa.resolved_strategy_id, 'unknown'),
                        t.market_regime, t.entry_price, t.exit_price,
                        t.stop_loss, t.take_profit, t.entry_time, t.exit_time,
                        t.pnl, t.r_multiple, t.position_size
                    FROM trades t
                    LEFT JOIN strategy_attributions sa ON sa.trade_id = t.id
                    WHERE t.status = 'closed'
                      AND t.exit_price IS NOT NULL
                      AND (t.trader_id = :tid OR :tid IS NULL)
                    ORDER BY t.exit_time DESC
                """),
                {"tid": trader_id},
            )
            trades: List[TradeRecord] = []
            for r in result.fetchall():
                try:
                    pnl = float(r[11] or 0)
                    trades.append(TradeRecord(
                        trade_id=str(r[0]),
                        symbol=r[1] or "",
                        strategy_name=r[3] or "unknown",
                        timeframe="",
                        regime_at_entry=str(r[4] or "unknown"),
                        entry_price=float(r[5] or 0),
                        exit_price=float(r[6] or 0),
                        stop_loss=float(r[7] or 0),
                        target_price=float(r[8] or 0),
                        entry_time=r[9] or datetime.now(timezone.utc),
                        exit_time=r[10] or datetime.now(timezone.utc),
                        outcome="win" if pnl > 0 else "loss",
                        actual_rr=float(r[12] or 0),
                        volume_at_entry=0.0,
                        trust_tier=DataTrustTier.USER_ALL,
                    ))
                except Exception as e:
                    logger.debug("Skipping trade row: %s", e)
            return trades
        except Exception:
            logger.debug("Failed to fetch completed trades", exc_info=True)
            return []

    def get_quality_trades(
        self,
        trader_id: Optional[str] = None,
    ) -> List[TradeRecord]:
        """
        Pull only 'clean execution' trades — no overrides with HIGH/CRITICAL severity.
        These get DataTrustTier.USER_QUALITY.
        """
        try:
            result = self._db.execute(
                text("""
                    SELECT
                        t.id, t.symbol, t.direction,
                        COALESCE(t.declared_strategy_id, sa.resolved_strategy_id, 'unknown'),
                        t.market_regime, t.entry_price, t.exit_price,
                        t.stop_loss, t.take_profit, t.entry_time, t.exit_time,
                        t.pnl, t.r_multiple
                    FROM trades t
                    LEFT JOIN strategy_attributions sa ON sa.trade_id = t.id
                    WHERE t.status = 'closed'
                      AND t.exit_price IS NOT NULL
                      AND (t.trader_id = :tid OR :tid IS NULL)
                      AND t.id NOT IN (
                          SELECT oe.trade_id FROM override_events oe
                          WHERE oe.severity IN ('high', 'critical')
                      )
                    ORDER BY t.exit_time DESC
                """),
                {"tid": trader_id},
            )
            trades: List[TradeRecord] = []
            for r in result.fetchall():
                try:
                    pnl = float(r[11] or 0)
                    trades.append(TradeRecord(
                        trade_id=str(r[0]),
                        symbol=r[1] or "",
                        strategy_name=r[3] or "unknown",
                        timeframe="",
                        regime_at_entry=str(r[4] or "unknown"),
                        entry_price=float(r[5] or 0),
                        exit_price=float(r[6] or 0),
                        stop_loss=float(r[7] or 0),
                        target_price=float(r[8] or 0),
                        entry_time=r[9] or datetime.now(timezone.utc),
                        exit_time=r[10] or datetime.now(timezone.utc),
                        outcome="win" if pnl > 0 else "loss",
                        actual_rr=float(r[12] or 0),
                        trust_tier=DataTrustTier.USER_QUALITY,
                    ))
                except Exception:
                    continue
            return trades
        except Exception:
            logger.debug("Failed to fetch quality trades", exc_info=True)
            return []

    def get_recent_trades(
        self,
        n: int = 10,
        trader_id: Optional[str] = None,
    ) -> List[TradeRecord]:
        """Most recent N completed trades."""
        all_trades = self.get_all_completed_trades(trader_id)
        return all_trades[:n]

    def get_losing_trades(
        self,
        trader_id: Optional[str] = None,
    ) -> List[TradeRecord]:
        """All losing trades for behavioral analysis."""
        all_trades = self.get_all_completed_trades(trader_id)
        return [t for t in all_trades if t.outcome == "loss"]

    # ─────────────────────────────────────────────
    # Regime Matrix Storage (coplanner_regime_matrix)
    # ─────────────────────────────────────────────

    def save_regime_matrix(
        self,
        matrix: Dict[str, Dict[str, RegimePerformanceStats]],
    ) -> None:
        """Upsert regime matrix rows into PostgreSQL."""
        try:
            for strategy, regimes in matrix.items():
                for regime, stats in regimes.items():
                    self._db.execute(
                        text("""
                            INSERT INTO coplanner_regime_matrix
                                (strategy_name, regime, wins, losses, total_trades,
                                 win_rate, avg_rr, total_rr, max_win_rr, max_loss_rr,
                                 avg_holding_hours, confidence, last_updated)
                            VALUES
                                (:strategy, :regime, :wins, :losses, :total,
                                 :wr, :avg_rr, :total_rr, :max_win, :max_loss,
                                 :hold_h, :conf, :ts)
                            ON CONFLICT (strategy_name, regime)
                            DO UPDATE SET
                                wins = EXCLUDED.wins,
                                losses = EXCLUDED.losses,
                                total_trades = EXCLUDED.total_trades,
                                win_rate = EXCLUDED.win_rate,
                                avg_rr = EXCLUDED.avg_rr,
                                total_rr = EXCLUDED.total_rr,
                                max_win_rr = EXCLUDED.max_win_rr,
                                max_loss_rr = EXCLUDED.max_loss_rr,
                                avg_holding_hours = EXCLUDED.avg_holding_hours,
                                confidence = EXCLUDED.confidence,
                                last_updated = EXCLUDED.last_updated
                        """),
                        {
                            "strategy": strategy,
                            "regime": regime,
                            "wins": stats.wins,
                            "losses": stats.losses,
                            "total": stats.total_trades,
                            "wr": stats.win_rate,
                            "avg_rr": stats.avg_rr,
                            "total_rr": stats.total_rr,
                            "max_win": stats.max_win_rr,
                            "max_loss": stats.max_loss_rr,
                            "hold_h": stats.avg_holding_period,
                            "conf": stats.sample_confidence.value,
                            "ts": datetime.now(timezone.utc).isoformat(),
                        },
                    )
            self._db.commit()
        except Exception:
            logger.error("Failed to save regime matrix", exc_info=True)
            self._db.rollback()

    def load_regime_matrix(self) -> Dict[str, Dict[str, RegimePerformanceStats]]:
        """Load stored regime matrix."""
        matrix: Dict[str, Dict[str, RegimePerformanceStats]] = {}
        try:
            result = self._db.execute(
                text("""
                    SELECT strategy_name, regime, wins, losses, total_trades,
                           win_rate, avg_rr, total_rr, max_win_rr, max_loss_rr,
                           avg_holding_hours, confidence, last_updated
                    FROM coplanner_regime_matrix
                """)
            )
            for r in result.fetchall():
                strategy = r[0]
                regime = r[1]
                if strategy not in matrix:
                    matrix[strategy] = {}
                try:
                    conf = ConfidenceLevel(r[11])
                except ValueError:
                    conf = ConfidenceLevel.INSUFFICIENT
                matrix[strategy][regime] = RegimePerformanceStats(
                    strategy_name=strategy,
                    regime=regime,
                    wins=r[2],
                    losses=r[3],
                    total_trades=r[4],
                    win_rate=float(r[5]),
                    avg_rr=float(r[6]),
                    total_rr=float(r[7]),
                    max_win_rr=float(r[8]),
                    max_loss_rr=float(r[9]),
                    avg_holding_period=float(r[10]),
                    sample_confidence=conf,
                )
        except Exception:
            logger.debug("Failed to load regime matrix (table may not exist yet)", exc_info=True)
        return matrix

    # ─────────────────────────────────────────────
    # Cluster Stats
    # ─────────────────────────────────────────────

    def get_cluster_stats(self, cluster_id: str) -> Dict[str, Any]:
        """Get performance stats for a pattern cluster."""
        try:
            result = self._db.execute(
                text("""
                    SELECT total_trades, wins, losses, win_rate, avg_rr
                    FROM coplanner_cluster_stats
                    WHERE cluster_id = :cid
                """),
                {"cid": cluster_id},
            )
            row = result.fetchone()
            if row:
                return {
                    "total_trades": row[0],
                    "wins": row[1],
                    "losses": row[2],
                    "win_rate": float(row[3]),
                    "avg_rr": float(row[4]),
                }
        except Exception:
            logger.debug("Cluster stats lookup failed", exc_info=True)
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.5, "avg_rr": 1.0}

    # ─────────────────────────────────────────────
    # Decision Logging
    # ─────────────────────────────────────────────

    def log_decision(self, session_id: str, decision_data: Dict[str, Any]) -> None:
        """Log a co-planner decision for effectiveness measurement."""
        try:
            rec = decision_data.get("recommendation", {})
            self._db.execute(
                text("""
                    INSERT INTO coplanner_decisions
                        (session_id, setups_evaluated, recommendation,
                         top_pick_symbol, top_pick_conviction, engine_output)
                    VALUES (:sid, :n, :rec, :sym, :conv, :output)
                """),
                {
                    "sid": session_id,
                    "n": decision_data.get("total_scanned", 0),
                    "rec": rec.get("action", ""),
                    "sym": rec.get("primary", ""),
                    "conv": decision_data.get("top_conviction", 0),
                    "output": json.dumps(decision_data, default=str)[:4000],
                },
            )
            self._db.commit()
        except Exception:
            logger.error("Failed to log decision", exc_info=True)
            self._db.rollback()

    # ─────────────────────────────────────────────
    # Behavioral helpers
    # ─────────────────────────────────────────────

    def get_override_counts_last_7_days(
        self,
        trader_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Count overrides by type in last 7 days."""
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            result = self._db.execute(
                text("""
                    SELECT oe.override_type, COUNT(*)
                    FROM override_events oe
                    JOIN trades t ON oe.trade_id = t.id
                    WHERE t.entry_time >= :cutoff
                      AND (t.trader_id = :tid OR :tid IS NULL)
                    GROUP BY oe.override_type
                """),
                {"cutoff": cutoff, "tid": trader_id},
            )
            return {r[0]: r[1] for r in result.fetchall()}
        except Exception:
            logger.debug("Override counts failed", exc_info=True)
            return {}

    def get_todays_trade_count(
        self,
        trader_id: Optional[str] = None,
    ) -> int:
        """Number of trades entered today."""
        try:
            today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
            result = self._db.execute(
                text("""
                    SELECT COUNT(*) FROM trades
                    WHERE entry_time >= :today
                      AND (trader_id = :tid OR trader_id IS NULL)
                """),
                {"today": today, "tid": trader_id},
            )
            row = result.fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
