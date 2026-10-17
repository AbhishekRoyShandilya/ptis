# agents/coplanner_context.py
"""
Co-Planner Context Assembler
==============================

Gathers trader-level context for the co-planner (pre-trade planning).
Unlike ContextAssembler (which assembles context for a specific trade_id),
this builds a snapshot of the trader's current state:

  - Recent trade history (last 10 closed trades)
  - Today's session stats (trade count, P&L, win/loss streak)
  - Strategy performance per symbol (win rate, avg R for this setup)
  - Current drawdown and portfolio heat
  - Behavioral patterns (recent overrides, emotional trends)
  - Current market regime + volatility

This context flows into every co-planner response so the AI can say
"You've lost 3 in a row, your breakout win rate on RELIANCE is 35%
in ranging markets — I'd skip this one" instead of generic advice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class TraderSnapshot:
    """Everything the co-planner needs to know about the trader right now."""

    # Session stats (today)
    todays_trade_count: int = 0
    todays_pnl: float = 0.0
    todays_winners: int = 0
    todays_losers: int = 0
    max_daily_trades: int = 5
    max_daily_loss: float = -5000.0

    # Streak
    current_streak: int = 0          # positive = wins, negative = losses
    streak_type: str = "neutral"     # "winning", "losing", "neutral"

    # Recent trades (last 10)
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)

    # Strategy performance for the current symbol
    symbol_stats: Dict[str, Any] = field(default_factory=dict)

    # Drawdown
    current_drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0

    # Portfolio heat
    open_positions: int = 0
    portfolio_heat_pct: float = 0.0
    open_trades: List[Dict[str, Any]] = field(default_factory=list)

    # Behavioral
    recent_overrides: List[Dict[str, Any]] = field(default_factory=list)
    discipline_score_avg: float = 7.0
    emotional_pattern: str = "stable"

    # Market context
    regime: str = "unknown"
    volatility_regime: str = "average"
    atr: float = 0.0

    # Trade frequency (for overtrading analysis)
    trades_last_7d: int = 0
    trades_last_14d: int = 0
    avg_trades_per_day_7d: float = 0.0
    max_trades_in_a_day_7d: int = 0
    total_closed_trades: int = 0

    # Trader profile
    preferred_strategies: List[str] = field(default_factory=list)
    strongest_regime: str = ""
    weakest_regime: str = ""


class CoPlannerContextAssembler:
    """
    Assembles TraderSnapshot from DB for co-planner.

    Each fetch method is independent and fails gracefully —
    partial context still produces useful co-planning.
    """

    def __init__(self, db_session: Any):
        self._db = db_session

    async def assemble(
        self,
        trader_id: str,
        symbol: Optional[str] = None,
    ) -> TraderSnapshot:
        """Build complete trader snapshot for co-planner context."""
        snap = TraderSnapshot()

        await self._fetch_todays_stats(snap, trader_id)
        await self._fetch_recent_trades(snap, trader_id)
        await self._fetch_streak(snap)
        await self._fetch_open_positions(snap, trader_id)
        await self._fetch_drawdown(snap, trader_id)

        if symbol:
            await self._fetch_symbol_stats(snap, trader_id, symbol)

        await self._fetch_behavioral_data(snap, trader_id)
        await self._fetch_regime(snap, symbol)

        return snap

    def assemble_sync(
        self,
        trader_id: str,
        symbol: Optional[str] = None,
    ) -> TraderSnapshot:
        """Synchronous version for non-async routes."""
        snap = TraderSnapshot()

        self._fetch_todays_stats_sync(snap, trader_id)
        self._fetch_recent_trades_sync(snap, trader_id)
        self._compute_streak(snap)  # computed from recent_trades, no DB
        self._fetch_open_positions_sync(snap, trader_id)
        self._fetch_behavioral_data_sync(snap, trader_id)
        self._fetch_trade_frequency_sync(snap, trader_id)

        if symbol:
            self._fetch_symbol_stats_sync(snap, trader_id, symbol)

        return snap

    # ── Today's session ──────────────────────────────────────────

    async def _fetch_todays_stats(self, snap: TraderSnapshot, trader_id: str) -> None:
        try:
            from sqlalchemy import text
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)

            result = self._db.execute(
                text("""
                    SELECT
                        COUNT(*) as total,
                        COALESCE(SUM(pnl), 0) as total_pnl,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losers
                    FROM trades
                    WHERE entry_time >= :today
                      AND (trader_id = :tid OR trader_id IS NULL)
                """),
                {"today": today_start, "tid": trader_id},
            )
            row = result.fetchone()
            if row:
                snap.todays_trade_count = row[0] or 0
                snap.todays_pnl = float(row[1] or 0)
                snap.todays_winners = row[2] or 0
                snap.todays_losers = row[3] or 0
        except Exception:
            logger.debug("Failed to fetch today's stats", exc_info=True)

    def _fetch_todays_stats_sync(self, snap: TraderSnapshot, trader_id: str) -> None:
        try:
            from sqlalchemy import text
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)

            result = self._db.execute(
                text("""
                    SELECT
                        COUNT(*) as total,
                        COALESCE(SUM(pnl), 0) as total_pnl,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losers
                    FROM trades
                    WHERE entry_time >= :today
                      AND (trader_id = :tid OR trader_id IS NULL)
                """),
                {"today": today_start, "tid": trader_id},
            )
            row = result.fetchone()
            if row:
                snap.todays_trade_count = row[0] or 0
                snap.todays_pnl = float(row[1] or 0)
                snap.todays_winners = row[2] or 0
                snap.todays_losers = row[3] or 0
        except Exception:
            logger.debug("Failed to fetch today's stats (sync)", exc_info=True)

    # ── Recent trades ────────────────────────────────────────────

    async def _fetch_recent_trades(self, snap: TraderSnapshot, trader_id: str) -> None:
        try:
            from sqlalchemy import text
            result = self._db.execute(
                text("""
                    SELECT id, symbol, direction, entry_price, exit_price,
                           pnl, r_multiple, market_regime, entry_time, exit_time
                    FROM trades
                    WHERE status = 'closed'
                      AND (trader_id = :tid OR trader_id IS NULL)
                    ORDER BY exit_time DESC
                    LIMIT 10
                """),
                {"tid": trader_id},
            )
            for r in result.fetchall():
                snap.recent_trades.append({
                    "id": str(r[0]),
                    "symbol": r[1],
                    "direction": r[2],
                    "entry_price": r[3],
                    "exit_price": r[4],
                    "pnl": float(r[5] or 0),
                    "r_multiple": float(r[6] or 0),
                    "regime": r[7],
                    "entry_time": str(r[8]) if r[8] else None,
                    "exit_time": str(r[9]) if r[9] else None,
                })
        except Exception:
            logger.debug("Failed to fetch recent trades", exc_info=True)

    def _fetch_recent_trades_sync(self, snap: TraderSnapshot, trader_id: str) -> None:
        try:
            from sqlalchemy import text
            result = self._db.execute(
                text("""
                    SELECT id, symbol, direction, entry_price, exit_price,
                           pnl, r_multiple, market_regime, entry_time, exit_time
                    FROM trades
                    WHERE status = 'closed'
                      AND (trader_id = :tid OR trader_id IS NULL)
                    ORDER BY exit_time DESC
                    LIMIT 10
                """),
                {"tid": trader_id},
            )
            for r in result.fetchall():
                snap.recent_trades.append({
                    "id": str(r[0]),
                    "symbol": r[1],
                    "direction": r[2],
                    "entry_price": r[3],
                    "exit_price": r[4],
                    "pnl": float(r[5] or 0),
                    "r_multiple": float(r[6] or 0),
                    "regime": r[7],
                })
        except Exception:
            logger.debug("Failed to fetch recent trades (sync)", exc_info=True)

    # ── Streak (computed from recent_trades, no DB) ──────────────

    async def _fetch_streak(self, snap: TraderSnapshot) -> None:
        self._compute_streak(snap)

    def _compute_streak(self, snap: TraderSnapshot) -> None:
        if not snap.recent_trades:
            return
        streak = 0
        first_pnl = snap.recent_trades[0].get("pnl", 0)
        streak_dir = 1 if first_pnl > 0 else -1 if first_pnl < 0 else 0
        for t in snap.recent_trades:
            pnl = t.get("pnl", 0)
            if (pnl > 0 and streak_dir >= 0):
                streak += 1
                streak_dir = 1
            elif (pnl < 0 and streak_dir <= 0):
                streak -= 1
                streak_dir = -1
            else:
                break
        snap.current_streak = streak
        snap.streak_type = "winning" if streak > 0 else "losing" if streak < 0 else "neutral"

    # ── Open positions ───────────────────────────────────────────

    async def _fetch_open_positions(self, snap: TraderSnapshot, trader_id: str) -> None:
        self._fetch_open_positions_sync(snap, trader_id)

    def _fetch_open_positions_sync(self, snap: TraderSnapshot, trader_id: str) -> None:
        try:
            from sqlalchemy import text
            result = self._db.execute(
                text("""
                    SELECT id, symbol, direction, entry_price, stop_loss,
                           position_size, risk_pct
                    FROM trades
                    WHERE status = 'open'
                      AND (trader_id = :tid OR trader_id IS NULL)
                """),
                {"tid": trader_id},
            )
            for r in result.fetchall():
                snap.open_trades.append({
                    "id": str(r[0]),
                    "symbol": r[1],
                    "direction": r[2],
                    "entry_price": r[3],
                    "stop_loss": r[4],
                    "position_size": r[5],
                    "risk_pct": float(r[6] or 0),
                })
            snap.open_positions = len(snap.open_trades)
            snap.portfolio_heat_pct = sum(t.get("risk_pct", 0) for t in snap.open_trades)
        except Exception:
            logger.debug("Failed to fetch open positions", exc_info=True)

    # ── Drawdown ─────────────────────────────────────────────────

    async def _fetch_drawdown(self, snap: TraderSnapshot, trader_id: str) -> None:
        try:
            from sqlalchemy import text
            result = self._db.execute(
                text("""
                    SELECT account_balance_at_entry
                    FROM trades
                    WHERE account_balance_at_entry IS NOT NULL
                      AND (trader_id = :tid OR trader_id IS NULL)
                    ORDER BY entry_time DESC
                    LIMIT 1
                """),
                {"tid": trader_id},
            )
            row = result.fetchone()
            if row and row[0]:
                snap.current_equity = float(row[0])

            # Get peak from all balances
            peak_result = self._db.execute(
                text("""
                    SELECT MAX(account_balance_at_entry)
                    FROM trades
                    WHERE account_balance_at_entry IS NOT NULL
                      AND (trader_id = :tid OR trader_id IS NULL)
                """),
                {"tid": trader_id},
            )
            peak_row = peak_result.fetchone()
            if peak_row and peak_row[0]:
                snap.peak_equity = float(peak_row[0])
                if snap.peak_equity > 0 and snap.current_equity > 0:
                    snap.current_drawdown_pct = (
                        (snap.peak_equity - snap.current_equity) / snap.peak_equity
                    )
        except Exception:
            logger.debug("Failed to fetch drawdown", exc_info=True)

    # ── Symbol-specific stats ────────────────────────────────────

    async def _fetch_symbol_stats(
        self, snap: TraderSnapshot, trader_id: str, symbol: str,
    ) -> None:
        self._fetch_symbol_stats_sync(snap, trader_id, symbol)

    def _fetch_symbol_stats_sync(
        self, snap: TraderSnapshot, trader_id: str, symbol: str,
    ) -> None:
        try:
            from sqlalchemy import text
            result = self._db.execute(
                text("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                        COALESCE(AVG(r_multiple), 0) as avg_r,
                        COALESCE(SUM(pnl), 0) as total_pnl,
                        COALESCE(AVG(pnl), 0) as avg_pnl,
                        MAX(market_regime) as last_regime
                    FROM trades
                    WHERE symbol = :symbol
                      AND status = 'closed'
                      AND (trader_id = :tid OR trader_id IS NULL)
                """),
                {"symbol": symbol, "tid": trader_id},
            )
            row = result.fetchone()
            if row and row[0] > 0:
                total = row[0]
                wins = row[1] or 0
                snap.symbol_stats = {
                    "symbol": symbol,
                    "total_trades": total,
                    "wins": wins,
                    "losses": total - wins,
                    "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                    "avg_r": round(float(row[2]), 2),
                    "total_pnl": round(float(row[3]), 2),
                    "avg_pnl": round(float(row[4]), 2),
                    "last_regime": row[5],
                }

            # Per-regime win rate for this symbol
            regime_result = self._db.execute(
                text("""
                    SELECT
                        market_regime,
                        COUNT(*) as total,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins
                    FROM trades
                    WHERE symbol = :symbol
                      AND status = 'closed'
                      AND market_regime IS NOT NULL
                      AND (trader_id = :tid OR trader_id IS NULL)
                    GROUP BY market_regime
                """),
                {"symbol": symbol, "tid": trader_id},
            )
            regime_breakdown = {}
            for r in regime_result.fetchall():
                regime_name = r[0]
                total = r[1]
                wins = r[2] or 0
                regime_breakdown[regime_name] = {
                    "total": total,
                    "wins": wins,
                    "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                }
            if regime_breakdown:
                snap.symbol_stats["regime_breakdown"] = regime_breakdown

        except Exception:
            logger.debug("Failed to fetch symbol stats for %s", symbol, exc_info=True)

    # ── Behavioral data ──────────────────────────────────────────

    async def _fetch_behavioral_data(self, snap: TraderSnapshot, trader_id: str) -> None:
        self._fetch_behavioral_data_sync(snap, trader_id)

    def _fetch_behavioral_data_sync(self, snap: TraderSnapshot, trader_id: str) -> None:
        try:
            from sqlalchemy import text

            # Recent overrides (last 7 days)
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            result = self._db.execute(
                text("""
                    SELECT oe.override_type, oe.severity, t.symbol, t.entry_time
                    FROM override_events oe
                    JOIN trades t ON oe.trade_id = t.id
                    WHERE t.entry_time >= :cutoff
                      AND (t.trader_id = :tid OR t.trader_id IS NULL)
                    ORDER BY t.entry_time DESC
                    LIMIT 10
                """),
                {"cutoff": cutoff, "tid": trader_id},
            )
            for r in result.fetchall():
                snap.recent_overrides.append({
                    "type": r[0],
                    "severity": r[1],
                    "symbol": r[2],
                    "time": str(r[3]) if r[3] else None,
                })

            # Identify emotional pattern from overrides
            high_severity = sum(1 for o in snap.recent_overrides if o.get("severity") in ("high", "critical"))
            if high_severity >= 3:
                snap.emotional_pattern = "stressed"
            elif high_severity >= 1:
                snap.emotional_pattern = "mixed"
            else:
                snap.emotional_pattern = "stable"

        except Exception:
            logger.debug("Failed to fetch behavioral data", exc_info=True)

    # ── Trade frequency (overtrading detection) ──────────────────

    def _fetch_trade_frequency_sync(self, snap: TraderSnapshot, trader_id: str) -> None:
        """Compute trade frequency stats for overtrading analysis."""
        try:
            from sqlalchemy import text
            cutoff_7d = datetime.now(timezone.utc) - timedelta(days=7)
            cutoff_14d = datetime.now(timezone.utc) - timedelta(days=14)

            # Count trades in last 7 and 14 days
            result = self._db.execute(
                text("""
                    SELECT
                        COUNT(*) FILTER (WHERE entry_time >= :c7d) as last_7d,
                        COUNT(*) FILTER (WHERE entry_time >= :c14d) as last_14d,
                        COUNT(*) as total
                    FROM trades
                    WHERE status = 'closed'
                      AND (trader_id = :tid OR trader_id IS NULL)
                """),
                {"c7d": cutoff_7d, "c14d": cutoff_14d, "tid": trader_id},
            )
            row = result.fetchone()
            if row:
                snap.trades_last_7d = row[0] or 0
                snap.trades_last_14d = row[1] or 0
                snap.total_closed_trades = row[2] or 0

            # Active trading days and max trades in a day (last 7d)
            result2 = self._db.execute(
                text("""
                    SELECT
                        DATE(entry_time AT TIME ZONE 'UTC') as trade_date,
                        COUNT(*) as day_count
                    FROM trades
                    WHERE status = 'closed'
                      AND entry_time >= :c7d
                      AND (trader_id = :tid OR trader_id IS NULL)
                    GROUP BY trade_date
                    ORDER BY trade_date
                """),
                {"c7d": cutoff_7d, "tid": trader_id},
            )
            rows2 = result2.fetchall()
            if rows2:
                day_counts = [r[1] for r in rows2]
                active_days = len(day_counts)
                snap.max_trades_in_a_day_7d = max(day_counts)
                snap.avg_trades_per_day_7d = round(sum(day_counts) / active_days, 1) if active_days else 0.0
        except Exception:
            logger.debug("Failed to fetch trade frequency", exc_info=True)

    # ── Market regime ────────────────────────────────────────────

    async def _fetch_regime(self, snap: TraderSnapshot, symbol: Optional[str]) -> None:
        try:
            from sqlalchemy import text
            result = self._db.execute(
                text("""
                    SELECT regime, confidence
                    FROM market_regime_logs
                    ORDER BY detected_at DESC
                    LIMIT 1
                """),
            )
            row = result.fetchone()
            if row:
                snap.regime = row[0] or "unknown"
        except Exception:
            logger.debug("Failed to fetch regime", exc_info=True)

    # ── Format for prompt injection ──────────────────────────────

    @staticmethod
    def format_for_prompt(snap: TraderSnapshot, symbol: Optional[str] = None) -> str:
        """Format TraderSnapshot as a text block for LLM system prompt injection."""
        lines = ["## Trader Context (live)"]

        # Session
        lines.append(f"\n### Today's Session")
        lines.append(f"- Trades: {snap.todays_trade_count} ({snap.todays_winners}W / {snap.todays_losers}L)")
        lines.append(f"- P&L: ₹{snap.todays_pnl:,.0f}")
        if snap.current_streak != 0:
            lines.append(f"- Streak: {abs(snap.current_streak)} consecutive {'wins' if snap.current_streak > 0 else 'losses'}")

        # Portfolio
        if snap.open_positions > 0:
            lines.append(f"\n### Open Positions ({snap.open_positions})")
            lines.append(f"- Portfolio heat: {snap.portfolio_heat_pct:.1f}%")
            for t in snap.open_trades[:5]:
                lines.append(f"  - {t['symbol']} {t['direction']} @ ₹{t.get('entry_price', 0):,.0f}")

        # Drawdown
        if snap.current_drawdown_pct > 0.02:
            lines.append(f"\n### Risk Alert")
            lines.append(f"- Current drawdown: {snap.current_drawdown_pct*100:.1f}% from peak")
            if snap.current_drawdown_pct > 0.10:
                lines.append("- ⚠ SIGNIFICANT DRAWDOWN — recommend reduced sizing")

        # Symbol stats
        if snap.symbol_stats and symbol:
            s = snap.symbol_stats
            lines.append(f"\n### Your History with {symbol}")
            lines.append(f"- Total trades: {s.get('total_trades', 0)}")
            lines.append(f"- Win rate: {s.get('win_rate', 0)}%")
            lines.append(f"- Average R: {s.get('avg_r', 0)}")
            lines.append(f"- Total P&L: ₹{s.get('total_pnl', 0):,.0f}")
            if "regime_breakdown" in s:
                lines.append("- Win rate by regime:")
                for regime, data in s["regime_breakdown"].items():
                    lines.append(f"  - {regime}: {data['win_rate']}% ({data['total']} trades)")

        # Behavioral
        if snap.recent_overrides:
            lines.append(f"\n### Recent Behavioral Patterns (7 days)")
            lines.append(f"- Override events: {len(snap.recent_overrides)}")
            lines.append(f"- Emotional pattern: {snap.emotional_pattern}")
            top_types = {}
            for o in snap.recent_overrides:
                t = o.get("type", "unknown")
                top_types[t] = top_types.get(t, 0) + 1
            if top_types:
                most_common = max(top_types, key=top_types.get)
                lines.append(f"- Most common override: {most_common} ({top_types[most_common]}x)")

        # Trade frequency / overtrading
        if snap.total_closed_trades > 0:
            lines.append(f"\n### Trading Frequency (for overtrading analysis)")
            lines.append(f"- Total closed trades (all time): {snap.total_closed_trades}")
            lines.append(f"- Trades last 7 days: {snap.trades_last_7d}")
            lines.append(f"- Trades last 14 days: {snap.trades_last_14d}")
            if snap.avg_trades_per_day_7d > 0:
                lines.append(f"- Avg trades/day (last 7d active days): {snap.avg_trades_per_day_7d}")
                lines.append(f"- Max trades in a single day (last 7d): {snap.max_trades_in_a_day_7d}")
            if snap.avg_trades_per_day_7d > 3:
                lines.append("- ⚠ HIGH FREQUENCY — potentially overtrading (>3 trades/active day)")
            elif snap.avg_trades_per_day_7d > 2:
                lines.append("- MODERATE frequency — borderline overtrading territory")
            else:
                lines.append("- Frequency looks controlled")

        # Recent trades
        if snap.recent_trades:
            lines.append(f"\n### Last {len(snap.recent_trades)} Trades")
            for t in snap.recent_trades[:5]:
                pnl = t.get("pnl", 0)
                r = t.get("r_multiple", 0)
                emoji = "✓" if pnl > 0 else "✗"
                lines.append(
                    f"  {emoji} {t['symbol']} {t.get('direction', '')} "
                    f"R={r:+.1f} ₹{pnl:+,.0f}"
                )

        # Regime
        if snap.regime and snap.regime != "unknown":
            lines.append(f"\n### Current Market")
            lines.append(f"- Regime: {snap.regime}")

        return "\n".join(lines)
