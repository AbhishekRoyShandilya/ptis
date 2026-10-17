"""
Context Assembler
=================

Gathers data from all PTIS subsystems and assembles a complete
TradeContext for decision quality evaluation.

Integration Points
------------------
• core.models          → trade record
• strategy_engine      → attribution, match score, conditions
• behavioral           → override events, emotional state
• shadow               → shadow trade data
• intelligence         → market environment, ATR, regime
• analytics            → recent trade history
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from analytics.decision_quality import (
    EmotionalStateTag,
    RecentTradeSnapshot,
    TradeContext,
    TradeOutcome,
)

logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Assembles TradeContext from disparate PTIS data sources.

    Each data source has a dedicated fetch method that fails
    gracefully and independently — partial context still produces
    a valid evaluation (with lower confidence).

    Parameters
    ----------
    db_session_factory
        Async session factory from core.database
    strategy_matcher
        Instance of strategy_engine.strategy_matcher.StrategyMatcher
    regime_detector
        Instance of strategy_engine.regime_detector.RegimeDetector
    shadow_engine
        Instance of shadow.engine.ShadowEngine
    market_env
        Instance of intelligence.market_environment.MarketEnvironment
    """

    def __init__(
        self,
        db_session_factory: Any,
        strategy_matcher: Any = None,
        regime_detector: Any = None,
        shadow_engine: Any = None,
        market_env: Any = None,
    ) -> None:
        self._session_factory = db_session_factory
        self._strategy_matcher = strategy_matcher
        self._regime_detector = regime_detector
        self._shadow_engine = shadow_engine
        self._market_env = market_env

    async def assemble(self, trade_id: UUID) -> TradeContext:
        """
        Build a complete TradeContext for the given trade.

        Each section is fetched independently; failures produce
        log warnings but do not abort assembly.
        """
        ctx = TradeContext(trade_id=trade_id)

        # 1. Core trade data
        await self._fetch_trade_data(ctx, trade_id)

        # 2. Strategy attribution
        await self._fetch_strategy_data(ctx, trade_id)

        # 3. Behavioral / emotional data
        await self._fetch_behavioral_data(ctx, trade_id)

        # 4. Shadow comparison
        await self._fetch_shadow_data(ctx, trade_id)

        # 5. Market environment
        await self._fetch_market_context(ctx)

        # 6. Recent trade history
        await self._fetch_recent_history(ctx)

        logger.info(
            "Context assembled for trade %s | symbol=%s strategy=%s "
            "emotional=%s shadow=%s market=%s",
            trade_id,
            ctx.symbol,
            ctx.strategy_name,
            ctx.emotional_state,
            ctx.has_shadow,
            ctx.volatility_regime,
        )

        return ctx

    # ── section fetchers ────────────────────────────────────────

    async def _fetch_trade_data(self, ctx: TradeContext, trade_id: UUID) -> None:
        """Load core trade record."""
        try:
            from sqlalchemy import text

            async with self._session_factory() as session:
                row = await session.execute(
                    text("""
                        SELECT
                            symbol, side, entry_price, exit_price,
                            entry_time, exit_time, quantity, pnl,
                            r_multiple, planned_stop, actual_stop,
                            planned_target, planned_position_size,
                            actual_position_size, planned_risk_reward,
                            portfolio_heat, emotional_state,
                            confidence_level, notes, trade_count_today,
                            session_type, max_daily_loss_limit,
                            current_daily_pnl
                        FROM trades
                        WHERE id = :trade_id
                    """),
                    {"trade_id": str(trade_id)},
                )
                r = row.fetchone()
                if r is None:
                    logger.warning("Trade %s not found in database.", trade_id)
                    return

                ctx.symbol = r[0]
                ctx.side = r[1]
                ctx.entry_price = r[2]
                ctx.exit_price = r[3]
                ctx.entry_time = r[4]
                ctx.exit_time = r[5]
                ctx.quantity = r[6]
                ctx.pnl = r[7]
                ctx.r_multiple = r[8]
                ctx.planned_stop_price = r[9]
                ctx.actual_stop_price = r[10]
                ctx.planned_target_price = r[11]
                ctx.planned_position_size = r[12]
                ctx.actual_position_size = r[13]
                ctx.planned_risk_reward = r[14]
                ctx.portfolio_heat = r[15]

                # Parse emotional state
                emo_str = r[16]
                if emo_str:
                    try:
                        ctx.emotional_state = EmotionalStateTag(emo_str.lower())
                    except ValueError:
                        ctx.emotional_state = EmotionalStateTag.NEUTRAL

                ctx.confidence_level = r[17]
                ctx.notes = r[18]
                ctx.trade_count_today = r[19] or 0
                ctx.session_type = r[20]
                ctx.max_daily_loss_limit = r[21]
                ctx.current_daily_pnl = r[22]

        except Exception:
            logger.exception("Failed to fetch trade data for %s.", trade_id)

    async def _fetch_strategy_data(self, ctx: TradeContext, trade_id: UUID) -> None:
        """Load strategy attribution results."""
        try:
            from sqlalchemy import text

            async with self._session_factory() as session:
                row = await session.execute(
                    text("""
                        SELECT
                            strategy_name, match_score,
                            conditions_met, conditions_total,
                            regime_compatible
                        FROM strategy_attributions
                        WHERE trade_id = :trade_id
                        ORDER BY created_at DESC
                        LIMIT 1
                    """),
                    {"trade_id": str(trade_id)},
                )
                r = row.fetchone()
                if r:
                    ctx.strategy_name = r[0]
                    ctx.strategy_match_score = r[1]
                    ctx.conditions_met = r[2]
                    ctx.conditions_total = r[3]
                    ctx.regime_compatible = r[4]

        except Exception:
            logger.exception(
                "Failed to fetch strategy data for trade %s.", trade_id
            )

    async def _fetch_behavioral_data(
        self, ctx: TradeContext, trade_id: UUID
    ) -> None:
        """Load override events and behavioral flags."""
        try:
            from sqlalchemy import text

            async with self._session_factory() as session:
                # Override events
                rows = await session.execute(
                    text("""
                        SELECT event_type, severity, details, created_at
                        FROM override_events
                        WHERE trade_id = :trade_id
                        ORDER BY created_at
                    """),
                    {"trade_id": str(trade_id)},
                )
                overrides = rows.fetchall()
                ctx.override_events = [
                    {
                        "event_type": o[0],
                        "severity": o[1],
                        "details": o[2],
                        "created_at": o[3].isoformat() if o[3] else None,
                    }
                    for o in overrides
                ]

                if overrides:
                    ctx.override_severity = max(o[1] for o in overrides)

                # Time since last trade
                if ctx.entry_time:
                    prev = await session.execute(
                        text("""
                            SELECT exit_time FROM trades
                            WHERE exit_time < :entry_time
                            ORDER BY exit_time DESC
                            LIMIT 1
                        """),
                        {"entry_time": ctx.entry_time},
                    )
                    prev_row = prev.fetchone()
                    if prev_row and prev_row[0]:
                        delta = (ctx.entry_time - prev_row[0]).total_seconds()
                        ctx.time_since_last_trade_seconds = max(0.0, delta)

        except Exception:
            logger.exception(
                "Failed to fetch behavioral data for trade %s.", trade_id
            )

    async def _fetch_shadow_data(self, ctx: TradeContext, trade_id: UUID) -> None:
        """Load shadow trade comparison data."""
        try:
            if self._shadow_engine is None:
                return

            from sqlalchemy import text

            async with self._session_factory() as session:
                row = await session.execute(
                    text("""
                        SELECT
                            shadow_entry_price, shadow_exit_price,
                            shadow_pnl, optimal_entry_price,
                            optimal_exit_price
                        FROM shadow_trades
                        WHERE real_trade_id = :trade_id
                        ORDER BY created_at DESC
                        LIMIT 1
                    """),
                    {"trade_id": str(trade_id)},
                )
                r = row.fetchone()
                if r:
                    ctx.shadow_entry_price = r[0]
                    ctx.shadow_exit_price = r[1]
                    ctx.shadow_pnl = r[2]
                    ctx.optimal_entry_price = r[3]
                    ctx.optimal_exit_price = r[4]

        except Exception:
            logger.exception(
                "Failed to fetch shadow data for trade %s.", trade_id
            )

    async def _fetch_market_context(self, ctx: TradeContext) -> None:
        """Enrich with market environment data."""
        try:
            if self._market_env is None or ctx.symbol is None:
                return

            # Try to get environment snapshot at trade time
            env = await self._market_env.get_environment(
                symbol=ctx.symbol,
                timestamp=ctx.entry_time,
            )

            if env:
                ctx.atr = env.get("atr")
                ctx.volatility_regime = env.get("volatility_regime")
                ctx.trend_direction = env.get("trend_direction")
                ctx.trend_strength = env.get("trend_strength")
                ctx.vix_level = env.get("vix_level")
                ctx.market_open_time = env.get("market_open_time")

        except Exception:
            logger.exception(
                "Failed to fetch market context for %s.", ctx.symbol
            )

    async def _fetch_recent_history(self, ctx: TradeContext) -> None:
        """Load recent trades for streak and pattern detection."""
        try:
            from sqlalchemy import text

            reference_time = ctx.entry_time or datetime.utcnow()

            async with self._session_factory() as session:
                rows = await session.execute(
                    text("""
                        SELECT
                            id, pnl, r_multiple, exit_time,
                            emotional_state, actual_position_size
                        FROM trades
                        WHERE exit_time < :ref_time
                          AND id != :trade_id
                        ORDER BY exit_time DESC
                        LIMIT 10
                    """),
                    {
                        "ref_time": reference_time,
                        "trade_id": str(ctx.trade_id) if ctx.trade_id else "",
                    },
                )

                recent = rows.fetchall()
                ctx.recent_trades = []
                for r in recent:
                    pnl_val = r[1] or 0.0
                    if pnl_val > 0:
                        outcome = TradeOutcome.WIN
                    elif pnl_val < 0:
                        outcome = TradeOutcome.LOSS
                    else:
                        outcome = TradeOutcome.BREAKEVEN

                    emo = None
                    if r[4]:
                        try:
                            emo = EmotionalStateTag(r[4].lower())
                        except ValueError:
                            emo = EmotionalStateTag.NEUTRAL

                    ctx.recent_trades.append(RecentTradeSnapshot(
                        trade_id=UUID(r[0]) if r[0] else None,
                        pnl=pnl_val,
                        r_multiple=r[2],
                        outcome=outcome,
                        exit_time=r[3],
                        emotional_state=emo,
                        position_size=r[5],
                    ))

        except Exception:
            logger.exception("Failed to fetch recent trade history.")