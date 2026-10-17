# automation/autonomous_shadow.py
"""
Autonomous Shadow Creator
=========================
Bridges the gap between SKE signal generation and shadow trade creation.

When the market scanner detects a signal with sufficient conviction, this
module automatically creates a virtual (shadow) trade — no real money at risk.
The result is tracked in the System Journal for performance review.

Flow:
    1. SignalGenerator.evaluate() produces a SignalOutput
    2. AutonomousShadowCreator.create_from_signal() converts it into a ShadowTrade
    3. The trade is journaled via SystemJournal
    4. ShadowEngine manages price updates, exits, and P&L tracking

This enables the system to "paper trade" autonomously, letting the user
compare their own decisions against what the system would have done.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session as DBSession

from core.enums import ShadowTradeType, TradeDirection
from core.events import Event, EventType, get_event_bus
from shadow.models import ShadowSession, ShadowTrade
from ske.models import SignalOutput

logger = logging.getLogger(__name__)

# Minimum conviction scores for autonomous trade creation
MIN_CONVICTION_STANDARD = 0.50  # standard position
MIN_CONVICTION_AGGRESSIVE = 0.70  # aggressive position
MIN_CONVICTION_MAX = 0.85  # max position


class AutonomousShadowCreator:
    """
    Creates shadow (virtual) trades from SKE signals automatically.

    Each signal that passes the conviction threshold gets a shadow trade
    with appropriate position sizing. The system tracks these independently
    of the trader's real trades.

    Usage:
        creator = AutonomousShadowCreator(default_capital=500000)
        result = creator.create_from_signal(db, signal, candles)
    """

    def __init__(
        self,
        default_capital: float = 500_000.0,
        default_risk_pct: float = 1.0,  # 1% risk per trade
        max_open_shadows: int = 20,
    ):
        self._default_capital = default_capital
        self._default_risk_pct = default_risk_pct
        self._max_open_shadows = max_open_shadows
        self._event_bus = get_event_bus()

    def create_from_signal(
        self,
        db: DBSession,
        signal: SignalOutput,
        candles: Optional[List[Dict[str, Any]]] = None,
        capital: Optional[float] = None,
    ) -> Optional[ShadowTrade]:
        """
        Create a shadow trade from a signal output.

        Args:
            db: Database session.
            signal: SKE signal output with conviction, entry zone, stop, targets.
            candles: Recent candles for context (optional).
            capital: Override capital for position sizing.

        Returns:
            Created ShadowTrade or None if signal doesn't qualify.
        """
        # 1. Conviction gate
        if signal.conviction_score < MIN_CONVICTION_STANDARD:
            logger.debug(
                "Signal for %s conviction %.2f below threshold — skipping",
                signal.symbol, signal.conviction_score,
            )
            return None

        # 2. Check max open shadows
        open_count = db.query(ShadowTrade).filter(
            ShadowTrade.status == "open",
            ShadowTrade.shadow_type == ShadowTradeType.AUTONOMOUS.value
            if hasattr(ShadowTradeType, "AUTONOMOUS")
            else ShadowTrade.shadow_type == "autonomous",
        ).count()

        if open_count >= self._max_open_shadows:
            logger.warning(
                "Max open autonomous shadows (%d) reached — skipping %s",
                self._max_open_shadows, signal.symbol,
            )
            return None

        # 3. Determine direction
        direction = "long"  # Default to long for equity signals
        if signal.entry_zone and signal.stop_level:
            if signal.stop_level > signal.entry_zone.sweet_spot:
                direction = "short"

        # 4. Compute entry price
        entry_price = (
            signal.entry_zone.sweet_spot
            if signal.entry_zone
            else (candles[-1]["close"] if candles else 0.0)
        )

        if entry_price <= 0:
            logger.warning("Invalid entry price for %s — skipping", signal.symbol)
            return None

        # 5. Compute position size based on risk
        stop_level = signal.stop_level
        if stop_level is None:
            # Fallback: 2% of entry as stop distance
            stop_dist = entry_price * 0.02
            stop_level = entry_price - stop_dist if direction == "long" else entry_price + stop_dist

        risk_per_share = abs(entry_price - stop_level)
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.01

        effective_capital = capital or self._default_capital
        risk_amount = effective_capital * (self._default_risk_pct / 100.0)

        # Apply sizing multiplier from signal
        risk_amount *= signal.size_multiplier
        position_size = max(1, int(risk_amount / risk_per_share))

        # 6. Determine target
        target_price = None
        if signal.target_levels:
            # Use the first target (1:1 R)
            target_price = signal.target_levels[0]

        # 7. Build trailing config if aggressive conviction
        trailing_config: Dict[str, Any] = {}
        if signal.conviction_score >= MIN_CONVICTION_AGGRESSIVE:
            trailing_config = {
                "enabled": True,
                "type": "atr",
                "params": {"atr_multiplier": 1.5},
                "activate_at_r": 1.0,
            }

        # 8. Create shadow session + trade
        session = ShadowSession(
            id=uuid.uuid4(),
            real_trade_id=None,  # No real trade — autonomous
            strategy_id=str(signal.sko_id) if signal.sko_id else None,
            status="active",
        )
        db.add(session)
        db.flush()

        shadow = ShadowTrade(
            id=uuid.uuid4(),
            session_id=session.id,
            real_trade_id=None,
            shadow_type="autonomous",
            label=f"Auto: {signal.symbol} ({signal.conviction_band.value})",
            symbol=signal.symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=signal.timestamp,
            position_size=position_size,
            stop_loss=stop_level,
            original_stop=stop_level,
            take_profit=target_price,
            trailing_config=trailing_config,
            risk_amount=risk_amount,
            remaining_size=position_size,
            high_since_entry=entry_price,
            low_since_entry=entry_price,
            strategy_id=str(signal.sko_id) if signal.sko_id else None,
            config={
                "source": "autonomous_shadow_creator",
                "sko_name": signal.sko_name,
                "conviction_score": signal.conviction_score,
                "conviction_band": signal.conviction_band.value,
                "sizing_recommendation": signal.sizing_recommendation,
                "matched_variant": signal.matched_variant,
                "variant_provenance": signal.variant_provenance,
                "variant_conviction_boost": signal.variant_conviction_boost,
                "regime_match_score": signal.regime_match_score,
            },
        )
        db.add(shadow)
        db.commit()
        db.refresh(shadow)

        logger.info(
            "Autonomous shadow created: %s %s @ %.2f | stop=%.2f | "
            "conviction=%.2f (%s) | variant=%s | pos_size=%d",
            direction.upper(),
            signal.symbol,
            entry_price,
            stop_level,
            signal.conviction_score,
            signal.conviction_band.value,
            signal.matched_variant or "none",
            position_size,
        )

        # Publish event
        self._event_bus.publish(Event(
            event_type=EventType.SHADOW_CREATED,
            source="autonomous_shadow_creator",
            data={
                "shadow_id": str(shadow.id),
                "session_id": str(session.id),
                "symbol": signal.symbol,
                "direction": direction,
                "conviction": signal.conviction_score,
                "conviction_band": signal.conviction_band.value,
                "autonomous": True,
            },
        ))

        return shadow

    def create_batch_from_signals(
        self,
        db: DBSession,
        signals: List[SignalOutput],
        candles_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        capital: Optional[float] = None,
    ) -> List[ShadowTrade]:
        """
        Create shadow trades from multiple signals.
        Deduplicates by symbol (only the highest conviction wins).
        """
        # Deduplicate: keep highest conviction per symbol
        best_signals: Dict[str, SignalOutput] = {}
        for sig in signals:
            existing = best_signals.get(sig.symbol)
            if existing is None or sig.conviction_score > existing.conviction_score:
                best_signals[sig.symbol] = sig

        created: List[ShadowTrade] = []
        for symbol, signal in best_signals.items():
            candles = (candles_map or {}).get(symbol)
            shadow = self.create_from_signal(db, signal, candles, capital)
            if shadow:
                created.append(shadow)

        logger.info(
            "Batch autonomous creation: %d signals → %d shadows",
            len(signals), len(created),
        )
        return created
