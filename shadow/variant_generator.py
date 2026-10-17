# shadow/variant_generator.py
"""
Variant Generator — creates alternative shadow trades with modified
risk parameters for what-if analysis.

Works with ShadowEngine.VARIANT_CONFIGS but adds dynamic variant
generation based on market conditions and historical performance.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session as DBSession

from core.enums import ShadowTradeType, TradeDirection
from core.models import Trade
from shadow.models import ShadowSession, ShadowTrade

logger = logging.getLogger(__name__)


@dataclass
class VariantConfig:
    """Configuration for a shadow variant."""
    variant_type: str
    label: str
    stop_multiplier: float = 1.0
    target_multiplier: float = 1.0
    trailing_enabled: bool = False
    trailing_atr_mult: float = 1.5
    size_multiplier: float = 1.0


# ── Standard variants (supplement engine.VARIANT_CONFIGS) ──

EXTENDED_VARIANTS = [
    VariantConfig(
        variant_type="breakeven_trail",
        label="Breakeven + Trail",
        stop_multiplier=1.0,
        trailing_enabled=True,
        trailing_atr_mult=2.0,
    ),
    VariantConfig(
        variant_type="half_size",
        label="Half Position Size",
        stop_multiplier=1.0,
        size_multiplier=0.5,
    ),
    VariantConfig(
        variant_type="double_target",
        label="Extended Target (2x)",
        stop_multiplier=1.0,
        target_multiplier=2.0,
    ),
]


class VariantGenerator:
    """
    Generates shadow trade variants for what-if comparison.

    Standard usage:
        gen = VariantGenerator()
        variants = gen.generate(db, session, trade)
    """

    def __init__(self, configs: list[VariantConfig] | None = None):
        self._configs = configs or EXTENDED_VARIANTS

    def generate(
        self,
        db: DBSession,
        session: ShadowSession,
        trade: Trade,
        market_snapshot: dict[str, Any] | None = None,
    ) -> list[ShadowTrade]:
        """Generate all configured variants for a trade."""
        created: list[ShadowTrade] = []

        for cfg in self._configs:
            shadow = self._build_variant(db, session, trade, cfg, market_snapshot)
            if shadow:
                created.append(shadow)

        logger.info(
            "Generated %d extended variants for trade %s",
            len(created), trade.id,
        )
        return created

    def generate_regime_aware(
        self,
        db: DBSession,
        session: ShadowSession,
        trade: Trade,
        regime: str,
        volatility_percentile: float = 50.0,
    ) -> list[ShadowTrade]:
        """
        Generate variants adapted to current market regime.
        High-volatility → wider stops; low-volatility → tighter stops.
        """
        dynamic_configs: list[VariantConfig] = []

        if volatility_percentile > 75:
            dynamic_configs.append(VariantConfig(
                variant_type="vol_adapted_wide",
                label="Vol-Adapted Wide Stop",
                stop_multiplier=1.8,
                trailing_enabled=True,
                trailing_atr_mult=2.0,
            ))
        elif volatility_percentile < 25:
            dynamic_configs.append(VariantConfig(
                variant_type="vol_adapted_tight",
                label="Vol-Adapted Tight Stop",
                stop_multiplier=0.5,
                target_multiplier=0.8,
            ))

        if "trend" in regime.lower():
            dynamic_configs.append(VariantConfig(
                variant_type="trend_trail",
                label="Trend Trailing Exit",
                stop_multiplier=1.2,
                trailing_enabled=True,
                trailing_atr_mult=1.5,
            ))

        if "rang" in regime.lower():
            dynamic_configs.append(VariantConfig(
                variant_type="range_tight",
                label="Range-Bound Tight",
                stop_multiplier=0.7,
                target_multiplier=0.7,
            ))

        created: list[ShadowTrade] = []
        for cfg in dynamic_configs:
            shadow = self._build_variant(db, session, trade, cfg)
            if shadow:
                created.append(shadow)

        return created

    # ── internals ──

    def _build_variant(
        self,
        db: DBSession,
        session: ShadowSession,
        trade: Trade,
        cfg: VariantConfig,
        market_snapshot: dict[str, Any] | None = None,
    ) -> ShadowTrade | None:
        if trade.stop_loss is None:
            return None

        is_long = (
            trade.direction == TradeDirection.LONG
            if isinstance(trade.direction, TradeDirection)
            else str(trade.direction).lower() == "long"
        )

        if is_long:
            stop_dist = trade.entry_price - trade.stop_loss
            variant_stop = trade.entry_price - (stop_dist * cfg.stop_multiplier)
        else:
            stop_dist = trade.stop_loss - trade.entry_price
            variant_stop = trade.entry_price + (stop_dist * cfg.stop_multiplier)

        # Target
        variant_target = None
        if trade.take_profit is not None:
            if is_long:
                target_dist = trade.take_profit - trade.entry_price
                variant_target = trade.entry_price + (target_dist * cfg.target_multiplier)
            else:
                target_dist = trade.entry_price - trade.take_profit
                variant_target = trade.entry_price - (target_dist * cfg.target_multiplier)

        variant_size = trade.position_size * cfg.size_multiplier
        risk_amount = abs(trade.entry_price - variant_stop) * variant_size

        trailing_config: dict[str, Any] = {}
        if cfg.trailing_enabled:
            trailing_config = {
                "enabled": True,
                "type": "atr",
                "params": {"atr_multiplier": cfg.trailing_atr_mult},
                "activate_at_r": 1.0,
            }

        direction_val = (
            trade.direction.value
            if isinstance(trade.direction, TradeDirection)
            else str(trade.direction)
        )

        shadow = ShadowTrade(
            id=uuid.uuid4(),
            session_id=session.id,
            real_trade_id=trade.id,
            shadow_type=cfg.variant_type,
            label=cfg.label,
            symbol=trade.symbol,
            direction=direction_val,
            entry_price=trade.entry_price,
            entry_time=trade.entry_time,
            position_size=variant_size,
            stop_loss=variant_stop,
            original_stop=variant_stop,
            take_profit=variant_target,
            trailing_config=trailing_config,
            risk_amount=risk_amount,
            remaining_size=variant_size,
            high_since_entry=trade.entry_price,
            low_since_entry=trade.entry_price,
            strategy_id=trade.declared_strategy_id,
            config={
                "source": "variant_generator",
                "variant_type": cfg.variant_type,
                "stop_multiplier": cfg.stop_multiplier,
                "target_multiplier": cfg.target_multiplier,
                "size_multiplier": cfg.size_multiplier,
            },
        )
        db.add(shadow)
        db.flush()
        return shadow
