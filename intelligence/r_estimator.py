"""
R-Multiple Estimator

Estimates expected R:R and R-multiple for a given setup based on:
1. Historical performance of this strategy in similar conditions
2. Regime-adjusted expectations
3. Volatility-adjusted targets
4. Trader's personal execution history
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from intelligence.market_environment import MarketEnvironment, RegimeType, TrendStrength
from core.models import Trade
from core.enums import TradeStatus

logger = logging.getLogger(__name__)


@dataclass
class REstimate:
    """Estimated R-multiple with confidence."""
    expected_r: float               # Mean expected R
    median_r: float                 # Median (more robust)
    confidence: float               # 0-1
    sample_size: int
    win_probability: float          # Historical win rate in similar conditions
    expected_rr_ratio: float        # Risk:Reward ratio
    upside_r: float                 # 75th percentile R
    downside_r: float               # 25th percentile R

    # Adjustment breakdown
    base_r: float                   # Raw historical R
    regime_adjustment: float        # Regime factor applied
    volatility_adjustment: float    # Volatility factor applied
    personal_adjustment: float      # Trader-specific factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_r": round(self.expected_r, 3),
            "median_r": round(self.median_r, 3),
            "confidence": round(self.confidence, 3),
            "sample_size": self.sample_size,
            "win_probability": round(self.win_probability, 3),
            "expected_rr_ratio": round(self.expected_rr_ratio, 2),
            "upside_r": round(self.upside_r, 3),
            "downside_r": round(self.downside_r, 3),
            "adjustments": {
                "base_r": round(self.base_r, 3),
                "regime": round(self.regime_adjustment, 3),
                "volatility": round(self.volatility_adjustment, 3),
                "personal": round(self.personal_adjustment, 3),
            },
        }


class RMultipleEstimator:
    """
    Estimates R-multiples using historical data and current conditions.

    Uses a layered estimation approach:
    1. Base: Historical R for this strategy
    2. Regime layer: Adjust for current market regime
    3. Volatility layer: Adjust for current volatility
    4. Personal layer: Adjust for trader's execution quality
    """

    def __init__(self, db: AsyncSession, min_sample_size: int = 10):
        self.db = db
        self.min_sample_size = min_sample_size

    async def estimate(
        self,
        strategy_name: str,
        environment: MarketEnvironment,
        trader_id: str,
        direction: Optional[int] = None,
    ) -> REstimate:
        """
        Estimate R-multiple for a strategy in current conditions.
        """
        # 1. Get base historical R-multiples for this strategy
        base_r_values = await self._fetch_historical_r(
            trader_id=trader_id,
            strategy_name=strategy_name,
        )

        # 2. Get regime-specific R-multiples
        regime_r_values = await self._fetch_regime_r(
            trader_id=trader_id,
            strategy_name=strategy_name,
            regime=environment.regime.value,
        )

        # 3. Compute base statistics
        if len(base_r_values) < self.min_sample_size:
            # Insufficient data — use strategy defaults
            return self._default_estimate(strategy_name, environment)

        base_stats = self._compute_stats(base_r_values)
        regime_stats = (
            self._compute_stats(regime_r_values)
            if len(regime_r_values) >= 5
            else None
        )

        # 4. Compute adjustments
        regime_adj = self._regime_adjustment(
            environment, base_stats, regime_stats
        )
        vol_adj = self._volatility_adjustment(environment)
        personal_adj = await self._personal_adjustment(
            trader_id, strategy_name
        )

        # 5. Compute final estimate
        base_r = base_stats["mean"]
        adjusted_r = base_r * regime_adj * vol_adj * personal_adj

        # Use regime-specific stats if available, else base
        effective_stats = regime_stats if regime_stats else base_stats

        # Confidence based on sample size and regime match
        confidence = self._compute_confidence(
            base_count=len(base_r_values),
            regime_count=len(regime_r_values),
            has_regime_data=regime_stats is not None,
        )

        return REstimate(
            expected_r=adjusted_r,
            median_r=effective_stats["median"] * regime_adj * vol_adj * personal_adj,
            confidence=confidence,
            sample_size=len(base_r_values),
            win_probability=effective_stats["win_rate"],
            expected_rr_ratio=max(
                0.5,
                effective_stats["avg_win_r"] / max(
                    abs(effective_stats["avg_loss_r"]), 0.1
                ),
            ),
            upside_r=effective_stats["p75"] * vol_adj,
            downside_r=effective_stats["p25"] * vol_adj,
            base_r=base_r,
            regime_adjustment=regime_adj,
            volatility_adjustment=vol_adj,
            personal_adjustment=personal_adj,
        )

    async def _fetch_historical_r(
        self, trader_id: str, strategy_name: str
    ) -> list[float]:
        """Fetch all historical R-multiples for a strategy."""
        stmt = (
            select(Trade.r_multiple)
            .where(
                and_(
                    Trade.trader_id == trader_id,
                    Trade.declared_strategy_id == strategy_name,
                    Trade.status == TradeStatus.CLOSED,
                    Trade.r_multiple.isnot(None),
                )
            )
            .order_by(Trade.exit_time.desc())
            .limit(200)
        )
        result = await self.db.execute(stmt)
        return [float(r) for r in result.scalars().all()]

    async def _fetch_regime_r(
        self, trader_id: str, strategy_name: str, regime: str
    ) -> list[float]:
        """Fetch R-multiples for strategy in specific regime."""
        stmt = (
            select(Trade.r_multiple)
            .where(
                and_(
                    Trade.trader_id == trader_id,
                    Trade.declared_strategy_id == strategy_name,
                    Trade.status == TradeStatus.CLOSED,
                    Trade.r_multiple.isnot(None),
                    Trade.market_regime == regime,
                )
            )
            .order_by(Trade.exit_time.desc())
            .limit(100)
        )
        result = await self.db.execute(stmt)
        return [float(r) for r in result.scalars().all()]

    def _compute_stats(self, r_values: list[float]) -> dict[str, float]:
        """Compute statistical summary of R-multiples."""
        arr = np.array(r_values)
        wins = arr[arr > 0]
        losses = arr[arr <= 0]

        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "win_rate": float(len(wins) / len(arr)) if len(arr) > 0 else 0.0,
            "avg_win_r": float(np.mean(wins)) if len(wins) > 0 else 0.0,
            "avg_loss_r": float(np.mean(losses)) if len(losses) > 0 else 0.0,
            "max_r": float(np.max(arr)) if len(arr) > 0 else 0.0,
            "min_r": float(np.min(arr)) if len(arr) > 0 else 0.0,
            "expectancy": float(
                (len(wins) / len(arr)) * np.mean(wins) +
                (len(losses) / len(arr)) * np.mean(losses)
            ) if len(arr) > 0 and len(wins) > 0 and len(losses) > 0 else 0.0,
            "count": len(arr),
        }

    def _regime_adjustment(
        self,
        environment: MarketEnvironment,
        base_stats: dict[str, float],
        regime_stats: Optional[dict[str, float]],
    ) -> float:
        """
        Compute regime-based adjustment factor.

        If we have regime-specific data, use the ratio of regime mean
        to base mean. Otherwise, use heuristic adjustments.
        """
        if regime_stats and regime_stats["count"] >= 5:
            base_mean = base_stats["mean"]
            regime_mean = regime_stats["mean"]
            if abs(base_mean) > 0.01:
                ratio = regime_mean / base_mean
                # Clamp to reasonable range
                return max(0.5, min(2.0, ratio))
            return 1.0

        # Heuristic adjustments when no regime-specific data
        regime = environment.regime
        trend = environment.trend

        if regime in {RegimeType.TRENDING_BULLISH, RegimeType.TRENDING_BEARISH}:
            if trend.strength == TrendStrength.STRONG:
                return 1.25
            return 1.10
        elif regime == RegimeType.BREAKOUT:
            return 1.30
        elif regime == RegimeType.RANGING:
            return 0.80
        elif regime == RegimeType.VOLATILE_EXPANSION:
            return 1.15  # More opportunity but more risk
        elif regime == RegimeType.LOW_VOLATILITY_COMPRESSION:
            return 0.70  # Pre-breakout — limited range
        elif regime == RegimeType.BREAKDOWN:
            return 1.20  # Sharp moves
        return 1.0

    def _volatility_adjustment(self, environment: MarketEnvironment) -> float:
        """
        Adjust R estimate based on current volatility.

        Higher volatility = wider targets but also wider stops,
        net effect depends on strategy type.
        """
        return environment.volatility.environment_multiplier

    async def _personal_adjustment(
        self, trader_id: str, strategy_name: str
    ) -> float:
        """
        Adjust for trader's personal execution quality.

        Compares the trader's actual R outcomes to the strategy's
        theoretical R to estimate an execution efficiency factor.
        """
        # Trade model has no planned_rr column — personal adjustment not computable yet
        return 1.0

    def _compute_confidence(
        self,
        base_count: int,
        regime_count: int,
        has_regime_data: bool,
    ) -> float:
        """Compute confidence score for the R estimate."""
        # Base confidence from sample size
        if base_count >= 100:
            base_conf = 0.90
        elif base_count >= 50:
            base_conf = 0.75
        elif base_count >= 20:
            base_conf = 0.60
        elif base_count >= 10:
            base_conf = 0.45
        else:
            base_conf = 0.20

        # Regime data bonus
        if has_regime_data:
            if regime_count >= 20:
                regime_bonus = 0.10
            elif regime_count >= 10:
                regime_bonus = 0.05
            else:
                regime_bonus = 0.02
        else:
            regime_bonus = -0.05

        return min(0.95, max(0.10, base_conf + regime_bonus))

    def _default_estimate(
        self, strategy_name: str, environment: MarketEnvironment
    ) -> REstimate:
        """Return a default estimate when insufficient historical data."""
        vol_mult = environment.volatility.environment_multiplier

        return REstimate(
            expected_r=0.5 * vol_mult,
            median_r=0.3 * vol_mult,
            confidence=0.15,
            sample_size=0,
            win_probability=0.50,
            expected_rr_ratio=2.0,
            upside_r=2.0 * vol_mult,
            downside_r=-1.0 * vol_mult,
            base_r=0.5,
            regime_adjustment=1.0,
            volatility_adjustment=vol_mult,
            personal_adjustment=1.0,
        )