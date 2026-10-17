"""
PTIS Market Context Enrichment Pipeline
========================================

Computes and attaches market environment context to trades.
Provides volatility regime classification, trend detection,
range analysis, and session characterization.

All computations are deterministic and use standard
technical analysis calculations.

Integration
-----------
• Called by ContextAssembler before decision quality evaluation
• Called by trade pipeline on ingestion for enrichment
• Cached per symbol/timeframe to avoid redundant computation
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════

class VolatilityRegime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class TrendDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class RangeState(str, Enum):
    EXPANDING = "expanding"
    COMPRESSING = "compressing"
    NORMAL = "normal"


class SessionPhase(str, Enum):
    PRE_MARKET = "pre_market"
    OPENING = "opening"
    MIDDAY = "midday"
    POWER_HOUR = "power_hour"
    CLOSING = "closing"
    AFTER_HOURS = "after_hours"


class VolatilityState(str, Enum):
    """Fine-grained volatility state (superset of VolatilityRegime)."""
    EXTREME_LOW = "extreme_low"
    COMPRESSING = "compressing"
    STABLE = "stable"
    EXPANDING = "expanding"
    EXTREME_HIGH = "extreme_high"


class RegimeType(str, Enum):
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    RANGING = "ranging"
    VOLATILE_EXPANSION = "volatile_expansion"
    LOW_VOLATILITY_COMPRESSION = "low_volatility_compression"
    UNKNOWN = "unknown"


class TrendStrength(str, Enum):
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MarketContextConfig:
    """Configuration for market context calculations."""
    # ATR
    atr_period: int = 14
    atr_volatility_low_percentile: float = 25.0
    atr_volatility_high_percentile: float = 75.0
    atr_volatility_extreme_percentile: float = 95.0

    # Trend detection
    trend_ema_fast: int = 9
    trend_ema_slow: int = 21
    trend_ema_anchor: int = 50
    trend_strength_lookback: int = 20
    trend_adx_period: int = 14
    trend_neutral_threshold: float = 25.0  # ADX below = no trend (25 is industry standard)
    trend_min_ema_separation_pct: float = 0.15  # EMA9/EMA21 must differ by ≥0.15% to call trend

    # Range analysis
    range_atr_lookback: int = 5
    range_expanding_threshold: float = 1.3
    range_compressing_threshold: float = 0.7

    # Session times (NSE/BSE, IST)
    market_open_hour: int = 9
    market_open_minute: int = 15
    market_close_hour: int = 15
    market_close_minute: int = 30
    opening_duration_minutes: int = 30
    power_hour_start_minutes_before_close: int = 60

    # Cache
    cache_ttl_seconds: int = 300  # 5 min


# ═══════════════════════════════════════════════════════════════════
# OHLCV BAR
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OHLCVBar:
    """Single price bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def true_range(self) -> float:
        """True range for a single bar (no previous close)."""
        return self.high - self.low


def compute_true_ranges(bars: Sequence[OHLCVBar]) -> List[float]:
    """Compute true range series from OHLCV bars."""
    if not bars:
        return []
    trs = [bars[0].range]
    for i in range(1, len(bars)):
        prev_close = bars[i - 1].close
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - prev_close),
            abs(bars[i].low - prev_close),
        )
        trs.append(tr)
    return trs


# ═══════════════════════════════════════════════════════════════════
# TECHNICAL CALCULATIONS
# ═══════════════════════════════════════════════════════════════════

class TechnicalCalculator:
    """Stateless technical indicator calculations."""

    @staticmethod
    def atr(bars: Sequence[OHLCVBar], period: int = 14) -> Optional[float]:
        """Average True Range."""
        trs = compute_true_ranges(bars)
        if len(trs) < period:
            return None
        # Wilder's smoothing
        atr_val = sum(trs[:period]) / period
        for i in range(period, len(trs)):
            atr_val = (atr_val * (period - 1) + trs[i]) / period
        return atr_val

    @staticmethod
    def atr_series(
        bars: Sequence[OHLCVBar], period: int = 14
    ) -> List[Optional[float]]:
        """ATR for each bar (None for insufficient data)."""
        trs = compute_true_ranges(bars)
        result: List[Optional[float]] = [None] * min(period - 1, len(trs))

        if len(trs) < period:
            return result

        atr_val = sum(trs[:period]) / period
        result.append(atr_val)
        for i in range(period, len(trs)):
            atr_val = (atr_val * (period - 1) + trs[i]) / period
            result.append(atr_val)
        return result

    @staticmethod
    def ema(values: Sequence[float], period: int) -> List[float]:
        """Exponential Moving Average."""
        if not values or period <= 0:
            return []
        multiplier = 2.0 / (period + 1)
        result = [values[0]]
        for i in range(1, len(values)):
            result.append(values[i] * multiplier + result[-1] * (1 - multiplier))
        return result

    @staticmethod
    def sma(values: Sequence[float], period: int) -> List[Optional[float]]:
        """Simple Moving Average."""
        result: List[Optional[float]] = []
        for i in range(len(values)):
            if i < period - 1:
                result.append(None)
            else:
                window = values[i - period + 1: i + 1]
                result.append(sum(window) / period)
        return result

    @staticmethod
    def adx(bars: Sequence[OHLCVBar], period: int = 14) -> Optional[float]:
        """
        Average Directional Index — measures trend strength (0-100).
        """
        if len(bars) < period + 1:
            return None

        plus_dm: List[float] = []
        minus_dm: List[float] = []
        trs: List[float] = []

        for i in range(1, len(bars)):
            high_diff = bars[i].high - bars[i - 1].high
            low_diff = bars[i - 1].low - bars[i].low

            plus_dm.append(max(high_diff, 0.0) if high_diff > low_diff else 0.0)
            minus_dm.append(max(low_diff, 0.0) if low_diff > high_diff else 0.0)

            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i - 1].close),
                abs(bars[i].low - bars[i - 1].close),
            )
            trs.append(tr)

        if len(trs) < period:
            return None

        # Wilder's smoothing
        def wilder_smooth(data: List[float], p: int) -> List[float]:
            result = [sum(data[:p])]
            for i in range(p, len(data)):
                result.append(result[-1] - result[-1] / p + data[i])
            return result

        smooth_tr = wilder_smooth(trs, period)
        smooth_plus = wilder_smooth(plus_dm, period)
        smooth_minus = wilder_smooth(minus_dm, period)

        dx_values: List[float] = []
        for i in range(len(smooth_tr)):
            if smooth_tr[i] == 0:
                dx_values.append(0.0)
                continue
            plus_di = 100 * smooth_plus[i] / smooth_tr[i]
            minus_di = 100 * smooth_minus[i] / smooth_tr[i]
            di_sum = plus_di + minus_di
            if di_sum == 0:
                dx_values.append(0.0)
            else:
                dx_values.append(100 * abs(plus_di - minus_di) / di_sum)

        if len(dx_values) < period:
            return dx_values[-1] if dx_values else None

        adx_val = sum(dx_values[:period]) / period
        for i in range(period, len(dx_values)):
            adx_val = (adx_val * (period - 1) + dx_values[i]) / period

        return adx_val

    @staticmethod
    def linear_regression_slope(
        values: Sequence[float], period: int
    ) -> Optional[float]:
        """Normalized slope of linear regression over last `period` bars."""
        if len(values) < period:
            return None

        window = list(values[-period:])
        n = len(window)
        x_sum = n * (n - 1) / 2
        x_sq_sum = n * (n - 1) * (2 * n - 1) / 6
        y_sum = sum(window)
        xy_sum = sum(i * window[i] for i in range(n))

        denom = n * x_sq_sum - x_sum * x_sum
        if denom == 0:
            return 0.0

        slope = (n * xy_sum - x_sum * y_sum) / denom
        avg_price = y_sum / n
        if avg_price == 0:
            return 0.0

        return slope / avg_price * 100  # normalized percentage slope


# ═══════════════════════════════════════════════════════════════════
# MARKET ENVIRONMENT SNAPSHOT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MarketEnvironment:
    """Complete market context for a symbol at a point in time."""
    symbol: str
    timestamp: datetime

    # Volatility
    atr: Optional[float] = None
    atr_percentile: Optional[float] = None
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    historical_volatility_20d: Optional[float] = None

    # Trend
    trend_direction: TrendDirection = TrendDirection.NEUTRAL
    trend_strength: float = 0.0  # 0-100 (ADX)
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ema_anchor: Optional[float] = None
    price_vs_ema_fast_pct: Optional[float] = None

    # Range
    range_state: RangeState = RangeState.NORMAL
    range_expansion_ratio: Optional[float] = None
    daily_range_atr: Optional[float] = None

    # Session
    session_phase: SessionPhase = SessionPhase.MIDDAY
    minutes_from_open: Optional[float] = None
    minutes_to_close: Optional[float] = None

    # Market-wide
    vix_level: Optional[float] = None
    spy_trend: Optional[str] = None
    market_open_time: Optional[datetime] = None

    # Volume
    relative_volume: Optional[float] = None  # vs 20-day avg

    # Price
    current_price: Optional[float] = None
    vwap: Optional[float] = None

    # Opening Range (first ~15 min of NSE session)
    orb_high: Optional[float] = None   # High of opening range period
    orb_low: Optional[float] = None    # Low of opening range period
    orb_bars_used: int = 0             # Number of bars used to compute ORB

    # Meta
    timeframe: str = "5m"

    # ── Computed views expected by intelligence callers ──────────

    @property
    def regime(self) -> "RegimeType":
        """Derive regime from flat fields."""
        # Volatile expansion: high/extreme volatility with expanding range
        if self.volatility_regime in (VolatilityRegime.HIGH, VolatilityRegime.EXTREME):
            if self.range_state == RangeState.EXPANDING:
                return RegimeType.VOLATILE_EXPANSION

        # Low-vol compression: explicit squeeze
        if (self.volatility_regime == VolatilityRegime.LOW
                and self.range_state == RangeState.COMPRESSING):
            return RegimeType.LOW_VOLATILITY_COMPRESSION

        # Neutral direction → ranging regardless of ADX
        if self.trend_direction == TrendDirection.NEUTRAL:
            return RegimeType.RANGING

        # Trending: directional move with meaningful ADX (≥ 25) and not compressing
        if self.trend_strength >= 25 and self.range_state != RangeState.COMPRESSING:
            if self.trend_direction == TrendDirection.UP:
                return RegimeType.TRENDING_BULLISH
            if self.trend_direction == TrendDirection.DOWN:
                return RegimeType.TRENDING_BEARISH

        # Weak trend / choppy
        if self.trend_strength < 25:
            return RegimeType.RANGING

        # Remaining case: strength >= 25 but range is compressing — treat as ranging
        # (price is directional but contracting, not a clean trending regime)
        return RegimeType.RANGING

    @property
    def volatility(self) -> Any:
        """Volatility view with attributes expected by callers."""
        _multipliers = {
            VolatilityRegime.LOW: 0.75,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 1.3,
            VolatilityRegime.EXTREME: 1.6,
        }
        _self = self

        # Map regime + range_state → fine-grained VolatilityState
        if _self.volatility_regime == VolatilityRegime.EXTREME:
            _vs = VolatilityState.EXTREME_HIGH
        elif _self.volatility_regime == VolatilityRegime.HIGH:
            _vs = VolatilityState.EXPANDING
        elif _self.volatility_regime == VolatilityRegime.LOW:
            if _self.range_state == RangeState.COMPRESSING:
                _vs = VolatilityState.EXTREME_LOW
            else:
                _vs = VolatilityState.COMPRESSING
        else:  # NORMAL
            if _self.range_state == RangeState.EXPANDING:
                _vs = VolatilityState.EXPANDING
            elif _self.range_state == RangeState.COMPRESSING:
                _vs = VolatilityState.COMPRESSING
            else:
                _vs = VolatilityState.STABLE

        class _VolatilityView:
            atr_ratio: float = _self.range_expansion_ratio or 1.0
            is_compression: bool = _self.range_state == RangeState.COMPRESSING
            is_expansion: bool = _self.range_state == RangeState.EXPANDING
            environment_multiplier: float = _multipliers.get(_self.volatility_regime, 1.0)
            state = _vs

        return _VolatilityView()

    @property
    def trend(self) -> Any:
        """Trend view with attributes expected by callers."""
        adx = self.trend_strength
        if adx >= 50:
            _strength = TrendStrength.VERY_STRONG
        elif adx >= 35:
            _strength = TrendStrength.STRONG
        elif adx >= 25:
            _strength = TrendStrength.MODERATE
        elif adx >= 15:
            _strength = TrendStrength.WEAK
        else:
            _strength = TrendStrength.VERY_WEAK

        _self = self

        _dir_num = (
            1 if _self.trend_direction == TrendDirection.UP
            else -1 if _self.trend_direction == TrendDirection.DOWN
            else 0
        )
        _ema_aligned = (
            _self.ema_fast is not None
            and _self.ema_slow is not None
            and (
                (_self.trend_direction == TrendDirection.UP and _self.ema_fast > _self.ema_slow)
                or (_self.trend_direction == TrendDirection.DOWN and _self.ema_fast < _self.ema_slow)
            )
        )

        class _TrendView:
            is_trending: bool = adx > 25
            strength = _strength
            direction: int = _dir_num      # numeric: 1=up, -1=down, 0=neutral
            ema_alignment: bool = _ema_aligned

        return _TrendView()

    @property
    def structure(self) -> Any:
        """Structure view — placeholder until structure detection is implemented."""
        class _StructureView:
            has_structure_break: bool = False
            has_liquidity_sweep: bool = False
            nearest_support: Optional[float] = None
            nearest_resistance: Optional[float] = None
            key_levels: list = []

        return _StructureView()

    @property
    def environment_score(self) -> float:
        """
        0–1 score representing how favourable the environment is for trading.
        Higher = cleaner / more readable market.
        """
        score = 0.5  # base

        # Trend contributes positively when strong
        adx = self.trend_strength
        if adx >= 35:
            score += 0.2
        elif adx >= 25:
            score += 0.1
        elif adx < 15:
            score -= 0.1

        # Normal vol is best; extreme vol penalises
        if self.volatility_regime == VolatilityRegime.NORMAL:
            score += 0.1
        elif self.volatility_regime == VolatilityRegime.EXTREME:
            score -= 0.2
        elif self.volatility_regime == VolatilityRegime.LOW:
            score -= 0.05

        # Expanding range in a trend is good
        if self.range_state == RangeState.EXPANDING and adx >= 25:
            score += 0.1

        return max(0.0, min(1.0, score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "atr": round(self.atr, 4) if self.atr else None,
            "atr_percentile": (
                round(self.atr_percentile, 1) if self.atr_percentile else None
            ),
            "volatility_regime": self.volatility_regime.value,
            "volatility_state": self.volatility_regime.value,   # alias
            "trend_direction": self.trend_direction.value,
            "trend_strength": round(self.trend_strength, 1),
            "ema_fast": round(self.ema_fast, 4) if self.ema_fast else None,
            "ema_slow": round(self.ema_slow, 4) if self.ema_slow else None,
            "range_state": self.range_state.value,
            "range_expansion_ratio": (
                round(self.range_expansion_ratio, 2)
                if self.range_expansion_ratio else None
            ),
            "session_phase": self.session_phase.value,
            "minutes_from_open": self.minutes_from_open,
            "minutes_to_close": self.minutes_to_close,
            "vix_level": self.vix_level,
            "relative_volume": (
                round(self.relative_volume, 2) if self.relative_volume else None
            ),
            "market_open_time": (
                self.market_open_time.isoformat() if self.market_open_time else None
            ),
            "regime": self.regime.value,
            "environment_score": round(self.environment_score, 3),
            "vwap": round(self.vwap, 4) if self.vwap else None,
            "orb_high": round(self.orb_high, 2) if self.orb_high else None,
            "orb_low": round(self.orb_low, 2) if self.orb_low else None,
            "timeframe": self.timeframe,
        }

    def for_context_assembler(self) -> Dict[str, Any]:
        """Return fields matching TradeContext expectations."""
        return {
            "atr": self.atr,
            "volatility_regime": self.volatility_regime.value,
            "trend_direction": self.trend_direction.value,
            "trend_strength": self.trend_strength,
            "vix_level": self.vix_level,
            "session_type": self.session_phase.value,
            "market_open_time": self.market_open_time,
        }


# ═══════════════════════════════════════════════════════════════════
# LRU CACHE
# ═══════════════════════════════════════════════════════════════════

class _TTLCache:
    """Simple TTL-bounded LRU cache for market environment snapshots."""

    def __init__(self, max_size: int = 200, ttl_seconds: int = 300) -> None:
        self._cache: OrderedDict[str, Tuple[datetime, Any]] = OrderedDict()
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            ts, value = self._cache[key]
            if datetime.utcnow() - ts < self._ttl:
                self._cache.move_to_end(key)
                return value
            del self._cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        self._cache[key] = (datetime.utcnow(), value)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


# ═══════════════════════════════════════════════════════════════════
# MARKET CONTEXT ENGINE
# ═══════════════════════════════════════════════════════════════════

class MarketContextEngine:
    """
    Computes market environment context from OHLCV data.

    Usage
    -----
    >>> engine = MarketContextEngine()
    >>> env = engine.compute(bars, symbol="AAPL", timestamp=now)
    >>> # Or with async data fetching:
    >>> env = await engine.get_environment(symbol="AAPL", timestamp=now)
    """

    def __init__(
        self,
        config: Optional[MarketContextConfig] = None,
        price_fetcher: Any = None,
    ) -> None:
        self.cfg = config or MarketContextConfig()
        self._calc = TechnicalCalculator()
        self._cache = _TTLCache(
            max_size=200, ttl_seconds=self.cfg.cache_ttl_seconds
        )
        self._price_fetcher = price_fetcher
        logger.info("MarketContextEngine initialized.")

    def compute(
        self,
        bars: Sequence[OHLCVBar],
        symbol: str,
        timestamp: Optional[datetime] = None,
        vix_level: Optional[float] = None,
    ) -> MarketEnvironment:
        """
        Compute full market environment from OHLCV bars.

        Parameters
        ----------
        bars : Sequence[OHLCVBar]
            Historical OHLCV bars, newest last.
        symbol : str
            Ticker symbol.
        timestamp : datetime, optional
            Point in time (defaults to last bar timestamp).
        vix_level : float, optional
            Current VIX level if available.
        """
        ts = timestamp or (bars[-1].timestamp if bars else datetime.utcnow())
        env = MarketEnvironment(symbol=symbol, timestamp=ts)

        if not bars or len(bars) < 2:
            return env

        closes = [b.close for b in bars]

        # ── Volatility ──────────────────────────────────────────
        env.atr = self._calc.atr(bars, self.cfg.atr_period)

        atr_series = self._calc.atr_series(bars, self.cfg.atr_period)
        valid_atrs = [a for a in atr_series if a is not None]
        if valid_atrs and env.atr is not None:
            sorted_atrs = sorted(valid_atrs)
            rank = sum(1 for a in sorted_atrs if a <= env.atr)
            env.atr_percentile = rank / len(sorted_atrs) * 100

            if env.atr_percentile <= self.cfg.atr_volatility_low_percentile:
                env.volatility_regime = VolatilityRegime.LOW
            elif env.atr_percentile <= self.cfg.atr_volatility_high_percentile:
                env.volatility_regime = VolatilityRegime.NORMAL
            elif env.atr_percentile <= self.cfg.atr_volatility_extreme_percentile:
                env.volatility_regime = VolatilityRegime.HIGH
            else:
                env.volatility_regime = VolatilityRegime.EXTREME

        # Historical volatility (annualized, 20-day)
        if len(closes) >= 21:
            log_returns = [
                math.log(closes[i] / closes[i - 1])
                for i in range(len(closes) - 20, len(closes))
                if closes[i - 1] > 0
            ]
            if log_returns:
                import statistics as st
                env.historical_volatility_20d = (
                    st.stdev(log_returns) * math.sqrt(252) * 100
                )

        # ── Trend ───────────────────────────────────────────────
        ema_fast = self._calc.ema(closes, self.cfg.trend_ema_fast)
        ema_slow = self._calc.ema(closes, self.cfg.trend_ema_slow)
        ema_anchor = self._calc.ema(closes, self.cfg.trend_ema_anchor)

        if ema_fast:
            env.ema_fast = ema_fast[-1]
        if ema_slow:
            env.ema_slow = ema_slow[-1]
        if ema_anchor:
            env.ema_anchor = ema_anchor[-1]

        if env.ema_fast and closes[-1] != 0:
            env.price_vs_ema_fast_pct = (
                (closes[-1] - env.ema_fast) / closes[-1] * 100
            )

        # ADX for trend strength
        adx_val = self._calc.adx(bars, self.cfg.trend_adx_period)
        env.trend_strength = adx_val if adx_val is not None else 0.0

        # Direction from EMA stack
        if env.ema_fast and env.ema_slow and closes[-1] > 0:
            ema_sep_pct = abs(env.ema_fast - env.ema_slow) / closes[-1] * 100
            _enough_sep = ema_sep_pct >= self.cfg.trend_min_ema_separation_pct
            _strong = env.trend_strength >= self.cfg.trend_neutral_threshold
            if (
                env.ema_fast > env.ema_slow
                and closes[-1] > env.ema_fast
                and _enough_sep
                and _strong
            ):
                env.trend_direction = TrendDirection.UP
            elif (
                env.ema_fast < env.ema_slow
                and closes[-1] < env.ema_fast
                and _enough_sep
                and _strong
            ):
                env.trend_direction = TrendDirection.DOWN
            else:
                env.trend_direction = TrendDirection.NEUTRAL

        # ── Range Analysis ──────────────────────────────────────
        if env.atr and env.atr > 0:
            recent_ranges = [
                b.range for b in bars[-self.cfg.range_atr_lookback:]
            ]
            if recent_ranges:
                avg_recent_range = sum(recent_ranges) / len(recent_ranges)
                env.range_expansion_ratio = avg_recent_range / env.atr

                if env.range_expansion_ratio > self.cfg.range_expanding_threshold:
                    env.range_state = RangeState.EXPANDING
                elif env.range_expansion_ratio < self.cfg.range_compressing_threshold:
                    env.range_state = RangeState.COMPRESSING
                else:
                    env.range_state = RangeState.NORMAL

            if bars[-1].range > 0:
                env.daily_range_atr = bars[-1].range / env.atr

        # ── Session Phase ───────────────────────────────────────
        env.session_phase, env.minutes_from_open, env.minutes_to_close = (
            self._determine_session_phase(ts)
        )

        open_dt = ts.replace(
            hour=self.cfg.market_open_hour,
            minute=self.cfg.market_open_minute,
            second=0,
            microsecond=0,
        )
        env.market_open_time = open_dt

        # ── Volume ──────────────────────────────────────────────
        volumes = [b.volume for b in bars if b.volume > 0]
        if len(volumes) >= 21 and bars[-1].volume > 0:
            avg_vol_20 = sum(volumes[-21:-1]) / 20
            if avg_vol_20 > 0:
                env.relative_volume = bars[-1].volume / avg_vol_20

        # ── Market-wide ─────────────────────────────────────────
        env.vix_level = vix_level

        return env

    def analyze(
        self,
        df: Any,
        symbol: str,
        timeframe: str = "5m",
    ) -> MarketEnvironment:
        """
        Convenience wrapper: convert a pandas DataFrame (OHLCV) to bars and compute.
        Column names accepted: open/Open, high/High, low/Low, close/Close, volume/Volume.
        """
        bars: List[OHLCVBar] = []
        for ts, row in df.iterrows():
            try:
                ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.utcnow()
                bars.append(OHLCVBar(
                    timestamp=ts_dt,
                    open=float(row.get("open", row.get("Open", 0))),
                    high=float(row.get("high", row.get("High", 0))),
                    low=float(row.get("low", row.get("Low", 0))),
                    close=float(row.get("close", row.get("Close", 0))),
                    volume=float(row.get("volume", row.get("Volume", 0))),
                ))
            except Exception:
                continue

        if not bars:
            raise ValueError("Insufficient data: no bars parsed from DataFrame")

        if len(bars) < 30:
            raise ValueError(
                f"Insufficient data: need at least 30 bars, got {len(bars)}"
            )

        env = self.compute(bars, symbol=symbol)
        env.timeframe = timeframe
        if bars:
            env.current_price = bars[-1].close

        # Session phase must reflect CURRENT real-world time in IST,
        # not the last candle's timestamp.  Otherwise evaluating on a
        # Saturday evening shows "Midday" because the last candle was
        # Friday afternoon.
        from datetime import timezone, timedelta
        _IST = timezone(timedelta(hours=5, minutes=30))
        _now_ist = datetime.now(_IST).replace(tzinfo=None)   # naive IST
        env.session_phase, env.minutes_from_open, env.minutes_to_close = (
            self._determine_session_phase(_now_ist)
        )
        env.market_open_time = _now_ist.replace(
            hour=self.cfg.market_open_hour,
            minute=self.cfg.market_open_minute,
            second=0, microsecond=0,
        )

        # VWAP — session-only approximation.
        # NSE trading day = 375 min. bars_per_day = 375 / bar_minutes.
        # We use only the most recent session's worth of bars so VWAP
        # doesn't bleed across days (timestamps may be unavailable when
        # the DataFrame index is stripped before passing here).
        _tf_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "60m": 60, "1d": 375, "D": 375,
        }
        _bar_min = _tf_minutes.get(env.timeframe, 5)
        _session_bars = max(1, 375 // _bar_min)   # bars in one NSE session
        session_bars = bars[-_session_bars:]
        cum_tp_vol = 0.0
        cum_vol = 0.0
        for b in session_bars:
            tp = (b.high + b.low + b.close) / 3.0
            cum_tp_vol += tp * b.volume
            cum_vol += b.volume
        if cum_vol > 0:
            env.vwap = cum_tp_vol / cum_vol

        # Opening Range — first 15 min of the NSE session.
        # ORB period = ceil(15 / bar_minutes) bars, min 1.
        # session_bars already contains only the current session's bars.
        _orb_period_bars = max(1, 15 // _bar_min)   # e.g. 15m→1, 5m→3, 1m→15
        orb_slice = session_bars[:_orb_period_bars]
        if orb_slice:
            env.orb_high = max(b.high for b in orb_slice)
            env.orb_low = min(b.low for b in orb_slice)
            env.orb_bars_used = len(orb_slice)

        return env

    async def get_environment(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Async interface for ContextAssembler.
        Fetches bars from price_fetcher if available, else returns cached.
        """
        cache_key = f"{symbol}:{timestamp.isoformat() if timestamp else 'now'}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if self._price_fetcher is None:
            logger.debug("No price fetcher — cannot compute environment for %s.", symbol)
            return None

        try:
            bars = await self._price_fetcher.get_daily_bars(
                symbol=symbol,
                count=100,
                end_date=timestamp,
            )
            if not bars:
                return None

            ohlcv_bars = [
                OHLCVBar(
                    timestamp=b["timestamp"],
                    open=b["open"],
                    high=b["high"],
                    low=b["low"],
                    close=b["close"],
                    volume=b.get("volume", 0),
                )
                for b in bars
            ]

            vix = None
            if hasattr(self._price_fetcher, "get_vix"):
                vix = await self._price_fetcher.get_vix()

            env = self.compute(ohlcv_bars, symbol, timestamp, vix)
            result = env.for_context_assembler()
            self._cache.put(cache_key, result)
            return result

        except Exception:
            logger.exception("Failed to compute environment for %s.", symbol)
            return None

    def _determine_session_phase(
        self, ts: datetime
    ) -> Tuple[SessionPhase, Optional[float], Optional[float]]:
        """Determine current market session phase."""
        market_open = ts.replace(
            hour=self.cfg.market_open_hour,
            minute=self.cfg.market_open_minute,
            second=0,
        )
        market_close = ts.replace(
            hour=self.cfg.market_close_hour,
            minute=self.cfg.market_close_minute,
            second=0,
        )

        if ts < market_open:
            return SessionPhase.PRE_MARKET, None, None

        if ts > market_close:
            return SessionPhase.AFTER_HOURS, None, None

        mins_from_open = (ts - market_open).total_seconds() / 60.0
        mins_to_close = (market_close - ts).total_seconds() / 60.0

        if mins_from_open <= self.cfg.opening_duration_minutes:
            phase = SessionPhase.OPENING
        elif mins_to_close <= self.cfg.power_hour_start_minutes_before_close:
            if mins_to_close <= 15:
                phase = SessionPhase.CLOSING
            else:
                phase = SessionPhase.POWER_HOUR
        else:
            phase = SessionPhase.MIDDAY

        return phase, mins_from_open, mins_to_close

    def enrich_trade(
        self,
        trade_data: Dict[str, Any],
        bars: Sequence[OHLCVBar],
        vix_level: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Enrich a trade dict with market context fields.
        Used in the ingestion pipeline.
        """
        symbol = trade_data.get("symbol", "")
        entry_time = trade_data.get("entry_time")

        env = self.compute(bars, symbol, entry_time, vix_level)
        context = env.for_context_assembler()

        enriched = dict(trade_data)
        enriched.update({
            "atr": context["atr"],
            "volatility_regime": context["volatility_regime"],
            "trend_direction": context["trend_direction"],
            "trend_strength": context["trend_strength"],
            "vix_level": context["vix_level"],
            "session_type": context["session_type"],
            "market_open_time": context["market_open_time"],
        })

        return enriched
# Alias expected by callers
MarketEnvironmentAnalyzer = MarketContextEngine


@dataclass
class StructureEvent:
    """
    A significant market structure event (swing high/low, breakout, breakdown).
    Used for pattern detection and regime confirmation.
    """
    event_type: str          # "swing_high" | "swing_low" | "breakout" | "breakdown" | "consolidation"
    price: float
    bar_index: int
    timestamp: Optional[Any] = None
    significance: float = 1.0   # 0-1, higher = more significant
    confirmed: bool = False

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "price": self.price,
            "bar_index": self.bar_index,
            "significance": self.significance,
            "confirmed": self.confirmed,
        }
