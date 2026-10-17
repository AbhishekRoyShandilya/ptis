"""
PTIS Market Context Enrichment
==============================

Automatically enriches trade data with market context including:
- ATR (Average True Range) and volatility metrics
- Trend strength and direction indicators
- Market regime classification
- Session behavior analysis
- Range expansion/compression detection

This context is essential for:
- Decision quality evaluation (market context dimension)
- Strategy performance analysis by regime
- Behavioral pattern detection
- Coaching and improvement insights

Design
------
- Event-driven enrichment on trade entry/close
- Configurable market data providers
- Caching for performance
- Graceful degradation when data unavailable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from core.events import Event, EventType, get_event_bus
from core.models import Trade

logger = logging.getLogger(__name__)


class VolatilityRegime(str, Enum):
    """Volatility regime classification."""
    EXTREME_LOW = "extreme_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME_HIGH = "extreme_high"


class TrendStrength(str, Enum):
    """Trend strength classification."""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class TrendDirection(str, Enum):
    """Trend direction classification."""
    STRONG_DOWN = "strong_down"
    DOWN = "down"
    SIDEWAYS = "sideways"
    UP = "up"
    STRONG_UP = "strong_up"


class RangeState(str, Enum):
    """Range expansion/compression state."""
    TIGHT_RANGE = "tight_range"
    NORMAL_RANGE = "normal_range"
    EXPANDING_RANGE = "expanding_range"
    BREAKOUT_MODE = "breakout_mode"


@dataclass
class MarketContext:
    """Complete market context for a specific time and symbol."""
    
    # Basic market data
    timestamp: datetime
    symbol: str
    
    # Price data
    current_price: float
    open_price: float
    high_price: float
    low_price: float
    volume: int
    
    # Volatility metrics
    atr_14: float  # 14-period ATR
    atr_ratio: float  # Current ATR / average ATR
    volatility_regime: VolatilityRegime
    volatility_percentile: float  # 0-1, where this volatility sits historically
    
    # Trend analysis
    trend_direction: TrendDirection
    trend_strength: TrendStrength
    trend_duration_bars: int  # How many bars trend has persisted
    
    # Range analysis
    range_state: RangeState
    range_width_pct: float  # Range width as % of price
    range_position: float  # Current price position in range (0-1)
    
    # Session context
    session_type: str  # pre_market, regular, after_hours
    session_progress: float  # 0-1 through session
    volume_profile: str  # low, normal, high volume for this time
    
    # Market internals (if available)
    vix_level: Optional[float] = None
    market_breadth: Optional[str] = None  # positive, negative, neutral
    sector_performance: Optional[Dict[str, float]] = None
    
    # Time-based metrics
    time_from_open_minutes: float = 0.0
    time_to_close_minutes: float = 0.0
    
    def refresh_from_price_feed(self) -> bool:
        """
        Update this MarketContext's price fields from the live price feed.

        Imports get_price_feed_manager, fetches the last tick for self.symbol,
        and overwrites price/volume/timestamp fields.  Returns True if the
        context was updated, False if no tick was available.
        """
        try:
            from shadow.price_feed import get_price_feed_manager
        except ImportError:
            logger.warning("shadow.price_feed not available — cannot refresh from live feed")
            return False

        manager = get_price_feed_manager()
        tick = manager.get_last_tick(self.symbol)

        if tick is None:
            logger.debug("No live tick available for %s", self.symbol)
            return False

        self.timestamp = tick.timestamp
        self.current_price = tick.price

        if tick.open is not None:
            self.open_price = tick.open
        if tick.high is not None:
            self.high_price = tick.high
        if tick.low is not None:
            self.low_price = tick.low
        if tick.volume is not None:
            self.volume = int(tick.volume)

        # Also update close alias if the tick carries it
        # (current_price already set above from tick.price)

        logger.debug("MarketContext for %s refreshed from live feed (price=%.2f)", self.symbol, tick.price)
        return True

    @classmethod
    def from_live_feed(cls, symbol: str) -> Optional["MarketContext"]:
        """
        Factory: construct a MarketContext populated with live tick data.

        Returns None if the price feed has no tick for the symbol.
        """
        try:
            from shadow.price_feed import get_price_feed_manager
        except ImportError:
            logger.warning("shadow.price_feed not available — cannot build from live feed")
            return None

        manager = get_price_feed_manager()
        tick = manager.get_last_tick(symbol.upper())

        if tick is None:
            logger.debug("No live tick for %s — cannot build MarketContext", symbol)
            return None

        return cls(
            timestamp=tick.timestamp,
            symbol=tick.symbol,
            current_price=tick.price,
            open_price=tick.open if tick.open is not None else tick.price,
            high_price=tick.high if tick.high is not None else tick.price,
            low_price=tick.low if tick.low is not None else tick.price,
            volume=int(tick.volume) if tick.volume is not None else 0,
            # Defaults for computed fields — caller should enrich further
            atr_14=0.0,
            atr_ratio=1.0,
            volatility_regime=VolatilityRegime.NORMAL,
            volatility_percentile=0.5,
            trend_direction=TrendDirection.SIDEWAYS,
            trend_strength=TrendStrength.WEAK,
            trend_duration_bars=0,
            range_state=RangeState.NORMAL_RANGE,
            range_width_pct=0.0,
            range_position=0.5,
            session_type="unknown",
            session_progress=0.5,
            volume_profile="normal",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "current_price": self.current_price,
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "volume": self.volume,
            "atr_14": self.atr_14,
            "atr_ratio": self.atr_ratio,
            "volatility_regime": self.volatility_regime.value,
            "volatility_percentile": round(self.volatility_percentile, 3),
            "trend_direction": self.trend_direction.value,
            "trend_strength": self.trend_strength.value,
            "trend_duration_bars": self.trend_duration_bars,
            "range_state": self.range_state.value,
            "range_width_pct": round(self.range_width_pct, 4),
            "range_position": round(self.range_position, 3),
            "session_type": self.session_type,
            "session_progress": round(self.session_progress, 3),
            "volume_profile": self.volume_profile,
            "vix_level": self.vix_level,
            "market_breadth": self.market_breadth,
            "sector_performance": self.sector_performance,
            "time_from_open_minutes": round(self.time_from_open_minutes, 1),
            "time_to_close_minutes": round(self.time_to_close_minutes, 1),
        }


@dataclass
class MarketContextConfig:
    """Configuration for market context enrichment."""
    
    # ATR calculation
    atr_period: int = 14
    atr_history_days: int = 100  # Days for ATR percentile calculation
    
    # Trend analysis
    trend_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    trend_strength_threshold: float = 0.6  # Correlation threshold for strong trend
    
    # Volatility regime thresholds
    volatility_percentile_thresholds: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    
    # Range analysis
    range_lookback_bars: int = 20
    breakout_threshold_pct: float = 0.02  # 2% of price
    
    # Session analysis
    market_open_time: Tuple[int, int] = (9, 15)   # 9:15 AM IST (NSE)
    market_close_time: Tuple[int, int] = (15, 30)  # 3:30 PM IST (NSE)
    
    # Data requirements
    min_data_points: int = 50
    max_data_age_hours: int = 24
    
    # Caching
    cache_ttl_minutes: int = 15
    max_cache_size: int = 1000


class MarketDataProvider:
    """
    Abstract interface for market data providers.
    
    Implementations can connect to:
    - Broker APIs (Interactive Brokers, Alpaca, etc.)
    - Market data services (Polygon, Quandl, etc.)
    - Internal data stores
    - Real-time feeds
    """
    
    def get_price_data(
        self, 
        symbol: str, 
        timestamp: datetime,
        lookback_bars: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get historical price data for context calculation.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        timestamp : datetime
            Target timestamp for context
        lookback_bars : int
            Number of historical bars to retrieve
        
        Returns
        -------
        Optional[List[Dict[str, Any]]]
            List of OHLCV data, newest first, or None if unavailable
        """
        raise NotImplementedError
    
    def get_market_internals(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Get market-wide indicators (VIX, breadth, etc.).
        
        Parameters
        ----------
        timestamp : datetime
            Target timestamp
        
        Returns
        -------
        Dict[str, Any]
            Market internals data
        """
        raise NotImplementedError


class MockMarketDataProvider(MarketDataProvider):
    """
    Mock market data provider for testing and development.
    Generates realistic-looking market data.
    """
    
    def get_price_data(
        self, 
        symbol: str, 
        timestamp: datetime,
        lookback_bars: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate mock price data."""
        import random
        
        # Generate realistic price series
        base_price = 100.0
        data = []
        current_price = base_price
        
        for i in range(lookback_bars):
            # Random walk with trend
            trend = 0.0001 * (i - lookback_bars // 2)  # Slight upward trend
            noise = random.gauss(0, 0.01)  # 1% daily volatility
            
            change = trend + noise
            current_price *= (1 + change)
            
            # Generate OHLC
            high = current_price * (1 + abs(random.gauss(0, 0.005)))
            low = current_price * (1 - abs(random.gauss(0, 0.005)))
            open_price = low + (high - low) * random.random()
            close = current_price
            volume = int(random.gauss(1000000, 200000))
            
            data.append({
                "timestamp": timestamp - timedelta(minutes=5 * (lookback_bars - i)),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            })
        
        return data
    
    def get_market_internals(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate mock market internals."""
        import random
        
        return {
            "vix_level": random.gauss(20, 5),
            "market_breadth": random.choice(["positive", "negative", "neutral"]),
            "sector_performance": {
                "technology": random.gauss(0, 0.02),
                "healthcare": random.gauss(0, 0.015),
                "finance": random.gauss(0, 0.018),
            }
        }


class MarketContextEnricher:
    """
    Enriches trades with market context data.
    
    Listens for trade events and automatically attaches market context
    to trades for decision quality evaluation and analysis.
    """
    
    def __init__(
        self, 
        config: Optional[MarketContextConfig] = None,
        data_provider: Optional[MarketDataProvider] = None
    ):
        self.cfg = config or MarketContextConfig()
        self.data_provider = data_provider or MockMarketDataProvider()
        self._event_bus = get_event_bus()
        
        # Cache for context data
        self._context_cache: Dict[Tuple[str, datetime], MarketContext] = {}
        
        # Subscribe to trade events
        self._setup_event_listeners()
        
        logger.info("MarketContextEnricher initialized")
    
    def _setup_event_listeners(self) -> None:
        """Subscribe to trade events for enrichment."""
        self._event_bus.subscribe(EventType.TRADE_ENTERED, self._on_trade_entered)
        self._event_bus.subscribe(EventType.TRADE_CLOSED, self._on_trade_closed)
    
    async def enrich_trade(
        self,
        trade: Trade,
        timestamp: Optional[datetime] = None,
        use_live_feed: bool = True
    ) -> MarketContext:
        """
        Enrich a trade with market context.
        
        Parameters
        ----------
        trade : Trade
            Trade to enrich
        timestamp : Optional[datetime]
            Timestamp for context (defaults to trade entry time)
        
        Returns
        -------
        MarketContext
            Market context for the trade
        """
        target_time = timestamp or trade.entry_time or datetime.utcnow()
        cache_key = (trade.symbol, target_time)
        
        # Check cache first
        if cache_key in self._context_cache:
            cached_context, cached_at = self._context_cache[cache_key]
            if (datetime.utcnow() - cached_at).total_seconds() < self.cfg.cache_ttl_minutes * 60:
                logger.debug(f"Using cached market context for {trade.symbol}")
                return cached_context
        
        # Calculate fresh context — try live feed first if requested
        context = await self._calculate_market_context(
            trade.symbol, target_time, use_live_feed=use_live_feed
        )

        # Cache the result
        self._context_cache[cache_key] = (context, datetime.utcnow())
        
        # Cleanup old cache entries
        self._cleanup_cache()
        
        return context
    
    async def _calculate_market_context(
        self,
        symbol: str,
        timestamp: datetime,
        use_live_feed: bool = True,
    ) -> MarketContext:
        """Calculate complete market context for symbol and timestamp.

        When *use_live_feed* is True the method first attempts to seed the
        current-bar data from the PriceFeedManager.  If no live tick is
        available it falls back to the configured data_provider as before.
        """

        # ── Attempt live price feed first ──
        live_tick = None
        if use_live_feed:
            try:
                from shadow.price_feed import get_price_feed_manager
                live_tick = get_price_feed_manager().get_last_tick(symbol.upper())
            except ImportError:
                logger.debug("shadow.price_feed not importable — skipping live feed")

        # Get price data
        price_data = self.data_provider.get_price_data(
            symbol, timestamp, self.cfg.atr_period + 50
        )
        
        if not price_data or len(price_data) < self.cfg.min_data_points:
            logger.warning(f"Insufficient price data for {symbol}")
            return self._create_default_context(symbol, timestamp)
        
        # Get market internals
        internals = self.data_provider.get_market_internals(timestamp)
        
        # Calculate context components
        current_bar = price_data[0]  # Most recent
        historical_bars = price_data[1:]  # Historical data

        # If a live tick is available, overlay it onto the current bar so
        # downstream calculations (range analysis, session, etc.) use the
        # freshest data while historical bars remain from the data provider.
        if live_tick is not None:
            current_bar = {
                "close": live_tick.price,
                "open": live_tick.open if live_tick.open is not None else current_bar["open"],
                "high": live_tick.high if live_tick.high is not None else current_bar["high"],
                "low": live_tick.low if live_tick.low is not None else current_bar["low"],
                "volume": int(live_tick.volume) if live_tick.volume is not None else current_bar["volume"],
            }
            timestamp = live_tick.timestamp  # use tick timestamp
            logger.debug("Using live tick for %s (price=%.2f)", symbol, live_tick.price)

        # Basic market data
        context = MarketContext(
            timestamp=timestamp,
            symbol=symbol,
            current_price=current_bar["close"],
            open_price=current_bar["open"],
            high_price=current_bar["high"],
            low_price=current_bar["low"],
            volume=current_bar["volume"],
        )
        
        # Volatility analysis
        context.atr_14 = self._calculate_atr(historical_bars[:self.cfg.atr_period])
        context.atr_ratio = self._calculate_atr_ratio(context.atr_14, historical_bars)
        context.volatility_regime = self._classify_volatility_regime(context.atr_ratio)
        context.volatility_percentile = self._calculate_volatility_percentile(context.atr_14, historical_bars)
        
        # Trend analysis
        trend_data = self._analyze_trend(historical_bars)
        context.trend_direction = trend_data["direction"]
        context.trend_strength = trend_data["strength"]
        context.trend_duration_bars = trend_data["duration"]
        
        # Range analysis
        range_data = self._analyze_range(historical_bars[:self.cfg.range_lookback_bars], current_bar)
        context.range_state = range_data["state"]
        context.range_width_pct = range_data["width_pct"]
        context.range_position = range_data["position"]
        
        # Session analysis
        session_data = self._analyze_session(timestamp, current_bar["volume"])
        context.session_type = session_data["type"]
        context.session_progress = session_data["progress"]
        context.volume_profile = session_data["volume_profile"]
        
        # Market internals
        context.vix_level = internals.get("vix_level")
        context.market_breadth = internals.get("market_breadth")
        context.sector_performance = internals.get("sector_performance")
        
        # Time metrics
        context.time_from_open_minutes = session_data["time_from_open"]
        context.time_to_close_minutes = session_data["time_to_close"]
        
        return context
    
    def _calculate_atr(self, bars: List[Dict[str, Any]]) -> float:
        """Calculate Average True Range for given bars."""
        if len(bars) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(len(bars)):
            if i == 0:
                tr = bars[i]["high"] - bars[i]["low"]
            else:
                high_low = bars[i]["high"] - bars[i]["low"]
                high_close = abs(bars[i]["high"] - bars[i-1]["close"])
                low_close = abs(bars[i]["low"] - bars[i-1]["close"])
                tr = max(high_low, high_close, low_close)
            
            true_ranges.append(tr)
        
        return sum(true_ranges) / len(true_ranges)
    
    def _calculate_atr_ratio(self, current_atr: float, historical_bars: List[Dict[str, Any]]) -> float:
        """Calculate current ATR relative to historical average."""
        if len(historical_bars) < self.cfg.atr_history_days:
            return 1.0
        
        # Calculate historical ATRs
        historical_atrs = []
        for i in range(self.cfg.atr_period, len(historical_bars)):
            atr_bars = historical_bars[i-self.cfg.atr_period:i]
            atr = self._calculate_atr(atr_bars)
            historical_atrs.append(atr)
        
        if not historical_atrs:
            return 1.0
        
        avg_historical_atr = sum(historical_atrs) / len(historical_atrs)
        return current_atr / avg_historical_atr if avg_historical_atr > 0 else 1.0
    
    def _classify_volatility_regime(self, atr_ratio: float) -> VolatilityRegime:
        """Classify volatility regime based on ATR ratio."""
        thresholds = self.cfg.volatility_percentile_thresholds
        
        if atr_ratio < thresholds[0]:
            return VolatilityRegime.EXTREME_LOW
        elif atr_ratio < thresholds[1]:
            return VolatilityRegime.LOW
        elif atr_ratio < thresholds[2]:
            return VolatilityRegime.NORMAL
        elif atr_ratio < thresholds[3]:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME_HIGH
    
    def _calculate_volatility_percentile(self, current_atr: float, historical_bars: List[Dict[str, Any]]) -> float:
        """Calculate percentile of current volatility vs historical."""
        if len(historical_bars) < 50:
            return 0.5  # Default to middle
        
        # Calculate historical ATRs
        historical_atrs = []
        for i in range(self.cfg.atr_period, len(historical_bars)):
            atr_bars = historical_bars[i-self.cfg.atr_period:i]
            atr = self._calculate_atr(atr_bars)
            historical_atrs.append(atr)
        
        if not historical_atrs:
            return 0.5
        
        # Calculate percentile
        sorted_atrs = sorted(historical_atrs)
        rank = sum(1 for atr in sorted_atrs if atr <= current_atr)
        percentile = rank / len(sorted_atrs)
        
        return percentile
    
    def _analyze_trend(self, bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trend direction and strength."""
        if len(bars) < max(self.cfg.trend_periods):
            return {
                "direction": TrendDirection.SIDEWAYS,
                "strength": TrendStrength.WEAK,
                "duration": 0
            }
        
        closes = [bar["close"] for bar in bars]
        
        # Calculate trend for different periods
        trend_scores = []
        for period in self.cfg.trend_periods:
            if len(closes) >= period:
                recent_closes = closes[:period]
                older_closes = closes[period:2*period] if len(closes) >= 2*period else closes[period:]
                
                if len(older_closes) > 0:
                    recent_avg = sum(recent_closes) / len(recent_closes)
                    older_avg = sum(older_closes) / len(older_closes)
                    
                    # Simple trend score
                    trend_score = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                    trend_scores.append(trend_score)
        
        if not trend_scores:
            return {
                "direction": TrendDirection.SIDEWAYS,
                "strength": TrendStrength.WEAK,
                "duration": 0
            }
        
        # Overall trend direction
        avg_trend = sum(trend_scores) / len(trend_scores)
        
        if avg_trend > 0.02:
            direction = TrendDirection.STRONG_UP
        elif avg_trend > 0.01:
            direction = TrendDirection.UP
        elif avg_trend > -0.01:
            direction = TrendDirection.SIDEWAYS
        elif avg_trend > -0.02:
            direction = TrendDirection.DOWN
        else:
            direction = TrendDirection.STRONG_DOWN
        
        # Trend strength based on consistency
        strength_score = min(1.0, abs(avg_trend) * 50)  # Normalize to 0-1
        
        if strength_score > 0.8:
            strength = TrendStrength.VERY_STRONG
        elif strength_score > 0.6:
            strength = TrendStrength.STRONG
        elif strength_score > 0.4:
            strength = TrendStrength.MODERATE
        elif strength_score > 0.2:
            strength = TrendStrength.WEAK
        else:
            strength = TrendStrength.VERY_WEAK
        
        # Trend duration (simplified)
        duration = len([score for score in trend_scores if 
                       (score > 0.01 and avg_trend > 0) or 
                       (score < -0.01 and avg_trend < 0)])
        
        return {
            "direction": direction,
            "strength": strength,
            "duration": duration
        }
    
    def _analyze_range(self, bars: List[Dict[str, Any]], current_bar: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price range state."""
        if not bars:
            return {
                "state": RangeState.NORMAL_RANGE,
                "width_pct": 0.0,
                "position": 0.5
            }
        
        highs = [bar["high"] for bar in bars]
        lows = [bar["low"] for bar in bars]
        
        range_high = max(highs)
        range_low = min(lows)
        range_width = range_high - range_low
        current_price = current_bar["close"]
        
        # Range width as percentage of price
        width_pct = (range_width / current_price) * 100 if current_price > 0 else 0
        
        # Position in range (0 = at low, 1 = at high)
        position = (current_price - range_low) / range_width if range_width > 0 else 0.5
        
        # Classify range state
        if width_pct < 1.0:
            state = RangeState.TIGHT_RANGE
        elif width_pct > 5.0:
            state = RangeState.EXPANDING_RANGE
        elif position > 0.9 or position < 0.1:
            state = RangeState.BREAKOUT_MODE
        else:
            state = RangeState.NORMAL_RANGE
        
        return {
            "state": state,
            "width_pct": width_pct,
            "position": position
        }
    
    def _analyze_session(self, timestamp: datetime, volume: int) -> Dict[str, Any]:
        """Analyze session context."""
        from zoneinfo import ZoneInfo
        from config.settings import settings

        try:
            # Convert to market timezone (from config)
            local = timestamp.astimezone(ZoneInfo(settings.MARKET_TIMEZONE))
        except Exception:
            local = timestamp

        hour, minute = local.hour, local.minute
        time_minutes = hour * 60 + minute

        # Parse market hours from config
        open_parts = settings.MARKET_OPEN_TIME.split(":")
        close_parts = settings.MARKET_CLOSE_TIME.split(":")
        market_open = int(open_parts[0]) * 60 + int(open_parts[1])   # e.g. 9:15 = 555
        market_close = int(close_parts[0]) * 60 + int(close_parts[1]) # e.g. 15:30 = 930

        # Session type
        if time_minutes < market_open:
            session_type = "pre_market"
        elif time_minutes < market_close:
            session_type = "regular"
        else:
            session_type = "after_hours"

        # Session progress
        if session_type == "regular":
            session_length = market_close - market_open

            if time_minutes < market_open:
                progress = 0.0
                time_from_open = 0.0
                time_to_close = session_length
            elif time_minutes > market_close:
                progress = 1.0
                time_from_open = session_length
                time_to_close = 0.0
            else:
                time_from_open = time_minutes - market_open
                time_to_close = market_close - time_minutes
                progress = time_from_open / session_length
        else:
            progress = 0.5  # Default for non-regular sessions
            time_from_open = 0.0
            time_to_close = 0.0

        # Volume profile (simplified)
        if session_type == "regular":
            if progress < 0.25:  # First hour
                volume_profile = "high" if volume > 1500000 else "normal"
            elif progress > 0.75:  # Last hour
                volume_profile = "high" if volume > 1200000 else "normal"
            else:
                volume_profile = "normal"
        else:
            volume_profile = "low"

        return {
            "type": session_type,
            "progress": progress,
            "volume_profile": volume_profile,
            "time_from_open": time_from_open,
            "time_to_close": time_to_close
        }
    
    def _create_default_context(self, symbol: str, timestamp: datetime) -> MarketContext:
        """Create default context when data unavailable."""
        return MarketContext(
            timestamp=timestamp,
            symbol=symbol,
            current_price=0.0,
            open_price=0.0,
            high_price=0.0,
            low_price=0.0,
            volume=0,
            atr_14=0.0,
            atr_ratio=1.0,
            volatility_regime=VolatilityRegime.NORMAL,
            volatility_percentile=0.5,
            trend_direction=TrendDirection.SIDEWAYS,
            trend_strength=TrendStrength.WEAK,
            trend_duration_bars=0,
            range_state=RangeState.NORMAL_RANGE,
            range_width_pct=0.0,
            range_position=0.5,
            session_type="unknown",
            session_progress=0.5,
            volume_profile="normal"
        )
    
    def _cleanup_cache(self) -> None:
        """Remove old cache entries."""
        if len(self._context_cache) <= self.cfg.max_cache_size:
            return
        
        # Remove oldest entries
        sorted_entries = sorted(
            self._context_cache.items(),
            key=lambda x: x[1][1]  # Sort by cache timestamp
        )
        
        # Keep only the most recent entries
        keep_count = self.cfg.max_cache_size - 100
        self._context_cache = dict(sorted_entries[-keep_count:])
        
        logger.debug(f"Cleaned up market context cache, kept {keep_count} entries")
    
    # Event handlers
    def _on_trade_entered(self, event: Event) -> None:
        """Handle trade entered events - enrich with market context."""
        if event.trade_id and event.data.get("symbol"):
            # In production, would fetch trade and enrich asynchronously
            logger.debug(f"Trade entered event for market context enrichment: {event.trade_id}")
    
    def _on_trade_closed(self, event: Event) -> None:
        """Handle trade closed events - update context if needed."""
        if event.trade_id and event.data.get("symbol"):
            logger.debug(f"Trade closed event for context update: {event.trade_id}")


# Global instance
_market_enricher: Optional[MarketContextEnricher] = None

def get_market_enricher() -> MarketContextEnricher:
    """Get global market context enricher."""
    global _market_enricher
    if _market_enricher is None:
        _market_enricher = MarketContextEnricher()
    return _market_enricher
