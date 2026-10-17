# avt/market_data.py
"""
Market Data Adapters for AVT
=============================

Concrete implementations of MarketDataInterface:

  1. BrokerMarketData — Live feed from broker WebSocket (Zerodha/Upstox/etc.)
  2. YFinanceMarketData — Free delayed data via yfinance (good for development)
  3. ReplayMarketData — Historical replay (already in engine.py, for testing)

Usage:
    from avt.market_data import YFinanceMarketData
    avt = init_avt(strategies=..., market_data=YFinanceMarketData())

    # Or with broker:
    from avt.market_data import BrokerMarketData
    avt = init_avt(strategies=..., market_data=BrokerMarketData(api_key="...", api_secret="..."))
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timezone, timedelta
from typing import Any, Dict, List, Optional

from avt.engine import MarketDataInterface

logger = logging.getLogger(__name__)


# ── NSE sector map (top FNO names) ───────────────────────────────

_SECTOR_MAP: Dict[str, str] = {
    "RELIANCE": "energy", "ONGC": "energy", "BPCL": "energy", "IOC": "energy", "GAIL": "energy",
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT", "LTIMINDTREE": "IT",
    "HDFCBANK": "banking", "ICICIBANK": "banking", "KOTAKBANK": "banking", "SBIN": "banking",
    "AXISBANK": "banking", "INDUSINDBK": "banking", "BANKBARODA": "banking", "PNB": "banking",
    "BAJFINANCE": "finance", "BAJAJFINSV": "finance", "HDFC": "finance", "SBILIFE": "finance",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG", "DABUR": "FMCG",
    "BHARTIARTL": "telecom", "TATAMOTORS": "auto", "M&M": "auto", "MARUTI": "auto", "HEROMOTOCO": "auto",
    "SUNPHARMA": "pharma", "DRREDDY": "pharma", "CIPLA": "pharma", "DIVISLAB": "pharma",
    "ADANIENT": "conglomerate", "ADANIPORTS": "infra", "TATASTEEL": "metals", "JSWSTEEL": "metals",
    "HINDALCO": "metals", "COALINDIA": "mining", "NTPC": "power", "POWERGRID": "power",
    "ASIANPAINT": "paints", "ULTRACEMCO": "cement", "GRASIM": "cement",
    "LT": "infra", "TITAN": "consumer", "BAJAJ-AUTO": "auto",
}


# ══════════════════════════════════════════════════════════════════
#  YFINANCE ADAPTER (development / free data)
# ══════════════════════════════════════════════════════════════════

class YFinanceMarketData(MarketDataInterface):
    """
    Market data via yfinance. Free, delayed (~15 min for NSE).
    Good for development and paper trading.

    Requires: pip install yfinance
    """

    def __init__(self, suffix: str = ".NS"):
        """
        Args:
            suffix: Yahoo Finance suffix for the exchange. ".NS" for NSE, ".BO" for BSE.
        """
        self._suffix = suffix
        self._price_cache: Dict[str, tuple] = {}  # symbol → (price, timestamp)
        self._cache_ttl = 60  # seconds

    async def get_current_price(self, symbol: str, side: str = "last") -> float:
        # Check cache
        cached = self._price_cache.get(symbol)
        if cached:
            price, ts = cached
            if (datetime.now(timezone.utc) - ts).total_seconds() < self._cache_ttl:
                return price

        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}{self._suffix}")
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                # Fallback to daily
                hist = ticker.history(period="5d")
            if hist.empty:
                return 0.0
            price = float(hist["Close"].iloc[-1])
            self._price_cache[symbol] = (price, datetime.now(timezone.utc))
            return price
        except Exception as e:
            logger.debug("yfinance price error for %s: %s", symbol, e)
            return 0.0

    async def get_candles(self, symbol: str, timeframe: str = "5m",
                          count: int = 100) -> List[Dict[str, Any]]:
        try:
            import yfinance as yf

            # Map timeframe to yfinance interval + period
            interval_map = {
                "1m": ("1m", "1d"),
                "5m": ("5m", "5d"),
                "15m": ("15m", "5d"),
                "1h": ("1h", "1mo"),
                "1d": ("1d", "6mo"),
            }
            interval, period = interval_map.get(timeframe, ("5m", "5d"))

            ticker = yf.Ticker(f"{symbol}{self._suffix}")
            hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                return []

            candles = []
            for idx, row in hist.tail(count).iterrows():
                candles.append({
                    "timestamp": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                })
            return candles

        except Exception as e:
            logger.debug("yfinance candles error for %s: %s", symbol, e)
            return []

    async def get_spread_pct(self, symbol: str) -> float:
        # yfinance doesn't provide bid/ask, estimate from typical NSE spreads
        return 0.05  # 0.05% typical for FNO stocks

    async def get_sector(self, symbol: str) -> str:
        return _SECTOR_MAP.get(symbol, "unknown")

    async def is_market_open(self) -> bool:
        return _is_nse_open()


# ══════════════════════════════════════════════════════════════════
#  BROKER ADAPTER (live trading — Zerodha Kite / Upstox / etc.)
# ══════════════════════════════════════════════════════════════════

class BrokerMarketData(MarketDataInterface):
    """
    Live market data from broker API (Zerodha Kite, Upstox, etc.).

    Expects a `broker_client` object with methods:
      - ltp(instruments: List[str]) → Dict[str, Dict]
      - historical_data(instrument_token, from_date, to_date, interval) → List[Dict]

    Example (Zerodha Kite):
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key="...")
        kite.set_access_token("...")
        market = BrokerMarketData(broker_client=kite, exchange="NSE")
    """

    def __init__(
        self,
        broker_client: Any = None,
        exchange: str = "NSE",
        instrument_map: Optional[Dict[str, int]] = None,
    ):
        self._client = broker_client
        self._exchange = exchange
        self._instrument_map = instrument_map or {}  # symbol → instrument_token
        self._price_cache: Dict[str, tuple] = {}
        self._cache_ttl = 5  # 5 seconds for live

    def set_client(self, client: Any) -> None:
        """Set or replace the broker client at runtime."""
        self._client = client

    def set_instrument_map(self, mapping: Dict[str, int]) -> None:
        """Set symbol → instrument_token mapping."""
        self._instrument_map = mapping

    async def get_current_price(self, symbol: str, side: str = "last") -> float:
        if not self._client:
            return 0.0

        # Check cache
        cached = self._price_cache.get(symbol)
        if cached:
            price, ts = cached
            if (datetime.now(timezone.utc) - ts).total_seconds() < self._cache_ttl:
                return price

        try:
            instrument = f"{self._exchange}:{symbol}"
            ltp_data = self._client.ltp([instrument])
            price = ltp_data.get(instrument, {}).get("last_price", 0.0)
            self._price_cache[symbol] = (float(price), datetime.now(timezone.utc))
            return float(price)
        except Exception as e:
            logger.debug("Broker price error for %s: %s", symbol, e)
            return 0.0

    async def get_candles(self, symbol: str, timeframe: str = "5m",
                          count: int = 100) -> List[Dict[str, Any]]:
        if not self._client:
            return []

        token = self._instrument_map.get(symbol)
        if not token:
            return []

        try:
            interval_map = {
                "1m": "minute",
                "5m": "5minute",
                "15m": "15minute",
                "1h": "60minute",
                "1d": "day",
            }
            interval = interval_map.get(timeframe, "5minute")

            # Compute date range
            to_date = datetime.now()
            if timeframe in ("1m", "5m", "15m"):
                from_date = to_date - timedelta(days=5)
            elif timeframe == "1h":
                from_date = to_date - timedelta(days=30)
            else:
                from_date = to_date - timedelta(days=180)

            raw = self._client.historical_data(
                token, from_date, to_date, interval,
            )

            candles = []
            for bar in raw[-count:]:
                candles.append({
                    "timestamp": bar["date"].isoformat() if hasattr(bar["date"], "isoformat") else str(bar["date"]),
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": int(bar["volume"]),
                })
            return candles

        except Exception as e:
            logger.debug("Broker candles error for %s: %s", symbol, e)
            return []

    async def get_spread_pct(self, symbol: str) -> float:
        if not self._client:
            return 0.05

        try:
            instrument = f"{self._exchange}:{symbol}"
            quote = self._client.quote([instrument])
            depth = quote.get(instrument, {}).get("depth", {})
            best_bid = depth.get("buy", [{}])[0].get("price", 0)
            best_ask = depth.get("sell", [{}])[0].get("price", 0)
            if best_bid > 0 and best_ask > 0:
                mid = (best_bid + best_ask) / 2
                return ((best_ask - best_bid) / mid) * 100
        except Exception:
            pass
        return 0.05

    async def get_sector(self, symbol: str) -> str:
        return _SECTOR_MAP.get(symbol, "unknown")

    async def is_market_open(self) -> bool:
        return _is_nse_open()


# ══════════════════════════════════════════════════════════════════
#  UPSTOX ADAPTER (live data via Upstox API)
# ══════════════════════════════════════════════════════════════════

class UpstoxMarketData(MarketDataInterface):
    """
    Live market data from Upstox via MarketDataService.

    Uses Upstox for real-time LTP/quotes and Angel One (via the same
    MarketDataService) for historical candles.  All methods use
    ``run_in_executor`` because the underlying HTTP calls are synchronous.

    Usage:
        from ingestion.market_data_service import get_market_data_service
        from avt.market_data import UpstoxMarketData
        svc = get_market_data_service()
        avt = init_avt(strategies=..., market_data=UpstoxMarketData(svc))
    """

    def __init__(self, market_data_service):
        from ingestion.market_data_service import MarketDataService
        self._svc: MarketDataService = market_data_service
        self._price_cache: Dict[str, tuple] = {}
        self._cache_ttl = 5  # 5 seconds for live data

    async def get_current_price(self, symbol: str, side: str = "last") -> float:
        # Check cache
        cached = self._price_cache.get(symbol)
        if cached:
            price, ts = cached
            if (datetime.now(timezone.utc) - ts).total_seconds() < self._cache_ttl:
                return price

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            ltp = await loop.run_in_executor(None, self._svc.get_ltp, symbol)
            if ltp and ltp > 0:
                self._price_cache[symbol] = (ltp, datetime.now(timezone.utc))
                return ltp
        except Exception as e:
            logger.debug("Upstox price error for %s: %s", symbol, e)
        return 0.0

    async def get_candles(self, symbol: str, timeframe: str = "5m",
                          count: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch candles via MarketDataService → CandleStore (the local OHLCV lake,
        Upstox v3 backed). Passes the requested timeframe through (previously it
        fell back to the scanner default, so AVT silently got 15m for a 5m ask).
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            candles = await loop.run_in_executor(
                None, self._svc.fetch_candles_for_scanner, symbol, count, timeframe,
            )
            return candles
        except Exception as e:
            logger.debug("Candles error for %s: %s", symbol, e)
            return []

    async def get_spread_pct(self, symbol: str) -> float:
        """Spread from Upstox live quote (bid/ask depth)."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            quote = await loop.run_in_executor(None, self._svc.get_market_quote, symbol)
            bid = quote.get("bid")
            ask = quote.get("ask")
            if bid and ask and bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                return ((ask - bid) / mid) * 100
        except Exception as e:
            logger.debug("Upstox spread error for %s: %s", symbol, e)
        return 0.05  # Default NSE FNO spread

    async def get_sector(self, symbol: str) -> str:
        return _SECTOR_MAP.get(symbol, "unknown")

    async def is_market_open(self) -> bool:
        return _is_nse_open()


# Keep AngelOneMarketData as alias for backward compatibility
AngelOneMarketData = UpstoxMarketData


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _is_nse_open() -> bool:
    """Check if NSE is currently in trading hours (9:15 AM - 3:30 PM IST)."""
    now_utc = datetime.now(timezone.utc)
    # IST = UTC + 5:30
    ist_offset = timedelta(hours=5, minutes=30)
    now_ist = now_utc + ist_offset

    # Weekday check (Mon=0, Sun=6)
    if now_ist.weekday() >= 5:
        return False

    market_open = time(9, 15)
    market_close = time(15, 30)
    current_time = now_ist.time()

    return market_open <= current_time <= market_close
