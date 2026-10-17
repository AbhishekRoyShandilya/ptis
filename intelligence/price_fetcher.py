"""
Price Data Fetcher

Provides OHLCV data to the intelligence layer from configured sources.
Supports multiple backends with caching.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine, Optional

import pandas as pd

logger = logging.getLogger(__name__)


async def fetch_from_alpaca(
    symbol: str, timeframe: str, bars: int = 200
) -> pd.DataFrame:
    """Fetch from Alpaca Markets API."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        import os

        tf_map = {
            "1m": TimeFrame(1, TimeFrameUnit.Minute),
            "5m": TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "1h": TimeFrame(1, TimeFrameUnit.Hour),
            "D": TimeFrame(1, TimeFrameUnit.Day),
        }

        client = StockHistoricalDataClient(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
        )

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf_map.get(timeframe, tf_map["5m"]),
            limit=bars,
        )

        bars_data = client.get_stock_bars(request)
        df = bars_data.df
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Missing columns: {required - set(df.columns)}")

        return df

    except ImportError:
        raise ImportError("alpaca-py package required: pip install alpaca-py")


async def fetch_from_polygon(
    symbol: str, timeframe: str, bars: int = 200
) -> pd.DataFrame:
    """Fetch from Polygon.io API."""
    try:
        import os
        import httpx
        from datetime import datetime, timedelta

        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            raise ValueError("POLYGON_API_KEY not set")

        multiplier_map = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "1h": (1, "hour"),
            "D": (1, "day"),
        }

        mult, span = multiplier_map.get(timeframe, (5, "minute"))
        end = datetime.utcnow()
        start = end - timedelta(days=5 if span == "minute" else 365)

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
            f"{mult}/{span}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
            f"?adjusted=true&sort=asc&limit={bars}&apiKey={api_key}"
        )

        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            raise ValueError(f"No data returned for {symbol}")

        df = pd.DataFrame(results)
        df = df.rename(columns={
            "o": "open", "h": "high", "l": "low",
            "c": "close", "v": "volume", "t": "timestamp",
        })

        return df[["open", "high", "low", "close", "volume"]]

    except ImportError:
        raise ImportError("httpx package required: pip install httpx")


async def fetch_from_yahoo(
    symbol: str, timeframe: str, bars: int = 200
) -> pd.DataFrame:
    """Fetch from Yahoo Finance (free, delayed)."""
    try:
        import yfinance as yf

        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "D": "1d",
            "1d": "1d",
        }

        period_map = {
            "1m": "1d",
            "5m": "5d",
            "15m": "5d",
            "30m": "5d",
            "1h": "1mo",
            "D": "1y",
            "1d": "1y",
        }

        interval = interval_map.get(timeframe, "5m")
        period = period_map.get(timeframe, "5d")

        # NSE Indian stocks need ".NS" suffix on Yahoo Finance
        # BSE stocks need ".BO" — default to NSE
        yahoo_symbol = symbol
        if "." not in symbol and not symbol.startswith("^"):
            yahoo_symbol = f"{symbol}.NS"

        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period=period, interval=interval)

        # If .NS returns nothing, try .BO as fallback
        if df.empty:
            ticker = yf.Ticker(f"{symbol}.BO")
            df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        df.columns = [c.lower() for c in df.columns]
        df = df.reset_index()

        return df[["open", "high", "low", "close", "volume"]].tail(bars)

    except ImportError:
        raise ImportError("yfinance package required: pip install yfinance")


def get_price_fetcher(
    settings: Any,
) -> Callable[..., Coroutine[Any, Any, pd.DataFrame]]:
    """
    Get the configured price data fetcher.

    Returns an async function: (symbol, timeframe) -> DataFrame
    """
    provider = getattr(settings, "PRICE_DATA_PROVIDER", "yahoo").lower()

    fetchers = {
        "alpaca": fetch_from_alpaca,
        "polygon": fetch_from_polygon,
        "yahoo": fetch_from_yahoo,
    }

    fetcher = fetchers.get(provider)
    if fetcher is None:
        logger.warning(
            f"Unknown price provider '{provider}', falling back to yahoo"
        )
        fetcher = fetch_from_yahoo

    logger.info(f"Price data provider: {provider}")
    return fetcher