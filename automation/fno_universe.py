# automation/fno_universe.py
"""
FNO Stock Universe Auto-Fetcher
================================
Automatically fetches and maintains the list of NSE F&O (Futures & Options)
eligible stocks.

The FNO universe is the primary watchlist for autonomous scanning because:
- These stocks have high liquidity (guaranteed by derivative activity)
- They represent the top ~200 NSE stocks by market cap
- They have well-defined lot sizes for position sizing

Data sources (in priority order):
1. NSE's official CSV/JSON via public endpoints
2. Zerodha's instruments dump (kiteconnect API)
3. Hardcoded fallback list of NIFTY 50 + NIFTY Next 50

The FNO universe is refreshed daily (or on-demand) and cached locally.
Changes (additions/removals) are logged for the system journal.

Usage:
    fetcher = FNOUniverseFetcher()
    stocks = fetcher.get_fno_stocks()           # cached list
    stocks = await fetcher.refresh()            # re-fetch from source
    lot_info = fetcher.get_lot_size("RELIANCE") # lot size for F&O
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Cache file path
_CACHE_DIR = Path(os.getenv("PTIS_DATA_DIR", "./data"))
_CACHE_FILE = _CACHE_DIR / "fno_universe.json"

# NSE public endpoint (scraped, may require browser-like headers)
_NSE_FNO_URL = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
_NSE_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
}


# ── NIFTY 50 Fallback List ──
# Used when NSE API is unavailable
_NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BHARTIARTL", "BPCL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
    "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN",
    "ULTRACEMCO", "UPL", "WIPRO",
]

_NIFTY_NEXT_50 = [
    "ABB", "ADANIGREEN", "AMBUJACEM", "ATGL", "BANKBARODA",
    "BEL", "BHEL", "BOSCHLTD", "CANBK", "CHOLAFIN",
    "COLPAL", "CONCOR", "DLF", "GAIL", "GODREJCP",
    "HAL", "HAVELLS", "ICICIPRULI", "IDEA", "IDFCFIRSTB",
    "INDIGO", "IOC", "IRCTC", "JINDALSTEL", "LICI",
    "LUPIN", "MCDOWELL-N", "MUTHOOTFIN", "NAUKRI", "NHPC",
    "NMDC", "OFSS", "PAGEIND", "PEL", "PFC",
    "PIDILITIND", "PNB", "POLYCAB", "RECLTD", "SBICARD",
    "SHREECEM", "SIEMENS", "SRF", "TATACONSUM", "TATAPOWER",
    "TORNTPHARM", "TRENT", "VEDL", "YESBANK", "ZOMATO",
]


class FNOStock:
    """Represents a single FNO-eligible stock."""

    def __init__(
        self,
        symbol: str,
        name: str = "",
        lot_size: int = 1,
        exchange: str = "NSE",
        series: str = "EQ",
        isin: str = "",
        sector: str = "",
        market_cap: Optional[float] = None,
    ):
        self.symbol = symbol
        self.name = name
        self.lot_size = lot_size
        self.exchange = exchange
        self.series = series
        self.isin = isin
        self.sector = sector
        self.market_cap = market_cap

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "lot_size": self.lot_size,
            "exchange": self.exchange,
            "series": self.series,
            "isin": self.isin,
            "sector": self.sector,
            "market_cap": self.market_cap,
        }


class FNOUniverseFetcher:
    """
    Fetches and caches the FNO-eligible stock universe.

    Provides:
        - get_fno_stocks()  — returns cached list (fast)
        - refresh()         — re-fetches from source
        - get_symbols()     — just the symbol strings
        - get_lot_size()    — lot size for a specific stock
    """

    def __init__(self):
        self._stocks: List[FNOStock] = []
        self._lot_sizes: Dict[str, int] = {}
        self._last_refresh: Optional[datetime] = None
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached FNO universe from disk."""
        if _CACHE_FILE.exists():
            try:
                data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
                self._stocks = [FNOStock(**s) for s in data.get("stocks", [])]
                self._lot_sizes = {s.symbol: s.lot_size for s in self._stocks}
                refresh_str = data.get("last_refresh")
                if refresh_str:
                    self._last_refresh = datetime.fromisoformat(refresh_str)
                logger.info(
                    "Loaded %d FNO stocks from cache (last refresh: %s)",
                    len(self._stocks),
                    self._last_refresh,
                )
            except Exception as e:
                logger.warning("Failed to load FNO cache: %s", e)
                self._load_fallback()
        else:
            self._load_fallback()

    def _load_fallback(self) -> None:
        """Use hardcoded NIFTY 50 + Next 50 as fallback."""
        all_symbols = _NIFTY_50 + _NIFTY_NEXT_50
        self._stocks = [FNOStock(symbol=s) for s in all_symbols]
        self._lot_sizes = {s: 1 for s in all_symbols}
        logger.info("Using fallback FNO universe: %d symbols", len(self._stocks))

    def _save_cache(self) -> None:
        """Persist current FNO universe to disk."""
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "stocks": [s.to_dict() for s in self._stocks],
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "count": len(self._stocks),
        }
        _CACHE_FILE.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("FNO universe cached: %d stocks", len(self._stocks))

    async def refresh(self) -> List[FNOStock]:
        """
        Re-fetch the FNO universe from NSE.
        Falls back to Kite instruments or hardcoded list on failure.
        """
        stocks = await self._fetch_from_nse()

        if not stocks:
            stocks = self._fetch_from_kite()

        if not stocks:
            logger.warning("All FNO fetch sources failed — using fallback")
            self._load_fallback()
            return self._stocks

        self._stocks = stocks
        self._lot_sizes = {s.symbol: s.lot_size for s in stocks}
        self._last_refresh = datetime.now(timezone.utc)
        self._save_cache()

        logger.info(
            "FNO universe refreshed: %d stocks from live source",
            len(stocks),
        )
        return stocks

    async def _fetch_from_nse(self) -> List[FNOStock]:
        """Fetch FNO stocks from NSE public API."""
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                # NSE requires a session cookie — hit homepage first
                await client.get("https://www.nseindia.com/", headers=_NSE_HEADERS)

                resp = await client.get(_NSE_FNO_URL, headers=_NSE_HEADERS)
                resp.raise_for_status()
                data = resp.json()

            records = data.get("data", [])
            stocks = []
            for rec in records:
                symbol = rec.get("symbol", "")
                if not symbol or symbol == "NIFTY 50":
                    continue
                stocks.append(FNOStock(
                    symbol=symbol,
                    name=rec.get("meta", {}).get("companyName", ""),
                    lot_size=1,  # NSE API doesn't provide lot sizes
                    exchange="NSE",
                    series=rec.get("series", "EQ"),
                    isin=rec.get("meta", {}).get("isin", ""),
                    sector=rec.get("meta", {}).get("industry", ""),
                ))

            logger.info("Fetched %d FNO stocks from NSE", len(stocks))
            return stocks

        except Exception as e:
            logger.warning("NSE FNO fetch failed: %s", e)
            return []

    def _fetch_from_kite(self) -> List[FNOStock]:
        """Fetch FNO instruments from Kite Connect instruments dump."""
        try:
            from config.settings import settings
            if not settings.KITE_API_KEY:
                return []

            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=settings.KITE_API_KEY)
            if settings.KITE_ACCESS_TOKEN:
                kite.set_access_token(settings.KITE_ACCESS_TOKEN)

            instruments = kite.instruments("NFO")
            # Get unique underlyings from NFO instruments
            symbols = set()
            lot_sizes: Dict[str, int] = {}

            for inst in instruments:
                underlying = inst.get("name", "")
                if underlying and inst.get("instrument_type") == "FUT":
                    symbols.add(underlying)
                    lot_sizes[underlying] = inst.get("lot_size", 1)

            stocks = [
                FNOStock(
                    symbol=sym,
                    lot_size=lot_sizes.get(sym, 1),
                    exchange="NSE",
                )
                for sym in sorted(symbols)
            ]

            logger.info("Fetched %d FNO stocks from Kite instruments", len(stocks))
            return stocks

        except Exception as e:
            logger.warning("Kite FNO fetch failed: %s", e)
            return []

    # ── Public API ──

    def get_fno_stocks(self) -> List[FNOStock]:
        """Return cached FNO stock list."""
        return list(self._stocks)

    def get_symbols(self) -> List[str]:
        """Return just the symbol strings."""
        return [s.symbol for s in self._stocks]

    def get_lot_size(self, symbol: str) -> int:
        """Get FNO lot size for a symbol (default 1 if unknown)."""
        return self._lot_sizes.get(symbol, 1)

    def get_nifty50(self) -> List[str]:
        """Return NIFTY 50 constituents."""
        return list(_NIFTY_50)

    def get_nifty_next50(self) -> List[str]:
        """Return NIFTY Next 50 constituents."""
        return list(_NIFTY_NEXT_50)

    def get_stock_info(self, symbol: str) -> Optional[FNOStock]:
        """Get full info for a specific symbol."""
        for s in self._stocks:
            if s.symbol == symbol:
                return s
        return None

    def is_fno_eligible(self, symbol: str) -> bool:
        """Check if a symbol is in the FNO universe."""
        return any(s.symbol == symbol for s in self._stocks)

    @property
    def last_refresh(self) -> Optional[datetime]:
        return self._last_refresh

    @property
    def count(self) -> int:
        return len(self._stocks)
