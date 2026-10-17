# ingestion/broker_adapters/upstox.py
"""
Upstox API integration for historical data, real-time context,
and trade enrichment. Read-only — no execution.

Endpoints used:
- Historical candles (1min, 5min, 15min, daily)
- Market quotes (LTP, depth)
- Option chain (for IV context)
- Intraday candles
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Any, Optional
from enum import Enum

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)


class UpstoxInterval(str, Enum):
    ONE_MIN = "1minute"
    FIVE_MIN = "5minute"
    FIFTEEN_MIN = "15minute"
    THIRTY_MIN = "30minute"
    SIXTY_MIN = "60minute"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class UpstoxInstrumentType(str, Enum):
    EQUITY = "EQ"
    FUTURES = "FUT"
    OPTIONS_CE = "CE"
    OPTIONS_PE = "PE"
    INDEX = "INDEX"


class UpstoxClient:
    """
    Read-only Upstox API client.
    Handles authentication, rate limiting, and data normalization.

    Instrument keys: Upstox v2 uses ISIN-based keys (e.g. NSE_EQ|INE002A01018).
    Call resolve_instrument_key("RELIANCE") to get the correct key automatically.
    """

    BASE_URL = "https://api.upstox.com/v2"
    # Upstox instruments CSV — maps trading symbols to ISINs
    INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"

    def __init__(
        self,
        access_token: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        settings = get_settings()
        self._access_token = access_token or settings.upstox_access_token
        self._api_key = api_key or settings.upstox_api_key
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers=self._build_headers(),
            timeout=30.0,
        )
        self._rate_limit_remaining: int = 100
        self._cache: dict[str, Any] = {}
        # symbol → ISIN-based instrument key, e.g. "RELIANCE" → "NSE_EQ|INE002A01018"
        self._instrument_map: dict[str, str] = {}
        self._instruments_loaded: bool = False

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Api-Version": "2.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            # httpx drops the base_url path prefix when path starts with '/'.
            # Strip leading slash so httpx appends correctly to /v2.
            response = self._client.request(method, path.lstrip("/"), **kwargs)
            self._rate_limit_remaining = int(
                response.headers.get("x-ratelimit-remaining", 100)
            )

            if response.status_code == 429:
                logger.warning("Upstox rate limit hit — backing off")
                raise UpstoxRateLimitError("Rate limit exceeded")

            response.raise_for_status()
            data = response.json()

            if data.get("status") == "error":
                raise UpstoxAPIError(data.get("errors", [{}])[0].get("message", "Unknown error"))

            return data.get("data", data)

        except httpx.HTTPStatusError as e:
            logger.error("Upstox API error: %s %s → %s", method, path, e.response.status_code)
            raise UpstoxAPIError(f"HTTP {e.response.status_code}: {e.response.text}")

    # ══════════════════════════════════════════
    #  INSTRUMENT RESOLUTION (symbol → ISIN key)
    # ══════════════════════════════════════════

    def _load_instruments(self) -> None:
        """
        Download Upstox instruments CSV and build symbol → instrument_key map.
        Cached in memory for the lifetime of the client.

        CSV columns: instrument_key, exchange_token, tradingsymbol, name,
                     last_price, expiry, strike, tick_size, lot_size,
                     instrument_type, option_type, exchange
        """
        if self._instruments_loaded:
            return
        try:
            import gzip, csv, io
            resp = httpx.get(self.INSTRUMENTS_URL, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            content = gzip.decompress(resp.content).decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                exchange = row.get("exchange", "")       # e.g. "NSE_EQ", "BSE_EQ"
                tradingsymbol = row.get("tradingsymbol", "").upper()  # e.g. "RELIANCE-EQ"
                inst_type = row.get("instrument_type", "")  # "EQUITY", "FUT", etc.
                key = row.get("instrument_key", "")

                if not key or not tradingsymbol or inst_type != "EQUITY":
                    continue

                # Normalise: strip "-EQ" suffix Angel One appends
                plain = tradingsymbol.removesuffix("-EQ")

                if exchange == "NSE_EQ":
                    self._instrument_map[plain] = key
                    self._instrument_map[tradingsymbol] = key
                elif exchange == "BSE_EQ":
                    # Only use BSE as fallback if NSE not already mapped
                    self._instrument_map.setdefault(plain, key)
                    self._instrument_map.setdefault(tradingsymbol, key)

            self._instruments_loaded = True
            logger.info("Upstox instruments loaded: %d symbols mapped", len(self._instrument_map))
        except Exception as e:
            logger.warning("Could not load Upstox instruments list: %s — using fallback key format", e)
            self._instruments_loaded = True  # Don't retry on every call

    def resolve_instrument_key(self, symbol: str) -> str:
        """
        Resolve a plain symbol (e.g. 'RELIANCE') to an Upstox ISIN-based instrument key.
        Falls back to NSE_EQ|SYMBOL if not found (works for some index symbols).
        """
        upper = symbol.upper()

        # Index shortcuts — NSE and BSE spot indices
        _index_map = {
            # NSE indices
            "NIFTY":        "NSE_INDEX|Nifty 50",
            "NIFTY 50":     "NSE_INDEX|Nifty 50",
            "NIFTY50":      "NSE_INDEX|Nifty 50",
            "BANKNIFTY":    "NSE_INDEX|Nifty Bank",
            "NIFTY BANK":   "NSE_INDEX|Nifty Bank",
            "FINNIFTY":     "NSE_INDEX|Nifty Fin Service",
            "MIDCPNIFTY":   "NSE_INDEX|NIFTY MID SELECT",
            "INDIA VIX":    "NSE_INDEX|India VIX",
            "VIX":          "NSE_INDEX|India VIX",
            "NIFTY IT":     "NSE_INDEX|Nifty IT",
            "NIFTY AUTO":   "NSE_INDEX|Nifty Auto",
            "NIFTY PHARMA": "NSE_INDEX|Nifty Pharma",
            "NIFTY FMCG":   "NSE_INDEX|Nifty FMCG",
            "NIFTY METAL":  "NSE_INDEX|Nifty Metal",
            "NIFTY REALTY": "NSE_INDEX|Nifty Realty",
            # BSE indices
            "SENSEX":       "BSE_INDEX|SENSEX",
            "BANKEX":       "BSE_INDEX|BANKEX",
        }
        if upper in _index_map:
            return _index_map[upper]

        # Load instruments if not yet done
        self._load_instruments()

        if upper in self._instrument_map:
            return self._instrument_map[upper]

        # Fallback — may not work but worth trying
        logger.debug("Instrument key not found for %s — using fallback NSE_EQ|%s", symbol, upper)
        return f"NSE_EQ|{upper}"

    @staticmethod
    def build_instrument_key(
        exchange: str,
        symbol: str,
        instrument_type: str = "EQ",
        expiry: Optional[str] = None,
        strike: Optional[float] = None,
    ) -> str:
        """Build Upstox instrument key (legacy helper — prefer resolve_instrument_key)."""
        if instrument_type == "EQ":
            return f"{exchange}_EQ|{symbol}"
        elif instrument_type in ("FUT", "CE", "PE"):
            return f"{exchange}_FO|{symbol}"
        return f"{exchange}_EQ|{symbol}"

    # ══════════════════════════════════════════
    #  HISTORICAL CANDLES
    # ══════════════════════════════════════════

    def get_historical_candles(
        self,
        instrument_key: str,
        interval: UpstoxInterval,
        from_date: date,
        to_date: date,
    ) -> list[dict[str, Any]]:
        """
        Fetch historical OHLCV candles.

        Returns list of:
        {
            "timestamp": datetime,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": int,
            "oi": int (for derivatives)
        }
        """
        path = f"/historical-candle/{instrument_key}/{interval.value}/{to_date.isoformat()}/{from_date.isoformat()}"

        data = self._request("GET", path)
        candles_raw = data.get("candles", [])

        candles = []
        for c in candles_raw:
            candles.append({
                "timestamp": datetime.fromisoformat(c[0].replace("T", " ").split("+")[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": int(c[5]),
                "oi": int(c[6]) if len(c) > 6 else 0,
            })

        # Upstox returns newest first — reverse to chronological
        candles.reverse()

        logger.debug(
            "Fetched %d candles for %s (%s) %s→%s",
            len(candles), instrument_key, interval.value, from_date, to_date,
        )
        return candles

    # Upstox V3 historical candle base (custom unit/interval, long 1-min history).
    V3_HISTORICAL_BASE = "https://api.upstox.com/v3/historical-candle"

    def get_historical_candles_v3(
        self,
        instrument_key: str,
        unit: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> list[dict[str, Any]]:
        """
        Fetch historical OHLCV via the Upstox **V3** historical-candle API.

        v3 path: /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}
          unit:     "minutes" | "hours" | "days" | "weeks" | "months"
          interval: numeric string, e.g. "1" (1-minute when unit="minutes")

        v3 limits (caller must chunk): 1–15 min → 1 month/request; daily → 1 decade.
        1-minute history is available from Jan 2022; daily from 2000.

        Returns the same shape as get_historical_candles():
        [{timestamp, open, high, low, close, volume, oi}] in chronological order.
        """
        url = (
            f"{self.V3_HISTORICAL_BASE}/{instrument_key}/{unit}/{interval}/"
            f"{to_date.isoformat()}/{from_date.isoformat()}"
        )
        try:
            response = self._client.request("GET", url)
            self._rate_limit_remaining = int(
                response.headers.get("x-ratelimit-remaining", 100)
            )
            if response.status_code == 429:
                raise UpstoxRateLimitError("Rate limit exceeded")
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") == "error":
                raise UpstoxAPIError(payload.get("errors", [{}])[0].get("message", "Unknown error"))
            data = payload.get("data", payload)
        except UpstoxRateLimitError:
            raise
        except httpx.HTTPStatusError as e:
            logger.error("Upstox v3 API error: GET %s → %s", url, e.response.status_code)
            raise UpstoxAPIError(f"HTTP {e.response.status_code}: {e.response.text}")

        candles_raw = data.get("candles", [])
        candles = []
        for c in candles_raw:
            candles.append({
                "timestamp": datetime.fromisoformat(c[0].replace("T", " ").split("+")[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": int(c[5]),
                "oi": int(c[6]) if len(c) > 6 else 0,
            })
        candles.reverse()  # Upstox returns newest-first
        logger.debug(
            "Fetched %d v3 candles for %s (%s/%s) %s→%s",
            len(candles), instrument_key, unit, interval, from_date, to_date,
        )
        return candles

    # Upstox V3 intraday base — TODAY's candles. The v2 intraday endpoint below rejects "5minute"
    # (HTTP 400 UDAPI1076, verified 2026-09-17); v3 takes unit + numeric interval instead.
    V3_INTRADAY_BASE = "https://api.upstox.com/v3/historical-candle/intraday"

    def get_intraday_candles_v3(
        self,
        instrument_key: str,
        unit: str = "minutes",
        interval: str = "5",
    ) -> list[dict[str, Any]]:
        """TODAY's candles via Upstox V3 (the live decision path: official bars, 50 req/s, ~0.1 s/call).

        Returns [{timestamp, open, high, low, close, volume, oi}] in CHRONOLOGICAL order (the API returns newest
        first). Verified 2026-09-17 against Angel 5-min bars: OHLC identical on every intraday bar; only the 15:25
        closing-auction bar differs (Upstox prints the auction, Angel does not).
        """
        url = f"{self.V3_INTRADAY_BASE}/{instrument_key}/{unit}/{interval}"
        response = self._client.request("GET", url)
        response.raise_for_status()
        raw = (response.json().get("data") or {}).get("candles") or []
        out = []
        for c in raw:
            out.append({
                "timestamp": datetime.fromisoformat(str(c[0]).replace("T", " ").split("+")[0]),
                "open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]),
                "volume": int(c[5]), "oi": int(c[6]) if len(c) > 6 else 0,
            })
        out.reverse()
        return out

    def get_intraday_candles(
        self,
        instrument_key: str,
        interval: UpstoxInterval = UpstoxInterval.FIVE_MIN,
    ) -> list[dict[str, Any]]:
        """Fetch today's intraday candles."""
        path = f"/historical-candle/intraday/{instrument_key}/{interval.value}"
        data = self._request("GET", path)
        candles_raw = data.get("candles", [])

        candles = []
        for c in candles_raw:
            candles.append({
                "timestamp": datetime.fromisoformat(c[0].replace("T", " ").split("+")[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": int(c[5]),
                "oi": int(c[6]) if len(c) > 6 else 0,
            })

        candles.reverse()
        return candles

    # ══════════════════════════════════════════
    #  MARKET QUOTES
    # ══════════════════════════════════════════

    def get_market_quote(
        self, symbol_or_key: str
    ) -> dict[str, Any]:
        """Get full market quote including LTP, OHLC, volume."""
        instrument_key = self.resolve_instrument_key(symbol_or_key) \
            if "|" not in symbol_or_key else symbol_or_key

        data = self._request("GET", "/market-quote/quotes", params={"instrument_key": instrument_key})
        # Response key uses ':' not '|' — take first value if exact key not found
        quote_data = data.get(instrument_key) or (next(iter(data.values()), {}) if data else {})
        return {
            "ltp": quote_data.get("last_price"),
            "open": quote_data.get("ohlc", {}).get("open"),
            "high": quote_data.get("ohlc", {}).get("high"),
            "low": quote_data.get("ohlc", {}).get("low"),
            "close": quote_data.get("ohlc", {}).get("close"),
            "prev_close": quote_data.get("ohlc", {}).get("prev_close"),
            "volume": quote_data.get("volume"),
            "oi": quote_data.get("oi"),
            "bid": quote_data.get("depth", {}).get("buy", [{}])[0].get("price"),
            "ask": quote_data.get("depth", {}).get("sell", [{}])[0].get("price"),
            "timestamp": quote_data.get("last_trade_time"),
        }

    def get_ltp(self, symbol_or_key: str) -> Optional[float]:
        """Get last traded price. Accepts plain symbol ('RELIANCE') or instrument key."""
        instrument_key = self.resolve_instrument_key(symbol_or_key) \
            if "|" not in symbol_or_key else symbol_or_key

        data = self._request("GET", "/market-quote/ltp", params={"instrument_key": instrument_key})
        # Response key uses ':' separator — take first value if exact key not found
        quote = data.get(instrument_key) or (next(iter(data.values()), {}) if data else {})
        return quote.get("last_price")

    # ══════════════════════════════════════════
    #  OPTION CHAIN (for IV context)
    # ══════════════════════════════════════════

    def get_option_chain(
        self, instrument_key: str, expiry_date: str
    ) -> dict[str, Any]:
        """
        Get option chain for IV context.
        Returns IV, PCR, max pain estimates.
        """
        path = "/option/chain"
        params = {
            "instrument_key": instrument_key,
            "expiry_date": expiry_date,
        }
        data = self._request("GET", path, params=params)

        chain = data if isinstance(data, list) else []
        total_call_oi = 0
        total_put_oi = 0
        iv_values = []

        for strike_data in chain:
            call = strike_data.get("call_options", {}).get("market_data", {})
            put = strike_data.get("put_options", {}).get("market_data", {})

            total_call_oi += call.get("oi", 0)
            total_put_oi += put.get("oi", 0)

            call_iv = call.get("iv")
            put_iv = put.get("iv")
            if call_iv:
                iv_values.append(call_iv)
            if put_iv:
                iv_values.append(put_iv)

        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        avg_iv = sum(iv_values) / len(iv_values) if iv_values else 0

        return {
            "pcr": round(pcr, 3),
            "avg_iv": round(avg_iv, 2),
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "strike_count": len(chain),
        }


class UpstoxAPIError(Exception):
    pass


class UpstoxRateLimitError(UpstoxAPIError):
    pass