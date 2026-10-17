# ingestion/broker_adapters/angel_one.py
"""
Angel One SmartAPI adapter — read-only broker integration for NSE/BSE.

Implements BrokerAdapter interface for:
- Historical OHLCV candles (getCandleData)
- Positions (position)
- Orders (orderBook)
- Account info (rmsLimit)
- Market quotes (ltpData, getMarketData)

Requires: smartapi-python>=1.4.0, pyotp
Auth: SmartAPI uses api_key + client_id + password/PIN + TOTP.

Rate limits:
- 10 requests/second general
- Historical data: specific limits per interval
  (1min → last 30 days, 1day → last 2000 days)
"""

from __future__ import annotations

import logging
import time as time_mod
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from config.settings import settings
from ingestion.broker_api import (
    BrokerAdapter,
    BrokerAPIError,
    BrokerAuthError,
    BrokerRateLimitError,
)

logger = logging.getLogger(__name__)


class AngelInterval:
    """Angel One candle interval constants."""
    ONE_MINUTE = "ONE_MINUTE"
    THREE_MINUTE = "THREE_MINUTE"
    FIVE_MINUTE = "FIVE_MINUTE"
    TEN_MINUTE = "TEN_MINUTE"
    FIFTEEN_MINUTE = "FIFTEEN_MINUTE"
    THIRTY_MINUTE = "THIRTY_MINUTE"
    ONE_HOUR = "ONE_HOUR"
    ONE_DAY = "ONE_DAY"

    _MAP = {
        "1m": "ONE_MINUTE", "1min": "ONE_MINUTE",
        "3m": "THREE_MINUTE", "3min": "THREE_MINUTE",
        "5m": "FIVE_MINUTE", "5min": "FIVE_MINUTE",
        "10m": "TEN_MINUTE", "10min": "TEN_MINUTE",
        "15m": "FIFTEEN_MINUTE", "15min": "FIFTEEN_MINUTE",
        "30m": "THIRTY_MINUTE", "30min": "THIRTY_MINUTE",
        "60m": "ONE_HOUR", "1h": "ONE_HOUR",
        "1d": "ONE_DAY", "day": "ONE_DAY",
    }

    @classmethod
    def resolve(cls, interval: str) -> str:
        """Map a generic interval string to Angel's format."""
        return cls._MAP.get(interval.lower(), interval)

    @classmethod
    def max_days(cls, interval: str) -> int:
        """Max days of data per interval (Angel One limits)."""
        limits = {
            "ONE_MINUTE": 30,
            "THREE_MINUTE": 90,
            "FIVE_MINUTE": 90,
            "TEN_MINUTE": 90,
            "FIFTEEN_MINUTE": 180,
            "THIRTY_MINUTE": 180,
            "ONE_HOUR": 365,
            "ONE_DAY": 2000,
        }
        return limits.get(interval, 90)


class AngelOneClient(BrokerAdapter):
    """
    Read-only Angel One SmartAPI client.
    Wraps the smartapi-python SDK for historical data, positions, orders, and quotes.
    """

    name: str = "angel_one"

    def __init__(
        self,
        api_key: Optional[str] = None,
        client_id: Optional[str] = None,
        password: Optional[str] = None,
        totp_secret: Optional[str] = None,
    ):
        self._api_key = api_key or settings.ANGEL_API_KEY
        self._client_id = client_id or settings.ANGEL_CLIENT_ID
        self._password = password or settings.ANGEL_PASSWORD
        self._totp_secret = totp_secret or settings.ANGEL_TOTP_SECRET

        if not self._api_key:
            raise BrokerAuthError("ANGEL_API_KEY is required for Angel One adapter")
        if not self._client_id:
            raise BrokerAuthError("ANGEL_CLIENT_ID is required for Angel One adapter")

        self._smart_api = None
        self._auth_token = None
        self._feed_token = None
        self._last_request_time: float = 0.0
        self._min_request_interval: float = 0.1  # 10 req/sec
        self._token_map: Dict[str, str] = {}  # symbol → token for instruments

        self._init_client()

    def _init_client(self) -> None:
        """Initialize the SmartAPI client object only — login is deferred to first use."""
        try:
            from SmartApi import SmartConnect
        except ImportError:
            raise ImportError(
                "smartapi-python package not installed. "
                "Install with: pip install smartapi-python>=1.4.0"
            )

        self._smart_api = SmartConnect(api_key=self._api_key)
        # Login deferred — called lazily on first data request via _ensure_session()

    def _ensure_session(self) -> None:
        """Login if not already authenticated. Called before every API request."""
        if self._auth_token:
            return
        if self._password:
            self._login()

    def _login(self) -> Dict[str, Any]:
        """
        Authenticate with Angel One SmartAPI.
        Requires client_id, password, and optionally TOTP.
        """
        try:
            totp = None
            if self._totp_secret:
                try:
                    import pyotp
                    totp = pyotp.TOTP(self._totp_secret).now()
                except ImportError:
                    logger.warning("pyotp not installed — TOTP will not be generated")
                except Exception as e:
                    logger.warning("TOTP generation failed: %s", e)

            data = self._smart_api.generateSession(
                clientCode=self._client_id,
                password=self._password,
                totp=totp,
            )

            if data.get("status"):
                self._auth_token = data["data"].get("jwtToken")
                self._feed_token = data["data"].get("feedToken")
                logger.info("Angel One login successful for client %s", self._client_id)
                return data["data"]
            else:
                raise BrokerAuthError(f"Angel One login failed: {data.get('message', 'Unknown error')}")

        except BrokerAuthError:
            raise
        except Exception as e:
            raise BrokerAuthError(f"Angel One authentication error: {e}")

    def _throttle(self) -> None:
        """Rate limiter — also ensures session is active before every request."""
        self._ensure_session()
        elapsed = time_mod.monotonic() - self._last_request_time
        if elapsed < self._min_request_interval:
            time_mod.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time_mod.monotonic()

    # ══════════════════════════════════════════
    #  INSTRUMENT TOKEN RESOLUTION
    # ══════════════════════════════════════════

    # Hardcoded Angel One instrument tokens for indices.
    # These bypass searchScrip() which often returns the nearest derivative (futures/options)
    # instead of the spot index when called with bare names like "NIFTY" or "BANKNIFTY".
    # Token values sourced from Angel One instrument master (stable identifiers).
    # NOTE: Angel's index tokens carry a "999" prefix (NSE indices: 99926xxx; BSE: 99919xxx). The bare
    # "26000"/"1" values here previously returned ZERO candles (getCandleData → empty) → indices went
    # stale/absent in the lake → the eye couldn't read index-option trades (e.g. SENSEX PE). VERIFIED
    # 2026-06-30: 99926000/99926009/99919000 each return 150 candles where the old tokens returned 0.
    _INDEX_TOKENS: dict[str, tuple[str, str]] = {
        # symbol → (exchange, token)
        "NIFTY":       ("NSE", "99926000"),
        "NIFTY 50":    ("NSE", "99926000"),
        "NIFTY50":     ("NSE", "99926000"),
        "BANKNIFTY":   ("NSE", "99926009"),
        "NIFTY BANK":  ("NSE", "99926009"),
        "FINNIFTY":    ("NSE", "99926037"),
        "MIDCPNIFTY":  ("NSE", "99926074"),
        "SENSEX":      ("BSE", "99919000"),
        "BANKEX":      ("BSE", "99919012"),
        "INDIA VIX":   ("NSE", "99926017"),
        "VIX":         ("NSE", "99926017"),
    }

    def get_symbol_token(
        self,
        symbol: str,
        exchange: str = "NSE",
    ) -> str:
        """
        Resolve a trading symbol to Angel One's symbol token.

        For known indices, returns a hardcoded token immediately — bypassing
        searchScrip() which frequently returns the nearest derivative contract
        instead of the spot index for bare names like NIFTY or BANKNIFTY.
        """
        upper = symbol.upper()

        # Fast path: known index tokens (avoids searchScrip returning futures)
        if upper in self._INDEX_TOKENS:
            idx_exchange, idx_token = self._INDEX_TOKENS[upper]
            self._token_map[f"{idx_exchange}:{upper}"] = idx_token
            return idx_token

        cache_key = f"{exchange}:{upper}"
        if cache_key in self._token_map:
            return self._token_map[cache_key]

        # For equity, Angel One uses a searchScrip API
        try:
            self._throttle()
            result = self._smart_api.searchScrip(exchange, upper)
            if result and result.get("status") and result.get("data"):
                items = result["data"]

                # Prefer exact match first
                for item in items:
                    if item["tradingsymbol"].upper() == upper:
                        self._token_map[cache_key] = item["symboltoken"]
                        return item["symboltoken"]

                # For equities Angel One appends "-EQ" — match that variant
                eq_symbol = f"{upper}-EQ"
                for item in items:
                    if item["tradingsymbol"].upper() == eq_symbol:
                        self._token_map[cache_key] = item["symboltoken"]
                        return item["symboltoken"]

                # Last resort: take the first result
                first = items[0]
                logger.debug(
                    "Exact match not found for %s — using first result: %s (%s)",
                    cache_key, first["tradingsymbol"], first["symboltoken"],
                )
                self._token_map[cache_key] = first["symboltoken"]
                return first["symboltoken"]

            raise BrokerAPIError(f"Symbol not found on Angel One: {cache_key}")
        except BrokerAPIError:
            raise
        except Exception as e:
            raise BrokerAPIError(f"Failed to resolve symbol {cache_key}: {e}")

    # ══════════════════════════════════════════
    #  HISTORICAL CANDLES
    # ══════════════════════════════════════════

    def get_historical_candles(
        self,
        symbol: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical OHLCV candles from Angel One SmartAPI.

        Args:
            symbol: Trading symbol (e.g., "RELIANCE", "NIFTY")
            interval: Candle interval (e.g., "5m", "15m", "1d")
            from_date: Start date
            to_date: End date

        Returns:
            List of {timestamp, open, high, low, close, volume} dicts
        """
        exchange = self._detect_exchange(symbol)
        token = self.get_symbol_token(symbol, exchange)
        angel_interval = AngelInterval.resolve(interval)

        from_str = datetime.combine(from_date, datetime.min.time()).strftime("%Y-%m-%d %H:%M")
        to_str = datetime.combine(to_date, datetime.max.time().replace(microsecond=0)).strftime("%Y-%m-%d %H:%M")

        historic_params = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": angel_interval,
            "fromdate": from_str,
            "todate": to_str,
        }

        self._throttle()
        try:
            result = self._smart_api.getCandleData(historic_params)
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "too many" in err_str:
                raise BrokerRateLimitError(f"Angel One rate limit: {e}")
            if "auth" in err_str or "token" in err_str or "session" in err_str:
                raise BrokerAuthError(f"Angel One auth error: {e}")
            raise BrokerAPIError(f"Angel One historical data error: {e}")

        if not result or not result.get("status") or not result.get("data"):
            logger.warning("No candle data returned for %s:%s", exchange, symbol)
            return []

        candles = []
        for c in result["data"]:
            # Angel returns: [timestamp, open, high, low, close, volume]
            candles.append({
                "timestamp": datetime.strptime(c[0], "%Y-%m-%dT%H:%M:%S%z")
                             if isinstance(c[0], str) else c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": int(c[5]),
            })

        logger.debug(
            "Fetched %d candles for %s:%s (%s) %s→%s",
            len(candles), exchange, symbol, angel_interval, from_date, to_date,
        )
        return candles

    # ══════════════════════════════════════════
    #  POSITIONS
    # ══════════════════════════════════════════

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch current open positions from Angel One."""
        self._throttle()
        try:
            result = self._smart_api.position()
        except Exception as e:
            raise BrokerAPIError(f"Failed to fetch positions: {e}")

        if not result or not result.get("status") or not result.get("data"):
            return []

        positions = []
        for pos in result["data"]:
            net_qty = int(pos.get("netqty", 0))
            if net_qty == 0:
                continue

            positions.append({
                "symbol": pos.get("tradingsymbol", ""),
                "exchange": pos.get("exchange", "NSE"),
                "quantity": net_qty,
                "avg_price": float(pos.get("averageprice", 0)),
                "pnl": float(pos.get("pnl", 0)),
                "side": "long" if net_qty > 0 else "short",
                "product": pos.get("producttype", ""),
                "ltp": float(pos.get("ltp", 0)),
            })

        return positions

    # ══════════════════════════════════════════
    #  ORDERS
    # ══════════════════════════════════════════

    def get_orders(
        self,
        from_date: Optional[date] = None,
        status: str = "filled",
    ) -> List[Dict[str, Any]]:
        """Fetch order history from Angel One."""
        self._throttle()
        try:
            result = self._smart_api.orderBook()
        except Exception as e:
            raise BrokerAPIError(f"Failed to fetch orders: {e}")

        if not result or not result.get("status") or not result.get("data"):
            return []

        # Angel One status mapping
        status_map = {
            "filled": "complete",
            "open": "open",
            "cancelled": "cancelled",
            "rejected": "rejected",
        }
        filter_status = status_map.get(status, status.lower())

        orders = []
        for order in result["data"]:
            order_status = order.get("orderstatus", "").lower()
            if filter_status and order_status != filter_status:
                continue

            order_time_str = order.get("updatetime") or order.get("ordertime")
            order_time = None
            if order_time_str:
                try:
                    order_time = datetime.strptime(order_time_str, "%d-%b-%Y %H:%M:%S")
                except (ValueError, TypeError):
                    order_time = None

            if from_date and order_time and order_time.date() < from_date:
                continue

            orders.append({
                "symbol": order.get("tradingsymbol", ""),
                "side": order.get("transactiontype", "").lower(),  # BUY/SELL
                "qty": int(order.get("filledshares", order.get("quantity", 0))),
                "price": float(order.get("averageprice", 0)),
                "filled_at": order_time,
                "status": order_status,
                "order_id": order.get("orderid"),
                "exchange": order.get("exchange", "NSE"),
                "product": order.get("producttype", ""),
            })

        return orders

    # ══════════════════════════════════════════
    #  ACCOUNT INFO
    # ══════════════════════════════════════════

    def get_account_info(self) -> Dict[str, Any]:
        """Fetch account margins/RMS limits from Angel One."""
        self._throttle()
        try:
            result = self._smart_api.rmsLimit()
        except Exception as e:
            raise BrokerAPIError(f"Failed to fetch account info: {e}")

        if not result or not result.get("status") or not result.get("data"):
            return {
                "equity": 0.0,
                "buying_power": 0.0,
                "cash": 0.0,
                "margin_used": 0.0,
            }

        data = result["data"]
        return {
            "equity": float(data.get("net", 0)),
            "buying_power": float(data.get("availablecash", 0)),
            "cash": float(data.get("availablecash", 0)),
            "margin_used": float(data.get("utiliseddebits", 0)),
            "collateral": float(data.get("collateral", 0)),
        }

    # ══════════════════════════════════════════
    #  MARKET QUOTES
    # ══════════════════════════════════════════

    def get_market_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch current market quote including LTP, OHLC."""
        exchange = self._detect_exchange(symbol)
        token = self.get_symbol_token(symbol, exchange)

        self._throttle()
        try:
            result = self._smart_api.getMarketData(
                mode="FULL",
                exchangeTokens={exchange: [token]},
            )
        except Exception as e:
            # Fallback to LTP API
            try:
                ltp_result = self._smart_api.ltpData(exchange, symbol, token)
                if ltp_result and ltp_result.get("data"):
                    return {
                        "ltp": float(ltp_result["data"].get("ltp", 0)),
                        "open": float(ltp_result["data"].get("open", 0)),
                        "high": float(ltp_result["data"].get("high", 0)),
                        "low": float(ltp_result["data"].get("low", 0)),
                        "close": float(ltp_result["data"].get("close", 0)),
                        "volume": None,
                        "bid": None,
                        "ask": None,
                    }
            except Exception:
                pass
            raise BrokerAPIError(f"Failed to fetch quote for {symbol}: {e}")

        if not result or not result.get("status") or not result.get("data"):
            return {"ltp": None}

        fetched = result["data"].get("fetched", [])
        if not fetched:
            return {"ltp": None}

        quote = fetched[0]
        depth = quote.get("depth", {})
        best_buy = depth.get("buy", [{}])
        best_sell = depth.get("sell", [{}])

        return {
            "ltp": float(quote.get("ltp", 0)),
            "open": float(quote.get("open", 0)),
            "high": float(quote.get("high", 0)),
            "low": float(quote.get("low", 0)),
            "close": float(quote.get("close", 0)),
            "prev_close": float(quote.get("close", 0)),
            "volume": int(quote.get("tradeVolume", 0)),
            "oi": int(quote.get("opnInterest", 0)),
            "bid": float(best_buy[0].get("price", 0)) if best_buy else None,
            "ask": float(best_sell[0].get("price", 0)) if best_sell else None,
            "timestamp": quote.get("exchFeedTime"),
        }

    def get_ltp(self, symbol: str) -> Optional[float]:
        """Get last traded price (lightweight)."""
        exchange = self._detect_exchange(symbol)
        token = self.get_symbol_token(symbol, exchange)

        self._throttle()
        try:
            result = self._smart_api.ltpData(exchange, symbol, token)
            if result and result.get("data"):
                return float(result["data"].get("ltp", 0))
        except Exception as e:
            logger.error("Failed to fetch LTP for %s: %s", symbol, e)
        return None

    # ══════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════

    @staticmethod
    def _detect_exchange(symbol: str) -> str:
        """Detect exchange from symbol name."""
        upper = symbol.upper()
        has_digits = any(c.isdigit() for c in upper)

        # BSE spot indices
        if upper in ("SENSEX", "BANKEX", "BSE500", "BSE200", "BSE100"):
            return "BSE"

        # NSE spot indices — bare names with no expiry digits
        if upper in ("NIFTY", "NIFTY 50", "NIFTY50", "BANKNIFTY", "NIFTY BANK",
                     "FINNIFTY", "MIDCPNIFTY", "INDIA VIX", "VIX"):
            return "NSE"

        # Names with digits starting with index prefix → derivative (futures/options)
        if has_digits and any(
            upper.startswith(idx)
            for idx in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")
        ):
            return "NFO"
        return "NSE"


class AngelOneAPIError(Exception):
    pass
