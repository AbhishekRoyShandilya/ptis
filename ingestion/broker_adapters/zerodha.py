# ingestion/broker_adapters/zerodha.py
"""
Zerodha Kite Connect adapter — read-only broker integration for NSE/BSE.

Implements BrokerAdapter interface for:
- Historical OHLCV candles
- Positions, orders, account info
- Market quotes (LTP, depth)

Requires: kiteconnect>=5.0.0
Auth: Kite Connect uses api_key + access_token (daily refresh via login redirect).

Rate limits:
- Historical data: 3 requests/second
- Other APIs: 10 requests/second
"""

from __future__ import annotations

import logging
import time as time_mod
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from config.settings import settings
from ingestion.broker_api import BrokerAdapter, BrokerAPIError, BrokerAuthError, BrokerRateLimitError

logger = logging.getLogger(__name__)


class KiteInterval:
    """Kite Connect candle interval constants."""
    MINUTE = "minute"
    THREE_MINUTE = "3minute"
    FIVE_MINUTE = "5minute"
    TEN_MINUTE = "10minute"
    FIFTEEN_MINUTE = "15minute"
    THIRTY_MINUTE = "30minute"
    SIXTY_MINUTE = "60minute"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

    _MAP = {
        "1m": "minute", "1min": "minute",
        "3m": "3minute", "3min": "3minute",
        "5m": "5minute", "5min": "5minute",
        "10m": "10minute", "10min": "10minute",
        "15m": "15minute", "15min": "15minute",
        "30m": "30minute", "30min": "30minute",
        "60m": "60minute", "1h": "60minute",
        "1d": "day", "day": "day",
        "1w": "week", "week": "week",
    }

    @classmethod
    def resolve(cls, interval: str) -> str:
        """Map a generic interval string to Kite's format."""
        return cls._MAP.get(interval.lower(), interval)


class ZerodhaClient(BrokerAdapter):
    """
    Read-only Zerodha Kite Connect client.
    Wraps the kiteconnect SDK for historical data, positions, orders, and quotes.
    """

    name: str = "zerodha"

    def __init__(
        self,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
    ):
        self._api_key = api_key or settings.KITE_API_KEY
        self._access_token = access_token or settings.KITE_ACCESS_TOKEN

        if not self._api_key:
            raise BrokerAuthError("KITE_API_KEY is required for Zerodha adapter")

        self._kite = None
        self._instrument_cache: Dict[str, int] = {}
        self._last_request_time: float = 0.0
        self._min_request_interval: float = 0.34  # ~3 req/sec for historical

        self._init_client()

    def _init_client(self) -> None:
        """Initialize the KiteConnect client."""
        try:
            from kiteconnect import KiteConnect
        except ImportError:
            raise ImportError(
                "kiteconnect package not installed. "
                "Install with: pip install kiteconnect>=5.0.0"
            )

        self._kite = KiteConnect(api_key=self._api_key)

        if self._access_token:
            self._kite.set_access_token(self._access_token)
            logger.info("Zerodha client initialized with access token")
        else:
            logger.warning(
                "Zerodha client initialized WITHOUT access token. "
                "Call generate_session() with a request_token to authenticate."
            )

    def generate_session(self, request_token: str) -> Dict[str, Any]:
        """
        Exchange request_token for access_token.
        Must be called once per day after the Kite login redirect.

        Returns session data including access_token.
        """
        if not settings.KITE_API_SECRET:
            raise BrokerAuthError("KITE_API_SECRET is required for session generation")

        try:
            data = self._kite.generate_session(
                request_token,
                api_secret=settings.KITE_API_SECRET,
            )
            self._access_token = data["access_token"]
            self._kite.set_access_token(self._access_token)
            logger.info("Zerodha session generated successfully")
            return data
        except Exception as e:
            raise BrokerAuthError(f"Failed to generate Zerodha session: {e}")

    def _throttle(self) -> None:
        """Simple rate limiter — ensures minimum interval between requests."""
        elapsed = time_mod.monotonic() - self._last_request_time
        if elapsed < self._min_request_interval:
            time_mod.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time_mod.monotonic()

    # ══════════════════════════════════════════
    #  INSTRUMENT TOKEN RESOLUTION
    # ══════════════════════════════════════════

    def get_instrument_token(
        self,
        symbol: str,
        exchange: str = "NSE",
    ) -> int:
        """
        Resolve a human-readable symbol to Kite's instrument_token.
        Caches results. Uses the instruments dump API.
        """
        cache_key = f"{exchange}:{symbol}"
        if cache_key in self._instrument_cache:
            return self._instrument_cache[cache_key]

        try:
            instruments = self._kite.instruments(exchange)
            for inst in instruments:
                key = f"{inst['exchange']}:{inst['tradingsymbol']}"
                self._instrument_cache[key] = inst["instrument_token"]

            token = self._instrument_cache.get(cache_key)
            if token is None:
                raise BrokerAPIError(
                    f"Instrument not found: {cache_key}. "
                    f"Cached {len(self._instrument_cache)} instruments from {exchange}."
                )
            return token
        except BrokerAPIError:
            raise
        except Exception as e:
            raise BrokerAPIError(f"Failed to fetch instruments: {e}")

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
        Fetch historical OHLCV candles from Kite Connect.

        Args:
            symbol: Trading symbol (e.g., "RELIANCE", "NIFTY 50")
            interval: Candle interval (e.g., "5m", "15m", "1d")
            from_date: Start date
            to_date: End date

        Returns:
            List of {timestamp, open, high, low, close, volume} dicts
        """
        exchange = self._detect_exchange(symbol)
        token = self.get_instrument_token(symbol, exchange)
        kite_interval = KiteInterval.resolve(interval)

        from_dt = datetime.combine(from_date, datetime.min.time())
        to_dt = datetime.combine(to_date, datetime.max.time().replace(microsecond=0))

        all_candles = []

        # Kite limits historical data per request — paginate by chunks
        chunk_days = self._get_chunk_days(kite_interval)
        current_from = from_dt

        while current_from < to_dt:
            current_to = min(current_from + timedelta(days=chunk_days), to_dt)

            self._throttle()
            try:
                raw = self._kite.historical_data(
                    instrument_token=token,
                    from_date=current_from,
                    to_date=current_to,
                    interval=kite_interval,
                )
            except Exception as e:
                err_str = str(e).lower()
                if "too many requests" in err_str or "rate" in err_str:
                    raise BrokerRateLimitError(f"Kite historical rate limit: {e}")
                if "token" in err_str or "auth" in err_str:
                    raise BrokerAuthError(f"Kite auth error: {e}")
                raise BrokerAPIError(f"Kite historical data error: {e}")

            for candle in raw:
                all_candles.append({
                    "timestamp": candle["date"],
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": int(candle["volume"]),
                })

            current_from = current_to + timedelta(seconds=1)

        logger.debug(
            "Fetched %d candles for %s:%s (%s) %s→%s",
            len(all_candles), exchange, symbol, kite_interval, from_date, to_date,
        )
        return all_candles

    # ══════════════════════════════════════════
    #  POSITIONS
    # ══════════════════════════════════════════

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch current open positions from Zerodha."""
        self._throttle()
        try:
            positions = self._kite.positions()
        except Exception as e:
            raise BrokerAPIError(f"Failed to fetch positions: {e}")

        result = []
        for pos in positions.get("net", []):
            if pos.get("quantity", 0) == 0:
                continue
            result.append({
                "symbol": pos["tradingsymbol"],
                "exchange": pos.get("exchange", "NSE"),
                "quantity": pos["quantity"],
                "avg_price": pos.get("average_price", 0.0),
                "pnl": pos.get("pnl", 0.0),
                "side": "long" if pos["quantity"] > 0 else "short",
                "product": pos.get("product", ""),
                "ltp": pos.get("last_price", 0.0),
            })

        return result

    # ══════════════════════════════════════════
    #  ORDERS
    # ══════════════════════════════════════════

    def get_orders(
        self,
        from_date: Optional[date] = None,
        status: str = "filled",
    ) -> List[Dict[str, Any]]:
        """Fetch order history from Zerodha."""
        self._throttle()
        try:
            orders = self._kite.orders()
        except Exception as e:
            raise BrokerAPIError(f"Failed to fetch orders: {e}")

        # Kite status mapping
        status_map = {
            "filled": "COMPLETE",
            "open": "OPEN",
            "cancelled": "CANCELLED",
            "rejected": "REJECTED",
        }
        kite_status = status_map.get(status, status.upper())

        result = []
        for order in orders:
            if kite_status and order.get("status") != kite_status:
                continue

            order_time = order.get("order_timestamp")
            if from_date and order_time:
                if isinstance(order_time, datetime) and order_time.date() < from_date:
                    continue

            result.append({
                "symbol": order["tradingsymbol"],
                "side": order.get("transaction_type", "").lower(),  # BUY/SELL → buy/sell
                "qty": order.get("filled_quantity", order.get("quantity", 0)),
                "price": order.get("average_price", 0.0),
                "filled_at": order.get("order_timestamp"),
                "status": order.get("status", "").lower(),
                "order_id": order.get("order_id"),
                "exchange": order.get("exchange", "NSE"),
                "product": order.get("product", ""),
            })

        return result

    # ══════════════════════════════════════════
    #  ACCOUNT INFO
    # ══════════════════════════════════════════

    def get_account_info(self) -> Dict[str, Any]:
        """Fetch account margins from Zerodha."""
        self._throttle()
        try:
            margins = self._kite.margins()
        except Exception as e:
            raise BrokerAPIError(f"Failed to fetch margins: {e}")

        equity = margins.get("equity", {})
        available = equity.get("available", {})
        utilised = equity.get("utilised", {})

        return {
            "equity": available.get("live_balance", 0.0),
            "buying_power": available.get("cash", 0.0),
            "cash": available.get("cash", 0.0),
            "margin_used": utilised.get("debits", 0.0),
            "collateral": available.get("collateral", 0.0),
        }

    # ══════════════════════════════════════════
    #  MARKET QUOTES
    # ══════════════════════════════════════════

    def get_market_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch current market quote including LTP, OHLC, depth."""
        exchange = self._detect_exchange(symbol)
        instrument = f"{exchange}:{symbol}"

        self._throttle()
        try:
            quotes = self._kite.quote([instrument])
        except Exception as e:
            raise BrokerAPIError(f"Failed to fetch quote for {instrument}: {e}")

        data = quotes.get(instrument, {})
        ohlc = data.get("ohlc", {})
        depth = data.get("depth", {})

        best_bid = depth.get("buy", [{}])[0] if depth.get("buy") else {}
        best_ask = depth.get("sell", [{}])[0] if depth.get("sell") else {}

        return {
            "ltp": data.get("last_price"),
            "open": ohlc.get("open"),
            "high": ohlc.get("high"),
            "low": ohlc.get("low"),
            "close": ohlc.get("close"),
            "prev_close": ohlc.get("close"),
            "volume": data.get("volume"),
            "oi": data.get("oi"),
            "bid": best_bid.get("price"),
            "ask": best_ask.get("price"),
            "timestamp": data.get("last_trade_time"),
        }

    def get_ltp(self, symbol: str) -> Optional[float]:
        """Get last traded price (lightweight call)."""
        exchange = self._detect_exchange(symbol)
        instrument = f"{exchange}:{symbol}"

        self._throttle()
        try:
            ltp_data = self._kite.ltp([instrument])
            return ltp_data.get(instrument, {}).get("last_price")
        except Exception as e:
            logger.error("Failed to fetch LTP for %s: %s", instrument, e)
            return None

    # ══════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════

    @staticmethod
    def _detect_exchange(symbol: str) -> str:
        """
        Detect exchange from symbol name.
        Index symbols (NIFTY, BANKNIFTY, etc.) go to NSE.
        Symbols with expiry-like suffixes go to NFO.
        """
        upper = symbol.upper()
        # Check index names first (take priority over F&O detection)
        _NSE_INDICES = {"NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE", "NIFTY MID SELECT"}
        if upper in _NSE_INDICES:
            return "NSE"

        # F&O symbols typically contain digits (expiry/strike)
        has_digits = any(c.isdigit() for c in upper)
        if has_digits and any(upper.startswith(idx) for idx in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")):
            return "NFO"
        return "NSE"

    @staticmethod
    def _get_chunk_days(interval: str) -> int:
        """Max days per historical data request (Kite limits)."""
        if interval in ("minute", "3minute"):
            return 60
        elif interval in ("5minute", "10minute", "15minute"):
            return 100
        elif interval in ("30minute", "60minute"):
            return 400
        else:
            return 2000  # daily, weekly, monthly
