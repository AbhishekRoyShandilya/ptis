# ingestion/broker_adapters/groww.py
"""
Groww Trading API adapter — read-only broker integration for NSE/BSE.

Uses the official `growwapi` SDK (>=1.5.0). Implements the BrokerAdapter
interface for historical candles, positions, orders, account info, and quotes.

Auth (growwapi 1.5.0):
  - GrowwAPI(token)                          → `token` is a valid ACCESS TOKEN.
  - GrowwAPI.get_access_token(api_key, totp=…, secret=…)
        mints an access token from an API key:
          • TOTP-type api key      → pass `totp` (a 6-digit code)
          • Approval-type api key  → pass `secret`

IMPORTANT: Groww access tokens are typically bound to the IP/network where they
were generated. This adapter must therefore run on the user's own machine
(their backend), not a remote host — otherwise Groww rejects calls as
"Access forbidden" / "token invalid" even with valid credentials.

The real SDK methods (verified via introspection):
  get_order_list, get_positions_for_user, get_user_profile,
  get_available_margin_details, get_holdings_for_user,
  get_historical_candle_data, get_ltp, get_quote, get_trade_list_for_order
"""

from __future__ import annotations

import logging
import time as time_mod
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from config.settings import settings
from ingestion.broker_api import (
    BrokerAdapter,
    BrokerAPIError,
    BrokerAuthError,
    BrokerRateLimitError,
)

logger = logging.getLogger(__name__)


def _unwrap(result: Any, *list_keys: str) -> Any:
    """
    Defensively unwrap a Groww SDK response to the payload of interest.
    Groww responses vary; try the common envelope keys then fall back to the
    object itself. *list_keys are candidate keys that hold the target list/dict.
    """
    if result is None:
        return None
    data = result
    if isinstance(result, dict):
        # common envelopes
        for env in ("payload", "data", "result"):
            if env in result and isinstance(result[env], (dict, list)):
                data = result[env]
                break
        if isinstance(data, dict):
            for k in list_keys:
                if k in data:
                    return data[k]
    return data


def _f(d: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


def _i(d: dict, *keys: str, default: int = 0) -> int:
    return int(_f(d, *keys, default=default))


def _s(d: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return str(v)
    return default


class GrowwClient(BrokerAdapter):
    """Read-only Groww Trading API client (growwapi SDK)."""

    name: str = "groww"

    # Groww segments to query for orders/positions (user trades CASH + FNO)
    _SEGMENTS = ("CASH", "FNO")

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        totp_secret: Optional[str] = None,
    ):
        self._api_key = api_key or settings.GROWW_API_KEY
        self._api_secret = api_secret or settings.GROWW_API_SECRET
        self._access_token = access_token or settings.GROWW_ACCESS_TOKEN
        self._totp_secret = totp_secret or settings.GROWW_TOTP_SECRET

        if not (self._totp_secret or self._access_token or self._api_key):
            raise BrokerAuthError(
                "Groww needs GROWW_TOTP_SECRET (+API key), GROWW_ACCESS_TOKEN, "
                "or GROWW_API_KEY (+secret)"
            )

        self._client = None
        self._token: Optional[str] = None
        self._token_exp: float = 0.0
        self._last_request_time: float = 0.0
        self._min_request_interval: float = 0.1  # 10 req/sec
        self._logged_shapes: set = set()

        self._init_client()

    # ── Auth (with TOTP auto-refresh) ─────────────────────────────

    def _init_client(self) -> None:
        try:
            from growwapi import GrowwAPI
        except ImportError:
            raise ImportError(
                "growwapi package not installed. Install: pip install growwapi>=1.5.0"
            )

        token = None
        # Preferred: TOTP auto-mint → a fresh token, no daily manual paste.
        if self._totp_secret and self._api_key:
            token = self._mint_token_via_totp()
        # Else: a directly-supplied (daily) access token.
        elif self._access_token:
            token = self._access_token
        # Else: approval flow (api_key + secret).
        elif self._api_key:
            try:
                resp = GrowwAPI.get_access_token(
                    api_key=self._api_key, secret=self._api_secret or None,
                )
                token = self._extract_token(resp)
            except Exception as e:
                raise BrokerAuthError(f"Groww get_access_token failed: {e}")

        if not token:
            raise BrokerAuthError("Could not obtain a Groww access token")

        try:
            self._client = GrowwAPI(token)
            self._token = token
            self._token_exp = self._token_expiry(token)
            logger.info("Groww client initialized (auth=%s)",
                        "totp" if self._totp_secret else "static-token")
        except Exception as e:
            raise BrokerAuthError(f"GrowwAPI(token) init failed: {e}")

    def _mint_token_via_totp(self) -> Optional[str]:
        try:
            import pyotp
        except ImportError:
            raise BrokerAuthError(
                "pyotp not installed (pip install pyotp) — needed for Groww TOTP auto-refresh"
            )
        from growwapi import GrowwAPI
        import time as _time
        totp = pyotp.TOTP(self._totp_secret)
        last_err = "unknown error"
        # The mint is a ONE-SHOT TOTP+network call; a 30s-window boundary (stale code) or a transient
        # network blip would otherwise fail the ENTIRE trade-sync service for the whole session — the
        # "Groww shows Not Configured until I restart" bug. Retry up to 3x with a fresh code each time
        # so a single blip can't disable trade sync.
        for attempt in range(1, 4):
            try:
                code = totp.now()
                resp = GrowwAPI.get_access_token(api_key=self._api_key, totp=code)
                tok = self._extract_token(resp)
                if tok:
                    logger.info("Groww token minted via TOTP (attempt %d/3)", attempt)
                    return tok
                last_err = "no token in get_access_token response"
            except Exception as e:
                last_err = str(e)
                logger.warning("Groww TOTP mint attempt %d/3 failed: %s", attempt, e)
            if attempt < 3:
                _time.sleep(3)  # brief backoff; lets the 30s TOTP window roll so the next code is fresh
        raise BrokerAuthError(f"Groww TOTP mint failed after 3 attempts: {last_err}")

    @staticmethod
    def _token_expiry(token: str) -> float:
        """Read the `exp` (epoch seconds) claim from the access-token JWT."""
        try:
            import base64
            import json
            payload = token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            return float(json.loads(base64.urlsafe_b64decode(payload)).get("exp", 0))
        except Exception:
            return 0.0

    def _ensure_token(self) -> None:
        """Re-mint the access token via TOTP when expired/near-expiry (60s buffer)."""
        if not (self._totp_secret and self._api_key):
            return  # static-token mode — nothing to refresh
        if self._token_exp and time_mod.time() < self._token_exp - 60:
            return  # still valid
        try:
            logger.info("Groww token expired/near-expiry — re-minting via TOTP")
            self._init_client()
        except Exception as e:
            logger.warning("Groww token re-mint failed: %s", e)

    @staticmethod
    def _extract_token(resp: Any) -> Optional[str]:
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            for k in ("access_token", "token", "accessToken"):
                if resp.get(k):
                    return resp[k]
            payload = resp.get("payload") or resp.get("data") or {}
            if isinstance(payload, dict):
                for k in ("access_token", "token", "accessToken"):
                    if payload.get(k):
                        return payload[k]
        return None

    def _throttle(self) -> None:
        # Auto-refresh the token before every API call (cheap no-op when valid).
        self._ensure_token()
        elapsed = time_mod.monotonic() - self._last_request_time
        if elapsed < self._min_request_interval:
            time_mod.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time_mod.monotonic()

    def _log_shape(self, tag: str, result: Any) -> None:
        """Log the raw response shape ONCE per tag, to finalize field mapping."""
        if tag in self._logged_shapes:
            return
        self._logged_shapes.add(tag)
        try:
            if isinstance(result, dict):
                logger.info("Groww %s raw keys: %s", tag, list(result.keys()))
            elif isinstance(result, list) and result:
                logger.info("Groww %s list[0] keys: %s",
                            tag, list(result[0].keys()) if isinstance(result[0], dict) else type(result[0]))
            else:
                logger.info("Groww %s raw type: %s", tag, type(result).__name__)
        except Exception:
            pass

    def _wrap_err(self, what: str, e: Exception) -> Exception:
        s = str(e).lower()
        if "rate" in s or "throttl" in s:
            return BrokerRateLimitError(f"Groww rate limit ({what}): {e}")
        if "auth" in s or "token" in s or "forbidden" in s or "permission" in s:
            return BrokerAuthError(f"Groww auth error ({what}): {e}")
        return BrokerAPIError(f"Groww {what} error: {e}")

    # ══════════════════════════════════════════
    #  ORDERS  (real method: get_order_list)
    # ══════════════════════════════════════════

    def get_orders(
        self,
        from_date: Optional[date] = None,
        status: str = "filled",
    ) -> List[Dict[str, Any]]:
        """Fetch order history across segments and normalise for the pipeline."""
        raw_orders: List[dict] = []
        for segment in self._SEGMENTS:
            self._throttle()
            try:
                result = self._client.get_order_list(page=0, page_size=100, segment=segment)
            except Exception as e:
                # Some accounts/segments may be disallowed — skip, don't abort all.
                logger.debug("get_order_list(%s) failed: %s", segment, e)
                continue
            self._log_shape(f"order_list[{segment}]", result)
            lst = _unwrap(result, "order_list", "orderList", "orders")
            if isinstance(lst, list):
                raw_orders.extend(o for o in lst if isinstance(o, dict))

        # status filter: 'filled' → Groww COMPLETE/EXECUTED
        want = {
            "filled": {"complete", "completed", "executed", "filled"},
            "open": {"open", "pending", "new"},
            "cancelled": {"cancelled", "canceled"},
            "rejected": {"rejected", "failed"},
        }.get(status.lower())

        out: List[Dict[str, Any]] = []
        for o in raw_orders:
            ostatus = _s(o, "order_status", "orderStatus", "status").lower()
            if want and ostatus not in want:
                continue

            ts = (o.get("trade_date_time") or o.get("exchange_time")
                  or o.get("created_at") or o.get("order_timestamp")
                  or o.get("orderTimestamp") or o.get("updateTime"))
            filled_at = self._parse_ts(ts)
            if from_date and filled_at and filled_at.date() < from_date:
                continue

            out.append({
                "order_id": _s(o, "groww_order_id", "order_id", "orderId", "id"),
                "symbol": _s(o, "trading_symbol", "tradingSymbol", "symbol"),
                "side": _s(o, "transaction_type", "transactionType", "side").lower(),
                "qty": _i(o, "filled_quantity", "filledQuantity", "quantity", "qty"),
                "price": _f(o, "average_fill_price", "average_price", "averagePrice", "price"),
                "filled_at": filled_at,
                "status": ostatus,
                "exchange": _s(o, "exchange", default="NSE"),
                "segment": _s(o, "segment"),
                "product": _s(o, "product", "productType"),
            })
        # Chronological order so buys are processed before their matching sells.
        out.sort(key=lambda o: o.get("filled_at") or datetime.min)
        return out

    # ══════════════════════════════════════════
    #  POSITIONS  (real method: get_positions_for_user)
    # ══════════════════════════════════════════

    def get_positions(self) -> List[Dict[str, Any]]:
        positions: List[Dict[str, Any]] = []
        for segment in self._SEGMENTS:
            self._throttle()
            try:
                result = self._client.get_positions_for_user(segment=segment)
            except Exception as e:
                logger.debug("get_positions_for_user(%s) failed: %s", segment, e)
                continue
            self._log_shape(f"positions[{segment}]", result)
            lst = _unwrap(result, "positions", "position_list", "net_positions")
            if not isinstance(lst, list):
                continue
            for p in lst:
                if not isinstance(p, dict):
                    continue
                qty = _i(p, "net_quantity", "netQuantity", "quantity", "qty")
                if qty == 0:
                    continue
                positions.append({
                    "symbol": _s(p, "trading_symbol", "tradingSymbol", "symbol"),
                    "exchange": _s(p, "exchange", default="NSE"),
                    "segment": segment,
                    "quantity": qty,
                    "avg_price": _f(p, "average_price", "net_price", "averagePrice", "avgPrice"),
                    "pnl": _f(p, "pnl", "unrealized_pnl", "unrealizedPnl"),
                    "side": "long" if qty > 0 else "short",
                    "product": _s(p, "product", "productType"),
                    "ltp": _f(p, "ltp", "last_price", "lastPrice"),
                })
        return positions

    def get_closed_positions(self) -> List[Dict[str, Any]]:
        """Round-tripped (net qty 0) positions that carry a realised P&L — i.e. CROSS-DAY closes whose
        opening leg is no longer in the intraday order book. Returns the raw legs + realised_pnl; the
        trade-sync reconstructs entry/exit direction from the pnl sign (Groww's credit/debit labels are
        ambiguous). Only these need position-based recovery; same-day trades pair fine from orders."""
        out: List[Dict[str, Any]] = []
        for segment in self._SEGMENTS:
            self._throttle()
            try:
                result = self._client.get_positions_for_user(segment=segment)
            except Exception as e:
                logger.debug("get_positions_for_user(%s) failed: %s", segment, e)
                continue
            lst = _unwrap(result, "positions", "position_list", "net_positions")
            if not isinstance(lst, list):
                continue
            for p in lst:
                if not isinstance(p, dict):
                    continue
                net = _i(p, "quantity", "net_quantity", "netQuantity")
                cq = _i(p, "credit_quantity", "creditQuantity")
                dq = _i(p, "debit_quantity", "debitQuantity")
                rpnl = _f(p, "realised_pnl", "realized_pnl", "realisedPnl")
                if net != 0 or (cq <= 0 and dq <= 0) or not rpnl:
                    continue    # only FULLY-closed positions with a realised result
                out.append({
                    "symbol": _s(p, "trading_symbol", "tradingSymbol", "symbol"),
                    "segment": segment,
                    "quantity": cq or dq,
                    "credit_price": _f(p, "credit_price", "creditPrice"),
                    "debit_price": _f(p, "debit_price", "debitPrice"),
                    "realised_pnl": rpnl,
                })
        return out

    # ══════════════════════════════════════════
    #  ACCOUNT  (profile + margins)
    # ══════════════════════════════════════════

    def get_account_info(self) -> Dict[str, Any]:
        self._throttle()
        profile, margin = {}, {}
        try:
            profile = _unwrap(self._client.get_user_profile()) or {}
            self._log_shape("user_profile", profile)
        except Exception as e:
            logger.debug("get_user_profile failed: %s", e)
        try:
            self._throttle()
            margin = _unwrap(self._client.get_available_margin_details()) or {}
            self._log_shape("margin", margin)
        except Exception as e:
            logger.debug("get_available_margin_details failed: %s", e)

        if not isinstance(profile, dict):
            profile = {}
        if not isinstance(margin, dict):
            margin = {}

        clear_cash = _f(margin, "clear_cash", "cash", "available_cash")
        collateral = _f(margin, "collateral_available")
        used = _f(margin, "net_margin_used", "used_margin", "utilised_margin")
        return {
            "name": _s(profile, "vendor_user_id", "ucc", "name", "user_name"),
            "user_id": _s(profile, "ucc", "vendor_user_id", "user_id"),
            "equity": round(clear_cash + collateral, 2),
            "buying_power": round(clear_cash + collateral - used, 2),
            "cash": clear_cash,
            "margin_used": used,
        }

    # ══════════════════════════════════════════
    #  HISTORICAL CANDLES  (get_historical_candle_data)
    # ══════════════════════════════════════════

    def get_historical_candles(
        self,
        symbol: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> List[Dict[str, Any]]:
        exchange = self._detect_exchange(symbol)
        segment = "FNO" if exchange == "NFO" else "CASH"
        interval_min = self._interval_minutes(interval)

        self._throttle()
        try:
            result = self._client.get_historical_candle_data(
                trading_symbol=symbol,
                exchange="NSE",
                segment=segment,
                start_time=from_date.isoformat(),
                end_time=to_date.isoformat(),
                interval_in_minutes=interval_min,
            )
        except Exception as e:
            raise self._wrap_err("historical candles", e)

        self._log_shape("historical_candles", result)
        rows = _unwrap(result, "candles", "candle_list", "data")
        if not isinstance(rows, list):
            return []

        candles: List[Dict[str, Any]] = []
        for c in rows:
            # Groww candles are often [ts, open, high, low, close, volume] arrays
            if isinstance(c, (list, tuple)) and len(c) >= 6:
                candles.append({
                    "timestamp": self._parse_ts(c[0]) or c[0],
                    "open": float(c[1]), "high": float(c[2]),
                    "low": float(c[3]), "close": float(c[4]),
                    "volume": int(c[5]),
                })
            elif isinstance(c, dict):
                candles.append({
                    "timestamp": self._parse_ts(c.get("timestamp") or c.get("time")) or c.get("timestamp"),
                    "open": _f(c, "open"), "high": _f(c, "high"),
                    "low": _f(c, "low"), "close": _f(c, "close"),
                    "volume": _i(c, "volume"),
                })
        return candles

    # ══════════════════════════════════════════
    #  QUOTES / LTP
    # ══════════════════════════════════════════

    def get_ltp(self, symbol: str) -> Optional[float]:
        segment = "FNO" if self._detect_exchange(symbol) == "NFO" else "CASH"
        self._throttle()
        try:
            result = self._client.get_ltp(
                exchange_trading_symbols=(f"NSE_{symbol}",),
                segment=segment,
            )
            data = _unwrap(result) or {}
            if isinstance(data, dict):
                # response often keyed by symbol → price
                for v in data.values():
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, dict):
                        return _f(v, "ltp", "last_price", "lastPrice")
                return _f(data, "ltp", "last_price", "lastPrice") or None
        except Exception as e:
            logger.debug("get_ltp(%s) failed: %s", symbol, e)
        return None

    def get_market_quote(self, symbol: str) -> Dict[str, Any]:
        segment = "FNO" if self._detect_exchange(symbol) == "NFO" else "CASH"
        self._throttle()
        try:
            result = self._client.get_quote(
                exchange="NSE", segment=segment, trading_symbol=symbol,
            )
        except Exception as e:
            raise self._wrap_err("quote", e)
        self._log_shape("quote", result)
        d = _unwrap(result) or {}
        if not isinstance(d, dict):
            return {"ltp": None}
        return {
            "ltp": _f(d, "last_price", "ltp", "lastPrice"),
            "open": _f(d, "open", "open_price"),
            "high": _f(d, "high", "day_high"),
            "low": _f(d, "low", "day_low"),
            "close": _f(d, "close", "close_price"),
            "prev_close": _f(d, "previous_close", "prev_close", "close"),
            "volume": _i(d, "volume", "total_traded_volume"),
            "oi": _i(d, "oi", "open_interest"),
            "bid": d.get("bid_price") or d.get("bestBid"),
            "ask": d.get("ask_price") or d.get("bestAsk"),
        }

    # ══════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════

    @staticmethod
    def _parse_ts(ts: Any) -> Optional[datetime]:
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            # epoch seconds or ms
            try:
                val = float(ts)
                if val > 1e12:
                    val /= 1000.0
                return datetime.fromtimestamp(val, tz=timezone.utc)
            except Exception:
                return None
        if isinstance(ts, str):
            for parse in (
                lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")),
            ):
                try:
                    return parse(ts)
                except Exception:
                    pass
        return None

    @staticmethod
    def _interval_minutes(interval: str) -> int:
        m = {
            "1m": 1, "1min": 1, "3m": 3, "5m": 5, "5min": 5,
            "10m": 10, "15m": 15, "15min": 15, "30m": 30, "30min": 30,
            "60m": 60, "1h": 60, "1d": 1440, "day": 1440,
        }
        return m.get(interval.lower(), 5)

    @staticmethod
    def _detect_exchange(symbol: str) -> str:
        upper = symbol.upper()
        has_digits = any(c.isdigit() for c in upper)
        if has_digits and any(
            upper.startswith(idx)
            for idx in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")
        ):
            return "NFO"
        return "NSE"


class GrowwAPIError(Exception):
    pass
