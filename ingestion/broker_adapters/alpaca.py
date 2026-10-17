# ingestion/broker_adapters/alpaca.py
"""
Alpaca Markets API client — read-only access to account data,
positions, orders, and historical candles.

Endpoints used:
- /v2/account
- /v2/positions
- /v2/orders
- /v2/stocks/{symbol}/bars (data API)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from ingestion.broker_api import BrokerAdapter, BrokerAPIError, BrokerRateLimitError

logger = logging.getLogger(__name__)


class AlpacaClient(BrokerAdapter):
    """Read-only Alpaca Markets API client."""

    name = "alpaca"

    TRADING_URL = "https://api.alpaca.markets"
    PAPER_URL = "https://paper-api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        base_url = self.PAPER_URL if paper else self.TRADING_URL
        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
            timeout=30.0,
        )
        self._data_client = httpx.Client(
            base_url=self.DATA_URL,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
            timeout=30.0,
        )

    def _request(self, client: httpx.Client, method: str, path: str, **kwargs) -> Any:
        try:
            resp = client.request(method, path, **kwargs)
            if resp.status_code == 429:
                raise BrokerRateLimitError("Alpaca rate limit exceeded")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise BrokerAPIError(f"Alpaca HTTP {e.response.status_code}: {e.response.text}")

    def get_account_info(self) -> Dict[str, Any]:
        data = self._request(self._client, "GET", "/v2/account")
        return {
            "equity": float(data.get("equity", 0)),
            "buying_power": float(data.get("buying_power", 0)),
            "cash": float(data.get("cash", 0)),
            "portfolio_value": float(data.get("portfolio_value", 0)),
            "pattern_day_trader": data.get("pattern_day_trader", False),
            "day_trade_count": int(data.get("daytrade_count", 0)),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        data = self._request(self._client, "GET", "/v2/positions")
        positions = []
        for p in data:
            positions.append({
                "symbol": p["symbol"],
                "quantity": float(p["qty"]),
                "avg_price": float(p["avg_entry_price"]),
                "current_price": float(p.get("current_price", 0)),
                "pnl": float(p.get("unrealized_pl", 0)),
                "pnl_pct": float(p.get("unrealized_plpc", 0)) * 100,
                "side": p.get("side", "long"),
                "market_value": float(p.get("market_value", 0)),
            })
        return positions

    def get_orders(
        self,
        from_date: Optional[date] = None,
        status: str = "filled",
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"status": status, "limit": 500}
        if from_date:
            params["after"] = from_date.isoformat()

        data = self._request(self._client, "GET", "/v2/orders", params=params)
        orders = []
        for o in data:
            orders.append({
                "order_id": o["id"],
                "symbol": o["symbol"],
                "side": o["side"],
                "qty": float(o.get("filled_qty", o.get("qty", 0))),
                "price": float(o.get("filled_avg_price", 0)),
                "filled_at": o.get("filled_at"),
                "status": o["status"],
                "type": o.get("type", "market"),
                "time_in_force": o.get("time_in_force", "day"),
            })
        return orders

    def get_historical_candles(
        self,
        symbol: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> List[Dict[str, Any]]:
        # Map interval to Alpaca timeframe
        tf_map = {
            "1m": "1Min", "5m": "5Min", "15m": "15Min",
            "30m": "30Min", "1h": "1Hour", "4h": "4Hour",
            "1d": "1Day", "day": "1Day",
        }
        timeframe = tf_map.get(interval, "15Min")

        params = {
            "start": from_date.isoformat(),
            "end": to_date.isoformat(),
            "timeframe": timeframe,
            "limit": 10000,
        }

        data = self._request(
            self._data_client, "GET",
            f"/v2/stocks/{symbol}/bars",
            params=params,
        )

        candles = []
        for bar in data.get("bars", []):
            candles.append({
                "timestamp": bar["t"],
                "open": float(bar["o"]),
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "close": float(bar["c"]),
                "volume": int(bar["v"]),
            })
        return candles

    def get_market_quote(self, symbol: str) -> Dict[str, Any]:
        data = self._request(
            self._data_client, "GET",
            f"/v2/stocks/{symbol}/snapshot",
        )
        latest = data.get("latestTrade", {})
        quote = data.get("latestQuote", {})
        bar = data.get("dailyBar", {})

        return {
            "ltp": float(latest.get("p", 0)),
            "bid": float(quote.get("bp", 0)),
            "ask": float(quote.get("ap", 0)),
            "volume": int(bar.get("v", 0)),
            "open": float(bar.get("o", 0)),
            "high": float(bar.get("h", 0)),
            "low": float(bar.get("l", 0)),
        }
