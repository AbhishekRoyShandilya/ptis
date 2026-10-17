# ingestion/broker_adapters/interactive_brokers.py
"""
Interactive Brokers (IBKR) Client Portal API adapter.
Read-only access to account, positions, orders, and market data.

Requires: IBKR Client Portal Gateway running locally.
Default: https://localhost:5000
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import httpx

from ingestion.broker_api import BrokerAdapter, BrokerAPIError

logger = logging.getLogger(__name__)


class IBKRClient(BrokerAdapter):
    """Read-only Interactive Brokers Client Portal API client."""

    name = "interactive_brokers"

    def __init__(
        self,
        gateway_url: str = "https://localhost:5000",
        account_id: Optional[str] = None,
        verify_ssl: bool = False,
    ):
        self._base_url = gateway_url
        self._account_id = account_id
        self._client = httpx.Client(
            base_url=gateway_url,
            verify=verify_ssl,
            timeout=30.0,
        )

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            resp = self._client.request(method, path, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise BrokerAPIError(
                f"IBKR HTTP {e.response.status_code}: {e.response.text}"
            )

    def _get_account_id(self) -> str:
        if self._account_id:
            return self._account_id
        accounts = self._request("GET", "/v1/api/iserver/accounts")
        accts = accounts.get("accounts", [])
        if not accts:
            raise BrokerAPIError("No IBKR accounts found")
        self._account_id = accts[0]
        return self._account_id

    def get_account_info(self) -> Dict[str, Any]:
        acct = self._get_account_id()
        data = self._request("GET", f"/v1/api/portfolio/{acct}/summary")

        return {
            "equity": data.get("netliquidation", {}).get("amount", 0),
            "buying_power": data.get("buyingpower", {}).get("amount", 0),
            "cash": data.get("totalcashvalue", {}).get("amount", 0),
            "margin_used": data.get("maintmarginreq", {}).get("amount", 0),
            "unrealized_pnl": data.get("unrealizedpnl", {}).get("amount", 0),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        acct = self._get_account_id()
        data = self._request("GET", f"/v1/api/portfolio/{acct}/positions/0")

        positions = []
        for p in data if isinstance(data, list) else []:
            positions.append({
                "symbol": p.get("ticker", p.get("contractDesc", "")),
                "conid": p.get("conid"),
                "quantity": float(p.get("position", 0)),
                "avg_price": float(p.get("avgCost", 0)),
                "current_price": float(p.get("mktPrice", 0)),
                "pnl": float(p.get("unrealizedPnl", 0)),
                "market_value": float(p.get("mktValue", 0)),
                "side": "long" if p.get("position", 0) > 0 else "short",
            })
        return positions

    def get_orders(
        self,
        from_date: Optional[date] = None,
        status: str = "filled",
    ) -> List[Dict[str, Any]]:
        data = self._request("GET", "/v1/api/iserver/account/orders")

        orders = []
        for o in data.get("orders", []):
            order_status = o.get("status", "").lower()
            if status != "all" and order_status != status:
                continue

            orders.append({
                "order_id": str(o.get("orderId", "")),
                "symbol": o.get("ticker", ""),
                "side": o.get("side", "").lower(),
                "qty": float(o.get("filledQuantity", o.get("totalSize", 0))),
                "price": float(o.get("avgPrice", 0)),
                "filled_at": o.get("lastExecutionTime_r"),
                "status": order_status,
                "type": o.get("orderType", ""),
            })
        return orders

    def get_historical_candles(
        self,
        symbol: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> List[Dict[str, Any]]:
        # Need conid — search for it
        search_data = self._request(
            "GET", "/v1/api/iserver/secdef/search",
            params={"symbol": symbol},
        )
        if not search_data:
            return []

        conid = search_data[0].get("conid") if isinstance(search_data, list) else None
        if not conid:
            return []

        # Map interval
        period_map = {
            "1m": "1min", "5m": "5min", "15m": "15min",
            "30m": "30min", "1h": "1h", "4h": "4h",
            "1d": "1d",
        }
        bar = period_map.get(interval, "15min")

        # Calculate period
        days = (to_date - from_date).days
        period = f"{days}d" if days <= 365 else "1y"

        data = self._request(
            "GET", f"/v1/api/iserver/marketdata/history",
            params={
                "conid": conid,
                "period": period,
                "bar": bar,
            },
        )

        candles = []
        for bar_data in data.get("data", []):
            candles.append({
                "timestamp": datetime.fromtimestamp(bar_data["t"] / 1000).isoformat(),
                "open": float(bar_data["o"]),
                "high": float(bar_data["h"]),
                "low": float(bar_data["l"]),
                "close": float(bar_data["c"]),
                "volume": int(bar_data.get("v", 0)),
            })
        return candles

    def get_market_quote(self, symbol: str) -> Dict[str, Any]:
        search = self._request(
            "GET", "/v1/api/iserver/secdef/search",
            params={"symbol": symbol},
        )
        if not search or not isinstance(search, list):
            return {}

        conid = search[0].get("conid")
        if not conid:
            return {}

        data = self._request(
            "GET", f"/v1/api/iserver/marketdata/snapshot",
            params={"conids": str(conid), "fields": "31,84,85,86,87,88"},
        )

        quote = data[0] if isinstance(data, list) and data else {}
        return {
            "ltp": quote.get("31"),
            "bid": quote.get("84"),
            "ask": quote.get("86"),
            "volume": quote.get("87"),
            "open": quote.get("7295"),
            "high": quote.get("70"),
            "low": quote.get("71"),
        }
