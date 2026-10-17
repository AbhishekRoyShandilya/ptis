# ingestion/broker_adapters/generic.py
"""
Generic Broker Adapter — a configurable adapter that works with
any REST-based broker API by mapping response fields.

Use this when a broker doesn't have a dedicated adapter:
    client = GenericBrokerClient(
        base_url="https://api.mybroker.com",
        headers={"Authorization": "Bearer ..."},
        field_map={"symbol": "instrument", "ltp": "last_price"},
    )
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import httpx

from ingestion.broker_api import BrokerAdapter, BrokerAPIError

logger = logging.getLogger(__name__)


# Default field mappings (broker field → our standard field)
_DEFAULT_FIELD_MAP = {
    "symbol": "symbol",
    "direction": "side",
    "entry_price": "price",
    "quantity": "quantity",
    "pnl": "pnl",
    "ltp": "last_price",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "timestamp": "timestamp",
}


class GenericBrokerClient(BrokerAdapter):
    """
    Configurable adapter for any REST-based broker.

    Usage:
        client = GenericBrokerClient(
            base_url="https://api.broker.com",
            headers={"Authorization": "Bearer token"},
            endpoints={
                "positions": "/api/positions",
                "orders": "/api/orders",
                "candles": "/api/candles/{symbol}",
                "quote": "/api/quote/{symbol}",
                "account": "/api/account",
            },
            field_map={"symbol": "ticker", "ltp": "last"},
        )
    """

    name = "generic"

    def __init__(
        self,
        base_url: str,
        headers: Dict[str, str] | None = None,
        endpoints: Dict[str, str] | None = None,
        field_map: Dict[str, str] | None = None,
        timeout: float = 30.0,
    ):
        self._base_url = base_url
        self._endpoints = endpoints or {}
        self._field_map = {**_DEFAULT_FIELD_MAP, **(field_map or {})}
        self._client = httpx.Client(
            base_url=base_url,
            headers=headers or {},
            timeout=timeout,
        )

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            resp = self._client.request(method, path, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise BrokerAPIError(
                f"Generic broker HTTP {e.response.status_code}: {e.response.text}"
            )

    def _map_fields(self, record: Dict, reverse: bool = False) -> Dict:
        """Map broker field names to standard names (or reverse)."""
        mapped = {}
        mapping = (
            {v: k for k, v in self._field_map.items()}
            if reverse
            else self._field_map
        )
        for std_name, broker_name in mapping.items():
            if broker_name in record:
                mapped[std_name] = record[broker_name]
        # Include unmapped fields as-is
        for k, v in record.items():
            if k not in mapped.values():
                mapped.setdefault(k, v)
        return mapped

    def get_account_info(self) -> Dict[str, Any]:
        endpoint = self._endpoints.get("account", "/account")
        data = self._request("GET", endpoint)
        return self._map_fields(data) if isinstance(data, dict) else data

    def get_positions(self) -> List[Dict[str, Any]]:
        endpoint = self._endpoints.get("positions", "/positions")
        data = self._request("GET", endpoint)
        if not isinstance(data, list):
            data = data.get("positions", data.get("data", []))
        return [self._map_fields(p) for p in data]

    def get_orders(
        self,
        from_date: Optional[date] = None,
        status: str = "filled",
    ) -> List[Dict[str, Any]]:
        endpoint = self._endpoints.get("orders", "/orders")
        params: Dict[str, Any] = {}
        if from_date:
            params["from"] = from_date.isoformat()
        if status != "all":
            params["status"] = status

        data = self._request("GET", endpoint, params=params)
        if not isinstance(data, list):
            data = data.get("orders", data.get("data", []))
        return [self._map_fields(o) for o in data]

    def get_historical_candles(
        self,
        symbol: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> List[Dict[str, Any]]:
        endpoint = self._endpoints.get("candles", "/candles/{symbol}")
        endpoint = endpoint.format(symbol=symbol)

        params = {
            "interval": interval,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
        }

        data = self._request("GET", endpoint, params=params)
        if not isinstance(data, list):
            data = data.get("candles", data.get("bars", data.get("data", [])))

        return [self._map_fields(c) for c in data]

    def get_market_quote(self, symbol: str) -> Dict[str, Any]:
        endpoint = self._endpoints.get("quote", "/quote/{symbol}")
        endpoint = endpoint.format(symbol=symbol)
        data = self._request("GET", endpoint)
        return self._map_fields(data) if isinstance(data, dict) else data
