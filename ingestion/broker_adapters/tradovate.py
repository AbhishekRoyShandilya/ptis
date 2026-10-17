# ingestion/broker_adapters/tradovate.py
"""
Tradovate API adapter — read-only access for futures trading data.

Endpoints used:
- /v1/account/list
- /v1/position/list
- /v1/order/list
- /v1/md/getChart (market data)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import httpx

from ingestion.broker_api import BrokerAdapter, BrokerAPIError, BrokerAuthError

logger = logging.getLogger(__name__)


class TradovateClient(BrokerAdapter):
    """Read-only Tradovate API client for futures data."""

    name = "tradovate"

    DEMO_URL = "https://demo.tradovateapi.com/v1"
    LIVE_URL = "https://live.tradovateapi.com/v1"
    MD_URL = "https://md.tradovateapi.com/v1"

    def __init__(
        self,
        access_token: str,
        live: bool = False,
    ):
        self._access_token = access_token
        base_url = self.LIVE_URL if live else self.DEMO_URL
        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        self._md_client = httpx.Client(
            base_url=self.MD_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    def _request(self, client: httpx.Client, method: str, path: str, **kwargs) -> Any:
        try:
            resp = client.request(method, path, **kwargs)
            if resp.status_code == 401:
                raise BrokerAuthError("Tradovate auth expired")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise BrokerAPIError(
                f"Tradovate HTTP {e.response.status_code}: {e.response.text}"
            )

    def get_account_info(self) -> Dict[str, Any]:
        accounts = self._request(self._client, "GET", "/account/list")
        if not accounts:
            return {}

        acct = accounts[0]
        return {
            "account_id": acct.get("id"),
            "name": acct.get("name", ""),
            "equity": acct.get("cashBalance", 0),
            "margin_used": acct.get("marginUsed", 0),
            "realized_pnl": acct.get("realizedPnl", 0),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        data = self._request(self._client, "GET", "/position/list")

        positions = []
        for p in (data if isinstance(data, list) else []):
            net_pos = p.get("netPos", 0)
            positions.append({
                "symbol": p.get("contractId", ""),
                "contract_id": p.get("contractId"),
                "quantity": abs(net_pos),
                "avg_price": float(p.get("netPrice", 0)),
                "pnl": float(p.get("openPnl", 0)),
                "side": "long" if net_pos > 0 else "short",
                "timestamp": p.get("timestamp"),
            })
        return positions

    def get_orders(
        self,
        from_date: Optional[date] = None,
        status: str = "filled",
    ) -> List[Dict[str, Any]]:
        data = self._request(self._client, "GET", "/order/list")

        orders = []
        for o in (data if isinstance(data, list) else []):
            order_status = o.get("ordStatus", "").lower()
            if status != "all" and order_status != status:
                continue

            orders.append({
                "order_id": str(o.get("id", "")),
                "symbol": str(o.get("contractId", "")),
                "side": o.get("action", "").lower(),
                "qty": int(o.get("filledQty", o.get("qty", 0))),
                "price": float(o.get("avgFillPrice", o.get("price", 0))),
                "filled_at": o.get("timestamp"),
                "status": order_status,
                "type": o.get("ordType", ""),
            })
        return orders

    def get_historical_candles(
        self,
        symbol: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> List[Dict[str, Any]]:
        # Tradovate uses elementSize + elementSizeUnit
        interval_map = {
            "1m": (1, "MinuteBar"), "5m": (5, "MinuteBar"),
            "15m": (15, "MinuteBar"), "30m": (30, "MinuteBar"),
            "1h": (60, "MinuteBar"), "1d": (1, "DailyBar"),
        }
        size, unit = interval_map.get(interval, (15, "MinuteBar"))

        params = {
            "symbol": symbol,
            "chartDescription": {
                "underlyingType": "MinuteBar" if "Minute" in unit else "DailyBar",
                "elementSize": size,
                "elementSizeUnit": unit,
            },
        }

        data = self._request(self._md_client, "GET", "/md/getChart", params=params)

        candles = []
        for bar in data.get("bars", []):
            candles.append({
                "timestamp": bar.get("timestamp", ""),
                "open": float(bar.get("open", 0)),
                "high": float(bar.get("high", 0)),
                "low": float(bar.get("low", 0)),
                "close": float(bar.get("close", 0)),
                "volume": int(bar.get("upVolume", 0) + bar.get("downVolume", 0)),
            })
        return candles

    def get_market_quote(self, symbol: str) -> Dict[str, Any]:
        data = self._request(
            self._md_client, "GET", "/md/getChart",
            params={"symbol": symbol},
        )
        bars = data.get("bars", [])
        if not bars:
            return {}

        latest = bars[-1]
        return {
            "ltp": latest.get("close"),
            "open": latest.get("open"),
            "high": latest.get("high"),
            "low": latest.get("low"),
            "volume": latest.get("upVolume", 0) + latest.get("downVolume", 0),
        }
