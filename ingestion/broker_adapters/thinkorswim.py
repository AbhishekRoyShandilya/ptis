# ingestion/broker_adapters/thinkorswim.py
"""
TD Ameritrade / Schwab (ThinkorSwim) API adapter.
Read-only access to account data, positions, orders, and market data.

Note: TD Ameritrade API transitioned to Schwab. This adapter supports
both the legacy TD API and the new Schwab API patterns.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import httpx

from ingestion.broker_api import BrokerAdapter, BrokerAPIError, BrokerAuthError

logger = logging.getLogger(__name__)


class ThinkorswimClient(BrokerAdapter):
    """Read-only ThinkorSwim / Schwab API client."""

    name = "thinkorswim"

    BASE_URL = "https://api.schwabapi.com/trader/v1"
    MARKET_URL = "https://api.schwabapi.com/marketdata/v1"

    def __init__(
        self,
        access_token: str,
        account_hash: Optional[str] = None,
    ):
        self._access_token = access_token
        self._account_hash = account_hash
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    def _request(self, method: str, url: str, **kwargs) -> Any:
        try:
            resp = self._client.request(method, url, **kwargs)
            if resp.status_code == 401:
                raise BrokerAuthError("ThinkorSwim auth expired — refresh token")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise BrokerAPIError(
                f"ThinkorSwim HTTP {e.response.status_code}: {e.response.text}"
            )

    def _get_account_hash(self) -> str:
        if self._account_hash:
            return self._account_hash
        data = self._request("GET", f"{self.BASE_URL}/accounts/accountNumbers")
        if not data:
            raise BrokerAPIError("No ThinkorSwim accounts found")
        self._account_hash = data[0].get("hashValue", "")
        return self._account_hash

    def get_account_info(self) -> Dict[str, Any]:
        acct = self._get_account_hash()
        data = self._request("GET", f"{self.BASE_URL}/accounts/{acct}")

        balances = data.get("securitiesAccount", {}).get("currentBalances", {})
        return {
            "equity": balances.get("equity", 0),
            "buying_power": balances.get("buyingPower", 0),
            "cash": balances.get("cashBalance", 0),
            "margin_used": balances.get("maintenanceRequirement", 0),
            "day_trade_buying_power": balances.get("dayTradingBuyingPower", 0),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        acct = self._get_account_hash()
        data = self._request(
            "GET", f"{self.BASE_URL}/accounts/{acct}",
            params={"fields": "positions"},
        )

        raw_positions = (
            data.get("securitiesAccount", {}).get("positions", [])
        )
        positions = []
        for p in raw_positions:
            inst = p.get("instrument", {})
            positions.append({
                "symbol": inst.get("symbol", ""),
                "quantity": float(p.get("longQuantity", 0) - p.get("shortQuantity", 0)),
                "avg_price": float(p.get("averagePrice", 0)),
                "current_price": float(p.get("marketValue", 0)) / max(
                    abs(p.get("longQuantity", 0) - p.get("shortQuantity", 0)), 1
                ),
                "pnl": float(p.get("currentDayProfitLoss", 0)),
                "pnl_pct": float(p.get("currentDayProfitLossPercentage", 0)),
                "side": "long" if p.get("longQuantity", 0) > 0 else "short",
                "asset_type": inst.get("assetType", "EQUITY"),
            })
        return positions

    def get_orders(
        self,
        from_date: Optional[date] = None,
        status: str = "filled",
    ) -> List[Dict[str, Any]]:
        acct = self._get_account_hash()
        params: Dict[str, Any] = {"maxResults": 500}
        if from_date:
            params["fromEnteredTime"] = from_date.isoformat()
        if status != "all":
            params["status"] = status.upper()

        data = self._request(
            "GET", f"{self.BASE_URL}/accounts/{acct}/orders",
            params=params,
        )

        orders = []
        for o in (data if isinstance(data, list) else []):
            legs = o.get("orderLegCollection", [])
            for leg in legs:
                inst = leg.get("instrument", {})
                orders.append({
                    "order_id": str(o.get("orderId", "")),
                    "symbol": inst.get("symbol", ""),
                    "side": leg.get("instruction", "").lower(),
                    "qty": float(o.get("filledQuantity", 0)),
                    "price": float(o.get("price", 0)),
                    "filled_at": o.get("closeTime"),
                    "status": o.get("status", "").lower(),
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
        freq_map = {
            "1m": ("minute", 1), "5m": ("minute", 5),
            "15m": ("minute", 15), "30m": ("minute", 30),
            "1d": ("daily", 1), "day": ("daily", 1),
            "1w": ("weekly", 1),
        }
        freq_type, freq = freq_map.get(interval, ("minute", 15))

        params = {
            "periodType": "day" if freq_type == "minute" else "year",
            "frequencyType": freq_type,
            "frequency": freq,
            "startDate": int(datetime.combine(from_date, datetime.min.time()).timestamp() * 1000),
            "endDate": int(datetime.combine(to_date, datetime.min.time()).timestamp() * 1000),
        }

        data = self._request(
            "GET", f"{self.MARKET_URL}/pricehistory",
            params={"symbol": symbol, **params},
        )

        candles = []
        for c in data.get("candles", []):
            candles.append({
                "timestamp": datetime.fromtimestamp(c["datetime"] / 1000).isoformat(),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": int(c.get("volume", 0)),
            })
        return candles

    def get_market_quote(self, symbol: str) -> Dict[str, Any]:
        data = self._request(
            "GET", f"{self.MARKET_URL}/quotes",
            params={"symbols": symbol},
        )
        quote = data.get(symbol, {}).get("quote", {})
        return {
            "ltp": quote.get("lastPrice"),
            "bid": quote.get("bidPrice"),
            "ask": quote.get("askPrice"),
            "volume": quote.get("totalVolume"),
            "open": quote.get("openPrice"),
            "high": quote.get("highPrice"),
            "low": quote.get("lowPrice"),
            "prev_close": quote.get("closePrice"),
        }
