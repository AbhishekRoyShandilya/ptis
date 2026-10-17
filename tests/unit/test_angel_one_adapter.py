# tests/unit/test_angel_one_adapter.py
"""
Unit tests for the Angel One SmartAPI broker adapter.
Uses mocked SmartAPI responses — no real API calls.
"""

import pytest
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch


class TestAngelOneClient:
    """Tests for AngelOneClient broker adapter."""

    @pytest.fixture
    def mock_smart_api(self):
        """Create a mock SmartConnect instance."""
        mock = MagicMock()
        mock.generateSession.return_value = {
            "status": True,
            "data": {
                "jwtToken": "test_jwt_token",
                "feedToken": "test_feed_token",
            },
        }
        mock.searchScrip.return_value = {
            "status": True,
            "data": [
                {
                    "tradingsymbol": "RELIANCE",
                    "symboltoken": "2885",
                    "exchange": "NSE",
                },
            ],
        }
        return mock

    @pytest.fixture
    def client(self, mock_smart_api):
        """Create an AngelOneClient with mocked SmartConnect."""
        with patch("ingestion.broker_adapters.angel_one.AngelOneClient._init_client"):
            from ingestion.broker_adapters.angel_one import AngelOneClient
            client = AngelOneClient.__new__(AngelOneClient)
            client._api_key = "test_api_key"
            client._client_id = "test_client"
            client._password = "1234"
            client._totp_secret = None
            client._smart_api = mock_smart_api
            client._auth_token = "test_jwt"
            client._feed_token = "test_feed"
            client._last_request_time = 0.0
            client._min_request_interval = 0.0  # No throttle in tests
            client._token_map = {}
            return client

    def test_name(self, client):
        assert client.name == "angel_one"

    def test_symbol_token_resolution(self, client, mock_smart_api):
        """Test symbol → token resolution via searchScrip."""
        token = client.get_symbol_token("RELIANCE", "NSE")
        assert token == "2885"
        mock_smart_api.searchScrip.assert_called_once_with("NSE", "RELIANCE")

    def test_symbol_token_caching(self, client, mock_smart_api):
        """Test that token is cached after first lookup."""
        client.get_symbol_token("RELIANCE", "NSE")
        client.get_symbol_token("RELIANCE", "NSE")
        mock_smart_api.searchScrip.assert_called_once()  # Only called once

    def test_get_historical_candles(self, client, mock_smart_api):
        """Test historical candle fetching."""
        mock_smart_api.getCandleData.return_value = {
            "status": True,
            "data": [
                ["2024-01-15T09:15:00+05:30", 2450.0, 2465.0, 2445.0, 2460.0, 125000],
                ["2024-01-15T09:20:00+05:30", 2460.0, 2470.0, 2455.0, 2468.0, 98000],
            ],
        }

        candles = client.get_historical_candles(
            symbol="RELIANCE",
            interval="5m",
            from_date=date(2024, 1, 15),
            to_date=date(2024, 1, 15),
        )

        assert len(candles) == 2
        assert candles[0]["open"] == 2450.0
        assert candles[0]["close"] == 2460.0
        assert candles[0]["volume"] == 125000

    def test_get_positions(self, client, mock_smart_api):
        """Test positions retrieval with zero-qty filtering."""
        mock_smart_api.position.return_value = {
            "status": True,
            "data": [
                {
                    "tradingsymbol": "RELIANCE",
                    "exchange": "NSE",
                    "netqty": "100",
                    "averageprice": "2450.0",
                    "pnl": "500",
                    "producttype": "CNC",
                    "ltp": "2455.0",
                },
                {
                    "tradingsymbol": "HDFCBANK",
                    "exchange": "NSE",
                    "netqty": "0",
                    "averageprice": "1600.0",
                    "pnl": "0",
                    "producttype": "CNC",
                    "ltp": "1600.0",
                },
            ],
        }

        positions = client.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "RELIANCE"
        assert positions[0]["side"] == "long"

    def test_get_orders(self, client, mock_smart_api):
        """Test order book with status filtering."""
        mock_smart_api.orderBook.return_value = {
            "status": True,
            "data": [
                {
                    "tradingsymbol": "RELIANCE",
                    "transactiontype": "BUY",
                    "filledshares": "100",
                    "averageprice": "2450.0",
                    "updatetime": "15-Jan-2024 09:30:00",
                    "orderstatus": "complete",
                    "orderid": "240115000001",
                    "exchange": "NSE",
                    "producttype": "CNC",
                },
                {
                    "tradingsymbol": "INFY",
                    "transactiontype": "BUY",
                    "filledshares": "0",
                    "averageprice": "0",
                    "updatetime": "15-Jan-2024 10:00:00",
                    "orderstatus": "cancelled",
                    "orderid": "240115000002",
                    "exchange": "NSE",
                    "producttype": "MIS",
                },
            ],
        }

        filled = client.get_orders(status="filled")
        assert len(filled) == 1
        assert filled[0]["symbol"] == "RELIANCE"

    def test_get_account_info(self, client, mock_smart_api):
        """Test account margins."""
        mock_smart_api.rmsLimit.return_value = {
            "status": True,
            "data": {
                "net": "500000",
                "availablecash": "450000",
                "utiliseddebits": "50000",
                "collateral": "25000",
            },
        }

        info = client.get_account_info()
        assert info["equity"] == 500000.0
        assert info["buying_power"] == 450000.0
        assert info["margin_used"] == 50000.0

    def test_interval_resolution(self):
        """Test interval mapping."""
        from ingestion.broker_adapters.angel_one import AngelInterval
        assert AngelInterval.resolve("5m") == "FIVE_MINUTE"
        assert AngelInterval.resolve("1h") == "ONE_HOUR"
        assert AngelInterval.resolve("1d") == "ONE_DAY"

    def test_detect_exchange(self, client):
        """Test exchange detection."""
        from ingestion.broker_adapters.angel_one import AngelOneClient
        assert AngelOneClient._detect_exchange("RELIANCE") == "NSE"
        assert AngelOneClient._detect_exchange("NIFTY24MARFUT") == "NFO"
