# tests/unit/test_zerodha_adapter.py
"""
Unit tests for the Zerodha/Kite Connect broker adapter.
Uses mocked Kite API responses — no real API calls.
"""

import pytest
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock


class TestZerodhaClient:
    """Tests for ZerodhaClient broker adapter."""

    @pytest.fixture
    def mock_kite(self):
        """Create a mock KiteConnect instance."""
        mock = MagicMock()
        mock.instruments.return_value = [
            {
                "instrument_token": 738561,
                "exchange_token": "2885",
                "tradingsymbol": "RELIANCE",
                "name": "RELIANCE INDUSTRIES",
                "exchange": "NSE",
                "segment": "NSE",
                "instrument_type": "EQ",
            },
            {
                "instrument_token": 256265,
                "exchange_token": "1001",
                "tradingsymbol": "NIFTY 50",
                "name": "NIFTY 50",
                "exchange": "NSE",
                "segment": "INDICES",
                "instrument_type": "EQ",
            },
        ]
        return mock

    @pytest.fixture
    def client(self, mock_kite):
        """Create a ZerodhaClient with mocked KiteConnect."""
        with patch.dict("os.environ", {
            "KITE_API_KEY": "test_api_key",
            "KITE_ACCESS_TOKEN": "test_access_token",
        }):
            with patch("ingestion.broker_adapters.zerodha.ZerodhaClient._init_client"):
                from ingestion.broker_adapters.zerodha import ZerodhaClient
                client = ZerodhaClient.__new__(ZerodhaClient)
                client._api_key = "test_api_key"
                client._access_token = "test_access_token"
                client._kite = mock_kite
                client._instrument_cache = {}
                client._last_request_time = 0.0
                client._min_request_interval = 0.0  # No throttle in tests
                return client

    def test_name(self, client):
        assert client.name == "zerodha"

    def test_instrument_token_resolution(self, client, mock_kite):
        """Test that symbol → instrument_token resolution works."""
        token = client.get_instrument_token("RELIANCE", "NSE")
        assert token == 738561
        mock_kite.instruments.assert_called_once_with("NSE")

    def test_instrument_token_caching(self, client, mock_kite):
        """Test that instrument tokens are cached after first resolution."""
        client.get_instrument_token("RELIANCE", "NSE")
        client.get_instrument_token("RELIANCE", "NSE")
        # Should only call instruments() once — second call hits cache
        mock_kite.instruments.assert_called_once()

    def test_instrument_not_found(self, client, mock_kite):
        """Test error when symbol is not found."""
        from ingestion.broker_api import BrokerAPIError
        with pytest.raises(BrokerAPIError, match="Instrument not found"):
            client.get_instrument_token("NONEXISTENT", "NSE")

    def test_get_historical_candles(self, client, mock_kite):
        """Test historical candle fetching and normalization."""
        mock_kite.historical_data.return_value = [
            {
                "date": datetime(2024, 1, 15, 9, 15, tzinfo=timezone.utc),
                "open": 2450.0,
                "high": 2465.0,
                "low": 2445.0,
                "close": 2460.0,
                "volume": 125000,
            },
            {
                "date": datetime(2024, 1, 15, 9, 20, tzinfo=timezone.utc),
                "open": 2460.0,
                "high": 2470.0,
                "low": 2455.0,
                "close": 2468.0,
                "volume": 98000,
            },
        ]

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
        assert candles[1]["close"] == 2468.0

    def test_get_positions(self, client, mock_kite):
        """Test positions fetching and normalization."""
        mock_kite.positions.return_value = {
            "net": [
                {
                    "tradingsymbol": "RELIANCE",
                    "exchange": "NSE",
                    "quantity": 100,
                    "average_price": 2450.0,
                    "pnl": 500.0,
                    "product": "CNC",
                    "last_price": 2455.0,
                },
                {
                    "tradingsymbol": "HDFCBANK",
                    "exchange": "NSE",
                    "quantity": 0,  # Flat — should be filtered out
                    "average_price": 1600.0,
                    "pnl": 0.0,
                    "product": "CNC",
                    "last_price": 1600.0,
                },
            ],
            "day": [],
        }

        positions = client.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "RELIANCE"
        assert positions[0]["quantity"] == 100
        assert positions[0]["side"] == "long"

    def test_get_orders(self, client, mock_kite):
        """Test order fetching with status filtering."""
        mock_kite.orders.return_value = [
            {
                "tradingsymbol": "RELIANCE",
                "transaction_type": "BUY",
                "filled_quantity": 100,
                "average_price": 2450.0,
                "order_timestamp": datetime(2024, 1, 15, 9, 30),
                "status": "COMPLETE",
                "order_id": "240115000001",
                "exchange": "NSE",
                "product": "CNC",
            },
            {
                "tradingsymbol": "HDFCBANK",
                "transaction_type": "SELL",
                "filled_quantity": 0,
                "average_price": 0,
                "order_timestamp": datetime(2024, 1, 15, 10, 0),
                "status": "CANCELLED",
                "order_id": "240115000002",
                "exchange": "NSE",
                "product": "MIS",
            },
        ]

        filled_orders = client.get_orders(status="filled")
        assert len(filled_orders) == 1
        assert filled_orders[0]["symbol"] == "RELIANCE"
        assert filled_orders[0]["side"] == "buy"

    def test_get_account_info(self, client, mock_kite):
        """Test account info (margins) fetching."""
        mock_kite.margins.return_value = {
            "equity": {
                "available": {
                    "live_balance": 500000.0,
                    "cash": 450000.0,
                    "collateral": 50000.0,
                },
                "utilised": {
                    "debits": 100000.0,
                },
            }
        }

        info = client.get_account_info()
        assert info["equity"] == 500000.0
        assert info["buying_power"] == 450000.0
        assert info["margin_used"] == 100000.0

    def test_get_market_quote(self, client, mock_kite):
        """Test market quote fetching."""
        mock_kite.quote.return_value = {
            "NSE:RELIANCE": {
                "last_price": 2460.0,
                "ohlc": {
                    "open": 2450.0,
                    "high": 2470.0,
                    "low": 2440.0,
                    "close": 2455.0,
                },
                "volume": 5000000,
                "oi": 0,
                "depth": {
                    "buy": [{"price": 2459.0, "quantity": 500}],
                    "sell": [{"price": 2461.0, "quantity": 300}],
                },
            }
        }

        quote = client.get_market_quote("RELIANCE")
        assert quote["ltp"] == 2460.0
        assert quote["open"] == 2450.0
        assert quote["bid"] == 2459.0
        assert quote["ask"] == 2461.0

    def test_detect_exchange_equity(self, client):
        """Test exchange detection for equity symbols."""
        from ingestion.broker_adapters.zerodha import ZerodhaClient
        assert ZerodhaClient._detect_exchange("RELIANCE") == "NSE"
        assert ZerodhaClient._detect_exchange("HDFCBANK") == "NSE"

    def test_detect_exchange_index(self, client):
        """Test exchange detection for index symbols."""
        from ingestion.broker_adapters.zerodha import ZerodhaClient
        assert ZerodhaClient._detect_exchange("NIFTY 50") == "NSE"
        assert ZerodhaClient._detect_exchange("NIFTY BANK") == "NSE"

    def test_detect_exchange_fno(self, client):
        """Test exchange detection for F&O symbols."""
        from ingestion.broker_adapters.zerodha import ZerodhaClient
        assert ZerodhaClient._detect_exchange("NIFTY24MARFUT") == "NFO"
        assert ZerodhaClient._detect_exchange("BANKNIFTY2411622000CE") == "NFO"

    def test_interval_resolution(self):
        """Test interval string mapping."""
        from ingestion.broker_adapters.zerodha import KiteInterval
        assert KiteInterval.resolve("5m") == "5minute"
        assert KiteInterval.resolve("15min") == "15minute"
        assert KiteInterval.resolve("1h") == "60minute"
        assert KiteInterval.resolve("1d") == "day"


class TestTimezoneConfig:
    """Verify timezone configuration is correctly applied."""

    def test_market_timezone_default(self):
        """Verify default market timezone is Asia/Kolkata."""
        from config.settings import Settings
        s = Settings(
            KITE_API_KEY=None,
            KITE_API_SECRET=None,
            KITE_ACCESS_TOKEN=None,
            _env_file=None,
        )
        assert s.MARKET_TIMEZONE == "Asia/Kolkata"

    def test_market_hours_default(self):
        """Verify default market open/close times are NSE hours."""
        from config.settings import Settings
        s = Settings(_env_file=None)
        assert s.MARKET_OPEN_TIME == "09:15"
        assert s.MARKET_CLOSE_TIME == "15:30"
