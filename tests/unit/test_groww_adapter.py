# tests/unit/test_groww_adapter.py
"""
Unit tests for the Groww Trading API broker adapter.
Uses mocked Groww API responses — no real API calls.
"""

import pytest
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch


class TestGrowwClient:
    """Tests for GrowwClient broker adapter."""

    @pytest.fixture
    def mock_groww_api(self):
        """Create a mock GrowwAPI instance."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def client(self, mock_groww_api):
        """A GrowwClient built through its real constructor, with the SDK client replaced by a mock."""
        from ingestion.broker_adapters.groww import GrowwClient
        with patch.object(GrowwClient, "_init_client"):
            client = GrowwClient(api_key="test_api_key", api_secret="test_api_secret",
                                 access_token="test_access_token")
        client._totp_secret = None          # static-token mode: never re-mint a token during tests
        client._client = mock_groww_api
        client._min_request_interval = 0.0  # no throttle in tests
        return client

    def test_name(self, client):
        assert client.name == "groww"

    def test_get_historical_candles(self, client, mock_groww_api):
        """Test historical candle fetching."""
        mock_groww_api.get_historical_candle_data.return_value = {
            "candles": [
                {
                    "timestamp": "2024-01-15T09:15:00+05:30",
                    "open": 2450.0,
                    "high": 2465.0,
                    "low": 2445.0,
                    "close": 2460.0,
                    "volume": 125000,
                },
                {
                    "timestamp": "2024-01-15T09:20:00+05:30",
                    "open": 2460.0,
                    "high": 2470.0,
                    "low": 2455.0,
                    "close": 2468.0,
                    "volume": 98000,
                },
            ]
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
        kwargs = mock_groww_api.get_historical_candle_data.call_args.kwargs
        assert kwargs["segment"] == "CASH" and kwargs["interval_in_minutes"] == 5

    def test_get_historical_candles_array_rows(self, client, mock_groww_api):
        """Groww also returns candles as [ts, open, high, low, close, volume] arrays."""
        mock_groww_api.get_historical_candle_data.return_value = {
            "payload": {"candles": [[1705290300, 2450.0, 2465.0, 2445.0, 2460.0, 125000]]}
        }
        candles = client.get_historical_candles("RELIANCE", "5m", date(2024, 1, 15), date(2024, 1, 15))
        assert len(candles) == 1
        assert (candles[0]["high"], candles[0]["low"], candles[0]["volume"]) == (2465.0, 2445.0, 125000)

    def test_get_positions(self, client, mock_groww_api):
        """Test positions with zero-qty filtering."""
        cash_positions = {
            "positions": [
                {
                    "tradingSymbol": "RELIANCE",
                    "exchange": "NSE",
                    "quantity": 100,
                    "averagePrice": 2450.0,
                    "pnl": 500.0,
                    "product": "CNC",
                    "ltp": 2455.0,
                },
                {
                    "tradingSymbol": "HDFCBANK",
                    "exchange": "NSE",
                    "quantity": 0,
                    "averagePrice": 1600.0,
                    "pnl": 0,
                    "product": "CNC",
                    "ltp": 1600.0,
                },
            ]
        }
        mock_groww_api.get_positions_for_user.side_effect = (
            lambda segment: cash_positions if segment == "CASH" else {"positions": []})

        positions = client.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "RELIANCE"
        assert positions[0]["side"] == "long"

    def test_get_orders(self, client, mock_groww_api):
        """Test order book with status filtering."""
        cash_orders = {
            "order_list": [
                {
                    "tradingSymbol": "RELIANCE",
                    "transactionType": "BUY",
                    "filledQuantity": 100,
                    "averagePrice": 2450.0,
                    "orderTimestamp": "2024-01-15T09:30:00+05:30",
                    "status": "complete",
                    "orderId": "GRW240115001",
                    "exchange": "NSE",
                    "product": "CNC",
                },
                {
                    "tradingSymbol": "INFY",
                    "transactionType": "SELL",
                    "filledQuantity": 0,
                    "averagePrice": 0,
                    "orderTimestamp": "2024-01-15T10:00:00+05:30",
                    "status": "cancelled",
                    "orderId": "GRW240115002",
                    "exchange": "NSE",
                    "product": "MIS",
                },
            ]
        }
        mock_groww_api.get_order_list.side_effect = (
            lambda page, page_size, segment: cash_orders if segment == "CASH" else {"order_list": []})

        filled = client.get_orders(status="filled")
        assert len(filled) == 1
        assert filled[0]["symbol"] == "RELIANCE"
        assert filled[0]["side"] == "buy"

    def test_get_account_info(self, client, mock_groww_api):
        """Test account funds."""
        mock_groww_api.get_user_profile.return_value = {"ucc": "AB1234", "vendor_user_id": "AB1234"}
        mock_groww_api.get_available_margin_details.return_value = {
            "clear_cash": 400000.0,
            "collateral_available": 100000.0,
            "net_margin_used": 50000.0,
        }

        info = client.get_account_info()
        assert info["equity"] == 500000.0          # cash + collateral
        assert info["buying_power"] == 450000.0    # equity - margin used
        assert info["margin_used"] == 50000.0
        assert info["user_id"] == "AB1234"

    def test_get_market_quote(self, client, mock_groww_api):
        """Test market quote."""
        mock_groww_api.get_quote.return_value = {
            "ltp": 2460.0,
            "open": 2450.0,
            "high": 2470.0,
            "low": 2440.0,
            "close": 2455.0,
            "previousClose": 2445.0,
            "volume": 5000000,
            "oi": 0,
            "bestBid": 2459.0,
            "bestAsk": 2461.0,
        }

        quote = client.get_market_quote("RELIANCE")
        assert quote["ltp"] == 2460.0
        assert quote["open"] == 2450.0
        assert quote["bid"] == 2459.0
        assert quote["ask"] == 2461.0

    def test_get_ltp(self, client, mock_groww_api):
        """Test LTP fetch."""
        mock_groww_api.get_ltp.return_value = {"ltp": 2460.0}

        ltp = client.get_ltp("RELIANCE")
        assert ltp == 2460.0

    def test_interval_resolution(self):
        """Test interval mapping."""
        from ingestion.broker_adapters.groww import GrowwClient
        assert GrowwClient._interval_minutes("5m") == 5
        assert GrowwClient._interval_minutes("1h") == 60
        assert GrowwClient._interval_minutes("60m") == 60
        assert GrowwClient._interval_minutes("1D") == 1440
        assert GrowwClient._interval_minutes("unknown") == 5     # safe default

    def test_detect_exchange(self, client):
        """Test exchange detection."""
        from ingestion.broker_adapters.groww import GrowwClient
        assert GrowwClient._detect_exchange("RELIANCE") == "NSE"
        assert GrowwClient._detect_exchange("NIFTY24MARFUT") == "NFO"

    def test_empty_positions(self, client, mock_groww_api):
        """Test empty positions response."""
        mock_groww_api.get_positions_for_user.return_value = None
        positions = client.get_positions()
        assert positions == []

    def test_empty_orders(self, client, mock_groww_api):
        """Test empty orders response."""
        mock_groww_api.get_order_list.return_value = None
        orders = client.get_orders()
        assert orders == []
