# shadow/price_adapters/__init__.py
"""
Price feed adapters for the shadow trading engine.

Each adapter implements a common interface and feeds PriceTick
objects into PriceFeedManager via its dispatch mechanism.
"""

from shadow.price_adapters.manual_injector import ManualPriceInjector
from shadow.price_adapters.yahoo_polling import YahooPollingAdapter
from shadow.price_adapters.alpaca_ws import AlpacaWebSocketAdapter
from shadow.price_adapters.polygon_ws import PolygonWebSocketAdapter
from shadow.price_adapters.nse_kite import NseKiteAdapter

__all__ = [
    "ManualPriceInjector",
    "YahooPollingAdapter",
    "AlpacaWebSocketAdapter",
    "PolygonWebSocketAdapter",
    "NseKiteAdapter",
]

