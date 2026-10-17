# ingestion/broker_adapters/__init__.py
"""
Broker-specific API adapters. Each adapter implements the
BrokerAdapter interface for read-only data access.
"""

from ingestion.broker_adapters.alpaca import AlpacaClient
from ingestion.broker_adapters.angel_one import AngelOneClient
from ingestion.broker_adapters.generic import GenericBrokerClient
from ingestion.broker_adapters.groww import GrowwClient
from ingestion.broker_adapters.interactive_brokers import IBKRClient
from ingestion.broker_adapters.thinkorswim import ThinkorswimClient
from ingestion.broker_adapters.tradovate import TradovateClient
from ingestion.broker_adapters.upstox import UpstoxClient
from ingestion.broker_adapters.zerodha import ZerodhaClient

__all__ = [
    "AlpacaClient",
    "AngelOneClient",
    "GenericBrokerClient",
    "GrowwClient",
    "IBKRClient",
    "ThinkorswimClient",
    "TradovateClient",
    "UpstoxClient",
    "ZerodhaClient",
]


