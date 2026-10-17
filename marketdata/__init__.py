"""
marketdata — local OHLCV lake (Parquet + DuckDB) behind a single CandleStore
read-through interface.

Phase 1: equity + index, 1-minute (Jan 2022+) and daily (2000+), provider =
Upstox v3 primary with Angel One / yfinance fallback. Schema is options-ready
(see project_ohlcv_local_store memory + plan).

Public surface:
  get_candle_store() -> CandleStore
  CandleStore.get(symbol, interval, start, end) -> pandas.DataFrame
"""
from marketdata.candle_store import CandleStore, get_candle_store

__all__ = ["CandleStore", "get_candle_store"]
