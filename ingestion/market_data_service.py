# ingestion/market_data_service.py
"""
Central Market Data Service — dual-source: Angel One + Upstox.

Broker roles:
  Angel One  → Historical OHLCV candles (discovery, backtesting, scanner patterns, regime)
  Upstox     → Live market data (LTP, quotes, intraday) for AVT, shadow, coaching, watchlist
  Groww      → Trade sync only (handled by ingestion/trade_sync.py, not here)

Consumer adapters:
  - SetupScanner:  fetch_candles_for_scanner(symbol, n_bars) → list[dict]   (Angel One historical)
  - Discovery:     fetch_candles_for_discovery(symbol, tf, start, end) → DataFrame  (Angel One historical)
  - Intelligence:  as_intelligence_provider() → MarketDataProvider  (Upstox live + Angel One historical)
  - AVT:           Separate UpstoxMarketData class in avt/market_data.py    (Upstox live)

Session management:
  - Angel One: TOTP sessions last ~24h, _ensure_angel_session() re-auths after 8h
  - Upstox: access_token based, no auto-refresh (user provides long-lived token)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Session refresh interval (hours) — Angel One sessions last ~24h, refresh at 8h
_SESSION_REFRESH_HOURS = 8


def _aggregate_1m_to_nm(bars_1m: List[Dict[str, Any]], n_minutes: int) -> List[Dict[str, Any]]:
    """
    Aggregate 1-minute OHLCV bars into N-minute bars.
    Groups by flooring each timestamp to the nearest N-minute boundary.
    """
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for bar in bars_1m:
        ts = bar.get("timestamp")
        if ts is None:
            continue
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        floored = ts.replace(second=0, microsecond=0)
        floored = floored - timedelta(minutes=floored.minute % n_minutes)
        buckets[floored].append(bar)

    result = []
    for ts_key in sorted(buckets.keys()):
        group = buckets[ts_key]
        result.append({
            "timestamp": ts_key,
            "open":   group[0]["open"],
            "high":   max(b["high"] for b in group),
            "low":    min(b["low"]  for b in group),
            "close":  group[-1]["close"],
            "volume": sum(b.get("volume", 0) for b in group),
        })
    return result


# Keep legacy alias for any existing callers
def _aggregate_1m_to_5m(bars_1m: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _aggregate_1m_to_nm(bars_1m, 5)


# ── yfinance index ticker map ──────────────────────────────────────────────────
# Used as fallback when broker APIs can't fetch BSE/NSE index historical candles.
_YFINANCE_INDEX_MAP: dict[str, str] = {
    "NIFTY":       "^NSEI",
    "NIFTY 50":    "^NSEI",
    "NIFTY50":     "^NSEI",
    "BANKNIFTY":   "^NSEBANK",
    "NIFTY BANK":  "^NSEBANK",
    "FINNIFTY":    "NIFTY_FIN_SERVICE.NS",
    "SENSEX":      "^BSESN",
    "BSE SENSEX":  "^BSESN",
    "BANKEX":      "BANKEX.BO",
    "INDIA VIX":   "^INDIAVIX",
    "VIX":         "^INDIAVIX",
    "MIDCPNIFTY":  "NIFTY_MIDCAP_SELECT.NS",
}

_YF_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m", "1d": "1d", "D": "1d",
}


def fetch_index_candles_yfinance(
    symbol: str,
    interval: str,
    from_date: date,
    to_date: date,
) -> List[Dict[str, Any]]:
    """
    Fetch index OHLCV candles via yfinance as a last-resort fallback.

    Covers SENSEX (^BSESN), NIFTY (^NSEI), BANKNIFTY (^NSEBANK), VIX (^INDIAVIX).
    Returns empty list if yfinance is not installed or the ticker is unknown.
    Volume is 0 for indices (spot index data has no volume).
    """
    ticker = _YFINANCE_INDEX_MAP.get(symbol.upper())
    if not ticker:
        return []

    try:
        import yfinance as yf
    except ImportError:
        logger.debug("yfinance not installed — index candle fallback unavailable")
        return []

    yf_interval = _YF_INTERVAL_MAP.get(interval, "15m")
    try:
        df = yf.download(
            ticker,
            start=from_date.isoformat(),
            end=(to_date + timedelta(days=1)).isoformat(),
            interval=yf_interval,
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            logger.debug("yfinance returned empty data for %s (%s)", ticker, yf_interval)
            return []

        candles = []
        for ts, row in df.iterrows():
            # Handle both tz-aware and tz-naive timestamps
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            candles.append({
                "timestamp": ts,
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": 0,   # spot index has no volume
            })
        logger.info("yfinance fallback: %d bars for %s (%s)", len(candles), symbol, ticker)
        return candles
    except Exception as e:
        logger.warning("yfinance fetch failed for %s (%s): %s", symbol, ticker, e)
        return []


class MarketDataService:
    """
    Singleton service providing market data from multiple sources:
      - angel_client   → historical candles (get_historical_candles) + portfolio
      - upstox_client  → live data (get_ltp, get_market_quote, get_intraday_candles)
      - zerodha_client → optional alternative for historical + portfolio

    Any client can be None — the service degrades gracefully.
    """

    def __init__(self, angel_client=None, upstox_client=None, zerodha_client=None):
        self._angel = angel_client
        self._upstox = upstox_client
        self._zerodha = zerodha_client
        self._angel_last_login: Optional[datetime] = None

    @property
    def has_historical(self) -> bool:
        # Historical candles now come through the OHLCV lake (CandleStore →
        # Upstox v3), not just Angel One / Zerodha. So historical is available
        # whenever ANY provider is configured — including an Upstox-only setup.
        return (
            self._angel is not None
            or self._zerodha is not None
            or self._upstox is not None
        )

    @property
    def has_live(self) -> bool:
        return self._upstox is not None

    # ══════════════════════════════════════════════════════════════
    #  Angel One Session Management
    # ══════════════════════════════════════════════════════════════

    def _ensure_angel_session(self) -> bool:
        """Re-authenticate Angel One if session > 8h old."""
        if not self._angel:
            return False

        now = datetime.now(timezone.utc)
        if self._angel_last_login and (now - self._angel_last_login).total_seconds() < _SESSION_REFRESH_HOURS * 3600:
            return True

        try:
            self._angel._login()
            self._angel_last_login = now
            logger.info("Angel One session refreshed")
            return True
        except Exception as e:
            logger.error("Angel One session refresh failed: %s", e)
            return False

    # ══════════════════════════════════════════════════════════════
    #  Historical Candles (Angel One)
    # ══════════════════════════════════════════════════════════════

    def _fetch_candles_raw(
        self,
        symbol: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical candles through the local OHLCV lake (CandleStore):
        lake-first, fetch-and-cache on miss (Upstox v3 → Angel/Zerodha/yfinance).
        Falls back to the legacy direct fetchers if the lake layer errors.
        """
        try:
            from marketdata.candle_store import get_candle_store
            df = get_candle_store().get(symbol, interval, from_date, to_date)
            if df is not None and not df.empty:
                out: List[Dict[str, Any]] = []
                for _, r in df.iterrows():
                    ts = r["ts"]
                    out.append({
                        "timestamp": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                        "volume": int(r["volume"] or 0),
                        "oi": int(r["oi"] or 0),
                    })
                return out
        except Exception as e:
            logger.warning("CandleStore fetch failed for %s (%s) — falling back: %s", symbol, interval, e)

        # Legacy direct fallback (CandleStore unavailable / errored)
        if self._angel:
            return self._fetch_candles_angel(symbol, interval, from_date, to_date)
        if self._zerodha:
            return self._fetch_candles_zerodha(symbol, interval, from_date, to_date)
        logger.debug("No historical data client — cannot fetch candles")
        return []

    def _fetch_candles_angel(
        self, symbol: str, interval: str, from_date: date, to_date: date,
    ) -> List[Dict[str, Any]]:
        """Fetch historical candles from Angel One with auto-chunking."""
        from ingestion.broker_adapters.angel_one import AngelInterval

        self._ensure_angel_session()

        angel_interval = AngelInterval.resolve(interval)
        max_days = AngelInterval.max_days(angel_interval)

        all_candles: List[Dict[str, Any]] = []
        chunk_start = from_date

        while chunk_start <= to_date:
            chunk_end = min(chunk_start + timedelta(days=max_days - 1), to_date)
            try:
                candles = self._angel.get_historical_candles(
                    symbol=symbol,
                    interval=interval,
                    from_date=chunk_start,
                    to_date=chunk_end,
                )
                all_candles.extend(candles)
            except Exception as e:
                logger.warning(
                    "Angel One candle fetch failed for %s (%s→%s): %s",
                    symbol, chunk_start, chunk_end, e,
                )
            chunk_start = chunk_end + timedelta(days=1)

        return all_candles

    def _fetch_candles_zerodha(
        self, symbol: str, interval: str, from_date: date, to_date: date,
    ) -> List[Dict[str, Any]]:
        """Fetch historical candles from Zerodha with auto-chunking."""
        from ingestion.broker_adapters.zerodha import KiteInterval

        kite_interval = KiteInterval.resolve(interval)

        # Chunk sizes for Kite
        chunk_days_map = {
            "minute": 60, "3minute": 60, "5minute": 100, "10minute": 100,
            "15minute": 200, "30minute": 400, "60minute": 400,
            "day": 2000, "week": 2000, "month": 2000,
        }
        chunk_days = chunk_days_map.get(kite_interval, 100)

        all_candles: List[Dict[str, Any]] = []
        chunk_start = from_date

        while chunk_start <= to_date:
            chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), to_date)
            try:
                candles = self._zerodha.get_historical_candles(
                    symbol=symbol,
                    interval=interval,
                    from_date=chunk_start,
                    to_date=chunk_end,
                )
                all_candles.extend(candles)
            except Exception as e:
                logger.warning(
                    "Zerodha candle fetch failed for %s (%s→%s): %s",
                    symbol, chunk_start, chunk_end, e,
                )
            chunk_start = chunk_end + timedelta(days=1)

        return all_candles

    # ══════════════════════════════════════════════════════════════
    #  Live Market Data (Upstox)
    # ══════════════════════════════════════════════════════════════

    def _symbol_to_instrument_key(self, symbol: str) -> str:
        """Convert plain symbol (e.g. 'RELIANCE') to Upstox ISIN-based instrument key."""
        if self._upstox:
            return self._upstox.resolve_instrument_key(symbol)
        return f"NSE_EQ|{symbol.upper()}"

    def get_ltp(self, symbol: str) -> Optional[float]:
        """Get last traded price. Prefers Upstox (live), falls back to Angel One/Zerodha."""
        # Try Upstox first (live)
        if self._upstox:
            try:
                # If the caller already passed a valid Upstox instrument_key (e.g. "NSE_FO|51377" for an option),
                # use it DIRECTLY — running it through _symbol_to_instrument_key mangles it into an invalid key
                # (UDAPI1087). Only resolve when given a plain tradingsymbol.
                key = symbol if "|" in str(symbol) else self._symbol_to_instrument_key(symbol)
                ltp = self._upstox.get_ltp(key)
                if ltp and ltp > 0:
                    return ltp
            except Exception as e:
                logger.warning("Upstox LTP failed for %s: %s", symbol, e)

        # Fallback: Angel One
        if self._angel:
            try:
                return self._angel.get_ltp(symbol)
            except Exception as e:
                logger.debug("Angel One LTP fallback failed for %s: %s", symbol, e)

        # Fallback: Zerodha
        if self._zerodha:
            try:
                return self._zerodha.get_ltp(symbol)
            except Exception as e:
                logger.debug("Zerodha LTP fallback failed for %s: %s", symbol, e)

        return None

    def get_market_quote(self, symbol: str) -> Dict[str, Any]:
        """Get full market quote. Prefers Upstox (live), falls back to Angel One/Zerodha."""
        # Try Upstox first (live)
        if self._upstox:
            try:
                key = self._symbol_to_instrument_key(symbol)
                quote = self._upstox.get_market_quote(key)
                if quote.get("ltp"):
                    return quote
            except Exception as e:
                logger.warning("Upstox quote failed for %s: %s", symbol, e)

        # Fallback: Angel One
        if self._angel:
            try:
                return self._angel.get_market_quote(symbol)
            except Exception as e:
                logger.debug("Angel One quote fallback failed for %s: %s", symbol, e)

        # Fallback: Zerodha
        if self._zerodha:
            try:
                return self._zerodha.get_market_quote(symbol)
            except Exception as e:
                logger.debug("Zerodha quote fallback failed for %s: %s", symbol, e)

        return {"ltp": None}

    def get_intraday_candles(self, symbol: str, interval: str = "5m") -> List[Dict[str, Any]]:
        """Get today's intraday candles via Upstox (live)."""
        if not self._upstox:
            return []
        try:
            from ingestion.broker_adapters.upstox import UpstoxInterval
            key = self._symbol_to_instrument_key(symbol)
            interval_map = {
                "1m": UpstoxInterval.ONE_MIN,
                "5m": UpstoxInterval.FIVE_MIN,
                "15m": UpstoxInterval.FIFTEEN_MIN,
                "30m": UpstoxInterval.THIRTY_MIN,
                "1h": UpstoxInterval.SIXTY_MIN,
            }
            upstox_interval = interval_map.get(interval, UpstoxInterval.FIVE_MIN)
            return self._upstox.get_intraday_candles(key, upstox_interval)
        except Exception as e:
            logger.warning("Upstox intraday failed for %s: %s", symbol, e)
            return []

    # ══════════════════════════════════════════════════════════════
    #  Live-bar merge helper
    # ══════════════════════════════════════════════════════════════

    def fetch_candles_with_live_bar(
        self,
        symbol: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical candles AND append/replace today's bars with Upstox
        intraday data — which includes the CURRENT in-progress candle.

        This makes regime detection, VWAP, and strategy conditions accurate to
        the current tick rather than the last closed bar.

        Merge strategy:
          1. Fetch historical candles (Angel One / Upstox historical)
          2. If market is open + Upstox available, fetch intraday candles
          3. Remove historical bars whose date == today (they may be incomplete)
          4. Append today's intraday bars (last bar = current live candle)
          5. Return sorted by timestamp

        For BSE indices (SENSEX) where Upstox intraday isn't supported,
        this falls back gracefully to historical-only data.
        """
        import pytz

        # Step 1: historical candles
        candles = self._fetch_candles_raw(symbol, interval, from_date, to_date)

        # Step 2: only merge live bar during market hours (NSE: Mon-Fri 09:15–15:30 IST)
        if not self._upstox:
            return candles

        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)
        is_market_open = (
            now_ist.weekday() < 5
            and (9, 15) <= (now_ist.hour, now_ist.minute) <= (15, 30)
        )
        if not is_market_open:
            return candles  # market closed — historical is already complete

        # Step 3: fetch today's intraday bars (includes current in-progress candle)
        try:
            intraday = self.get_intraday_candles(symbol, interval)
        except Exception as e:
            logger.debug("Live bar fetch skipped for %s: %s", symbol, e)
            return candles

        if not intraday:
            return candles

        # Step 4: remove historical bars from today, replace with intraday data
        today = date.today()
        historical_prior = []
        for c in candles:
            ts = c.get("timestamp")
            bar_date = ts.date() if isinstance(ts, datetime) else today
            if bar_date < today:
                historical_prior.append(c)
            # today's historical bars are discarded — intraday is fresher

        merged = historical_prior + intraday

        # Step 5: sort by timestamp
        def _ts_key(c):
            ts = c.get("timestamp")
            if ts is None:
                return datetime.min
            if isinstance(ts, datetime):
                return ts.replace(tzinfo=None) if ts.tzinfo else ts
            return ts

        merged.sort(key=_ts_key)
        logger.debug(
            "Live-bar merge for %s: %d historical + %d intraday = %d total bars",
            symbol, len(historical_prior), len(intraday), len(merged),
        )
        return merged

    # ══════════════════════════════════════════════════════════════
    #  Consumer Adapters
    # ══════════════════════════════════════════════════════════════

    # ── SetupScanner (CandleFetchFn = Callable[[str, int, str], list[dict]])
    def fetch_candles_for_scanner(
        self, symbol: str, n_bars: int, interval: str = "15m"
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent candles for the setup scanner at the requested interval.
        Uses Angel One historical. Falls back to Upstox intraday aggregation
        for 5m/1m (Upstox intraday only supports 1m and 30m natively).

        interval: "5m" | "15m" | "30m" | "1h" | "1d" etc.
        """
        # How many calendar days of history we need given bars-per-day estimates
        bars_per_day = {
            "1m": 375, "3m": 125, "5m": 75, "10m": 38,
            "15m": 25, "30m": 13, "1h": 6, "4h": 2, "1d": 1,
        }
        bpd = bars_per_day.get(interval.lower().replace("min", "m").replace("minute", "m"), 25)
        days_needed = max(2, (n_bars // bpd) + 3)

        to_date   = date.today()
        from_date = to_date - timedelta(days=days_needed)

        candles = self._fetch_candles_raw(symbol, interval, from_date, to_date)

        # Supplement with Upstox intraday for sub-30m intervals.
        # Upstox intraday endpoint only supports 1m and 30m natively —
        # fetch 1m bars and aggregate to the requested interval.
        if self._upstox and len(candles) < n_bars and interval in ("1m", "3m", "5m", "10m", "15m"):
            try:
                intraday_1m = self.get_intraday_candles(symbol, "1m")
                if intraday_1m:
                    if interval == "1m":
                        intraday_agg = intraday_1m
                    else:
                        # Use a generic aggregator for any N-minute target
                        minutes = int(
                            interval.lower()
                            .replace("min", "")
                            .replace("minute", "")
                            .replace("m", "")
                        )
                        intraday_agg = _aggregate_1m_to_nm(intraday_1m, minutes)
                    existing_ts = {str(c.get("timestamp")) for c in candles}
                    for bar in intraday_agg:
                        if str(bar.get("timestamp")) not in existing_ts:
                            candles.append(bar)
            except Exception:
                pass

        return candles[-n_bars:] if len(candles) > n_bars else candles

    # ── Discovery (CandleFetcher = Callable[[str, str, datetime, datetime], pd.DataFrame])
    def fetch_candles_for_discovery(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ):
        """Fetch candles for discovery pipeline. Returns a pandas DataFrame."""
        import pandas as pd

        candles = self._fetch_candles_raw(
            symbol=symbol,
            interval=timeframe,
            from_date=start.date() if isinstance(start, datetime) else start,
            to_date=end.date() if isinstance(end, datetime) else end,
        )

        if not candles:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles)
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "datetime"})
        return df

    # ── Intelligence (MarketDataProvider)
    def as_intelligence_provider(self) -> "LiveMarketDataProvider":
        """Return a MarketDataProvider for the Intelligence module."""
        return LiveMarketDataProvider(self)


# ══════════════════════════════════════════════════════════════════
#  Intelligence MarketDataProvider — uses both sources
# ══════════════════════════════════════════════════════════════════

class LiveMarketDataProvider:
    """
    Wraps MarketDataService into the intelligence/market_context.py
    MarketDataProvider interface.

    get_price_data → Angel One historical
    get_market_internals → Upstox live (VIX via India VIX LTP)
    """

    def __init__(self, service: MarketDataService):
        self._svc = service

    def get_price_data(
        self,
        symbol: str,
        timestamp: datetime,
        lookback_bars: int = 100,
    ) -> Optional[List[Dict[str, Any]]]:
        """Historical price data for context (Angel One)."""
        days = max(5, (lookback_bars // 75) + 3)
        to_date = timestamp.date() if isinstance(timestamp, datetime) else date.today()
        from_date = to_date - timedelta(days=days)

        candles = self._svc._fetch_candles_raw(symbol, "5m", from_date, to_date)

        if not candles:
            return None

        # Return newest first, trimmed to lookback_bars
        candles.reverse()
        return candles[:lookback_bars]

    def get_market_internals(self, timestamp: datetime) -> Dict[str, Any]:
        """Market-wide indicators via Upstox live (India VIX)."""
        result: Dict[str, Any] = {
            "vix_level": None,
            "market_breadth": None,
            "sector_performance": None,
        }

        try:
            vix = self._svc.get_ltp("INDIA VIX")
            if vix:
                result["vix_level"] = vix
        except Exception as e:
            logger.debug("VIX fetch failed: %s", e)

        return result


# ══════════════════════════════════════════════════════════════════
#  Singleton
# ══════════════════════════════════════════════════════════════════

_service: Optional[MarketDataService] = None


def init_market_data_service() -> Optional[MarketDataService]:
    """
    Initialize the global MarketDataService from .env credentials.
    Initializes Angel One, Upstox, and/or Zerodha independently.
    Returns None only if NONE are configured.
    """
    global _service

    angel_client = None
    upstox_client = None
    zerodha_client = None

    # ── Angel One (historical candles + portfolio) ──
    if settings.ANGEL_API_KEY and settings.ANGEL_CLIENT_ID:
        try:
            from ingestion.broker_adapters.angel_one import AngelOneClient
            angel_client = AngelOneClient(
                api_key=settings.ANGEL_API_KEY,
                client_id=settings.ANGEL_CLIENT_ID,
                password=settings.ANGEL_PASSWORD,
                totp_secret=settings.ANGEL_TOTP_SECRET,
            )
            logger.info("Angel One client initialized (historical data + portfolio)")
        except Exception as e:
            logger.warning("Angel One init failed: %s", e)
    else:
        logger.info("Angel One not configured — historical candles unavailable")

    # ── Upstox (live market data) ──
    # Token priority: saved token file → UPSTOX_ACCESS_TOKEN env var
    from ingestion.token_store import get_upstox_token
    upstox_token = get_upstox_token()
    if upstox_token:
        try:
            from ingestion.broker_adapters.upstox import UpstoxClient
            upstox_client = UpstoxClient(
                access_token=upstox_token,
                api_key=settings.UPSTOX_API_KEY,
            )
            logger.info("Upstox client initialized (live market data)")
        except Exception as e:
            logger.warning("Upstox init failed: %s", e)
    else:
        logger.info("Upstox not configured — live market data unavailable")

    # ── Zerodha (optional — historical + portfolio fallback) ──
    if settings.KITE_API_KEY and settings.KITE_ACCESS_TOKEN:
        try:
            from ingestion.broker_adapters.zerodha import ZerodhaClient
            zerodha_client = ZerodhaClient(
                api_key=settings.KITE_API_KEY,
                access_token=settings.KITE_ACCESS_TOKEN,
            )
            logger.info("Zerodha client initialized (historical + portfolio)")
        except Exception as e:
            logger.warning("Zerodha init failed: %s", e)
    else:
        logger.info("Zerodha not configured — Kite Connect unavailable")

    if not angel_client and not upstox_client and not zerodha_client:
        logger.warning("No market data sources configured (need Angel One, Upstox, and/or Zerodha)")
        return None

    _service = MarketDataService(
        angel_client=angel_client,
        upstox_client=upstox_client,
        zerodha_client=zerodha_client,
    )
    if angel_client:
        _service._angel_last_login = datetime.now(timezone.utc)

    sources = []
    if angel_client:
        sources.append("Angel One (historical + portfolio)")
    if upstox_client:
        sources.append("Upstox (live)")
    if zerodha_client:
        sources.append("Zerodha (historical + portfolio)")
    logger.info("MarketDataService initialized: %s", " + ".join(sources))

    return _service


def get_market_data_service() -> Optional[MarketDataService]:
    """Get the global MarketDataService (None if not initialized)."""
    return _service


def reload_upstox_token(new_token: str) -> bool:
    """
    Hot-reload the Upstox access token without restarting the server.
    Reinitializes the UpstoxClient with the new token and replaces it in the
    running MarketDataService. Returns True on success.
    """
    global _service

    try:
        from ingestion.broker_adapters.upstox import UpstoxClient
        from config.settings import settings

        new_client = UpstoxClient(
            access_token=new_token,
            api_key=settings.UPSTOX_API_KEY,
        )

        if _service is None:
            # Service not yet initialized — create it now
            _service = MarketDataService(upstox_client=new_client)
        else:
            _service._upstox = new_client

        logger.info("Upstox client hot-reloaded with new access token")
        return True

    except Exception as e:
        logger.error("Failed to hot-reload Upstox token: %s", e)
        return False
