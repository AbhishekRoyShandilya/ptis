# bootstrap/adapters.py
"""
Adapter Bridges
================

Bridges between bootstrap module's expected interfaces
and PTIS's actual system implementations.

The bootstrap files (fast_backtester, warm_starter) were designed
with clean interface contracts. These adapters translate between
those contracts and the real PTIS classes.

Each adapter is thin — just argument/format translation.
No business logic here.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  ADAPTER 1: Market Data
# ══════════════════════════════════════════════════════════════


class MarketDataAdapter:
    """
    Bridges:
        bootstrap expects: get_historical_candles(symbol, timeframe, lookback_days)
                           → List[Dict] with {timestamp, open, high, low, close, volume}
        PTIS has:          BrokerAPIFacade.get_candles(broker, symbol, interval, from_date, to_date)
                           → List[Dict] same format
    """

    def __init__(
        self,
        broker_facade: Any,   # ingestion.broker_api.BrokerAPIFacade
        broker_name: str = "",
    ):
        self._facade = broker_facade
        self._broker = broker_name

        # Auto-detect broker if not specified
        if not self._broker and hasattr(broker_facade, "available_brokers"):
            brokers = broker_facade.available_brokers
            if brokers:
                self._broker = brokers[0]

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str = "5min",
        lookback_days: int = 180,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical candles via the local OHLCV lake (CandleStore) — system
        market data, NOT the user's execution broker. Cached + reproducible.
        """
        interval = self._normalize_interval(timeframe)   # "5min" → "5m"
        to_date = date.today()
        from_date = to_date - timedelta(days=lookback_days)

        try:
            from marketdata.candle_store import get_candle_store
            df = get_candle_store().get(symbol, interval, from_date, to_date)
            if df is None or df.empty:
                return []
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
                })
            return out
        except Exception as e:
            logger.warning("CandleStore fetch failed for %s: %s", symbol, e)
            return []

    @staticmethod
    def _normalize_interval(timeframe: str) -> str:
        """Convert bootstrap timeframe to broker interval format."""
        mapping = {
            "1min": "1m", "5min": "5m", "15min": "15m",
            "30min": "30m", "1h": "1h", "4h": "4h",
            "1d": "1d", "daily": "1d",
            # Pass through if already in broker format
            "1m": "1m", "5m": "5m", "15m": "15m",
        }
        return mapping.get(timeframe, timeframe)


# ══════════════════════════════════════════════════════════════
#  ADAPTER 2: Feature Computer
# ══════════════════════════════════════════════════════════════


class FeatureComputerAdapter:
    """
    Bridges:
        bootstrap expects: compute_from_candles(symbol, context_candles, current_candle)
                           → Dict[str, float]
        PTIS has:          IndicatorEngine (static methods on pd.Series/DataFrame)
                           + EventExtractor.compute_fingerprint(df, bar_idx)

    Computes a flat Dict[str, float] of features from raw candles.
    These feed into hard rules, soft scorers, and the feature scaler.
    """

    def __init__(self, feature_keys: Optional[List[str]] = None):
        """
        Args:
            feature_keys: Subset of features to compute. If None, computes all.
        """
        self._feature_keys = feature_keys

    async def compute_from_candles(
        self,
        symbol: str,
        context_candles: List[Dict[str, Any]],
        current_candle: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute features from OHLCV candle data.

        Returns a flat dict of feature_name → float value.
        Uses IndicatorEngine for technical indicators.
        """
        # Build DataFrame from context + current candle
        all_candles = context_candles + [current_candle]
        df = self._candles_to_df(all_candles)

        if len(df) < 20:
            return {}

        features: Dict[str, float] = {}

        try:
            from strategy_engine.indicators import IndicatorEngine

            close = df["close"]
            bar_idx = len(df) - 1

            # Core indicators
            atr_series = IndicatorEngine.atr(df, period=14)
            features["atr"] = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 0.0

            rsi_series = IndicatorEngine.rsi(close, period=14)
            features["rsi"] = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 else 50.0

            ema_9 = IndicatorEngine.ema(close, period=9)
            ema_20 = IndicatorEngine.ema(close, period=20)
            features["ema_9"] = float(ema_9.iloc[-1]) if len(ema_9) > 0 else 0.0
            features["ema_20"] = float(ema_20.iloc[-1]) if len(ema_20) > 0 else 0.0

            sma_50 = IndicatorEngine.sma(close, period=50) if len(close) >= 50 else close
            features["sma_50"] = float(sma_50.iloc[-1]) if len(sma_50) > 0 else 0.0

            adx_series = IndicatorEngine.adx(df, period=14)
            features["adx"] = float(adx_series.iloc[-1]) if len(adx_series) > 0 else 0.0

            bb_mid, bb_upper, bb_lower = IndicatorEngine.bollinger_bands(close)
            features["bb_width"] = (
                (float(bb_upper.iloc[-1]) - float(bb_lower.iloc[-1]))
                / float(bb_mid.iloc[-1])
                if float(bb_mid.iloc[-1]) > 0 else 0.0
            )

            # Price-derived features
            current_close = float(close.iloc[-1])
            features["close"] = current_close
            features["volume"] = float(df["volume"].iloc[-1])

            # Relative volume (current vs 20-bar average)
            vol_avg = float(df["volume"].iloc[-20:].mean()) if len(df) >= 20 else 1.0
            features["relative_volume"] = (
                float(df["volume"].iloc[-1]) / vol_avg if vol_avg > 0 else 1.0
            )

            # Distance from VWAP (if volume available)
            if "volume" in df.columns and df["volume"].sum() > 0:
                typical = (df["high"] + df["low"] + df["close"]) / 3
                cum_tp_vol = (typical * df["volume"]).cumsum()
                cum_vol = df["volume"].cumsum()
                vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
                vwap_val = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else current_close
                features["vwap"] = vwap_val
                features["vwap_distance"] = (
                    (current_close - vwap_val) / features["atr"]
                    if features["atr"] > 0 else 0.0
                )

            # Price position relative to EMAs
            if features["ema_20"] > 0:
                features["price_vs_ema20"] = (
                    (current_close - features["ema_20"]) / features["atr"]
                    if features["atr"] > 0 else 0.0
                )

            # Candle body ratio
            candle_range = float(df["high"].iloc[-1] - df["low"].iloc[-1])
            candle_body = abs(float(df["close"].iloc[-1] - df["open"].iloc[-1]))
            features["body_ratio"] = (
                candle_body / candle_range if candle_range > 0 else 0.0
            )

            # ATR percentile (volatility context)
            if len(df) >= 50:
                features["atr_percentile"] = IndicatorEngine.atr_percentile(
                    df, period=14, lookback=min(252, len(df))
                )

        except Exception as e:
            logger.warning("Feature computation failed: %s", e)
            return features

        # Filter to requested keys if specified
        if self._feature_keys:
            features = {
                k: v for k, v in features.items()
                if k in self._feature_keys
            }

        return features

    @staticmethod
    def _candles_to_df(candles: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert list of candle dicts to DataFrame."""
        df = pd.DataFrame(candles)
        # Ensure required columns
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df


# ══════════════════════════════════════════════════════════════
#  ADAPTER 3: Regime Engine
# ══════════════════════════════════════════════════════════════


class RegimeEngineAdapter:
    """
    Bridges:
        bootstrap expects: classify_from_candles(candles) → str
        PTIS has:          MarketRegimeDetector.detect(df: pd.DataFrame) → RegimeAnalysis
    """

    def __init__(self, detector: Optional[Any] = None):
        """
        Args:
            detector: MarketRegimeDetector instance. Created lazily if None.
        """
        self._detector = detector

    def _get_detector(self) -> Any:
        if self._detector is None:
            from strategy_engine.regime_detector import MarketRegimeDetector
            self._detector = MarketRegimeDetector()
        return self._detector

    async def classify_from_candles(
        self, candles: List[Dict[str, Any]]
    ) -> str:
        """Classify market regime from raw candle dicts."""
        if len(candles) < 50:
            return "unknown"

        try:
            df = pd.DataFrame(candles)
            for col in ("open", "high", "low", "close", "volume"):
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

            detector = self._get_detector()
            analysis = detector.detect(df)
            return analysis.regime.value if hasattr(analysis.regime, "value") else str(analysis.regime)
        except Exception as e:
            logger.warning("Regime classification failed: %s", e)
            return "unknown"


# ══════════════════════════════════════════════════════════════
#  ADAPTER 4: Strategy Engine Facade
# ══════════════════════════════════════════════════════════════


class StrategyEngineFacade:
    """
    Wraps StrategyUnderstandingEngine for warm_starter.

    warm_starter expects:
        engine._regime → RegimeStabilityChecker (with .record_outcome, .compute_stability)
        engine.evaluate(features) → MetaLayerOutput
    """

    def __init__(
        self,
        understanding_engine: Optional[Any] = None,
    ):
        self._engine = understanding_engine
        # Expose _regime directly for warm_starter access
        if understanding_engine and hasattr(understanding_engine, "_regime"):
            self._regime = understanding_engine._regime
        else:
            from strategy_quality.engine import RegimeStabilityChecker
            self._regime = RegimeStabilityChecker()

    def evaluate(self, features: Dict[str, Any]) -> Any:
        """Evaluate signal features through the strategy engine."""
        if self._engine is None:
            # Return a minimal stub output
            return _StubMetaLayerOutput(features)
        return self._engine.evaluate(features)


class _StubMetaLayerOutput:
    """Minimal output when no engine is available yet."""

    def __init__(self, features: Dict[str, Any]):
        self.overall_match_score = 0.5
        self.strategy_confidence = 0.3
        self.decision = "weak"
        self.soft_score_components = {}
        self.context_components = {}


# ══════════════════════════════════════════════════════════════
#  ADAPTER 5: Pattern Intelligence Facade
# ══════════════════════════════════════════════════════════════


class PatternIntelFacade:
    """
    Wraps PatternAwareFirewall for warm_starter.

    warm_starter expects:
        intel.scaler → FeatureScaler (with .update, .is_ready, .to_vector)
        intel.clusterer → DynamicPatternClusterer (with .add_trade, .recluster)
    """

    def __init__(
        self,
        firewall: Optional[Any] = None,
        feature_keys: Optional[List[str]] = None,
    ):
        if firewall is not None:
            self._firewall = firewall
        else:
            # Create a standalone pattern intelligence stack
            from strategy_quality.pattern_intelligence import (
                FeatureScaler,
                DynamicPatternClusterer,
                PatternAwareFirewall,
            )
            keys = feature_keys or _DEFAULT_FEATURE_KEYS
            self._scaler = FeatureScaler(keys)
            self._clusterer = DynamicPatternClusterer(
                feature_keys=keys, scaler=self._scaler,
            )
            self._firewall = None

    @property
    def scaler(self):
        if self._firewall:
            return self._firewall.scaler
        return self._scaler

    @property
    def clusterer(self):
        if self._firewall:
            return self._firewall.clusterer
        return self._clusterer


# ══════════════════════════════════════════════════════════════
#  DEFAULT FEATURE KEYS
# ══════════════════════════════════════════════════════════════

_DEFAULT_FEATURE_KEYS = [
    "atr", "rsi", "adx", "bb_width",
    "relative_volume", "vwap_distance",
    "price_vs_ema20", "body_ratio",
    "atr_percentile", "ema_9", "ema_20",
]


# ══════════════════════════════════════════════════════════════
#  FACTORY: Build all adapters from PTIS dependencies
# ══════════════════════════════════════════════════════════════


def build_bootstrap_adapters(
    broker_facade: Optional[Any] = None,
    broker_name: str = "",
    understanding_engine: Optional[Any] = None,
    pattern_firewall: Optional[Any] = None,
    regime_detector: Optional[Any] = None,
    feature_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build all adapter instances needed by the bootstrap pipeline.

    Returns dict with keys matching bootstrap constructor args:
        market_data, feature_computer, regime_engine,
        strategy_engine, pattern_intel
    """
    keys = feature_keys or _DEFAULT_FEATURE_KEYS

    return {
        "market_data": MarketDataAdapter(
            broker_facade=broker_facade,
            broker_name=broker_name,
        ) if broker_facade else MarketDataAdapter(
            broker_facade=_NullBrokerFacade(),
        ),
        "feature_computer": FeatureComputerAdapter(
            feature_keys=keys,
        ),
        "regime_engine": RegimeEngineAdapter(
            detector=regime_detector,
        ),
        "strategy_engine": StrategyEngineFacade(
            understanding_engine=understanding_engine,
        ),
        "pattern_intel": PatternIntelFacade(
            firewall=pattern_firewall,
            feature_keys=keys,
        ),
    }


class _NullBrokerFacade:
    """Stub broker when no real broker is configured."""

    @property
    def available_brokers(self) -> List[str]:
        return []

    def get_candles(self, **kwargs) -> List[Dict]:
        return []
