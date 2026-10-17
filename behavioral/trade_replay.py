"""
Trade Replay Engine
=====================

Backwards reconstruction of market state at historical trade timestamps.

The single biggest gap in PTIS pre-this: every existing analyzer enriches
trades with context AT ENTRY only. We need full backwards replay so we can
answer "was the move 70% complete when you entered?" or "did price reach
+1R favorable before your early exit?" — questions that require seeing the
price PATH, not just snapshots.

Tiered replay (driven by trade age):
  - FULL     (≤ 30 days):   1m + 5m + 15m + 1h candles, all indicators, bar-by-bar excursion
  - STANDARD (31-90 days):  15m + 1h candles, core indicators, MFE/MAE only
  - SAMPLE   (> 90 days):   summary stats only — entry/exit prices vs SMA20

Reuses extensively:
  - `ingestion/market_data_service.py:MarketDataService._fetch_candles_raw` for backward fetches
  - `strategy_engine/indicators.py:IndicatorEngine` for EMA/RSI/ATR
  - `ml/probabilistic_regime.py:ProbabilisticRegimeClassifier` (improvement #2) for regime tagging
  - `automation/setup_scanner.py:_compute_vwap` pattern for VWAP

Aggressive caching: candle data is cached per (symbol, timeframe, date_window)
so multiple trades on the same symbol/day reuse the same fetch.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from behavioral.replay_models import (
    ExcursionPoint,
    IndicatorSnapshot,
    REPLAY_TIER_FULL,
    REPLAY_TIER_SAMPLE,
    REPLAY_TIER_STANDARD,
    ReplayObservations,
    ReplayedTrade,
)

logger = logging.getLogger(__name__)


# ── Tier thresholds (configurable via constructor) ─────────────────


DEFAULT_TIER_FULL_DAYS = 30
DEFAULT_TIER_STANDARD_DAYS = 90

# Bars to fetch around entry/exit (per timeframe)
FULL_BARS_BEFORE_ENTRY = 30        # for setup-formation context
FULL_BARS_AFTER_EXIT = 5            # for "was there more move after exit?"
STANDARD_BARS_BEFORE_ENTRY = 20
STANDARD_BARS_AFTER_EXIT = 3

# Timeframes per tier
FULL_TIMEFRAMES = ["1m", "5m", "15m", "1h"]
STANDARD_TIMEFRAMES = ["15m", "1h"]


# ── Engine ──────────────────────────────────────────────────────────


class TradeReplayEngine:
    """
    Replays a single closed trade against historical market data.

    Public API:
        engine = TradeReplayEngine(market_data_service)
        replay = engine.replay_trade(trade)        # returns ReplayedTrade
        replays = engine.replay_trades(trade_list) # bulk; reuses candle cache

    The engine is stateless across replay() calls EXCEPT for the cache,
    which speeds up bulk replay of trades on the same symbol/day.
    """

    def __init__(
        self,
        market_data_service: Any,
        regime_classifier: Optional[Any] = None,
        tier_full_days: int = DEFAULT_TIER_FULL_DAYS,
        tier_standard_days: int = DEFAULT_TIER_STANDARD_DAYS,
    ):
        self._mds = market_data_service
        self._tier_full_days = tier_full_days
        self._tier_standard_days = tier_standard_days

        # Probabilistic regime classifier (improvement #2)
        if regime_classifier is None:
            try:
                from ml.probabilistic_regime import get_probabilistic_classifier
                regime_classifier = get_probabilistic_classifier()
            except Exception as e:
                logger.warning("ProbabilisticRegimeClassifier unavailable: %s", e)
        self._regime_clf = regime_classifier

        # Per-(symbol, timeframe, date_window) candle cache
        # Key: (symbol, timeframe, from_date_iso, to_date_iso)
        # Value: list[dict] candles
        self._candle_cache: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}

    # ── Public ──

    def clear_cache(self) -> None:
        self._candle_cache.clear()

    def select_tier(self, trade_age_days: int) -> str:
        if trade_age_days <= self._tier_full_days:
            return REPLAY_TIER_FULL
        if trade_age_days <= self._tier_standard_days:
            return REPLAY_TIER_STANDARD
        return REPLAY_TIER_SAMPLE

    def replay_trade(self, trade: Any, now: Optional[datetime] = None) -> ReplayedTrade:
        """
        Replay a single trade. `trade` should be a Trade ORM object OR a dict
        with at minimum: id, symbol, direction, entry_time, entry_price,
        exit_time (optional), exit_price (optional), stop_loss (optional).
        """
        start_ms = time.monotonic()
        now = now or datetime.now(timezone.utc)

        # Normalize trade fields (works for ORM or dict)
        trade_dict = self._normalize_trade(trade)

        # Decide replay tier from trade age
        entry_time = trade_dict["entry_time"]
        if entry_time is None:
            return self._empty_replay(trade_dict, "missing entry_time")
        age_days = max(0, (self._strip_tz(now) - self._strip_tz(entry_time)).days)
        tier = self.select_tier(age_days)

        replay = ReplayedTrade(
            trade_id=str(trade_dict["id"]),
            symbol=trade_dict["symbol"],
            direction=trade_dict["direction"],
            entry_time=entry_time,
            exit_time=trade_dict.get("exit_time"),
            entry_price=float(trade_dict["entry_price"]),
            exit_price=trade_dict.get("exit_price"),
            stop_loss=trade_dict.get("stop_loss"),
            tier=tier,
            replayed_at=now,
        )

        try:
            if tier == REPLAY_TIER_SAMPLE:
                self._replay_sample(replay)
            elif tier == REPLAY_TIER_STANDARD:
                self._replay_standard(replay)
            else:
                self._replay_full(replay)
        except Exception as e:
            logger.warning("Replay failed for trade %s: %s", trade_dict.get("id"), e)
            replay.fetch_errors.append(f"{type(e).__name__}: {e}")

        replay.replay_duration_ms = int((time.monotonic() - start_ms) * 1000)
        return replay

    def replay_trades(self, trades: List[Any]) -> List[ReplayedTrade]:
        """Bulk replay; cache reuse benefits trades on same symbol/day."""
        return [self.replay_trade(t) for t in trades]

    # ── Replay logic per tier ──

    def _replay_full(self, replay: ReplayedTrade) -> None:
        """Full detail: 1m+5m+15m+1h candles, indicators, bar-by-bar excursion."""
        # Primary timeframe for snapshots: 5m (matches typical intraday strategies)
        primary_tf = "5m"
        replay.snapshot_at_entry = self._build_snapshot(
            replay.symbol, primary_tf, replay.entry_time,
            bars_before=FULL_BARS_BEFORE_ENTRY,
        )
        if replay.exit_time:
            replay.snapshot_at_exit = self._build_snapshot(
                replay.symbol, primary_tf, replay.exit_time,
                bars_before=FULL_BARS_BEFORE_ENTRY,
            )

        # HTF context (composes with #2)
        replay.snapshot_15m_at_entry = self._build_snapshot(
            replay.symbol, "15m", replay.entry_time, bars_before=20,
        )
        replay.snapshot_1h_at_entry = self._build_snapshot(
            replay.symbol, "1h", replay.entry_time, bars_before=20,
        )

        # Bar-by-bar excursion (only FULL tier)
        replay.excursion_path = self._compute_excursion(
            replay.symbol, primary_tf,
            replay.entry_time, replay.exit_time,
            replay.entry_price, replay.direction, replay.stop_loss,
        )

        # Find peak favorable bar + snapshot at that bar
        if replay.excursion_path:
            peak = max(replay.excursion_path, key=lambda p: p.mfe_so_far)
            replay.snapshot_at_peak_favorable = self._build_snapshot(
                replay.symbol, primary_tf, peak.timestamp,
                bars_before=10,
            )

        # Synthesize observations
        replay.observations = self._synthesize_observations(replay)

    def _replay_standard(self, replay: ReplayedTrade) -> None:
        """Lighter: 15m+1h only, snapshot at entry/exit, MFE/MAE summary."""
        replay.snapshot_at_entry = self._build_snapshot(
            replay.symbol, "15m", replay.entry_time,
            bars_before=STANDARD_BARS_BEFORE_ENTRY,
        )
        if replay.exit_time:
            replay.snapshot_at_exit = self._build_snapshot(
                replay.symbol, "15m", replay.exit_time,
                bars_before=STANDARD_BARS_BEFORE_ENTRY,
            )
        replay.snapshot_15m_at_entry = replay.snapshot_at_entry
        replay.snapshot_1h_at_entry = self._build_snapshot(
            replay.symbol, "1h", replay.entry_time, bars_before=15,
        )

        # MFE/MAE on 15m bars (no per-bar storage, just summary)
        path = self._compute_excursion(
            replay.symbol, "15m",
            replay.entry_time, replay.exit_time,
            replay.entry_price, replay.direction, replay.stop_loss,
        )
        if path:
            peak = max(path, key=lambda p: p.mfe_so_far)
            replay.observations.peak_r_during_trade = peak.mfe_so_far
            replay.observations.peak_r_timestamp = peak.timestamp
            replay.snapshot_at_peak_favorable = self._build_snapshot(
                replay.symbol, "15m", peak.timestamp, bars_before=10,
            )

        replay.observations = self._synthesize_observations(replay, simple_path=path)

    def _replay_sample(self, replay: ReplayedTrade) -> None:
        """Minimal: only daily/15m close vs SMA20 at entry. Background-job tier."""
        # Single 15m snapshot — no excursion, no HTF
        replay.snapshot_at_entry = self._build_snapshot(
            replay.symbol, "15m", replay.entry_time, bars_before=20,
        )
        replay.observations = self._synthesize_observations(replay)

    # ── Snapshot construction ──

    def _build_snapshot(
        self,
        symbol: str,
        timeframe: str,
        target_ts: datetime,
        bars_before: int,
    ) -> Optional[IndicatorSnapshot]:
        """
        Fetch candles ending at-or-before target_ts; compute indicators on
        the window; return IndicatorSnapshot for the bar nearest target_ts.
        """
        if target_ts is None:
            return None

        # Fetch a window: target_date - some buffer through target_date
        candles = self._fetch_candles_window(symbol, timeframe, target_ts, bars_before)
        if not candles:
            return None

        # Pick the bar closest to (but not exceeding) target_ts
        # candles are list[dict] with 'timestamp' field
        target_naive = self._strip_tz(target_ts)
        idx = None
        for i in range(len(candles) - 1, -1, -1):
            ts = self._strip_tz(self._candle_ts(candles[i]))
            if ts is None:
                continue
            if ts <= target_naive:
                idx = i
                break
        if idx is None:
            idx = len(candles) - 1

        # Build DataFrame for indicator computation
        df = self._candles_to_df(candles[: idx + 1])
        if df is None or df.empty:
            return None

        last_row = df.iloc[-1]
        snap = IndicatorSnapshot(
            timestamp=df.index[-1].to_pydatetime() if hasattr(df.index[-1], "to_pydatetime") else target_ts,
            timeframe=timeframe,
            close=float(last_row["close"]),
            high=float(last_row.get("high", last_row["close"])),
            low=float(last_row.get("low", last_row["close"])),
            open=float(last_row.get("open", last_row["close"])),
            volume=float(last_row.get("volume", 0) or 0),
        )

        # Indicators (use existing IndicatorEngine when available)
        try:
            from strategy_engine.indicators import IndicatorEngine
            ie = IndicatorEngine()
            if len(df) >= 9:
                ema9 = ie.ema(df["close"], 9)
                snap.ema_9 = float(ema9.iloc[-1]) if len(ema9) else None
            if len(df) >= 20:
                ema20 = ie.ema(df["close"], 20)
                snap.ema_20 = float(ema20.iloc[-1]) if len(ema20) else None
                snap.sma_20 = float(df["close"].iloc[-20:].mean())
            if len(df) >= 50:
                ema50 = ie.ema(df["close"], 50)
                snap.ema_50 = float(ema50.iloc[-1]) if len(ema50) else None
            if len(df) >= 14:
                rsi = ie.rsi(df["close"], 14)
                snap.rsi_14 = float(rsi.iloc[-1]) if len(rsi) else None
                atr = ie.atr(df, 14)
                snap.atr_14 = float(atr.iloc[-1]) if len(atr) else None
                if snap.atr_14 and snap.close > 0:
                    snap.atr_pct = round(snap.atr_14 / snap.close * 100, 4)
        except Exception as e:
            logger.debug("Indicator computation skipped: %s", e)

        # Bollinger bands
        if len(df) >= 20:
            sma = df["close"].iloc[-20:].mean()
            std = float(df["close"].iloc[-20:].std())
            snap.bb_upper = float(sma + 2 * std)
            snap.bb_lower = float(sma - 2 * std)
            if sma > 0:
                snap.bb_width_pct = round(((snap.bb_upper - snap.bb_lower) / sma) * 100, 4)

        # VWAP — only meaningful intraday. Compute on bars from same trading day.
        snap.vwap = self._compute_vwap_intraday(candles[: idx + 1], target_ts)

        # Volume ratio (current / 20-bar avg)
        if len(df) >= 20:
            recent_vol = df["volume"].iloc[-20:].mean()
            if recent_vol > 0:
                snap.volume_ratio = round(float(snap.volume) / float(recent_vol), 4)

        # Probabilistic regime tagging (improvement #2)
        if self._regime_clf is not None and len(df) >= 50:
            try:
                probs = self._regime_clf.classify(df)
                snap.regime_label = probs.dominant
                snap.regime_probs = probs.as_dict()
                snap.regime_confidence = float(probs.dominant_prob)
            except Exception as e:
                logger.debug("Regime classify failed: %s", e)

        return snap

    # ── Excursion computation ──

    def _compute_excursion(
        self,
        symbol: str,
        timeframe: str,
        entry_time: datetime,
        exit_time: Optional[datetime],
        entry_price: float,
        direction: str,
        stop_loss: Optional[float],
    ) -> List[ExcursionPoint]:
        """
        Build bar-by-bar excursion from entry to exit.
        Each ExcursionPoint records signed R-mult relative to entry.
        """
        if exit_time is None or entry_price <= 0:
            return []

        # Fetch all bars from entry through exit (+ buffer)
        candles = self._fetch_candles_range(symbol, timeframe, entry_time, exit_time)
        if not candles:
            return []

        # Filter to bars between entry and exit
        entry_naive = self._strip_tz(entry_time)
        exit_naive = self._strip_tz(exit_time)
        in_trade = []
        for c in candles:
            ts = self._strip_tz(self._candle_ts(c))
            if ts is None:
                continue
            if entry_naive <= ts <= exit_naive:
                in_trade.append(c)
        if not in_trade:
            return []

        # 1R = stop distance, fall back to 1% of entry if no SL
        if stop_loss and stop_loss > 0:
            risk_per_unit = abs(entry_price - stop_loss)
        else:
            risk_per_unit = entry_price * 0.01
        if risk_per_unit <= 0:
            return []

        is_long = (direction or "long").lower() == "long"
        path: List[ExcursionPoint] = []
        peak_mfe = 0.0
        peak_mae = 0.0
        for i, c in enumerate(in_trade):
            close = float(c.get("close", 0) or 0)
            if close <= 0:
                continue
            move = (close - entry_price) if is_long else (entry_price - close)
            r = move / risk_per_unit
            peak_mfe = max(peak_mfe, r)
            peak_mae = min(peak_mae, r)
            path.append(ExcursionPoint(
                timestamp=self._candle_ts(c),
                bar_index=i,
                close=close,
                r_at_this_bar=round(r, 4),
                mfe_so_far=round(peak_mfe, 4),
                mae_so_far=round(peak_mae, 4),
            ))
        return path

    # ── Observations synthesis ──

    def _synthesize_observations(
        self,
        replay: ReplayedTrade,
        simple_path: Optional[List[ExcursionPoint]] = None,
    ) -> ReplayObservations:
        """Roll up snapshots + excursion into trader-facing observations."""
        obs = ReplayObservations()
        snap = replay.snapshot_at_entry
        snap_exit = replay.snapshot_at_exit

        # RSI at entry
        if snap and snap.rsi_14 is not None:
            obs.rsi_at_entry = round(snap.rsi_14, 2)

        # Trend alignment vs HTF
        ht = replay.snapshot_1h_at_entry
        if ht and ht.ema_50 is not None:
            is_long = (replay.direction or "long").lower() == "long"
            if ht.close > ht.ema_50:
                obs.trend_alignment_at_entry = "with_trend" if is_long else "against_trend"
            elif ht.close < ht.ema_50:
                obs.trend_alignment_at_entry = "against_trend" if is_long else "with_trend"
            else:
                obs.trend_alignment_at_entry = "no_trend"

        # Position vs key levels
        if snap:
            if snap.vwap is not None:
                obs.above_vwap_at_entry = snap.close > snap.vwap
            if snap.ema_20 is not None:
                obs.above_ema20_at_entry = snap.close > snap.ema_20
            if snap.ema_50 is not None:
                obs.above_ema50_at_entry = snap.close > snap.ema_50

        # Regime
        if snap and snap.regime_label:
            obs.regime_at_entry = snap.regime_label
        if snap_exit and snap_exit.regime_label:
            obs.regime_at_exit = snap_exit.regime_label
        obs.regime_changed = bool(
            obs.regime_at_entry and obs.regime_at_exit
            and obs.regime_at_entry != obs.regime_at_exit
        )

        # Peak R + early-exit severity (FULL tier path; STANDARD uses simple_path)
        path = replay.excursion_path or simple_path or []
        if path:
            peak = max(p.mfe_so_far for p in path)
            final = path[-1].r_at_this_bar
            obs.peak_r_during_trade = round(peak, 4)
            obs.final_r = round(final, 4)
            # Early-exit severity: how much R was given back before exit
            if peak > 0.5:  # only meaningful if trade had real upside
                give_back = peak - final
                # Normalize: give_back of 1R = severity 0.5; 2R = 1.0
                obs.early_exit_severity = round(min(give_back / 2.0, 1.0), 4)
            # Find timestamp of peak
            for p in path:
                if abs(p.mfe_so_far - peak) < 0.001:
                    obs.peak_r_timestamp = p.timestamp
                    break

        # Move-pct-complete-at-entry: rough proxy via where entry sits vs recent range
        obs.move_pct_complete_at_entry = self._estimate_move_pct_complete(replay)

        return obs

    def _estimate_move_pct_complete(self, replay: ReplayedTrade) -> Optional[float]:
        """
        Rough proxy: at entry, how far into the recent move are we?
        - Look at last N (≈20) bars before entry on primary timeframe
        - Find swing low (for longs) or swing high (for shorts)
        - move_pct = (entry_price - swing_low) / (recent_high - swing_low)
        """
        snap = replay.snapshot_at_entry
        if snap is None:
            return None
        # Need recent high and low — fetch a window
        candles = self._fetch_candles_window(
            replay.symbol,
            snap.timeframe or "15m",
            replay.entry_time,
            bars_before=20,
        )
        if not candles or len(candles) < 5:
            return None
        df = self._candles_to_df(candles)
        if df is None or df.empty:
            return None
        recent_high = float(df["high"].max())
        recent_low = float(df["low"].min())
        rng = recent_high - recent_low
        if rng <= 0:
            return None
        is_long = (replay.direction or "long").lower() == "long"
        if is_long:
            pct = (replay.entry_price - recent_low) / rng
        else:
            pct = (recent_high - replay.entry_price) / rng
        return round(max(0.0, min(1.0, pct)), 4)

    # ── Candle fetching with cache ──

    def _fetch_candles_window(
        self,
        symbol: str,
        timeframe: str,
        target_ts: datetime,
        bars_before: int,
    ) -> List[Dict[str, Any]]:
        """Fetch a window of `bars_before` candles ending at-or-before target_ts."""
        # Compute date range: target_ts - some days, through target_ts
        target_date = target_ts.date() if isinstance(target_ts, datetime) else target_ts
        days_back = self._estimate_days_for_bars(timeframe, bars_before + 5)
        from_d = target_date - timedelta(days=days_back)
        to_d = target_date
        return self._fetch_candles_cached(symbol, timeframe, from_d, to_d)

    def _fetch_candles_range(
        self,
        symbol: str,
        timeframe: str,
        from_ts: datetime,
        to_ts: datetime,
    ) -> List[Dict[str, Any]]:
        """Fetch all candles between two timestamps."""
        from_d = from_ts.date() if isinstance(from_ts, datetime) else from_ts
        to_d = to_ts.date() if isinstance(to_ts, datetime) else to_ts
        return self._fetch_candles_cached(symbol, timeframe, from_d, to_d)

    def _fetch_candles_cached(
        self,
        symbol: str,
        timeframe: str,
        from_d: date,
        to_d: date,
    ) -> List[Dict[str, Any]]:
        """Wrapped fetch that uses an in-memory per-(sym, tf, range) cache."""
        key = (symbol.upper(), timeframe, from_d.isoformat(), to_d.isoformat())
        cached = self._candle_cache.get(key)
        if cached is not None:
            return cached
        try:
            candles = self._mds._fetch_candles_raw(symbol, timeframe, from_d, to_d) or []
        except Exception as e:
            logger.debug("Candle fetch failed for %s %s %s..%s: %s",
                         symbol, timeframe, from_d, to_d, e)
            candles = []
        self._candle_cache[key] = candles
        return candles

    @staticmethod
    def _estimate_days_for_bars(timeframe: str, n_bars: int) -> int:
        """Approximate calendar days needed to get n_bars at given timeframe."""
        # ~78 5-min bars per trading day, ~26 15m bars, ~6 1h bars
        bars_per_day = {
            "1m": 375, "5m": 78, "15m": 26, "30m": 13, "1h": 6, "1d": 1,
        }.get(timeframe, 26)
        days = max(2, int(n_bars / max(bars_per_day, 1)) * 2 + 5)  # buffer for holidays
        return days

    # ── VWAP computation ──

    @staticmethod
    def _compute_vwap_intraday(candles: List[Dict[str, Any]], target_ts: datetime) -> Optional[float]:
        """
        Compute VWAP using only bars from the same trading day as target_ts.
        Mirrors automation/setup_scanner.py:_compute_vwap pattern.
        """
        if not candles or target_ts is None:
            return None
        target_date = target_ts.date() if isinstance(target_ts, datetime) else target_ts
        same_day = []
        for c in candles:
            ts = TradeReplayEngine._candle_ts(c)
            if ts and ts.date() == target_date:
                same_day.append(c)
        if not same_day:
            return None
        total_vol = sum(float(c.get("volume", 0) or 0) for c in same_day)
        if total_vol <= 0:
            return None
        total_pv = sum(
            ((float(c.get("high", 0) or 0) + float(c.get("low", 0) or 0)
              + float(c.get("close", 0) or 0)) / 3.0) * float(c.get("volume", 0) or 0)
            for c in same_day
        )
        return round(total_pv / total_vol, 4)

    # ── DataFrame helper ──

    @staticmethod
    def _candles_to_df(candles: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        if not candles:
            return None
        try:
            df = pd.DataFrame(candles)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            elif "datetime" in df.columns:
                df = df.set_index("datetime")
            for col in ("open", "high", "low", "close", "volume"):
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close"])
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass
            df = df.sort_index()
            return df
        except Exception as e:
            logger.debug("candles_to_df failed: %s", e)
            return None

    # ── Trade normalization ──

    @staticmethod
    def _normalize_trade(trade: Any) -> Dict[str, Any]:
        """Accept Trade ORM object or dict; return uniform dict."""
        if isinstance(trade, dict):
            d = dict(trade)
        else:
            d = {
                "id": getattr(trade, "id", None),
                "symbol": getattr(trade, "symbol", ""),
                "direction": _stringify(getattr(trade, "direction", "long")),
                "entry_time": getattr(trade, "entry_time", None),
                "exit_time": getattr(trade, "exit_time", None),
                "entry_price": getattr(trade, "entry_price", 0.0),
                "exit_price": getattr(trade, "exit_price", None),
                "stop_loss": getattr(trade, "stop_loss", None),
            }
        # Normalize direction string
        d["direction"] = (d.get("direction") or "long").lower().replace("tradedirection.", "")
        if "long" in d["direction"]:
            d["direction"] = "long"
        elif "short" in d["direction"]:
            d["direction"] = "short"
        return d

    @staticmethod
    def _empty_replay(trade_dict: Dict[str, Any], reason: str) -> ReplayedTrade:
        return ReplayedTrade(
            trade_id=str(trade_dict.get("id", "")),
            symbol=trade_dict.get("symbol", ""),
            direction=trade_dict.get("direction", "long"),
            entry_time=trade_dict.get("entry_time") or datetime.utcnow(),
            exit_time=trade_dict.get("exit_time"),
            entry_price=float(trade_dict.get("entry_price", 0.0)),
            exit_price=trade_dict.get("exit_price"),
            tier=REPLAY_TIER_SAMPLE,
            fetch_errors=[reason],
        )

    @staticmethod
    def _strip_tz(ts: Optional[datetime]) -> Optional[datetime]:
        if ts is None:
            return None
        if isinstance(ts, datetime) and ts.tzinfo is not None:
            return ts.replace(tzinfo=None)
        return ts

    @staticmethod
    def _candle_ts(candle: Dict[str, Any]) -> Optional[datetime]:
        ts = candle.get("timestamp") or candle.get("datetime") or candle.get("date")
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        try:
            return pd.to_datetime(ts).to_pydatetime()
        except Exception:
            return None


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    if "." in s:
        s = s.split(".")[-1]
    return s.lower()
