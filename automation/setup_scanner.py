"""
Technical Pattern Scanner
=========================
Continuous background scanner that detects trade setups in real-time.
Runs on a configurable tick interval and emits SETUP_DETECTED events
that the alert engine and WebSocket layer broadcast to the trader's browser.

Note: This is the *technical pattern* scanner (candle patterns, pullbacks, reversals).
For the *strategy readiness* scanner (conditions vs market environment),
see intelligence/setup_scanner.py.

Architecture:
  SetupScanner.scan_once(candles_by_symbol) — pure, synchronous, testable
  ScannerLoop                               — asyncio background task wrapper
  get_scanner_loop()                        — app-level singleton

Detectors used (all pure functions, no I/O):
  • strategy_engine.candle_patterns   — single/multi-candle patterns
  • strategy_engine.pullback_logic    — EMA pullback with Fibonacci depth
  • strategy_engine.reversal_logic    — S/R reversals, exhaustion, failed breakouts

Setup quality threshold: only setups scoring ≥ 50/100 are surfaced.
Each symbol is scanned at most once per DEBOUNCE_MINUTES to avoid alert fatigue.

Integration:
  Start scanner at startup via lifespan hook in api/main.py:
      from automation.setup_scanner import get_scanner_loop
      asyncio.create_task(get_scanner_loop().run())
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Callable, Optional

from core.events import Event, EventType, get_event_bus
from strategy_engine.candle_patterns import scan_all
from strategy_engine.pullback_logic import detect_pullback, PullbackSignal
from strategy_engine.reversal_logic import detect_reversal, ReversalSignal

# Strategy profile integration (lazy import to avoid circular deps)
_StrategyProfileModel = None

def _get_strategy_profile_model():
    global _StrategyProfileModel
    if _StrategyProfileModel is None:
        from core.models import StrategyProfile
        _StrategyProfileModel = StrategyProfile
    return _StrategyProfileModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_SCAN_INTERVAL_SECONDS = 60      # how often to re-scan each symbol
DEBOUNCE_MINUTES              = 15      # min gap between same-setup alerts per symbol
MIN_QUALITY_SCORE             = 50.0   # only emit setups scoring ≥ this
MIN_CANDLES_REQUIRED          = 55      # need at least 55 bars for EMA-50 + context


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SetupSignal:
    signal_id: str
    symbol: str
    setup_type: str          # "candle_pattern" | "pullback" | "reversal"
    setup_name: str          # e.g. "bullish_engulfing", "support_bounce"
    direction: str           # "long" | "short" | "neutral"
    quality_score: float     # 0-100
    entry_zone_low: float
    entry_zone_high: float
    suggested_stop: float
    target_1: float
    target_2: float
    description: str
    timeframe: str = "15m"   # trading timeframe this signal was detected on
    detected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "setup_type": self.setup_type,
            "setup_name": self.setup_name,
            "direction": self.direction,
            "quality_score": round(self.quality_score, 1),
            "entry_zone": {"low": self.entry_zone_low, "high": self.entry_zone_high},
            "suggested_stop": self.suggested_stop,
            "targets": {"t1": self.target_1, "t2": self.target_2},
            "description": self.description,
            "detected_at": self.detected_at,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Core scanner (pure, synchronous — easy to unit test)
# ---------------------------------------------------------------------------

class SetupScanner:
    """
    Scans a dict of {symbol: list[candle]} and returns all detected setups
    above the quality threshold.
    """

    # Detectors enabled per timeframe family
    # "fast"  = 1m/3m/5m  — intraday momentum/ORB/VWAP
    # "mid"   = 10m/15m   — compression breakout, pullback, S/R
    # "slow"  = 30m/1h+   — macro S/R, trend reversals
    _TF_DETECTORS: dict[str, set[str]] = {
        "fast": {"candle_pattern", "pullback", "orb", "vwap", "sr_breakout"},
        "mid":  {"candle_pattern", "pullback", "reversal", "sr_breakout", "vwap"},
        "slow": {"candle_pattern", "reversal", "sr_breakout"},
    }

    @staticmethod
    def _tf_family(timeframe: str) -> str:
        """Map a timeframe string to a detector family."""
        tf = timeframe.lower().replace("min", "m").replace("minute", "m")
        if tf in ("1m", "2m", "3m", "5m"):
            return "fast"
        if tf in ("10m", "15m", "20m"):
            return "mid"
        return "slow"  # 30m, 1h, 4h, 1d, etc.

    def scan_once(
        self,
        candles_by_symbol: dict[str, list[dict]],
        min_score: float = MIN_QUALITY_SCORE,
        timeframe: str = "15m",
    ) -> list[SetupSignal]:
        """
        candles_by_symbol: {symbol: [{open, high, low, close, volume}, ...]}
        timeframe: the candle interval these bars represent (used for routing
                   detectors and tagging signals).
        Returns list of SetupSignal for all qualifying setups.
        """
        tf_family  = self._tf_family(timeframe)
        detectors  = self._TF_DETECTORS.get(tf_family, self._TF_DETECTORS["mid"])
        signals: list[SetupSignal] = []

        for symbol, candles in candles_by_symbol.items():
            if len(candles) < MIN_CANDLES_REQUIRED:
                logger.debug(f"[{symbol}] Only {len(candles)} bars — skipping (need {MIN_CANDLES_REQUIRED})")
                continue

            current = candles[-1]["close"]

            # 1. Candle patterns (all timeframes)
            if "candle_pattern" in detectors:
                patterns = scan_all(candles)
                atr = self._estimate_atr(candles)
                for pattern in patterns:
                    score = pattern.strength * 100
                    if score < min_score:
                        continue
                    if pattern.direction == "bullish":
                        entry_lo = current * 0.998
                        entry_hi = current * 1.001
                        stop     = current - atr * 1.5
                        t1       = current + atr * 2.0
                        t2       = current + atr * 3.0
                    elif pattern.direction == "bearish":
                        entry_lo = current * 0.999
                        entry_hi = current * 1.002
                        stop     = current + atr * 1.5
                        t1       = current - atr * 2.0
                        t2       = current - atr * 3.0
                    else:
                        continue

                    signals.append(SetupSignal(
                        signal_id=str(uuid.uuid4()),
                        symbol=symbol,
                        setup_type="candle_pattern",
                        setup_name=pattern.pattern,
                        direction=pattern.direction,
                        quality_score=score,
                        entry_zone_low=round(entry_lo, 4),
                        entry_zone_high=round(entry_hi, 4),
                        suggested_stop=round(stop, 4),
                        target_1=round(t1, 4),
                        target_2=round(t2, 4),
                        description=pattern.description,
                        timeframe=timeframe,
                        metadata={"pattern_strength": pattern.strength},
                    ))

            # 2. Pullback setup (fast + mid)
            if "pullback" in detectors:
                pb: PullbackSignal = detect_pullback(candles)
                if pb.detected and pb.quality_score >= min_score:
                    signals.append(SetupSignal(
                        signal_id=str(uuid.uuid4()),
                        symbol=symbol,
                        setup_type="pullback",
                        setup_name=f"pullback_{pb.direction}",
                        direction=pb.direction,
                        quality_score=pb.quality_score,
                        entry_zone_low=pb.entry_zone_low,
                        entry_zone_high=pb.entry_zone_high,
                        suggested_stop=pb.suggested_stop,
                        target_1=round(pb.entry_zone_high + abs(pb.entry_zone_high - pb.suggested_stop) * 2, 4)
                                  if pb.direction == "long" else
                                  round(pb.entry_zone_low - abs(pb.entry_zone_low - pb.suggested_stop) * 2, 4),
                        target_2=round(pb.entry_zone_high + abs(pb.entry_zone_high - pb.suggested_stop) * 3, 4)
                                  if pb.direction == "long" else
                                  round(pb.entry_zone_low - abs(pb.entry_zone_low - pb.suggested_stop) * 3, 4),
                        description="; ".join(pb.reasons),
                        timeframe=timeframe,
                        metadata={"warnings": pb.warnings, "invalidation": pb.invalidation},
                    ))

            # 3. Reversal setup (mid + slow)
            if "reversal" in detectors:
                rv: ReversalSignal = detect_reversal(candles)
                if rv.detected and rv.quality_score >= min_score:
                    signals.append(SetupSignal(
                        signal_id=str(uuid.uuid4()),
                        symbol=symbol,
                        setup_type="reversal",
                        setup_name=rv.reversal_type,
                        direction=rv.direction,
                        quality_score=rv.quality_score,
                        entry_zone_low=rv.suggested_entry * 0.999,
                        entry_zone_high=rv.suggested_entry * 1.001,
                        suggested_stop=rv.suggested_stop,
                        target_1=rv.target_1,
                        target_2=rv.target_2,
                        description="; ".join(rv.reasons),
                        timeframe=timeframe,
                        metadata={"key_level": rv.key_level, "warnings": rv.warnings},
                    ))

            # 4. ORB Breakout / Breakdown — fast (5m) only
            if "orb" in detectors:
                try:
                    orb_sig = self.detect_orb_setup(symbol, candles)
                    if orb_sig and orb_sig.quality_score >= min_score:
                        orb_sig.timeframe = timeframe
                        signals.append(orb_sig)
                except Exception as _e:
                    logger.debug("ORB detection error for %s: %s", symbol, _e)

            # 5. VWAP Bounce / Rejection — fast + mid
            if "vwap" in detectors:
                try:
                    vwap_sig = self.detect_vwap_setup(symbol, candles)
                    if vwap_sig and vwap_sig.quality_score >= min_score:
                        vwap_sig.timeframe = timeframe
                        signals.append(vwap_sig)
                except Exception as _e:
                    logger.debug("VWAP detection error for %s: %s", symbol, _e)

            # 6. S/R Breakout / Breakdown — all timeframes
            if "sr_breakout" in detectors:
                try:
                    bo_sigs = self.detect_sr_breakout(symbol, candles, timeframe=timeframe)
                    for sig in bo_sigs:
                        if sig.quality_score >= min_score:
                            signals.append(sig)
                except Exception as _e:
                    logger.debug("SR breakout detection error for %s: %s", symbol, _e)

        return signals

    def detect_sr_breakout(
        self, symbol: str, candles: list[dict], timeframe: str = "15m"
    ) -> list["SetupSignal"]:
        """
        Wire intelligence.breakout_detector.BreakoutDetector into the scanner.

        Returns up to 2 signals:
          • Confirmed breakout/breakdown  (quality_score based on detector)
          • "Approaching resistance" pre-alert (quality 52) when within 0.8%
            of a key level AND price is in compression for 6+ bars
        """
        if len(candles) < 20:
            return []

        try:
            from intelligence.breakout_detector import BreakoutDetector
            from intelligence.market_environment import MarketEnvironmentAnalyzer
            from types import SimpleNamespace
        except ImportError as e:
            logger.debug("BreakoutDetector import failed: %s", e)
            return []

        # Convert candle dicts → lightweight bar objects (BreakoutDetector uses
        # attribute access: bar.close, bar.high, bar.low, bar.volume)
        bars = [
            SimpleNamespace(
                open=c["open"],
                high=c["high"],
                low=c["low"],
                close=c["close"],
                volume=c.get("volume", 0),
            )
            for c in candles
        ]

        atr = self._estimate_atr(candles)
        detector = BreakoutDetector()
        sig = detector.detect(bars, regime="unknown", atr=atr, timeframe=timeframe)

        results: list[SetupSignal] = []
        current = candles[-1]["close"]

        # ── Confirmed breakout / breakdown ────────────────────────────
        if sig.detected and sig.broken_level and sig.signal_type in ("breakout", "breakdown"):
            direction = "long" if sig.signal_type == "breakout" else "short"
            quality   = round(sig.quality_score * 100, 1)    # detector is 0-1; scanner is 0-100

            if direction == "long":
                entry_lo = sig.broken_level * 1.001
                entry_hi = sig.broken_level * 1.005
                stop     = sig.next_support or (sig.broken_level - atr * 1.5)
                t1       = sig.next_resistance or (current + atr * 2.0)
                t2       = t1 + atr * 1.5
            else:
                entry_hi = sig.broken_level * 0.999
                entry_lo = sig.broken_level * 0.995
                stop     = sig.next_resistance or (sig.broken_level + atr * 1.5)
                t1       = sig.next_support or (current - atr * 2.0)
                t2       = t1 - atr * 1.5

            vol_note  = "volume confirmed" if sig.volume_confirmed else "low volume"
            atr_note  = ", ATR expanding" if sig.atr_expansion else ""
            sr_pct    = int(sig.regime_success_rate * 100)
            results.append(SetupSignal(
                signal_id=str(uuid.uuid4()),
                symbol=symbol,
                setup_type="sr_breakout" if direction == "long" else "sr_breakdown",
                setup_name=sig.signal_type,
                direction=direction,
                quality_score=quality,
                entry_zone_low=round(entry_lo, 2),
                entry_zone_high=round(entry_hi, 2),
                suggested_stop=round(stop, 2),
                target_1=round(t1, 2),
                target_2=round(t2, 2),
                description=(
                    f"{symbol} broke ₹{sig.broken_level:.2f} "
                    f"({sig.distance_pct:.1f}% through, {vol_note}{atr_note}) — "
                    f"~{sr_pct}% regime success rate"
                ),
                timeframe=timeframe,
                metadata={
                    "broken_level": sig.broken_level,
                    "volume_confirmed": sig.volume_confirmed,
                    "atr_expansion": sig.atr_expansion,
                    "quality_score_raw": sig.quality_score,
                    "regime_success_rate": sig.regime_success_rate,
                    "notes": sig.notes,
                    "key_levels": [lv.to_dict() for lv in sig.key_levels[:5]],
                },
            ))

        # ── Approaching resistance pre-alert ──────────────────────────
        # Fire when: no confirmed breakout YET, price within 0.8% of next
        # resistance, AND stock is in compression (tight recent bars).
        elif (
            not sig.detected
            and sig.next_resistance
            and sig.signal_type not in ("prior_breakout", "prior_breakdown")
        ):
            gap_to_res = (sig.next_resistance - current) / current
            if 0 < gap_to_res <= 0.008:  # within 0.8% of resistance
                # Compression check: recent 8-bar avg range vs prior 20-bar avg range
                if len(candles) >= 28:
                    recent_ranges = [c["high"] - c["low"] for c in candles[-8:]]
                    prior_ranges  = [c["high"] - c["low"] for c in candles[-28:-8]]
                    avg_recent = sum(recent_ranges) / len(recent_ranges)
                    avg_prior  = sum(prior_ranges) / len(prior_ranges)
                    compression = avg_prior > 0 and (avg_recent / avg_prior) < 0.65

                    if compression:
                        t1 = sig.next_resistance + atr * 1.5
                        t2 = sig.next_resistance + atr * 3.0
                        stop_est = current - atr * 1.2
                        results.append(SetupSignal(
                            signal_id=str(uuid.uuid4()),
                            symbol=symbol,
                            setup_type="compression",
                            setup_name="approaching_resistance",
                            direction="long",
                            quality_score=52.0,   # below normal threshold by design — pre-alert
                            entry_zone_low=round(sig.next_resistance, 2),
                            entry_zone_high=round(sig.next_resistance * 1.003, 2),
                            suggested_stop=round(stop_est, 2),
                            target_1=round(t1, 2),
                            target_2=round(t2, 2),
                            description=(
                                f"{symbol} compressing — ₹{sig.next_resistance:.2f} resistance "
                                f"just {gap_to_res*100:.1f}% away, bars tightening "
                                f"({avg_recent/avg_prior*100:.0f}% of normal range). "
                                f"Watch for volume-confirmed break."
                            ),
                            timeframe=timeframe,
                            metadata={
                                "next_resistance": sig.next_resistance,
                                "gap_pct": round(gap_to_res * 100, 2),
                                "compression_ratio": round(avg_recent / avg_prior, 2),
                                "key_levels": [lv.to_dict() for lv in sig.key_levels[:5]],
                            },
                        ))

        return results

    @staticmethod
    def _estimate_atr(candles: list[dict], period: int = 14) -> float:
        """Simple ATR estimate from recent candles."""
        if len(candles) < 2:
            return candles[-1]["close"] * 0.01 if candles else 1.0
        trs = []
        for i in range(max(1, len(candles) - period), len(candles)):
            prev_close = candles[i - 1]["close"]
            h, l, c = candles[i]["high"], candles[i]["low"], prev_close
            trs.append(max(h - l, abs(h - c), abs(l - c)))
        return sum(trs) / len(trs) if trs else candles[-1]["close"] * 0.01

    @staticmethod
    def _today_candles(candles: list[dict]) -> list[dict]:
        """Filter candles to today's session only (using timestamp if available)."""
        today = date.today()
        dated = [c for c in candles if isinstance(c.get("timestamp"), datetime)
                 and c["timestamp"].date() == today]
        # Fallback: use last 78 bars (≈1 trading day on 5m)
        return dated if dated else candles[-78:]

    @staticmethod
    def _compute_vwap(candles: list[dict]) -> Optional[float]:
        """Compute intraday VWAP from candles with volume."""
        total_vol = sum(c.get("volume", 0) for c in candles)
        if total_vol == 0:
            return None
        total_pv = sum(
            ((c["high"] + c["low"] + c["close"]) / 3) * c.get("volume", 0)
            for c in candles
        )
        return total_pv / total_vol

    def detect_orb_setup(self, symbol: str, candles: list[dict]) -> Optional["SetupSignal"]:
        """
        Detect Opening Range Breakout / Breakdown.
        ORB = high/low of first 15 minutes (9:15-9:29 IST on 5m = first 3 bars).
        Fires when current price breaks out with volume confirmation.
        """
        today = self._today_candles(candles)
        if len(today) < 4:
            return None  # Not enough intraday bars yet

        # ORB window: bars with time < 09:30
        orb_bars = [c for c in today
                    if isinstance(c.get("timestamp"), datetime)
                    and c["timestamp"].time() < time(9, 30)]
        if len(orb_bars) < 2:
            # Fallback: use first 3 intraday bars
            orb_bars = today[:3]

        orb_high = max(c["high"] for c in orb_bars)
        orb_low  = min(c["low"]  for c in orb_bars)
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            return None

        current = today[-1]["close"]
        current_high = today[-1]["high"]
        current_low  = today[-1]["low"]

        # Volume confirmation: last bar volume > 1.3x avg of today's bars
        avg_vol = sum(c.get("volume", 0) for c in today) / len(today) if today else 0
        last_vol = today[-1].get("volume", 0)
        vol_ok = last_vol > avg_vol * 1.2

        atr = self._estimate_atr(candles)

        if current_high > orb_high and vol_ok:
            # Bullish ORB breakout
            stop  = orb_low
            t1    = orb_high + orb_range
            t2    = orb_high + orb_range * 2
            score = min(90.0, 65.0 + (last_vol / max(avg_vol, 1) - 1.2) * 20)
            return SetupSignal(
                signal_id=str(uuid.uuid4()),
                symbol=symbol,
                setup_type="orb_breakout",
                setup_name="orb_breakout",
                direction="long",
                quality_score=round(score, 1),
                entry_zone_low=round(orb_high, 2),
                entry_zone_high=round(orb_high * 1.002, 2),
                suggested_stop=round(stop, 2),
                target_1=round(t1, 2),
                target_2=round(t2, 2),
                description=(
                    f"ORB breakout: price {current:.2f} crossed ORB high {orb_high:.2f} "
                    f"(range {orb_range:.2f}, vol {last_vol/avg_vol:.1f}x avg)"
                ),
                metadata={"orb_high": orb_high, "orb_low": orb_low, "orb_range": orb_range},
            )

        if current_low < orb_low and vol_ok:
            # Bearish ORB breakdown
            stop  = orb_high
            t1    = orb_low - orb_range
            t2    = orb_low - orb_range * 2
            score = min(90.0, 65.0 + (last_vol / max(avg_vol, 1) - 1.2) * 20)
            return SetupSignal(
                signal_id=str(uuid.uuid4()),
                symbol=symbol,
                setup_type="orb_breakdown",
                setup_name="orb_breakdown",
                direction="short",
                quality_score=round(score, 1),
                entry_zone_low=round(orb_low * 0.998, 2),
                entry_zone_high=round(orb_low, 2),
                suggested_stop=round(stop, 2),
                target_1=round(t1, 2),
                target_2=round(t2, 2),
                description=(
                    f"ORB breakdown: price {current:.2f} broke ORB low {orb_low:.2f} "
                    f"(range {orb_range:.2f}, vol {last_vol/avg_vol:.1f}x avg)"
                ),
                metadata={"orb_high": orb_high, "orb_low": orb_low, "orb_range": orb_range},
            )
        return None

    def detect_vwap_setup(self, symbol: str, candles: list[dict]) -> Optional["SetupSignal"]:
        """
        Detect VWAP bounce or rejection setup.
        Price touches VWAP within 0.3% and shows reversal candle with volume.
        """
        today = self._today_candles(candles)
        if len(today) < 10:
            return None

        vwap = self._compute_vwap(today)
        if not vwap:
            return None

        current     = today[-1]["close"]
        prev        = today[-2]["close"]
        low_bar     = today[-1]["low"]
        high_bar    = today[-1]["high"]

        proximity_pct = abs(current - vwap) / vwap
        if proximity_pct > 0.003:  # must be within 0.3% of VWAP
            return None

        atr = self._estimate_atr(candles)
        avg_vol = sum(c.get("volume", 0) for c in today) / len(today)
        last_vol = today[-1].get("volume", 0)
        vol_ok = last_vol > avg_vol * 1.15

        # Bullish VWAP bounce: price dipped below VWAP and closed back above
        if prev < vwap and current > vwap and vol_ok:
            stop  = low_bar - atr * 0.5
            t1    = vwap + atr * 1.5
            t2    = vwap + atr * 2.5
            return SetupSignal(
                signal_id=str(uuid.uuid4()),
                symbol=symbol,
                setup_type="vwap_bounce",
                setup_name="vwap_bounce_long",
                direction="long",
                quality_score=72.0,
                entry_zone_low=round(vwap, 2),
                entry_zone_high=round(vwap * 1.001, 2),
                suggested_stop=round(stop, 2),
                target_1=round(t1, 2),
                target_2=round(t2, 2),
                description=(
                    f"VWAP bounce: price reclaimed VWAP ₹{vwap:.2f} "
                    f"(current ₹{current:.2f}, vol {last_vol/avg_vol:.1f}x)"
                ),
                metadata={"vwap": vwap, "proximity_pct": round(proximity_pct * 100, 3)},
            )

        # Bearish VWAP rejection: price pushed above VWAP and closed back below
        if prev > vwap and current < vwap and vol_ok:
            stop  = high_bar + atr * 0.5
            t1    = vwap - atr * 1.5
            t2    = vwap - atr * 2.5
            return SetupSignal(
                signal_id=str(uuid.uuid4()),
                symbol=symbol,
                setup_type="vwap_rejection",
                setup_name="vwap_rejection_short",
                direction="short",
                quality_score=70.0,
                entry_zone_low=round(vwap * 0.999, 2),
                entry_zone_high=round(vwap, 2),
                suggested_stop=round(stop, 2),
                target_1=round(t1, 2),
                target_2=round(t2, 2),
                description=(
                    f"VWAP rejection: price failed above VWAP ₹{vwap:.2f} "
                    f"(current ₹{current:.2f}, vol {last_vol/avg_vol:.1f}x)"
                ),
                metadata={"vwap": vwap, "proximity_pct": round(proximity_pct * 100, 3)},
            )
        return None


# ---------------------------------------------------------------------------
# Candle data fetcher interface
# ---------------------------------------------------------------------------

CandleFetchFn = Callable[[str, int, str], list[dict]]
"""
Type alias for a function that fetches candles:
    fetch(symbol, n_bars, interval) -> list[{open, high, low, close, volume}]

interval: timeframe string, e.g. "5m", "15m", "1h"
"""


def _noop_fetcher(symbol: str, n_bars: int, interval: str = "15m") -> list[dict]:
    """Default no-op fetcher — returns empty list (scanner does nothing without data)."""
    return []


# ---------------------------------------------------------------------------
# Async scanner loop
# ---------------------------------------------------------------------------

class ScannerLoop:
    """
    Runs SetupScanner on a periodic asyncio task.
    Debounces repeat alerts per symbol+setup_type.
    Publishes detected setups as EventType.SETUP_DETECTED events.
    """

    # Default timeframe when a strategy/symbol has no explicit preference
    DEFAULT_TIMEFRAME = "15m"

    def __init__(
        self,
        symbols: list[str],
        fetch_candles: CandleFetchFn = _noop_fetcher,
        scan_interval: int = DEFAULT_SCAN_INTERVAL_SECONDS,
        min_score: float = MIN_QUALITY_SCORE,
        n_bars: int = 100,
    ):
        # _symbol_timeframes: symbol → set of timeframes to scan on
        # e.g. {"RELIANCE": {"5m", "15m"}, "TCS": {"15m"}}
        self._symbol_timeframes: dict[str, set[str]] = {
            s: {self.DEFAULT_TIMEFRAME} for s in symbols
        }
        self._fetch         = fetch_candles
        self._interval      = scan_interval
        self._min_score     = min_score
        self._n_bars        = n_bars
        self._scanner       = SetupScanner()
        self._bus           = get_event_bus()
        self._running       = False
        self._task: Optional[asyncio.Task] = None
        # Debounce: (symbol, setup_name, timeframe) → last alert timestamp
        self._last_alert: dict[tuple[str, str, str], datetime] = {}

    @property
    def symbols(self) -> list[str]:
        return list(self._symbol_timeframes.keys())

    def add_symbol(self, symbol: str, timeframes: Optional[list[str]] = None) -> None:
        tfs = set(timeframes or [self.DEFAULT_TIMEFRAME])
        if symbol not in self._symbol_timeframes:
            self._symbol_timeframes[symbol] = tfs
            logger.info("SetupScanner: added %s [%s]", symbol, ",".join(tfs))
        else:
            before = len(self._symbol_timeframes[symbol])
            self._symbol_timeframes[symbol].update(tfs)
            added = self._symbol_timeframes[symbol] - set()
            if len(self._symbol_timeframes[symbol]) > before:
                logger.info("SetupScanner: added timeframes %s for %s", tfs - self._symbol_timeframes[symbol], symbol)

    def remove_symbol(self, symbol: str) -> None:
        if symbol in self._symbol_timeframes:
            del self._symbol_timeframes[symbol]
            logger.info("SetupScanner: removed %s", symbol)

    def set_fetcher(self, fetch_fn: CandleFetchFn) -> None:
        """Inject a real candle fetcher (e.g. from Zerodha/IBKR live feed)."""
        self._fetch = fetch_fn

    def set_db_factory(self, db_factory: Callable) -> None:
        """Inject DB session factory for loading StrategyProfile watchlists."""
        self._db_factory = db_factory

    def sync_from_strategy_profiles(self) -> dict:
        """
        Load active StrategyProfile records from DB and auto-populate
        scanner watchlist with symbols from their tags/metadata.

        Also caches strategy definitions so detected setups can be
        enriched with matched strategy info (direction, regime prefs).

        Called at startup and periodically during scan cycles.
        """
        if not hasattr(self, "_db_factory") or self._db_factory is None:
            return {"synced": False, "reason": "no db_factory"}

        try:
            StrategyProfileModel = _get_strategy_profile_model()
            db = self._db_factory()
            try:
                profiles = (
                    db.query(StrategyProfileModel)
                    .filter(StrategyProfileModel.status.in_(["active", "learning"]))
                    .all()
                )

                if not profiles:
                    return {"synced": True, "profiles": 0, "symbols_added": 0}

                # Cache strategy definitions for setup enrichment
                self._strategy_profiles = {}
                symbols_added = 0

                for profile in profiles:
                    # Resolve preferred timeframes for this profile
                    _meta = profile.profile_metadata or {}
                    _raw_tfs = _meta.get("preferred_timeframes") or _meta.get("timeframes") or []
                    _single  = _meta.get("preferred_timeframe")
                    if _single and _single not in _raw_tfs:
                        _raw_tfs = [_single] + list(_raw_tfs)
                    _pref_tfs = [t.strip() for t in _raw_tfs if t] or [self.DEFAULT_TIMEFRAME]

                    profile_info = {
                        "id": str(profile.id),
                        "name": profile.name,
                        "short_code": profile.short_code,
                        "entry_criteria": profile.entry_criteria or [],
                        "regime_preferences": profile.regime_preferences or {},
                        "risk_parameters": profile.risk_parameters or {},
                        "confidence_scores": profile.confidence_scores or {},
                        "preferred_timeframes": _pref_tfs,
                    }
                    self._strategy_profiles[profile.short_code] = profile_info

                    # Extract preferred timeframes for this strategy profile
                    meta = profile.profile_metadata or {}
                    raw_tfs = meta.get("preferred_timeframes") or meta.get("timeframes") or []
                    # Also accept a single string "preferred_timeframe"
                    single_tf = meta.get("preferred_timeframe")
                    if single_tf and single_tf not in raw_tfs:
                        raw_tfs = [single_tf] + list(raw_tfs)
                    # Normalise (remove None, strip whitespace)
                    profile_tfs = [t.strip() for t in raw_tfs if t] or [self.DEFAULT_TIMEFRAME]

                    # Extract watchlist symbols from profile metadata/tags
                    tags = profile.tags or []
                    watchlist = meta.get("watchlist", [])
                    # Also check tags for symbols (uppercase, 1-20 chars)
                    for tag in tags:
                        if isinstance(tag, str) and tag.isupper() and 1 <= len(tag) <= 20:
                            watchlist.append(tag)

                    for sym in watchlist:
                        sym = sym.strip().upper()
                        if not sym:
                            continue
                        if sym not in self._symbol_timeframes:
                            self._symbol_timeframes[sym] = set(profile_tfs)
                            symbols_added += 1
                        else:
                            self._symbol_timeframes[sym].update(profile_tfs)

                # Also pull symbols from Smart Picks + strategy watchlists.
                # Strategy watchlists are linked to StrategyProfiles, so their
                # preferred timeframes can be looked up; Smart Picks default to DEFAULT_TIMEFRAME.
                try:
                    from core.models import Watchlist, WatchlistItem
                    watchlists = db.query(Watchlist).filter(
                        Watchlist.watchlist_type == "strategy",
                        Watchlist.is_active == True,
                    ).all()
                    for wl in watchlists:
                        # Resolve this watchlist's preferred timeframes
                        wl_tfs: set[str] = {self.DEFAULT_TIMEFRAME}
                        if wl.strategy_profile_id:
                            # Look up the linked profile's timeframes from our cache
                            for code, pinfo in self._strategy_profiles.items():
                                if pinfo.get("id") == str(wl.strategy_profile_id):
                                    wl_tfs = set(pinfo.get("preferred_timeframes", [self.DEFAULT_TIMEFRAME]))
                                    break

                        items = db.query(WatchlistItem).filter(
                            WatchlistItem.watchlist_id == wl.id,
                            WatchlistItem.status == "matched",
                        ).all()
                        for item in items:
                            sym = item.symbol.strip().upper()
                            if not sym:
                                continue
                            if sym not in self._symbol_timeframes:
                                self._symbol_timeframes[sym] = set(wl_tfs)
                                symbols_added += 1
                            else:
                                self._symbol_timeframes[sym].update(wl_tfs)
                except Exception as _we:
                    logger.debug("Strategy watchlist sync skipped: %s", _we)

                total_symbols = len(self._symbol_timeframes)
                # Build a compact tf→count summary for logging
                tf_counts: dict[str, int] = {}
                for tfs in self._symbol_timeframes.values():
                    for tf in tfs:
                        tf_counts[tf] = tf_counts.get(tf, 0) + 1
                logger.info(
                    "Scanner synced: %d profiles, %d symbols added, %d total | TF breakdown: %s",
                    len(profiles), symbols_added, total_symbols,
                    ", ".join(f"{tf}:{cnt}" for tf, cnt in sorted(tf_counts.items())),
                )
                return {
                    "synced": True,
                    "profiles": len(profiles),
                    "symbols_added": symbols_added,
                    "total_symbols": total_symbols,
                    "timeframe_breakdown": tf_counts,
                    "strategies": list(self._strategy_profiles.keys()),
                }
            finally:
                db.close()
        except Exception as e:
            logger.warning("Strategy profile sync failed: %s", e)
            return {"synced": False, "reason": str(e)}

    def refresh_strategy_templates(self) -> None:
        """
        Load/reload strategy templates from the YAML registry.
        Called at startup and after STRATEGY_RELOADED event so the scanner
        is instantly aware of any new user-added strategies.
        """
        try:
            from strategy_engine.strategy_registry import get_registry
            reg = get_registry()
            if not reg.is_initialized:
                reg.initialize()
            self._strategy_templates = {s.id: s for s in reg.get_all_active()}
            logger.info("Scanner loaded %d strategy templates from registry", len(self._strategy_templates))
        except Exception as e:
            logger.warning("Could not load strategy templates: %s", e)
            self._strategy_templates = {}

    # Setup type → strategy template IDs that handle it
    _SETUP_STRATEGY_MAP: dict[str, list[str]] = {
        "orb_breakout":       ["orb"],
        "orb_breakdown":      ["orb"],
        "vwap_bounce":        ["vwap_bounce"],
        "vwap_rejection":     ["vwap_bounce"],
        "pullback":           ["ema_pullback"],
        "reversal":           ["momentum_exhaustion", "vwap_bounce"],
        "candle_pattern":     ["compression_breakout", "ema_pullback"],
        "compression":        ["compression_breakout"],
        "gap_fill":           ["gap_fill"],
    }

    def get_matching_strategies(self, signal: SetupSignal) -> list[dict]:
        """
        Given a detected setup signal, find which user strategies match.
        Checks both YAML strategy templates (primary) and DB StrategyProfiles.
        Returns enriched matches with strategy name + readiness hint.
        """
        matches: list[dict] = []
        seen: set[str] = set()

        # 1. Match against YAML strategy templates (ORB, VWAP, EMA Pullback, etc.)
        templates = getattr(self, "_strategy_templates", {})
        candidate_ids = self._SETUP_STRATEGY_MAP.get(signal.setup_type, [])
        # Also check setup_name for more specific matches
        for sid in list(candidate_ids):
            if sid in templates and sid not in seen:
                t = templates[sid]
                matches.append({
                    "strategy_id": sid,
                    "strategy_name": t.name,
                    "match_type": "setup_type",
                    "fit": "primary",
                })
                seen.add(sid)

        # Also scan all templates for keyword match on setup name
        for sid, t in templates.items():
            if sid in seen:
                continue
            t_desc = (getattr(t, "description", "") or "").lower()
            t_name = t.name.lower()
            sig_type = signal.setup_type.lower()
            sig_name = signal.setup_name.lower()
            if any(kw in t_desc or kw in t_name for kw in [sig_type, sig_name]):
                matches.append({
                    "strategy_id": sid,
                    "strategy_name": t.name,
                    "match_type": "keyword",
                    "fit": "secondary",
                })
                seen.add(sid)

        # 2. Also check DB-loaded StrategyProfiles (user-taught strategies)
        for code, profile in getattr(self, "_strategy_profiles", {}).items():
            if code in seen:
                continue
            criteria = profile.get("entry_criteria", [])
            for criterion in criteria:
                desc = (criterion.get("description", "") or "").lower()
                if signal.setup_name.lower() in desc or signal.setup_type.lower() in desc:
                    matches.append({
                        "strategy_id": code,
                        "strategy_name": profile["name"],
                        "match_type": "db_profile",
                        "fit": "user_defined",
                    })
                    seen.add(code)
                    break

        return matches

    async def run(self) -> None:
        """Main asyncio loop — call once as a background task."""
        self._running = True
        logger.info(
            "SetupScanner loop started: %d symbols, interval=%ss, min_score=%s",
            len(self._symbol_timeframes), self._interval, self._min_score,
        )
        while self._running:
            try:
                await self._scan_cycle()
            except Exception as e:
                logger.error(f"SetupScanner cycle error: {e}", exc_info=True)
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("SetupScanner loop stopped")

    async def _scan_cycle(self) -> None:
        """Fetch candles for all symbols and run the scanner."""
        # Periodically re-sync from StrategyProfile (picks up new strategies/symbols)
        if not hasattr(self, "_last_profile_sync"):
            self._last_profile_sync = None
        now = datetime.utcnow()
        if (
            self._last_profile_sync is None
            or (now - self._last_profile_sync) > timedelta(minutes=5)
        ):
            self.sync_from_strategy_profiles()
            self.refresh_strategy_templates()  # picks up any newly added strategies
            self._last_profile_sync = now

        if not self._symbol_timeframes:
            return

        # Build (symbol, timeframe) pairs to fetch
        pairs: list[tuple[str, str]] = []
        for sym, tfs in self._symbol_timeframes.items():
            for tf in tfs:
                pairs.append((sym, tf))

        # Fetch candles concurrently — one executor task per (symbol, timeframe)
        loop = asyncio.get_event_loop()
        fetch_tasks: dict[tuple[str, str], asyncio.Future] = {}
        for sym, tf in pairs:
            fetch_tasks[(sym, tf)] = loop.run_in_executor(
                None, self._fetch, sym, self._n_bars, tf
            )

        # Collect results grouped by timeframe
        candles_by_tf: dict[str, dict[str, list[dict]]] = {}
        for (sym, tf), coro in fetch_tasks.items():
            try:
                candles = await coro
                if candles:
                    candles_by_tf.setdefault(tf, {})[sym] = candles
            except Exception as e:
                logger.warning("Candle fetch failed for %s@%s: %s", sym, tf, e)

        if not candles_by_tf:
            return

        # Scan each timeframe group with appropriate detectors
        for tf, candles_by_symbol in candles_by_tf.items():
            signals = self._scanner.scan_once(
                candles_by_symbol, self._min_score, timeframe=tf
            )
            for sig in signals:
                if self._is_debounced(sig.symbol, sig.setup_name, sig.timeframe):
                    logger.debug("[%s@%s] %s debounced", sig.symbol, sig.timeframe, sig.setup_name)
                    continue
                self._record_alert(sig.symbol, sig.setup_name, sig.timeframe)
                self._emit(sig)

    def _is_debounced(self, symbol: str, setup_name: str, timeframe: str = "15m") -> bool:
        key = (symbol, setup_name, timeframe)
        last = self._last_alert.get(key)
        if last is None:
            return False
        return (datetime.utcnow() - last) < timedelta(minutes=DEBOUNCE_MINUTES)

    def _record_alert(self, symbol: str, setup_name: str, timeframe: str = "15m") -> None:
        self._last_alert[(symbol, setup_name, timeframe)] = datetime.utcnow()

    def _emit(self, sig: SetupSignal) -> None:
        """Publish the setup signal as an event (WebSocket picks it up)."""
        event_data = sig.to_dict()

        # Enrich with matched user strategies
        matched = self.get_matching_strategies(sig)
        if matched:
            event_data["matched_strategies"] = matched

        self._bus.publish(Event(
            event_type=EventType.SETUP_DETECTED,
            source="setup_scanner",
            data=event_data,
        ))
        strategy_note = ""
        if matched:
            codes = [m.get("strategy_id", m.get("strategy_name", "?")) for m in matched]
            strategy_note = f" -> matches [{', '.join(codes)}]"
        logger.info(
            f"[SETUP] {sig.symbol} — {sig.setup_name} ({sig.direction}) "
            f"score={sig.quality_score:.0f}{strategy_note}"
        )

        # Persist setup_status to WatchlistItem.last_analysis for real-time Strategy tab
        self._update_watchlist_status(sig)

    def _update_watchlist_status(self, sig: SetupSignal) -> None:
        """Write setup_status tag back to all strategy watchlist items for this symbol."""
        if not hasattr(self, "_db_factory") or self._db_factory is None:
            return

        # Map setup type + score → status tag
        if sig.setup_type in ("sr_breakout", "sr_breakdown", "orb_breakout", "orb_breakdown") and sig.quality_score >= 70:
            setup_status = "TRIGGERED"
        elif sig.setup_type in ("sr_breakout", "sr_breakdown", "pullback", "reversal") and sig.quality_score >= 55:
            setup_status = "READY"
        elif sig.setup_type == "compression":
            setup_status = "FORMING"
        else:
            return  # Don't overwrite status for low-confidence candle patterns

        try:
            from core.models import WatchlistItem, Watchlist
            db = self._db_factory()
            try:
                items = (
                    db.query(WatchlistItem)
                    .join(Watchlist, WatchlistItem.watchlist_id == Watchlist.id)
                    .filter(
                        Watchlist.watchlist_type == "strategy",
                        Watchlist.is_active == True,
                        WatchlistItem.symbol == sig.symbol,
                        WatchlistItem.status.in_(["matched", "active"]),
                    )
                    .all()
                )
                for item in items:
                    analysis = dict(item.last_analysis or {})
                    analysis["setup_status"] = setup_status
                    analysis["last_signal_type"] = sig.setup_name
                    analysis["last_signal_score"] = round(sig.quality_score, 1)
                    analysis["last_signal_at"] = datetime.utcnow().isoformat()
                    item.last_analysis = analysis
                if items:
                    db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.debug("WatchlistItem status update failed for %s: %s", sig.symbol, e)

    # ------------------------------------------------------------------
    # Synchronous scan (for REST endpoints / on-demand use)
    # ------------------------------------------------------------------

    def scan_now(self, symbol: str, timeframe: Optional[str] = None) -> list[SetupSignal]:
        """
        Immediate synchronous scan for a single symbol.
        Used by the /api/v1/intelligence/scan/watchlist endpoint.
        Uses the symbol's registered timeframes (or DEFAULT_TIMEFRAME if not registered).
        """
        tfs = self._symbol_timeframes.get(symbol, {timeframe or self.DEFAULT_TIMEFRAME})
        signals: list[SetupSignal] = []
        for tf in tfs:
            candles = self._fetch(symbol, self._n_bars, tf)
            if candles:
                signals.extend(
                    self._scanner.scan_once({symbol: candles}, self._min_score, timeframe=tf)
                )
        return signals

    def scan_watchlist_now(self) -> list[SetupSignal]:
        """Synchronous scan of all registered symbols at their registered timeframes."""
        signals: list[SetupSignal] = []
        for sym, tfs in self._symbol_timeframes.items():
            for tf in tfs:
                try:
                    candles = self._fetch(sym, self._n_bars, tf)
                    if candles:
                        signals.extend(
                            self._scanner.scan_once({sym: candles}, self._min_score, timeframe=tf)
                        )
                except Exception as e:
                    logger.warning("Candle fetch failed for %s@%s: %s", sym, tf, e)
        return signals


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_loop: Optional[ScannerLoop] = None


def get_scanner_loop(
    symbols: Optional[list[str]] = None,
    fetch_candles: CandleFetchFn = _noop_fetcher,
    scan_interval: int = DEFAULT_SCAN_INTERVAL_SECONDS,
) -> ScannerLoop:
    """
    Return (or create) the module-level ScannerLoop singleton.

    First call should pass symbols + fetch_candles to configure it.
    Subsequent calls return the existing instance.
    """
    global _loop
    if _loop is None:
        _loop = ScannerLoop(
            symbols=symbols or [],
            fetch_candles=fetch_candles,
            scan_interval=scan_interval,
        )
        logger.info(
            f"ScannerLoop created: {len(_loop.symbols)} symbols, "
            f"interval={scan_interval}s"
        )
    return _loop
