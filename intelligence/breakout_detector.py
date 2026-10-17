"""
Breakout / Breakdown Detector
==============================

Detects breakouts and breakdowns from OHLCV bar data using pivot-based
support/resistance levels.  Provides:

  • Key S/R levels (swing highs/lows, clustered within ATR)
  • Breakout / breakdown signal with quality score 0-1
  • Regime-adjusted success rate estimate
  • Human-readable notes for the UI and coaching layer

No additional API calls required — works from the same candle series
already fetched for the market environment analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class KeyLevel:
    """A significant support or resistance price level."""
    price: float
    level_type: str          # "resistance" | "support"
    touches: int = 1         # how many times price has tested this level
    strength: float = 0.5    # 0-1 composite strength

    def to_dict(self) -> dict:
        return {
            "price": round(self.price, 2),
            "type": self.level_type,
            "touches": self.touches,
            "strength": round(self.strength, 2),
        }


@dataclass
class BreakoutSignal:
    """Result of breakout/breakdown detection."""
    detected: bool = False
    signal_type: str = "none"       # "breakout" | "breakdown" | "prior_breakout" | "prior_breakdown" | "none"
    broken_level: Optional[float] = None
    distance_pct: float = 0.0       # % price is above/below broken level
    volume_confirmed: bool = False
    atr_expansion: bool = False
    quality_score: float = 0.0      # 0-1 composite quality
    regime_success_rate: float = 0.0  # estimated historical win % for this pattern
    next_resistance: Optional[float] = None   # nearest level above (for breakouts)
    next_support: Optional[float] = None      # nearest level below (for breakdowns)
    room_to_next_level_pct: float = 0.0       # % move until next S/R
    key_levels: List[KeyLevel] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "signal_type": self.signal_type,
            "broken_level": round(self.broken_level, 2) if self.broken_level else None,
            "distance_pct": round(self.distance_pct, 2),
            "volume_confirmed": self.volume_confirmed,
            "atr_expansion": self.atr_expansion,
            "quality_score": round(self.quality_score, 2),
            "regime_success_rate": round(self.regime_success_rate, 2),
            "next_resistance": round(self.next_resistance, 2) if self.next_resistance else None,
            "next_support": round(self.next_support, 2) if self.next_support else None,
            "room_to_next_level_pct": round(self.room_to_next_level_pct, 2),
            "key_levels": [lv.to_dict() for lv in self.key_levels],
            "notes": self.notes,
        }


# ═══════════════════════════════════════════════════════════════════
# REGIME SUCCESS RATES
# These are empirically derived estimates for NSE FnO universe.
# breakouts in trending/compression regimes succeed far more often
# than in ranging/volatile regimes.
# ═══════════════════════════════════════════════════════════════════

_REGIME_BREAKOUT_SUCCESS: dict[str, float] = {
    "trending_bullish":          0.65,
    "trending_bearish":          0.30,   # breakouts against bearish trend fail
    "breakout":                  0.70,   # regime itself is in breakout mode
    "breakdown":                 0.65,
    "low_volatility_compression": 0.72,  # compression → expansion = highest
    "volatile_expansion":        0.50,
    "ranging":                   0.35,   # most breakouts fail in ranging
    "unknown":                   0.45,
}

_REGIME_BREAKDOWN_SUCCESS: dict[str, float] = {
    "trending_bearish":          0.65,
    "trending_bullish":          0.30,
    "breakdown":                 0.70,
    "breakout":                  0.30,
    "low_volatility_compression": 0.65,
    "volatile_expansion":        0.50,
    "ranging":                   0.35,
    "unknown":                   0.45,
}


# ═══════════════════════════════════════════════════════════════════
# DETECTOR
# ═══════════════════════════════════════════════════════════════════

class BreakoutDetector:
    """
    Pivot-based breakout / breakdown detector.

    Usage
    -----
    >>> from intelligence.market_environment import OHLCVBar
    >>> detector = BreakoutDetector()
    >>> signal = detector.detect(bars, regime="ranging", atr=30.5)
    """

    def __init__(
        self,
        pivot_lookback: int = 3,       # bars each side for swing high/low
        sr_lookback: int = 100,        # bars to search for pivots
        cluster_atr_frac: float = 0.4, # levels within 0.4×ATR → merge
        min_distance_pct: float = 0.1, # min % above level to call breakout
    ) -> None:
        self.pivot_lookback = pivot_lookback
        self.sr_lookback = sr_lookback
        self.cluster_atr_frac = cluster_atr_frac
        self.min_distance_pct = min_distance_pct

    # ── Public entry point ───────────────────────────────────────

    # NSE session length in bars per timeframe
    _SESSION_BARS: dict[str, int] = {
        "1m": 375, "5m": 75, "15m": 25, "30m": 13,
        "1h": 7,   "60m": 7, "1d": 252, "D": 252,
    }

    def detect(
        self,
        bars: Sequence,           # OHLCVBar sequence, newest last
        regime: str = "unknown",
        atr: Optional[float] = None,
        timeframe: str = "5m",
        vwap: Optional[float] = None,  # inject VWAP as a structural level
    ) -> BreakoutSignal:
        """
        Run detection on the provided bar sequence.

        Parameters
        ----------
        bars      : OHLCV bar sequence (oldest first, newest last)
        regime    : Current market regime string
        atr       : Average True Range (used for level clustering)
        timeframe : Candle timeframe — limits S/R search to current session
                    for intraday TFs so multi-day lows don't pollute results.
        """
        if len(bars) < max(self.pivot_lookback * 2 + 1, 15):
            return BreakoutSignal(notes=["Insufficient bars for breakout detection"])

        # For intraday timeframes use 2 sessions of bars so structure from
        # earlier in the day (where breakout often originates) is captured.
        session_bars = self._SESSION_BARS.get(timeframe, 75) * 2
        effective_lookback = min(self.sr_lookback, session_bars)

        # Use last effective_lookback bars for pivot search
        analysis_bars = bars[-effective_lookback:]
        current_bar  = bars[-1]
        current_price = current_bar.close

        # ATR default: use last 14 bars' avg range
        if atr is None or atr <= 0:
            atr = self._simple_atr(bars[-14:])
        if atr <= 0:
            atr = current_price * 0.01  # 1% fallback

        # ── Find pivot highs and lows ──────────────────────────
        pivot_highs, pivot_lows = self._find_pivots(analysis_bars)

        # ── Cluster nearby levels ──────────────────────────────
        resistance_levels = self._cluster_levels(
            [b.high for b in analysis_bars],
            pivot_highs, atr, "resistance"
        )
        support_levels = self._cluster_levels(
            [b.low for b in analysis_bars],
            pivot_lows, atr, "support"
        )

        # Merge and sort
        all_levels = sorted(resistance_levels + support_levels, key=lambda l: l.price)

        # ── Check breakout / breakdown ─────────────────────────
        prev_close = bars[-2].close if len(bars) >= 2 else current_price

        # Average volume for confirmation
        volumes = [b.volume for b in bars[-21:-1] if b.volume > 0]
        avg_vol = sum(volumes) / len(volumes) if volumes else 0

        # Inject VWAP as a structural support/resistance level
        if vwap and vwap > 0:
            vwap_level = KeyLevel(
                price=vwap,
                level_type="support" if current_price > vwap else "resistance",
                touches=3,           # VWAP is tested continuously
                strength=0.70,
            )
            if current_price > vwap:
                support_levels.append(vwap_level)
            else:
                resistance_levels.append(vwap_level)
            all_levels = sorted(resistance_levels + support_levels, key=lambda l: l.price)

        signal = self._evaluate_signal(
            current_bar=current_bar,
            prev_close=prev_close,
            resistance_levels=resistance_levels,
            support_levels=support_levels,
            all_levels=all_levels,
            atr=atr,
            avg_vol=avg_vol,
            regime=regime,
        )

        # Sort key_levels by distance from current price (nearest first)
        all_levels_sorted = sorted(
            all_levels,
            key=lambda l: abs(l.price - current_price)
        )
        signal.key_levels = all_levels_sorted[:10]  # 10 nearest levels for UI
        return signal

    # ── Core evaluation ──────────────────────────────────────────

    def _evaluate_signal(
        self,
        current_bar,
        prev_close: float,
        resistance_levels: List[KeyLevel],
        support_levels: List[KeyLevel],
        all_levels: List[KeyLevel],
        atr: float,
        avg_vol: float,
        regime: str,
    ) -> BreakoutSignal:
        price = current_bar.close
        bar_range = current_bar.high - current_bar.low
        volume_confirmed = avg_vol > 0 and current_bar.volume > avg_vol * 1.2
        atr_expansion = bar_range > atr * 1.4

        # ── Look for broken resistance (BREAKOUT) ──────────────
        broken_resistance = None
        prior_breakout_level = None     # resistance broken on an earlier bar
        for level in sorted(resistance_levels, key=lambda l: l.price):
            if level.price < price * 0.995:   # price is above by ≥ 0.5%
                distance_pct = (price - level.price) / level.price * 100
                if distance_pct < self.min_distance_pct:
                    continue
                # Was price below this level on previous bar? → current-bar breakout
                if prev_close <= level.price * 1.002:
                    broken_resistance = (level, distance_pct)
                    break
                # prev_close already above → breakout happened on a prior bar.
                # Always overwrite so we end up with the HIGHEST broken resistance
                # (most recent breakout), not the lowest (oldest move).
                else:
                    prior_breakout_level = (level, distance_pct)

        # ── Look for broken support (BREAKDOWN) ────────────────
        broken_support = None
        prior_breakdown_level = None    # support broken on an earlier bar
        for level in sorted(support_levels, key=lambda l: l.price, reverse=True):
            if level.price > price * 1.005:   # price is below by ≥ 0.5%
                distance_pct = (level.price - price) / level.price * 100
                if distance_pct < self.min_distance_pct:
                    continue
                if prev_close >= level.price * 0.998:
                    broken_support = (level, distance_pct)
                    break
                # Always overwrite so we end up with the LOWEST broken support
                # (most recent breakdown), not the highest (oldest move).
                else:
                    prior_breakdown_level = (level, distance_pct)

        # ── Build signal ───────────────────────────────────────
        if broken_resistance:
            level, dist_pct = broken_resistance
            quality = self._score_breakout(dist_pct, volume_confirmed, atr_expansion, level.strength)
            success_rate = _REGIME_BREAKOUT_SUCCESS.get(regime, 0.45)
            notes = self._breakout_notes(dist_pct, volume_confirmed, atr_expansion, quality, success_rate, regime)

            max_dist = atr * 3.0
            # Next resistance above (within 3×ATR)
            next_res = next((
                l.price for l in sorted(resistance_levels, key=lambda x: x.price)
                if l.price > price * 1.001 and (l.price - price) <= max_dist
            ), None)
            next_sup = next((
                l.price for l in sorted(support_levels, key=lambda x: x.price, reverse=True)
                if l.price < price * 0.999 and (price - l.price) <= max_dist
            ), None)
            room_pct = (next_res - price) / price * 100 if next_res else 0.0

            return BreakoutSignal(
                detected=True,
                signal_type="breakout",
                broken_level=level.price,
                distance_pct=dist_pct,
                volume_confirmed=volume_confirmed,
                atr_expansion=atr_expansion,
                quality_score=quality,
                regime_success_rate=success_rate,
                next_resistance=next_res,
                next_support=next_sup,
                room_to_next_level_pct=room_pct,
                notes=notes,
            )

        if broken_support:
            level, dist_pct = broken_support
            quality = self._score_breakout(dist_pct, volume_confirmed, atr_expansion, level.strength)
            success_rate = _REGIME_BREAKDOWN_SUCCESS.get(regime, 0.45)
            notes = self._breakdown_notes(dist_pct, volume_confirmed, atr_expansion, quality, success_rate, regime)

            max_dist = atr * 3.0
            next_res = next((
                l.price for l in sorted(resistance_levels, key=lambda x: x.price)
                if l.price > price * 1.001 and (l.price - price) <= max_dist
            ), None)
            next_sup = next((
                l.price for l in sorted(support_levels, key=lambda x: x.price, reverse=True)
                if l.price < price * 0.999 and (price - l.price) <= max_dist
            ), None)
            room_pct = (price - next_sup) / price * 100 if next_sup else 0.0

            return BreakoutSignal(
                detected=True,
                signal_type="breakdown",
                broken_level=level.price,
                distance_pct=dist_pct,
                volume_confirmed=volume_confirmed,
                atr_expansion=atr_expansion,
                quality_score=quality,
                regime_success_rate=success_rate,
                next_resistance=next_res,
                next_support=next_sup,
                room_to_next_level_pct=room_pct,
                notes=notes,
            )

        # ── Prior breakout/breakdown (happened on an earlier bar) ──
        if prior_breakout_level and not broken_resistance:
            level, dist_pct = prior_breakout_level
            success_rate = _REGIME_BREAKOUT_SUCCESS.get(regime, 0.45)
            max_dist = atr * 6.0
            next_res = next((
                l.price for l in sorted(resistance_levels, key=lambda x: x.price)
                if l.price > price * 1.001 and (l.price - price) <= max_dist
            ), None)
            next_sup = next((
                l.price for l in sorted(support_levels, key=lambda x: x.price, reverse=True)
                if l.price < price * 0.999 and (price - l.price) <= max_dist
            ), None)
            room_pct = (next_res - price) / price * 100 if next_res else 0.0
            return BreakoutSignal(
                detected=False,
                signal_type="prior_breakout",
                broken_level=level.price,
                distance_pct=dist_pct,
                volume_confirmed=volume_confirmed,
                atr_expansion=atr_expansion,
                quality_score=0.0,
                regime_success_rate=success_rate,
                next_resistance=next_res,
                next_support=next_sup,
                room_to_next_level_pct=room_pct,
                notes=[
                    f"Breakout occurred earlier in the session — ₹{level.price:.1f} resistance already cleared",
                    f"Price is {dist_pct:.1f}% above the broken level",
                    "Evaluating a retest or continuation entry — not a fresh breakout",
                    *(
                        [f"Next resistance: ₹{next_res:.1f} ({room_pct:.1f}% away)"]
                        if next_res else []
                    ),
                ],
            )

        if prior_breakdown_level and not broken_support:
            level, dist_pct = prior_breakdown_level
            success_rate = _REGIME_BREAKDOWN_SUCCESS.get(regime, 0.45)
            max_dist = atr * 6.0
            next_res = next((
                l.price for l in sorted(resistance_levels, key=lambda x: x.price)
                if l.price > price * 1.001 and (l.price - price) <= max_dist
            ), None)
            next_sup = next((
                l.price for l in sorted(support_levels, key=lambda x: x.price, reverse=True)
                if l.price < price * 0.999 and (price - l.price) <= max_dist
            ), None)
            room_pct = (price - next_sup) / price * 100 if next_sup else 0.0
            return BreakoutSignal(
                detected=False,
                signal_type="prior_breakdown",
                broken_level=level.price,
                distance_pct=dist_pct,
                volume_confirmed=volume_confirmed,
                atr_expansion=atr_expansion,
                quality_score=0.0,
                regime_success_rate=success_rate,
                next_resistance=next_res,
                next_support=next_sup,
                room_to_next_level_pct=room_pct,
                notes=[
                    f"Breakdown occurred earlier in the session — ₹{level.price:.1f} support already broken",
                    f"Price is {dist_pct:.1f}% below the broken level",
                    "Evaluating a retest or continuation short — not a fresh breakdown",
                    *(
                        [f"Next support: ₹{next_sup:.1f} ({room_pct:.1f}% below)"]
                        if next_sup else []
                    ),
                ],
            )

        # No breakout — show nearest levels within 6×ATR.
        # Using 6×ATR so VWAP and prior structure (e.g. breakout origin level)
        # remain visible even after a strong trending move.
        max_distance = atr * 6.0
        above_levels = [
            l.price for l in resistance_levels
            if l.price > price and (l.price - price) <= max_distance
        ]
        below_levels = [
            l.price for l in support_levels
            if l.price < price and (price - l.price) <= max_distance
        ]
        next_res = min(above_levels) if above_levels else None
        next_sup = max(below_levels) if below_levels else None

        return BreakoutSignal(
            detected=False,
            signal_type="none",
            next_resistance=next_res,
            next_support=next_sup,
            room_to_next_level_pct=(next_res - price) / price * 100 if next_res else 0.0,
            notes=self._no_signal_notes(price, next_res, next_sup, regime),
        )

    # ── Quality scoring ──────────────────────────────────────────

    def _score_breakout(
        self,
        distance_pct: float,
        volume_confirmed: bool,
        atr_expansion: bool,
        level_strength: float,
    ) -> float:
        score = 0.0
        # Distance above broken level (0-0.30)
        score += min(0.30, distance_pct / 3.0 * 0.30)
        # Volume confirmation (0.30)
        if volume_confirmed:
            score += 0.30
        # ATR expansion (0.20)
        if atr_expansion:
            score += 0.20
        # Level strength — stronger the level, better the breakout (0.20)
        score += level_strength * 0.20
        return round(min(1.0, score), 2)

    # ── Pivot detection ──────────────────────────────────────────

    def _find_pivots(
        self, bars: Sequence
    ) -> Tuple[List[int], List[int]]:
        """Return bar indices that are swing highs / swing lows."""
        n = len(bars)
        k = self.pivot_lookback
        highs: List[int] = []
        lows:  List[int] = []

        for i in range(k, n - k):
            h = bars[i].high
            l = bars[i].low

            is_high = all(h >= bars[j].high for j in range(i - k, i + k + 1) if j != i)
            is_low  = all(l <= bars[j].low  for j in range(i - k, i + k + 1) if j != i)

            if is_high:
                highs.append(i)
            if is_low:
                lows.append(i)

        return highs, lows

    def _cluster_levels(
        self,
        prices: List[float],
        pivot_indices: List[int],
        atr: float,
        level_type: str,
    ) -> List[KeyLevel]:
        """Merge nearby pivot prices into significant levels."""
        if not pivot_indices:
            return []

        raw = [prices[i] for i in pivot_indices]
        threshold = atr * self.cluster_atr_frac
        merged: List[KeyLevel] = []

        for price in sorted(raw):
            # Find existing cluster within threshold
            added = False
            for lv in merged:
                if abs(lv.price - price) <= threshold:
                    # Merge: weighted average
                    lv.price = (lv.price * lv.touches + price) / (lv.touches + 1)
                    lv.touches += 1
                    lv.strength = min(1.0, lv.touches / 5.0 * 0.8 + 0.2)
                    added = True
                    break
            if not added:
                merged.append(KeyLevel(price=price, level_type=level_type))

        return sorted(merged, key=lambda l: l.price)

    # ── Helpers ──────────────────────────────────────────────────

    def _simple_atr(self, bars: Sequence) -> float:
        if not bars:
            return 0.0
        ranges = [b.high - b.low for b in bars]
        return sum(ranges) / len(ranges)

    def _breakout_notes(
        self,
        dist_pct: float,
        vol_confirmed: bool,
        atr_exp: bool,
        quality: float,
        success_rate: float,
        regime: str,
    ) -> List[str]:
        notes = []
        if quality >= 0.7:
            notes.append(f"Strong breakout — {dist_pct:.1f}% above broken level")
        elif quality >= 0.4:
            notes.append(f"Moderate breakout — {dist_pct:.1f}% above resistance")
        else:
            notes.append(f"Weak breakout — only {dist_pct:.1f}% above level, watch for fakeout")

        if vol_confirmed:
            notes.append("Volume confirmed — institutional participation likely")
        else:
            notes.append("Low volume — breakout may lack conviction, wait for retest")

        if atr_exp:
            notes.append("ATR expanding — momentum behind the move")

        sr_pct = int(success_rate * 100)
        regime_label = regime.replace("_", " ").title()
        notes.append(f"In {regime_label} regime, breakouts succeed ~{sr_pct}% of the time")

        if success_rate < 0.45:
            notes.append("⚠ Current regime has poor breakout follow-through — consider waiting for retest")
        elif success_rate >= 0.65:
            notes.append("✓ Regime favours breakout continuation — can trail from breakout level")

        return notes

    def _breakdown_notes(
        self,
        dist_pct: float,
        vol_confirmed: bool,
        atr_exp: bool,
        quality: float,
        success_rate: float,
        regime: str,
    ) -> List[str]:
        notes = []
        if quality >= 0.7:
            notes.append(f"Strong breakdown — {dist_pct:.1f}% below broken support")
        elif quality >= 0.4:
            notes.append(f"Moderate breakdown — {dist_pct:.1f}% below support")
        else:
            notes.append(f"Weak breakdown — only {dist_pct:.1f}% below level, watch for recovery")

        if vol_confirmed:
            notes.append("Volume confirmed — sellers in control")
        else:
            notes.append("Low volume — possible fakeout, wait for retest")

        sr_pct = int(success_rate * 100)
        regime_label = regime.replace("_", " ").title()
        notes.append(f"In {regime_label} regime, breakdowns succeed ~{sr_pct}% of the time")

        if success_rate < 0.45:
            notes.append("⚠ Regime historically causes breakdown reversals")

        return notes

    def _no_signal_notes(
        self,
        price: float,
        next_res: Optional[float],
        next_sup: Optional[float],
        regime: str,
    ) -> List[str]:
        notes = ["No active breakout or breakdown detected"]
        if next_res:
            gap_pct = (next_res - price) / price * 100
            notes.append(f"Nearest resistance: ₹{next_res:.1f} ({gap_pct:.1f}% away — potential hurdle)")
        if next_sup:
            gap_pct = (price - next_sup) / price * 100
            notes.append(f"Nearest support: ₹{next_sup:.1f} ({gap_pct:.1f}% below)")
        return notes
