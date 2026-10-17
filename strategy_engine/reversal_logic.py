"""
Reversal Logic
===============
Detects high-probability reversal setups at key structural levels.
Combines price action (candle pattern), volume, and structural context.

Reversal types supported:
- Support / Resistance reversal
- Exhaustion reversal (climax volume + rejection wick)
- Failed breakout (false break above/below key level)
- Double-top / Double-bottom structure

Pure functions — no I/O, no DB access.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReversalSignal:
    detected: bool
    reversal_type: str          # "support_bounce" | "resistance_rejection" | "exhaustion" | "failed_breakout" | "double_bottom" | "double_top" | "none"
    direction: str              # "long" | "short" | "none"
    quality_score: float        # 0-100
    key_level: float            # S/R level the reversal is occurring at
    suggested_entry: float
    suggested_stop: float
    target_1: float
    target_2: float
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _body(c: dict) -> float:
    return abs(c["close"] - c["open"])

def _range(c: dict) -> float:
    return c["high"] - c["low"]

def _lower_wick(c: dict) -> float:
    return min(c["open"], c["close"]) - c["low"]

def _upper_wick(c: dict) -> float:
    return c["high"] - max(c["open"], c["close"])


def _find_levels(highs: list[float], lows: list[float],
                 tolerance: float = 0.005) -> list[float]:
    """
    Find significant price levels by clustering swing highs and lows.
    Two swings within tolerance% of each other form one level.
    """
    candidates = highs + lows
    if not candidates:
        return []
    levels: list[float] = []
    for price in sorted(set(candidates)):
        if not levels or abs(price - levels[-1]) / levels[-1] > tolerance:
            levels.append(price)
    return levels


def _nearest_level(price: float, levels: list[float]) -> Optional[float]:
    if not levels:
        return None
    return min(levels, key=lambda l: abs(price - l))


def _swing_lows(lows: list[float], window: int = 3) -> list[float]:
    pivots = []
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i - window: i + window + 1]):
            pivots.append(lows[i])
    return pivots


def _swing_highs(highs: list[float], window: int = 3) -> list[float]:
    pivots = []
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            pivots.append(highs[i])
    return pivots


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

def detect_reversal(candles: list[dict], level_window: int = 5) -> ReversalSignal:
    """
    candles: list of {"open":f, "high":f, "low":f, "close":f, "volume":f}
    Runs all reversal checks and returns the highest-scoring signal.
    """
    _no_signal = ReversalSignal(
        detected=False, reversal_type="none", direction="none", quality_score=0.0,
        key_level=0.0, suggested_entry=0.0, suggested_stop=0.0, target_1=0.0, target_2=0.0,
    )

    if len(candles) < 20:
        return _no_signal

    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]
    closes  = [c["close"]  for c in candles]
    volumes = [c.get("volume", 0) for c in candles]

    swing_h = _swing_highs(highs, level_window)
    swing_l = _swing_lows(lows, level_window)
    levels  = _find_levels(swing_h, swing_l)

    current = closes[-1]
    last    = candles[-1]

    # --- Check each reversal type ---
    signals = [
        _check_support_bounce(candles, current, last, levels, swing_l, volumes),
        _check_resistance_rejection(candles, current, last, levels, swing_h, volumes),
        _check_exhaustion(candles, last, volumes, closes),
        _check_failed_breakout(candles, last, levels, closes, highs, lows),
        _check_double_bottom(lows, closes, levels, current),
        _check_double_top(highs, closes, levels, current),
    ]

    # Pick highest scoring
    best = max(signals, key=lambda s: s.quality_score)
    return best if best.quality_score >= 40.0 else _no_signal


def _check_support_bounce(candles, current, last, levels, swing_lows_list, volumes):
    reasons, warnings = [], []
    score = 0.0
    level = _nearest_level(current, levels)
    if level is None:
        return _empty("support_bounce", "long")

    proximity = abs(current - level) / level
    if proximity > 0.01:
        return _empty("support_bounce", "long")

    score += 25.0
    reasons.append(f"Price at key support level {level:.2f}")

    # Hammer / lower wick rejection
    b = _body(last) or 0.001
    lw = _lower_wick(last)
    if lw >= b * 1.5:
        score += 20.0
        reasons.append("Lower wick rejection — buyers defending level")

    # Volume spike
    avg_vol = statistics.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    curr_vol = volumes[-1]
    if avg_vol > 0 and curr_vol > avg_vol * 1.3:
        score += 15.0
        reasons.append("Volume surge on test of support")

    # Bullish close
    if last["close"] > last["open"]:
        score += 10.0
        reasons.append("Bullish candle close above open")

    # Multiple tests of level (stronger support)
    tests = sum(1 for sl in swing_lows_list
                if abs(sl - level) / (level or 1) < 0.005)
    if tests >= 2:
        score += 10.0
        reasons.append(f"Level tested {tests}× — strong support")

    # Recent structure not broken
    last_swing_low = min(swing_lows_list[-3:]) if swing_lows_list else current * 0.98
    if current > last_swing_low:
        score += 10.0

    stop = level * 0.993
    entry = current * 1.001
    r = current - stop
    return ReversalSignal(
        detected=True, reversal_type="support_bounce", direction="long",
        quality_score=round(min(score, 100.0), 1), key_level=round(level, 4),
        suggested_entry=round(entry, 4), suggested_stop=round(stop, 4),
        target_1=round(entry + r * 2, 4), target_2=round(entry + r * 3, 4),
        reasons=reasons, warnings=warnings,
    )


def _check_resistance_rejection(candles, current, last, levels, swing_highs_list, volumes):
    reasons, warnings = [], []
    score = 0.0
    level = _nearest_level(current, levels)
    if level is None:
        return _empty("resistance_rejection", "short")

    proximity = abs(current - level) / level
    if proximity > 0.01:
        return _empty("resistance_rejection", "short")

    score += 25.0
    reasons.append(f"Price at key resistance level {level:.2f}")

    b = _body(last) or 0.001
    uw = _upper_wick(last)
    if uw >= b * 1.5:
        score += 20.0
        reasons.append("Upper wick rejection — sellers defending level")

    avg_vol = statistics.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    curr_vol = volumes[-1]
    if avg_vol > 0 and curr_vol > avg_vol * 1.3:
        score += 15.0
        reasons.append("Volume surge on test of resistance")

    if last["close"] < last["open"]:
        score += 10.0
        reasons.append("Bearish candle close below open")

    tests = sum(1 for sh in swing_highs_list
                if abs(sh - level) / (level or 1) < 0.005)
    if tests >= 2:
        score += 10.0
        reasons.append(f"Level tested {tests}× — strong resistance")

    stop = level * 1.007
    entry = current * 0.999
    r = stop - current
    return ReversalSignal(
        detected=True, reversal_type="resistance_rejection", direction="short",
        quality_score=round(min(score, 100.0), 1), key_level=round(level, 4),
        suggested_entry=round(entry, 4), suggested_stop=round(stop, 4),
        target_1=round(entry - r * 2, 4), target_2=round(entry - r * 3, 4),
        reasons=reasons, warnings=warnings,
    )


def _check_exhaustion(candles, last, volumes, closes):
    """Climax volume + long wick = exhaustion reversal."""
    avg_vol = statistics.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    curr_vol = volumes[-1]
    if avg_vol == 0:
        return _empty("exhaustion", "none")

    vol_spike = curr_vol > avg_vol * 2.0
    big_wick_up   = _upper_wick(last) > _body(last) * 2
    big_wick_down = _lower_wick(last) > _body(last) * 2

    if vol_spike and big_wick_up:
        stop = last["high"] * 1.003
        entry = last["close"]
        r = stop - entry
        return ReversalSignal(
            detected=True, reversal_type="exhaustion", direction="short",
            quality_score=70.0, key_level=round(last["high"], 4),
            suggested_entry=round(entry, 4), suggested_stop=round(stop, 4),
            target_1=round(entry - r * 2, 4), target_2=round(entry - r * 3, 4),
            reasons=["Climax volume spike", "Large upper wick — buying exhaustion"],
        )

    if vol_spike and big_wick_down:
        stop = last["low"] * 0.997
        entry = last["close"]
        r = entry - stop
        return ReversalSignal(
            detected=True, reversal_type="exhaustion", direction="long",
            quality_score=70.0, key_level=round(last["low"], 4),
            suggested_entry=round(entry, 4), suggested_stop=round(stop, 4),
            target_1=round(entry + r * 2, 4), target_2=round(entry + r * 3, 4),
            reasons=["Climax volume spike", "Large lower wick — selling exhaustion"],
        )
    return _empty("exhaustion", "none")


def _check_failed_breakout(candles, last, levels, closes, highs, lows):
    """Price broke a level but closed back inside — failed breakout."""
    level = _nearest_level(closes[-1], levels)
    if not level:
        return _empty("failed_breakout", "none")

    # Bullish false break (above resistance, closed below)
    if last["high"] > level and last["close"] < level:
        stop = last["high"] * 1.003
        entry = last["close"]
        r = stop - entry
        return ReversalSignal(
            detected=True, reversal_type="failed_breakout", direction="short",
            quality_score=65.0, key_level=round(level, 4),
            suggested_entry=round(entry, 4), suggested_stop=round(stop, 4),
            target_1=round(entry - r * 2, 4), target_2=round(entry - r * 3, 4),
            reasons=[f"Failed breakout above {level:.2f} — trapped bulls"],
        )

    # Bearish false break (below support, closed above)
    if last["low"] < level and last["close"] > level:
        stop = last["low"] * 0.997
        entry = last["close"]
        r = entry - stop
        return ReversalSignal(
            detected=True, reversal_type="failed_breakout", direction="long",
            quality_score=65.0, key_level=round(level, 4),
            suggested_entry=round(entry, 4), suggested_stop=round(stop, 4),
            target_1=round(entry + r * 2, 4), target_2=round(entry + r * 3, 4),
            reasons=[f"Failed breakdown below {level:.2f} — trapped bears"],
        )
    return _empty("failed_breakout", "none")


def _check_double_bottom(lows, closes, levels, current):
    sl = _swing_lows(lows, 3)
    if len(sl) < 2:
        return _empty("double_bottom", "long")
    last_two = sl[-2:]
    if abs(last_two[0] - last_two[1]) / (last_two[0] or 1) < 0.01 and current > last_two[1]:
        stop = min(last_two) * 0.995
        r = current - stop
        return ReversalSignal(
            detected=True, reversal_type="double_bottom", direction="long",
            quality_score=72.0, key_level=round(statistics.mean(last_two), 4),
            suggested_entry=round(current, 4), suggested_stop=round(stop, 4),
            target_1=round(current + r * 2, 4), target_2=round(current + r * 3, 4),
            reasons=["Double bottom pattern — equal lows with bounce"],
        )
    return _empty("double_bottom", "long")


def _check_double_top(highs, closes, levels, current):
    sh = _swing_highs(highs, 3)
    if len(sh) < 2:
        return _empty("double_top", "short")
    last_two = sh[-2:]
    if abs(last_two[0] - last_two[1]) / (last_two[0] or 1) < 0.01 and current < last_two[1]:
        stop = max(last_two) * 1.005
        r = stop - current
        return ReversalSignal(
            detected=True, reversal_type="double_top", direction="short",
            quality_score=72.0, key_level=round(statistics.mean(last_two), 4),
            suggested_entry=round(current, 4), suggested_stop=round(stop, 4),
            target_1=round(current - r * 2, 4), target_2=round(current - r * 3, 4),
            reasons=["Double top pattern — equal highs with rejection"],
        )
    return _empty("double_top", "short")


def _empty(reversal_type: str, direction: str) -> ReversalSignal:
    return ReversalSignal(
        detected=False, reversal_type=reversal_type, direction=direction,
        quality_score=0.0, key_level=0.0, suggested_entry=0.0,
        suggested_stop=0.0, target_1=0.0, target_2=0.0,
    )
