"""
Candle Patterns
================
Pure-function pattern detection on OHLCV candlestick data.
Returns structured PatternResult objects — no side-effects, no I/O.

Supported patterns:
  Single candle: doji, hammer, shooting_star, marubozu, spinning_top
  Two candle:    bullish_engulfing, bearish_engulfing, harami
  Three candle:  morning_star, evening_star, three_white_soldiers, three_black_crows

All detectors take a list of candle dicts:
  {"open": float, "high": float, "low": float, "close": float, "volume": float}
and return a PatternResult.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PatternResult:
    pattern: str
    detected: bool
    direction: str          # "bullish" | "bearish" | "neutral"
    strength: float         # 0-1; 1 = textbook example
    description: str


def _body(c: dict) -> float:
    return abs(c["close"] - c["open"])

def _range(c: dict) -> float:
    return c["high"] - c["low"]

def _upper_wick(c: dict) -> float:
    return c["high"] - max(c["open"], c["close"])

def _lower_wick(c: dict) -> float:
    return min(c["open"], c["close"]) - c["low"]

def _is_bullish(c: dict) -> bool:
    return c["close"] > c["open"]

def _is_bearish(c: dict) -> bool:
    return c["close"] < c["open"]


# ---------------------------------------------------------------------------
# Single-candle patterns
# ---------------------------------------------------------------------------

def detect_doji(candles: list[dict]) -> PatternResult:
    """Doji: open ≈ close (body < 10% of range)."""
    if not candles:
        return PatternResult("doji", False, "neutral", 0.0, "Insufficient data")
    c = candles[-1]
    r = _range(c) or 1.0
    b = _body(c)
    ratio = b / r
    detected = ratio < 0.10
    strength = max(0.0, 1.0 - ratio / 0.10) if detected else 0.0
    return PatternResult("doji", detected, "neutral", round(strength, 3),
                         "Indecision — body < 10% of range" if detected else "Not a doji")


def detect_hammer(candles: list[dict]) -> PatternResult:
    """Hammer: small body at top, lower wick ≥ 2× body, upper wick small."""
    if not candles:
        return PatternResult("hammer", False, "bullish", 0.0, "Insufficient data")
    c = candles[-1]
    b = _body(c) or 0.001
    lw = _lower_wick(c)
    uw = _upper_wick(c)
    detected = lw >= 2 * b and uw <= b * 0.5
    strength = min(lw / (b * 3), 1.0) if detected else 0.0
    return PatternResult("hammer", detected, "bullish", round(strength, 3),
                         "Bullish reversal signal — rejection of lower prices" if detected else "Not a hammer")


def detect_shooting_star(candles: list[dict]) -> PatternResult:
    """Shooting Star: small body at bottom, upper wick ≥ 2× body."""
    if not candles:
        return PatternResult("shooting_star", False, "bearish", 0.0, "Insufficient data")
    c = candles[-1]
    b = _body(c) or 0.001
    uw = _upper_wick(c)
    lw = _lower_wick(c)
    detected = uw >= 2 * b and lw <= b * 0.5
    strength = min(uw / (b * 3), 1.0) if detected else 0.0
    return PatternResult("shooting_star", detected, "bearish", round(strength, 3),
                         "Bearish reversal — rejection of higher prices" if detected else "Not a shooting star")


def detect_marubozu(candles: list[dict]) -> PatternResult:
    """Marubozu: full body, very small wicks (wicks < 5% of range each)."""
    if not candles:
        return PatternResult("marubozu", False, "neutral", 0.0, "Insufficient data")
    c = candles[-1]
    r = _range(c) or 1.0
    b = _body(c)
    uw = _upper_wick(c)
    lw = _lower_wick(c)
    detected = b / r > 0.90 and uw / r < 0.05 and lw / r < 0.05
    direction = "bullish" if _is_bullish(c) else "bearish"
    strength = (b / r) if detected else 0.0
    return PatternResult("marubozu", detected, direction, round(strength, 3),
                         f"Strong {'bullish' if _is_bullish(c) else 'bearish'} momentum — no wicks"
                         if detected else "Not a marubozu")


# ---------------------------------------------------------------------------
# Two-candle patterns
# ---------------------------------------------------------------------------

def detect_bullish_engulfing(candles: list[dict]) -> PatternResult:
    """Bullish engulfing: bearish candle followed by larger bullish candle."""
    if len(candles) < 2:
        return PatternResult("bullish_engulfing", False, "bullish", 0.0, "Insufficient data")
    prev, curr = candles[-2], candles[-1]
    detected = (_is_bearish(prev) and _is_bullish(curr) and
                curr["open"] < prev["close"] and curr["close"] > prev["open"])
    strength = min(_body(curr) / (_body(prev) or 0.001), 2.0) / 2.0 if detected else 0.0
    return PatternResult("bullish_engulfing", detected, "bullish", round(strength, 3),
                         "Strong bullish reversal — buyers absorbed all sellers" if detected
                         else "Not a bullish engulfing")


def detect_bearish_engulfing(candles: list[dict]) -> PatternResult:
    """Bearish engulfing: bullish candle followed by larger bearish candle."""
    if len(candles) < 2:
        return PatternResult("bearish_engulfing", False, "bearish", 0.0, "Insufficient data")
    prev, curr = candles[-2], candles[-1]
    detected = (_is_bullish(prev) and _is_bearish(curr) and
                curr["open"] > prev["close"] and curr["close"] < prev["open"])
    strength = min(_body(curr) / (_body(prev) or 0.001), 2.0) / 2.0 if detected else 0.0
    return PatternResult("bearish_engulfing", detected, "bearish", round(strength, 3),
                         "Strong bearish reversal — sellers absorbed all buyers" if detected
                         else "Not a bearish engulfing")


def detect_harami(candles: list[dict]) -> PatternResult:
    """Harami: small second candle contained within first candle's body."""
    if len(candles) < 2:
        return PatternResult("harami", False, "neutral", 0.0, "Insufficient data")
    prev, curr = candles[-2], candles[-1]
    prev_hi = max(prev["open"], prev["close"])
    prev_lo = min(prev["open"], prev["close"])
    curr_hi = max(curr["open"], curr["close"])
    curr_lo = min(curr["open"], curr["close"])
    detected = curr_hi <= prev_hi and curr_lo >= prev_lo and _body(curr) < _body(prev) * 0.5
    direction = "bullish" if _is_bearish(prev) and _is_bullish(curr) else "bearish"
    strength = (1 - _body(curr) / (_body(prev) or 1)) if detected else 0.0
    return PatternResult("harami", detected, direction, round(strength, 3),
                         f"{direction.capitalize()} indecision inside prior candle" if detected
                         else "Not a harami")


# ---------------------------------------------------------------------------
# Three-candle patterns
# ---------------------------------------------------------------------------

def detect_morning_star(candles: list[dict]) -> PatternResult:
    """Morning Star: bearish candle → small-body doji/gap → bullish candle."""
    if len(candles) < 3:
        return PatternResult("morning_star", False, "bullish", 0.0, "Insufficient data")
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    small_body = _body(c2) < _body(c1) * 0.3
    # c3 must close above midpoint of c1's body
    c1_midpoint = (c1["open"] + c1["close"]) / 2
    detected = (_is_bearish(c1) and small_body and _is_bullish(c3) and
                c3["close"] > c1_midpoint)
    strength = min(_body(c3) / (_body(c1) or 1), 1.0) if detected else 0.0
    return PatternResult("morning_star", detected, "bullish", round(strength, 3),
                         "Bullish reversal — bears exhausted, bulls take control" if detected
                         else "Not a morning star")


def detect_evening_star(candles: list[dict]) -> PatternResult:
    """Evening Star: bullish candle → small-body → bearish candle."""
    if len(candles) < 3:
        return PatternResult("evening_star", False, "bearish", 0.0, "Insufficient data")
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    small_body = _body(c2) < _body(c1) * 0.3
    detected = (_is_bullish(c1) and small_body and _is_bearish(c3) and
                c3["close"] < c1["open"] - _body(c1) * 0.3)
    strength = min(_body(c3) / (_body(c1) or 1), 1.0) if detected else 0.0
    return PatternResult("evening_star", detected, "bearish", round(strength, 3),
                         "Bearish reversal — bulls exhausted, bears take control" if detected
                         else "Not an evening star")


def detect_three_white_soldiers(candles: list[dict]) -> PatternResult:
    """Three white soldiers: three consecutive bullish candles with higher closes."""
    if len(candles) < 3:
        return PatternResult("three_white_soldiers", False, "bullish", 0.0, "Insufficient data")
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    detected = (all(_is_bullish(c) for c in [c1, c2, c3]) and
                c2["close"] > c1["close"] and c3["close"] > c2["close"] and
                all(_lower_wick(c) < _body(c) * 0.3 for c in [c1, c2, c3]))
    return PatternResult("three_white_soldiers", detected, "bullish",
                         0.85 if detected else 0.0,
                         "Strong bullish continuation" if detected else "Not three white soldiers")


def detect_three_black_crows(candles: list[dict]) -> PatternResult:
    """Three black crows: three consecutive bearish candles with lower closes."""
    if len(candles) < 3:
        return PatternResult("three_black_crows", False, "bearish", 0.0, "Insufficient data")
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    detected = (all(_is_bearish(c) for c in [c1, c2, c3]) and
                c2["close"] < c1["close"] and c3["close"] < c2["close"] and
                all(_upper_wick(c) < _body(c) * 0.3 for c in [c1, c2, c3]))
    return PatternResult("three_black_crows", detected, "bearish",
                         0.85 if detected else 0.0,
                         "Strong bearish continuation" if detected else "Not three black crows")


# ---------------------------------------------------------------------------
# Composite scanner
# ---------------------------------------------------------------------------

ALL_DETECTORS = [
    detect_doji, detect_hammer, detect_shooting_star, detect_marubozu,
    detect_bullish_engulfing, detect_bearish_engulfing, detect_harami,
    detect_morning_star, detect_evening_star,
    detect_three_white_soldiers, detect_three_black_crows,
]


def scan_all(candles: list[dict]) -> list[PatternResult]:
    """Run all pattern detectors and return only those that fired."""
    results = [d(candles) for d in ALL_DETECTORS]
    return [r for r in results if r.detected]


def scan_all_dict(candles: list[dict]) -> list[dict]:
    """Same as scan_all but returns dicts (JSON serialisable)."""
    return [{"pattern": r.pattern, "direction": r.direction,
             "strength": r.strength, "description": r.description}
            for r in scan_all(candles)]
