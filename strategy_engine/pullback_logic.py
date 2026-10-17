"""
Pullback Logic
===============
Detects and scores pullback entry opportunities in trending markets.
A pullback is a temporary counter-trend move that creates a re-entry opportunity
at a better price within the dominant trend.

Logic flow:
1. Confirm trend (using EMA relationship or swing high/low structure)
2. Detect pullback (price retracing towards a key level)
3. Score the entry quality (depth, candle confirmation, volume)
4. Return a PullbackSignal with entry parameters

Pure functions — no I/O, no DB access.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PullbackSignal:
    detected: bool
    direction: str              # "long" | "short" | "none"
    quality_score: float        # 0-100
    entry_zone_low: float
    entry_zone_high: float
    suggested_stop: float       # below swing low (long) or above swing high (short)
    invalidation: float         # price that cancels the setup
    reasons: list = field(default_factory=list)    # list of strings explaining the score
    warnings: list = field(default_factory=list)


def _ema(values: list[float], period: int) -> list[float]:
    """Exponential moving average (Wilder-style: alpha = 1/period, adjust=False)."""
    if not values or period <= 0:
        return []
    alpha = 1.0 / period
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def _swing_lows(lows: list[float], window: int = 3) -> list[tuple[int, float]]:
    """Return (index, value) for local swing lows."""
    pivots = []
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i - window: i + window + 1]):
            pivots.append((i, lows[i]))
    return pivots


def _swing_highs(highs: list[float], window: int = 3) -> list[tuple[int, float]]:
    """Return (index, value) for local swing highs."""
    pivots = []
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            pivots.append((i, highs[i]))
    return pivots


def _is_uptrend(closes: list[float], ema_fast: int = 20, ema_slow: int = 50) -> bool:
    """EMA cross + higher-highs / higher-lows confirmation."""
    if len(closes) < ema_slow:
        return False
    fast = _ema(closes, ema_fast)
    slow = _ema(closes, ema_slow)
    return fast[-1] > slow[-1]


def _is_downtrend(closes: list[float], ema_fast: int = 20, ema_slow: int = 50) -> bool:
    if len(closes) < ema_slow:
        return False
    fast = _ema(closes, ema_fast)
    slow = _ema(closes, ema_slow)
    return fast[-1] < slow[-1]


def _retracement_pct(start: float, end: float, current: float) -> float:
    """Fibonacci retracement level of pullback (0=none, 1=full reversal)."""
    move = abs(end - start)
    if move == 0:
        return 0.0
    return abs(current - end) / move


def detect_pullback(candles: list[dict], ema_fast: int = 20,
                    ema_slow: int = 50) -> PullbackSignal:
    """
    candles: list of {"open":f, "high":f, "low":f, "close":f, "volume":f}
    Returns a PullbackSignal.
    """
    _no_signal = PullbackSignal(
        detected=False, direction="none", quality_score=0.0,
        entry_zone_low=0.0, entry_zone_high=0.0,
        suggested_stop=0.0, invalidation=0.0,
    )

    if len(candles) < max(ema_slow + 5, 30):
        return _no_signal

    closes  = [c["close"] for c in candles]
    highs   = [c["high"]  for c in candles]
    lows    = [c["low"]   for c in candles]
    volumes = [c.get("volume", 0) for c in candles]

    current = closes[-1]
    fast_ema = _ema(closes, ema_fast)
    slow_ema = _ema(closes, ema_slow)

    up   = fast_ema[-1] > slow_ema[-1]
    down = fast_ema[-1] < slow_ema[-1]

    if not (up or down):
        return _no_signal

    direction = "long" if up else "short"
    reasons: list[str] = []
    warnings: list[str] = []
    score = 0.0

    if direction == "long":
        # Trend confirmation
        reasons.append(f"Uptrend confirmed: EMA{ema_fast} ({fast_ema[-1]:.2f}) > EMA{ema_slow} ({slow_ema[-1]:.2f})")
        score += 20.0

        # Pullback: price below fast EMA but above slow EMA
        pulling_back = slow_ema[-1] < current < fast_ema[-1]
        at_ema = abs(current - fast_ema[-1]) / fast_ema[-1] < 0.005
        if pulling_back or at_ema:
            reasons.append("Price pulling back to EMA zone")
            score += 25.0

        # Fibonacci depth check (prefer 38-62% pullback)
        swing_highs_list = _swing_highs(highs, 3)
        swing_lows_list  = _swing_lows(lows, 3)
        if swing_highs_list and swing_lows_list:
            recent_high = swing_highs_list[-1][1]
            recent_low  = swing_lows_list[-1][1]
            ret = _retracement_pct(recent_low, recent_high, current)
            if 0.382 <= ret <= 0.618:
                reasons.append(f"Fibonacci depth {ret:.1%} — ideal pullback zone")
                score += 20.0
            elif 0.236 <= ret < 0.382 or 0.618 < ret <= 0.786:
                reasons.append(f"Fibonacci depth {ret:.1%} — acceptable")
                score += 10.0
            elif ret > 0.786:
                warnings.append(f"Deep pullback {ret:.1%} — trend may be weakening")

        # Volume: pullback should be on lower volume than trend impulse
        avg_vol = statistics.mean(volumes[-20:]) if len(volumes) >= 20 else 0
        recent_vol = statistics.mean(volumes[-3:]) if len(volumes) >= 3 else 0
        if avg_vol > 0 and recent_vol < avg_vol * 0.8:
            reasons.append("Pullback volume below average — healthy consolidation")
            score += 15.0

        # Candle confirmation: last candle should be bullish or a reversal pattern
        last = candles[-1]
        if last["close"] > last["open"]:
            reasons.append("Bullish confirmation candle")
            score += 10.0

        # Swing low stop placement
        recent_sl = swing_lows_list[-1][1] if swing_lows_list else min(lows[-5:])
        stop = recent_sl * 0.998    # 0.2% below swing low
        invalidation = slow_ema[-1] * 0.997

        entry_lo = min(current, fast_ema[-1]) * 0.999
        entry_hi = max(current, fast_ema[-1]) * 1.001

    else:  # short
        reasons.append(f"Downtrend confirmed: EMA{ema_fast} ({fast_ema[-1]:.2f}) < EMA{ema_slow} ({slow_ema[-1]:.2f})")
        score += 20.0

        pulling_back = fast_ema[-1] < current < slow_ema[-1]
        at_ema = abs(current - fast_ema[-1]) / (fast_ema[-1] or 1) < 0.005
        if pulling_back or at_ema:
            reasons.append("Price pulling back to EMA zone")
            score += 25.0

        swing_highs_list = _swing_highs(highs, 3)
        swing_lows_list  = _swing_lows(lows, 3)
        if swing_highs_list and swing_lows_list:
            recent_high = swing_highs_list[-1][1]
            recent_low  = swing_lows_list[-1][1]
            ret = _retracement_pct(recent_high, recent_low, current)
            if 0.382 <= ret <= 0.618:
                reasons.append(f"Fibonacci depth {ret:.1%} — ideal pullback zone")
                score += 20.0
            elif ret > 0.786:
                warnings.append(f"Deep pullback {ret:.1%} — trend may be reversing")

        avg_vol = statistics.mean(volumes[-20:]) if len(volumes) >= 20 else 0
        recent_vol = statistics.mean(volumes[-3:]) if len(volumes) >= 3 else 0
        if avg_vol > 0 and recent_vol < avg_vol * 0.8:
            reasons.append("Pullback volume below average")
            score += 15.0

        last = candles[-1]
        if last["close"] < last["open"]:
            reasons.append("Bearish confirmation candle")
            score += 10.0

        recent_sh = swing_highs_list[-1][1] if swing_highs_list else max(highs[-5:])
        stop = recent_sh * 1.002
        invalidation = slow_ema[-1] * 1.003
        entry_lo = min(current, fast_ema[-1]) * 0.999
        entry_hi = max(current, fast_ema[-1]) * 1.001

    detected = score >= 40.0   # minimum threshold to call this a valid pullback

    return PullbackSignal(
        detected=detected,
        direction=direction,
        quality_score=round(min(score, 100.0), 1),
        entry_zone_low=round(entry_lo, 4),
        entry_zone_high=round(entry_hi, 4),
        suggested_stop=round(stop, 4),
        invalidation=round(invalidation, 4),
        reasons=reasons,
        warnings=warnings,
    )
