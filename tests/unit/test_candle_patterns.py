# tests/unit/test_candle_patterns.py
"""Tests for strategy_engine/candle_patterns.py."""

import pytest
from strategy_engine.candle_patterns import (
    detect_doji, detect_hammer, detect_shooting_star, detect_marubozu,
    detect_bullish_engulfing, detect_bearish_engulfing, detect_harami,
    detect_morning_star, detect_evening_star,
    detect_three_white_soldiers, detect_three_black_crows,
    scan_all, PatternResult,
)
from tests.factories import (
    make_candle, make_doji_candle, make_hammer_candle,
    make_shooting_star_candle, make_marubozu_candle,
)


class TestDoji:
    def test_detected(self):
        r = detect_doji([make_doji_candle(100.0, 4.0)])
        assert r.detected
        assert r.pattern == "doji"
        assert r.direction == "neutral"
        assert r.strength > 0

    def test_not_detected_on_big_body(self):
        r = detect_doji([make_candle(open_=95, high=105, low=94, close=104)])
        assert not r.detected

    def test_empty_candles(self):
        r = detect_doji([])
        assert not r.detected


class TestHammer:
    def test_detected(self):
        r = detect_hammer([make_hammer_candle(100.0)])
        assert r.detected
        assert r.direction == "bullish"

    def test_not_detected_on_marubozu(self):
        r = detect_hammer([make_marubozu_candle(98, 102)])
        assert not r.detected


class TestShootingStar:
    def test_detected(self):
        r = detect_shooting_star([make_shooting_star_candle(100.0)])
        assert r.detected
        assert r.direction == "bearish"

    def test_not_detected_on_hammer(self):
        r = detect_shooting_star([make_hammer_candle(100.0)])
        assert not r.detected


class TestMarubozu:
    def test_bullish_detected(self):
        r = detect_marubozu([make_marubozu_candle(98.0, 102.0)])
        assert r.detected
        assert r.direction == "bullish"

    def test_bearish_detected(self):
        r = detect_marubozu([make_marubozu_candle(102.0, 98.0)])
        assert r.detected
        assert r.direction == "bearish"

    def test_not_detected_on_doji(self):
        r = detect_marubozu([make_doji_candle()])
        assert not r.detected


class TestBullishEngulfing:
    def test_detected(self):
        prev = make_candle(open_=102, high=103, low=99, close=100)
        curr = make_candle(open_=99, high=104, low=98, close=103)
        r = detect_bullish_engulfing([prev, curr])
        assert r.detected
        assert r.direction == "bullish"

    def test_not_detected_same_direction(self):
        prev = make_candle(open_=100, high=103, low=99, close=102)
        curr = make_candle(open_=101, high=104, low=100, close=103)
        r = detect_bullish_engulfing([prev, curr])
        assert not r.detected

    def test_insufficient_data(self):
        r = detect_bullish_engulfing([make_candle()])
        assert not r.detected


class TestBearishEngulfing:
    def test_detected(self):
        prev = make_candle(open_=100, high=103, low=99, close=102)
        curr = make_candle(open_=103, high=104, low=98, close=99)
        r = detect_bearish_engulfing([prev, curr])
        assert r.detected
        assert r.direction == "bearish"


class TestHarami:
    def test_detected(self):
        prev = make_candle(open_=105, high=106, low=95, close=96)
        curr = make_candle(open_=99, high=101, low=98, close=100)
        r = detect_harami([prev, curr])
        assert r.detected

    def test_not_detected_body_too_large(self):
        prev = make_candle(open_=102, high=103, low=99, close=100)
        curr = make_candle(open_=99, high=103, low=98, close=102)
        r = detect_harami([prev, curr])
        assert not r.detected


class TestMorningStar:
    def test_detected(self):
        c1 = make_candle(open_=105, high=106, low=99, close=100)
        c2 = make_candle(open_=100, high=100.5, low=99.5, close=100.1)
        c3 = make_candle(open_=101, high=107, low=100, close=106)
        r = detect_morning_star([c1, c2, c3])
        assert r.detected
        assert r.direction == "bullish"

    def test_insufficient_data(self):
        r = detect_morning_star([make_candle(), make_candle()])
        assert not r.detected


class TestEveningStar:
    def test_detected(self):
        c1 = make_candle(open_=100, high=106, low=99, close=105)
        c2 = make_candle(open_=105, high=105.5, low=104.5, close=105.1)
        c3 = make_candle(open_=104, high=105, low=97, close=98)
        r = detect_evening_star([c1, c2, c3])
        assert r.detected
        assert r.direction == "bearish"


class TestThreeWhiteSoldiers:
    def test_detected(self):
        c1 = make_candle(open_=100, high=103, low=100, close=102.9)
        c2 = make_candle(open_=103, high=106, low=103, close=105.9)
        c3 = make_candle(open_=106, high=109, low=106, close=108.9)
        r = detect_three_white_soldiers([c1, c2, c3])
        assert r.detected
        assert r.direction == "bullish"


class TestThreeBlackCrows:
    def test_detected(self):
        c1 = make_candle(open_=109, high=109, low=106.1, close=106.2)
        c2 = make_candle(open_=106, high=106, low=103.1, close=103.2)
        c3 = make_candle(open_=103, high=103, low=100.1, close=100.2)
        r = detect_three_black_crows([c1, c2, c3])
        assert r.detected
        assert r.direction == "bearish"


class TestScanAll:
    def test_returns_only_detected(self):
        candles = [make_doji_candle(100.0, 4.0)]
        results = scan_all(candles)
        assert all(r.detected for r in results)
        patterns = [r.pattern for r in results]
        assert "doji" in patterns

    def test_empty_candles_returns_empty(self):
        assert scan_all([]) == []
