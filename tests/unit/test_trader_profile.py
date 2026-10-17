"""Tests for Trader Profile Intelligence."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from intelligence.trader_profile_intelligence import (
    TraderProfileBuilder,
    TraderIntelligenceProfile,
    TradingStyle,
    EntryStyle,
    RiskPersonality,
    DisciplinePattern,
    RiskProfile,
    TimingProfile,
    EmotionalProfile,
)


def _mock_trade(**kwargs):
    t = MagicMock()
    t.entry_time = kwargs.get("entry_time", datetime(2024, 1, 15, 10, 30))
    t.exit_time = kwargs.get("exit_time", datetime(2024, 1, 15, 11, 0))
    t.r_multiple = kwargs.get("r_multiple", 1.0)
    t.pnl = kwargs.get("pnl", 100)
    t.risk_percent = kwargs.get("risk_percent", 0.01)
    t.planned_rr = kwargs.get("planned_rr", 2.0)
    t.outcome = kwargs.get("outcome", "win")
    t.strategy = kwargs.get("strategy", "opening_breakout")
    t.market_regime = kwargs.get("market_regime", "trending_bullish")
    t.emotional_state = kwargs.get("emotional_state", "focused")
    t.metadata = kwargs.get("metadata", {})
    t.created_at = kwargs.get("created_at", t.entry_time)
    t.closed_at = kwargs.get("closed_at", t.exit_time)
    return t


class TestTradingStyleClassification:

    def setup_method(self):
        self.builder = TraderProfileBuilder(db=AsyncMock())

    def test_scalper_classification(self):
        trades = [
            _mock_trade(
                entry_time=datetime(2024, 1, 15, 10, 0),
                exit_time=datetime(2024, 1, 15, 10, 3),
            )
            for _ in range(20)
        ]
        assert self.builder._classify_trading_style(trades) == TradingStyle.SCALPER

    def test_day_trader_classification(self):
        trades = [
            _mock_trade(
                entry_time=datetime(2024, 1, 15, 10, 0),
                exit_time=datetime(2024, 1, 15, 11, 30),
            )
            for _ in range(20)
        ]
        assert self.builder._classify_trading_style(trades) == TradingStyle.DAY_TRADER

    def test_swing_trader_classification(self):
        trades = [
            _mock_trade(
                entry_time=datetime(2024, 1, 15, 10, 0),
                exit_time=datetime(2024, 1, 17, 10, 0),
            )
            for _ in range(20)
        ]
        assert self.builder._classify_trading_style(trades) == TradingStyle.SWING_TRADER


class TestEntryStyleClassification:

    def setup_method(self):
        self.builder = TraderProfileBuilder(db=AsyncMock())

    def test_confirmation_style(self):
        trades = [_mock_trade(metadata={}) for _ in range(20)]
        result = self.builder._classify_entry_style(trades)
        assert result == EntryStyle.CONFIRMATION

    def test_reactive_style(self):
        trades = [_mock_trade(metadata={"chased_entry": True}) for _ in range(20)]
        result = self.builder._classify_entry_style(trades)
        assert result == EntryStyle.REACTIVE

    def test_anticipation_style(self):
        trades = [_mock_trade(metadata={"early_entry": True}) for _ in range(20)]
        result = self.builder._classify_entry_style(trades)
        assert result == EntryStyle.ANTICIPATION

    def test_mixed_style(self):
        trades = (
            [_mock_trade(metadata={"chased_entry": True}) for _ in range(7)] +
            [_mock_trade(metadata={"early_entry": True}) for _ in range(6)] +
            [_mock_trade(metadata={}) for _ in range(7)]
        )
        result = self.builder._classify_entry_style(trades)
        assert result == EntryStyle.MIXED


class TestRiskProfile:

    def setup_method(self):
        self.builder = TraderProfileBuilder(db=AsyncMock())

    def test_conservative_profile(self):
        trades = [
            _mock_trade(risk_percent=0.008)
            for _ in range(30)
        ]
        profile = self.builder._build_risk_profile(trades)
        assert profile.personality == RiskPersonality.CONSERVATIVE
        assert profile.risk_consistency > 0.7

    def test_aggressive_profile(self):
        trades = [
            _mock_trade(risk_percent=0.04)
            for _ in range(30)
        ]
        profile = self.builder._build_risk_profile(trades)
        assert profile.personality == RiskPersonality.AGGRESSIVE

    def test_inconsistent_profile(self):
        trades = (
            [_mock_trade(risk_percent=0.005) for _ in range(15)] +
            [_mock_trade(risk_percent=0.05) for _ in range(15)]
        )
        profile = self.builder._build_risk_profile(trades)
        assert profile.personality == RiskPersonality.INCONSISTENT

    def test_stop_widening_detected(self):
        total = 30
        widened = int(total * 0.20)  # 20% > 15% threshold
        trades = (
            [_mock_trade(metadata={"stop_moved_against": True}) for _ in range(widened)] +
            [_mock_trade(metadata={}) for _ in range(total - widened)]
        )
        profile = self.builder._build_risk_profile(trades)
        assert profile.tends_to_widen_stops is True


class TestTimingProfile:

    def setup_method(self):
        self.builder = TraderProfileBuilder(db=AsyncMock())

    def test_best_hours_identified(self):
        # Morning trades win, afternoon trades lose
        trades = (
            [_mock_trade(
                entry_time=datetime(2024, 1, d, 10, 0),
                exit_time=datetime(2024, 1, d, 10, 30),
                r_multiple=1.5,
            ) for d in range(1, 16)] +
            [_mock_trade(
                entry_time=datetime(2024, 1, d, 15, 0),
                exit_time=datetime(2024, 1, d, 15, 30),
                r_multiple=-0.8,
            ) for d in range(1, 16)]
        )
        timing = self.builder._build_timing_profile(trades, [])
        assert 10 in timing.best_hours
        assert 15 in timing.worst_hours

    def test_optimal_trade_count(self):
        # First 3 trades each day are good, after that quality drops
        trades = []
        for day in range(1, 21):
            for trade_num in range(6):
                r = 1.0 if trade_num < 3 else -0.5
                entry = datetime(2024, 1, day, 9 + trade_num, 30)
                trades.append(_mock_trade(
                    entry_time=entry,
                    exit_time=entry.replace(minute=59),
                    r_multiple=r,
                ))

        count = self.builder._find_optimal_trade_count(trades, [])
        assert count <= 4


class TestEmotionalProfile:

    def setup_method(self):
        self.builder = TraderProfileBuilder(db=AsyncMock())

    def test_triggers_from_events(self):
        events = [
            MagicMock(event_type="cognitive_bias", description="revenge_trading", created_at=datetime.utcnow()),
            MagicMock(event_type="cognitive_bias", description="revenge_trading", created_at=datetime.utcnow()),
            MagicMock(event_type="tilt_detected", description="tilt", created_at=datetime.utcnow()),
            MagicMock(event_type="cognitive_bias", description="fomo", created_at=datetime.utcnow()),
        ]

        profile = self.builder._build_emotional_profile([], events)
        assert "revenge_trading" in profile.primary_triggers

    def test_emotional_awareness_score(self):
        # All trades have emotional state logged
        trades = [
            _mock_trade(emotional_state="focused")
            for _ in range(20)
        ]
        profile = self.builder._build_emotional_profile(trades, [])
        assert profile.emotional_awareness_score >= 0.9

    def test_low_awareness_when_not_logged(self):
        trades = [
            _mock_trade(emotional_state=None)
            for _ in range(20)
        ]
        profile = self.builder._build_emotional_profile(trades, [])
        assert profile.emotional_awareness_score < 0.1


class TestProfileSerialization:

    def test_to_dict(self):
        profile = TraderIntelligenceProfile(
            trader_id="test",
            trading_style="day_trader",
            entry_style="confirmation",
            risk=RiskProfile(personality="moderate", avg_risk_per_trade=0.01),
            timing=TimingProfile(best_hours=[9, 10], optimal_trade_count=3),
            emotional=EmotionalProfile(primary_triggers=["fomo"]),
            current_discipline_score=0.82,
            sample_size=150,
            confidence=0.80,
        )

        d = profile.to_dict()
        assert d["trader_id"] == "test"
        assert d["trading_style"] == "day_trader"
        assert d["risk"]["personality"] == "moderate"
        assert d["timing"]["best_hours"] == [9, 10]
        assert d["emotional"]["primary_triggers"] == ["fomo"]
        assert d["sample_size"] == 150