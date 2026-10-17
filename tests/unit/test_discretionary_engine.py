"""Tests for the Discretionary Query Engine."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from intelligence.discretionary_engine import (
    DiscretionaryQueryEngine,
    DiscretionaryAssessment,
    QueryIntent,
    BehavioralGuardrail,
)
from intelligence.market_environment import MarketEnvironmentAnalyzer
from intelligence.readiness_scanner import SetupScanner, ScanResult, SetupReadiness
from intelligence.trader_profile_intelligence import (
    TraderIntelligenceProfile,
    RiskProfile,
    TimingProfile,
    EmotionalProfile,
    StrategyPreference,
)


def _make_df(bars=200):
    np.random.seed(42)
    closes = [150.0]
    for _ in range(bars - 1):
        closes.append(closes[-1] * (1 + np.random.normal(0.001, 0.008)))
    closes = np.array(closes)
    return pd.DataFrame({
        "open": closes * 0.999,
        "high": closes * 1.003,
        "low": closes * 0.997,
        "close": closes,
        "volume": np.random.poisson(100000, bars),
    })


def _make_profile(**overrides) -> TraderIntelligenceProfile:
    return TraderIntelligenceProfile(
        trader_id=overrides.get("trader_id", "test-trader"),
        trading_style=overrides.get("trading_style", "day_trader"),
        entry_style=overrides.get("entry_style", "confirmation"),
        risk=overrides.get("risk", RiskProfile(
            personality="moderate",
            avg_risk_per_trade=0.01,
            stop_adherence_rate=0.85,
        )),
        timing=overrides.get("timing", TimingProfile(
            best_hours=[9, 10, 11],
            worst_hours=[15, 16],
            optimal_trade_count=4,
            quality_decay_after_n_trades=4,
        )),
        emotional=overrides.get("emotional", EmotionalProfile(
            primary_triggers=["consecutive_losses"],
            post_loss_behavior="stays_disciplined",
            tilt_frequency=0.5,
        )),
        current_discipline_score=overrides.get("discipline_score", 0.80),
        strategy_preferences=overrides.get("strategy_preferences", [
            StrategyPreference(
                strategy_name="opening_breakout",
                frequency=0.40, win_rate=0.65, avg_r=1.2, is_core=True,
            ),
        ]),
        sample_size=overrides.get("sample_size", 100),
        confidence=overrides.get("confidence", 0.75),
    )


class TestQueryIntentParsing:

    def setup_method(self):
        self.engine = DiscretionaryQueryEngine(
            env_analyzer=MarketEnvironmentAnalyzer(),
            setup_scanner=MagicMock(spec=SetupScanner),
        )

    def test_setup_evaluation(self):
        assert self.engine._parse_intent("What do you think about this breakout?") == QueryIntent.SETUP_EVALUATION

    def test_entry_timing(self):
        assert self.engine._parse_intent("Should I enter this trade now?") == QueryIntent.ENTRY_TIMING

    def test_risk_assessment(self):
        assert self.engine._parse_intent("How much should I risk on this?") == QueryIntent.RISK_ASSESSMENT

    def test_exit_planning(self):
        assert self.engine._parse_intent("Where should I take profit?") == QueryIntent.EXIT_PLANNING

    def test_direction_bias(self):
        assert self.engine._parse_intent("Is NVDA bullish or bearish?") == QueryIntent.DIRECTION_BIAS

    def test_strategy_match(self):
        assert self.engine._parse_intent("Which strategy fits this pattern?") == QueryIntent.STRATEGY_MATCH

    def test_general_fallback(self):
        assert self.engine._parse_intent("Tell me about the market") == QueryIntent.GENERAL_ANALYSIS


class TestBehavioralGuardrails:

    def setup_method(self):
        self.engine = DiscretionaryQueryEngine(
            env_analyzer=MarketEnvironmentAnalyzer(),
            setup_scanner=MagicMock(spec=SetupScanner),
        )

    def test_critical_low_discipline(self):
        profile = _make_profile(discipline_score=0.35)
        assessment = DiscretionaryAssessment(symbol="AAPL", query="test", intent=QueryIntent.SETUP_EVALUATION)

        guardrails = self.engine._generate_guardrails(
            assessment, profile, todays_trade_count=2, current_pnl=0.0
        )

        critical = [g for g in guardrails if g.severity == "critical"]
        assert len(critical) >= 1
        assert "discipline" in critical[0].message.lower()

    def test_critical_overtrading(self):
        profile = _make_profile()
        assessment = DiscretionaryAssessment(
            symbol="AAPL", query="test", intent=QueryIntent.SETUP_EVALUATION,
            within_daily_trade_limit=False,
        )

        guardrails = self.engine._generate_guardrails(
            assessment, profile, todays_trade_count=6, current_pnl=0.0
        )

        critical = [g for g in guardrails if g.severity == "critical"]
        assert any("trade" in g.message.lower() for g in critical)

    def test_warning_worst_hours(self):
        profile = _make_profile()
        assessment = DiscretionaryAssessment(
            symbol="AAPL", query="test", intent=QueryIntent.SETUP_EVALUATION,
            timing_appropriate=False,
        )

        guardrails = self.engine._generate_guardrails(
            assessment, profile, todays_trade_count=1, current_pnl=0.0
        )

        warnings = [g for g in guardrails if g.severity == "warning"]
        assert any("hour" in g.message.lower() for g in warnings)

    def test_revenge_trading_warning(self):
        profile = _make_profile(
            emotional=EmotionalProfile(
                post_loss_behavior="revenge_trades",
                primary_triggers=["consecutive_losses"],
            )
        )
        assessment = DiscretionaryAssessment(
            symbol="AAPL", query="test", intent=QueryIntent.SETUP_EVALUATION,
        )

        guardrails = self.engine._generate_guardrails(
            assessment, profile, todays_trade_count=3, current_pnl=-500.0
        )

        warnings = [g for g in guardrails if g.severity == "warning"]
        assert any("revenge" in g.message.lower() for g in warnings)

    def test_no_guardrails_when_clean(self):
        profile = _make_profile(discipline_score=0.90)
        assessment = DiscretionaryAssessment(
            symbol="AAPL", query="test", intent=QueryIntent.SETUP_EVALUATION,
            timing_appropriate=True,
            within_daily_trade_limit=True,
            matches_trading_style=True,
            regime_aligned=True,
            environment_suitable=True,
        )

        guardrails = self.engine._generate_guardrails(
            assessment, profile, todays_trade_count=1, current_pnl=200.0
        )

        critical = [g for g in guardrails if g.severity == "critical"]
        assert len(critical) == 0


class TestRecommendationFinalization:

    def setup_method(self):
        self.engine = DiscretionaryQueryEngine(
            env_analyzer=MarketEnvironmentAnalyzer(),
            setup_scanner=MagicMock(spec=SetupScanner),
        )

    def test_strong_consider_when_all_aligned(self):
        profile = _make_profile(discipline_score=0.90)
        assessment = DiscretionaryAssessment(
            symbol="AAPL", query="test", intent=QueryIntent.SETUP_EVALUATION,
            setup_valid=True,
            setup_confidence=0.85,
            regime_aligned=True,
            environment_suitable=True,
            timing_appropriate=True,
            within_daily_trade_limit=True,
            within_risk_limits=True,
            matches_trading_style=True,
            risk_reward_assessment="favorable",
            environment_score=0.80,
        )
        assessment.best_match = MagicMock()
        assessment.best_match.strategy_name = "opening_breakout"
        assessment.best_match.readiness_score = 0.85
        assessment.best_match.is_forming = False
        assessment.best_match.conditions_met = 9
        assessment.best_match.conditions_total = 10
        assessment.r_estimate = MagicMock()
        assessment.r_estimate.expected_rr_ratio = 3.0
        assessment.r_estimate.win_probability = 0.65
        assessment.r_estimate.confidence = 0.8

        result = self.engine._finalize(
            assessment, profile, todays_trade_count=1, current_pnl=0.0
        )

        assert result.recommendation in {"strong_consider", "consider"}

    def test_avoid_when_critical_guardrails(self):
        profile = _make_profile(discipline_score=0.25)
        assessment = DiscretionaryAssessment(
            symbol="AAPL", query="test", intent=QueryIntent.SETUP_EVALUATION,
            setup_valid=False,
            within_daily_trade_limit=False,
            timing_appropriate=False,
        )
        assessment.guardrails = [
            BehavioralGuardrail("critical", "Low discipline", "Stop"),
            BehavioralGuardrail("critical", "Overtrading", "Stop"),
        ]

        result = self.engine._finalize(
            assessment, profile, todays_trade_count=8, current_pnl=-1000.0
        )

        assert result.recommendation == "avoid"

    def test_recommendation_reasons_populated(self):
        profile = _make_profile()
        assessment = DiscretionaryAssessment(
            symbol="AAPL", query="test", intent=QueryIntent.SETUP_EVALUATION,
            setup_valid=True,
            setup_confidence=0.75,
            regime_aligned=False,
        )
        assessment.best_match = MagicMock()
        assessment.best_match.strategy_name = "test"
        assessment.best_match.readiness_score = 0.75

        result = self.engine._finalize(
            assessment, profile, todays_trade_count=2, current_pnl=0.0
        )

        assert len(result.recommendation_reasons) > 0

        def test_context_for_coaching_built(self):
         profile = _make_profile()
         assessment = DiscretionaryAssessment(
            symbol="AAPL", query="test", intent=QueryIntent.SETUP_EVALUATION,
            setup_valid=True,
            setup_confidence=0.80,
        )
        assessment.best_match = MagicMock()
        assessment.best_match.strategy_name = "opening_breakout"
        assessment.best_match.readiness_score = 0.80
        assessment.best_match.conditions = []
        assessment.best_match.conditions_met = 8
        assessment.best_match.conditions_total = 10
        assessment.best_match.is_triggered = False
        assessment.best_match.is_forming = True

        assessment.environment = MagicMock()
        assessment.environment.regime.value = "trending_bullish"
        assessment.environment.trend.direction = 1
        assessment.environment.trend.strength.value = "moderate"
        assessment.environment.volatility.state.value = "stable"
        assessment.environment.environment_score = 0.72
        assessment.environment.current_price = 185.50
        assessment.environment.vwap = 184.20
        assessment.environment.structure.nearest_support = 182.0
        assessment.environment.structure.nearest_resistance = 188.0
        assessment.environment.structure.has_structure_break = False
        assessment.environment.structure.has_liquidity_sweep = False

        assessment.r_estimate = MagicMock()
        assessment.r_estimate.expected_r = 1.5
        assessment.r_estimate.expected_rr_ratio = 2.5
        assessment.r_estimate.win_probability = 0.60
        assessment.r_estimate.confidence = 0.65
        assessment.r_estimate.upside_r = 2.8
        assessment.r_estimate.downside_r = -1.0

        result = self.engine._finalize(
            assessment, profile, todays_trade_count=2, current_pnl=100.0
        )

        ctx = result.context_for_coaching
        assert "query_type" in ctx
        assert "environment" in ctx
        assert "setup" in ctx
        assert "r_estimate" in ctx
        assert "trader_state" in ctx
        assert "guardrails" in ctx
        assert ctx["symbol"] == "AAPL"
        assert ctx["trader_state"]["trading_style"] == "day_trader"


class TestDiscretionaryFullEvaluation:
    """Integration-style tests for the full evaluation pipeline."""

    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self):
        """Test the complete evaluation from query to assessment."""
        # Mock setup scanner
        mock_setup = SetupReadiness(
            strategy_name="opening_breakout",
            symbol="NVDA",
            timestamp=datetime.utcnow(),
            readiness_score=0.82,
            conditions_met=8,
            conditions_total=10,
            conditions_met_ratio=0.80,
            regime_aligned=True,
            volatility_suitable=True,
            environment_score=0.75,
            is_forming=True,
            is_triggered=False,
            estimated_r_multiple=1.5,
            r_confidence=0.60,
            estimated_rr_ratio=2.5,
            conditions=[],
        )

        mock_scan_result = ScanResult(
            symbol="NVDA",
            timestamp=datetime.utcnow(),
            environment=MagicMock(),
            setups=[mock_setup],
            top_setup=mock_setup,
            alerts=[],
        )

        mock_scanner = MagicMock(spec=SetupScanner)
        mock_scanner.scan.return_value = mock_scan_result

        engine = DiscretionaryQueryEngine(
            env_analyzer=MarketEnvironmentAnalyzer(),
            setup_scanner=mock_scanner,
            r_estimator=None,
        )

        df = _make_df()
        profile = _make_profile()

        assessment = await engine.evaluate(
            query="What do you think about this NVDA breakout?",
            symbol="NVDA",
            df=df,
            trader_profile=profile,
            direction=1,
            todays_trade_count=2,
            current_pnl=150.0,
        )

        assert isinstance(assessment, DiscretionaryAssessment)
        assert assessment.symbol == "NVDA"
        assert assessment.intent == QueryIntent.SETUP_EVALUATION
        assert assessment.recommendation in {
            "strong_consider", "consider", "caution", "avoid"
        }
        assert 0 <= assessment.confidence <= 1.0
        assert len(assessment.recommendation_reasons) > 0
        assert assessment.context_for_coaching is not None

    @pytest.mark.asyncio
    async def test_evaluation_with_bad_conditions(self):
        """Test evaluation when everything is against the trade."""
        mock_scanner = MagicMock(spec=SetupScanner)
        mock_scan_result = ScanResult(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            environment=MagicMock(),
            setups=[],
            top_setup=None,
            alerts=[],
        )
        mock_scanner.scan.return_value = mock_scan_result

        engine = DiscretionaryQueryEngine(
            env_analyzer=MarketEnvironmentAnalyzer(),
            setup_scanner=mock_scanner,
        )

        df = _make_df()
        profile = _make_profile(
            discipline_score=0.30,
            emotional=EmotionalProfile(
                post_loss_behavior="revenge_trades",
                primary_triggers=["consecutive_losses"],
            ),
        )

        assessment = await engine.evaluate(
            query="Should I enter this trade?",
            symbol="AAPL",
            df=df,
            trader_profile=profile,
            todays_trade_count=8,
            current_pnl=-800.0,
        )

        assert assessment.recommendation in {"caution", "avoid"}
        critical_guardrails = [
            g for g in assessment.guardrails if g.severity == "critical"
        ]
        assert len(critical_guardrails) >= 1

    @pytest.mark.asyncio
    async def test_evaluation_handles_env_failure(self):
        """Test graceful handling of environment analysis failure."""
        mock_scanner = MagicMock(spec=SetupScanner)

        engine = DiscretionaryQueryEngine(
            env_analyzer=MarketEnvironmentAnalyzer(),
            setup_scanner=mock_scanner,
        )

        # Very short dataframe that will fail analysis
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [10000, 10000],
        })

        profile = _make_profile()

        assessment = await engine.evaluate(
            query="Quick check on this",
            symbol="FAIL",
            df=df,
            trader_profile=profile,
        )

        # Should still return an assessment with reduced confidence
        assert isinstance(assessment, DiscretionaryAssessment)
        assert assessment.confidence < 0.5
        assert any(
            g.severity == "warning" for g in assessment.guardrails
        )