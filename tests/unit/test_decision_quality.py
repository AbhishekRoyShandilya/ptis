"""
Unit tests for the Decision Quality Engine.

Tests each dimension scorer independently, the composite calculation,
the classifier, and the insight generator.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from analytics.decision_quality import (
    DecisionQualityAnalytics,
    DecisionQualityConfig,
    DecisionQualityEngine,
    DimensionName,
    EmotionalStateTag,
    ExecutionGrade,
    InsightSeverity,
    RecentTradeSnapshot,
    TradeContext,
    TradeOutcome,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def config() -> DecisionQualityConfig:
    return DecisionQualityConfig()


@pytest.fixture
def engine(config: DecisionQualityConfig) -> DecisionQualityEngine:
    return DecisionQualityEngine(config)


@pytest.fixture
def textbook_context() -> TradeContext:
    """A near‑perfect trade context for baseline testing."""
    now = datetime.utcnow()
    market_open = now - timedelta(minutes=45)

    return TradeContext(
        trade_id=uuid4(),
        symbol="AAPL",
        side="long",
        entry_price=150.00,
        exit_price=153.00,
        entry_time=now,
        exit_time=now + timedelta(minutes=20),
        quantity=100,
        pnl=300.0,
        r_multiple=2.0,
        strategy_name="opening_breakout",
        strategy_match_score=92.0,
        conditions_met=8,
        conditions_total=9,
        regime_compatible=True,
        planned_stop_price=148.50,
        actual_stop_price=148.50,
        planned_target_price=153.50,
        planned_position_size=100,
        actual_position_size=100,
        planned_risk_reward=2.3,
        portfolio_heat=0.02,
        max_daily_loss_limit=-500.0,
        current_daily_pnl=200.0,
        emotional_state=EmotionalStateTag.FOCUSED,
        confidence_level=80.0,
        override_events=[],
        override_severity=0.0,
        atr=1.80,
        volatility_regime="normal",
        trend_direction="up",
        trend_strength=72.0,
        vix_level=16.5,
        session_type="open",
        market_open_time=market_open,
        shadow_entry_price=149.90,
        shadow_exit_price=152.80,
        shadow_pnl=290.0,
        optimal_entry_price=149.95,
        optimal_exit_price=153.20,
        recent_trades=[
            RecentTradeSnapshot(
                pnl=100.0,
                outcome=TradeOutcome.WIN,
                exit_time=now - timedelta(minutes=30),
            ),
        ],
        trade_count_today=2,
        time_since_last_trade_seconds=1800.0,
    )


@pytest.fixture
def poor_context() -> TradeContext:
    """A poorly executed trade context."""
    now = datetime.utcnow()
    market_open = now - timedelta(hours=4)

    return TradeContext(
        trade_id=uuid4(),
        symbol="TSLA",
        side="long",
        entry_price=250.00,
        exit_price=245.00,
        entry_time=now,
        exit_time=now + timedelta(seconds=45),
        quantity=500,
        pnl=-2500.0,
        r_multiple=-3.5,
        strategy_name=None,
        strategy_match_score=None,
        conditions_met=None,
        conditions_total=None,
        regime_compatible=None,
        planned_stop_price=248.00,
        actual_stop_price=None,  # no stop placed!
        planned_target_price=255.00,
        planned_position_size=200,
        actual_position_size=500,  # 2.5x oversized
        planned_risk_reward=2.5,
        portfolio_heat=0.12,  # way over limit
        max_daily_loss_limit=-1000.0,
        current_daily_pnl=-1500.0,  # past daily limit
        emotional_state=EmotionalStateTag.REVENGE,
        confidence_level=95.0,
        override_events=[
            {"event_type": "position_size_override", "severity": 0.9},
            {"event_type": "stop_removed", "severity": 1.0},
        ],
        override_severity=1.0,
        atr=3.50,
        volatility_regime="extreme",
        trend_direction="down",
        trend_strength=80.0,
        vix_level=35.0,
        session_type="close",
        market_open_time=market_open,
        shadow_entry_price=251.00,
        shadow_exit_price=249.00,
        shadow_pnl=-200.0,
        recent_trades=[
            RecentTradeSnapshot(
                pnl=-500.0,
                outcome=TradeOutcome.LOSS,
                exit_time=now - timedelta(seconds=60),
            ),
            RecentTradeSnapshot(
                pnl=-300.0,
                outcome=TradeOutcome.LOSS,
                exit_time=now - timedelta(seconds=180),
            ),
            RecentTradeSnapshot(
                pnl=-200.0,
                outcome=TradeOutcome.LOSS,
                exit_time=now - timedelta(seconds=300),
            ),
        ],
        trade_count_today=12,
        time_since_last_trade_seconds=60.0,
    )


@pytest.fixture
def minimal_context() -> TradeContext:
    """Minimal data — tests graceful degradation."""
    return TradeContext(
        trade_id=uuid4(),
        symbol="SPY",
        side="long",
        entry_price=450.0,
        exit_price=451.0,
        pnl=100.0,
    )


# ═══════════════════════════════════════════════════════════════════
# CORE ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════

class TestDecisionQualityEngine:
    """Tests for the main engine."""

    def test_evaluate_returns_result(
        self, engine: DecisionQualityEngine, textbook_context: TradeContext
    ):
        result = engine.evaluate(textbook_context)
        assert result is not None
        assert result.trade_id == textbook_context.trade_id
        assert 0 <= result.final_score <= 100
        assert isinstance(result.grade, ExecutionGrade)
        assert len(result.dimensions) == 6

    def test_textbook_trade_scores_high(
        self, engine: DecisionQualityEngine, textbook_context: TradeContext
    ):
        result = engine.evaluate(textbook_context)
        assert result.final_score >= 75.0
        assert result.grade in (ExecutionGrade.TEXTBOOK, ExecutionGrade.GOOD)

    def test_poor_trade_scores_low(
        self, engine: DecisionQualityEngine, poor_context: TradeContext
    ):
        result = engine.evaluate(poor_context)
        assert result.final_score < 40.0
        assert result.grade in (
            ExecutionGrade.POOR,
            ExecutionGrade.UNDISCIPLINED,
        )

    def test_minimal_data_does_not_crash(
        self, engine: DecisionQualityEngine, minimal_context: TradeContext
    ):
        result = engine.evaluate(minimal_context)
        assert result is not None
        assert result.data_completeness < 0.5  # low completeness expected
        # Score should be near neutral
        assert 30 <= result.final_score <= 70

    def test_all_six_dimensions_present(
        self, engine: DecisionQualityEngine, textbook_context: TradeContext
    ):
        result = engine.evaluate(textbook_context)
        dim_names = {d.name for d in result.dimensions}
        assert dim_names == set(DimensionName)

    def test_dimension_weights_sum_to_one(self, config: DecisionQualityConfig):
        total = sum(config.weights.as_dict().values())
        assert abs(total - 1.0) < 1e-6

    def test_to_dict_serialization(
        self, engine: DecisionQualityEngine, textbook_context: TradeContext
    ):
        result = engine.evaluate(textbook_context)
        d = result.to_dict()
        assert "final_score" in d
        assert "dimensions" in d
        assert "insights" in d
        assert len(d["dimensions"]) == 6


# ═══════════════════════════════════════════════════════════════════
# DIMENSION SCORER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSetupAlignmentScorer:

    def test_high_match_scores_well(
        self, engine: DecisionQualityEngine, textbook_context: TradeContext
    ):
        result = engine.evaluate(textbook_context)
        dim = result.dimension(DimensionName.SETUP_ALIGNMENT)
        assert dim is not None
        assert dim.score >= 70.0

    def test_no_strategy_penalized(
        self, engine: DecisionQualityEngine, minimal_context: TradeContext
    ):
        result = engine.evaluate(minimal_context)
        dim = result.dimension(DimensionName.SETUP_ALIGNMENT)
        assert dim is not None
        assert dim.score < 70.0


class TestRiskDisciplineScorer:

    def test_no_stop_critical(
        self, engine: DecisionQualityEngine, poor_context: TradeContext
    ):
        result = engine.evaluate(poor_context)
        dim = result.dimension(DimensionName.RISK_DISCIPLINE)
        assert dim is not None
        assert dim.score < 30.0

    def test_oversizing_penalized(self, engine: DecisionQualityEngine):
        ctx = TradeContext(
            trade_id=uuid4(),
            planned_position_size=100,
            actual_position_size=300,  # 3x oversize
            planned_stop_price=95.0,
            actual_stop_price=95.0,
            entry_price=100.0,
            atr=2.0,
        )
        result = engine.evaluate(ctx)
        dim = result.dimension(DimensionName.RISK_DISCIPLINE)
        assert dim is not None
        assert dim.components.get("position_sizing", 100) < 15.0


class TestEmotionalStabilityScorer:

    def test_revenge_state_penalized(
        self, engine: DecisionQualityEngine, poor_context: TradeContext
    ):
        result = engine.evaluate(poor_context)
        dim = result.dimension(DimensionName.EMOTIONAL_STABILITY)
        assert dim is not None
        assert dim.score < 25.0

    def test_calm_state_scores_well(
        self, engine: DecisionQualityEngine, textbook_context: TradeContext
    ):
        result = engine.evaluate(textbook_context)
        dim = result.dimension(DimensionName.EMOTIONAL_STABILITY)
        assert dim is not None
        assert dim.score >= 70.0

    def test_overtrading_detected(self, engine: DecisionQualityEngine):
        ctx = TradeContext(
            trade_id=uuid4(),
            trade_count_today=15,
            emotional_state=EmotionalStateTag.BORED,
        )
        result = engine.evaluate(ctx)
        dim = result.dimension(DimensionName.EMOTIONAL_STABILITY)
        assert dim is not None
        assert dim.components.get("overtrading", 100) < 5.0


class TestExitDisciplineScorer:

    def test_good_exit_scores_well(
        self, engine: DecisionQualityEngine, textbook_context: TradeContext
    ):
        result = engine.evaluate(textbook_context)
        dim = result.dimension(DimensionName.EXIT_DISCIPLINE)
        assert dim is not None
        assert dim.score >= 60.0

    def test_premature_exit_penalized(self, engine: DecisionQualityEngine):
        ctx = TradeContext(
            trade_id=uuid4(),
            side="long",
            entry_price=100.0,
            exit_price=101.0,
            planned_target_price=105.0,  # captured only 20%
            planned_stop_price=98.0,
            actual_stop_price=98.0,
            pnl=100.0,
            atr=2.0,
            entry_time=datetime.utcnow(),
            exit_time=datetime.utcnow() + timedelta(minutes=5),
        )
        result = engine.evaluate(ctx)
        dim = result.dimension(DimensionName.EXIT_DISCIPLINE)
        assert dim is not None
        assert dim.components.get("target_adherence", 100) < 15.0


class TestMarketContextScorer:

    def test_counter_trend_penalized(self, engine: DecisionQualityEngine):
        ctx = TradeContext(
            trade_id=uuid4(),
            side="long",
            trend_direction="down",
            trend_strength=80.0,
            volatility_regime="high",
            regime_compatible=False,
            vix_level=32.0,
            session_type="close",
        )
        result = engine.evaluate(ctx)
        dim = result.dimension(DimensionName.MARKET_CONTEXT)
        assert dim is not None
        assert dim.score < 45.0


# ═══════════════════════════════════════════════════════════════════
# INSIGHT GENERATION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestInsightGeneration:

    def test_poor_trade_produces_critical_insights(
        self, engine: DecisionQualityEngine, poor_context: TradeContext
    ):
        result = engine.evaluate(poor_context)
        assert len(result.critical_insights) >= 1

    def test_textbook_trade_produces_positive_insights(
        self, engine: DecisionQualityEngine, textbook_context: TradeContext
    ):
        result = engine.evaluate(textbook_context)
        assert len(result.positive_insights) >= 1

    def test_insights_are_ordered_by_severity(
        self, engine: DecisionQualityEngine, poor_context: TradeContext
    ):
        result = engine.evaluate(poor_context)
        severity_order = {
            InsightSeverity.CRITICAL: 0,
            InsightSeverity.WARNING: 1,
            InsightSeverity.SUGGESTION: 2,
            InsightSeverity.POSITIVE: 3,
        }
        orders = [severity_order[i.severity] for i in result.insights]
        assert orders == sorted(orders)

    def test_cross_dimensional_tilt_cascade(
        self, engine: DecisionQualityEngine, poor_context: TradeContext
    ):
        result = engine.evaluate(poor_context)
        titles = [i.title for i in result.insights]
        assert "Tilt Cascade Detected" in titles

    def test_well_managed_loss_positive_reinforcement(
        self, engine: DecisionQualityEngine, textbook_context: TradeContext
    ):
        # Make it a loss but keep everything disciplined
        textbook_context.pnl = -150.0
        textbook_context.exit_price = 148.50
        textbook_context.r_multiple = -1.0
        result = engine.evaluate(textbook_context)
        titles = [i.title for i in result.insights]
        assert "Well‑Managed Loss" in titles


# ═══════════════════════════════════════════════════════════════════
# ANALYTICS AGGREGATOR TESTS
# ═══════════════════════════════════════════════════════════════════

class TestDecisionQualityAnalytics:

    def test_compute_averages(
        self, engine: DecisionQualityEngine, textbook_context: TradeContext
    ):
        results = [engine.evaluate(textbook_context) for _ in range(5)]
        avgs = DecisionQualityAnalytics.compute_averages(results)
        assert "overall" in avgs
        assert avgs["overall"] > 0

    def test_grade_distribution(
        self,
        engine: DecisionQualityEngine,
        textbook_context: TradeContext,
        poor_context: TradeContext,
    ):
        from analytics.decision_quality import DecisionQualityAnalytics

        results = [
            engine.evaluate(textbook_context),
            engine.evaluate(poor_context),
        ]
        dist = DecisionQualityAnalytics.grade_distribution(results)
        total = sum(dist.values())
        assert total == 2

    def test_coaching_summary_structure(
        self, engine: DecisionQualityEngine, textbook_context: TradeContext
    ):
        results = [engine.evaluate(textbook_context) for _ in range(10)]
        summary = DecisionQualityAnalytics.coaching_summary(results)
        assert "trade_count" in summary
        assert "avg_score" in summary
        assert "trend_direction" in summary
        assert summary["trade_count"] == 10

    def test_empty_results_handled(self):
        avgs = DecisionQualityAnalytics.compute_averages([])
        assert avgs["overall"] == 0.0

        summary = DecisionQualityAnalytics.coaching_summary([])
        assert summary["trade_count"] == 0


# ═══════════════════════════════════════════════════════════════════
# EXECUTION CLASSIFIER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestExecutionClassifier:

    @pytest.mark.parametrize(
        "score, expected",
        [
            (95.0, ExecutionGrade.TEXTBOOK),
            (85.0, ExecutionGrade.TEXTBOOK),
            (78.0, ExecutionGrade.GOOD),
            (70.0, ExecutionGrade.GOOD),
            (62.0, ExecutionGrade.ACCEPTABLE),
            (55.0, ExecutionGrade.ACCEPTABLE),
            (45.0, ExecutionGrade.POOR),
            (40.0, ExecutionGrade.POOR),
            (30.0, ExecutionGrade.UNDISCIPLINED),
            (10.0, ExecutionGrade.UNDISCIPLINED),
            (0.0, ExecutionGrade.UNDISCIPLINED),
        ],
    )
    def test_grade_boundaries(
        self,
        config: DecisionQualityConfig,
        score: float,
        expected: ExecutionGrade,
    ):
        from analytics.decision_quality import ExecutionClassifier

        classifier = ExecutionClassifier(config.grade_thresholds)
        assert classifier.classify(score) == expected


# ═══════════════════════════════════════════════════════════════════
# OUTCOME INDEPENDENCE TEST
# ═══════════════════════════════════════════════════════════════════

class TestOutcomeIndependence:
    """
    The CORE principle: decision quality should NOT change
    based on profit/loss alone.
    """

    def test_same_setup_different_outcomes(
        self, engine: DecisionQualityEngine, textbook_context: TradeContext
    ):
        # Evaluate as winner
        textbook_context.pnl = 500.0
        textbook_context.exit_price = 154.0
        textbook_context.r_multiple = 3.0
        result_win = engine.evaluate(textbook_context)

        # Evaluate as loser (but same process)
        textbook_context.pnl = -150.0
        textbook_context.exit_price = 148.50
        textbook_context.r_multiple = -1.0
        result_loss = engine.evaluate(textbook_context)

        # Scores should be close — the process was identical
        # We allow some variation because exit discipline
        # scoring does look at exit price relative to stop/target
        delta = abs(result_win.final_score - result_loss.final_score)
        assert delta < 20.0, (
            f"Outcome changed score by {delta:.1f} points — "
            f"win={result_win.final_score}, loss={result_loss.final_score}. "
            f"Should be outcome‑independent."
        )