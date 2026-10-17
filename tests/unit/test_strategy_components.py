import pandas as pd
import numpy as np
from strategy_engine.indicators import IndicatorEngine
from strategy_engine.consolidation import ConsolidationDetector
from strategy_engine.breakout_logic import BreakoutValidator
from strategy_engine.risk_rules import AdaptiveRiskManager
from strategy_engine.confidence import ConfidenceScorer
from core.enums import TradeDirection, VolatilityRegime, MarketRegime


def make_simple_df(n=20):
    ts = pd.date_range(start="2024-01-01", periods=n, freq="5min")
    prices = np.linspace(100, 120, n)
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.randint(1000, 2000, size=n),
    }, index=ts)
    return df


def test_indicator_engine_basic():
    df = make_simple_df(50)
    ema10 = IndicatorEngine.ema(df['close'], 10)
    assert len(ema10) == 50
    assert not ema10.isna().all()
    adx = IndicatorEngine.adx(df)
    assert isinstance(adx, pd.Series)


def test_consolidation_detector_no_zone():
    df = make_simple_df(50)
    cd = ConsolidationDetector()
    found, high, low, cnt = cd.find_consolidation(df)
    assert isinstance(found, bool)


def test_breakout_validator():
    df = make_simple_df(10)
    validator = BreakoutValidator()
    # resistance above all closes to force invalid
    valid, vol_ratio, body_ratio, reason = validator.validate(df, resistance=1000)
    assert not valid
    assert "Low volume" in reason or "Small body" in reason


def test_risk_manager_rr():
    df = make_simple_df(10)
    rm = AdaptiveRiskManager()
    sl = rm.calculate_sl(df, atr=1.0, volatility=VolatilityRegime.AVERAGE, direction=TradeDirection.LONG)
    assert isinstance(sl, float)
    target, trigger, rr = rm.calculate_rr(110, sl, TradeDirection.LONG, VolatilityRegime.AVERAGE, MarketRegime.RANGE)
    assert rr >= rm.min_rr


def test_confidence_scorer():
    scorer = ConfidenceScorer()
    class Dummy:
        daily_trend_up = True
        weekly_trend_up = True
        h1_supertrend_up = True
        m15_supertrend_up = True
        h1_above_50dma = True
        m15_above_50dma = True
    score = scorer.score(Dummy, 1.2, 1.3, 10, TradeDirection.LONG)
    assert 0 <= score <= 100


def test_regime_detector_state():
    from strategy_engine.regime_detector import MarketRegimeDetector

    # create minimal dataframes with linearly increasing prices
    def mk_df(n):
        ts = pd.date_range(start="2024-01-01", periods=n, freq="h")
        prices = np.linspace(100, 200, n)
        df = pd.DataFrame({
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.ones(n) * 1000,
        }, index=ts)
        return df

    detector = MarketRegimeDetector()
    d_daily = mk_df(100)
    d_weekly = mk_df(200)
    d_h1 = mk_df(50)
    d_m15 = mk_df(100)
    state = detector.get_regime_state(d_daily, d_weekly, d_h1, d_m15)
    assert isinstance(state, object)
    assert hasattr(state, "regime")
    assert hasattr(state, "volatility")
