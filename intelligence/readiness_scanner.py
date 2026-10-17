"""
Strategy Readiness Scanner — Detects when market conditions align with strategy templates.

Scans real-time market data against the trader's configured strategies and
generates readiness scores, estimated time-to-trigger, and setup alerts.

Note: This is the *strategy-level* scanner (conditions vs market environment).
For the *technical pattern* scanner (candle patterns, pullbacks, reversals),
see automation/setup_scanner.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from intelligence.market_environment import (
    MarketEnvironment,
    MarketEnvironmentAnalyzer,
    RegimeType,
    VolatilityState,
    TrendStrength,
)
from strategy_engine.strategy_registry import StrategyRegistry
from strategy_engine.conditions.registry import ConditionRegistry

logger = logging.getLogger(__name__)


@dataclass
class ConditionStatus:
    """Status of a single entry condition."""
    name: str
    description: str
    is_met: bool
    value: Optional[Any] = None
    threshold: Optional[Any] = None
    proximity: float = 0.0  # 0-1, how close to being met
    notes: str = ""


@dataclass
class SetupReadiness:
    """
    How ready a specific strategy setup is to trigger.

    This is the core output of the scanner for each strategy/symbol pair.
    """
    strategy_name: str
    symbol: str
    timestamp: datetime

    # Readiness
    readiness_score: float  # 0-1, composite readiness
    conditions: list[ConditionStatus] = field(default_factory=list)
    conditions_met: int = 0
    conditions_total: int = 0
    conditions_met_ratio: float = 0.0

    # Estimation
    estimated_r_multiple: Optional[float] = None
    r_confidence: float = 0.0  # 0-1
    estimated_rr_ratio: Optional[float] = None

    # Environment alignment
    regime_aligned: bool = False
    regime_suitability: float = 0.0  # 0-1
    volatility_suitable: bool = True
    environment_score: float = 0.0

    # Timing
    time_window: str = ""  # e.g., "next 15 minutes", "opening range"
    is_forming: bool = False  # True if setup is actively forming
    is_triggered: bool = False  # True if all conditions met

    # Alert metadata
    alert_priority: str = "low"  # "critical", "high", "medium", "low"
    alert_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "readiness_score": round(self.readiness_score, 4),
            "conditions_met": self.conditions_met,
            "conditions_total": self.conditions_total,
            "conditions": [
                {
                    "name": c.name,
                    "description": c.description,
                    "is_met": c.is_met,
                    "proximity": round(c.proximity, 3),
                    "notes": c.notes,
                }
                for c in self.conditions
            ],
            "estimated_r_multiple": (
                round(self.estimated_r_multiple, 2)
                if self.estimated_r_multiple else None
            ),
            "r_confidence": round(self.r_confidence, 3),
            "estimated_rr_ratio": (
                round(self.estimated_rr_ratio, 2)
                if self.estimated_rr_ratio else None
            ),
            "regime_aligned": self.regime_aligned,
            "environment_score": round(self.environment_score, 3),
            "is_forming": self.is_forming,
            "is_triggered": self.is_triggered,
            "alert_priority": self.alert_priority,
            "alert_message": self.alert_message,
        }


@dataclass
class ScanResult:
    """Result of scanning all strategies against current market conditions."""
    symbol: str
    timestamp: datetime
    environment: MarketEnvironment
    setups: list[SetupReadiness]
    top_setup: Optional[SetupReadiness] = None
    alerts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "environment": self.environment.to_dict(),
            "setups": [s.to_dict() for s in self.setups],
            "top_setup": self.top_setup.to_dict() if self.top_setup else None,
            "alerts": self.alerts,
        }


class SetupScanner:
    """
    Scans market conditions against a trader's strategy templates.

    For each strategy, evaluates:
    1. Entry condition readiness
    2. Environment alignment
    3. R:R estimation
    4. Alert generation
    """

    def __init__(
        self,
        strategy_registry: StrategyRegistry,
        condition_registry: ConditionRegistry,
        env_analyzer: MarketEnvironmentAnalyzer,
        r_estimator: Optional[Any] = None,  # RMultipleEstimator
        forming_threshold: float = 0.60,
        alert_threshold: float = 0.75,
    ):
        self.strategies = strategy_registry
        self.conditions = condition_registry
        self.env_analyzer = env_analyzer
        self.r_estimator = r_estimator
        self.forming_threshold = forming_threshold
        self.alert_threshold = alert_threshold

    def scan(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "5m",
        strategy_names: Optional[list[str]] = None,
        trader_id: Optional[str] = None,
    ) -> ScanResult:
        """
        Scan all (or specified) strategies against current market data.

        Args:
            df: OHLCV DataFrame
            symbol: Ticker symbol
            timeframe: Candle timeframe
            strategy_names: Specific strategies to check (None = all)
            trader_id: For personalized R:R estimation

        Returns:
            ScanResult with all setup readiness assessments
        """
        # 1. Analyze market environment
        environment = self.env_analyzer.analyze(df, symbol, timeframe)

        # 2. Get strategies to scan
        if strategy_names:
            strategies = [
                self.strategies.get(name)
                for name in strategy_names
                if self.strategies.get(name) is not None
            ]
        else:
            strategies = self.strategies.get_all_active()

        # 3. Evaluate each strategy
        setups: list[SetupReadiness] = []
        for strategy in strategies:
            readiness = self._evaluate_strategy(
                strategy=strategy,
                df=df,
                symbol=symbol,
                environment=environment,
                trader_id=trader_id,
            )
            setups.append(readiness)

        # 4. Sort by readiness
        setups.sort(key=lambda s: s.readiness_score, reverse=True)
        top_setup = setups[0] if setups else None

        # 5. Generate alerts
        alerts = self._generate_alerts(setups, environment)

        return ScanResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            environment=environment,
            setups=setups,
            top_setup=top_setup,
            alerts=alerts,
        )

    def _evaluate_strategy(
        self,
        strategy: Any,
        df: pd.DataFrame,
        symbol: str,
        environment: MarketEnvironment,
        trader_id: Optional[str] = None,
    ) -> SetupReadiness:
        """Evaluate a single strategy against current conditions."""
        strategy_name = getattr(strategy, "name", str(strategy))
        now = datetime.utcnow()

        # Evaluate entry conditions
        condition_statuses = self._check_conditions(strategy, df, environment)
        met = sum(1 for c in condition_statuses if c.is_met)
        total = len(condition_statuses) if condition_statuses else 1
        ratio = met / total

        # Regime alignment
        regime_aligned = self._check_regime_alignment(strategy, environment)
        regime_suitability = self._compute_regime_suitability(
            strategy, environment
        )

        # Volatility suitability
        vol_suitable = self._check_volatility_suitability(strategy, environment)

        # R:R estimation
        estimated_r, r_confidence, estimated_rr = self._estimate_r_multiple(
            strategy=strategy,
            df=df,
            environment=environment,
            trader_id=trader_id,
        )

        # Environment score
        env_score = environment.environment_score

        # Composite readiness score
        readiness = self._compute_readiness_score(
            conditions_ratio=ratio,
            regime_suitability=regime_suitability,
            vol_suitable=vol_suitable,
            env_score=env_score,
        )

        # Determine state
        is_triggered = ratio >= 0.95 and regime_aligned and vol_suitable
        is_forming = (
            ratio >= self.forming_threshold and not is_triggered
        )

        # Alert priority
        alert_priority = self._determine_alert_priority(
            readiness, is_triggered, is_forming, regime_aligned
        )

        # Generate alert message
        alert_message = self._build_alert_message(
            strategy_name=strategy_name,
            symbol=symbol,
            readiness=readiness,
            is_forming=is_forming,
            is_triggered=is_triggered,
            conditions_met=met,
            conditions_total=total,
            estimated_r=estimated_r,
            regime_aligned=regime_aligned,
        )

        # Time window from strategy config
        time_window = getattr(strategy, "time_window", "") or ""

        return SetupReadiness(
            strategy_name=strategy_name,
            symbol=symbol,
            timestamp=now,
            readiness_score=readiness,
            conditions=condition_statuses,
            conditions_met=met,
            conditions_total=total,
            conditions_met_ratio=ratio,
            estimated_r_multiple=estimated_r,
            r_confidence=r_confidence,
            estimated_rr_ratio=estimated_rr,
            regime_aligned=regime_aligned,
            regime_suitability=regime_suitability,
            volatility_suitable=vol_suitable,
            environment_score=env_score,
            time_window=time_window,
            is_forming=is_forming,
            is_triggered=is_triggered,
            alert_priority=alert_priority,
            alert_message=alert_message,
        )

    def _check_conditions(
        self,
        strategy: Any,
        df: pd.DataFrame,
        environment: MarketEnvironment,
    ) -> list[ConditionStatus]:
        """Check each entry condition for the strategy."""
        statuses = []

        # Support both StrategyTemplate (entry_confirmations: list[ConditionDef])
        # and DNA-style objects (entry_conditions: list[dict]).
        raw_conditions: list[Any] = (
            getattr(strategy, "entry_confirmations", None)
            or getattr(strategy, "entry_conditions", None)
            or []
        )

        for cond in raw_conditions:
            # Normalise to dict-like access
            if hasattr(cond, "type"):
                # ConditionDef dataclass
                condition_name = cond.type
                condition_params = cond.params if cond.params else {}
                description = cond.description or condition_name
            elif isinstance(cond, dict):
                condition_name = cond.get("type", "unknown")
                condition_params = cond.get("params", {})
                description = cond.get("description", condition_name)
            else:
                continue

            try:
                result = self._eval_condition_env(
                    condition_name, condition_params, df, environment
                )
                statuses.append(ConditionStatus(
                    name=condition_name,
                    description=description,
                    is_met=result.get("met", False),
                    value=result.get("value"),
                    threshold=result.get("threshold"),
                    proximity=result.get("proximity", 0.0),
                    notes=result.get("notes", ""),
                ))
            except Exception as e:
                logger.warning("Condition evaluation failed for %s: %s", condition_name, e)
                statuses.append(ConditionStatus(
                    name=condition_name,
                    description=description,
                    is_met=False,
                    notes=f"Evaluation error: {e}",
                ))

        return statuses

    # ── regime value mapping ────────────────────────────────────────
    # Bridges YAML MarketRegime values → RegimeType values used by MarketEnvironment
    _REGIME_MAP: dict[str, str] = {
        "high_volatility":           "volatile_expansion",
        "volatile":                  "volatile_expansion",
        "low_volatility":            "low_volatility_compression",
        "low_vol":                   "low_volatility_compression",
        "trending_up":               "trending_bullish",
        "trending_down":             "trending_bearish",
        "mean_reverting":            "ranging",
        "range":                     "ranging",
    }

    def _eval_condition_env(
        self,
        condition_type: str,
        params: dict,
        df: pd.DataFrame,
        environment: MarketEnvironment,
    ) -> dict:
        """
        Evaluate a condition using market environment data.
        Returns dict with keys: met (bool), proximity (float 0-1), notes (str),
        optionally value and threshold.
        """
        try:
            price = environment.current_price or (df["close"].iloc[-1] if len(df) else 0.0)
            vwap  = environment.vwap
            atr   = environment.atr or 0.0
            rel_vol = environment.relative_volume or 1.0
            trend_dir = environment.trend_direction.value   # "up"/"down"/"neutral"
            vol_state  = environment.volatility.state.value  # VolatilityState
            range_state = environment.range_state.value      # "expanding"/"compressing"/"normal"
            trend_str  = environment.trend_strength           # ADX

            # ── Volume conditions ──────────────────────────────────
            if condition_type in ("volume_surge", "volume_spike"):
                min_ratio = params.get("min_ratio", 2.0)
                met = rel_vol >= min_ratio
                return {"met": met, "value": rel_vol, "threshold": min_ratio,
                        "proximity": min(1.0, rel_vol / max(min_ratio, 0.01))}

            if condition_type in ("volume_above_avg", "high_volume"):
                min_ratio = params.get("min_ratio", 1.5)
                met = rel_vol >= min_ratio
                return {"met": met, "value": rel_vol, "threshold": min_ratio,
                        "proximity": min(1.0, rel_vol / max(min_ratio, 0.01))}

            # ── Trend / EMA conditions ─────────────────────────────
            if condition_type in ("ema_alignment", "trend_aligned"):
                met = trend_dir == "up" and environment.ema_fast and environment.ema_slow and environment.ema_fast > environment.ema_slow
                return {"met": met, "proximity": 0.85 if met else 0.15}

            if condition_type in ("price_above_ema", "above_ema"):
                ema = environment.ema_fast or environment.ema_slow
                if ema:
                    met = price > ema
                    dist_pct = (price - ema) / ema * 100
                    return {"met": met, "value": dist_pct, "proximity": 0.8 if met else 0.2}
                return {"met": False, "proximity": 0.3, "notes": "EMA not available"}

            if condition_type == "pullback_to_ema":
                ema = environment.ema_fast or environment.ema_slow
                if ema:
                    dist_pct = abs(price - ema) / ema * 100
                    met = dist_pct <= 1.0  # within 1% of EMA
                    proximity = max(0.0, 1.0 - dist_pct / 3.0)
                    return {"met": met, "value": dist_pct, "threshold": 1.0,
                            "proximity": proximity, "notes": f"EMA distance: {dist_pct:.1f}%"}
                return {"met": False, "proximity": 0.3}

            # ── Breakout / structure conditions ────────────────────
            if condition_type in ("breakout_candle", "price_breakout", "close_above_resistance"):
                # Use ATR expansion + uptrend as proxy for breakout candle
                expanding = range_state == "expanding"
                bullish = trend_dir == "up"
                met = expanding and bullish
                return {"met": met, "proximity": 0.75 if met else (0.45 if expanding or bullish else 0.2)}

            if condition_type == "bb_squeeze":
                is_comp = vol_state in ("compressing", "extreme_low", "extreme low")
                return {"met": is_comp, "proximity": 0.85 if is_comp else 0.2}

            if condition_type in ("atr_expansion", "range_expansion"):
                is_exp = range_state == "expanding"
                return {"met": is_exp, "proximity": 0.85 if is_exp else 0.25}

            # ── VWAP conditions ────────────────────────────────────
            if condition_type in ("close_above_vwap", "price_above_vwap"):
                if vwap:
                    met = price > vwap
                    dist_pct = (price - vwap) / vwap * 100
                    return {"met": met, "value": dist_pct,
                            "proximity": min(1.0, 0.5 + dist_pct / 2.0) if met else max(0.0, 0.5 + dist_pct / 2.0)}
                return {"met": False, "proximity": 0.4, "notes": "VWAP not available"}

            if condition_type == "vwap_support":
                if vwap:
                    dist_pct = abs(price - vwap) / vwap * 100
                    met = price > vwap and dist_pct <= 1.5
                    proximity = max(0.0, 1.0 - dist_pct / 3.0)
                    return {"met": met, "proximity": proximity}
                return {"met": False, "proximity": 0.3}

            # ── Momentum / RSI conditions ──────────────────────────
            if condition_type == "rsi_extreme":
                # No RSI in environment; use trend + vol as proxy
                overbought = params.get("overbought", 75)
                oversold = params.get("oversold", 25)
                # High vol + strong trend direction = likely extreme RSI
                is_extreme = (
                    (trend_dir == "up" and trend_str > 35 and rel_vol > 1.5)
                    or (trend_dir == "down" and trend_str > 35 and rel_vol > 1.5)
                )
                return {"met": is_extreme, "proximity": 0.6 if is_extreme else 0.25,
                        "notes": "Estimated from trend strength + volume (no direct RSI)"}

            if condition_type == "price_extension":
                min_atr = params.get("min_atr_distance", 2.0)
                if vwap and atr > 0:
                    dist_atr = abs(price - vwap) / atr
                    met = dist_atr >= min_atr
                    return {"met": met, "value": round(dist_atr, 2), "threshold": min_atr,
                            "proximity": min(1.0, dist_atr / min_atr)}
                return {"met": False, "proximity": 0.3}

            # ── ORB conditions ─────────────────────────────────────
            if condition_type in ("orb_high_break", "orb_low_break"):
                # Proxy: early session + range expanding + bullish/bearish
                from intelligence.market_environment import SessionPhase
                is_early = environment.session_phase in (SessionPhase.OPENING, SessionPhase.MIDDAY)
                expanding = range_state == "expanding"
                is_long = condition_type == "orb_high_break"
                dir_ok = (trend_dir == "up") == is_long
                met = is_early and expanding and dir_ok
                return {"met": met, "proximity": 0.7 if (is_early and expanding) else 0.3}

            # ── Candle pattern conditions ──────────────────────────
            if condition_type in ("exhaustion_candle", "reversal_candle"):
                # Need actual candle analysis from df
                if len(df) >= 2:
                    last = df.iloc[-1]
                    hi = float(last.get("high", last.get("High", 0)))
                    lo = float(last.get("low", last.get("Low", 0)))
                    op = float(last.get("open", last.get("Open", 0)))
                    cl = float(last.get("close", last.get("Close", 0)))
                    total_range = hi - lo
                    if total_range > 0:
                        body = abs(cl - op)
                        wick = total_range - body
                        wick_ratio = wick / total_range
                        min_wick = params.get("min_wick_ratio", 0.6)
                        met = wick_ratio >= min_wick
                        return {"met": met, "value": round(wick_ratio, 2), "threshold": min_wick,
                                "proximity": min(1.0, wick_ratio / min_wick)}
                return {"met": False, "proximity": 0.3}

            # ── Gap conditions ─────────────────────────────────────
            if condition_type == "gap_detected":
                if len(df) >= 2:
                    prev_close = float(df.iloc[-2].get("close", df.iloc[-2].get("Close", 0)))
                    curr_open  = float(df.iloc[-1].get("open", df.iloc[-1].get("Open", 0)))
                    if prev_close > 0:
                        gap_pct = abs(curr_open - prev_close) / prev_close * 100
                        min_gap = params.get("min_gap_pct", 1.0)
                        max_gap = params.get("max_gap_pct", 3.0)
                        met = min_gap <= gap_pct <= max_gap
                        return {"met": met, "value": round(gap_pct, 2),
                                "proximity": min(1.0, gap_pct / min_gap) if gap_pct < max_gap else 0.1}
                return {"met": False, "proximity": 0.1}

            # ── Expiry / calendar conditions ───────────────────────
            if condition_type == "expiry_day":
                from datetime import datetime as _dt
                day = _dt.now().weekday()  # 0=Mon … 6=Sun
                expected = params.get("day_of_week", 3)  # 3=Thu
                met = (day == expected)
                return {"met": met, "proximity": 1.0 if met else 0.0}

            if condition_type == "overextension":
                min_atr = params.get("min_atr_distance", 1.5)
                if vwap and atr > 0:
                    dist_atr = abs(price - vwap) / atr
                    met = dist_atr >= min_atr
                    return {"met": met, "value": round(dist_atr, 2), "threshold": min_atr,
                            "proximity": min(1.0, dist_atr / min_atr)}
                return {"met": False, "proximity": 0.3}

            # ── VWAP proximity (for VWAP Bounce) ──────────────────
            if condition_type == "vwap_proximity":
                max_dist = params.get("max_distance_pct", 0.3)
                if vwap and vwap > 0:
                    dist_pct = abs(price - vwap) / vwap * 100
                    met = dist_pct <= max_dist
                    proximity = max(0.0, 1.0 - dist_pct / (max_dist * 5))
                    return {"met": met, "value": round(dist_pct, 2), "threshold": max_dist,
                            "proximity": proximity, "notes": f"VWAP distance: {dist_pct:.2f}%"}
                return {"met": False, "proximity": 0.3, "notes": "VWAP not available"}

            if condition_type == "trend_alignment":
                ema_f = environment.ema_fast
                ema_s = environment.ema_slow
                if ema_f and ema_s:
                    bullish_aligned = (trend_dir == "up" and ema_f > ema_s and price > ema_f)
                    bearish_aligned = (trend_dir == "down" and ema_f < ema_s and price < ema_f)
                    met = bullish_aligned or bearish_aligned
                    return {"met": met, "proximity": 0.85 if met else 0.2}
                # Fallback: use trend direction
                met = trend_dir != "neutral" and trend_str > 25
                return {"met": met, "proximity": 0.7 if met else 0.25}

            if condition_type == "ema_proximity":
                max_dist = params.get("max_distance_pct", 0.5)
                ema = environment.ema_fast or environment.ema_slow
                if ema and ema > 0:
                    dist_pct = abs(price - ema) / ema * 100
                    met = dist_pct <= max_dist
                    proximity = max(0.0, 1.0 - dist_pct / (max_dist * 4))
                    return {"met": met, "value": round(dist_pct, 2), "threshold": max_dist,
                            "proximity": proximity, "notes": f"EMA distance: {dist_pct:.2f}%"}
                return {"met": False, "proximity": 0.3}

            if condition_type == "momentum_reset":
                # RSI in mid zone (40-60) after a trend move — proxy: trend strength moderate
                rsi_min = params.get("rsi_min", 35)
                rsi_max = params.get("rsi_max", 65)
                # Estimate: range compressing after expansion = momentum cooling
                is_reset = vol_state in ("compressing", "stable")
                return {"met": is_reset, "proximity": 0.65 if is_reset else 0.3,
                        "notes": "Estimated from volatility state (no direct RSI)"}

            if condition_type == "rejection_candle":
                # Candle wick analysis
                if len(df) >= 1:
                    last = df.iloc[-1]
                    hi = float(last.get("high", last.get("High", 0)))
                    lo = float(last.get("low", last.get("Low", 0)))
                    op = float(last.get("open", last.get("Open", 0)))
                    cl = float(last.get("close", last.get("Close", 0)))
                    total_range = hi - lo
                    if total_range > 0:
                        body = abs(cl - op)
                        wick = total_range - body
                        wick_ratio = wick / total_range
                        min_wick = params.get("min_wick_ratio", 0.5)
                        met = wick_ratio >= min_wick
                        return {"met": met, "value": round(wick_ratio, 2), "threshold": min_wick,
                                "proximity": min(1.0, wick_ratio / min_wick)}
                return {"met": False, "proximity": 0.3}

            if condition_type in ("orb_breakout", "range_breakout"):
                from intelligence.market_environment import SessionPhase
                orb_high = environment.orb_high
                orb_low = environment.orb_low
                curr = price
                is_after_hours = environment.session_phase == SessionPhase.AFTER_HOURS

                if orb_high and orb_low and curr:
                    # Exact check: price above ORB high = bullish breakout; below ORB low = bearish
                    bullish_bo = curr > orb_high
                    bearish_bo = curr < orb_low
                    met = bullish_bo or bearish_bo

                    if bullish_bo:
                        dist_pct = (curr - orb_high) / orb_high * 100
                        proximity = min(1.0, 0.7 + dist_pct * 0.1)
                        direction = "bullish"
                    elif bearish_bo:
                        dist_pct = (orb_low - curr) / orb_low * 100
                        proximity = min(1.0, 0.7 + dist_pct * 0.1)
                        direction = "bearish"
                    else:
                        orb_range = orb_high - orb_low
                        dist_to_high = (orb_high - curr) / max(orb_range, 0.01)
                        dist_to_low = (curr - orb_low) / max(orb_range, 0.01)
                        proximity = 1.0 - min(dist_to_high, dist_to_low)
                        direction = "neutral"

                    if is_after_hours and met:
                        status = f"Triggered {direction} earlier today — evaluating continuation/retest"
                    elif is_after_hours and not met:
                        status = "ORB never triggered today — price stayed inside opening range"
                    elif met:
                        status = f"Broken {direction} — ₹{curr:.1f} vs ORB {direction} edge"
                    else:
                        status = "Inside ORB — no breakout yet"

                    notes = (
                        f"ORB range: ₹{orb_low:.1f}–₹{orb_high:.1f} | "
                        f"Current: ₹{curr:.1f} | {status}"
                    )
                    return {"met": met, "proximity": round(proximity, 3), "notes": notes}

                # ORB not yet formed (very early session or no session bars)
                in_session = environment.session_phase in (
                    SessionPhase.OPENING, SessionPhase.MIDDAY
                )
                return {"met": False, "proximity": 0.3 if in_session else 0.1,
                        "notes": "Opening range not yet established"}

            if condition_type == "prior_day_context":
                # Price above/below prior close — proxy: trend direction
                met = trend_dir != "neutral"
                return {"met": met, "proximity": 0.7 if met else 0.3}

            if condition_type in ("gap_fill_momentum",):
                # First candle shows reversal — proxy: vol expanding on opposite direction
                met = rel_vol >= 1.5
                return {"met": met, "proximity": min(1.0, rel_vol / 1.5)}

            if condition_type in ("volume_climax", "climax_volume"):
                # Exhaustion volume: unusually high surge (≥ 3× average)
                min_ratio = params.get("min_ratio", 3.0)
                met = rel_vol >= min_ratio
                return {"met": met, "value": rel_vol, "threshold": min_ratio,
                        "proximity": min(1.0, rel_vol / max(min_ratio, 0.01))}

            # ── Conditions requiring external data — partial credit ──
            if condition_type in ("sector_confirmation", "sector_relative_strength"):
                # Needs sector ETF feed — not wired yet; soft pass (0.5 proximity)
                return {"met": False, "proximity": 0.5,
                        "notes": "Sector data not yet available — condition skipped"}

            if condition_type in ("max_pain_proximity", "max_pain"):
                # Needs options chain data — not wired yet; soft pass
                return {"met": False, "proximity": 0.5,
                        "notes": "Options chain data not yet available — condition skipped"}

            if condition_type in ("vix_elevated", "india_vix"):
                # Needs India VIX feed — not wired yet; soft pass
                return {"met": False, "proximity": 0.5,
                        "notes": "India VIX data not yet available — condition skipped"}

            # ── Default: low proximity, don't penalise heavily ─────
            return {"met": False, "proximity": 0.3,
                    "notes": f"No evaluator for '{condition_type}' — assuming not met"}

        except Exception as e:
            return {"met": False, "proximity": 0.2, "notes": f"Eval error: {e}"}

    def _check_regime_alignment(
        self, strategy: Any, environment: MarketEnvironment
    ) -> bool:
        """Check if current regime matches strategy's preferred regimes."""
        preferred_regimes = getattr(strategy, "preferred_regimes", [])
        if not preferred_regimes:
            return True  # No preference = always aligned

        current = environment.regime.value
        # Build set of canonical values to check against (with alias expansion)
        current_set = {current, self._REGIME_MAP.get(current, current)}

        for pr in preferred_regimes:
            pr_val = pr.value if hasattr(pr, "value") else str(pr)
            pr_canonical = self._REGIME_MAP.get(pr_val, pr_val)
            if pr_val in current_set or pr_canonical in current_set:
                return True
        return False

    def _compute_regime_suitability(
        self, strategy: Any, environment: MarketEnvironment
    ) -> float:
        """Compute how suitable the current regime is (0-1)."""
        preferred = getattr(strategy, "preferred_regimes", [])
        avoid = getattr(strategy, "avoid_regimes", [])
        current = environment.regime.value
        current_set = {current, self._REGIME_MAP.get(current, current)}

        def _matches(regime_list: list) -> bool:
            for r in regime_list:
                rv = r.value if hasattr(r, "value") else str(r)
                rc = self._REGIME_MAP.get(rv, rv)
                if rv in current_set or rc in current_set:
                    return True
            return False

        if _matches(avoid):
            return 0.1
        if not preferred:
            return 0.6  # Neutral if no preference
        if _matches(preferred):
            return 0.9
        return 0.4  # Not preferred but not avoided

    def _check_volatility_suitability(
        self, strategy: Any, environment: MarketEnvironment
    ) -> bool:
        """Check if volatility conditions suit this strategy."""
        vol = environment.volatility

        # Strategy-specific volatility preferences
        min_atr_ratio = getattr(strategy, "min_atr_ratio", None)
        max_atr_ratio = getattr(strategy, "max_atr_ratio", None)
        requires_compression = getattr(strategy, "requires_compression", False)
        requires_expansion = getattr(strategy, "requires_expansion", False)

        if min_atr_ratio and vol.atr_ratio < min_atr_ratio:
            return False
        if max_atr_ratio and vol.atr_ratio > max_atr_ratio:
            return False
        if requires_compression and not vol.is_compression:
            return False
        if requires_expansion and not vol.is_expansion:
            return False

        return True

    def _estimate_r_multiple(
        self,
        strategy: Any,
        df: pd.DataFrame,
        environment: MarketEnvironment,
        trader_id: Optional[str] = None,
    ) -> tuple[Optional[float], float, Optional[float]]:
        """
        Estimate expected R-multiple for this setup.

        Returns: (estimated_r, confidence, estimated_rr_ratio)
        """
        if self.r_estimator is not None:
            try:
                import inspect
                estimate = self.r_estimator.estimate(
                    strategy_name=strategy.name if hasattr(strategy, "name") else str(strategy),
                    environment=environment,
                    trader_id=trader_id,
                )
                # estimate() is async — if called from sync context we get a coroutine
                if inspect.iscoroutine(estimate):
                    estimate.close()  # clean up coroutine, fall through to fallback
                    raise RuntimeError("r_estimator.estimate is async — use fallback")
                return (
                    estimate.get("expected_r"),
                    estimate.get("confidence", 0.5),
                    estimate.get("rr_ratio"),
                )
            except Exception as e:
                logger.debug(f"R estimation skipped: {e}")

        # Fallback: use strategy default R:R and adjust for volatility
        default_rr = getattr(strategy, "default_rr", 2.0)
        vol_multiplier = environment.volatility.environment_multiplier

        # Regime adjustment
        regime_factor = 1.0
        if environment.trend.is_trending and environment.trend.strength == TrendStrength.STRONG:
            regime_factor = 1.2
        elif not environment.trend.is_trending:
            regime_factor = 0.8

        adjusted_rr = default_rr * vol_multiplier * regime_factor
        confidence = 0.4  # Low confidence for fallback estimate

        return (adjusted_rr * 0.5, confidence, adjusted_rr)  # Expected R ~ 50% of RR

    def _compute_readiness_score(
        self,
        conditions_ratio: float,
        regime_suitability: float,
        vol_suitable: bool,
        env_score: float,
    ) -> float:
        """Compute composite readiness score."""
        score = (
            conditions_ratio * 0.50
            + regime_suitability * 0.20
            + env_score * 0.15
            + (0.15 if vol_suitable else 0.0)
        )
        return max(0.0, min(1.0, score))

    def _determine_alert_priority(
        self,
        readiness: float,
        is_triggered: bool,
        is_forming: bool,
        regime_aligned: bool,
    ) -> str:
        if is_triggered and regime_aligned:
            return "critical"
        if is_triggered:
            return "high"
        if is_forming and regime_aligned and readiness > self.alert_threshold:
            return "high"
        if is_forming:
            return "medium"
        if readiness > self.forming_threshold:
            return "low"
        return "low"

    def _build_alert_message(
        self,
        strategy_name: str,
        symbol: str,
        readiness: float,
        is_forming: bool,
        is_triggered: bool,
        conditions_met: int,
        conditions_total: int,
        estimated_r: Optional[float],
        regime_aligned: bool,
    ) -> str:
        if is_triggered:
            msg = (
                f"🔔 {strategy_name} TRIGGERED on {symbol} — "
                f"{conditions_met}/{conditions_total} conditions met"
            )
            if estimated_r:
                msg += f" — Est. R: {estimated_r:+.1f}"
            if regime_aligned:
                msg += " — ✅ Regime aligned"
            return msg

        if is_forming:
            msg = (
                f"⚡ {strategy_name} forming on {symbol} — "
                f"{conditions_met}/{conditions_total} conditions met "
                f"({readiness:.0%} ready)"
            )
            if not regime_aligned:
                msg += " — ⚠️ Regime not aligned"
            return msg

        return ""

    def _generate_alerts(
        self,
        setups: list[SetupReadiness],
        environment: MarketEnvironment,
    ) -> list[dict[str, Any]]:
        """Generate alert objects for all noteworthy setups."""
        alerts = []

        for setup in setups:
            if setup.alert_message:
                alerts.append({
                    "priority": setup.alert_priority,
                    "strategy": setup.strategy_name,
                    "symbol": setup.symbol,
                    "message": setup.alert_message,
                    "readiness": setup.readiness_score,
                    "is_triggered": setup.is_triggered,
                    "is_forming": setup.is_forming,
                    "estimated_r": setup.estimated_r_multiple,
                    "timestamp": setup.timestamp.isoformat(),
                })

        # Regime shift alert
        if environment.structure.has_structure_break:
            alerts.append({
                "priority": "high",
                "strategy": "_system",
                "symbol": environment.symbol,
                "message": (
                    f"📊 Structure break detected on {environment.symbol} — "
                    f"Regime: {environment.regime.value}"
                ),
                "timestamp": environment.timestamp.isoformat(),
            })

        if environment.structure.has_liquidity_sweep:
            alerts.append({
                "priority": "medium",
                "strategy": "_system",
                "symbol": environment.symbol,
                "message": (
                    f"💧 Liquidity sweep detected on {environment.symbol}"
                ),
                "timestamp": environment.timestamp.isoformat(),
            })

        return alerts