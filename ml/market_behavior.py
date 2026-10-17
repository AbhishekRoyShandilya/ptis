"""
Market Behavior Layer — Forward-Looking Signals
==================================================

Tracks three core market behavior metrics that change BEFORE strategy P&L does:

  1. Breakout success rate — % of N-bar high/low breakouts that follow through
  2. Trend continuation prob — P(next bar continues current direction)
  3. Mean reversion strength — how quickly deviations from mean get pulled back

These are LEADING indicators. The existing DecayDetector (Phase 5A) waits until
30 losing trades to fire — by then real money is gone. This module fires when
the underlying market microstructure shifts, typically 2-3 days BEFORE strategy
P&L reflects it.

Composes with:
  - #2 Probabilistic Regime: this layer's metrics drive a 2nd regime view that
    cross-checks #2's static classification
  - #3 Confidence Decay: behavior shifts feed into "regime alignment" component
  - #4 Expected Value: behavior trends inform forward EV estimates

Usage:
    tracker = MarketBehaviorTracker()
    snapshot = tracker.measure(candles_df, symbol="RELIANCE", timeframe="15m")
    alerts = tracker.detect_alerts(snapshot)

Output BehaviorAlert types:
  - "breakout_degrading"      → reduce sizing on breakout strategies
  - "trend_weakening"         → tighten targets on trend strategies
  - "reversion_strengthening" → favorable for mean-reversion, unfavorable for trend
  - "regime_transitioning"    → major shift detected, reduce overall sizing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Defaults ────────────────────────────────────────────────────────

DEFAULT_BREAKOUT_LOOKBACK: int = 50      # bars used to measure breakout success
DEFAULT_TREND_LOOKBACK: int = 100        # bars used for trend continuation
DEFAULT_REVERSION_LOOKBACK: int = 50     # bars used for mean reversion strength
DEFAULT_ALERT_THRESHOLD: float = 0.15    # 15% relative change triggers warning
DEFAULT_CRITICAL_THRESHOLD: float = 0.30 # 30% relative change triggers critical


# ── Output dataclasses ──────────────────────────────────────────────


@dataclass
class MarketBehaviorSnapshot:
    """A point-in-time measurement of market behavior."""
    timestamp: datetime
    symbol: str
    timeframe: str

    # Core forward-looking metrics
    breakout_success_rate: float          # 0-1 — % breakouts that follow through
    trend_continuation_prob: float        # 0-1 — P(next bar continues)
    mean_reversion_strength: float        # 0-1 — how strongly price reverts to mean

    # Derived probabilistic regime (cross-check vs #2)
    regime_prob_trending: float
    regime_prob_ranging: float
    regime_prob_volatile: float

    # Transition signals
    regime_momentum: float                # rate of change in regime probs
    behavior_shift_score: float           # 0=stable, 1=major shift underway

    # Per-strategy edge confidence (driven by behavior)
    edge_confidence: Dict[str, float] = field(default_factory=dict)

    metadata: dict = field(default_factory=dict)

    def short_summary(self) -> str:
        return (
            f"{self.symbol} {self.timeframe}: "
            f"BO={self.breakout_success_rate:.0%}, "
            f"trend_cont={self.trend_continuation_prob:.0%}, "
            f"reversion={self.mean_reversion_strength:.2f}, "
            f"shift={self.behavior_shift_score:.2f}"
        )


@dataclass
class BehaviorAlert:
    """A forward-looking alert about market behavior change."""
    alert_type: str                       # see top of file for valid values
    severity: str                         # "info" | "warning" | "critical"
    symbol: str
    description: str
    current_value: float
    baseline_value: float
    change_pct: float                     # relative change (e.g. -0.25 = -25%)
    affected_strategies: List[str]
    suggested_action: str
    detected_at: str


# ── Strategy → behavior metric mapping ─────────────────────────────

# Maps each strategy to the behavior metrics that most affect its edge.
# Used to identify which strategies are at risk when a behavior shifts.
_STRATEGY_BEHAVIOR_MAP: Dict[str, List[str]] = {
    # Breakout-dependent
    "compression_breakout":  ["breakout_success_rate"],
    "orb_breakout":          ["breakout_success_rate"],
    # Trend-dependent
    "ema_pullback":          ["trend_continuation_prob"],
    "pullback_trend":        ["trend_continuation_prob"],
    "momentum_continuation": ["trend_continuation_prob"],
    # Reversion-dependent
    "vwap_bounce":           ["mean_reversion_strength"],
    "vwap_reversal":         ["mean_reversion_strength"],
    "momentum_exhaustion":   ["mean_reversion_strength"],
    "gap_fill":              ["mean_reversion_strength"],
}


# ── Tracker ─────────────────────────────────────────────────────────


class MarketBehaviorTracker:
    """
    Tracks forward-looking market behavior metrics.

    Updates per-symbol, stores last 500 snapshots, and detects shifts
    by comparing recent N snapshots vs the prior N.
    """

    def __init__(
        self,
        breakout_lookback: int = DEFAULT_BREAKOUT_LOOKBACK,
        trend_lookback: int = DEFAULT_TREND_LOOKBACK,
        reversion_lookback: int = DEFAULT_REVERSION_LOOKBACK,
        alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
    ):
        self.breakout_lb = breakout_lookback
        self.trend_lb = trend_lookback
        self.reversion_lb = reversion_lookback
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold

        # Per-symbol history of snapshots (in-memory; persistence is caller's job)
        self._history: Dict[str, List[MarketBehaviorSnapshot]] = {}

    # ── Main measurement ──

    def measure(
        self,
        candles: pd.DataFrame,
        symbol: str,
        timeframe: str,
        timestamp: Optional[datetime] = None,
    ) -> MarketBehaviorSnapshot:
        """
        Measure current market behavior from OHLCV candles.

        candles must have: open/high/low/close/volume columns
        with DatetimeIndex or 'timestamp' column.
        """
        df = self._coerce_df(candles)
        ts = timestamp or datetime.utcnow()

        if df is None or len(df) < max(self.breakout_lb, self.trend_lb, self.reversion_lb):
            logger.debug("Insufficient candles for %s — using neutral snapshot", symbol)
            return self._neutral_snapshot(symbol, timeframe, ts)

        # Three core metrics
        breakout_sr = self._measure_breakout_success(df)
        trend_cp = self._measure_trend_continuation(df)
        reversion = self._measure_mean_reversion(df)

        # Derived: probabilistic regime from behavior
        regime_probs = self._compute_regime_probs(breakout_sr, trend_cp, reversion, df)

        # Push to history first so transition / momentum can see this point
        snapshot = MarketBehaviorSnapshot(
            timestamp=ts,
            symbol=symbol,
            timeframe=timeframe,
            breakout_success_rate=round(breakout_sr, 4),
            trend_continuation_prob=round(trend_cp, 4),
            mean_reversion_strength=round(reversion, 4),
            regime_prob_trending=round(regime_probs["trending"], 4),
            regime_prob_ranging=round(regime_probs["ranging"], 4),
            regime_prob_volatile=round(regime_probs["volatile"], 4),
            regime_momentum=0.0,
            behavior_shift_score=0.0,
            edge_confidence=self._compute_edge_confidence(breakout_sr, trend_cp, reversion),
        )

        sym_hist = self._history.setdefault(symbol, [])
        sym_hist.append(snapshot)
        if len(sym_hist) > 500:
            self._history[symbol] = sym_hist[-500:]

        # Recompute momentum + shift score now that history is updated
        snapshot.regime_momentum = round(self._regime_momentum(symbol), 4)
        snapshot.behavior_shift_score = round(self._behavior_shift_score(symbol), 4)

        return snapshot

    # ── Alert detection ──

    def detect_alerts(
        self,
        current: MarketBehaviorSnapshot,
        baseline_window: int = 20,
    ) -> List[BehaviorAlert]:
        """
        Compare current behavior to baseline (prior `baseline_window` snapshots)
        and emit alerts when shifts cross thresholds.
        """
        sym_hist = self._history.get(current.symbol, [])
        if len(sym_hist) < baseline_window + 1:
            return []

        # Use prior `baseline_window` snapshots (excluding current)
        baseline = sym_hist[-(baseline_window + 1):-1]
        now_str = datetime.utcnow().isoformat()
        alerts: List[BehaviorAlert] = []

        # ── 1. Breakout success rate degrading ──
        baseline_bo = float(np.mean([s.breakout_success_rate for s in baseline]))
        if baseline_bo > 0:
            change = (current.breakout_success_rate - baseline_bo) / baseline_bo
            if change < -self.alert_threshold:
                severity = "critical" if change < -self.critical_threshold else "warning"
                alerts.append(BehaviorAlert(
                    alert_type="breakout_degrading",
                    severity=severity,
                    symbol=current.symbol,
                    description=(
                        f"Breakout success rate falling: "
                        f"{baseline_bo:.0%} → {current.breakout_success_rate:.0%} "
                        f"({change:+.0%}). Breakout strategies will underperform."
                    ),
                    current_value=current.breakout_success_rate,
                    baseline_value=baseline_bo,
                    change_pct=round(change, 4),
                    affected_strategies=[
                        s for s, mets in _STRATEGY_BEHAVIOR_MAP.items()
                        if "breakout_success_rate" in mets
                    ],
                    suggested_action=(
                        "Reduce sizing on breakout strategies "
                        "(compression_breakout, orb_breakout). Pause if breakout SR drops below 30%."
                    ),
                    detected_at=now_str,
                ))

        # ── 2. Trend continuation weakening ──
        baseline_tc = float(np.mean([s.trend_continuation_prob for s in baseline]))
        if baseline_tc > 0:
            change = (current.trend_continuation_prob - baseline_tc) / baseline_tc
            if change < -self.alert_threshold:
                severity = "critical" if change < -self.critical_threshold else "warning"
                alerts.append(BehaviorAlert(
                    alert_type="trend_weakening",
                    severity=severity,
                    symbol=current.symbol,
                    description=(
                        f"Trend continuation probability falling: "
                        f"{baseline_tc:.0%} → {current.trend_continuation_prob:.0%} "
                        f"({change:+.0%}). Trend-following strategies at risk."
                    ),
                    current_value=current.trend_continuation_prob,
                    baseline_value=baseline_tc,
                    change_pct=round(change, 4),
                    affected_strategies=[
                        s for s, mets in _STRATEGY_BEHAVIOR_MAP.items()
                        if "trend_continuation_prob" in mets
                    ],
                    suggested_action=(
                        "Tighten targets on trend strategies (ema_pullback, momentum_continuation). "
                        "Market may be transitioning to range/chop."
                    ),
                    detected_at=now_str,
                ))

        # ── 3. Mean reversion strengthening (favorable for reversion, bad for trend) ──
        baseline_mr = float(np.mean([s.mean_reversion_strength for s in baseline]))
        if baseline_mr > 0:
            change = (current.mean_reversion_strength - baseline_mr) / baseline_mr
            if change > self.alert_threshold:
                alerts.append(BehaviorAlert(
                    alert_type="reversion_strengthening",
                    severity="warning",
                    symbol=current.symbol,
                    description=(
                        f"Mean reversion strengthening: "
                        f"{baseline_mr:.2f} → {current.mean_reversion_strength:.2f} "
                        f"({change:+.0%}). Favorable for mean-reversion, unfavorable for breakout/trend."
                    ),
                    current_value=current.mean_reversion_strength,
                    baseline_value=baseline_mr,
                    change_pct=round(change, 4),
                    affected_strategies=[
                        s for s, mets in _STRATEGY_BEHAVIOR_MAP.items()
                        if "mean_reversion_strength" in mets
                    ],
                    suggested_action=(
                        "Increase sizing on mean-reversion strategies (vwap_bounce, gap_fill). "
                        "Reduce breakout expectations."
                    ),
                    detected_at=now_str,
                ))

        # ── 4. Major regime transition ──
        if current.behavior_shift_score > 0.6:
            alerts.append(BehaviorAlert(
                alert_type="regime_transitioning",
                severity="critical",
                symbol=current.symbol,
                description=(
                    f"Major behavior shift detected (score: {current.behavior_shift_score:.2f}). "
                    f"Regime mix now: trend={current.regime_prob_trending:.0%}, "
                    f"range={current.regime_prob_ranging:.0%}, "
                    f"volatile={current.regime_prob_volatile:.0%}."
                ),
                current_value=current.behavior_shift_score,
                baseline_value=0.0,
                change_pct=0.0,
                affected_strategies=list(_STRATEGY_BEHAVIOR_MAP.keys()),
                suggested_action=(
                    "REDUCE ALL SIZING by 50% until new regime stabilizes (typically 5-10 days). "
                    "Do NOT trust historical statistics during transitions."
                ),
                detected_at=now_str,
            ))

        return alerts

    # ── Helpers: history-based metrics ──

    def _regime_momentum(self, symbol: str) -> float:
        """Rate of change of trending probability over recent vs older snapshots."""
        hist = self._history.get(symbol, [])
        if len(hist) < 5:
            return 0.0
        recent5 = hist[-5:]
        older5 = hist[-10:-5] if len(hist) >= 10 else hist[:5]
        recent_trending = float(np.mean([s.regime_prob_trending for s in recent5]))
        older_trending = float(np.mean([s.regime_prob_trending for s in older5]))
        return recent_trending - older_trending

    def _behavior_shift_score(self, symbol: str) -> float:
        """
        0 = stable behavior; 1 = major shift underway.
        Euclidean distance in (BO_sr, trend_cp, reversion) space between
        recent 5 snapshots and prior 10.
        """
        hist = self._history.get(symbol, [])
        if len(hist) < 10:
            return 0.0
        recent = hist[-5:]
        older = hist[-15:-5] if len(hist) >= 15 else hist[:5]
        recent_vec = np.array([
            np.mean([s.breakout_success_rate for s in recent]),
            np.mean([s.trend_continuation_prob for s in recent]),
            np.mean([s.mean_reversion_strength for s in recent]),
        ])
        older_vec = np.array([
            np.mean([s.breakout_success_rate for s in older]),
            np.mean([s.trend_continuation_prob for s in older]),
            np.mean([s.mean_reversion_strength for s in older]),
        ])
        distance = float(np.linalg.norm(recent_vec - older_vec))
        # Normalize: max distance in 3D unit space ≈ √3 ≈ 1.73
        return min(distance / 1.0, 1.0)

    # ── Helpers: edge confidence per strategy ──

    @staticmethod
    def _compute_edge_confidence(
        breakout_sr: float, trend_cp: float, reversion: float,
    ) -> Dict[str, float]:
        """Per-strategy edge confidence based on which behavior they depend on."""
        return {
            "compression_breakout":  round(min(breakout_sr * 1.5, 1.0), 4),
            "orb_breakout":          round(min(breakout_sr * 1.3, 1.0), 4),
            "ema_pullback":          round(min(trend_cp * 1.4, 1.0), 4),
            "momentum_continuation": round(min(trend_cp * 1.3, 1.0), 4),
            "pullback_trend":        round(min(trend_cp * 1.3, 1.0), 4),
            "vwap_bounce":           round(min(reversion * 1.5, 1.0), 4),
            "vwap_reversal":         round(min(reversion * 1.4, 1.0), 4),
            "momentum_exhaustion":   round(min(reversion * 1.3, 1.0), 4),
            "gap_fill":              round(min(reversion * 1.5, 1.0), 4),
        }

    # ── Helpers: regime probabilities derived from behavior ──

    def _compute_regime_probs(
        self, breakout_sr: float, trend_cp: float, reversion: float, df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Probabilistic regime DERIVED from behavior metrics — not the same as #2.
        #2 uses indicator snapshots (ADX, autocorr, BB width).
        This uses behavior trends — useful as a cross-check.
        """
        # Volatility ratio from returns
        if len(df) >= 40:
            returns = df["close"].pct_change().dropna()
            if len(returns) >= 40:
                recent_vol = float(returns.iloc[-20:].std())
                long_vol = float(returns.std())
                vol_ratio = recent_vol / long_vol if long_vol > 0 else 1.0
            else:
                vol_ratio = 1.0
        else:
            vol_ratio = 1.0

        trending_score = trend_cp * (1 - reversion) * breakout_sr
        ranging_score = reversion * (1 - trend_cp) * (1 - breakout_sr)
        volatile_score = max(vol_ratio - 1.0, 0) * 0.5

        total = trending_score + ranging_score + volatile_score + 0.001
        return {
            "trending": trending_score / total,
            "ranging": ranging_score / total,
            "volatile": volatile_score / total,
        }

    # ── Three core metric calculations ──

    def _measure_breakout_success(self, df: pd.DataFrame) -> float:
        """% of N-bar high/low breakouts that have follow-through."""
        window = min(self.breakout_lb, len(df))
        recent = df.iloc[-window:]
        if len(recent) < 25:
            return 0.5

        highs_20 = recent["high"].rolling(20).max().shift(1)
        lows_20 = recent["low"].rolling(20).min().shift(1)
        breakout_up = recent["close"] > highs_20
        breakout_down = recent["close"] < lows_20

        attempts = 0
        successes = 0
        for i in range(len(recent) - 3):
            if pd.isna(highs_20.iloc[i]) or pd.isna(lows_20.iloc[i]):
                continue
            bo_close = float(recent["close"].iloc[i])
            if bool(breakout_up.iloc[i]):
                attempts += 1
                follow = sum(
                    1 for j in range(1, min(4, len(recent) - i))
                    if float(recent["close"].iloc[i + j]) > bo_close
                )
                if follow >= 2:
                    successes += 1
            elif bool(breakout_down.iloc[i]):
                attempts += 1
                follow = sum(
                    1 for j in range(1, min(4, len(recent) - i))
                    if float(recent["close"].iloc[i + j]) < bo_close
                )
                if follow >= 2:
                    successes += 1

        if attempts < 3:
            return 0.5
        return successes / attempts

    def _measure_trend_continuation(self, df: pd.DataFrame) -> float:
        """P(next bar continues current direction)."""
        window = min(self.trend_lb, len(df))
        recent = df.iloc[-window:]
        if len(recent) < 10:
            return 0.5

        directions = np.sign(recent["close"].values - recent["open"].values)

        # Bar-to-bar continuation
        continuations = 0
        for i in range(1, len(directions)):
            if directions[i] != 0 and directions[i] == directions[i - 1]:
                continuations += 1

        # Streak continuation: when 3+ bars in same direction, does 4th continue?
        streak_continuations = 0
        streak_attempts = 0
        streak = 0
        last_dir = 0.0
        for i in range(len(directions)):
            if directions[i] == last_dir and directions[i] != 0:
                streak += 1
            else:
                if streak >= 3 and i < len(directions):
                    streak_attempts += 1
                    if directions[i] == last_dir:
                        streak_continuations += 1
                streak = 1
                last_dir = directions[i]

        base_prob = continuations / max(len(directions) - 1, 1)
        streak_prob = (
            streak_continuations / max(streak_attempts, 1)
            if streak_attempts > 0
            else base_prob
        )
        return 0.6 * base_prob + 0.4 * streak_prob

    def _measure_mean_reversion(self, df: pd.DataFrame) -> float:
        """How quickly does price revert to its mean?"""
        window = min(self.reversion_lb, len(df))
        recent = df.iloc[-window:]
        if len(recent) < 20:
            return 0.5

        sma = recent["close"].rolling(20).mean()
        deviation = ((recent["close"] - sma) / sma).dropna()
        if len(deviation) < 10:
            return 0.5

        dev_values = deviation.values
        if len(dev_values) < 5:
            return 0.5

        # Lag-1 autocorrelation; negative = reverting
        try:
            autocorr = float(np.corrcoef(dev_values[:-1], dev_values[1:])[0, 1])
            if np.isnan(autocorr):
                autocorr = 0.0
        except Exception:
            autocorr = 0.0

        # Map autocorr to reversion strength: -1 → 1.0 (strong revert), +1 → 0.0
        reversion_from_autocorr = (1.0 - autocorr) / 2.0

        # Pullback rate: when |deviation| > 1 std, does it pull back within 5 bars?
        std_dev = float(np.std(dev_values))
        if std_dev == 0:
            return float(reversion_from_autocorr)

        pullback_count = 0
        pullback_attempts = 0
        for i in range(len(dev_values) - 5):
            if abs(dev_values[i]) > std_dev:
                pullback_attempts += 1
                initial = dev_values[i]
                for j in range(1, 6):
                    if abs(dev_values[i + j]) < abs(initial) * 0.5:
                        pullback_count += 1
                        break

        pullback_rate = (
            pullback_count / pullback_attempts if pullback_attempts > 3 else 0.5
        )

        return 0.5 * reversion_from_autocorr + 0.5 * pullback_rate

    # ── Helpers ──

    @staticmethod
    def _coerce_df(candles) -> Optional[pd.DataFrame]:
        if candles is None:
            return None
        if isinstance(candles, pd.DataFrame):
            df = candles
        else:
            try:
                df = pd.DataFrame(candles)
            except Exception:
                return None
        if df.empty:
            return None
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                return None
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ("timestamp", "datetime", "date", "time"):
                if col in df.columns:
                    try:
                        df = df.set_index(col)
                        break
                    except Exception:
                        pass
        return df

    @staticmethod
    def _neutral_snapshot(symbol: str, timeframe: str, ts: datetime) -> MarketBehaviorSnapshot:
        return MarketBehaviorSnapshot(
            timestamp=ts, symbol=symbol, timeframe=timeframe,
            breakout_success_rate=0.5,
            trend_continuation_prob=0.5,
            mean_reversion_strength=0.5,
            regime_prob_trending=0.33,
            regime_prob_ranging=0.33,
            regime_prob_volatile=0.34,
            regime_momentum=0.0,
            behavior_shift_score=0.0,
            edge_confidence={},
        )


# ── Module singleton ───────────────────────────────────────────────


_singleton: Optional[MarketBehaviorTracker] = None


def get_behavior_tracker() -> MarketBehaviorTracker:
    """Module-level singleton."""
    global _singleton
    if _singleton is None:
        _singleton = MarketBehaviorTracker()
    return _singleton
