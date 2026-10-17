"""
Edge Emergence Detector
=======================

Catches the OTHER half of the strategy-rotation problem: a new edge is forming.
A strategy that was mediocre starts winning more often as its preferred regime
becomes more frequent in the market.

Algorithm:
  For each strategy, check two things in a recent window vs long-term baseline:
    1. Is its PREFERRED REGIME becoming more common in the market?
       (regime distribution shift)
    2. Is its WIN RATE rising in recent trades?

  If BOTH are rising → emergence finding (consider enabling / increasing sizing)

This is the natural counterpart to decay_detector. Together they tell the
full story: "your old edge is dying AND a new one is forming, here's the rotation."

Real-money safety:
  - Alerts only — no auto-enable, no auto-sizing increase
  - Trader reviews and decides
  - Approved findings still go through the AVT proof cycle before going live
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from research_agent.models import EmergenceFinding

logger = logging.getLogger(__name__)


# ── Default parameters ─────────────────────────────────────────────

DEFAULT_RECENT_DAYS: int = 30
DEFAULT_BASELINE_DAYS: int = 180          # 6 months
DEFAULT_MIN_RECENT_TRADES: int = 15
DEFAULT_MIN_BASELINE_TRADES: int = 60

# Thresholds for declaring emergence
DEFAULT_REGIME_PCT_RISE: float = 0.10     # +10pp regime frequency = meaningful shift
DEFAULT_WIN_RATE_RISE: float = 0.08       # +8pp WR = meaningful improvement


# ── Data interfaces ────────────────────────────────────────────────

# Caller provides loaders — this module is data-source agnostic.
TradeRecordLoader = Callable[[str, datetime, datetime], List[dict]]
RegimeFreqLoader = Callable[[str, datetime, datetime], float]
StrategyPreferredRegimeLoader = Callable[[str], List[str]]


# ── Emergence detector ────────────────────────────────────────────

class EmergenceDetector:
    """
    Detects strategies whose edge is emerging due to favorable regime shift.

    Usage:
        detector = EmergenceDetector(
            trade_loader=fn(strategy_id, start_dt, end_dt) -> [{r_multiple, ...}, ...],
            regime_freq_loader=fn(regime_name, start_dt, end_dt) -> 0..1,
            strategy_regime_loader=fn(strategy_id) -> ["trending_up", "breakout"],
        )
        finding = detector.detect("compression_breakout")
    """

    def __init__(
        self,
        trade_loader: TradeRecordLoader,
        regime_freq_loader: RegimeFreqLoader,
        strategy_regime_loader: StrategyPreferredRegimeLoader,
        recent_days: int = DEFAULT_RECENT_DAYS,
        baseline_days: int = DEFAULT_BASELINE_DAYS,
        min_recent_trades: int = DEFAULT_MIN_RECENT_TRADES,
        min_baseline_trades: int = DEFAULT_MIN_BASELINE_TRADES,
        regime_pct_rise: float = DEFAULT_REGIME_PCT_RISE,
        win_rate_rise: float = DEFAULT_WIN_RATE_RISE,
    ):
        self._trade_loader = trade_loader
        self._regime_freq = regime_freq_loader
        self._strat_regimes = strategy_regime_loader
        self.recent_days = recent_days
        self.baseline_days = baseline_days
        self.min_recent_trades = min_recent_trades
        self.min_baseline_trades = min_baseline_trades
        self.regime_pct_rise_threshold = regime_pct_rise
        self.win_rate_rise_threshold = win_rate_rise

    def detect(self, strategy_id: str, now: Optional[datetime] = None) -> Optional[EmergenceFinding]:
        """
        Check if `strategy_id` shows BOTH:
          - rising preferred-regime frequency
          - rising win rate
        Returns EmergenceFinding only when both conditions met.
        """
        now = now or datetime.now()

        # Date ranges
        recent_start = _shift_days(now, -self.recent_days)
        baseline_end = recent_start
        baseline_start = _shift_days(now, -(self.baseline_days + self.recent_days))

        # ── 1. Trade-level WR comparison ──
        try:
            recent_trades = self._trade_loader(strategy_id, recent_start, now) or []
            baseline_trades = self._trade_loader(strategy_id, baseline_start, baseline_end) or []
        except Exception as e:
            logger.warning("EmergenceDetector trade load failed for %s: %s", strategy_id, e)
            return None

        if len(recent_trades) < self.min_recent_trades:
            return None
        if len(baseline_trades) < self.min_baseline_trades:
            return None

        recent_wr = _win_rate(recent_trades)
        baseline_wr = _win_rate(baseline_trades)
        wr_change = recent_wr - baseline_wr

        if wr_change < self.win_rate_rise_threshold:
            return None  # not an emergence

        # ── 2. Regime distribution shift ──
        try:
            preferred_regimes = self._strat_regimes(strategy_id) or []
        except Exception as e:
            logger.debug("Strategy regime load failed for %s: %s", strategy_id, e)
            preferred_regimes = []

        if not preferred_regimes:
            return None

        # For each preferred regime, find the one with the largest positive shift
        best_regime: Optional[str] = None
        best_baseline_pct: float = 0.0
        best_recent_pct: float = 0.0
        best_change: float = 0.0
        for regime in preferred_regimes:
            try:
                pct_recent = self._regime_freq(regime, recent_start, now)
                pct_baseline = self._regime_freq(regime, baseline_start, baseline_end)
            except Exception as e:
                logger.debug("Regime freq load failed for %s/%s: %s", strategy_id, regime, e)
                continue
            change = pct_recent - pct_baseline
            if change > best_change:
                best_change = change
                best_regime = regime
                best_baseline_pct = pct_baseline
                best_recent_pct = pct_recent

        if best_regime is None or best_change < self.regime_pct_rise_threshold:
            return None  # regime not actually rising

        # ── 3. Co-emergence — both conditions met ──
        action = (
            f"Consider enabling {strategy_id} or increasing sizing — "
            f"its preferred regime ({best_regime}) is becoming more common AND win rate is rising."
        )
        rationale = (
            f"Recent {self.recent_days}d: {len(recent_trades)} trades, "
            f"WR {recent_wr:.0%} (was {baseline_wr:.0%}, +{wr_change:.0%}). "
            f"Regime '{best_regime}' freq {best_baseline_pct:.0%} → {best_recent_pct:.0%} "
            f"(+{best_change:.0%}). Co-emergence: edge appears to be strengthening."
        )

        return EmergenceFinding(
            strategy_id=strategy_id,
            detected_at=now,
            preferred_regime=best_regime,
            regime_pct_baseline=best_baseline_pct,
            regime_pct_recent=best_recent_pct,
            regime_pct_change=best_change,
            win_rate_baseline=baseline_wr,
            win_rate_recent=recent_wr,
            win_rate_change=wr_change,
            baseline_n=len(baseline_trades),
            recent_n=len(recent_trades),
            suggested_action=action,
            rationale=rationale,
        )

    def detect_all(self, strategy_ids: List[str]) -> List[EmergenceFinding]:
        findings: List[EmergenceFinding] = []
        for sid in strategy_ids:
            try:
                f = self.detect(sid)
                if f:
                    findings.append(f)
                    logger.info("Emergence finding: %s", f.short_summary())
            except Exception as e:
                logger.warning("EmergenceDetector failed for %s: %s", sid, e)
        return findings


# ── Helpers ─────────────────────────────────────────────────────────

def _win_rate(trades: List[dict]) -> float:
    """Compute win rate from trade records (each must have 'r_multiple')."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if float(t.get("r_multiple", 0) or 0) > 0)
    return wins / len(trades)


def _shift_days(dt: datetime, delta_days: int) -> datetime:
    """Shift a datetime by N days (handles month/year rollover via timedelta)."""
    from datetime import timedelta
    return dt + timedelta(days=delta_days)
