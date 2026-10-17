# coplanner/behavioral_guard.py
"""
Behavioral Planner Guard — Planning-Stage Behavioral Checks.

Runs BEFORE user decides to trade, not after.
Uses historical patterns of user behavior to warn proactively.

Existing live_support.py runs at TRADE ENTRY time.
This runs at PLANNING stage — upstream, preventive.

Checks:
  1. Time of day loss pattern
  2. Day of week loss pattern
  3. Volume context (low-vol breakout losses)
  4. Winning/losing streak behavior
  5. Regime mismatch history
  6. Daily trade count (overtrading)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from coplanner.db_interface import CoPlannerDB
from coplanner.models import BehavioralCheckResult, TradeRecord

logger = logging.getLogger(__name__)


class BehavioralPlannerGuard:
    """
    Runs all behavioral checks at planning stage.

    Usage:
        guard = BehavioralPlannerGuard(db, trader_id)
        result = guard.check(current_hour=10, current_volume=...)
        if not result.proceed:
            # recommend not trading
    """

    # Thresholds
    MAX_DAILY_TRADES = 5
    LOSS_PATTERN_MIN = 5           # minimum losses to flag a pattern
    STREAK_OVERCONFIDENCE = 4      # wins in a row → flag
    STREAK_REVENGE_RISK = 3        # losses in a row → flag
    BEHAVIORAL_BLOCK_THRESHOLD = 5 # risk_score >= this → don't trade

    def __init__(
        self,
        db: CoPlannerDB,
        trader_id: Optional[str] = None,
    ):
        self._db = db
        self._trader_id = trader_id
        self._loss_patterns: Optional[Dict[str, Any]] = None

    def check(
        self,
        current_hour: Optional[int] = None,
        current_volume: float = 0,
        avg_volume: float = 0,
        current_regime: str = "unknown",
        proposed_regime_mismatch: bool = False,
    ) -> BehavioralCheckResult:
        """Run ALL behavioral checks and return combined result."""
        # Build loss patterns lazily
        if self._loss_patterns is None:
            self._loss_patterns = self._build_loss_patterns()

        warnings: List[Dict[str, Any]] = []
        risk_score = 0

        # Check 1: Time of day
        w, s = self._check_time_of_day(current_hour)
        warnings.extend(w)
        risk_score += s

        # Check 2: Day of week
        w, s = self._check_day_of_week()
        warnings.extend(w)
        risk_score += s

        # Check 3: Volume context
        w, s = self._check_volume_context(current_volume, avg_volume)
        warnings.extend(w)
        risk_score += s

        # Check 4: Streak behavior
        w, s = self._check_streaks()
        warnings.extend(w)
        risk_score += s

        # Check 5: Regime mismatch
        if proposed_regime_mismatch:
            warnings.append({
                "type": "REGIME_MISMATCH",
                "severity": "HIGH",
                "message": (
                    "Strategy doesn't match current regime. "
                    "This is historically your #1 source of losses."
                ),
            })
            risk_score += 2

        # Check 6: Daily trade count
        w, s = self._check_overtrading()
        warnings.extend(w)
        risk_score += s

        # Verdict
        proceed = risk_score < self.BEHAVIORAL_BLOCK_THRESHOLD
        verdict = self._verdict(risk_score, warnings)

        return BehavioralCheckResult(
            warnings=warnings,
            risk_score=risk_score,
            max_risk_score=10,
            verdict=verdict,
            proceed=proceed,
        )

    # ─────────────────────────────────────────────
    # LOSS PATTERN BUILDER
    # ─────────────────────────────────────────────

    def _build_loss_patterns(self) -> Dict[str, Any]:
        """Analyze user's losing trades to find behavioral patterns."""
        losses = self._db.get_losing_trades(self._trader_id)
        patterns: Dict[str, Any] = {
            "time_of_day": {},
            "day_of_week": {},
            "low_volume_losses": 0,
            "total_losses": len(losses),
        }

        for trade in losses:
            if trade.entry_time:
                hour = trade.entry_time.hour
                patterns["time_of_day"][hour] = (
                    patterns["time_of_day"].get(hour, 0) + 1
                )
                day = trade.entry_time.strftime("%A")
                patterns["day_of_week"][day] = (
                    patterns["day_of_week"].get(day, 0) + 1
                )

            if (
                trade.volume_at_entry > 0
                and trade.volume_at_entry < 0.6  # placeholder ratio
            ):
                patterns["low_volume_losses"] += 1

        return patterns

    # ─────────────────────────────────────────────
    # INDIVIDUAL CHECKS
    # ─────────────────────────────────────────────

    def _check_time_of_day(
        self, current_hour: Optional[int],
    ) -> tuple[List[Dict[str, Any]], int]:
        if current_hour is None:
            return [], 0

        p = self._loss_patterns or {}
        tod = p.get("time_of_day", {})
        loss_count = tod.get(current_hour, 0)

        if loss_count >= self.LOSS_PATTERN_MIN:
            return [
                {
                    "type": "TIME_PATTERN",
                    "severity": "HIGH",
                    "message": (
                        f"You have {loss_count} losses in trades taken "
                        f"at {current_hour}:00 hour. Historically weak time."
                    ),
                }
            ], 2
        return [], 0

    def _check_day_of_week(self) -> tuple[List[Dict[str, Any]], int]:
        p = self._loss_patterns or {}
        dow = p.get("day_of_week", {})
        today = datetime.now(timezone.utc).strftime("%A")
        day_losses = dow.get(today, 0)

        if day_losses >= self.LOSS_PATTERN_MIN:
            return [
                {
                    "type": "DAY_PATTERN",
                    "severity": "MEDIUM",
                    "message": (
                        f"{today}s have been historically weak "
                        f"({day_losses} losses)."
                    ),
                }
            ], 1
        return [], 0

    def _check_volume_context(
        self, current_vol: float, avg_vol: float,
    ) -> tuple[List[Dict[str, Any]], int]:
        if avg_vol <= 0 or current_vol <= 0:
            return [], 0

        p = self._loss_patterns or {}
        low_vol_losses = p.get("low_volume_losses", 0)

        if current_vol < avg_vol * 0.6 and low_vol_losses >= 3:
            return [
                {
                    "type": "LOW_VOLUME",
                    "severity": "HIGH",
                    "message": (
                        f"Low volume detected. You've lost {low_vol_losses} times "
                        f"in similar low-volume situations."
                    ),
                }
            ], 2
        return [], 0

    def _check_streaks(self) -> tuple[List[Dict[str, Any]], int]:
        recent = self._db.get_recent_trades(n=5, trader_id=self._trader_id)
        if not recent:
            return [], 0

        warnings: List[Dict[str, Any]] = []
        score = 0

        wins = sum(1 for t in recent if t.outcome == "win")
        losses = sum(1 for t in recent if t.outcome == "loss")

        if wins >= self.STREAK_OVERCONFIDENCE:
            warnings.append({
                "type": "OVERCONFIDENCE_RISK",
                "severity": "MEDIUM",
                "message": (
                    f"On a {wins}-win streak. Historical pattern: "
                    f"tendency to oversize or take low-conviction trades."
                ),
            })
            score += 1

        if losses >= self.STREAK_REVENGE_RISK:
            warnings.append({
                "type": "REVENGE_TRADING_RISK",
                "severity": "HIGH",
                "message": (
                    f"{losses}+ consecutive losses. High revenge trading risk. "
                    f"Consider stepping back."
                ),
            })
            score += 3

        return warnings, score

    def _check_overtrading(self) -> tuple[List[Dict[str, Any]], int]:
        count = self._db.get_todays_trade_count(self._trader_id)
        if count >= self.MAX_DAILY_TRADES:
            return [
                {
                    "type": "OVERTRADING",
                    "severity": "HIGH",
                    "message": (
                        f"Already {count} trades today (max: {self.MAX_DAILY_TRADES}). "
                        f"Overtrading destroys edge."
                    ),
                }
            ], 3
        if count >= self.MAX_DAILY_TRADES - 1:
            return [
                {
                    "type": "OVERTRADING_WARNING",
                    "severity": "MEDIUM",
                    "message": f"{count} trades today — approaching daily limit.",
                }
            ], 1
        return [], 0

    # ─────────────────────────────────────────────
    # VERDICT
    # ─────────────────────────────────────────────

    @staticmethod
    def _verdict(score: int, warnings: List[Dict[str, Any]]) -> str:
        if score == 0:
            return "No behavioral red flags. Clear to plan."
        if score <= 2:
            return "Minor concerns. Proceed with awareness."
        if score <= 4:
            return (
                "Multiple behavioral flags. Reduce size or "
                "pick only highest conviction setup."
            )
        return "HIGH behavioral risk. Strongly recommend NOT trading right now."
