# avt/observer.py
"""
AVT Observer — Behavioral Observation Layer
=============================================

Watches the USER's real trades for behavioral/discipline issues.
These are DIRECT observations — no AVT proof needed.

Examples:
  - "You widened your SL 3 times today, cost you 1.2R"
  - "3 consecutive losses — possible revenge trading"
  - "You followed your plan on 4/5 trades today. Good discipline."

NOT for strategy improvements. Strategy improvements are discovered
by the improvement_lab analyzing AVT's OWN clean trades, applied to
AVT, proven by AVT trading with them, then presented to the user.
This module has NOTHING to do with that pipeline.

Scope:
  - Overtrading detection
  - Revenge trade detection
  - SL widening
  - Position oversizing
  - Early exit (left money on table)
  - Plan adherence (session end summary)
  - Positive reinforcement (good discipline)
  - AVT proof lifecycle notifications (proven/started/reverted)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  OBSERVATION TYPES
# ══════════════════════════════════════════════════════════════════

class ObservationType:
    # Behavioral (direct feedback, no proof needed)
    PLAN_DEVIATION = "plan_deviation"
    DISCIPLINE_DROP = "discipline_drop"
    OVERTRADING = "overtrading"
    REVENGE_TRADE = "revenge_trade"
    SL_WIDENED = "sl_widened"
    EARLY_EXIT = "early_exit"
    POSITION_OVERSIZE = "position_oversize"
    EMOTION_FLAG = "emotion_flag"

    # Positive
    GOOD_DISCIPLINE = "good_discipline"

    # AVT lifecycle events (informational)
    IMPROVEMENT_PROVEN = "improvement_proven"
    IMPROVEMENT_STARTED = "improvement_started"
    IMPROVEMENT_REVERTED = "improvement_reverted"


class Observation:
    """Single observation from the observer."""

    def __init__(
        self,
        obs_type: str,
        title: str,
        description: str,
        severity: str = "info",    # "info" | "warning" | "critical" | "positive"
        data: Optional[Dict[str, Any]] = None,
        strategy_id: Optional[str] = None,
        trade_ids: Optional[List[str]] = None,
    ):
        self.id = str(uuid4())
        self.type = obs_type
        self.title = title
        self.description = description
        self.severity = severity
        self.data = data or {}
        self.strategy_id = strategy_id
        self.trade_ids = trade_ids or []
        self.timestamp = datetime.now(timezone.utc)
        self.acknowledged = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "data": self.data,
            "strategy_id": self.strategy_id,
            "trade_ids": self.trade_ids,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


# ══════════════════════════════════════════════════════════════════
#  BEHAVIORAL CHECKS
# ══════════════════════════════════════════════════════════════════

class BehavioralObserver:
    """
    Watches user's real trades for behavioral issues.
    These observations go DIRECTLY to the user — no AVT proof needed.
    """

    def __init__(self):
        self._trades_today: List[Dict[str, Any]] = []
        self._last_reset_date: Optional[str] = None
        self._consecutive_losses: int = 0

    def observe_trade(self, trade: Dict[str, Any]) -> List[Observation]:
        """Check a user trade for behavioral issues."""
        observations: List[Observation] = []
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._last_reset_date != today:
            self._trades_today = []
            self._last_reset_date = today

        self._trades_today.append(trade)
        strategy_id = trade.get("strategy_id") or trade.get("declared_strategy_id")

        # ── Overtrading ──
        if len(self._trades_today) > 8:
            observations.append(Observation(
                obs_type=ObservationType.OVERTRADING,
                title="Overtrading detected",
                description=(
                    f"You've taken {len(self._trades_today)} trades today. "
                    f"Your typical max is 6-8. More trades usually means "
                    f"lower quality setups."
                ),
                severity="warning",
                strategy_id=strategy_id,
                data={"trades_today": len(self._trades_today)},
            ))

        # ── Revenge trade ──
        r = trade.get("r_multiple", 0)
        if r and r < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if self._consecutive_losses >= 3:
            observations.append(Observation(
                obs_type=ObservationType.REVENGE_TRADE,
                title="Possible revenge trading",
                description=(
                    f"{self._consecutive_losses} consecutive losses. "
                    f"Take a break. The market will be here tomorrow."
                ),
                severity="critical",
                strategy_id=strategy_id,
                data={"consecutive_losses": self._consecutive_losses},
            ))

        # ── SL widened ──
        overrides = trade.get("overrides", [])
        for ov in overrides:
            ov_type = ov.get("override_type", "") if isinstance(ov, dict) else getattr(ov, "override_type", "")
            if str(ov_type) in ("stop_moved", "stop_too_wide", "sl_widened"):
                observations.append(Observation(
                    obs_type=ObservationType.SL_WIDENED,
                    title="Stop loss widened",
                    description=(
                        "You widened your stop loss during this trade. "
                        "This increases risk beyond your original plan "
                        "and is one of the most costly behavioral patterns."
                    ),
                    severity="warning",
                    strategy_id=strategy_id,
                    trade_ids=[str(trade.get("id", ""))],
                ))
                break

        # ── Position oversize ──
        risk_pct = trade.get("risk_pct", 0)
        if risk_pct and risk_pct > 2.0:
            observations.append(Observation(
                obs_type=ObservationType.POSITION_OVERSIZE,
                title="Position too large",
                description=(
                    f"Risk per trade: {risk_pct:.1f}% — exceeds your 2% rule. "
                    f"Oversized positions lead to emotional decision-making."
                ),
                severity="warning",
                strategy_id=strategy_id,
                trade_ids=[str(trade.get("id", ""))],
                data={"risk_pct": risk_pct},
            ))

        # ── Early exit (left money on table) ──
        max_fav_r = trade.get("max_favorable_r", 0) or 0
        if r and r > 0 and max_fav_r >= 2.0 and r < max_fav_r * 0.4:
            observations.append(Observation(
                obs_type=ObservationType.EARLY_EXIT,
                title="Exited too early",
                description=(
                    f"Trade reached {max_fav_r:.1f}R but you exited at {r:.1f}R. "
                    f"You captured only {(r / max_fav_r * 100):.0f}% of the move."
                ),
                severity="info",
                strategy_id=strategy_id,
                trade_ids=[str(trade.get("id", ""))],
                data={"max_favorable_r": max_fav_r, "actual_r": r},
            ))

        return observations

    def observe_session_end(self, trades: List[Dict[str, Any]]) -> List[Observation]:
        """End-of-day session summary observations."""
        observations: List[Observation] = []
        if not trades:
            return observations

        total = len(trades)
        plan_followed = sum(
            1 for t in trades
            if not t.get("overrides") or len(t.get("overrides", [])) == 0
        )
        discipline_pct = plan_followed / total if total > 0 else 1.0

        if discipline_pct >= 0.8:
            observations.append(Observation(
                obs_type=ObservationType.GOOD_DISCIPLINE,
                title="Good discipline today",
                description=(
                    f"You followed your plan on {plan_followed}/{total} trades "
                    f"({discipline_pct:.0%}). Keep it up."
                ),
                severity="positive",
                data={"discipline_pct": discipline_pct, "total_trades": total},
            ))
        elif discipline_pct < 0.5:
            observations.append(Observation(
                obs_type=ObservationType.PLAN_DEVIATION,
                title="Plan adherence was low today",
                description=(
                    f"Only {plan_followed}/{total} trades followed your strategy rules. "
                    f"Deviations are the #1 source of avoidable losses."
                ),
                severity="warning",
                data={"discipline_pct": discipline_pct, "total_trades": total},
            ))

        return observations


# ══════════════════════════════════════════════════════════════════
#  AVT OBSERVER (unified interface)
# ══════════════════════════════════════════════════════════════════

class AVTObserver:
    """
    Unified observation interface for AVT.

    Two responsibilities:
    1. Watch user trades for behavioral issues (BehavioralObserver)
    2. Emit AVT lifecycle events (improvement proven/started/reverted)

    Does NOT analyze user trades for strategy improvements.
    That's the improvement_lab's job, and it only runs on AVT's data.
    """

    def __init__(self):
        self._behavioral = BehavioralObserver()
        self._observation_log: List[Observation] = []
        self._callbacks: List[Callable[[Observation], None]] = []
        self._max_log = 500

    def on_observation(self, callback: Callable[[Observation], None]) -> None:
        """Register a callback for new observations."""
        self._callbacks.append(callback)

    def _emit(self, obs: Observation) -> None:
        """Log and emit an observation."""
        self._observation_log.append(obs)
        if len(self._observation_log) > self._max_log:
            self._observation_log = self._observation_log[-self._max_log:]

        for cb in self._callbacks:
            try:
                cb(obs)
            except Exception as e:
                logger.debug("Observer callback error: %s", e)

    # ── User trade events ─────────────────────────────────────

    def on_user_trade_closed(self, trade: Dict[str, Any]) -> List[Observation]:
        """Called when a user trade closes. Returns behavioral observations."""
        observations = self._behavioral.observe_trade(trade)
        for obs in observations:
            self._emit(obs)
        return observations

    def on_session_end(self, trades: List[Dict[str, Any]]) -> List[Observation]:
        """Called at end of trading day."""
        observations = self._behavioral.observe_session_end(trades)
        for obs in observations:
            self._emit(obs)
        return observations

    # ── AVT lifecycle events ──────────────────────────────────

    def on_improvement_proven(
        self, strategy_id: str, title: str, delta_r: float,
        trades_count: int, evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called when AVT proves an improvement through virtual trading."""
        obs = Observation(
            obs_type=ObservationType.IMPROVEMENT_PROVEN,
            title=f"Proven: {title}",
            description=(
                f"I traded with this improvement for {trades_count} virtual trades. "
                f"Result: +{delta_r:.3f}R per trade. This is real evidence from "
                f"my own trading, not just statistics."
            ),
            severity="positive",
            strategy_id=strategy_id,
            data={"delta_r": delta_r, "trades_count": trades_count, **(evidence or {})},
        )
        self._emit(obs)

    def on_improvement_started(
        self, strategy_id: str, title: str, description: str,
    ) -> None:
        """Called when AVT starts testing a new hypothesis."""
        obs = Observation(
            obs_type=ObservationType.IMPROVEMENT_STARTED,
            title=f"Testing: {title}",
            description=(
                f"I found a potential improvement and applied it to my own trading. "
                f"I'll trade with it for 50+ trades to see if it actually works. "
                f"You'll only hear about it if it proves out."
            ),
            severity="info",
            strategy_id=strategy_id,
        )
        self._emit(obs)

    def on_improvement_reverted(
        self, strategy_id: str, title: str, reason: str,
    ) -> None:
        """Called when AVT reverts a failed improvement."""
        obs = Observation(
            obs_type=ObservationType.IMPROVEMENT_REVERTED,
            title=f"Reverted: {title}",
            description=f"Didn't work out: {reason}",
            severity="info",
            strategy_id=strategy_id,
        )
        self._emit(obs)

    # ── Query ─────────────────────────────────────────────────

    def get_recent_observations(
        self,
        limit: int = 50,
        obs_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent observations, newest first."""
        filtered = self._observation_log
        if obs_type:
            filtered = [o for o in filtered if o.type == obs_type]
        return [o.to_dict() for o in reversed(filtered[-limit:])]

    def get_today_summary(self) -> Dict[str, Any]:
        """Summary of today's observations."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_obs = [
            o for o in self._observation_log
            if o.timestamp.strftime("%Y-%m-%d") == today
        ]

        by_severity = {"critical": 0, "warning": 0, "info": 0, "positive": 0}
        for o in today_obs:
            by_severity[o.severity] = by_severity.get(o.severity, 0) + 1

        return {
            "date": today,
            "total_observations": len(today_obs),
            "by_severity": by_severity,
            "observations": [o.to_dict() for o in today_obs],
        }
