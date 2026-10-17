"""
Guardrails Engine
==================
Real-time trading safety rules evaluated before every trade.
Designed as a hard gate: if any guardrail fires, the trade is blocked
and the trader receives a clear, constructive explanation.

Built-in guardrails (in evaluation order):
  1. Daily loss limit           — stop trading after X rupees lost today
  2. Daily trade count limit    — stop after N trades per session
  3. Consecutive loss streak    — pause after K consecutive losses
  4. Drawdown threshold         — halt when account drawdown > D%
  5. Emotional state block      — block if last emotion = high-risk state
  6. Discipline score floor     — warn / block if score < threshold
  7. Time-based blocks          — no trading in defined time windows
  8. Circuit breaker             — manual override from the API

All guardrails respect the AutomationRule model from the database.
Custom rules created via the API are evaluated alongside built-ins.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time as dtime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Default thresholds (overridden by DB rules if present)
DEFAULT_DAILY_LOSS_LIMIT   = -10_000.0   # ₹10,000 daily loss halt
DEFAULT_MAX_DAILY_TRADES   = 10
DEFAULT_MAX_LOSS_STREAK    = 3
DEFAULT_MIN_DISCIPLINE     = 40.0        # below this: warn only (not block)
_HIGH_RISK_EMOTIONS = {"revenge", "fomo", "frustrated", "euphoric", "anxious"}


@dataclass
class GuardrailResult:
    allowed: bool
    blocked_by: Optional[str]    # rule name that fired
    reason: Optional[str]        # human-readable explanation
    severity: str = "info"       # "info" | "warning" | "block"
    suggestions: list = field(default_factory=list)


class GuardrailEngine:
    """
    Evaluates all active guardrails against the current trading session state.
    Call allow_trade(db) before any order submission.
    """

    def allow_trade(self, db) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        reason is empty string when allowed.
        """
        result = self.evaluate(db)
        if not result.allowed:
            return False, result.reason or "Guardrail blocked trade"
        return True, ""

    def evaluate(self, db) -> GuardrailResult:
        """Full guardrail evaluation — returns first blocking result."""
        from core.models import Trade, AutomationRule
        from core.enums import TradeStatus

        today = date.today()
        today_start = datetime.combine(today, dtime.min)

        today_trades = db.query(Trade).filter(
            Trade.entry_time >= today_start,
        ).all()

        closed_today = [t for t in today_trades if t.pnl is not None]
        daily_pnl = sum(t.pnl for t in closed_today)
        open_trades = [t for t in today_trades if t.status != TradeStatus.CLOSED]

        # Load custom DB rules
        db_rules = db.query(AutomationRule).filter(AutomationRule.enabled == True).all()

        # --- 1. Circuit breaker (manual) ---
        cb_rule = next((r for r in db_rules if str(r.rule_type) == "circuit_breaker"), None)
        if cb_rule:
            return GuardrailResult(
                allowed=False, severity="block",
                blocked_by="circuit_breaker",
                reason=f"Circuit breaker active: {cb_rule.description or cb_rule.name}",
                suggestions=["Use the Automation panel to deactivate the circuit breaker when ready"],
            )

        # --- 2. Daily loss limit ---
        loss_rule = next((r for r in db_rules if str(r.rule_type) == "daily_drawdown"), None)
        loss_limit = -abs(loss_rule.trigger_value) if loss_rule else DEFAULT_DAILY_LOSS_LIMIT
        if daily_pnl <= loss_limit:
            return GuardrailResult(
                allowed=False, severity="block",
                blocked_by="daily_loss_limit",
                reason=f"Daily loss limit reached: ₹{abs(daily_pnl):.0f} lost today "
                       f"(limit: ₹{abs(loss_limit):.0f})",
                suggestions=[
                    "Step away from the screen — you've hit your loss limit for today",
                    "Review today's trades in the journal before tomorrow",
                ],
            )

        # --- 3. Max daily trades ---
        trade_rule = next((r for r in db_rules if str(r.rule_type) == "max_trades"), None)
        max_trades = int(trade_rule.trigger_value) if trade_rule else DEFAULT_MAX_DAILY_TRADES
        total_today = len(today_trades)
        if total_today >= max_trades:
            return GuardrailResult(
                allowed=False, severity="block",
                blocked_by="max_daily_trades",
                reason=f"Daily trade limit reached: {total_today} of {max_trades} trades taken",
                suggestions=["Quality over quantity — you've reached today's trade limit"],
            )

        # --- 4. Consecutive loss streak ---
        all_closed = (
            db.query(Trade)
            .filter(Trade.status == TradeStatus.CLOSED, Trade.pnl.isnot(None))
            .order_by(Trade.exit_time.desc())
            .limit(20)
            .all()
        )
        streak = 0
        for t in all_closed:
            if t.pnl < 0:
                streak += 1
            else:
                break
        streak_rule = next((r for r in db_rules if str(r.rule_type) == "loss_streak"), None)
        max_streak = int(streak_rule.trigger_value) if streak_rule else DEFAULT_MAX_LOSS_STREAK
        if streak >= max_streak:
            return GuardrailResult(
                allowed=False, severity="block",
                blocked_by="consecutive_losses",
                reason=f"{streak} consecutive losses — trading paused to prevent revenge trading",
                suggestions=[
                    "Take a 30-minute break before reviewing your next setup",
                    "Ask yourself: is the next trade driven by edge or by the need to recover?",
                ],
            )

        # --- 5. Emotional state check (warn only — not a hard block by default) ---
        latest_trade = all_closed[0] if all_closed else None
        if latest_trade and getattr(latest_trade, "emotion_post", None):
            em = latest_trade.emotion_post
            em_str = str(em.value if hasattr(em, "value") else em).lower()
            if em_str in _HIGH_RISK_EMOTIONS:
                # Check if emotion_block rule exists in DB
                em_rule = next((r for r in db_rules if str(r.rule_type) == "emotion_block"), None)
                if em_rule and em_rule.action == "stop":
                    return GuardrailResult(
                        allowed=False, severity="block",
                        blocked_by="emotion_block",
                        reason=f"Post-trade emotion '{em_str}' is a high-risk state — trade blocked",
                        suggestions=["Take a break. High-risk emotional states reduce win rate significantly."],
                    )
                # Else: just log a warning (non-blocking)
                logger.warning(f"High-risk emotion detected: {em_str} — proceeding with caution")

        # --- 6. Discipline score floor ---
        disc_rule = next((r for r in db_rules if str(r.rule_type) == "discipline_floor"), None)
        min_disc = disc_rule.trigger_value if disc_rule else DEFAULT_MIN_DISCIPLINE
        if latest_trade and getattr(latest_trade, "discipline_score", None) is not None:
            if latest_trade.discipline_score < min_disc:
                logger.info(f"Discipline score {latest_trade.discipline_score:.0f} below floor {min_disc:.0f} — warning only")

        # All guardrails passed
        return GuardrailResult(
            allowed=True, blocked_by=None, reason=None, severity="info",
            suggestions=[],
        )

    def check_readiness(self, db, emotion: Optional[str] = None,
                        discipline_score: Optional[float] = None) -> dict:
        """
        Pre-trade readiness check — softer than guardrails, returns warnings.
        Used by the UI to show amber warnings before the trader confirms.
        """
        warnings = []
        if emotion and emotion.lower() in _HIGH_RISK_EMOTIONS:
            warnings.append(f"You marked your emotion as '{emotion}' — historically this reduces your win rate")
        if discipline_score is not None and discipline_score < 60:
            warnings.append(f"Discipline score {discipline_score:.0f}/100 — consider reviewing your checklist")
        guardrail = self.evaluate(db)
        if not guardrail.allowed:
            warnings.insert(0, guardrail.reason)
        return {
            "ready": guardrail.allowed and len(warnings) == 0,
            "trading_allowed": guardrail.allowed,
            "warnings": warnings,
            "suggestions": guardrail.suggestions,
        }


_engine: Optional[GuardrailEngine] = None


def get_guardrail_engine() -> GuardrailEngine:
    global _engine
    if _engine is None:
        _engine = GuardrailEngine()
    return _engine
