# strategy_engine/override_detector.py
"""
Tier 3: Detect behavioral overrides and rule deviations.
Compares actual trade execution against strategy rules.
Produces severity-scored override events for DB persistence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from core.enums import (
    MarketRegime,
    OverrideSeverity,
    OverrideType,
    TradeDirection,
)
from strategy_engine.conditions.base_condition import MarketSnapshot, TradeContext
from strategy_engine.templates.loader import StrategyTemplate

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  OVERRIDE RESULT
# ══════════════════════════════════════════════

@dataclass
class OverrideDetection:
    """A single detected override / rule deviation."""
    override_type: OverrideType
    severity: OverrideSeverity
    severity_score: float              # 0.0 - 1.0
    description: str
    rule_expected: str = ""
    actual_value: str = ""
    expected_value: str = ""
    strategy_id: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "override_type": self.override_type.value,
            "severity": self.severity.value,
            "severity_score": round(self.severity_score, 3),
            "description": self.description,
            "rule_expected": self.rule_expected,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
            "strategy_id": self.strategy_id,
            "context": self.context,
        }


@dataclass
class OverrideReport:
    """Complete override analysis for a trade."""
    trade_id: Optional[str] = None
    strategy_id: Optional[str] = None
    overrides: list[OverrideDetection] = field(default_factory=list)
    total_count: int = 0
    max_severity: Optional[OverrideSeverity] = None
    max_severity_score: float = 0.0
    composite_deviation_score: float = 0.0    # 0.0 (clean) - 1.0 (major deviations)
    discipline_score: float = 100.0           # 100 (perfect) - 0 (failed)

    @property
    def has_overrides(self) -> bool:
        return self.total_count > 0

    @property
    def by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.overrides:
            key = o.override_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def severity_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {s.value: 0 for s in OverrideSeverity}
        for o in self.overrides:
            dist[o.severity.value] += 1
        return dist

    def summary(self) -> dict[str, Any]:
        return {
            "total_overrides": self.total_count,
            "by_type": self.by_type,
            "severity_distribution": self.severity_distribution,
            "max_severity": self.max_severity.value if self.max_severity else None,
            "composite_deviation_score": round(self.composite_deviation_score, 3),
            "discipline_score": round(self.discipline_score, 1),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "strategy_id": self.strategy_id,
            **self.summary(),
            "overrides": [o.to_dict() for o in self.overrides],
        }


# ══════════════════════════════════════════════
#  SEVERITY SCORING
# ══════════════════════════════════════════════

SEVERITY_WEIGHTS: dict[OverrideSeverity, float] = {
    OverrideSeverity.LOW: 0.1,
    OverrideSeverity.MEDIUM: 0.3,
    OverrideSeverity.HIGH: 0.6,
    OverrideSeverity.CRITICAL: 1.0,
}

OVERRIDE_BASE_SEVERITY: dict[OverrideType, OverrideSeverity] = {
    OverrideType.EARLY_ENTRY: OverrideSeverity.MEDIUM,
    OverrideType.LATE_ENTRY: OverrideSeverity.LOW,
    OverrideType.WRONG_DIRECTION: OverrideSeverity.CRITICAL,
    OverrideType.MISSING_CONFIRMATION: OverrideSeverity.HIGH,
    OverrideType.AVOID_CONDITION_VIOLATED: OverrideSeverity.HIGH,
    OverrideType.OVERSIZING: OverrideSeverity.HIGH,
    OverrideType.UNDERSIZING: OverrideSeverity.LOW,
    OverrideType.STOP_TOO_WIDE: OverrideSeverity.MEDIUM,
    OverrideType.STOP_TOO_TIGHT: OverrideSeverity.LOW,
    OverrideType.NO_STOP_SET: OverrideSeverity.CRITICAL,
    OverrideType.STOP_MOVED: OverrideSeverity.HIGH,
    OverrideType.EARLY_EXIT: OverrideSeverity.MEDIUM,
    OverrideType.LATE_EXIT: OverrideSeverity.MEDIUM,
    OverrideType.TARGET_SKIPPED: OverrideSeverity.MEDIUM,
    OverrideType.TIME_WINDOW_VIOLATION: OverrideSeverity.MEDIUM,
    OverrideType.REGIME_MISMATCH: OverrideSeverity.MEDIUM,
    OverrideType.INVALIDATION_IGNORED: OverrideSeverity.HIGH,
    OverrideType.RAPID_RE_ENTRY: OverrideSeverity.HIGH,
}


def _severity_score(severity: OverrideSeverity) -> float:
    return SEVERITY_WEIGHTS[severity]


# ══════════════════════════════════════════════
#  SESSION CONTEXT (for behavioral checks)
# ══════════════════════════════════════════════

@dataclass
class SessionContext:
    """Intraday session state for behavioral override detection."""
    trades_today: int = 0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    last_trade_time: Optional[datetime] = None
    last_trade_pnl: Optional[float] = None
    last_trade_symbol: Optional[str] = None
    daily_max_loss_threshold: Optional[float] = None
    account_balance: Optional[float] = None


# ══════════════════════════════════════════════
#  OVERRIDE DETECTOR
# ══════════════════════════════════════════════

class OverrideDetector:
    """
    Detects behavioral overrides and rule deviations by comparing
    actual trade parameters against strategy template rules.
    """

    def __init__(self):
        self._detectors = [
            self._check_time_window,
            self._check_regime_compatibility,
            self._check_risk_sizing,
            self._check_stop_loss,
            self._check_missing_confirmations,
            self._check_avoid_violations,
            self._check_rapid_re_entry,
            self._check_daily_limits,
        ]

    def detect(
        self,
        trade: TradeContext,
        template: StrategyTemplate,
        market: MarketSnapshot,
        session: SessionContext | None = None,
        match_result: Any | None = None,
    ) -> OverrideReport:
        """
        Run all override detectors against a trade.

        Args:
            trade: The actual trade context
            template: Strategy template the trade is attributed to
            market: Market snapshot at trade entry
            session: Session state for behavioral checks
            match_result: StrategyMatchResult from Tier 2 (for condition re-use)
        """
        report = OverrideReport(
            trade_id=str(getattr(trade, "trade_id", None)),
            strategy_id=template.id,
        )

        if session is None:
            session = SessionContext(
                trades_today=trade.trades_today,
                daily_pnl=trade.daily_pnl,
                consecutive_losses=trade.consecutive_losses,
            )
            if trade.time_since_last_trade_seconds is not None:
                session.last_trade_time = trade.entry_time - timedelta(
                    seconds=trade.time_since_last_trade_seconds
                )
            session.last_trade_pnl = trade.last_trade_pnl
            session.account_balance = trade.account_balance

        for detector_fn in self._detectors:
            try:
                overrides = detector_fn(trade, template, market, session, match_result)
                report.overrides.extend(overrides)
            except Exception as e:
                logger.error("Override detector '%s' failed: %s", detector_fn.__name__, e)

        # ── Compute aggregate scores ──
        report.total_count = len(report.overrides)

        if report.overrides:
            severities = [o.severity for o in report.overrides]
            severity_order = [
                OverrideSeverity.LOW,
                OverrideSeverity.MEDIUM,
                OverrideSeverity.HIGH,
                OverrideSeverity.CRITICAL,
            ]
            report.max_severity = max(severities, key=lambda s: severity_order.index(s))
            report.max_severity_score = max(o.severity_score for o in report.overrides)

            # Composite deviation: weighted sum capped at 1.0
            total_weight = sum(o.severity_score for o in report.overrides)
            report.composite_deviation_score = min(1.0, total_weight / 2.0)
            # Divide by 2.0 so it takes ~2 CRITICAL overrides to hit 1.0

        report.discipline_score = max(0.0, (1.0 - report.composite_deviation_score) * 100)

        # Tag strategy on all overrides
        for o in report.overrides:
            o.strategy_id = template.id

        logger.info(
            "Override detection for trade %s against '%s': %d overrides, discipline=%.1f",
            report.trade_id, template.id, report.total_count, report.discipline_score,
        )

        return report

    def detect_from_profile(
        self,
        trade: TradeContext,
        profile: Any,
        market: MarketSnapshot,
        session: SessionContext | None = None,
    ) -> OverrideReport:
        """
        Run override detection against a learned StrategyProfile.

        Converts the profile to a StrategyTemplate via the adapter,
        then delegates to the standard detect() method. This reuses all
        8 existing detectors without modification.

        Args:
            trade: The actual trade context
            profile: A StrategyProfile DB model (from strategy_learning)
            market: Market snapshot at trade entry
            session: Session state for behavioral checks
        """
        from strategy_learning.profile_adapter import profile_to_template

        template = profile_to_template(profile)
        return self.detect(trade, template, market, session)

    # ── Individual Detectors ──

    def _check_time_window(
        self, trade: TradeContext, template: StrategyTemplate,
        market: MarketSnapshot, session: SessionContext, match_result: Any,
    ) -> list[OverrideDetection]:
        if not template.time_window:
            return []

        try:
            tz = ZoneInfo(template.time_window.timezone)
        except Exception:
            tz = ZoneInfo("Asia/Kolkata")

        from datetime import time

        entry_local = trade.entry_time.astimezone(tz)
        entry_t = entry_local.time()
        start_t = time(*map(int, template.time_window.start.split(":")))
        end_t = time(*map(int, template.time_window.end.split(":")))

        overrides = []

        if entry_t < start_t:
            minutes_early = (
                (start_t.hour * 60 + start_t.minute)
                - (entry_t.hour * 60 + entry_t.minute)
            )
            severity = OverrideSeverity.MEDIUM if minutes_early > 10 else OverrideSeverity.LOW
            overrides.append(OverrideDetection(
                override_type=OverrideType.EARLY_ENTRY,
                severity=severity,
                severity_score=_severity_score(severity),
                description=f"Entry {minutes_early}min before strategy window opens",
                rule_expected=f"Entry after {template.time_window.start}",
                actual_value=entry_t.strftime("%H:%M:%S"),
                expected_value=f">= {template.time_window.start}",
                context={"minutes_early": minutes_early},
            ))

        if entry_t > end_t:
            minutes_late = (
                (entry_t.hour * 60 + entry_t.minute)
                - (end_t.hour * 60 + end_t.minute)
            )
            severity = OverrideSeverity.MEDIUM if minutes_late > 15 else OverrideSeverity.LOW
            overrides.append(OverrideDetection(
                override_type=OverrideType.TIME_WINDOW_VIOLATION,
                severity=severity,
                severity_score=_severity_score(severity),
                description=f"Entry {minutes_late}min after strategy window closes",
                rule_expected=f"Entry before {template.time_window.end}",
                actual_value=entry_t.strftime("%H:%M:%S"),
                expected_value=f"<= {template.time_window.end}",
                context={"minutes_late": minutes_late},
            ))

        return overrides

    def _check_regime_compatibility(
        self, trade: TradeContext, template: StrategyTemplate,
        market: MarketSnapshot, session: SessionContext, match_result: Any,
    ) -> list[OverrideDetection]:
        regime = trade.market_regime or market.regime
        if not regime or regime == MarketRegime.UNKNOWN:
            return []

        if not template.is_regime_compatible(regime):
            return [OverrideDetection(
                override_type=OverrideType.REGIME_MISMATCH,
                severity=OverrideSeverity.MEDIUM,
                severity_score=_severity_score(OverrideSeverity.MEDIUM),
                description=f"Trade taken in '{regime.value}' regime which strategy avoids",
                rule_expected=f"Avoid regimes: {[r.value for r in template.avoid_regimes]}",
                actual_value=regime.value,
                expected_value=f"Not in {[r.value for r in template.avoid_regimes]}",
                context={"regime": regime.value},
            )]
        return []

    def _check_risk_sizing(
        self, trade: TradeContext, template: StrategyTemplate,
        market: MarketSnapshot, session: SessionContext, match_result: Any,
    ) -> list[OverrideDetection]:
        overrides = []
        risk = template.risk_rules

        # ── Position risk % check ──
        if trade.risk_pct is not None and risk.max_risk_pct:
            if trade.risk_pct > risk.max_risk_pct * 1.5:
                overrides.append(OverrideDetection(
                    override_type=OverrideType.OVERSIZING,
                    severity=OverrideSeverity.CRITICAL,
                    severity_score=_severity_score(OverrideSeverity.CRITICAL),
                    description=f"Risk {trade.risk_pct:.2f}% exceeds max {risk.max_risk_pct}% by >50%",
                    rule_expected=f"Max risk {risk.max_risk_pct}% per trade",
                    actual_value=f"{trade.risk_pct:.2f}%",
                    expected_value=f"<= {risk.max_risk_pct}%",
                    context={"risk_pct": trade.risk_pct, "max_allowed": risk.max_risk_pct},
                ))
            elif trade.risk_pct > risk.max_risk_pct:
                excess = trade.risk_pct - risk.max_risk_pct
                severity = OverrideSeverity.HIGH if excess > 0.5 else OverrideSeverity.MEDIUM
                overrides.append(OverrideDetection(
                    override_type=OverrideType.OVERSIZING,
                    severity=severity,
                    severity_score=_severity_score(severity),
                    description=f"Risk {trade.risk_pct:.2f}% exceeds max {risk.max_risk_pct}%",
                    rule_expected=f"Max risk {risk.max_risk_pct}% per trade",
                    actual_value=f"{trade.risk_pct:.2f}%",
                    expected_value=f"<= {risk.max_risk_pct}%",
                    context={"risk_pct": trade.risk_pct, "max_allowed": risk.max_risk_pct},
                ))

        # ── Risk amount check ──
        if trade.risk_amount is not None and risk.max_risk_amount:
            if trade.risk_amount > risk.max_risk_amount:
                severity = (
                    OverrideSeverity.CRITICAL
                    if trade.risk_amount > risk.max_risk_amount * 1.5
                    else OverrideSeverity.HIGH
                )
                overrides.append(OverrideDetection(
                    override_type=OverrideType.OVERSIZING,
                    severity=severity,
                    severity_score=_severity_score(severity),
                    description=f"Risk ${trade.risk_amount:.0f} exceeds max ${risk.max_risk_amount:.0f}",
                    rule_expected=f"Max risk ${risk.max_risk_amount:.0f} per trade",
                    actual_value=f"${trade.risk_amount:.2f}",
                    expected_value=f"<= ${risk.max_risk_amount:.2f}",
                ))

        # ── Min R-target check ──
        if trade.take_profit and trade.stop_loss and trade.entry_price:
            if trade.direction == TradeDirection.LONG:
                risk_dist = trade.entry_price - trade.stop_loss
                reward_dist = trade.take_profit - trade.entry_price
            else:
                risk_dist = trade.stop_loss - trade.entry_price
                reward_dist = trade.entry_price - trade.take_profit

            if risk_dist > 0:
                actual_rr = reward_dist / risk_dist
                if actual_rr < risk.min_r_target:
                    severity = (
                        OverrideSeverity.HIGH
                        if actual_rr < risk.min_r_target * 0.5
                        else OverrideSeverity.MEDIUM
                    )
                    overrides.append(OverrideDetection(
                        override_type=OverrideType.TARGET_SKIPPED,
                        severity=severity,
                        severity_score=_severity_score(severity),
                        description=f"R:R ratio {actual_rr:.1f} below minimum {risk.min_r_target}",
                        rule_expected=f"Min R:R target {risk.min_r_target}",
                        actual_value=f"{actual_rr:.2f}R",
                        expected_value=f">= {risk.min_r_target}R",
                        context={"actual_rr": actual_rr},
                    ))

        return overrides

    def _check_stop_loss(
        self, trade: TradeContext, template: StrategyTemplate,
        market: MarketSnapshot, session: SessionContext, match_result: Any,
    ) -> list[OverrideDetection]:
        overrides = []

        # ── No stop set ──
        if trade.stop_loss is None:
            overrides.append(OverrideDetection(
                override_type=OverrideType.NO_STOP_SET,
                severity=OverrideSeverity.CRITICAL,
                severity_score=_severity_score(OverrideSeverity.CRITICAL),
                description="No stop loss set on trade",
                rule_expected="Stop loss required by strategy",
            ))
            return overrides

        # ── Stop distance vs ATR ──
        atr = market.atr
        if atr and atr > 0:
            if trade.direction == TradeDirection.LONG:
                stop_dist = trade.entry_price - trade.stop_loss
            else:
                stop_dist = trade.stop_loss - trade.entry_price

            atr_ratio = stop_dist / atr

            if atr_ratio > 3.0:
                overrides.append(OverrideDetection(
                    override_type=OverrideType.STOP_TOO_WIDE,
                    severity=OverrideSeverity.MEDIUM,
                    severity_score=_severity_score(OverrideSeverity.MEDIUM),
                    description=f"Stop distance {atr_ratio:.1f}x ATR — unusually wide",
                    rule_expected="Stop within reasonable ATR multiple",
                    actual_value=f"{atr_ratio:.2f}x ATR",
                    expected_value="<= 3.0x ATR",
                    context={"atr_ratio": atr_ratio, "stop_dist": stop_dist, "atr": atr},
                ))
            elif atr_ratio < 0.3:
                overrides.append(OverrideDetection(
                    override_type=OverrideType.STOP_TOO_TIGHT,
                    severity=OverrideSeverity.LOW,
                    severity_score=_severity_score(OverrideSeverity.LOW),
                    description=f"Stop distance {atr_ratio:.1f}x ATR — very tight, likely noise stop",
                    rule_expected="Stop beyond noise range",
                    actual_value=f"{atr_ratio:.2f}x ATR",
                    expected_value=">= 0.3x ATR",
                    context={"atr_ratio": atr_ratio},
                ))

        return overrides

    def _check_missing_confirmations(
        self, trade: TradeContext, template: StrategyTemplate,
        market: MarketSnapshot, session: SessionContext, match_result: Any,
    ) -> list[OverrideDetection]:
        """
        Use Tier 2 match result to identify missing required confirmations.
        """
        if match_result is None:
            return []

        overrides = []

        # match_result is a StrategyMatchResult from the matcher
        for cond_eval in getattr(match_result, "condition_results", []):
            if cond_eval.category != "confirmation":
                continue
            if not cond_eval.condition_def.required:
                continue
            if cond_eval.result.is_failed:
                overrides.append(OverrideDetection(
                    override_type=OverrideType.MISSING_CONFIRMATION,
                    severity=OverrideSeverity.HIGH,
                    severity_score=_severity_score(OverrideSeverity.HIGH),
                    description=(
                        f"Required confirmation '{cond_eval.condition_def.type}' not met"
                    ),
                    rule_expected=f"Condition '{cond_eval.condition_def.type}' required",
                    actual_value=cond_eval.result.actual_value or "not met",
                    expected_value=cond_eval.result.expected_value or "met",
                    context={
                        "condition_type": cond_eval.condition_def.type,
                        "condition_params": cond_eval.condition_def.params,
                        "details": cond_eval.result.details,
                    },
                ))

        return overrides

    def _check_avoid_violations(
        self, trade: TradeContext, template: StrategyTemplate,
        market: MarketSnapshot, session: SessionContext, match_result: Any,
    ) -> list[OverrideDetection]:
        """
        Use Tier 2 match result to identify triggered avoid conditions.
        """
        if match_result is None:
            return []

        overrides = []

        for cond_eval in getattr(match_result, "condition_results", []):
            if cond_eval.category != "avoid":
                continue
            if cond_eval.result.is_met:  # MET = avoid condition triggered = bad
                overrides.append(OverrideDetection(
                    override_type=OverrideType.AVOID_CONDITION_VIOLATED,
                    severity=OverrideSeverity.HIGH,
                    severity_score=_severity_score(OverrideSeverity.HIGH),
                    description=(
                        f"Avoid condition triggered: "
                        f"{cond_eval.condition_def.description or cond_eval.condition_def.type}"
                    ),
                    rule_expected=f"Avoid: {cond_eval.condition_def.description}",
                    actual_value=cond_eval.result.actual_value or "triggered",
                    expected_value="not triggered",
                    context={
                        "condition_type": cond_eval.condition_def.type,
                        "details": cond_eval.result.details,
                    },
                ))

        return overrides

    def _check_rapid_re_entry(
        self, trade: TradeContext, template: StrategyTemplate,
        market: MarketSnapshot, session: SessionContext, match_result: Any,
    ) -> list[OverrideDetection]:
        """Detect rapid re-entry after a loss (revenge trading signal)."""
        if session.last_trade_time is None:
            return []

        seconds_since = (trade.entry_time - session.last_trade_time).total_seconds()
        minutes_since = seconds_since / 60.0

        # Rapid re-entry after loss
        if session.last_trade_pnl is not None and session.last_trade_pnl < 0:
            if minutes_since < 2.0:
                return [OverrideDetection(
                    override_type=OverrideType.RAPID_RE_ENTRY,
                    severity=OverrideSeverity.CRITICAL,
                    severity_score=_severity_score(OverrideSeverity.CRITICAL),
                    description=(
                        f"Re-entry {minutes_since:.1f}min after a loss "
                        f"(${session.last_trade_pnl:.0f}) — likely revenge trade"
                    ),
                    rule_expected="Wait for next clean setup after a loss",
                    actual_value=f"{minutes_since:.1f} minutes",
                    expected_value="> 5 minutes minimum",
                    context={
                        "minutes_since_last": minutes_since,
                        "last_pnl": session.last_trade_pnl,
                        "consecutive_losses": session.consecutive_losses,
                    },
                )]
            elif minutes_since < 5.0:
                return [OverrideDetection(
                    override_type=OverrideType.RAPID_RE_ENTRY,
                    severity=OverrideSeverity.HIGH,
                    severity_score=_severity_score(OverrideSeverity.HIGH),
                    description=(
                        f"Re-entry {minutes_since:.1f}min after a loss — "
                        f"potential emotional reaction"
                    ),
                    rule_expected="Allow cooldown period after loss",
                    actual_value=f"{minutes_since:.1f} minutes",
                    expected_value="> 5 minutes",
                    context={
                        "minutes_since_last": minutes_since,
                        "last_pnl": session.last_trade_pnl,
                    },
                )]

        return []

    def _check_daily_limits(
        self, trade: TradeContext, template: StrategyTemplate,
        market: MarketSnapshot, session: SessionContext, match_result: Any,
    ) -> list[OverrideDetection]:
        overrides = []
        risk = template.risk_rules

        # ── Max daily trades ──
        if risk.max_daily_trades and session.trades_today >= risk.max_daily_trades:
            overrides.append(OverrideDetection(
                override_type=OverrideType.TIME_WINDOW_VIOLATION,
                severity=OverrideSeverity.HIGH,
                severity_score=_severity_score(OverrideSeverity.HIGH),
                description=(
                    f"Daily trade limit exceeded: {session.trades_today + 1} trades "
                    f"(max {risk.max_daily_trades})"
                ),
                rule_expected=f"Max {risk.max_daily_trades} trades per day",
                actual_value=str(session.trades_today + 1),
                expected_value=f"<= {risk.max_daily_trades}",
                context={"trades_today": session.trades_today},
            ))

        # ── Max daily loss ──
        if risk.max_daily_loss_pct and session.account_balance:
            max_loss = session.account_balance * (risk.max_daily_loss_pct / 100.0)
            if session.daily_pnl < -max_loss:
                overrides.append(OverrideDetection(
                    override_type=OverrideType.OVERSIZING,
                    severity=OverrideSeverity.CRITICAL,
                    severity_score=_severity_score(OverrideSeverity.CRITICAL),
                    description=(
                        f"Trading after daily loss limit breached: "
                        f"${session.daily_pnl:.0f} (max -${max_loss:.0f})"
                    ),
                    rule_expected=f"Stop trading after {risk.max_daily_loss_pct}% daily loss",
                    actual_value=f"${session.daily_pnl:.2f}",
                    expected_value=f">= -${max_loss:.2f}",
                    context={"daily_pnl": session.daily_pnl, "max_loss": max_loss},
                ))

        return overrides