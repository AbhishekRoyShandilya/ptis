"""Real-time Cognitive Alert Engine.

Generates, routes, and stores structured cognitive alerts for five categories:
  1. Volatility Spike     — ATR has spiked beyond normal trading range
  2. Risk Deviation       — position size or stop deviates from plan
  3. Stop Widening        — stop has been moved further away from entry
  4. Emotional Override   — rapid re-entry, revenge trade, FOMO signature
  5. Regime Mismatch      — trade taken in a regime the strategy avoids

Plus supplementary alerts:
  6. Daily Loss Limit     — daily P&L approaching/breaching halt threshold
  7. Portfolio Heat       — total open risk approaching cap
  8. Overtrading          — trade count exceeds session recommended max
  9. Drawdown Warning     — account in significant drawdown

Architecture:
  AlertEngine is stateful (holds active alerts in memory with TTL).
  All alerts are also published on EventType.COGNITIVE_ALERT so the
  WebSocket route can broadcast them to the trader's browser immediately.

Usage:
    engine = get_alert_engine()
    alerts = engine.evaluate_trade(trade_params...)
    active = engine.get_active()
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from core.enums import MarketRegime, TradeDirection
from core.events import Event, EventType, get_event_bus

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  ALERT TYPES & SEVERITY
# ──────────────────────────────────────────────────────────────────────────────

class AlertType(str, Enum):
    VOLATILITY_SPIKE   = "volatility_spike"
    RISK_DEVIATION     = "risk_deviation"
    STOP_WIDENING      = "stop_widening"
    EMOTIONAL_OVERRIDE = "emotional_override"
    REGIME_MISMATCH    = "regime_mismatch"
    DAILY_LOSS_LIMIT   = "daily_loss_limit"
    OVERTRADING        = "overtrading"
    PORTFOLIO_HEAT     = "portfolio_heat"
    DRAWDOWN_WARNING   = "drawdown_warning"
    DISCIPLINE_BREACH  = "discipline_breach"


class AlertSeverity(str, Enum):
    INFO     = "info"       # informational, no action required
    WARNING  = "warning"    # notable — trader should be aware
    CRITICAL = "critical"   # immediate attention needed


# ──────────────────────────────────────────────────────────────────────────────
#  ALERT DATA STRUCTURE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CognitiveAlert:
    id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    action_required: str
    timestamp: datetime
    trade_id: Optional[str] = None
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    ttl_seconds: int = 300

    @property
    def is_expired(self) -> bool:
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "action_required": self.action_required,
            "timestamp": self.timestamp.isoformat(),
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AlertEngineConfig:
    vol_spike_atr_ratio: float = 1.8       # flag if ATR > N× baseline
    vol_spike_percentile: float = 85.0     # or if ATR percentile > this
    risk_deviation_pct: float = 0.5        # flag if risk_pct exceeds plan by >N%
    stop_widen_atr_threshold: float = 0.5  # flag if SL widens > N×ATR
    rapid_reentry_minutes: float = 2.0
    revenge_trade_minutes: float = 5.0
    consecutive_loss_threshold: int = 2
    drawdown_warning_pct: float = 5.0
    drawdown_critical_pct: float = 10.0
    daily_loss_halt_pct: float = 3.0
    max_portfolio_heat_pct: float = 6.0
    max_trades_per_session: int = 6


# ──────────────────────────────────────────────────────────────────────────────
#  COGNITIVE ALERT ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class CognitiveAlertEngine:
    """
    Real-time cognitive alert engine.

    Maintains an in-memory store of active alerts with TTL-based expiry.
    All generated alerts are published on the event bus so the WebSocket
    route can broadcast them to connected trader clients immediately.
    """

    def __init__(self, config: Optional[AlertEngineConfig] = None):
        self.cfg = config or AlertEngineConfig()
        self._active: Dict[str, CognitiveAlert] = {}
        self._bus = get_event_bus()

    # ── Master evaluation entry point ─────────────────────────────────────────

    def evaluate_trade(
        self,
        trade_id: str,
        symbol: str,
        direction: TradeDirection,
        entry_price: float,
        stop_loss: float,
        risk_pct: float,
        plan_risk_pct: float,
        current_atr: float,
        atr_percentile: float,
        baseline_atr: float,
        regime: MarketRegime,
        avoid_regimes: List[MarketRegime],
        session_daily_pnl: float,
        session_account_balance: float,
        session_trade_count: int,
        consecutive_losses: int,
        last_trade_pnl: Optional[float] = None,
        last_trade_time: Optional[datetime] = None,
        open_risk_pct: float = 0.0,
        current_drawdown_pct: float = 0.0,
        initial_stop: Optional[float] = None,
    ) -> List[CognitiveAlert]:
        """Run all cognitive checks for a newly entered or updated trade."""
        checks = [
            self._check_volatility_spike(trade_id, symbol, current_atr, baseline_atr, atr_percentile),
            self._check_risk_deviation(trade_id, symbol, risk_pct, plan_risk_pct),
            self._check_stop_widening(trade_id, symbol, direction, entry_price, stop_loss, initial_stop, current_atr),
            self._check_emotional_override(trade_id, symbol, consecutive_losses, last_trade_pnl, last_trade_time, risk_pct, plan_risk_pct),
            self._check_regime_mismatch(trade_id, symbol, regime, avoid_regimes),
            self._check_daily_loss_limit(trade_id, symbol, session_daily_pnl, session_account_balance),
            self._check_portfolio_heat(trade_id, symbol, open_risk_pct),
            self._check_overtrading(trade_id, symbol, session_trade_count),
            self._check_drawdown_warning(trade_id, symbol, current_drawdown_pct),
        ]

        alerts = []
        for alert in checks:
            if alert is not None:
                self._store_and_publish(alert)
                alerts.append(alert)
        return alerts

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_volatility_spike(self, trade_id, symbol, current_atr, baseline_atr, atr_percentile) -> Optional[CognitiveAlert]:
        if baseline_atr <= 0:
            return None
        atr_ratio = current_atr / baseline_atr
        if atr_ratio < self.cfg.vol_spike_atr_ratio and atr_percentile < self.cfg.vol_spike_percentile:
            return None
        severity = AlertSeverity.CRITICAL if atr_ratio >= 2.5 else AlertSeverity.WARNING
        return self._make(
            AlertType.VOLATILITY_SPIKE, severity,
            "Volatility Spike Detected",
            f"ATR is {atr_ratio:.1f}× baseline (percentile: {atr_percentile:.0f}th). "
            f"Current ATR {current_atr:.4f} vs baseline {baseline_atr:.4f}.",
            "Reduce position size 25–50%. Widen stops. Consider waiting for vol to normalise.",
            trade_id=trade_id, symbol=symbol,
            metadata={"atr_ratio": round(atr_ratio, 2), "current_atr": current_atr,
                      "baseline_atr": baseline_atr, "atr_percentile": atr_percentile},
            ttl_seconds=600,
        )

    def _check_risk_deviation(self, trade_id, symbol, actual_risk_pct, plan_risk_pct) -> Optional[CognitiveAlert]:
        excess = actual_risk_pct - plan_risk_pct
        if excess <= self.cfg.risk_deviation_pct:
            return None
        severity = AlertSeverity.CRITICAL if excess > 1.0 else AlertSeverity.WARNING
        return self._make(
            AlertType.RISK_DEVIATION, severity,
            "Risk Deviation from Plan",
            f"Position risk {actual_risk_pct:.2f}% vs planned {plan_risk_pct:.2f}% ({excess:+.2f}% over plan).",
            f"Reduce position to match plan risk of {plan_risk_pct:.2f}%.",
            trade_id=trade_id, symbol=symbol,
            metadata={"actual": actual_risk_pct, "plan": plan_risk_pct, "excess": round(excess, 3)},
        )

    def _check_stop_widening(self, trade_id, symbol, direction, entry, current_stop, initial_stop, atr) -> Optional[CognitiveAlert]:
        if initial_stop is None or atr <= 0:
            return None
        if direction == TradeDirection.LONG:
            initial_dist = entry - initial_stop
            current_dist = entry - current_stop
        else:
            initial_dist = initial_stop - entry
            current_dist = current_stop - entry
        widened_by = current_dist - initial_dist
        widened_atr = widened_by / atr
        if widened_atr < self.cfg.stop_widen_atr_threshold:
            return None
        severity = AlertSeverity.CRITICAL if widened_atr > 1.0 else AlertSeverity.WARNING
        return self._make(
            AlertType.STOP_WIDENING, severity,
            "Stop Widened Beyond Plan",
            f"Stop widened by {widened_by:.4f} ({widened_atr:.1f}× ATR). "
            f"Original: {initial_stop:.4f}, current: {current_stop:.4f}.",
            "Stop widening is a common emotional response. Restore original stop if placement was correct. "
            "Widening increases loss exposure and breaks your plan.",
            trade_id=trade_id, symbol=symbol,
            metadata={"initial_stop": initial_stop, "current_stop": current_stop,
                      "widened_atr": round(widened_atr, 2)},
            ttl_seconds=600,
        )

    def _check_emotional_override(self, trade_id, symbol, consecutive_losses, last_pnl, last_time, actual_risk, plan_risk) -> Optional[CognitiveAlert]:
        now = datetime.now(timezone.utc)
        is_oversized = actual_risk > plan_risk * 1.3

        if last_pnl is not None and last_pnl < 0 and last_time is not None:
            mins = (now - last_time).total_seconds() / 60
            if mins < self.cfg.rapid_reentry_minutes:
                return self._make(
                    AlertType.EMOTIONAL_OVERRIDE, AlertSeverity.CRITICAL,
                    "Rapid Re-entry After Loss — Likely Revenge Trade",
                    f"Entry {mins:.1f}min after ${abs(last_pnl):.2f} loss. "
                    f"{consecutive_losses} consecutive loss(es). "
                    f"{'Position oversized.' if is_oversized else ''}",
                    "STOP. Consider closing. Take a 10-minute break. "
                    "Revenge trading dramatically increases blow-up risk.",
                    trade_id=trade_id, symbol=symbol,
                    metadata={"minutes_since": round(mins, 1), "last_pnl": last_pnl,
                              "consecutive_losses": consecutive_losses, "oversized": is_oversized},
                    ttl_seconds=900,
                )
            elif mins < self.cfg.revenge_trade_minutes:
                sev = AlertSeverity.CRITICAL if consecutive_losses >= 2 else AlertSeverity.WARNING
                return self._make(
                    AlertType.EMOTIONAL_OVERRIDE, sev,
                    "Emotional Re-entry Risk — Cooldown Not Observed",
                    f"Entry {mins:.1f}min after a loss. {consecutive_losses} consecutive loss(es). "
                    "5-minute cooldown not observed.",
                    "Verify this is a clean pre-planned setup. If uncertain, step away.",
                    trade_id=trade_id, symbol=symbol,
                    metadata={"minutes_since": round(mins, 1), "consecutive_losses": consecutive_losses},
                )

        if consecutive_losses >= self.cfg.consecutive_loss_threshold and is_oversized:
            return self._make(
                AlertType.EMOTIONAL_OVERRIDE, AlertSeverity.CRITICAL,
                "Oversizing After Losses — Loss-Chasing Pattern",
                f"Risk {actual_risk:.2f}% vs plan {plan_risk:.2f}% after "
                f"{consecutive_losses} consecutive losses. Classic loss-chasing.",
                "Reduce to standard size. After consecutive losses, protocol is to REDUCE (not increase) size.",
                trade_id=trade_id, symbol=symbol,
                metadata={"consecutive_losses": consecutive_losses, "actual_risk": actual_risk, "plan_risk": plan_risk},
            )
        return None

    def _check_regime_mismatch(self, trade_id, symbol, regime, avoid_regimes) -> Optional[CognitiveAlert]:
        if not avoid_regimes or regime not in avoid_regimes:
            return None
        return self._make(
            AlertType.REGIME_MISMATCH, AlertSeverity.WARNING,
            "Regime Mismatch — Strategy Not Suited Here",
            f"Current regime '{regime.value}' is in the strategy's avoid list. "
            "Historical performance in this regime is poor.",
            "Reduce size to 50% or pass on this trade. Wait for regime alignment.",
            trade_id=trade_id, symbol=symbol,
            metadata={"current_regime": regime.value, "avoid_regimes": [r.value for r in avoid_regimes]},
        )

    def _check_daily_loss_limit(self, trade_id, symbol, daily_pnl, account_balance) -> Optional[CognitiveAlert]:
        if account_balance <= 0 or daily_pnl >= 0:
            return None
        loss_pct = abs(daily_pnl) / account_balance * 100
        if loss_pct >= self.cfg.daily_loss_halt_pct:
            return self._make(
                AlertType.DAILY_LOSS_LIMIT, AlertSeverity.CRITICAL,
                "Daily Loss Limit Breached — Stop Trading",
                f"Daily loss ${abs(daily_pnl):.2f} ({loss_pct:.1f}%) exceeds {self.cfg.daily_loss_halt_pct:.1f}% limit.",
                "STOP TRADING for today. Log off. Come back fresh tomorrow.",
                trade_id=trade_id, symbol=symbol,
                metadata={"daily_pnl": daily_pnl, "loss_pct": round(loss_pct, 2)},
                ttl_seconds=86400,
            )
        elif loss_pct >= self.cfg.daily_loss_halt_pct * 0.7:
            return self._make(
                AlertType.DAILY_LOSS_LIMIT, AlertSeverity.WARNING,
                "Approaching Daily Loss Limit",
                f"Daily loss ${abs(daily_pnl):.2f} ({loss_pct:.1f}%). Near {self.cfg.daily_loss_halt_pct:.1f}% limit.",
                "Reduce to minimum size or take a break.",
                trade_id=trade_id, symbol=symbol,
                metadata={"daily_pnl": daily_pnl, "loss_pct": round(loss_pct, 2)},
            )
        return None

    def _check_portfolio_heat(self, trade_id, symbol, open_risk_pct) -> Optional[CognitiveAlert]:
        ratio = open_risk_pct / max(self.cfg.max_portfolio_heat_pct, 0.01)
        if ratio < 0.8:
            return None
        sev = AlertSeverity.CRITICAL if ratio >= 1.0 else AlertSeverity.WARNING
        return self._make(
            AlertType.PORTFOLIO_HEAT, sev,
            "Portfolio Heat Cap Breached" if ratio >= 1.0 else "Portfolio Heat Approaching Cap",
            f"Total open risk {open_risk_pct:.1f}% (cap: {self.cfg.max_portfolio_heat_pct:.1f}%).",
            "Do not add new positions until existing ones close. Multiple positions compound correlation risk.",
            trade_id=trade_id, symbol=symbol,
            metadata={"open_risk_pct": round(open_risk_pct, 2), "cap": self.cfg.max_portfolio_heat_pct},
        )

    def _check_overtrading(self, trade_id, symbol, trade_count) -> Optional[CognitiveAlert]:
        if trade_count < self.cfg.max_trades_per_session:
            return None
        return self._make(
            AlertType.OVERTRADING, AlertSeverity.WARNING,
            "Overtrading Risk",
            f"{trade_count} trades this session (max: {self.cfg.max_trades_per_session}). Quality over quantity.",
            "Step back. Assess whether remaining setups have true edge.",
            trade_id=trade_id, symbol=symbol,
            metadata={"trade_count": trade_count, "max": self.cfg.max_trades_per_session},
        )

    def _check_drawdown_warning(self, trade_id, symbol, drawdown_pct) -> Optional[CognitiveAlert]:
        warning_threshold = self.cfg.drawdown_warning_pct / 100
        critical_threshold = self.cfg.drawdown_critical_pct / 100
        if drawdown_pct < warning_threshold:
            return None
        is_critical = drawdown_pct >= critical_threshold
        return self._make(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.CRITICAL if is_critical else AlertSeverity.WARNING,
            "Critical Drawdown — Mandatory Size Reduction" if is_critical else "Drawdown Warning",
            f"Account {drawdown_pct*100:.1f}% below peak equity.",
            f"Reduce size to {'25%' if is_critical else '50%'} of normal. Protect remaining capital.",
            trade_id=trade_id, symbol=symbol,
            metadata={"drawdown_pct": round(drawdown_pct * 100, 2)},
            ttl_seconds=3600,
        )

    # ── Factory helper ────────────────────────────────────────────────────────

    def _make(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        action_required: str,
        trade_id: Optional[str] = None,
        symbol: Optional[str] = None,
        metadata: Optional[Dict] = None,
        ttl_seconds: int = 300,
    ) -> CognitiveAlert:
        return CognitiveAlert(
            id=str(uuid.uuid4()),
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            action_required=action_required,
            timestamp=datetime.now(timezone.utc),
            trade_id=trade_id,
            symbol=symbol,
            metadata=metadata or {},
            ttl_seconds=ttl_seconds,
        )

    def _store_and_publish(self, alert: CognitiveAlert) -> None:
        self._active[alert.id] = alert
        self._bus.publish(Event(
            event_type=EventType.COGNITIVE_ALERT,
            data={"cognitive_alert": alert.to_dict(), "timestamp": alert.timestamp.isoformat()},
        ))
        log_fn = logger.critical if alert.severity == AlertSeverity.CRITICAL else logger.warning
        log_fn("ALERT [%s] %s: %s", alert.severity.value, alert.alert_type.value, alert.title)

    # ── Public query API ──────────────────────────────────────────────────────

    def get_active(
        self,
        trade_id: Optional[str] = None,
        min_severity: AlertSeverity = AlertSeverity.INFO,
    ) -> List[CognitiveAlert]:
        order = {AlertSeverity.INFO: 0, AlertSeverity.WARNING: 1, AlertSeverity.CRITICAL: 2}
        min_lvl = order[min_severity]
        return sorted(
            [
                a for a in self._active.values()
                if not a.is_expired and not a.acknowledged
                and order[a.severity] >= min_lvl
                and (trade_id is None or a.trade_id == trade_id)
            ],
            key=lambda a: a.timestamp,
            reverse=True,
        )

    def acknowledge(self, alert_id: str) -> bool:
        if alert_id in self._active:
            self._active[alert_id].acknowledged = True
            return True
        return False

    def clear_expired(self) -> int:
        expired = [k for k, v in self._active.items() if v.is_expired]
        for k in expired:
            del self._active[k]
        return len(expired)

    def get_summary(self) -> Dict[str, Any]:
        active = self.get_active()
        return {
            "total_active": len(active),
            "critical": sum(1 for a in active if a.severity == AlertSeverity.CRITICAL),
            "warnings": sum(1 for a in active if a.severity == AlertSeverity.WARNING),
            "by_type": {t.value: sum(1 for a in active if a.alert_type == t) for t in AlertType},
        }


# ── Module singleton ──────────────────────────────────────────────────────────

_engine: Optional[CognitiveAlertEngine] = None


def get_alert_engine(config: Optional[AlertEngineConfig] = None) -> CognitiveAlertEngine:
    global _engine
    if _engine is None:
        _engine = CognitiveAlertEngine(config)
    return _engine
