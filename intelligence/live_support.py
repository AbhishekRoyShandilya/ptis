"""
Live Cognitive Support
======================

Module responsible for monitoring live trade activity, session state and
market conditions in order to detect discipline violations, emotional overrides
or other cognitive risk factors while a trade is in progress.  When a potential
issue is identified the support engine emits a structured alert that can be
consumed by the realtime service, coaching agent or user interface.

This is intentionally lightweight and event-driven; actual rules can be added
incrementally as the product matures.

Design notes:
- subscribes to core.events for trade lifecycle and override events
- uses core.session_manager to keep track of intraday context
- publishes `EventType.COGNITIVE_ALERT` events with metadata
- integrates with RealtimeIntelligenceService via callback registration
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from core.events import Event, EventType, get_event_bus
from core.session_manager import TradingSessionManager, get_session_manager
from strategy_engine.override_detector import OverrideDetector, SessionContext
from core.models import Trade

logger = logging.getLogger(__name__)


class LiveCognitiveSupport:
    """Entry point for live cognitive rule monitoring.

    The class is mostly a coordinator; individual check methods implement
    specific rules (stop widening, oversizing, volatility spike, etc.).  New
    rules should append to ``self._checks`` and return alert dictionaries when
    triggered.
    """

    def __init__(self, alert_callback: Optional[Callable[[dict[str, Any]], None]] = None):
        self.session_mgr: TradingSessionManager = get_session_manager()
        self.override_detector = OverrideDetector()
        self._checks: list[Callable[[Trade], Optional[dict[str, Any]]]] = []
        self._alert_callback = alert_callback

        # subscribe to trade events
        bus = get_event_bus()
        bus.subscribe(EventType.TRADE_ENTERED, self._on_trade_entered)
        bus.subscribe(EventType.TRADE_CLOSED, self._on_trade_closed)
        bus.subscribe(EventType.OVERRIDE_DETECTED, self._on_override)

        # Built-in cognitive checks
        self._checks.extend([
            self._check_stop_widen,
            self._check_oversize,
            self._check_volatility_spike,
            self._check_emotional_override,
            self._check_regime_mismatch,
        ])

    def register_check(self, fn: Callable[[Trade], Optional[dict[str, Any]]]) -> None:
        """Allow external code to add custom checks."""
        self._checks.append(fn)

    # ── Event handlers ───────────────────────────────────────────

    def _on_trade_entered(self, event: Event) -> None:
        trade = event.data.get("trade")
        if trade:
            self.session_mgr.on_trade_entered(
                trade_id=trade.id,
                symbol=trade.symbol,
                entry_time=trade.entry_time,
                risk_amount=trade.risk_amount if hasattr(trade, "risk_amount") else None,
            )
            self._evaluate_trade(trade)

    def _on_trade_closed(self, event: Event) -> None:
        trade = event.data.get("trade")
        if trade:
            self.session_mgr.on_trade_closed(
                trade_id=trade.id,
                symbol=trade.symbol,
                pnl=trade.pnl or 0.0,
                exit_time=trade.exit_time or datetime.utcnow(),
            )

    def _on_override(self, event: Event) -> None:
        # simply forward overrides as cognitive alerts
        alert = {
            "type": "override",
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
        }
        self._publish(alert)

    # ── Rule evaluation ───────────────────────────────────────────

    def _evaluate_trade(self, trade: Trade) -> None:
        """Run all registered checks against a new trade entry."""
        for check in self._checks:
            try:
                result = check(trade)
                if result:
                    self._publish(result)
            except Exception as e:
                logger.error("Live check %s failed: %s", check.__name__, e)

    def _check_stop_widen(self, trade: Trade) -> Optional[dict[str, Any]]:
        """Flag if current SL has been widened beyond the original plan."""
        initial_stop = getattr(trade, "initial_stop_loss", None)
        current_stop = getattr(trade, "stop_loss", None)
        direction    = getattr(trade, "direction", None)
        entry        = getattr(trade, "entry_price", None)
        atr          = getattr(trade, "atr_at_entry", None)
        if any(v is None for v in [initial_stop, current_stop, direction, entry, atr]) or atr <= 0:
            return None
        from core.enums import TradeDirection
        is_long = str(direction).lower() == "long" or direction == TradeDirection.LONG
        initial_dist = (entry - initial_stop) if is_long else (initial_stop - entry)
        current_dist = (entry - current_stop) if is_long else (current_stop - entry)
        widened_atr  = (current_dist - initial_dist) / atr
        if widened_atr < 0.5:
            return None
        return {
            "type": "stop_widening",
            "severity": "critical" if widened_atr > 1.0 else "warning",
            "title": "Stop Widened Beyond Plan",
            "message": (
                f"Stop moved {widened_atr:.1f}× ATR further from entry. "
                f"Original: {initial_stop:.4f} → Now: {current_stop:.4f}."
            ),
            "action_required": (
                "Restore original stop if thesis is still valid. "
                "Widening a stop after unrealised loss is a discipline failure."
            ),
            "trade_id": str(trade.id), "symbol": trade.symbol,
            "metadata": {"initial_stop": initial_stop, "current_stop": current_stop,
                         "widened_atr": round(widened_atr, 2)},
        }

    def _check_oversize(self, trade: Trade) -> Optional[dict[str, Any]]:
        """Flag if actual position risk exceeds plan risk by > 0.5%."""
        risk_pct  = getattr(trade, "risk_pct", None)
        plan_risk = getattr(trade, "plan_risk_pct", 2.0)
        if risk_pct is None:
            return None
        excess = risk_pct - plan_risk
        if excess <= 0.5:
            return None
        return {
            "type": "risk_deviation",
            "severity": "critical" if risk_pct > plan_risk * 2 else "warning",
            "title": "Position Oversized",
            "message": f"Risk {risk_pct:.2f}% vs plan {plan_risk:.2f}% ({excess:+.2f}% over).",
            "action_required": f"Reduce to {plan_risk:.2f}% risk. Oversizing compounds losses.",
            "trade_id": str(trade.id), "symbol": trade.symbol,
            "metadata": {"actual_risk_pct": risk_pct, "plan_risk_pct": plan_risk},
        }

    def _check_volatility_spike(self, trade: Trade) -> Optional[dict[str, Any]]:
        """Flag if ATR has spiked > 1.8× baseline at entry."""
        current_atr  = getattr(trade, "atr_at_entry", None)
        baseline_atr = getattr(trade, "atr_baseline", None)
        atr_pct      = getattr(trade, "atr_percentile", 50.0)
        if current_atr is None or baseline_atr is None or baseline_atr <= 0:
            return None
        ratio = current_atr / baseline_atr
        if ratio < 1.8 and atr_pct < 85.0:
            return None
        return {
            "type": "volatility_spike",
            "severity": "critical" if ratio >= 2.5 else "warning",
            "title": "Volatility Spike at Entry",
            "message": (
                f"ATR is {ratio:.1f}× baseline (percentile {atr_pct:.0f}th). "
                f"ATR {current_atr:.4f} vs baseline {baseline_atr:.4f}."
            ),
            "action_required": "Reduce size 25–50%. High vol increases stop-out risk.",
            "trade_id": str(trade.id), "symbol": trade.symbol,
            "metadata": {"atr_ratio": round(ratio, 2), "atr_percentile": atr_pct},
        }

    def _check_emotional_override(self, trade: Trade) -> Optional[dict[str, Any]]:
        """Detect rapid re-entry after a loss — revenge trade / FOMO signal."""
        last_pnl        = getattr(trade, "last_trade_pnl", None)
        time_since_secs = getattr(trade, "time_since_last_trade_seconds", None)
        consecutive     = getattr(trade, "consecutive_losses", 0)
        risk_pct        = getattr(trade, "risk_pct", 1.0)
        plan_risk       = getattr(trade, "plan_risk_pct", 2.0)
        if last_pnl is None or last_pnl >= 0 or time_since_secs is None:
            return None
        mins         = time_since_secs / 60.0
        is_oversized = risk_pct > plan_risk * 1.3
        if mins < 2.0:
            return {
                "type": "emotional_override",
                "severity": "critical",
                "title": "Rapid Re-entry After Loss — Likely Revenge Trade",
                "message": (
                    f"Entry {mins:.1f}min after ${abs(last_pnl):.2f} loss. "
                    f"{consecutive} consecutive loss(es)."
                    + (" Position oversized." if is_oversized else "")
                ),
                "action_required": "STOP. Take a 10-minute break. Revenge trading causes blow-ups.",
                "trade_id": str(trade.id), "symbol": trade.symbol,
                "metadata": {"mins_since_loss": round(mins, 1), "last_pnl": last_pnl,
                             "consecutive_losses": consecutive},
            }
        if mins < 5.0 and consecutive >= 1:
            return {
                "type": "emotional_override",
                "severity": "warning",
                "title": "Emotional Re-entry Risk",
                "message": (
                    f"Entry {mins:.1f}min after loss. {consecutive} consecutive loss(es). "
                    "5-min cooldown not observed."
                ),
                "action_required": "Confirm this is a pre-planned setup — not a reaction.",
                "trade_id": str(trade.id), "symbol": trade.symbol,
                "metadata": {"mins_since_loss": round(mins, 1), "consecutive_losses": consecutive},
            }
        return None

    def _check_regime_mismatch(self, trade: Trade) -> Optional[dict[str, Any]]:
        """Flag if current regime is in the strategy's avoid list."""
        from core.enums import MarketRegime
        current_regime = getattr(trade, "market_regime", None)
        avoid_regimes  = getattr(trade, "strategy_avoid_regimes", [])
        if not current_regime or not avoid_regimes:
            return None
        if isinstance(current_regime, str):
            try:
                current_regime = MarketRegime(current_regime)
            except ValueError:
                return None
        if current_regime not in avoid_regimes:
            return None
        return {
            "type": "regime_mismatch",
            "severity": "warning",
            "title": f"Regime Mismatch — {current_regime.value.replace('_', ' ').title()}",
            "message": (
                f"Strategy avoids '{current_regime.value}' regime. "
                "Historical performance here is below breakeven."
            ),
            "action_required": "Reduce size 50% or pass. Adverse regimes destroy edge.",
            "trade_id": str(trade.id), "symbol": trade.symbol,
            "metadata": {"current_regime": current_regime.value,
                         "avoid_regimes": [str(r) for r in avoid_regimes]},
        }

    # ── Alert publishing ──────────────────────────────────────────

    def _publish(self, alert: dict[str, Any]) -> None:
        message = {
            "cognitive_alert": alert,
            "timestamp": datetime.utcnow().isoformat(),
        }
        # send to callback if provided
        if self._alert_callback:
            try:
                self._alert_callback(message)
            except Exception:
                logger.exception("Alert callback failed")
        # also publish on internal event bus
        bus = get_event_bus()
        bus.publish(Event(event_type=EventType.COGNITIVE_ALERT, data=message))


# module-level helper
_support: Optional[LiveCognitiveSupport] = None

def get_live_support(alert_callback: Optional[Callable[[dict[str, Any]], None]] = None) -> LiveCognitiveSupport:
    global _support
    if _support is None:
        _support = LiveCognitiveSupport(alert_callback=alert_callback)
    return _support
