"""
Automation API Routes
======================
Endpoints for automation rules, guardrail status, strategy evolution triggers,
and circuit-breaker controls.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from api.auth import get_current_user

from core.database import get_db_dependency as get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/automation", tags=["automation"], dependencies=[Depends(get_current_user)])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AutomationRuleRequest(BaseModel):
    name: str
    rule_type: str = Field(..., description="loss_limit | daily_drawdown | max_trades | emotion_block | time_block")
    trigger_value: float
    action: str = Field(..., description="pause | stop | alert | reduce_size")
    enabled: bool = True
    description: Optional[str] = None


class AutomationRuleResponse(BaseModel):
    id: str
    name: str
    rule_type: str
    trigger_value: float
    action: str
    enabled: bool
    description: Optional[str]
    created_at: str
    triggered_count: int


class GuardrailStatusResponse(BaseModel):
    trading_allowed: bool
    active_blocks: list[dict]
    daily_pnl: float
    daily_loss_limit: float
    daily_trade_count: int
    max_daily_trades: Optional[int]
    circuit_breaker_active: bool
    reset_at: Optional[str]   # when the block resets (e.g. next day)


class CircuitBreakerRequest(BaseModel):
    reason: str
    duration_hours: int = Field(24, ge=1, le=168)


class StrategyEvolutionRequest(BaseModel):
    strategy_id: UUID
    min_trades: int = Field(30, ge=10, description="Min trades before evolution analysis")
    lookback_days: int = Field(90, ge=14)


class StrategyEvolutionResponse(BaseModel):
    strategy_id: str
    current_win_rate: float
    recommended_adjustments: list[dict]
    edge_decay_detected: bool
    confidence: float
    analysis_summary: str


# ---------------------------------------------------------------------------
# Automation Rules CRUD
# ---------------------------------------------------------------------------

@router.get("/rules", response_model=list[AutomationRuleResponse])
def list_rules(db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """List all automation rules."""
    from core.models import AutomationRule
    rules = db.query(AutomationRule).filter(AutomationRule.trader_id == user.id).order_by(AutomationRule.created_at.desc()).all()
    return [
        AutomationRuleResponse(
            id=str(r.id), name=r.name, rule_type=str(r.rule_type),
            trigger_value=r.trigger_value, action=str(r.action),
            enabled=r.enabled, description=r.description,
            created_at=r.created_at.isoformat() if r.created_at else "",
            triggered_count=r.triggered_count or 0,
        )
        for r in rules
    ]


@router.post("/rules", status_code=201)
def create_rule(request: AutomationRuleRequest, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """Create a new automation rule."""
    from core.models import AutomationRule
    rule = AutomationRule(
        name=request.name,
        rule_type=request.rule_type,
        trigger_value=request.trigger_value,
        action=request.action,
        enabled=request.enabled,
        description=request.description,
        trader_id=user.id,
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    logger.info(f"Automation rule created: {rule.name} ({rule.rule_type})")
    return {"status": "created", "id": str(rule.id)}


@router.patch("/rules/{rule_id}/toggle")
def toggle_rule(rule_id: UUID, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """Enable or disable an automation rule."""
    from core.models import AutomationRule
    rule = db.query(AutomationRule).filter(AutomationRule.id == rule_id, AutomationRule.trader_id == user.id).first()
    if not rule:
        raise HTTPException(404, f"Rule {rule_id} not found")
    rule.enabled = not rule.enabled
    db.commit()
    return {"status": "ok", "enabled": rule.enabled}


@router.delete("/rules/{rule_id}", status_code=204)
def delete_rule(rule_id: UUID, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """Delete an automation rule."""
    from core.models import AutomationRule
    rule = db.query(AutomationRule).filter(AutomationRule.id == rule_id, AutomationRule.trader_id == user.id).first()
    if not rule:
        raise HTTPException(404, f"Rule {rule_id} not found")
    db.delete(rule)
    db.commit()


# ---------------------------------------------------------------------------
# Guardrail Status
# ---------------------------------------------------------------------------

@router.get("/guardrails/status", response_model=GuardrailStatusResponse)
def guardrail_status(db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """
    Check current guardrail status.
    Returns whether trading is allowed and what blocks are active.
    """
    from core.models import Trade, AutomationRule
    from core.enums import TradeStatus
    from datetime import date

    today = date.today()
    today_trades = db.query(Trade).filter(
        Trade.trader_id == user.id,
        Trade.entry_time >= datetime.combine(today, datetime.min.time()),
        Trade.status != TradeStatus.CANCELLED,
    ).all()

    daily_pnl = sum(t.pnl or 0 for t in today_trades if t.pnl is not None)
    daily_count = len(today_trades)

    active_blocks: list[dict] = []
    circuit_breaker_active = False

    # Check enabled rules
    rules = db.query(AutomationRule).filter(AutomationRule.trader_id == user.id, AutomationRule.enabled == True).all()
    trading_allowed = True

    for rule in rules:
        rt = str(rule.rule_type)
        triggered = False
        if rt == "daily_drawdown" and daily_pnl < -abs(rule.trigger_value):
            triggered = True
            active_blocks.append({
                "rule": rule.name, "type": rt,
                "message": f"Daily loss limit hit: ₹{abs(daily_pnl):.0f} > ₹{abs(rule.trigger_value):.0f}"
            })
        elif rt == "max_trades" and daily_count >= rule.trigger_value:
            triggered = True
            active_blocks.append({
                "rule": rule.name, "type": rt,
                "message": f"Max daily trades reached: {daily_count} of {int(rule.trigger_value)}"
            })

        if triggered and rule.action in ("stop", "pause"):
            trading_allowed = False
            if rule.action == "stop":
                circuit_breaker_active = True

    # Max daily trades from first matching rule
    max_trades_rule = next((r for r in rules if str(r.rule_type) == "max_trades"), None)
    max_trades = int(max_trades_rule.trigger_value) if max_trades_rule else None
    daily_loss_rule = next((r for r in rules if str(r.rule_type) == "daily_drawdown"), None)
    loss_limit = abs(daily_loss_rule.trigger_value) if daily_loss_rule else 0.0

    reset_at = datetime.combine(
        date.fromordinal(today.toordinal() + 1), datetime.min.time()
    ).isoformat() if not trading_allowed else None

    return GuardrailStatusResponse(
        trading_allowed=trading_allowed,
        active_blocks=active_blocks,
        daily_pnl=round(daily_pnl, 2),
        daily_loss_limit=round(loss_limit, 2),
        daily_trade_count=daily_count,
        max_daily_trades=max_trades,
        circuit_breaker_active=circuit_breaker_active,
        reset_at=reset_at,
    )


@router.post("/guardrails/circuit-breaker")
def activate_circuit_breaker(request: CircuitBreakerRequest, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """
    Manually activate the circuit breaker (trading halt).
    Useful when a trader wants to force a cooldown period.
    """
    from core.models import AutomationRule
    rule = AutomationRule(
        name=f"Manual circuit breaker — {request.reason[:50]}",
        rule_type="circuit_breaker",
        trigger_value=0,
        action="stop",
        enabled=True,
        description=f"Manual halt: {request.reason}. Duration: {request.duration_hours}h",
        trader_id=user.id,
    )
    db.add(rule)
    db.commit()
    logger.warning(f"Circuit breaker activated: {request.reason}")
    return {"status": "circuit_breaker_active", "duration_hours": request.duration_hours,
            "rule_id": str(rule.id)}


@router.delete("/guardrails/circuit-breaker/{rule_id}")
def deactivate_circuit_breaker(rule_id: UUID, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """Deactivate a manual circuit breaker."""
    from core.models import AutomationRule
    rule = db.query(AutomationRule).filter(
        AutomationRule.id == rule_id,
        AutomationRule.trader_id == user.id,
        AutomationRule.rule_type == "circuit_breaker",
    ).first()
    if not rule:
        raise HTTPException(404, "Circuit breaker rule not found")
    rule.enabled = False
    db.commit()
    logger.info(f"Circuit breaker {rule_id} deactivated")
    return {"status": "deactivated"}


# ---------------------------------------------------------------------------
# Strategy Evolution
# ---------------------------------------------------------------------------

@router.post("/strategy/evolve", response_model=StrategyEvolutionResponse)
def evolve_strategy(request: StrategyEvolutionRequest, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """
    Analyse a strategy's recent performance and suggest rule adjustments.
    Uses edge analysis and performance predictor to detect decay.
    """
    from datetime import timedelta
    from core.models import Trade
    from core.enums import TradeStatus
    from analytics.edge_analysis import EdgeAnalyzer

    cutoff = datetime.utcnow() - timedelta(days=request.lookback_days)
    trades = (
        db.query(Trade)
        .filter(
            Trade.trader_id == user.id,
            Trade.status == TradeStatus.CLOSED,
            Trade.entry_time >= cutoff,
        )
        .all()
    )

    # Filter by strategy
    strategy_trades = [
        t for t in trades
        if any(str(a.strategy_id) == str(request.strategy_id)
               for a in getattr(t, "strategy_attributions", []))
    ]

    if len(strategy_trades) < request.min_trades:
        return StrategyEvolutionResponse(
            strategy_id=str(request.strategy_id),
            current_win_rate=0.0,
            recommended_adjustments=[],
            edge_decay_detected=False,
            confidence=0.0,
            analysis_summary=(
                f"Insufficient data: {len(strategy_trades)} trades "
                f"(minimum {request.min_trades} required for evolution analysis)"
            ),
        )

    edge = EdgeAnalyzer().compute(strategy_trades)
    adjustments = []

    # Win rate recommendations
    if edge.rolling_win_rate_30d < edge.rolling_win_rate_90d * 0.85:
        adjustments.append({
            "type": "filter_tightening",
            "suggestion": "Tighten entry filters — recent win rate has dropped significantly",
            "priority": "high",
        })

    # Profit factor recommendations
    if edge.profit_factor < 1.2:
        adjustments.append({
            "type": "reward_risk",
            "suggestion": "Review reward:risk targets — profit factor below 1.2 is unsustainable",
            "priority": "high",
        })
    elif edge.profit_factor > 2.5:
        adjustments.append({
            "type": "sizing",
            "suggestion": "Strong edge detected — consider modest position size increase",
            "priority": "low",
        })

    if not adjustments:
        adjustments.append({
            "type": "maintain",
            "suggestion": "Strategy performing within normal parameters — no changes needed",
            "priority": "info",
        })

    summary = (
        f"Analysed {len(strategy_trades)} trades over {request.lookback_days} days. "
        f"Win rate: {edge.rolling_win_rate_90d:.0%} (90d) → {edge.rolling_win_rate_30d:.0%} (30d). "
        f"Profit factor: {edge.profit_factor:.2f}. "
        f"Edge decay: {'YES' if edge.edge_decay_detected else 'no'}."
    )

    return StrategyEvolutionResponse(
        strategy_id=str(request.strategy_id),
        current_win_rate=round(edge.rolling_win_rate_30d, 4),
        recommended_adjustments=adjustments,
        edge_decay_detected=edge.edge_decay_detected,
        confidence=min(len(strategy_trades) / 50, 1.0),
        analysis_summary=summary,
    )


# ---------------------------------------------------------------------------
# Setup Scanner control
# ---------------------------------------------------------------------------

class ScannerSymbolRequest(BaseModel):
    symbol: str


@router.get("/scanner/symbols")
def get_scanner_symbols():
    """List all symbols currently being scanned for setups."""
    from automation.setup_scanner import get_scanner_loop
    return {"symbols": get_scanner_loop().symbols}


@router.post("/scanner/symbols", status_code=201)
def add_scanner_symbol(request: ScannerSymbolRequest):
    """Add a symbol to the live setup scanner watchlist."""
    from automation.setup_scanner import get_scanner_loop
    get_scanner_loop().add_symbol(request.symbol.upper())
    return {"status": "added", "symbol": request.symbol.upper(),
            "symbols": get_scanner_loop().symbols}


@router.delete("/scanner/symbols/{symbol}")
def remove_scanner_symbol(symbol: str):
    """Remove a symbol from the live setup scanner watchlist."""
    from automation.setup_scanner import get_scanner_loop
    get_scanner_loop().remove_symbol(symbol.upper())
    return {"status": "removed", "symbol": symbol.upper(),
            "symbols": get_scanner_loop().symbols}


@router.get("/scanner/scan-now/{symbol}")
def scan_symbol_now(symbol: str):
    """
    Trigger an immediate on-demand scan for a single symbol.
    Returns detected setups without waiting for the next scheduled cycle.
    """
    from automation.setup_scanner import get_scanner_loop
    signals = get_scanner_loop().scan_now(symbol.upper())
    return {
        "symbol": symbol.upper(),
        "setups_found": len(signals),
        "signals": [s.to_dict() for s in signals],
    }


# ---------------------------------------------------------------------------
# Kill Switch (live order block)
# ---------------------------------------------------------------------------

class KillSwitchRequest(BaseModel):
    reason: str = Field("", description="Optional reason for activating the kill switch")


@router.post("/kill-switch", status_code=200)
def activate_kill_switch(request: KillSwitchRequest, db: DBSession = Depends(get_db)):
    """
    Activate the kill switch — blocks ALL live order submissions immediately.
    No orders will be sent to the broker until the kill switch is deactivated.

    Also achievable instantly without a DB call by setting env var PTIS_KILL_SWITCH=1.
    The DB flag persists across restarts; the env var is process-scoped only.
    """
    from core.models import SystemFlag
    from datetime import datetime, timezone

    flag = db.query(SystemFlag).filter(SystemFlag.key == "kill_switch").first()
    if flag:
        flag.value = "active"
        flag.reason = request.reason or flag.reason
        flag.updated_at = datetime.now(timezone.utc)
    else:
        flag = SystemFlag(key="kill_switch", value="active", reason=request.reason)
        db.add(flag)
    db.commit()
    logger.warning("Kill switch ACTIVATED via API. Reason: %s", request.reason or "(none)")
    return {
        "status": "kill_switch_active",
        "message": "All live order submissions are now blocked.",
        "reason": request.reason or None,
    }


@router.delete("/kill-switch", status_code=200)
def deactivate_kill_switch(db: DBSession = Depends(get_db)):
    """
    Deactivate the kill switch — resumes live order submissions.
    Has no effect if the kill switch was activated via env var (PTIS_KILL_SWITCH=1);
    that must be cleared in the environment and the process restarted.
    """
    from core.models import SystemFlag
    from datetime import datetime, timezone

    flag = db.query(SystemFlag).filter(SystemFlag.key == "kill_switch").first()
    if flag:
        flag.value = "inactive"
        flag.updated_at = datetime.now(timezone.utc)
        db.commit()
    logger.info("Kill switch DEACTIVATED via API.")
    return {
        "status": "kill_switch_inactive",
        "message": "Live order submissions are now allowed (DB flag cleared).",
        "note": "If PTIS_KILL_SWITCH=1 is set in the environment, orders are still blocked. Clear that env var and restart the process.",
    }


@router.get("/kill-switch/status", status_code=200)
def kill_switch_status(db: DBSession = Depends(get_db)):
    """Return the current kill switch state (env var + DB flag)."""
    import os
    from core.models import SystemFlag

    env_active = os.environ.get("PTIS_KILL_SWITCH", "0") == "1"
    flag = db.query(SystemFlag).filter(SystemFlag.key == "kill_switch").first()
    db_active = flag is not None and flag.value == "active"

    return {
        "kill_switch_active": env_active or db_active,
        "env_var_active": env_active,
        "db_flag_active": db_active,
        "db_flag_reason": flag.reason if flag else None,
        "db_flag_updated_at": flag.updated_at.isoformat() if flag and flag.updated_at else None,
    }
