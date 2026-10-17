# strategies.py
# api/routes/strategies.py
"""
Strategy DNA endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from collections import defaultdict

from api.auth import get_current_user
from typing import Optional

from core.database import get_db_dependency
from api.dependencies import get_query_engine, get_dna_loader

router = APIRouter(prefix="/strategies", tags=["strategies"], dependencies=[Depends(get_current_user)])


@router.get("/edge-matrix")
def get_edge_matrix(
    strategy_code: Optional[str] = None,
    db: Session = Depends(get_db_dependency),
    user=Depends(get_current_user),
):
    """
    Edge-decay matrix for the caller: per (strategy, context) win_rate / avg_r /
    expectancy + edge_trend (strengthening | weakening | stable). Aggregated EOD
    from the CLEAN AVT strategy-research record (perfect-discipline executions) —
    NOT from raw user trades, which are contaminated by behavioural noise.
    context_key e.g. 'regime:breakout', 'vol:high'.
    """
    from core.models import StrategyContextPerformance
    q = db.query(StrategyContextPerformance).filter(
        StrategyContextPerformance.trader_id == str(user.id)
    )
    if strategy_code:
        q = q.filter(StrategyContextPerformance.strategy_code == strategy_code)
    rows = q.order_by(
        StrategyContextPerformance.strategy_code, StrategyContextPerformance.context_key
    ).all()
    return [
        {
            "strategy_code": r.strategy_code,
            "context_key": r.context_key,
            "trade_count": r.trade_count,
            "win_rate": r.win_rate,
            "avg_r": r.avg_r,
            "expectancy": r.expectancy,
            "last_n_avg_r": r.last_n_avg_r,
            "edge_trend": r.edge_trend,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]


@router.get("/symbol-performance")
def get_symbol_performance(
    strategy_code: Optional[str] = None,
    symbol: Optional[str] = None,
    regime: Optional[str] = None,
    db: Session = Depends(get_db_dependency),
    user=Depends(get_current_user),
):
    """
    Per-(strategy, stock[, regime]) ranking baseline for the caller: win_rate /
    avg_r / expectancy + a confidence layer. Aggregated from the CLEAN benchmark
    replay (marketdata/strategy_research) — NOT user trades. This is what seeds
    opportunity ranking for a freshly-taught strategy.

    Ordered by CONFIDENCE-WEIGHTED expectancy (expectancy × confidence_score) so a
    thin 3-trade/100% cell never outranks a deep 150-trade/58% cell.
    """
    from core.models import StrategySymbolPerformance
    q = db.query(StrategySymbolPerformance).filter(
        StrategySymbolPerformance.trader_id == str(user.id)
    )
    if strategy_code:
        q = q.filter(StrategySymbolPerformance.strategy_code == strategy_code)
    if symbol:
        q = q.filter(StrategySymbolPerformance.symbol == symbol)
    # regime filter: default to the symbol-overall rows ('*') unless asked otherwise
    q = q.filter(StrategySymbolPerformance.regime == (regime or "*"))
    rows = q.all()
    out = [
        {
            "strategy_code": r.strategy_code,
            "symbol": r.symbol,
            "regime": r.regime,
            "market_state": r.market_state,
            "strategy_version": r.strategy_version,
            "trade_count": r.trade_count,
            "win_rate": r.win_rate,
            "avg_r": r.avg_r,
            "expectancy": r.expectancy,
            "last_n_avg_r": r.last_n_avg_r,
            "effective_sample_size": r.effective_sample_size,
            "confidence_score": r.confidence_score,
            "ranking_score": round((r.expectancy or 0.0) * (r.confidence_score or 0.0), 4),
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]
    out.sort(key=lambda x: x["ranking_score"], reverse=True)
    return out


@router.get("")
def list_strategies(
    db: Session = Depends(get_db_dependency),
    user=Depends(get_current_user),
):
    """
    List the CALLER'S OWN strategies — only the ones they explicitly taught and
    approved (Bootstrap finalize or Learning), scoped to their trader_id and
    status == 'active'.

    Deliberately does NOT return the global built-in DNA template library. A
    strategy only becomes "mine" once the user approves it, because these drive
    AVT stock selection and behavioral discipline scoring — surfacing templates
    the user never chose would corrupt both (and grade them "undisciplined" for
    not following strategies they never adopted).
    """
    from core.models import StrategyProfile, Trade
    from core.enums import TradeStatus

    profiles = (
        db.query(StrategyProfile)
        .filter(
            StrategyProfile.trader_id == user.id,
            StrategyProfile.status == "active",
        )
        .order_by(StrategyProfile.created_at.desc())
        .all()
    )

    if not profiles:
        return []

    # Aggregate live metrics from the user's own closed trades, grouped by the
    # strategy they were attributed to (declared_strategy_id == profile short_code).
    closed = (
        db.query(Trade)
        .filter(Trade.trader_id == user.id, Trade.status == TradeStatus.CLOSED)
        .all()
    )
    by_code: dict[str, list] = defaultdict(list)
    for t in closed:
        if t.declared_strategy_id:
            by_code[t.declared_strategy_id].append(t)

    def _metrics(code: str, name: str) -> dict:
        rows = by_code.get(code) or by_code.get(name) or []
        n = len(rows)
        if n == 0:
            return {"pnl": 0, "win_rate": 0, "avg_r": 0, "trades": 0}
        wins = sum(1 for t in rows if (t.pnl or 0) > 0)
        rs = [t.r_multiple for t in rows if t.r_multiple is not None]
        return {
            "pnl": round(sum(t.pnl or 0 for t in rows), 2),
            "win_rate": round(wins / n * 100, 1),
            "avg_r": round(sum(rs) / len(rs), 2) if rs else 0,
            "trades": n,
        }

    out = []
    for p in profiles:
        regimes = [
            r.get("regime") for r in (p.regime_preferences or {}).get("preferred", [])
            if isinstance(r, dict) and r.get("regime")
        ]
        conditions = [
            c.get("criterion") for c in (p.entry_criteria or [])
            if isinstance(c, dict) and c.get("criterion")
        ]
        out.append({
            "id": str(p.id),
            "name": p.name,
            "display_name": p.name,
            "short_code": p.short_code,
            "description": p.narrative or "",
            "status": p.status,
            "regimes": regimes,
            "conditions": conditions,
            "metrics": _metrics(p.short_code, p.name),
            "tags": p.tags or [],
        })
    return out


# ── SYSTEM strategy templates (the system's OWN strategies — cascade futures/options — the user SELECTS, not
#    teaches). Defined BEFORE /{strategy_name} so "templates" isn't captured by that catch-all. ──

@router.get("/templates")
def list_system_templates(user=Depends(get_current_user)):
    """Library of system templates the user can one-click activate + which they've already activated."""
    from avt.templates import list_templates
    from avt.template_store import activated
    act = set(activated(str(user.id)))
    return {"templates": [{**t, "activated": t["id"] in act} for t in list_templates()]}


@router.post("/templates/{template_id}/activate")
def activate_system_template(template_id: str, user=Depends(get_current_user)):
    """Activate a system template for this user → its SKOs load into their AVT on (re)start, routed to the
    cascade generator with real F&O charges. Idempotent."""
    from avt.template_store import activate
    try:
        act = activate(str(user.id), template_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown system template: {template_id}")
    return {"activated": act, "message": f"'{template_id}' activated — (re)start your AVT to trade it."}


@router.delete("/templates/{template_id}")
def deactivate_system_template(template_id: str, user=Depends(get_current_user)):
    """Deactivate a system template for this user (applies on next AVT start)."""
    from avt.template_store import deactivate
    act = deactivate(str(user.id), template_id)
    return {"activated": act, "message": f"'{template_id}' deactivated — (re)start your AVT to apply."}


@router.get("/{strategy_name}")
def get_strategy_detail(strategy_name: str):
    """Get full strategy DNA for a strategy."""
    query = get_query_engine()
    dna = query.get_strategy(strategy_name)

    if not dna:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")

    return {
        "name": dna.name,
        "display_name": dna.display_name,
        "short_code": dna.short_code,
        "category": dna.category,
        "description": dna.description,
        "timeframe": dna.timeframe_primary,
        "typical_hold": dna.typical_hold,

        "entry_logic": {
            "narrative": dna.entry_narrative,
            "conditions": [
                {
                    "name": c.name,
                    "description": c.description,
                    "importance": c.importance,
                    "what_good_looks_like": c.what_good_looks_like,
                    "what_bad_looks_like": c.what_bad_looks_like,
                }
                for c in dna.setup_conditions
            ],
            "a_plus_conditions": dna.a_plus_conditions,
            "skip_conditions": dna.skip_conditions,
        },

        "execution": {
            "entry_trigger": dna.entry_trigger,
            "entry_timing": dna.entry_timing,
            "common_mistakes": dna.common_entry_mistakes,
        },

        "risk": {
            "stop_method": dna.stop_method,
            "risk_per_trade": dna.risk_per_trade,
            "target_primary": dna.target_primary,
            "target_secondary": dna.target_secondary,
            "size_adjustments": dna.size_adjustments,
        },

        "market_conditions": {
            "best_regimes": [
                {"regime": r.regime, "notes": r.notes, "win_rate": r.expected_win_rate}
                for r in dna.market_conditions.works_best_regimes
            ],
            "worst_regimes": [
                {"regime": r.regime, "notes": r.notes, "recommendation": r.recommendation}
                for r in dna.market_conditions.works_poorly_regimes
            ],
            "regime_sizing": dna.market_conditions.regime_sizing,
        },

        "psychology": {
            "emotional_traps": [
                {"trap": t.trap, "description": t.description,
                 "solution": t.solution, "frequency": t.frequency}
                for t in dna.emotional_traps
            ],
            "ideal_mental_state": dna.ideal_mental_state,
            "do_not_trade_when": dna.do_not_trade_when,
        },

        "relationships": {
            "fallbacks": dna.relationships.fallback_strategies,
            "conflicts": dna.relationships.conflicts_with,
            "pairs_with": dna.relationships.pairs_well_with,
            "regime_alternatives": dna.relationships.regime_alternatives,
        },

        "personal_insights": {
            "strengths": dna.personal_insights.strengths,
            "weaknesses": dna.personal_insights.weaknesses,
            "edge_conditions": dna.personal_insights.edge_conditions,
            "learned_statistics": dna.personal_insights.learned_statistics,
        },

        "checklist": dna.checklist,
        "tags": dna.tags,
    }


@router.get("/{strategy_name}/checklist")
def get_strategy_checklist(strategy_name: str):
    """Get checklist for trade entry form."""
    query = get_query_engine()
    dna = query.get_strategy(strategy_name)

    if not dna:
        return {"checklist": [], "strategy_name": strategy_name}

    return {
        "strategy_name": dna.name,
        "display_name": dna.display_name,
        "checklist": [
            {"index": i, "condition": c}
            for i, c in enumerate(dna.checklist)
        ],
        "total_conditions": len(dna.checklist),
    }


@router.get("/{strategy_name}/guidance")
def get_pre_trade_guidance(
    strategy_name: str,
    regime: str = "unknown",
    emotion: Optional[str] = None,
    db: Session = Depends(get_db_dependency),
    user=Depends(get_current_user),
):
    """Get pre-trade guidance before entering a trade."""
    query = get_query_engine()

    # Count today's state
    from datetime import datetime
    today = datetime.combine(datetime.now().date(), datetime.min.time())
    from core.models import Trade
    from core.enums import TradeStatus
    trades_today = db.query(Trade).filter(Trade.trader_id == user.id, Trade.entry_time >= today).count()

    recent = db.query(Trade).filter(
        Trade.trader_id == user.id,
        Trade.status == TradeStatus.CLOSED,
    ).order_by(Trade.entry_time.desc()).limit(10).all()

    consecutive_losses = 0
    for t in recent:
        if t.pnl and t.pnl < 0:
            consecutive_losses += 1
        else:
            break

    guidance = query.get_pre_trade_guidance(
        strategy_name=strategy_name,
        current_regime=regime,
        trader_emotion=emotion,
        consecutive_losses=consecutive_losses,
        trades_today=trades_today,
    )

    return {
        "strategy_name": guidance.strategy_name,
        "regime_fitness": guidance.regime_fitness,
        "sizing_recommendation": guidance.sizing_recommendation,
        "checklist": [
            {"index": i, "condition": c}
            for i, c in enumerate(guidance.checklist)
        ],
        "warnings": guidance.key_warnings,
        "traps_to_watch": guidance.emotional_traps_to_watch,
        "skip_conditions": guidance.skip_conditions_active,
        "coaching_message": guidance.coaching_message,
    }


@router.get("/regime/{regime_name}/recommendations")
def get_regime_recommendations(regime_name: str):
    """Get strategy recommendations for a market regime."""
    query = get_query_engine()
    recs = query.get_strategies_for_regime(regime_name)

    return {
        "regime": regime_name,
        "recommendations": [
            {
                "strategy": r.strategy_name,
                "display_name": r.display_name,
                "fitness_score": r.fitness_score,
                "sizing": r.sizing_recommendation,
                "notes": r.notes,
                "warnings": r.warnings,
                "alternative": r.alternative,
            }
            for r in recs
        ],
    }


@router.get("/{strategy_name}/performance")
def get_strategy_performance(
    strategy_name: str,
    days: int = 90,
    db: Session = Depends(get_db_dependency),
    user=Depends(get_current_user),
):
    """Get learned performance data for a strategy."""
    from strategy_engine.strategy_dna.learning_updater import StrategyDNALearningUpdater

    loader = get_dna_loader()
    updater = StrategyDNALearningUpdater(db, loader)
    report = updater.generate_learning_report(strategy_name, lookback_days=days)

    return report