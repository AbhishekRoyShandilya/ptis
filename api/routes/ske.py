# api/routes/ske.py
"""
Strategy Knowledge Engine API routes — CRUD for SKOs, signals, variants, DQAL.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from api.dependencies import get_db
from core.events import Event, EventType, get_event_bus
from core.models import StrategyKnowledgeObject, SKOVariant, SKOSignal
from ske.builder import SKOBuilder
from ske.dqal import DQALValidator
from ske.models import (
    BatchSignalRequest,
    DQALResult,
    ScoreRequest,
    SignalRequest,
    SKOCreateRequest,
    SKODetail,
    SKOSummary,
    SKOUpdateRequest,
    FeatureScore,
    ScoringParameter,
)
from ske.scoring import compute_conviction, compute_feature_score
from ske.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ske", tags=["strategy-knowledge-engine"], dependencies=[Depends(get_current_user)])


# ══════════════════════════════════════════════
#  SKO CRUD
# ══════════════════════════════════════════════

@router.post("/objects", response_model=SKOSummary, status_code=201)
async def create_sko(
    req: SKOCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new Strategy Knowledge Object."""
    sko = StrategyKnowledgeObject(
        id=uuid4(),
        name=req.name,
        short_code=req.short_code,
        description=req.description,
        tags=req.tags,
        scoring_config=req.scoring_config,
        conviction_map=req.conviction_map or {
            "skip": [0.0, 0.30],
            "small": [0.30, 0.50],
            "standard": [0.50, 0.70],
            "aggressive": [0.70, 0.85],
            "max": [0.85, 1.0],
        },
        entry_zones=req.entry_zones,
        exit_rules=req.exit_rules,
        risk_parameters=req.risk_parameters,
    )
    db.add(sko)
    await db.commit()
    await db.refresh(sko)

    bus = get_event_bus()
    bus.publish(Event(
        event_type=EventType.SKO_CREATED,
        source="api",
        data={"sko_id": str(sko.id), "name": sko.name},
    ))

    return SKOSummary.model_validate(sko)


@router.get("/objects", response_model=List[SKOSummary])
async def list_skos(
    status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """List all Strategy Knowledge Objects."""
    stmt = select(StrategyKnowledgeObject)
    if status:
        stmt = stmt.where(StrategyKnowledgeObject.status == status)
    stmt = stmt.order_by(StrategyKnowledgeObject.created_at.desc())

    result = await db.execute(stmt)
    skos = result.scalars().all()
    return [SKOSummary.model_validate(s) for s in skos]


@router.get("/objects/{sko_id}", response_model=SKODetail)
async def get_sko(
    sko_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get full SKO detail."""
    sko = await db.get(StrategyKnowledgeObject, sko_id)
    if not sko:
        raise HTTPException(404, "SKO not found")

    # Count variants and signals
    variant_count = await db.scalar(
        select(sa_func.count(SKOVariant.id)).where(SKOVariant.sko_id == sko_id)
    )
    signal_count = await db.scalar(
        select(sa_func.count(SKOSignal.id)).where(SKOSignal.sko_id == sko_id)
    )

    detail = SKODetail.model_validate(sko)
    detail.variant_count = variant_count or 0
    detail.signal_count = signal_count or 0
    return detail


@router.put("/objects/{sko_id}", response_model=SKOSummary)
async def update_sko(
    sko_id: UUID,
    req: SKOUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update SKO configuration."""
    sko = await db.get(StrategyKnowledgeObject, sko_id)
    if not sko:
        raise HTTPException(404, "SKO not found")

    for field, value in req.model_dump(exclude_unset=True).items():
        setattr(sko, field, value)

    sko.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(sko)

    bus = get_event_bus()
    bus.publish(Event(
        event_type=EventType.SKO_UPDATED,
        source="api",
        data={"sko_id": str(sko_id)},
    ))

    return SKOSummary.model_validate(sko)


@router.delete("/objects/{sko_id}", status_code=204)
async def archive_sko(
    sko_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Archive (soft-delete) an SKO."""
    sko = await db.get(StrategyKnowledgeObject, sko_id)
    if not sko:
        raise HTTPException(404, "SKO not found")

    sko.status = "archived"
    sko.updated_at = datetime.now(timezone.utc)
    await db.commit()


# ══════════════════════════════════════════════
#  BUILD & VALIDATE
# ══════════════════════════════════════════════

@router.post("/objects/{sko_id}/build")
async def build_sko(
    sko_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Build/rebuild SKO from trade history."""
    sko = await db.get(StrategyKnowledgeObject, sko_id)
    if not sko:
        raise HTTPException(404, "SKO not found")

    # For now, return the builder with empty trades — real integration
    # will query trades by declared_strategy_id matching sko.short_code
    builder = SKOBuilder()

    from core.models import Trade
    stmt = select(Trade).where(Trade.declared_strategy_id == sko.short_code)
    result = await db.execute(stmt)
    trades = result.scalars().all()

    trade_dicts = [
        {
            "id": str(t.id),
            "symbol": t.symbol,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_loss": t.stop_loss,
            "direction": t.direction.value if t.direction else "long",
            "r_multiple": t.r_multiple,
            "risk_pct": t.risk_pct,
            "conviction_score": t.conviction_score,
        }
        for t in trades
    ]

    built = builder.build_from_trades(
        trade_dicts,
        name=sko.name,
        short_code=sko.short_code,
        description=sko.description or "",
    )

    # Update SKO fields
    for field in [
        "status", "tier", "total_trades", "win_rate", "expectancy",
        "profit_factor", "scoring_config", "conviction_map",
        "exit_rules", "risk_parameters", "dqal_score", "dqal_grade",
    ]:
        if field in built:
            setattr(sko, field, built[field])

    sko.updated_at = datetime.now(timezone.utc)

    # Create variants
    for v_data in built.get("variants", []):
        variant = SKOVariant(
            id=uuid4(),
            sko_id=sko_id,
            variant_type="anti_variant" if v_data.get("is_anti_variant") else "variant",
            label=v_data.get("label", "unknown"),
            centroid=v_data.get("centroid", {}),
            trade_count=v_data.get("trade_count", 0),
            win_rate=v_data.get("win_rate"),
            avg_r=v_data.get("avg_r"),
            expectancy=v_data.get("expectancy"),
            distinguishing_features=v_data.get("distinguishing_features", []),
        )
        db.add(variant)

    await db.commit()
    await db.refresh(sko)

    return {
        "status": "built",
        "total_trades": built.get("total_trades", 0),
        "tier": built.get("status"),
        "dqal_grade": built.get("dqal_grade"),
        "variants_discovered": len(built.get("variants", [])),
    }


@router.post("/objects/{sko_id}/validate", response_model=DQALResult)
async def validate_sko(
    sko_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Run DQAL validation on an SKO."""
    sko = await db.get(StrategyKnowledgeObject, sko_id)
    if not sko:
        raise HTTPException(404, "SKO not found")

    validator = DQALValidator()
    sko_data = {
        "name": sko.name,
        "short_code": sko.short_code,
        "scoring_config": sko.scoring_config or {},
        "conviction_map": sko.conviction_map or {},
        "exit_rules": sko.exit_rules or {},
        "risk_parameters": sko.risk_parameters or {},
        "total_trades": sko.total_trades or 0,
        "status": sko.status.value if hasattr(sko.status, "value") else str(sko.status),
        "time_preferences": sko.time_preferences or {},
    }

    # Get trade outcomes for edge verification
    from core.models import Trade
    stmt = select(Trade.r_multiple).where(
        Trade.declared_strategy_id == sko.short_code,
        Trade.r_multiple.is_not(None),
    )
    result = await db.execute(stmt)
    outcomes = [row[0] for row in result.all()]

    dqal_result = validator.validate(sko_data, outcomes)

    # Update SKO with DQAL results
    sko.dqal_score = dqal_result.overall_score
    sko.dqal_grade = dqal_result.grade
    await db.commit()

    bus = get_event_bus()
    bus.publish(Event(
        event_type=EventType.DQAL_COMPLETED,
        source="api",
        data={"sko_id": str(sko_id), "grade": dqal_result.grade, "score": dqal_result.overall_score},
    ))

    return dqal_result


# ══════════════════════════════════════════════
#  VARIANTS
# ══════════════════════════════════════════════

@router.get("/objects/{sko_id}/variants")
async def list_variants(
    sko_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """List variants for an SKO."""
    stmt = select(SKOVariant).where(SKOVariant.sko_id == sko_id).order_by(SKOVariant.trade_count.desc())
    result = await db.execute(stmt)
    variants = result.scalars().all()

    return [
        {
            "id": str(v.id),
            "label": v.label,
            "variant_type": v.variant_type,
            "trade_count": v.trade_count,
            "win_rate": v.win_rate,
            "avg_r": v.avg_r,
            "expectancy": v.expectancy,
            "distinguishing_features": v.distinguishing_features,
            "is_active": v.is_active,
            "narrative": getattr(v, "narrative", None),
            "provenance": getattr(v, "provenance", "system"),
            "tags": getattr(v, "tags", []),
            "confidence_level": getattr(v, "confidence_level", 0.0),
            "scoring_overrides": getattr(v, "scoring_overrides", {}),
            "regime_affinity": getattr(v, "regime_affinity", None),
            "preferred_timeframes": getattr(v, "preferred_timeframes", []),
        }
        for v in variants
    ]


@router.post("/objects/{sko_id}/variants", status_code=201)
async def create_variant(
    sko_id: UUID,
    payload: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
):
    """
    Create a user-taught variant for an SKO.
    
    Body:
        label: str — variant name (e.g., "Morning Breakout Gap-Up")
        narrative: str — detailed description of when/why this variant works
        centroid: dict — feature centroid (optional, can be auto-learned)
        scoring_overrides: dict — per-variant scoring adjustments
        regime_affinity: dict — preferred regime fingerprint
        tags: list[str] — user labels
        preferred_timeframes: list[str] — e.g., ["5m", "15m"]
    """
    sko = await db.get(StrategyKnowledgeObject, sko_id)
    if not sko:
        raise HTTPException(404, "SKO not found")

    label = payload.get("label")
    if not label:
        raise HTTPException(400, "Variant label is required")

    variant = SKOVariant(
        id=uuid4(),
        sko_id=sko_id,
        variant_type="user_taught",
        label=label,
        centroid=payload.get("centroid", {}),
        trade_count=0,
        win_rate=None,
        avg_r=None,
        expectancy=None,
        distinguishing_features=payload.get("distinguishing_features", []),
        is_active=True,
    )

    # Store enrichment data in the variant's JSON fields
    if hasattr(variant, "narrative"):
        variant.narrative = payload.get("narrative", "")
    if hasattr(variant, "provenance"):
        variant.provenance = "user_taught"
    if hasattr(variant, "scoring_overrides"):
        variant.scoring_overrides = payload.get("scoring_overrides", {})
    if hasattr(variant, "regime_affinity"):
        variant.regime_affinity = payload.get("regime_affinity")
    if hasattr(variant, "tags"):
        variant.tags = payload.get("tags", [])
    if hasattr(variant, "preferred_timeframes"):
        variant.preferred_timeframes = payload.get("preferred_timeframes", [])
    if hasattr(variant, "confidence_level"):
        variant.confidence_level = 0.1  # low confidence until validated

    db.add(variant)
    await db.commit()
    await db.refresh(variant)

    logger.info(
        "Created user-taught variant '%s' for SKO %s",
        label, sko_id,
    )

    return {
        "id": str(variant.id),
        "label": variant.label,
        "provenance": "user_taught",
        "status": "created",
    }


@router.put("/objects/{sko_id}/variants/{variant_id}")
async def update_variant(
    sko_id: UUID,
    variant_id: UUID,
    payload: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
):
    """
    Update a variant's metadata (narrative, scoring overrides, regime affinity, etc.).
    """
    variant = await db.get(SKOVariant, variant_id)
    if not variant or variant.sko_id != sko_id:
        raise HTTPException(404, "Variant not found")

    updatable_fields = [
        "label", "narrative", "scoring_overrides", "regime_affinity",
        "tags", "preferred_timeframes", "centroid", "is_active",
    ]

    for field in updatable_fields:
        if field in payload and hasattr(variant, field):
            setattr(variant, field, payload[field])

    # If user refines a system variant, upgrade provenance
    if getattr(variant, "provenance", "system") == "system" and any(
        f in payload for f in ["narrative", "scoring_overrides", "regime_affinity"]
    ):
        if hasattr(variant, "provenance"):
            variant.provenance = "hybrid"

    await db.commit()
    await db.refresh(variant)

    return {
        "id": str(variant.id),
        "label": variant.label,
        "status": "updated",
    }


# ══════════════════════════════════════════════
#  SIGNALS
# ══════════════════════════════════════════════

@router.post("/objects/{sko_id}/signal")
async def generate_signal(
    sko_id: UUID,
    req: SignalRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate a signal for a symbol against an SKO."""
    sko = await db.get(StrategyKnowledgeObject, sko_id)
    if not sko:
        raise HTTPException(404, "SKO not found")

    if not req.candles:
        raise HTTPException(400, "Candles data required (candle fetcher not yet integrated)")

    generator = SignalGenerator()

    # Load variants
    stmt = select(SKOVariant).where(SKOVariant.sko_id == sko_id, SKOVariant.is_active == True)
    result = await db.execute(stmt)
    db_variants = result.scalars().all()

    from ske.models import VariantInfo, DistinguishingFeature, RegimeFingerprint
    variants = []
    for v in db_variants:
        regime_aff = None
        raw_regime = getattr(v, "regime_affinity", None)
        if raw_regime and isinstance(raw_regime, dict):
            try:
                regime_aff = RegimeFingerprint(**raw_regime)
            except Exception as e:
                logger.debug("Regime fingerprint parse failed: %s", e)

        variants.append(VariantInfo(
            label=v.label,
            centroid=v.centroid or {},
            trade_count=v.trade_count or 0,
            win_rate=v.win_rate or 0.0,
            avg_r=v.avg_r or 0.0,
            expectancy=v.expectancy or 0.0,
            is_anti_variant=(v.variant_type == "anti_variant"),
            narrative=getattr(v, "narrative", "") or "",
            regime_affinity=regime_aff,
            scoring_overrides=getattr(v, "scoring_overrides", {}) or {},
            provenance=getattr(v, "provenance", "system") or "system",
            tags=getattr(v, "tags", []) or [],
            preferred_timeframes=getattr(v, "preferred_timeframes", []) or [],
            confidence_level=getattr(v, "confidence_level", 0.0) or 0.0,
        ))

    sko_dict = {
        "id": sko.id,
        "name": sko.name,
        "scoring_config": sko.scoring_config or {},
        "conviction_map": sko.conviction_map or {},
        "entry_zones": sko.entry_zones or [],
        "exit_rules": sko.exit_rules or {},
        "risk_parameters": sko.risk_parameters or {},
        "regime_fingerprint": sko.regime_fingerprint or {},
    }

    signal = generator.evaluate(sko_dict, req.candles, req.symbol, variants)

    if signal is None:
        return {"status": "skip", "message": "Conviction too low — no signal generated"}

    # Persist signal
    db_signal = SKOSignal(
        id=uuid4(),
        sko_id=sko_id,
        symbol=signal.symbol,
        timestamp=signal.timestamp,
        feature_vector={fs.feature_name: fs.raw_value for fs in signal.individual_scores},
        individual_scores={fs.feature_name: fs.normalized_score for fs in signal.individual_scores},
        conviction_score=signal.conviction_score,
        conviction_band=signal.conviction_band.value,
        sizing_recommendation=signal.sizing_recommendation,
        size_multiplier=signal.size_multiplier,
        regime_fingerprint=signal.regime_fingerprint.model_dump() if signal.regime_fingerprint else {},
        regime_match_score=signal.regime_match_score,
        entry_zone=signal.entry_zone.model_dump() if signal.entry_zone else {},
        stop_level=signal.stop_level,
        target_levels=signal.target_levels,
    )
    db.add(db_signal)

    sko.last_signal_at = datetime.now(timezone.utc)
    await db.commit()

    bus = get_event_bus()
    bus.publish(Event(
        event_type=EventType.SKO_SIGNAL_GENERATED,
        source="api",
        data={
            "sko_id": str(sko_id),
            "symbol": signal.symbol,
            "conviction_band": signal.conviction_band.value,
            "conviction_score": signal.conviction_score,
        },
    ))

    return signal.model_dump()


@router.post("/objects/{sko_id}/signal/batch")
async def batch_signal(
    sko_id: UUID,
    req: BatchSignalRequest,
    db: AsyncSession = Depends(get_db),
):
    """Batch signal generation (requires candle fetcher integration)."""
    raise HTTPException(501, "Batch signal requires candle fetcher — not yet integrated")


@router.get("/signals")
async def list_signals(
    status: Optional[str] = Query(None),
    band: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List recent signals."""
    stmt = select(SKOSignal).order_by(SKOSignal.created_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(SKOSignal.status == status)
    if band:
        stmt = stmt.where(SKOSignal.conviction_band == band)

    result = await db.execute(stmt)
    signals = result.scalars().all()

    return [
        {
            "id": str(s.id),
            "sko_id": str(s.sko_id),
            "symbol": s.symbol,
            "conviction_score": s.conviction_score,
            "conviction_band": s.conviction_band,
            "sizing_recommendation": s.sizing_recommendation,
            "entry_zone": s.entry_zone,
            "stop_level": s.stop_level,
            "target_levels": s.target_levels,
            "status": s.status,
            "outcome_r": s.outcome_r,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in signals
    ]


@router.put("/signals/{signal_id}/outcome")
async def record_outcome(
    signal_id: UUID,
    outcome_r: float = Query(...),
    status: str = Query("taken"),
    db: AsyncSession = Depends(get_db),
):
    """Record the outcome of a signal."""
    signal = await db.get(SKOSignal, signal_id)
    if not signal:
        raise HTTPException(404, "Signal not found")

    signal.outcome_r = outcome_r
    signal.status = status
    await db.commit()

    return {"status": "updated", "signal_id": str(signal_id), "outcome_r": outcome_r}


# ══════════════════════════════════════════════
#  AD-HOC SCORING
# ══════════════════════════════════════════════

@router.post("/score")
async def adhoc_score(req: ScoreRequest):
    """Ad-hoc scoring: pass feature values + scoring params → get conviction."""
    feature_scores = []
    for feat_name, param_dict in req.scoring_config.items():
        raw_value = req.feature_values.get(feat_name)
        if raw_value is None:
            continue
        param = ScoringParameter(**param_dict)
        score = compute_feature_score(raw_value, param)
        feature_scores.append(FeatureScore(
            feature_name=feat_name,
            raw_value=raw_value,
            normalized_score=score,
            weight=param.weight,
        ))

    conviction = compute_conviction(feature_scores, req.conviction_map)
    return conviction.model_dump()


# ══════════════════════════════════════════════
#  REGIME MATCH
# ══════════════════════════════════════════════

@router.get("/objects/{sko_id}/regime-match")
async def regime_match(
    sko_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get current regime match score for an SKO (placeholder — needs live candles)."""
    sko = await db.get(StrategyKnowledgeObject, sko_id)
    if not sko:
        raise HTTPException(404, "SKO not found")

    regime_fp = sko.regime_fingerprint or {}
    preferred = regime_fp.get("preferred", [])

    if not preferred:
        return {"regime_match_score": 0.5, "message": "No regime preferences configured"}

    return {
        "regime_match_score": 0.5,
        "message": "Regime match requires live candle data — returning neutral",
        "preferred_regimes": preferred,
    }
