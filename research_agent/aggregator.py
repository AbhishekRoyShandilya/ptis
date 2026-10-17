"""
Research Aggregator
====================

Persists individual findings (from Phase 5 detectors + Phase 4 explorer outputs)
to the research_findings table, then computes META findings via cross-correlation.

Cross-correlation is where real value compounds. Examples:
  - "3+ strategies decaying simultaneously" → it's not your strategies, it's a regime shift
  - "2+ strategies show reversal cliff at same time slot" → it's not the entries, it's the time

Real-money safety:
  - All findings written with status='pending_review'
  - Trader must explicitly accept via /research page before any action
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from research_agent.models import DecayFinding, EmergenceFinding

logger = logging.getLogger(__name__)


# ── Meta-finding thresholds ────────────────────────────────────────

META_REGIME_SHIFT_MIN_DECAY: int = 3      # ≥3 strategies decaying = regime shift
META_TIME_SLOT_MIN_REPEATS: int = 2       # ≥2 strategies share a cliff time slot


# ── Persistence ────────────────────────────────────────────────────

def _build_finding_id(prefix: str, *parts: str) -> str:
    """Stable ID from a prefix + key parts (for upserts to avoid duplicates)."""
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"


def persist_decay_finding(db, finding: DecayFinding) -> str:
    """
    Insert (or upsert by id) a DecayFinding into research_findings.

    Returns the row id. Idempotent — repeated calls with same finding update
    instead of duplicating.
    """
    from core.models import ResearchFinding

    fid = _build_finding_id(
        "decay",
        finding.strategy_id,
        finding.detected_at.strftime("%Y%m%d"),
    )

    title = f"Strategy decay detected: {finding.strategy_id}"
    summary = (
        f"Recent {finding.recent_n} trades avg R = {finding.recent_avg_r:.2f} "
        f"vs baseline {finding.baseline_n} trades avg R = {finding.baseline_avg_r:.2f}. "
        f"Drift = {finding.drift_magnitude:+.2f}R."
    )

    existing = db.query(ResearchFinding).filter_by(id=fid).first()
    if existing:
        # Update payload (statistics may have refined since last run)
        existing.payload = finding.to_dict()
        existing.confidence = finding.confidence.value
        existing.p_value = finding.p_value
        existing.sample_size = finding.recent_n
        existing.summary = summary
        existing.rationale = finding.rationale
        existing.suggested_action = finding.suggested_action
        existing.updated_at = datetime.now()
    else:
        row = ResearchFinding(
            id=fid,
            type="decay_detected",
            strategy_id=finding.strategy_id,
            symbol=None,
            detected_at=finding.detected_at,
            confidence=finding.confidence.value,
            p_value=finding.p_value,
            sample_size=finding.recent_n,
            title=title,
            summary=summary,
            rationale=finding.rationale,
            suggested_action=finding.suggested_action,
            payload=finding.to_dict(),
            status="pending_review",
            status_history=[],
            related_finding_ids=None,
        )
        db.add(row)

    return fid


def persist_emergence_finding(db, finding: EmergenceFinding) -> str:
    """Insert (or upsert) an EmergenceFinding into research_findings."""
    from core.models import ResearchFinding

    fid = _build_finding_id(
        "emerge",
        finding.strategy_id,
        finding.preferred_regime,
        finding.detected_at.strftime("%Y%m%d"),
    )

    title = f"Edge emerging: {finding.strategy_id} ({finding.preferred_regime})"
    summary = (
        f"Recent {finding.recent_n} trades WR = {finding.win_rate_recent:.0%} "
        f"(baseline {finding.win_rate_baseline:.0%}, +{finding.win_rate_change:.0%}). "
        f"Regime '{finding.preferred_regime}' frequency {finding.regime_pct_baseline:.0%} "
        f"→ {finding.regime_pct_recent:.0%}."
    )

    existing = db.query(ResearchFinding).filter_by(id=fid).first()
    if existing:
        existing.payload = finding.to_dict()
        existing.sample_size = finding.recent_n
        existing.summary = summary
        existing.rationale = finding.rationale
        existing.suggested_action = finding.suggested_action
        existing.updated_at = datetime.now()
    else:
        row = ResearchFinding(
            id=fid,
            type="edge_emerging",
            strategy_id=finding.strategy_id,
            symbol=None,
            detected_at=finding.detected_at,
            confidence="moderate",   # emergence is always moderate (no p-value gate)
            p_value=None,
            sample_size=finding.recent_n,
            title=title,
            summary=summary,
            rationale=finding.rationale,
            suggested_action=finding.suggested_action,
            payload=finding.to_dict(),
            status="pending_review",
            status_history=[],
            related_finding_ids=None,
        )
        db.add(row)

    return fid


# ── Cross-correlation meta-findings ────────────────────────────────

def cross_correlate(
    db,
    decay_findings: List[DecayFinding],
    emergence_findings: List[EmergenceFinding],
) -> List[str]:
    """
    Compute meta-findings from individual findings. Persists them to DB.
    Returns the list of created meta-finding IDs.

    These are HIGHER VALUE than individual findings — they tell the trader
    "it's not your strategy, it's the market".
    """
    from core.models import ResearchFinding

    meta_ids: List[str] = []
    today = datetime.now()

    # ── Meta 1: Regime shift detected (≥3 strategies decaying simultaneously) ──
    if len(decay_findings) >= META_REGIME_SHIFT_MIN_DECAY:
        related = [
            _build_finding_id("decay", f.strategy_id, f.detected_at.strftime("%Y%m%d"))
            for f in decay_findings
        ]
        fid = _build_finding_id("meta_regime_shift", today.strftime("%Y%m%d"))
        title = "META: Possible regime shift detected"
        summary = (
            f"{len(decay_findings)} strategies are decaying simultaneously: "
            f"{', '.join(f.strategy_id for f in decay_findings)}. "
            f"This pattern usually means the MARKET STRUCTURE has shifted, not the strategies."
        )
        action = (
            "Review all active strategies. Consider reducing overall sizing until you "
            "identify the new dominant regime. Check the emergence findings for clues "
            "about which strategies might be coming online."
        )
        existing = db.query(ResearchFinding).filter_by(id=fid).first()
        if not existing:
            row = ResearchFinding(
                id=fid,
                type="regime_shift",
                strategy_id=None,
                symbol=None,
                detected_at=today,
                confidence="moderate",
                p_value=None,
                sample_size=len(decay_findings),
                title=title,
                summary=summary,
                rationale=f"Decaying strategies: {', '.join(f.strategy_id for f in decay_findings)}",
                suggested_action=action,
                payload={
                    "n_decaying": len(decay_findings),
                    "decaying_strategies": [f.strategy_id for f in decay_findings],
                },
                status="pending_review",
                status_history=[],
                related_finding_ids=related,
            )
            db.add(row)
            meta_ids.append(fid)
            logger.info("Meta-finding: regime_shift (%d strategies decaying)", len(decay_findings))

    # ── Meta 2: Time slot risk (≥2 strategies share a cliff time slot) ──
    # (This needs reversal_cliff findings from Phase 2 — for now, scaffold the logic)
    cliff_findings = (
        db.query(ResearchFinding)
        .filter(ResearchFinding.type == "reversal_cliff")
        .filter(ResearchFinding.status == "pending_review")
        .all()
    )
    time_slots: Counter[str] = Counter()
    cliff_ids_by_slot: Dict[str, List[str]] = {}
    for f in cliff_findings:
        slot = (f.payload or {}).get("peak_reversal_time_slot")
        if slot:
            time_slots[slot] += 1
            cliff_ids_by_slot.setdefault(slot, []).append(f.id)

    for slot, count in time_slots.items():
        if count < META_TIME_SLOT_MIN_REPEATS:
            continue
        fid = _build_finding_id("meta_time_slot", slot, today.strftime("%Y%m%d"))
        title = f"META: Time slot risk ({slot})"
        summary = (
            f"{count} strategies show reversal cliffs at the {slot} time slot. "
            f"This is a TIME-OF-DAY risk, not a strategy-specific issue."
        )
        action = (
            f"Avoid holding through {slot} or tighten stops as the slot approaches. "
            "Consider scheduling exit logic to fire at the start of this window."
        )
        existing = db.query(ResearchFinding).filter_by(id=fid).first()
        if not existing:
            row = ResearchFinding(
                id=fid,
                type="time_slot_risk",
                strategy_id=None,
                symbol=None,
                detected_at=today,
                confidence="moderate",
                p_value=None,
                sample_size=count,
                title=title,
                summary=summary,
                rationale=f"Cliff reports at {slot}: {count}",
                suggested_action=action,
                payload={"time_slot": slot, "n_strategies_affected": count},
                status="pending_review",
                status_history=[],
                related_finding_ids=cliff_ids_by_slot.get(slot, []),
            )
            db.add(row)
            meta_ids.append(fid)
            logger.info("Meta-finding: time_slot_risk @ %s (%d strategies)", slot, count)

    return meta_ids


# ── Run-all helper ─────────────────────────────────────────────────

def aggregate_and_persist(
    db,
    decay_findings: List[DecayFinding],
    emergence_findings: List[EmergenceFinding],
    commit: bool = True,
) -> Dict[str, Any]:
    """
    Persist all individual findings + run cross-correlation. One-call API for the scheduler.

    Args:
        db: SQLAlchemy session
        decay_findings: from DecayDetector.detect_all()
        emergence_findings: from EmergenceDetector.detect_all()
        commit: if True, commits the transaction. If False, caller commits.

    Returns: {n_decay, n_emergence, n_meta, finding_ids: [...]}
    """
    decay_ids: List[str] = []
    emerge_ids: List[str] = []

    for df in decay_findings:
        try:
            decay_ids.append(persist_decay_finding(db, df))
        except Exception as e:
            logger.warning("Failed to persist decay finding for %s: %s", df.strategy_id, e)

    for ef in emergence_findings:
        try:
            emerge_ids.append(persist_emergence_finding(db, ef))
        except Exception as e:
            logger.warning("Failed to persist emergence finding for %s: %s", ef.strategy_id, e)

    # Flush so meta-correlation can see the new rows
    try:
        db.flush()
    except Exception as e:
        logger.warning("DB flush failed before cross-correlation: %s", e)

    meta_ids = cross_correlate(db, decay_findings, emergence_findings)

    if commit:
        try:
            db.commit()
        except Exception as e:
            logger.error("DB commit failed in aggregator: %s", e)
            db.rollback()

    return {
        "n_decay": len(decay_ids),
        "n_emergence": len(emerge_ids),
        "n_meta": len(meta_ids),
        "decay_ids": decay_ids,
        "emergence_ids": emerge_ids,
        "meta_ids": meta_ids,
    }
