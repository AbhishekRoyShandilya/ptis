# avt/repository.py
"""
AVT Repository — Database access layer.

All DB reads/writes for AVT tables go through here.
The engine, self_improver, and competition modules use this
instead of raw SQL or in-memory lists for persistence.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import desc, and_
from sqlalchemy.orm import Session

from avt.db_models import (
    AVTCompetitionSnapshotRow,
    AVTImprovementRow,
    AVTScanLogRow,
    AVTStrategyVersionRow,
    AVTVirtualTradeRow,
)
from avt.models import (
    ActiveImprovement,
    CompetitionReport,
    PerformanceDifference,
    ScanCycleLog,
    VirtualTrade,
)

logger = logging.getLogger(__name__)


def _to_float(v: Any) -> Optional[float]:
    """Convert Decimal/None to float."""
    if v is None:
        return None
    return float(v)


# ══════════════════════════════════════════════════════════════════
#  VIRTUAL TRADE REPOSITORY
# ══════════════════════════════════════════════════════════════════

class AVTTradeRepository:
    """CRUD for avt_virtual_trades."""

    def __init__(self, db: Session):
        self.db = db

    def save(self, trade: VirtualTrade) -> None:
        """Insert or update a virtual trade."""
        row = self.db.get(AVTVirtualTradeRow, trade.id)
        if row is None:
            row = AVTVirtualTradeRow(id=trade.id)
            self.db.add(row)

        if trade.trader_id is not None:
            row.trader_id = trade.trader_id
        row.strategy_id = trade.strategy_id
        row.strategy_version = trade.strategy_version
        row.variant_id = trade.variant_id
        row.symbol = trade.symbol
        row.direction = trade.direction
        row.entry_time = trade.entry_time
        row.entry_price = Decimal(str(trade.entry_price))
        row.quantity = trade.quantity
        row.stop_price = Decimal(str(trade.stop_price))
        row.conviction_at_entry = (
            Decimal(str(trade.conviction_at_entry)) if trade.conviction_at_entry else None
        )
        row.feature_vector = trade.feature_vector_at_entry
        row.regime_at_entry = trade.regime_at_entry
        row.ranking_data = {
            "position": trade.ranking_position,
            "score": trade.ranking_score,
            "explanation": trade.ranking_explanation,
            "competing": trade.competing_setups_count,
        }
        row.max_favorable_r = Decimal(str(trade.max_favorable_r))
        row.features_at_peak = trade.features_at_peak
        row.exit_time = trade.exit_time
        row.exit_price = Decimal(str(trade.exit_price)) if trade.exit_price else None
        row.exit_type = trade.exit_type
        # Map the domain VirtualTrade's pnl fields to the DB columns (domain: realized_pnl / r_multiple / exit_type;
        # DB: pnl_absolute / pnl_r / exit_reason). getattr fallbacks keep it robust to either field set.
        row.exit_reason = getattr(trade, "exit_reason", None) or trade.exit_type
        _pnl_r = getattr(trade, "pnl_r", None)
        _pnl_r = _pnl_r if _pnl_r is not None else getattr(trade, "r_multiple", None)
        row.pnl_r = Decimal(str(_pnl_r)) if _pnl_r is not None else None
        _pnl_pct = getattr(trade, "pnl_pct", None)
        row.pnl_pct = Decimal(str(_pnl_pct)) if _pnl_pct is not None else None
        _pnl_abs = getattr(trade, "pnl_absolute", None)
        _pnl_abs = _pnl_abs if _pnl_abs is not None else getattr(trade, "realized_pnl", None)
        row.pnl_absolute = Decimal(str(_pnl_abs)) if _pnl_abs is not None else None
        row.active_improvements = [uid for uid in (trade.active_improvements or [])]
        row.improvement_exits = getattr(trade, "improvement_exits", None) or getattr(trade, "partial_exits", None) or []
        _st = getattr(trade, "status", "open")
        row.status = _st.value if hasattr(_st, "value") else str(_st)   # enum -> "open", not "VirtualTradeStatus.OPEN"

        # Options (M3b): carry option metadata so the dashboard renders them correctly.
        if getattr(trade, "instrument", "future") == "option":
            row.instrument_meta = {
                "instrument": "option", "tradeable": trade.tradeable,
                "strike": trade.option_strike, "expiry": trade.option_expiry, "right": trade.option_right,
                "lots": trade.option_lots, "lot_size": trade.option_lot_size,
                "underlying_stop": trade.underlying_stop, "entry_premium": float(trade.entry_price),
            }

        self.db.flush()

    def get_by_id(self, trade_id: UUID) -> Optional[VirtualTrade]:
        """Load a single trade."""
        row = self.db.get(AVTVirtualTradeRow, trade_id)
        return self._to_model(row) if row else None

    def get_open_trades(
        self, strategy_id: Optional[UUID] = None, trader_id: Optional[str] = None,
    ) -> List[VirtualTrade]:
        """All currently open virtual trades (optionally scoped to one trader)."""
        q = self.db.query(AVTVirtualTradeRow).filter(
            AVTVirtualTradeRow.status == "open"
        )
        if strategy_id:
            q = q.filter(AVTVirtualTradeRow.strategy_id == strategy_id)
        if trader_id is not None:
            q = q.filter(AVTVirtualTradeRow.trader_id == trader_id)
        return [self._to_model(r) for r in q.all()]

    def get_closed_trades(
        self,
        strategy_id: Optional[UUID] = None,
        since: Optional[datetime] = None,
        limit: int = 500,
        trader_id: Optional[str] = None,
    ) -> List[VirtualTrade]:
        """Closed trades, optionally filtered by strategy/date/trader."""
        q = self.db.query(AVTVirtualTradeRow).filter(
            AVTVirtualTradeRow.status == "closed"
        )
        if strategy_id:
            q = q.filter(AVTVirtualTradeRow.strategy_id == strategy_id)
        if trader_id is not None:
            q = q.filter(AVTVirtualTradeRow.trader_id == trader_id)
        if since:
            q = q.filter(AVTVirtualTradeRow.exit_time >= since)
        q = q.order_by(desc(AVTVirtualTradeRow.exit_time)).limit(limit)
        return [self._to_model(r) for r in q.all()]

    def get_trades_with_improvement(self, improvement_id: str) -> List[VirtualTrade]:
        """Trades where a specific improvement was active."""
        rows = self.db.query(AVTVirtualTradeRow).filter(
            AVTVirtualTradeRow.active_improvements.any(improvement_id)
        ).all()
        return [self._to_model(r) for r in rows]

    def get_trades_by_version(
        self,
        strategy_id: UUID,
        version: str,
    ) -> List[VirtualTrade]:
        """All trades for a specific strategy version."""
        rows = self.db.query(AVTVirtualTradeRow).filter(
            and_(
                AVTVirtualTradeRow.strategy_id == strategy_id,
                AVTVirtualTradeRow.strategy_version == version,
            )
        ).order_by(AVTVirtualTradeRow.entry_time).all()
        return [self._to_model(r) for r in rows]

    def count(self, strategy_id: Optional[UUID] = None) -> int:
        q = self.db.query(AVTVirtualTradeRow)
        if strategy_id:
            q = q.filter(AVTVirtualTradeRow.strategy_id == strategy_id)
        return q.count()

    def get_avt_trades(
        self,
        strategy_id: str,
        version: Optional[str] = None,
        since: Optional[datetime] = None,
        before: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[VirtualTrade]:
        """
        Get AVT's closed trades for self-improver analysis.

        Combines version + since + before filtering.
        - since: trades with exit_time >= since (after-period)
        - before: trades with exit_time < before (before-period baseline)
        """
        q = self.db.query(AVTVirtualTradeRow).filter(
            and_(
                AVTVirtualTradeRow.strategy_id == strategy_id,
                AVTVirtualTradeRow.status == "closed",
            )
        )
        if version:
            q = q.filter(AVTVirtualTradeRow.strategy_version == version)
        if since:
            q = q.filter(AVTVirtualTradeRow.exit_time >= since)
        if before:
            q = q.filter(AVTVirtualTradeRow.exit_time < before)
        q = q.order_by(AVTVirtualTradeRow.exit_time.desc() if before else AVTVirtualTradeRow.exit_time).limit(limit)
        return [self._to_model(r) for r in q.all()]

    def save_improvement(self, hypothesis) -> None:
        """
        Persist an ImprovementHypothesis to the avt_improvements table.

        Maps improvement_lab.models.ImprovementHypothesis fields
        to AVTImprovementRow columns.
        """
        from uuid import UUID as _UUID

        hyp_id = hypothesis.id
        if isinstance(hyp_id, str):
            try:
                hyp_id = _UUID(hyp_id)
            except ValueError:
                hyp_id = uuid4()

        row = self.db.get(AVTImprovementRow, hyp_id)
        if row is None:
            row = AVTImprovementRow(id=hyp_id)
            self.db.add(row)

        strategy_id = hypothesis.strategy_id
        if isinstance(strategy_id, str):
            try:
                strategy_id = _UUID(strategy_id)
            except ValueError:
                pass

        row.strategy_id = strategy_id
        row.hypothesis_type = hypothesis.improvement_type or hypothesis.analysis_layer
        row.description = hypothesis.description or hypothesis.title
        row.params = hypothesis.proposed_rule or {}
        row.status = hypothesis.proof_status  # maps ProofStatus values
        row.applied_at = hypothesis.applied_to_avt_at
        row.applied_at_version = hypothesis.applied_at_version
        row.validation_result = {
            "expected_impact_r": hypothesis.expected_impact_r,
            "confidence": hypothesis.confidence,
            "detection_method": hypothesis.detection_method,
            "proof_trades_count": hypothesis.proof_trades_count,
            "proof_p_value": hypothesis.proof_p_value,
        }
        row.performance_before = hypothesis.proof_performance_before
        row.performance_after = hypothesis.proof_performance_after
        row.evidence_strength = Decimal(
            str(hypothesis.confidence)
        ) if hypothesis.confidence else Decimal("0")
        row.reverted_at = hypothesis.reverted_at
        row.revert_reason = hypothesis.revert_reason

        self.db.flush()

    @staticmethod
    def _to_model(row: AVTVirtualTradeRow) -> VirtualTrade:
        ranking = row.ranking_data or {}
        vt = VirtualTrade(
            id=row.id,
            strategy_id=row.strategy_id,
            strategy_version=row.strategy_version,
            variant_id=row.variant_id,
            symbol=row.symbol,
            direction=row.direction,
            entry_time=row.entry_time,
            entry_price=float(row.entry_price),
            quantity=row.quantity,
            stop_price=float(row.stop_price),
            conviction_at_entry=_to_float(row.conviction_at_entry) or 0.0,
            feature_vector_at_entry=row.feature_vector or [],
            regime_at_entry=row.regime_at_entry or {},
            ranking_position=ranking.get("position", 0),
            ranking_score=ranking.get("score", 0.0),
            ranking_explanation=ranking.get("explanation", ""),
            competing_setups_count=ranking.get("competing", 0),
            active_improvements=row.active_improvements or [],
            max_favorable_r=_to_float(row.max_favorable_r) or 0.0,
            features_at_peak=row.features_at_peak,
            exit_time=row.exit_time,
            exit_price=_to_float(row.exit_price),
            exit_type=row.exit_type,
            # DB pnl columns -> domain fields (realized_pnl / r_multiple); the domain has no separate
            # exit_reason/pnl_pct/improvement_exits, so those DB columns are read-only for the API layer.
            realized_pnl=_to_float(row.pnl_absolute) or 0.0,
            r_multiple=_to_float(row.pnl_r) or 0.0,
            remaining_quantity=int(row.quantity or 0),
            status=row.status or "open",
        )
        # Options (M3b): hydrate the option fields from instrument_meta so the dashboard renders them.
        meta = getattr(row, "instrument_meta", None)
        if meta:
            vt.instrument = meta.get("instrument", "option")
            vt.tradeable = meta.get("tradeable")
            vt.option_strike = meta.get("strike")
            vt.option_expiry = meta.get("expiry")
            vt.option_right = meta.get("right")
            vt.option_lots = meta.get("lots")
            vt.option_lot_size = meta.get("lot_size")
            vt.underlying_stop = meta.get("underlying_stop")
        return vt


# ══════════════════════════════════════════════════════════════════
#  IMPROVEMENT REPOSITORY
# ══════════════════════════════════════════════════════════════════

class AVTImprovementRepository:
    """CRUD for avt_improvements."""

    def __init__(self, db: Session):
        self.db = db

    def save(self, imp: ActiveImprovement) -> None:
        row = self.db.get(AVTImprovementRow, imp.id)
        if row is None:
            row = AVTImprovementRow(id=imp.id)
            self.db.add(row)

        row.strategy_id = imp.strategy_id
        row.hypothesis_type = imp.type
        row.description = imp.description
        row.params = imp.params
        row.status = imp.status
        row.applied_at = imp.applied_at
        row.applied_at_version = imp.applied_at_version
        row.validation_result = imp.validation_result
        row.performance_before = imp.performance_before
        row.performance_after = imp.performance_after
        row.evidence_strength = Decimal(str(imp.evidence_strength))
        row.reverted_at = imp.reverted_at
        row.revert_reason = imp.revert_reason

        self.db.flush()

    def get_by_id(self, imp_id: UUID) -> Optional[ActiveImprovement]:
        row = self.db.get(AVTImprovementRow, imp_id)
        return self._to_model(row) if row else None

    def get_by_strategy(
        self,
        strategy_id: UUID,
        status: Optional[str] = None,
    ) -> List[ActiveImprovement]:
        q = self.db.query(AVTImprovementRow).filter(
            AVTImprovementRow.strategy_id == strategy_id
        )
        if status:
            q = q.filter(AVTImprovementRow.status == status)
        return [self._to_model(r) for r in q.order_by(AVTImprovementRow.created_at).all()]

    def get_proven(self, strategy_id: Optional[UUID] = None) -> List[ActiveImprovement]:
        q = self.db.query(AVTImprovementRow).filter(
            AVTImprovementRow.status == "proven"
        )
        if strategy_id:
            q = q.filter(AVTImprovementRow.strategy_id == strategy_id)
        return [self._to_model(r) for r in q.all()]

    def update_status(self, imp_id: UUID, status: str, reason: str = "") -> None:
        row = self.db.get(AVTImprovementRow, imp_id)
        if row:
            row.status = status
            if status == "reverted":
                row.reverted_at = datetime.now(timezone.utc)
                row.revert_reason = reason
            self.db.flush()

    @staticmethod
    def _to_model(row: AVTImprovementRow) -> ActiveImprovement:
        return ActiveImprovement(
            id=row.id,
            hypothesis_id=row.id,  # same as id for DB-backed improvements
            strategy_id=row.strategy_id,
            type=row.hypothesis_type,
            description=row.description,
            params=row.params or {},
            status=row.status,
            applied_at=row.applied_at or datetime.now(timezone.utc),
            applied_at_version=row.applied_at_version or "1.0",
            validation_result=row.validation_result or {},
            performance_before=row.performance_before,
            performance_after=row.performance_after,
            evidence_strength=_to_float(row.evidence_strength) or 0.0,
            reverted_at=row.reverted_at,
            revert_reason=row.revert_reason,
        )


# ══════════════════════════════════════════════════════════════════
#  STRATEGY VERSION REPOSITORY
# ══════════════════════════════════════════════════════════════════

class AVTVersionRepository:
    """CRUD for avt_strategy_versions."""

    def __init__(self, db: Session):
        self.db = db

    def save_version(
        self,
        strategy_id: UUID,
        version: str,
        improvement_ids: List[UUID],
        notes: str = "",
    ) -> None:
        """Record a new strategy version (closes the previous one)."""
        # Close current version
        current = self.db.query(AVTStrategyVersionRow).filter(
            and_(
                AVTStrategyVersionRow.strategy_id == strategy_id,
                AVTStrategyVersionRow.ended_at.is_(None),
            )
        ).first()
        if current:
            current.ended_at = datetime.now(timezone.utc)

        row = AVTStrategyVersionRow(
            id=uuid4(),
            strategy_id=strategy_id,
            version=version,
            active_improvements=improvement_ids,
            notes=notes,
        )
        self.db.add(row)
        self.db.flush()

    def get_current(self, strategy_id: UUID) -> Optional[Dict[str, Any]]:
        row = self.db.query(AVTStrategyVersionRow).filter(
            and_(
                AVTStrategyVersionRow.strategy_id == strategy_id,
                AVTStrategyVersionRow.ended_at.is_(None),
            )
        ).first()
        if not row:
            return None
        return {
            "id": row.id,
            "version": row.version,
            "started_at": row.started_at,
            "active_improvements": row.active_improvements or [],
            "trade_count": row.trade_count,
            "performance_stats": row.performance_stats or {},
        }

    def get_history(self, strategy_id: UUID) -> List[Dict[str, Any]]:
        rows = self.db.query(AVTStrategyVersionRow).filter(
            AVTStrategyVersionRow.strategy_id == strategy_id
        ).order_by(AVTStrategyVersionRow.started_at).all()
        return [
            {
                "id": r.id,
                "version": r.version,
                "started_at": r.started_at,
                "ended_at": r.ended_at,
                "active_improvements": r.active_improvements or [],
                "trade_count": r.trade_count,
                "performance_stats": r.performance_stats or {},
                "notes": r.notes,
            }
            for r in rows
        ]

    def increment_trade_count(self, strategy_id: UUID) -> None:
        current = self.db.query(AVTStrategyVersionRow).filter(
            and_(
                AVTStrategyVersionRow.strategy_id == strategy_id,
                AVTStrategyVersionRow.ended_at.is_(None),
            )
        ).first()
        if current:
            current.trade_count = (current.trade_count or 0) + 1
            self.db.flush()

    def update_performance(self, strategy_id: UUID, stats: Dict[str, Any]) -> None:
        current = self.db.query(AVTStrategyVersionRow).filter(
            and_(
                AVTStrategyVersionRow.strategy_id == strategy_id,
                AVTStrategyVersionRow.ended_at.is_(None),
            )
        ).first()
        if current:
            current.performance_stats = stats
            self.db.flush()


# ══════════════════════════════════════════════════════════════════
#  SCAN LOG REPOSITORY
# ══════════════════════════════════════════════════════════════════

class AVTScanLogRepository:
    """CRUD for avt_scan_logs."""

    def __init__(self, db: Session):
        self.db = db

    def save(self, log: ScanCycleLog, strategy_id: Optional[UUID] = None) -> None:
        row = AVTScanLogRow(
            id=uuid4(),
            strategy_id=strategy_id,
            cycle_time=log.timestamp,
            candidates_found=log.candidates_found,
            trades_opened=log.executed_count,
            trades_closed=log.exits_triggered,
            skipped=[],
            top_ranked=[],
            duration_ms=int(log.duration_ms),
        )
        self.db.add(row)
        self.db.flush()

    def save_detailed(
        self,
        strategy_id: Optional[UUID],
        candidates_found: int,
        trades_opened: int,
        trades_closed: int,
        skipped: List[Dict[str, Any]],
        top_ranked: List[Dict[str, Any]],
        duration_ms: int,
    ) -> None:
        row = AVTScanLogRow(
            id=uuid4(),
            strategy_id=strategy_id,
            candidates_found=candidates_found,
            trades_opened=trades_opened,
            trades_closed=trades_closed,
            skipped=skipped,
            top_ranked=top_ranked,
            duration_ms=duration_ms,
        )
        self.db.add(row)
        self.db.flush()

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self.db.query(AVTScanLogRow).order_by(
            desc(AVTScanLogRow.cycle_time)
        ).limit(limit).all()
        return [
            {
                "id": r.id,
                "strategy_id": r.strategy_id,
                "cycle_time": r.cycle_time,
                "candidates_found": r.candidates_found,
                "trades_opened": r.trades_opened,
                "trades_closed": r.trades_closed,
                "skipped": r.skipped or [],
                "top_ranked": r.top_ranked or [],
                "duration_ms": r.duration_ms,
            }
            for r in rows
        ]


# ══════════════════════════════════════════════════════════════════
#  COMPETITION SNAPSHOT REPOSITORY
# ══════════════════════════════════════════════════════════════════

class AVTCompetitionRepository:
    """CRUD for avt_competition_snapshots."""

    def __init__(self, db: Session):
        self.db = db

    def save_snapshot(self, report: CompetitionReport, strategy_id: UUID) -> None:
        row = AVTCompetitionSnapshotRow(
            id=uuid4(),
            strategy_id=strategy_id,
            period_days=report.period_days,
            snapshot_date=date.today(),
            user_stats=report.user_stats,
            avt_stats=report.avt_stats,
            performance_gap=Decimal(str(report.performance_gap)),
            differences=[d.model_dump() for d in report.differences],
            proven_improvements=[i.model_dump(mode="json") for i in report.proven_improvements],
        )
        self.db.add(row)
        self.db.flush()

    def get_history(
        self,
        strategy_id: UUID,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        rows = self.db.query(AVTCompetitionSnapshotRow).filter(
            AVTCompetitionSnapshotRow.strategy_id == strategy_id
        ).order_by(desc(AVTCompetitionSnapshotRow.snapshot_date)).limit(limit).all()
        return [
            {
                "snapshot_date": r.snapshot_date,
                "period_days": r.period_days,
                "user_stats": r.user_stats,
                "avt_stats": r.avt_stats,
                "performance_gap": _to_float(r.performance_gap),
                "differences": r.differences,
                "proven_improvements": r.proven_improvements,
            }
            for r in rows
        ]

    def get_latest(self, strategy_id: UUID) -> Optional[Dict[str, Any]]:
        history = self.get_history(strategy_id, limit=1)
        return history[0] if history else None


# ══════════════════════════════════════════════════════════════════
#  LEARNER STATE REPOSITORY
# ══════════════════════════════════════════════════════════════════

class AVTLearnerStateRepository:
    """
    Persist and restore TwoSpeedLearner state (Bayesian tables + deep model blobs)
    across server restarts using the avt_learner_state table (migration 028).
    """

    def __init__(self, db: Session):
        self.db = db

    def save(
        self,
        strategy_id: str,
        detector_type: str,
        bayesian_state: Optional[Dict[str, Any]],
        model_blob: Optional[bytes],
        trade_count: int,
        model_score: float,
    ) -> None:
        """Upsert learner state for (strategy_id, detector_type)."""
        from sqlalchemy import text
        now = datetime.now(timezone.utc)
        self.db.execute(
            text("""
                INSERT INTO avt_learner_state
                    (strategy_id, detector_type, bayesian_state, model_blob,
                     trade_count_at_train, model_score, trained_at, updated_at)
                VALUES
                    (:sid, :dtype, :bstate::jsonb, :blob, :tc, :score, :trained, :now)
                ON CONFLICT (strategy_id, detector_type) DO UPDATE SET
                    bayesian_state    = EXCLUDED.bayesian_state,
                    model_blob        = EXCLUDED.model_blob,
                    trade_count_at_train = EXCLUDED.trade_count_at_train,
                    model_score       = EXCLUDED.model_score,
                    trained_at        = EXCLUDED.trained_at,
                    updated_at        = EXCLUDED.updated_at
            """),
            {
                "sid": strategy_id,
                "dtype": detector_type,
                "bstate": __import__("json").dumps(bayesian_state) if bayesian_state else None,
                "blob": model_blob,
                "tc": trade_count,
                "score": model_score,
                "trained": now,
                "now": now,
            },
        )
        self.db.flush()

    def load(
        self, strategy_id: str, detector_type: str
    ) -> Optional[Dict[str, Any]]:
        """Return persisted state dict or None if not found."""
        from sqlalchemy import text
        row = self.db.execute(
            text("""
                SELECT bayesian_state, model_blob, trade_count_at_train,
                       model_score, trained_at
                FROM avt_learner_state
                WHERE strategy_id = :sid AND detector_type = :dtype
            """),
            {"sid": strategy_id, "dtype": detector_type},
        ).fetchone()
        if row is None:
            return None
        return {
            "bayesian_state": row.bayesian_state,
            "model_blob": bytes(row.model_blob) if row.model_blob else None,
            "trade_count_at_train": row.trade_count_at_train,
            "model_score": row.model_score,
            "trained_at": row.trained_at,
        }

    def delete(self, strategy_id: str, detector_type: str) -> None:
        """Remove persisted state (e.g., when strategy is reset)."""
        from sqlalchemy import text
        self.db.execute(
            text("""
                DELETE FROM avt_learner_state
                WHERE strategy_id = :sid AND detector_type = :dtype
            """),
            {"sid": strategy_id, "dtype": detector_type},
        )
        self.db.flush()


# ══════════════════════════════════════════════════════════════════
#  CALIBRATION REPOSITORY
# ══════════════════════════════════════════════════════════════════

class AVTCalibrationRepository:
    """
    Persist and restore SignalCalibrationTracker state across restarts.

    Stores both AVT signals and user trades in avt_calibration_signals,
    distinguished by signal_details.record_type ("avt_signal" | "user_trade").
    Restoration replays records chronologically so the in-memory matching
    logic reconstructs TP/FP/FN counts correctly.
    """

    def __init__(self, db: Session):
        self.db = db

    def save_signal(self, strategy_id: str, signal: Any) -> None:
        """Upsert an AVT calibration signal."""
        from avt.db_models import AVTCalibrationSignalRow
        import uuid

        try:
            sig_id = signal.id if isinstance(signal.id, uuid.UUID) else uuid.UUID(str(signal.id))
        except (ValueError, AttributeError):
            sig_id = uuid.uuid4()

        details = dict(getattr(signal, "signal_details", {}) or {})
        details["record_type"] = "avt_signal"

        existing = self.db.get(AVTCalibrationSignalRow, sig_id)
        if existing:
            # Update match info in case it changed (signal was later matched)
            existing.matched_trade_id = (
                uuid.UUID(str(signal.matched_trade_id))
                if signal.matched_trade_id else None
            )
            existing.match_type = signal.match_type
            existing.user_annotation = signal.user_annotation
            existing.signal_details = details
        else:
            row = AVTCalibrationSignalRow(
                id=sig_id,
                strategy_id=strategy_id,
                symbol=signal.symbol,
                direction=signal.direction,
                signal_time=signal.timestamp,
                conviction=Decimal(str(signal.conviction)) if signal.conviction else None,
                signal_details=details,
                matched_trade_id=(
                    uuid.UUID(str(signal.matched_trade_id))
                    if signal.matched_trade_id else None
                ),
                match_type=signal.match_type,
                user_annotation=signal.user_annotation,
            )
            self.db.add(row)
        self.db.flush()

    def save_user_trade(self, strategy_id: str, trade: Any) -> None:
        """Upsert a user trade recorded during calibration."""
        from avt.db_models import AVTCalibrationSignalRow
        import uuid

        try:
            trade_id = trade.id if isinstance(trade.id, uuid.UUID) else uuid.UUID(str(trade.id))
        except (ValueError, AttributeError):
            trade_id = uuid.uuid4()

        details = dict(getattr(trade, "trade_details", {}) or {})
        details["record_type"] = "user_trade"

        existing = self.db.get(AVTCalibrationSignalRow, trade_id)
        if existing:
            existing.matched_trade_id = (
                uuid.UUID(str(trade.matched_signal_id))
                if trade.matched_signal_id else None
            )
            existing.match_type = trade.match_type
            existing.signal_details = details
        else:
            row = AVTCalibrationSignalRow(
                id=trade_id,
                strategy_id=strategy_id,
                symbol=trade.symbol,
                direction=trade.direction,
                signal_time=trade.timestamp,
                conviction=None,  # user trades have no conviction
                signal_details=details,
                matched_trade_id=(
                    uuid.UUID(str(trade.matched_signal_id))
                    if trade.matched_signal_id else None
                ),
                match_type=trade.match_type,
            )
            self.db.add(row)
        self.db.flush()

    def restore_tracker(self, strategy_id: str, tracker: Any) -> int:
        """
        Restore a SignalCalibrationTracker from DB.

        Replays records in chronological order. The tracker's matching
        logic reconstructs TP/FP/FN counts correctly via the same
        record_avt_signal() / record_user_trade() calls.

        Returns number of records restored.
        """
        from avt.db_models import AVTCalibrationSignalRow

        rows = (
            self.db.query(AVTCalibrationSignalRow)
            .filter(AVTCalibrationSignalRow.strategy_id == strategy_id)
            .order_by(AVTCalibrationSignalRow.signal_time)
            .all()
        )
        if not rows:
            return 0

        # Suppress DB callbacks during restore to avoid re-saving
        old_sig_cb = tracker._persist_signal_callback
        old_trade_cb = tracker._persist_trade_callback
        tracker._persist_signal_callback = None
        tracker._persist_trade_callback = None

        # Set started_at to the earliest record
        tracker._started_at = rows[0].signal_time

        count = 0
        try:
            for row in rows:
                details = dict(row.signal_details or {})
                record_type = details.pop("record_type", "avt_signal")

                if record_type == "avt_signal":
                    tracker.record_avt_signal(
                        symbol=row.symbol,
                        direction=row.direction,
                        conviction=float(row.conviction) if row.conviction else 0.0,
                        timestamp=row.signal_time,
                        details=details,
                    )
                elif record_type == "user_trade":
                    tracker.record_user_trade(
                        symbol=row.symbol,
                        direction=row.direction,
                        timestamp=row.signal_time,
                        trade_id=str(row.id),
                        details=details,
                    )
                count += 1
        finally:
            tracker._persist_signal_callback = old_sig_cb
            tracker._persist_trade_callback = old_trade_cb

        return count

    def get_signals(self, strategy_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Return recent calibration signals for the UI report."""
        from avt.db_models import AVTCalibrationSignalRow

        rows = (
            self.db.query(AVTCalibrationSignalRow)
            .filter(AVTCalibrationSignalRow.strategy_id == strategy_id)
            .order_by(desc(AVTCalibrationSignalRow.signal_time))
            .limit(limit)
            .all()
        )
        return [
            {
                "id": str(r.id),
                "record_type": (r.signal_details or {}).get("record_type", "avt_signal"),
                "symbol": r.symbol,
                "direction": r.direction,
                "signal_time": r.signal_time.isoformat() if r.signal_time else None,
                "conviction": _to_float(r.conviction),
                "match_type": r.match_type,
                "user_annotation": r.user_annotation,
            }
            for r in rows
        ]
