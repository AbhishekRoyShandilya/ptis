# api/routes/export.py
"""
Data export endpoints — CSV downloads for trades and journal entries.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session as DBSession

from api.auth import get_current_user
from core.database import get_db_dependency as get_db
from core.models import Trade, JournalEntry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/export", tags=["export"], dependencies=[Depends(get_current_user)])


@router.get("/trades")
def export_trades(
    format: str = Query("csv", pattern="^csv$"),
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Export all trades as CSV."""
    trades = (
        db.query(Trade)
        .filter(Trade.trader_id == user.id)
        .order_by(Trade.entry_time.desc())
        .all()
    )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "symbol", "direction", "status",
        "entry_price", "exit_price", "entry_time", "exit_time",
        "position_size", "stop_loss", "take_profit",
        "pnl", "pnl_pct", "r_multiple",
        "risk_amount", "risk_pct",
        "declared_strategy_id", "market_regime",
        "emotion_pre", "emotion_post", "conviction_score",
        "notes",
    ])

    for t in trades:
        writer.writerow([
            str(t.id),
            t.symbol,
            t.direction.value if t.direction else "",
            t.status.value if t.status else "",
            t.entry_price,
            t.exit_price,
            t.entry_time.isoformat() if t.entry_time else "",
            t.exit_time.isoformat() if t.exit_time else "",
            t.position_size,
            t.stop_loss,
            t.take_profit,
            t.pnl,
            t.pnl_pct,
            t.r_multiple,
            t.risk_amount,
            t.risk_pct,
            t.declared_strategy_id or "",
            t.market_regime.value if t.market_regime else "",
            t.emotion_pre.value if t.emotion_pre else "",
            t.emotion_post.value if t.emotion_post else "",
            t.conviction_score,
            t.notes or "",
        ])

    buf.seek(0)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=ptis_trades_{timestamp}.csv"},
    )


@router.get("/journal")
def export_journal(
    format: str = Query("csv", pattern="^csv$"),
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Export journal entries as CSV."""
    entries = (
        db.query(JournalEntry)
        .join(Trade, JournalEntry.trade_id == Trade.id)
        .filter(Trade.trader_id == user.id)
        .order_by(JournalEntry.created_at.desc())
        .all()
    )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "trade_id", "created_at",
        "pre_trade_notes", "post_trade_notes",
        "emotional_state", "lessons_learned",
    ])

    for e in entries:
        writer.writerow([
            str(e.id),
            str(e.trade_id) if e.trade_id else "",
            e.created_at.isoformat() if e.created_at else "",
            getattr(e, "pre_trade_notes", "") or "",
            getattr(e, "post_trade_notes", "") or "",
            getattr(e, "emotional_state", "") or "",
            getattr(e, "lessons_learned", "") or "",
        ])

    buf.seek(0)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=ptis_journal_{timestamp}.csv"},
    )
