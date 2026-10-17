"""
Journal API Routes
===================
Endpoints for trade journaling: pre-trade notes, post-trade reviews,
lesson capture, and journal entry retrieval.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import UUID

_IST = timezone(timedelta(hours=5, minutes=30))

def _to_ist(dt: Optional[datetime]) -> Optional[str]:
    """Convert a UTC (or naive-UTC) datetime to IST ISO string."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from api.auth import get_current_user

from core.database import get_db_dependency as get_db
from core.models import Trade, JournalEntry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/journal", tags=["journal"], dependencies=[Depends(get_current_user)])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PreTradeNoteRequest(BaseModel):
    trade_id: UUID
    plan: str = Field(..., description="What is the trade plan?")
    conviction_score: int = Field(5, ge=1, le=10, description="Conviction 1-10")
    emotion_pre: Optional[str] = None
    sleep_quality: Optional[int] = Field(None, ge=1, le=10)
    energy_level: Optional[int] = Field(None, ge=1, le=10)
    stress_level: Optional[int] = Field(None, ge=1, le=10)
    was_planned: bool = True
    notes: Optional[str] = None


class PostTradeReviewRequest(BaseModel):
    trade_id: UUID
    what_went_well: Optional[str] = None
    what_went_wrong: Optional[str] = None
    lesson: Optional[str] = None
    emotion_post: Optional[str] = None
    followed_plan: bool = True
    would_take_again: bool = True
    notes: Optional[str] = None


class JournalEntryResponse(BaseModel):
    trade_id: str
    plan: Optional[str]
    conviction_score: Optional[int]
    emotion_pre: Optional[str]
    emotion_post: Optional[str]
    what_went_well: Optional[str]
    what_went_wrong: Optional[str]
    lesson: Optional[str]
    was_planned: bool
    followed_plan: Optional[bool]
    would_take_again: Optional[bool]
    created_at: Optional[str]
    updated_at: Optional[str]


class LessonRequest(BaseModel):
    trade_id: UUID
    lesson: str
    tags: list[str] = Field(default_factory=list)
    save_to_knowledge_base: bool = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/pre-trade", status_code=201)
def add_pre_trade_note(request: PreTradeNoteRequest, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """
    Add or update the pre-trade journal entry for a trade.
    Captures trader's mental state and plan before entry.
    """
    trade = db.query(Trade).filter(Trade.id == request.trade_id, Trade.trader_id == user.id).first()
    if not trade:
        raise HTTPException(404, f"Trade {request.trade_id} not found")

    entry = db.query(JournalEntry).filter(JournalEntry.trade_id == request.trade_id).first()
    if entry:
        # Update existing
        entry.plan = request.plan
        entry.pre_trade_confidence = request.conviction_score
        entry.pre_trade_emotion = request.emotion_pre
        entry.sleep_quality = request.sleep_quality
        entry.energy_level = request.energy_level
        entry.stress_level = request.stress_level
        entry.was_planned = request.was_planned
        entry.notes = request.notes
        entry.updated_at = datetime.utcnow()
    else:
        entry = JournalEntry(
            trade_id=request.trade_id,
            plan=request.plan,
            pre_trade_confidence=request.conviction_score,
            pre_trade_emotion=request.emotion_pre,
            sleep_quality=request.sleep_quality,
            energy_level=request.energy_level,
            stress_level=request.stress_level,
            was_planned=request.was_planned,
            notes=request.notes,
        )
        db.add(entry)

    db.commit()
    db.refresh(entry)
    logger.info(f"Pre-trade journal saved for trade {request.trade_id}")
    return {"status": "ok", "trade_id": str(request.trade_id)}


@router.post("/post-trade", status_code=200)
def add_post_trade_review(request: PostTradeReviewRequest, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """
    Add or update the post-trade review for a trade.
    Captures lessons learned and emotional state after closure.
    """
    trade = db.query(Trade).filter(Trade.id == request.trade_id, Trade.trader_id == user.id).first()
    if not trade:
        raise HTTPException(404, f"Trade {request.trade_id} not found")

    entry = db.query(JournalEntry).filter(JournalEntry.trade_id == request.trade_id).first()
    if not entry:
        entry = JournalEntry(trade_id=request.trade_id)
        db.add(entry)

    entry.what_went_well = request.what_went_well
    entry.what_went_wrong = request.what_went_wrong
    entry.lesson = request.lesson
    entry.post_trade_emotion = request.emotion_post
    entry.followed_plan = request.followed_plan
    entry.would_take_again = request.would_take_again
    if request.notes:
        entry.notes = (entry.notes or "") + "\n\n[Post review] " + request.notes
    entry.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(entry)
    logger.info(f"Post-trade review saved for trade {request.trade_id}")
    return {"status": "ok", "trade_id": str(request.trade_id)}


@router.get("/{trade_id}", response_model=JournalEntryResponse)
def get_journal_entry(trade_id: UUID, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """Retrieve the full journal entry for a trade."""
    # Verify trade belongs to current user
    trade = db.query(Trade).filter(Trade.id == trade_id, Trade.trader_id == user.id).first()
    if not trade:
        raise HTTPException(404, f"Trade {trade_id} not found")
    entry = db.query(JournalEntry).filter(JournalEntry.trade_id == trade_id).first()
    if not entry:
        raise HTTPException(404, f"No journal entry for trade {trade_id}")
    return JournalEntryResponse(
        trade_id=str(trade_id),
        plan=entry.plan,
        conviction_score=entry.pre_trade_confidence,
        emotion_pre=str(entry.pre_trade_emotion.value
                        if hasattr(entry.pre_trade_emotion, "value")
                        else entry.pre_trade_emotion) if entry.pre_trade_emotion else None,
        emotion_post=str(entry.post_trade_emotion.value
                         if hasattr(entry.post_trade_emotion, "value")
                         else entry.post_trade_emotion) if entry.post_trade_emotion else None,
        what_went_well=entry.what_went_well,
        what_went_wrong=entry.what_went_wrong,
        lesson=entry.lesson,
        was_planned=bool(entry.was_planned),
        followed_plan=entry.followed_plan,
        would_take_again=entry.would_take_again,
        created_at=_to_ist(entry.created_at),
        updated_at=_to_ist(entry.updated_at),
    )


@router.get("/{trade_id}/eye-review")
def eye_review(trade_id: UUID, timeframe: str = "5m",
               db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """EYE-grounded review of a stored trade (the Learning-Journal core): what the frozen-core eye SAW
    at your entry and exit — with/against its read, its exit-trigger, and whether you got out while the
    move was alive vs done. Heavy (~2 eye reads); sync def → threadpooled."""
    trade = db.query(Trade).filter(Trade.id == trade_id, Trade.trader_id == user.id).first()
    if not trade:
        raise HTTPException(404, f"Trade {trade_id} not found")
    from intelligence.trade_review import review_trade_model
    try:
        tr = review_trade_model(trade, interval=timeframe)
    except Exception as e:
        raise HTTPException(500, f"eye review failed: {e}")
    if not tr.ok:
        raise HTTPException(404, tr.note)
    return tr.to_dict()


@router.get("/", response_model=list[JournalEntryResponse])
def list_journal_entries(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    has_lesson: Optional[bool] = Query(None),
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """List recent journal entries. Optionally filter to only those with lessons."""
    q = db.query(JournalEntry).join(Trade, JournalEntry.trade_id == Trade.id).filter(Trade.trader_id == user.id)
    if has_lesson is True:
        q = q.filter(JournalEntry.lesson.isnot(None))
    entries = q.order_by(Trade.entry_time.desc()).offset(offset).limit(limit).all()

    results = []
    for entry in entries:
        results.append(JournalEntryResponse(
            trade_id=str(entry.trade_id),
            plan=entry.plan,
            conviction_score=entry.pre_trade_confidence,
            emotion_pre=str(entry.pre_trade_emotion.value
                            if hasattr(entry.pre_trade_emotion, "value")
                            else entry.pre_trade_emotion) if entry.pre_trade_emotion else None,
            emotion_post=str(entry.post_trade_emotion.value
                             if hasattr(entry.post_trade_emotion, "value")
                             else entry.post_trade_emotion) if entry.post_trade_emotion else None,
            what_went_well=entry.what_went_well,
            what_went_wrong=entry.what_went_wrong,
            lesson=entry.lesson,
            was_planned=bool(entry.was_planned),
            followed_plan=entry.followed_plan,
            would_take_again=entry.would_take_again,
            created_at=entry.created_at.isoformat() if entry.created_at else None,
            updated_at=entry.updated_at.isoformat() if entry.updated_at else None,
        ))
    return results


@router.post("/lesson", status_code=201)
def capture_lesson(request: LessonRequest, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """
    Capture a lesson from a trade and optionally save it to the knowledge base.
    The lesson will appear in future AI context for similar setups.
    """
    trade = db.query(Trade).filter(Trade.id == request.trade_id, Trade.trader_id == user.id).first()
    if not trade:
        raise HTTPException(404, f"Trade {request.trade_id} not found")

    # Update journal entry lesson field
    entry = db.query(JournalEntry).filter(JournalEntry.trade_id == request.trade_id).first()
    if not entry:
        entry = JournalEntry(trade_id=request.trade_id)
        db.add(entry)
    entry.lesson = request.lesson
    entry.updated_at = datetime.utcnow()
    db.flush()

    # Save to knowledge base if requested
    kb_entry_id = None
    if request.save_to_knowledge_base:
        from rag.knowledge_base import KnowledgeBase
        from rag.indexer import KnowledgeIndexer
        kb = KnowledgeBase(db)
        strategy_name = (
            trade.attribution.resolved_strategy_id
            if trade.attribution else None
        )
        kb_entry = kb.add_lesson_from_trade(str(trade.id), request.lesson, strategy_name)
        db.flush()
        # Index for semantic search
        try:
            indexer = KnowledgeIndexer()
            indexer.index_entry(kb_entry)
        except Exception as e:
            logger.warning(f"Knowledge indexing failed (non-fatal): {e}")
        kb_entry_id = str(kb_entry.id)

    db.commit()
    logger.info(f"Lesson captured for trade {request.trade_id}")
    return {"status": "ok", "trade_id": str(request.trade_id),
            "knowledge_entry_id": kb_entry_id}
