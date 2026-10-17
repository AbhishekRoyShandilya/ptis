# knowledge.py
# api/routes/knowledge.py
"""
Knowledge base endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.auth import get_current_user
from pydantic import BaseModel
from typing import Optional, List

from core.database import get_db_dependency
from rag.knowledge_base import KnowledgeBase

router = APIRouter(prefix="/knowledge", tags=["knowledge"], dependencies=[Depends(get_current_user)])


class KnowledgeEntryRequest(BaseModel):
    category: str          # lesson, rule, insight, pattern
    title: str
    content: str
    tags: Optional[List[str]] = None


class TradeLesson(BaseModel):
    trade_id: str
    lesson: str
    strategy_name: Optional[str] = None


@router.post("/entry")
def add_knowledge(
    request: KnowledgeEntryRequest,
    db: Session = Depends(get_db_dependency),
):
    """Add a knowledge entry."""
    kb = KnowledgeBase(db)
    entry = kb.add_entry(
        category=request.category,
        title=request.title,
        content=request.content,
        tags=request.tags,
    )
    db.commit()
    return {"id": entry.id, "title": entry.title, "category": entry.category}


@router.post("/lesson")
def add_trade_lesson(
    request: TradeLesson,
    db: Session = Depends(get_db_dependency),
):
    """Add a lesson learned from a trade."""
    kb = KnowledgeBase(db)
    entry = kb.add_lesson_from_trade(
        trade_id=request.trade_id,
        lesson=request.lesson,
        strategy_name=request.strategy_name,
    )
    db.commit()
    return {"id": entry.id, "title": entry.title}


@router.get("/search")
def search_knowledge(
    q: str,
    category: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db_dependency),
):
    """Search knowledge base."""
    kb = KnowledgeBase(db)
    entries = kb.search(q, category=category, limit=limit)
    db.commit()

    return {
        "query": q,
        "results": [
            {
                "id": e.id,
                "category": e.category,
                "title": e.title,
                "content": e.content[:200],
                "tags": e.tags,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "relevance_count": e.relevance_count,
            }
            for e in entries
        ],
    }


@router.get("/lessons/recent")
def recent_lessons(
    limit: int = 10,
    db: Session = Depends(get_db_dependency),
):
    """Get recent lessons learned."""
    kb = KnowledgeBase(db)
    entries = kb.get_recent_lessons(limit)

    return {
        "lessons": [
            {
                "id": e.id,
                "title": e.title,
                "content": e.content,
                "tags": e.tags,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in entries
        ],
    }