# rag/ingestion_store.py
"""
RAG Ingestion Log — Postgres-side structured storage.

DUAL STORE pattern: embedding text → Vector DB, structured data → Postgres.
This table tracks what's been ingested and stores the structured half
for computation/filtering without parsing text.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, Integer, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from core.models import Base


class IngestionRecord(Base):
    """Tracks what's been ingested into RAG + stores structured data."""
    __tablename__ = "rag_ingestion_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    layer = Column(String(20), nullable=False, index=True)       # strategy|pattern|trade|insight
    source = Column(String(20), nullable=False, index=True)      # user|avt|system
    source_id = Column(String(64), nullable=False, index=True)   # trade_id, profile_id, etc.
    structured_data = Column(JSONB, nullable=True)                # the structured half of DualOutput
    quality_score = Column(Float, nullable=True)
    chunk_count = Column(Integer, default=1)                      # how many vector chunks created
    title = Column(String(200), nullable=True)
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())
