# rag/grounded_llm.py
"""
Grounded LLM — Anti-hallucination wrapper with RAG adjustment clamping.

Architecture Rules:
  - RAG adjustment clamped to [-0.25, +0.15] (asymmetric: easier to reduce than boost)
  - LLM ONLY for explanation — never for scanning/ranking/matching
  - Confidence auto-capped by sample size
  - Decision trace: every RAG-influenced decision logged with full breakdown (GAP 3)
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rag.context_builder import StructuredContext
from rag.ingestor import cap_confidence

logger = logging.getLogger(__name__)


# ── Grounding system prompt ──────────────────────────────────────

GROUNDING_SYSTEM_PROMPT = """You are a trading intelligence assistant grounded in the user's personal data.

RULES:
1. ONLY answer based on the CONTEXT provided below.
2. If the context does not contain information, say: "I don't have enough data on that yet."
3. Do NOT invent statistics, trade examples, or performance numbers.
4. Cite specific data from the context. Always mention sample sizes.
5. Max 300 tokens for explanation.
6. When discussing performance, always include sample size: "Based on N trades..."
7. If data consistency is LOW, mention it: "Note: your recent results show inconsistency..."
"""


@dataclass
class GroundedResponse:
    """Response from grounded LLM with full provenance."""
    content: str
    confidence: float                    # from retrieval scores, auto-capped by sample size
    grounding_sources: List[Dict]        # [{chunk_id, score, layer, title}]
    is_grounded: bool                    # True if answer came from context
    sample_sizes: Dict[str, int]         # {strategy: 12, trade: 45, ...}
    not_found_topics: List[str] = field(default_factory=list)
    consistency_level: str = "unknown"
    unknown_pattern_warning: bool = False


@dataclass
class DecisionTrace:
    """
    Full trace of a RAG-influenced decision (GAP 3).
    Logged to DB for auditability and debugging.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str = ""
    mode: str = ""                       # "coaching" | "signal_eval" | "co_planning" | etc.
    base_score: float = 0.0              # engine score BEFORE RAG
    rag_adjustment: float = 0.0          # clamped [-0.25, +0.15]
    final_score: float = 0.0             # base + clamped adjustment
    sample_size: int = 0
    confidence: float = 0.0              # auto-capped confidence
    top_reason: str = ""                 # highest-scoring chunk title
    consistency_level: str = "unknown"
    unknown_pattern: bool = False
    sources: List[Dict] = field(default_factory=list)  # grounding sources


class GroundedLLM:
    """
    Wraps an LLM client with grounding enforcement.

    Engine computes → LLM explains. No LLM scoring. No LLM ranking.
    RAG can only NUDGE, never dominate.
    """

    def __init__(self, llm_client=None, db_factory=None):
        self._llm_client = llm_client
        self._db_factory = db_factory

    def _get_llm(self):
        """Lazy-load LLM client."""
        if self._llm_client is not None:
            return self._llm_client
        try:
            from agents.llm_client import LLMClient
            from config.settings import settings
            if settings.LLM_API_KEY:
                self._llm_client = LLMClient(
                    provider=settings.LLM_PROVIDER,
                    model=settings.LLM_MODEL,
                    api_key=settings.LLM_API_KEY,
                    base_url=getattr(settings, "LLM_BASE_URL", None) or None,
                )
                return self._llm_client
        except Exception:
            pass
        return None

    async def generate(
        self,
        user_query: str,
        context: StructuredContext,
        mode: str = "coaching",
    ) -> GroundedResponse:
        """
        Generate a grounded response using context from RAG.
        If no LLM available, returns a structured summary from context alone.
        """
        llm = self._get_llm()

        # Calculate total sample size
        total_samples = sum(
            v for k, v in context.sample_sizes.items()
            if not k.startswith("_") and isinstance(v, int)
        )

        # Auto-cap confidence by sample size
        confidence = cap_confidence(context.confidence_score, total_samples)

        grounding_sources = context.get_grounding_sources()
        is_grounded = len(grounding_sources) > 0

        if llm is None:
            # No LLM — return context summary directly
            content = context.format_for_prompt(max_tokens=500)
            if not content or content == "No relevant personal knowledge found.":
                content = "I don't have enough data on that yet."
                is_grounded = False

            return GroundedResponse(
                content=content,
                confidence=confidence,
                grounding_sources=grounding_sources,
                is_grounded=is_grounded,
                sample_sizes=context.sample_sizes,
                consistency_level=context.consistency_level,
                unknown_pattern_warning=context.unknown_pattern_warning,
            )

        # Build grounded prompt
        context_text = context.format_for_prompt(max_tokens=2000)
        system_prompt = GROUNDING_SYSTEM_PROMPT + "\n\n" + context_text

        try:
            response_text = await llm.generate(
                prompt="",
                system_prompt=system_prompt,
                user_message=user_query,
                max_tokens=400,
                caller="grounded_llm",
            )
        except Exception as e:
            logger.warning("Grounded LLM generation failed: %s", e)
            response_text = context.format_for_prompt(max_tokens=500)

        return GroundedResponse(
            content=response_text,
            confidence=confidence,
            grounding_sources=grounding_sources,
            is_grounded=is_grounded,
            sample_sizes=context.sample_sizes,
            consistency_level=context.consistency_level,
            unknown_pattern_warning=context.unknown_pattern_warning,
        )

    # ── RAG adjustment clamping ───────────────────────────────────

    @staticmethod
    def clamp_rag_adjustment(raw_adjustment: float) -> float:
        """
        RAG can only NUDGE, never dominate.
        Asymmetric: easier to reduce conviction (-0.25) than boost (+0.15).
        """
        return max(-0.25, min(0.15, raw_adjustment))

    def compute_rag_adjustment(
        self,
        context: StructuredContext,
        base_score: float = 0.0,
    ) -> DecisionTrace:
        """
        Compute the RAG adjustment for an engine decision.
        Returns a full DecisionTrace for logging (GAP 3).

        The adjustment is based on retrieval confidence and consistency,
        NOT on LLM judgment.
        """
        total_samples = sum(
            v for k, v in context.sample_sizes.items()
            if not k.startswith("_") and isinstance(v, int)
        )

        # Raw adjustment = retrieval confidence, scaled
        raw = (context.confidence_score - 0.5) * 0.5  # center around 0

        # Apply consistency penalty
        if context.consistency_level == "low":
            raw *= 0.5  # halve the adjustment for inconsistent data

        # Unknown pattern safety (GAP 2): reduce by 50%
        if context.unknown_pattern_warning:
            raw *= 0.5

        # Clamp
        clamped = self.clamp_rag_adjustment(raw)

        # Auto-cap confidence
        confidence = cap_confidence(context.confidence_score, total_samples)

        # Top reason
        top_reason = ""
        sources = context.get_grounding_sources()
        if sources:
            top_reason = sources[0].get("title", "")

        trace = DecisionTrace(
            query=context.query,
            mode="rag_adjustment",
            base_score=base_score,
            rag_adjustment=clamped,
            final_score=base_score + clamped,
            sample_size=total_samples,
            confidence=confidence,
            top_reason=top_reason,
            consistency_level=context.consistency_level,
            unknown_pattern=context.unknown_pattern_warning,
            sources=sources[:5],
        )

        # Log the trace (GAP 3)
        self._log_decision_trace(trace)

        return trace

    def _log_decision_trace(self, trace: DecisionTrace) -> None:
        """
        Log decision trace to DB for auditability (GAP 3).
        Non-critical: failures are logged but don't block.
        """
        try:
            if not self._db_factory:
                logger.debug(
                    "Decision trace: base=%.2f rag=%.2f final=%.2f samples=%d conf=%.2f reason='%s'",
                    trace.base_score, trace.rag_adjustment, trace.final_score,
                    trace.sample_size, trace.confidence, trace.top_reason,
                )
                return

            db = self._db_factory()
            if hasattr(db, "__next__"):
                db = next(db)
            try:
                from rag.ingestion_store import IngestionRecord
                # Store trace as an ingestion record with layer="decision_trace"
                record = IngestionRecord(
                    layer="decision_trace",
                    source="system",
                    source_id=trace.trace_id,
                    structured_data={
                        "query": trace.query[:200],
                        "mode": trace.mode,
                        "base_score": trace.base_score,
                        "rag_adjustment": trace.rag_adjustment,
                        "final_score": trace.final_score,
                        "sample_size": trace.sample_size,
                        "confidence": trace.confidence,
                        "top_reason": trace.top_reason,
                        "consistency_level": trace.consistency_level,
                        "unknown_pattern": trace.unknown_pattern,
                    },
                    quality_score=trace.confidence,
                    title=f"Decision: {trace.mode}",
                )
                db.add(record)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.debug("Decision trace logging failed (non-critical): %s", e)


# ── Singleton ─────────────────────────────────────────────────────

_grounded_llm_instance: Optional[GroundedLLM] = None


def get_grounded_llm(db_factory=None) -> GroundedLLM:
    """Get or create the singleton GroundedLLM."""
    global _grounded_llm_instance
    if _grounded_llm_instance is None:
        _grounded_llm_instance = GroundedLLM(db_factory=db_factory)
    return _grounded_llm_instance
