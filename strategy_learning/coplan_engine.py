# strategy_learning/coplan_engine.py
"""
Co-Plan Engine — Real-time trade assistance using learned strategy profiles.

When a trader shares a current chart, the co-planner:
1. Analyses the chart via vision
2. Searches for similar past examples (vector + heuristic)
3. Synthesises actionable, strategy-specific recommendations
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.llm_client import LLMClient, LLMMessage
from config.settings import settings
from core.models import LearningExample, LearningSession, StrategyProfile
from strategy_learning.models import (
    CoplanResponse,
    ProfileConfidence,
    SimilarExample,
    VisionAnalysis,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  Prompt
# ──────────────────────────────────────────────────────────────

COPLAN_PROMPT = """\
You are a trading co-pilot helping a trader evaluate a potential trade setup.

══ TRADER'S LEARNED STRATEGY: {strategy_name} ══
{narrative}

══ LEARNED ENTRY CRITERIA ══
{entry_criteria}

══ LEARNED EXIT RULES ══
{exit_rules}

══ CURRENT CHART ANALYSIS ══
{chart_analysis}

══ SIMILAR PAST SETUPS ({similar_count} found) ══
{similar_examples}

══ TRADER'S CONTEXT ══
{context}

══ INSTRUCTIONS ══
Compare the CURRENT chart against the learned strategy pattern.
Be specific — reference the learned criteria by name.
If the setup is weak, say so honestly.
Do NOT give generic advice — tie everything to THIS trader's pattern.

Return ONLY a JSON object (no markdown fences):
{{
  "setup_match_score": <0-100>,
  "setup_detected": <true|false>,
  "entry_recommendation": "<specific advice>",
  "stop_recommendation": "<specific stop advice>",
  "target_recommendations": ["<target 1>", "<target 2>"],
  "risk_assessment": "<risk and sizing advice>",
  "narrative": "<2-3 paragraph personalised co-planning advice>",
  "warnings": ["<any concerns>"],
  "missing_criteria": ["<any learned criteria NOT met on this chart>"]
}}"""


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines) if lines[-1].strip() != "```" else -1
        text = "\n".join(lines[start:end]).strip()
    return text


def _safe_json(raw: str) -> Optional[dict]:
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return None


# ──────────────────────────────────────────────────────────────
#  Similarity scoring (keyword-based until vector search is live)
# ──────────────────────────────────────────────────────────────

def _keyword_similarity(vision: VisionAnalysis, example: LearningExample) -> float:
    """
    Simple keyword-overlap similarity between current chart analysis
    and a stored example's interpretation + vision data.

    Returns 0.0–1.0.  This is a bridge until vector similarity is wired up.
    """
    # Build keyword set from current vision
    current_tokens: set[str] = set()
    if vision.trend_direction:
        current_tokens.add(vision.trend_direction.lower())
    if vision.market_structure:
        current_tokens.update(vision.market_structure.lower().split())
    for obs in vision.key_observations:
        current_tokens.update(obs.lower().split())
    for cp in vision.candle_patterns:
        current_tokens.update(cp.pattern.lower().split())
    if vision.overall_context:
        current_tokens.update(vision.overall_context.lower().split())

    # Build keyword set from example
    example_tokens: set[str] = set()
    interp = (example.ai_interpretation or "").lower()
    example_tokens.update(interp.split())

    ex_vision = example.vision_analysis or {}
    if ex_vision.get("trend_direction"):
        example_tokens.add(ex_vision["trend_direction"].lower())
    if ex_vision.get("overall_context"):
        example_tokens.update(ex_vision["overall_context"].lower().split())

    ann = example.user_annotations or {}
    if ann.get("direction"):
        example_tokens.add(ann["direction"].lower())

    # Remove stop words
    _STOP = {"the", "a", "an", "is", "at", "in", "on", "to", "and", "of", "for", "with"}
    current_tokens -= _STOP
    example_tokens -= _STOP

    if not current_tokens or not example_tokens:
        return 0.0

    overlap = current_tokens & example_tokens
    # Jaccard similarity
    union = current_tokens | example_tokens
    return len(overlap) / len(union) if union else 0.0


# ──────────────────────────────────────────────────────────────
#  Engine
# ──────────────────────────────────────────────────────────────

class CoplanEngine:
    """Real-time trade co-planning using learned profiles."""

    def __init__(
        self,
        db: Session,
        llm: Optional[LLMClient] = None,
        vector_store: Any = None,
    ):
        self.db = db
        self._llm = llm
        self._vector_store = vector_store

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(
                provider=settings.LLM_PROVIDER,
                model=settings.LLM_MODEL,
            )
        return self._llm

    # ── main entry point ──────────────────────────────────────

    async def coplan(
        self,
        vision_analysis: VisionAnalysis,
        profile: StrategyProfile,
        context: str = "",
        symbol: str = "",
        timeframe: str = "",
    ) -> CoplanResponse:
        """Generate co-planning recommendations for a potential trade."""
        # Guard: profile not ready
        if not profile.narrative:
            return CoplanResponse(
                narrative=(
                    "This strategy profile hasn't been built yet. "
                    "Complete a teaching session first so I can learn your pattern."
                ),
                confidence=0.0,
                warnings=["Profile has no learned narrative."],
            )

        conf = ProfileConfidence(**(profile.confidence_scores or {}))
        if not conf.is_tradeable:
            return CoplanResponse(
                narrative=(
                    f"Profile confidence is low (overall={conf.overall:.0%}). "
                    f"Weak areas: {', '.join(conf.weak_areas())}. "
                    "Add more annotated examples before using co-planning."
                ),
                confidence=conf.overall,
                warnings=[
                    f"Low confidence in: {', '.join(conf.weak_areas())}",
                ],
            )

        # Find similar past examples
        similar_examples = await self._find_similar_examples(
            profile.id, vision_analysis
        )

        # Build prompt
        chart_text = vision_analysis.to_text(symbol=symbol, timeframe=timeframe)
        similar_text = self._format_similar(similar_examples)

        prompt = COPLAN_PROMPT.format(
            strategy_name=profile.name,
            narrative=profile.narrative,
            entry_criteria=json.dumps(profile.entry_criteria or [], indent=2),
            exit_rules=json.dumps(profile.exit_rules or {}, indent=2),
            chart_analysis=chart_text,
            similar_count=len(similar_examples),
            similar_examples=similar_text or "No similar past examples found yet.",
            context=context or "No additional context from trader.",
        )

        # Call LLM
        from agents.prompts import COPLAN_ADVISOR

        llm = self._get_llm()
        try:
            response = await llm.chat(
                messages=[
                    LLMMessage(role="system", content=COPLAN_ADVISOR),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=0.3,
                max_tokens=1500,
                caller="coplan_engine",
            )
        except Exception as exc:
            logger.error("LLM call failed in coplan: %s", exc)
            return CoplanResponse(
                narrative="Co-planning temporarily unavailable. Please try again.",
                confidence=0.0,
                warnings=[f"LLM error: {str(exc)[:100]}"],
            )

        return self._parse_response(response.content, similar_examples, conf)

    # ── similarity search ─────────────────────────────────────

    async def _find_similar_examples(
        self,
        profile_id: UUID,
        vision: VisionAnalysis,
    ) -> list[tuple[LearningExample, float]]:
        """
        Find the most relevant examples.  Uses vector search when
        available, falls back to keyword similarity + recency.

        Returns list of (example, similarity_score) tuples.
        """
        # Try vector search first
        if self._vector_store and vision.overall_context:
            try:
                return await self._vector_similarity(profile_id, vision)
            except Exception as exc:
                logger.warning("Vector search failed, falling back: %s", exc)

        # Fallback: keyword heuristic
        return self._keyword_search(profile_id, vision)

    async def _vector_similarity(
        self,
        profile_id: UUID,
        vision: VisionAnalysis,
    ) -> list[tuple[LearningExample, float]]:
        """Search via vector store."""
        collection = getattr(
            settings, "LEARNING_VECTOR_COLLECTION", "learning_examples"
        )
        # Build query text from vision
        query_text = vision.to_text()

        results = await self._vector_store.search(
            collection=collection,
            query_text=query_text,
            filter={"profile_id": str(profile_id), "is_positive": True},
            top_k=5,
        )

        example_ids = [r["metadata"]["example_id"] for r in results]
        if not example_ids:
            return []

        examples_by_id = {}
        db_results = self.db.execute(
            select(LearningExample).where(
                LearningExample.id.in_(example_ids)
            )
        )
        for ex in db_results.scalars().all():
            examples_by_id[str(ex.id)] = ex

        scored = []
        for r in results:
            ex = examples_by_id.get(r["metadata"]["example_id"])
            if ex:
                scored.append((ex, float(r.get("score", 0.5))))

        return scored

    def _keyword_search(
        self,
        profile_id: UUID,
        vision: VisionAnalysis,
    ) -> list[tuple[LearningExample, float]]:
        """Keyword-based fallback similarity search."""
        result = self.db.execute(
            select(LearningExample)
            .join(LearningSession)
            .where(
                LearningExample.strategy_profile_id == profile_id,
                LearningSession.status == "completed",
                LearningExample.is_positive_example.is_(True),
            )
            .order_by(LearningExample.created_at.desc())
            .limit(20)  # search pool
        )
        candidates = list(result.scalars().all())

        if not candidates:
            return []

        scored = [
            (ex, _keyword_similarity(vision, ex))
            for ex in candidates
        ]
        # Sort by score descending, take top 5
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:5]

    # ── formatting ────────────────────────────────────────────

    def _format_similar(
        self, scored_examples: list[tuple[LearningExample, float]]
    ) -> str:
        if not scored_examples:
            return ""

        parts: list[str] = []
        for i, (ex, score) in enumerate(scored_examples, 1):
            lines = [f"── Similar Example {i} (similarity: {score:.0%}) ──"]
            if ex.ai_interpretation:
                lines.append(ex.ai_interpretation[:500])

            ann = ex.user_annotations or {}
            if ann.get("direction"):
                lines.append(f"Direction: {ann['direction']}")
            if ann.get("setup_quality"):
                lines.append(f"Quality: {ann['setup_quality']}")
            if ann.get("reasoning"):
                lines.append(f"Reasoning: {ann['reasoning'][:200]}")

            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    # ── response parsing ──────────────────────────────────────

    def _parse_response(
        self,
        raw: str,
        scored_examples: list[tuple[LearningExample, float]],
        confidence: ProfileConfidence,
    ) -> CoplanResponse:
        """Parse LLM JSON output into CoplanResponse."""
        data = _safe_json(raw)
        if data is None:
            logger.warning("Failed to parse coplan JSON — returning raw narrative")
            return CoplanResponse(
                narrative=raw[:1500],
                confidence=confidence.overall * 0.5,  # penalise for parse failure
                warnings=["Response could not be parsed as structured JSON."],
            )

        # Build SimilarExample models
        similar_responses: list[SimilarExample] = []
        for ex, score in scored_examples[:3]:
            similar_responses.append(SimilarExample(
                example_id=ex.id,
                similarity_score=round(score, 2),
                ai_interpretation=(ex.ai_interpretation or "")[:300],
            ))

        # Merge missing_criteria into warnings
        warnings = data.get("warnings", [])
        missing = data.get("missing_criteria", [])
        if missing:
            warnings.append(
                f"Missing learned criteria: {', '.join(missing)}"
            )

        return CoplanResponse(
            setup_match_score=min(100, max(0, data.get("setup_match_score", 0))),
            setup_detected=bool(data.get("setup_detected", False)),
            entry_recommendation=data.get("entry_recommendation", ""),
            stop_recommendation=data.get("stop_recommendation", ""),
            target_recommendations=data.get("target_recommendations", []),
            risk_assessment=data.get("risk_assessment", ""),
            similar_examples=similar_responses,
            narrative=data.get("narrative", ""),
            confidence=confidence.overall,
            warnings=warnings,
        )