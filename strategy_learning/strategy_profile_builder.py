# strategy_learning/strategy_profile_builder.py
"""
Profile Builder — Constructs StrategyProfile from accumulated examples.

After a teaching session completes, this module:
1. Loads all examples for the strategy
2. Uses LLM to synthesise a narrative understanding
3. Extracts entry criteria, exit rules, regime preferences
4. Computes confidence scores based on example coverage
5. Embeds examples in vector store for similarity search
6. Updates the StrategyProfile in the database
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.llm_client import LLMClient, LLMMessage
from config.settings import settings
from core.events import Event, EventType, get_event_bus
from core.models import LearningExample, LearningSession, StrategyProfile
from strategy_learning.models import (
    LearnedEntryCriterion,
    LearnedExitRules,
    ProfileConfidence,
    UserAnnotation,
    VisionAnalysis,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  LLM Prompt Templates
# ──────────────────────────────────────────────────────────────

NARRATIVE_PROMPT = """\
You are analysing a trader's personal strategy based on {example_count} \
annotated chart examples they have provided.

Each example includes an AI chart analysis and the trader's own annotations \
about their entry/exit logic, reasoning, and edge.

Synthesise a clear, *specific* narrative of THIS trader's strategy.  Cover:
1. Market conditions and context the strategy targets
2. The specific entry trigger and confirmation signals
3. Stop-loss placement logic
4. Target / exit approach (including partials and time-based exits)
5. The key edge conditions that make this strategy profitable
6. What the trader deliberately avoids

Write in second person ("You look for …") as if coaching the trader.
Be concrete — reference patterns, levels, and indicators seen in the examples.
Do NOT give generic trading advice.

EXAMPLES:
{examples}

Write the strategy narrative (3-5 paragraphs):"""

CRITERIA_PROMPT = """\
You are extracting the recurring ENTRY CRITERIA from {example_count} \
annotated trading examples.

Focus on conditions that appear in MULTIPLE examples.  Assign importance:
- critical   → appears in ≥80 % of positive examples
- important  → appears in 60-80 %
- preferred  → appears in 40-60 %
- nice_to_have → <40 % but explicitly mentioned by the trader

Return a JSON array (no markdown fences):
[
  {{
    "criterion": "short name",
    "description": "detailed description",
    "importance": "critical|important|preferred|nice_to_have",
    "frequency": 0.85
  }}
]

EXAMPLES:
{examples}

Return ONLY the JSON array:"""

EXIT_RULES_PROMPT = """\
You are extracting EXIT RULES (stop loss, targets, trailing stops, time exits) \
from {example_count} annotated trading examples.

Return a JSON object (no markdown fences):
{{
  "stop_method": "how stops are placed (e.g. 'below previous swing low')",
  "typical_stop_atr": null or float,
  "stop_description": "detailed stop-placement description",
  "target_patterns": [
    {{"target": "1R", "action": "take 50 % off"}},
    {{"target": "2R", "action": "trail remainder"}}
  ],
  "trailing_rules": "trailing-stop description or empty string",
  "time_exit": "time-based exit rule or empty string",
  "partial_exit_rules": ["rule 1", "rule 2"]
}}

EXAMPLES:
{examples}

Return ONLY the JSON object:"""

# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

_MAX_LLM_RETRIES = 2


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that LLMs love to add."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (```json) and last line (```)
        start = 1
        end = len(lines)
        if lines[-1].strip() == "```":
            end = -1
        text = "\n".join(lines[start:end])
    return text.strip()


def _safe_parse_json(raw: str, fallback: Any = None) -> Any:
    """Parse JSON from LLM output, tolerating markdown fences."""
    cleaned = _strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed.  Raw (first 300 chars): %s", raw[:300])
        return fallback


# ──────────────────────────────────────────────────────────────
#  ProfileBuilder
# ──────────────────────────────────────────────────────────────


class ProfileBuilder:
    """Builds and rebuilds StrategyProfile from accumulated examples."""

    def __init__(
        self,
        db: Session,
        llm: Optional[LLMClient] = None,
        vector_store: Any = None,
        embedding_provider: Any = None,
    ):
        self.db = db
        self._llm = llm
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(
                provider=settings.LLM_PROVIDER,
                model=settings.LLM_MODEL,
            )
        return self._llm

    # ── main entry point ──────────────────────────────────────

    async def build_profile(self, profile_id: UUID) -> StrategyProfile:
        """
        Rebuild the full strategy profile from every completed example.
        """
        profile = self.db.get(StrategyProfile, profile_id)
        if profile is None:
            raise ValueError(f"Profile not found: {profile_id}")

        examples = self._load_examples(profile_id)
        if not examples:
            logger.warning("No completed examples for profile %s", profile_id)
            return profile

        n = len(examples)
        logger.info("Building profile '%s' from %d examples", profile.name, n)

        formatted = self._format_examples(examples)

        # Synthesise components (sequential — each is independent)
        narrative = await self._synthesize_narrative(formatted, n)
        entry_criteria = await self._extract_entry_criteria(formatted, n, examples)
        exit_rules = await self._extract_exit_rules(formatted, n)
        regime_prefs = self._compute_regime_preferences(examples)
        time_prefs = self._compute_time_preferences(examples)
        risk_params = self._compute_risk_parameters(examples)
        confidence = self._compute_confidence(examples, entry_criteria, exit_rules)

        # Embed for similarity search
        await self._embed_examples(examples, profile_id)

        # Persist
        profile.narrative = narrative
        profile.entry_criteria = [c.model_dump() for c in entry_criteria]
        profile.exit_rules = exit_rules.model_dump()
        profile.regime_preferences = regime_prefs
        profile.time_preferences = time_prefs
        profile.risk_parameters = risk_params
        profile.confidence_scores = confidence.model_dump()
        profile.examples_count = n
        profile.last_rebuilt_at = datetime.now(timezone.utc)

        min_examples = getattr(settings, "LEARNING_MIN_EXAMPLES", 5)
        profile.status = "active" if n >= min_examples else "learning"

        self.db.commit()
        self.db.refresh(profile)

        get_event_bus().publish(Event(
            event_type=EventType.STRATEGY_PROFILE_UPDATED,
            source="profile_builder",
            strategy_id=str(profile_id),
            data={
                "examples_count": n,
                "confidence": confidence.model_dump(),
                "status": profile.status,
            },
        ))

        logger.info(
            "Profile '%s' rebuilt — %d examples, confidence=%.2f, status=%s",
            profile.name, n, confidence.overall, profile.status,
        )
        return profile

    # ── data loading ──────────────────────────────────────────

    def _load_examples(self, profile_id: UUID) -> list[LearningExample]:
        result = self.db.execute(
            select(LearningExample)
            .join(LearningSession)
            .where(
                LearningExample.strategy_profile_id == profile_id,
                LearningSession.status == "completed",
            )
            .order_by(LearningExample.created_at)
        )
        return list(result.scalars().all())

    def _format_examples(self, examples: list[LearningExample]) -> str:
        parts: list[str] = []
        for i, ex in enumerate(examples, 1):
            lines = [f"═══ Example {i}/{len(examples)} ═══"]

            if ex.ai_interpretation:
                lines.append(f"AI Interpretation: {ex.ai_interpretation}")

            ann = ex.user_annotations or {}
            for key, label in [
                ("direction", "Direction"),
                ("entry_price", "Entry"),
                ("stop_price", "Stop"),
                ("target_prices", "Targets"),
                ("reasoning", "Reasoning"),
                ("what_makes_this_work", "Edge"),
                ("what_would_invalidate", "Invalidation"),
                ("setup_quality", "Quality"),
            ]:
                val = ann.get(key)
                if val:
                    lines.append(f"  {label}: {val}")

            vision = ex.vision_analysis or {}
            if vision.get("trend_direction"):
                lines.append(f"  Chart Trend: {vision['trend_direction']}")
            if vision.get("overall_context"):
                lines.append(f"  Chart Context: {vision['overall_context']}")

            ctx = ex.market_context or {}
            if ctx.get("regime"):
                lines.append(f"  Market Regime: {ctx['regime']}")
            if ctx.get("time"):
                lines.append(f"  Time of Day: {ctx['time']}")

            if not ex.is_positive_example:
                lines.append("  ⚠ COUNTER-EXAMPLE — this is NOT a valid setup")

            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    # ── LLM synthesis ─────────────────────────────────────────

    async def _llm_call(
        self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 1200
    ) -> str:
        """LLM call with retry."""
        from agents.prompts import STRATEGY_ANALYST

        llm = self._get_llm()
        last_error: Optional[Exception] = None

        for attempt in range(_MAX_LLM_RETRIES + 1):
            try:
                resp = await llm.chat(
                    messages=[
                        LLMMessage(role="system", content=STRATEGY_ANALYST),
                        LLMMessage(role="user", content=prompt),
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    caller="profile_builder",
                )
                return resp.content.strip()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1, _MAX_LLM_RETRIES + 1, exc,
                )

        raise RuntimeError(
            f"LLM call failed after {_MAX_LLM_RETRIES + 1} attempts"
        ) from last_error

    async def _synthesize_narrative(self, formatted: str, count: int) -> str:
        prompt = NARRATIVE_PROMPT.format(examples=formatted, example_count=count)
        return await self._llm_call(prompt, temperature=0.35, max_tokens=1500)

    async def _extract_entry_criteria(
        self, formatted: str, count: int, examples: list[LearningExample]
    ) -> list[LearnedEntryCriterion]:
        prompt = CRITERIA_PROMPT.format(examples=formatted, example_count=count)
        raw = await self._llm_call(prompt, temperature=0.15, max_tokens=1000)
        data = _safe_parse_json(raw, fallback=[])
        if not isinstance(data, list):
            return []

        total_examples = len(examples)
        criteria: list[LearnedEntryCriterion] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            freq = float(item.get("frequency", 0.5))
            criteria.append(LearnedEntryCriterion(
                criterion=item.get("criterion", "unknown"),
                description=item.get("description", ""),
                importance=item.get("importance", "important"),
                confidence=round(freq, 2),
                # BUG FIX: was `len(data)` — now correctly uses total examples
                example_count=max(1, round(freq * total_examples)),
            ))
        return criteria

    async def _extract_exit_rules(self, formatted: str, count: int) -> LearnedExitRules:
        prompt = EXIT_RULES_PROMPT.format(examples=formatted, example_count=count)
        raw = await self._llm_call(prompt, temperature=0.15, max_tokens=800)
        data = _safe_parse_json(raw, fallback={})
        if not isinstance(data, dict):
            return LearnedExitRules()
        return LearnedExitRules(**data)

    # ── statistical analysis ──────────────────────────────────

    def _compute_regime_preferences(
        self, examples: list[LearningExample]
    ) -> dict[str, Any]:
        regime_counts: dict[str, int] = {}
        positive_in_regime: dict[str, int] = {}

        for ex in examples:
            regime = (ex.market_context or {}).get("regime", "")
            if not regime:
                continue
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            if ex.is_positive_example:
                positive_in_regime[regime] = positive_in_regime.get(regime, 0) + 1

        preferred, avoid, neutral = [], [], []
        for regime, count in regime_counts.items():
            pos_ratio = positive_in_regime.get(regime, 0) / count if count else 0
            entry = {
                "regime": regime,
                "example_count": count,
                "positive_ratio": round(pos_ratio, 2),
            }
            if pos_ratio >= 0.6:
                preferred.append(entry)
            elif pos_ratio < 0.3 and count >= 2:
                avoid.append(entry)
            else:
                neutral.append(entry)

        return {"preferred": preferred, "avoid": avoid, "neutral": neutral}

    def _compute_time_preferences(
        self, examples: list[LearningExample]
    ) -> dict[str, Any]:
        hour_counts: dict[int, int] = {}
        time_labels: dict[str, int] = {}

        for ex in examples:
            ctx = ex.market_context or {}
            time_str = ctx.get("time", "")
            if time_str:
                time_labels[time_str] = time_labels.get(time_str, 0) + 1
                # Try to extract hour
                try:
                    hour = int(time_str.split(":")[0])
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                except (ValueError, IndexError):
                    pass

        peak_hours = sorted(hour_counts, key=hour_counts.get, reverse=True)[:3]  # type: ignore[arg-type]
        return {
            "time_distribution": time_labels,
            "hour_distribution": hour_counts,
            "peak_hours": peak_hours,
        }

    def _compute_risk_parameters(
        self, examples: list[LearningExample]
    ) -> dict[str, Any]:
        stop_distances: list[float] = []
        rr_ratios: list[float] = []
        risk_pcts: list[float] = []

        for ex in examples:
            ann = ex.user_annotations or {}
            entry = ann.get("entry_price")
            stop = ann.get("stop_price")
            targets = ann.get("target_prices", [])
            risk_pct = ann.get("risk_pct")  # if user specifies

            if risk_pct is not None:
                risk_pcts.append(float(risk_pct))

            if entry is not None and stop is not None:
                try:
                    entry_f = float(entry)
                    stop_f = float(stop)
                except (TypeError, ValueError):
                    continue

                distance = abs(entry_f - stop_f)
                if distance <= 0:
                    continue

                stop_distances.append(distance)

                # Compute distance as percentage of entry
                pct_risk = (distance / entry_f) * 100 if entry_f > 0 else 0
                if pct_risk > 0:
                    risk_pcts.append(round(pct_risk, 3))

                # R:R from first target
                if targets and isinstance(targets, list):
                    try:
                        primary_target = float(targets[0])
                        reward = abs(primary_target - entry_f)
                        rr = reward / distance
                        if 0.1 <= rr <= 50:  # sanity bounds
                            rr_ratios.append(round(rr, 2))
                    except (TypeError, ValueError, IndexError):
                        pass

        result: dict[str, Any] = {}

        if stop_distances:
            sorted_dists = sorted(stop_distances)
            result["avg_stop_distance"] = round(
                sum(stop_distances) / len(stop_distances), 4
            )
            result["median_stop_distance"] = round(
                sorted_dists[len(sorted_dists) // 2], 4
            )
            result["stop_sample_count"] = len(stop_distances)

        if rr_ratios:
            sorted_rr = sorted(rr_ratios)
            result["avg_rr_ratio"] = round(sum(rr_ratios) / len(rr_ratios), 2)
            result["typical_rr"] = round(sorted_rr[len(sorted_rr) // 2], 2)
            result["min_rr"] = sorted_rr[0]
            result["max_rr"] = sorted_rr[-1]
            result["rr_sample_count"] = len(rr_ratios)

        if risk_pcts:
            sorted_pcts = sorted(risk_pcts)
            result["typical_risk_pct"] = round(
                sorted_pcts[len(sorted_pcts) // 2], 3
            )
            result["avg_risk_pct"] = round(
                sum(risk_pcts) / len(risk_pcts), 3
            )

        return result

    # ── confidence scoring ────────────────────────────────────

    def _compute_confidence(
        self,
        examples: list[LearningExample],
        criteria: list[LearnedEntryCriterion],
        exit_rules: LearnedExitRules,
    ) -> ProfileConfidence:
        """
        Compute per-aspect confidence based on:
        - Number of examples (more = higher ceiling)
        - Annotation completeness per aspect
        - LLM extraction quality
        """
        n = len(examples)
        if n == 0:
            return ProfileConfidence()

        # Coverage factor: scales from 0→1 as examples go from 0→20+
        coverage = min(1.0, n / 20)

        # ── entry confidence ──
        entry_conf = 0.0
        if criteria:
            avg_crit_conf = sum(c.confidence for c in criteria) / len(criteria)
            # Blend: 40% from having enough examples, 60% from criterion quality
            entry_conf = min(1.0, coverage * 0.4 + avg_crit_conf * 0.6)
        else:
            entry_conf = coverage * 0.3  # have examples but couldn't extract criteria

        # ── stop confidence ──
        stops_annotated = sum(
            1 for ex in examples
            if (ex.user_annotations or {}).get("stop_price") is not None
        )
        stop_conf = min(1.0, (stops_annotated / n) * 0.7 + coverage * 0.3)

        # ── exit confidence ──
        exit_conf = exit_rules.completeness

        # ── regime confidence ──
        regime_count = sum(
            1 for ex in examples
            if (ex.market_context or {}).get("regime")
        )
        regime_conf = min(1.0, (regime_count / n) if n else 0)

        # ── timing confidence ──
        time_count = sum(
            1 for ex in examples
            if (ex.market_context or {}).get("time")
        )
        timing_conf = min(1.0, (time_count / n) if n else 0)

        # ── risk confidence ──
        full_plan_count = sum(
            1 for ex in examples
            if (
                (ex.user_annotations or {}).get("stop_price") is not None
                and (ex.user_annotations or {}).get("target_prices")
                and (ex.user_annotations or {}).get("entry_price") is not None
            )
        )
        risk_conf = min(1.0, (full_plan_count / n) if n else 0)

        return ProfileConfidence(
            entry=round(entry_conf, 3),
            exit=round(exit_conf, 3),
            stop=round(stop_conf, 3),
            regime=round(regime_conf, 3),
            timing=round(timing_conf, 3),
            risk=round(risk_conf, 3),
            # overall is auto-computed by the model_validator
        )

    # ── vector embedding ──────────────────────────────────────

    async def _embed_examples(
        self, examples: list[LearningExample], profile_id: UUID
    ) -> None:
        """Embed example interpretations for similarity search."""
        if not self._embedding_provider or not self._vector_store:
            logger.debug("No embedding/vector store configured — skipping")
            return

        collection = getattr(
            settings, "LEARNING_VECTOR_COLLECTION", "learning_examples"
        )
        embedded_count = 0

        for ex in examples:
            if ex.embedding_id:
                continue  # already embedded

            text = ex.ai_interpretation or ""
            if len(text) < 20:
                continue  # too short to embed meaningfully

            try:
                embedding = await self._embedding_provider.embed(text)
                doc_id = f"learning_example_{ex.id}"

                await self._vector_store.upsert(
                    collection=collection,
                    doc_id=doc_id,
                    embedding=embedding,
                    metadata={
                        "profile_id": str(profile_id),
                        "example_id": str(ex.id),
                        "is_positive": ex.is_positive_example,
                        "direction": (ex.user_annotations or {}).get("direction", ""),
                        "regime": (ex.market_context or {}).get("regime", ""),
                    },
                    text=text,
                )

                ex.embedding_id = doc_id
                embedded_count += 1
            except Exception as exc:
                logger.warning("Failed to embed example %s: %s", ex.id, exc)

        if embedded_count:
            self.db.commit()
            logger.info("Embedded %d/%d examples", embedded_count, len(examples))

    # ── coaching questions ────────────────────────────────────

    async def get_next_questions(self, profile_id: UUID) -> list[str]:
        """
        Generate AI-driven questions targeting low-confidence areas
        so the trader knows *exactly* what to teach next.
        """
        profile = self.db.get(StrategyProfile, profile_id)
        if not profile or not profile.confidence_scores:
            return [
                "Share a chart screenshot of a setup you would take with this strategy.",
                "Mark where your stop loss goes and explain why.",
                "What's your target? Do you scale out in partials?",
                "What market condition makes this setup work best?",
            ]

        confidence = ProfileConfidence(**profile.confidence_scores)
        weak = confidence.weak_areas(threshold=0.5)

        _question_bank: dict[str, list[str]] = {
            "entry": [
                "Share 2-3 more examples showing your ideal entry trigger.",
                "What specific candle pattern or price-action signal tells you to enter?",
                "How do you confirm the entry — what must be true on the chart?",
            ],
            "exit": [
                "Walk me through how you decide to exit — partial and full.",
                "Do you have any time-based exit rules (e.g. close by EOD)?",
                "When a trade goes in your favour, how do you manage it?",
            ],
            "stop": [
                "Show me on a chart where exactly you'd place the stop and why.",
                "Is your stop based on structure, ATR, fixed points, or something else?",
                "Do you ever move your stop? Under what conditions?",
            ],
            "regime": [
                "In what market conditions does this strategy shine?",
                "Can you mark the market regime (trending/ranging/volatile) for each example?",
                "Are there conditions where you deliberately skip this setup?",
            ],
            "timing": [
                "What time of day do you typically see these setups?",
                "Do you avoid certain hours or sessions?",
                "Is there a 'golden window' when this strategy works best?",
            ],
            "risk": [
                "What percentage of your capital do you risk per trade?",
                "Annotate entry, stop, AND target on your next example.",
                "What's your minimum acceptable risk-reward ratio for this setup?",
            ],
        }

        questions: list[str] = []
        for area in weak:
            bank = _question_bank.get(area, [])
            # Pick first 2 from each weak area
            questions.extend(bank[:2])

        if not questions:
            # Everything above threshold — push for excellence
            questions = [
                "Great coverage! Share more A+ setups to sharpen the profile.",
                "Can you share a counter-example — a chart that looks like this "
                "setup but you would NOT take?",
            ]

        return questions[:6]