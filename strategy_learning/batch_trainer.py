# strategy_learning/batch_trainer.py
"""
Batch Screenshot Trainer — Multi-screenshot strategy synthesis.

Like autonomous vehicle training: feed 15-20 labeled screenshots of the
SAME strategy, and the system cross-references them to extract:
  • What's CONSISTENT across all screenshots (the real signal)
  • What VARIES (noise / context differences)
  • Contradictions that need clarification
  • Auto-generated follow-up questions

Architecture:
  1. Analyze each screenshot individually (ChartVisionAnalyzer — already exists)
  2. Aggregate all VisionAnalysis results
  3. Use LLM to cross-reference and synthesise
  4. Produce a BatchTrainingReport
  5. Feed consistent patterns into the existing profile builder

This sits between vision_analyzer (per-image) and strategy_profile_builder (synthesis).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Any, Optional

from agents.llm_client import LLMClient, LLMMessage
from config.settings import settings
from strategy_learning.models import (
    BatchTrainingReport,
    ConsistentPattern,
    StrategyContradiction,
    SynthesisedRule,
    VisionAnalysis,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  LLM Prompt for cross-referencing multiple analyses
# ──────────────────────────────────────────────────────────────

CROSS_REFERENCE_PROMPT = """\
You are a professional trading strategy analyst. A trader has provided \
{n_screenshots} chart screenshots of the SAME strategy setup to teach \
you what their strategy looks like.

Each screenshot has been individually analysed. Your job is to \
CROSS-REFERENCE all analyses and extract what is CONSISTENT \
(the real strategy signal) vs what VARIES (noise / different contexts).

═══ INDIVIDUAL ANALYSES ═══
{analyses}

═══ USER CONTEXT ═══
{user_context}

Respond with a JSON object (no markdown fences):
{{
  "consistent_patterns": [
    {{
      "pattern": "short name",
      "description": "detailed description of what's consistent",
      "frequency": 0.85,
      "category": "entry_signal|confirmation|context|structure|volume|indicator|timing",
      "importance": "critical|important|preferred|nice_to_have"
    }}
  ],
  "synthesised_rules": [
    {{
      "rule": "specific rule derived from examples",
      "confidence": 0.8,
      "evidence_count": 14,
      "category": "entry|exit|stop|regime|timing",
      "conditions": ["condition 1", "condition 2"]
    }}
  ],
  "contradictions": [
    {{
      "aspect": "what aspect differs",
      "description": "description of the contradiction",
      "variants": ["variant A", "variant B"],
      "suggested_question": "question to ask the trader to resolve this",
      "severity": "low|medium|high"
    }}
  ],
  "common_indicators": ["EMA20", "VWAP", "RSI"],
  "common_trend_direction": "bullish|bearish|sideways|mixed",
  "common_timeframe": "15m|5m|1h|etc or mixed",
  "common_market_structure": "description of typical structure",
  "volume_patterns": ["pattern 1", "pattern 2"],
  "price_level_patterns": ["pattern about S/R, VWAP proximity, etc."],
  "auto_generated_questions": [
    "Follow-up question about ambiguity 1",
    "Follow-up question about ambiguity 2"
  ],
  "synthesis_narrative": "3-5 paragraph narrative of the complete strategy as understood from all screenshots"
}}

Focus on:
1. What appears in >60% of screenshots = CONSISTENT (the strategy)
2. What appears in <30% = probably noise or edge case
3. CONTRADICTIONS = things that flip between screenshots (bullish vs bearish, different indicators)
4. Questions the trader MUST answer to resolve ambiguities
5. Be specific — "compression near VWAP with declining volume" not "consolidation pattern"
"""


class BatchTrainer:
    """
    Cross-references multiple chart screenshots to learn a strategy.

    Usage:
        trainer = BatchTrainer(llm_client=my_llm)
        report = await trainer.train(
            analyses=[va1, va2, ..., va20],
            user_context="This is my compression breakout strategy",
        )
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client

    def _get_client(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(
                provider="anthropic",
                model=settings.VISION_MODEL,
            )
        return self._llm

    async def train(
        self,
        analyses: list[VisionAnalysis],
        user_context: str = "",
        annotations: Optional[list[dict[str, Any]]] = None,
    ) -> BatchTrainingReport:
        """
        Run batch training on multiple screenshot analyses.

        Args:
            analyses: List of VisionAnalysis from individual screenshots.
            user_context: User's description of what this strategy is.
            annotations: Optional per-screenshot annotations from the user.

        Returns:
            BatchTrainingReport with consistent patterns, contradictions,
            synthesised rules, and follow-up questions.
        """
        n_total = len(analyses)
        valid_analyses = [a for a in analyses if not a.is_empty]
        n_valid = len(valid_analyses)
        n_failed = n_total - n_valid

        logger.info(
            "Batch training: %d screenshots (%d valid, %d empty)",
            n_total, n_valid, n_failed,
        )

        if n_valid < 3:
            return BatchTrainingReport(
                total_screenshots=n_total,
                analysed_successfully=n_valid,
                failed_analyses=n_failed,
                consistency_score=0.0,
                synthesis_narrative="Too few valid screenshots to synthesise a strategy.",
                auto_generated_questions=[
                    "Please provide at least 10 chart screenshots showing your strategy setup.",
                ],
            )

        # Step 1: Statistical aggregation (fast, no LLM needed)
        stat_report = self._statistical_aggregation(valid_analyses)

        # Step 2: LLM cross-reference synthesis
        llm_report = await self._llm_cross_reference(
            valid_analyses, user_context, annotations
        )

        # Step 3: Merge statistical and LLM insights
        report = self._merge_reports(stat_report, llm_report, n_total, n_valid, n_failed)

        logger.info(
            "Batch training complete: %.0f%% consistency, %d patterns, %d contradictions, %d questions",
            report.consistency_score * 100,
            len(report.consistent_patterns),
            len(report.contradictions),
            len(report.auto_generated_questions),
        )

        return report

    # ── Step 1: Statistical Aggregation ──────────────────────

    def _statistical_aggregation(
        self, analyses: list[VisionAnalysis]
    ) -> dict[str, Any]:
        """
        Fast statistical pass — no LLM, just counting occurrences.

        Extracts:
        - Indicator frequency (what indicators appear in >50% of charts?)
        - Trend direction consensus
        - Timeframe consensus
        - Common candle patterns
        - Volume behavior consensus
        """
        n = len(analyses)

        # Indicators
        indicator_counter: Counter = Counter()
        for a in analyses:
            for ind in a.indicators_visible:
                indicator_counter[ind.upper().strip()] += 1

        common_indicators = [
            ind for ind, count in indicator_counter.most_common()
            if count >= n * 0.4  # appears in 40%+ of screenshots
        ]

        # Trend direction
        trend_counter: Counter = Counter()
        for a in analyses:
            if a.trend_direction:
                trend_counter[a.trend_direction] += 1
        most_common_trend = trend_counter.most_common(1)
        trend_consensus = most_common_trend[0][0] if most_common_trend else ""
        trend_agreement = (
            most_common_trend[0][1] / n if most_common_trend else 0
        )

        # Timeframe
        tf_counter: Counter = Counter()
        for a in analyses:
            if a.timeframe_detected:
                tf_counter[a.timeframe_detected] += 1
        most_common_tf = tf_counter.most_common(1)
        tf_consensus = most_common_tf[0][0] if most_common_tf else ""

        # Market structure
        structure_counter: Counter = Counter()
        for a in analyses:
            if a.market_structure:
                structure_counter[a.market_structure] += 1
        most_common_structure = structure_counter.most_common(1)
        structure_consensus = (
            most_common_structure[0][0] if most_common_structure else ""
        )

        # Candle pattern frequency
        pattern_counter: Counter = Counter()
        for a in analyses:
            for cp in a.candle_patterns:
                pattern_counter[cp.pattern.lower().strip()] += 1

        common_patterns = [
            (pat, count)
            for pat, count in pattern_counter.most_common()
            if count >= max(2, n * 0.3)
        ]

        # Volume patterns
        volume_descriptions: list[str] = []
        for a in analyses:
            if a.volume_profile:
                volume_descriptions.append(a.volume_profile)

        # Consistency score
        consistency_factors = []
        if trend_agreement > 0:
            consistency_factors.append(trend_agreement)
        if common_indicators:
            consistency_factors.append(
                min(1.0, len(common_indicators) / 3)
            )
        if common_patterns:
            consistency_factors.append(
                min(1.0, len(common_patterns) / 3)
            )
        if tf_counter and most_common_tf:
            consistency_factors.append(most_common_tf[0][1] / n)

        consistency_score = (
            sum(consistency_factors) / len(consistency_factors)
            if consistency_factors
            else 0.0
        )

        return {
            "common_indicators": common_indicators,
            "common_trend_direction": trend_consensus,
            "trend_agreement": trend_agreement,
            "common_timeframe": tf_consensus,
            "common_market_structure": structure_consensus,
            "common_candle_patterns": common_patterns,
            "volume_descriptions": volume_descriptions,
            "consistency_score": round(consistency_score, 3),
        }

    # ── Step 2: LLM Cross-Reference ─────────────────────────

    async def _llm_cross_reference(
        self,
        analyses: list[VisionAnalysis],
        user_context: str,
        annotations: Optional[list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        LLM-based cross-referencing of all screenshot analyses.

        Sends all individual VisionAnalysis texts to the LLM,
        which finds patterns, contradictions, and generates rules.
        """
        # Build analysis summaries
        analysis_texts = []
        for i, a in enumerate(analyses):
            text = f"--- Screenshot {i+1} ---\n{a.to_text()}"
            if annotations and i < len(annotations) and annotations[i]:
                ann = annotations[i]
                ann_parts = []
                if ann.get("direction"):
                    ann_parts.append(f"Direction: {ann['direction']}")
                if ann.get("reasoning"):
                    ann_parts.append(f"Reasoning: {ann['reasoning']}")
                if ann.get("what_makes_this_work"):
                    ann_parts.append(f"Edge: {ann['what_makes_this_work']}")
                if ann.get("setup_quality"):
                    ann_parts.append(f"Quality: {ann['setup_quality']}")
                if ann_parts:
                    text += f"\nUser annotations: {'; '.join(ann_parts)}"
            analysis_texts.append(text)

        analyses_str = "\n\n".join(analysis_texts)

        prompt = CROSS_REFERENCE_PROMPT.format(
            n_screenshots=len(analyses),
            analyses=analyses_str,
            user_context=user_context or "No additional context provided.",
        )

        from agents.prompts import STRATEGY_ANALYST

        client = self._get_client()
        messages = [
            LLMMessage(role="system", content=STRATEGY_ANALYST),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = await client.chat(
                messages=messages,
                temperature=0.3,
                max_tokens=4000,
                caller="batch_trainer",
            )
            return self._parse_llm_response(response.content)
        except Exception as e:
            logger.error("LLM cross-reference failed: %s", e)
            return {}

    def _parse_llm_response(self, raw: str) -> dict[str, Any]:
        """Parse the LLM's JSON response."""
        content = raw.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM cross-reference as JSON")
            return {"synthesis_narrative": raw[:2000]}

    # ── Step 3: Merge Reports ────────────────────────────────

    def _merge_reports(
        self,
        stat: dict[str, Any],
        llm: dict[str, Any],
        n_total: int,
        n_valid: int,
        n_failed: int,
    ) -> BatchTrainingReport:
        """Merge statistical aggregation and LLM synthesis into final report."""

        # Consistent patterns from LLM
        consistent_patterns = []
        for p in llm.get("consistent_patterns", []):
            try:
                consistent_patterns.append(ConsistentPattern(
                    pattern=p.get("pattern", "unknown"),
                    description=p.get("description", ""),
                    frequency=p.get("frequency", 0.5),
                    importance=p.get("importance", "important"),
                    category=p.get("category", ""),
                ))
            except Exception:
                continue

        # Synthesised rules from LLM
        synthesised_rules = []
        for r in llm.get("synthesised_rules", []):
            try:
                synthesised_rules.append(SynthesisedRule(
                    rule=r.get("rule", "unknown"),
                    confidence=r.get("confidence", 0.5),
                    evidence_count=r.get("evidence_count", 0),
                    category=r.get("category", ""),
                    conditions=r.get("conditions", []),
                ))
            except Exception:
                continue

        # Contradictions from LLM
        contradictions = []
        for c in llm.get("contradictions", []):
            try:
                contradictions.append(StrategyContradiction(
                    aspect=c.get("aspect", "unknown"),
                    description=c.get("description", ""),
                    variants=c.get("variants", []),
                    suggested_question=c.get("suggested_question", ""),
                    severity=c.get("severity", "medium"),
                ))
            except Exception:
                continue

        # Auto-generated questions: merge LLM questions + contradiction questions
        questions = list(llm.get("auto_generated_questions", []))
        for c in contradictions:
            if c.suggested_question and c.suggested_question not in questions:
                questions.append(c.suggested_question)

        # Add questions for low-coverage areas
        covered_categories = {p.category for p in consistent_patterns}
        essential = {"entry_signal", "confirmation", "volume", "structure"}
        for missing in essential - covered_categories:
            questions.append(
                f"Your screenshots don't clearly show a consistent {missing.replace('_', ' ')} pattern. "
                f"Can you describe or show examples of what you look for?"
            )

        # Consistency score: blend stat and LLM
        stat_consistency = stat.get("consistency_score", 0.0)
        llm_consistency = (
            len(consistent_patterns) / max(1, len(consistent_patterns) + len(contradictions))
        )
        consistency_score = round(stat_consistency * 0.4 + llm_consistency * 0.6, 3)

        # Coverage score
        all_categories = {p.category for p in consistent_patterns}
        all_categories |= {r.category for r in synthesised_rules}
        full_categories = {"entry", "exit", "stop", "regime", "timing",
                           "entry_signal", "confirmation", "volume", "structure"}
        coverage_score = round(
            len(all_categories & full_categories) / len(full_categories), 3
        )

        # Volume patterns: merge stat + LLM
        volume_patterns = llm.get("volume_patterns", [])
        if not volume_patterns and stat.get("volume_descriptions"):
            volume_patterns = stat["volume_descriptions"][:3]

        return BatchTrainingReport(
            total_screenshots=n_total,
            analysed_successfully=n_valid,
            failed_analyses=n_failed,
            consistent_patterns=consistent_patterns,
            synthesised_rules=synthesised_rules,
            contradictions=contradictions,
            common_indicators=llm.get("common_indicators", stat.get("common_indicators", [])),
            common_trend_direction=llm.get("common_trend_direction", stat.get("common_trend_direction", "")),
            common_timeframe=llm.get("common_timeframe", stat.get("common_timeframe", "")),
            common_market_structure=llm.get("common_market_structure", stat.get("common_market_structure", "")),
            volume_patterns=volume_patterns,
            price_level_patterns=llm.get("price_level_patterns", []),
            auto_generated_questions=questions,
            consistency_score=consistency_score,
            coverage_score=coverage_score,
            synthesis_narrative=llm.get("synthesis_narrative", ""),
        )
