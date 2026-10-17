# bootstrap/nl_interpreter.py
"""
LLM Strategy Interpreter
==========================

Converts user's natural language description + screenshots
into a structured strategy draft.

Flow:
1. User describes strategy (text/voice)
2. LLM extracts structured components
3. Maps to Pattern Library template(s)
4. Generates customized teaching questions
5. User answers questions
6. LLM refines understanding
7. Produces initial SKO draft

The LLM does NOT replace domain logic.
It's a TRANSLATOR between human language and system structures.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bootstrap.pattern_library import (
    PatternLibrary,
    PatternTemplate,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedComponent:
    """A component extracted from user's description."""
    name: str
    raw_text: str              # what user said
    component_type: str        # "entry_rule" | "exit_rule" | "filter" | "sizing"
    confidence: float          # LLM's confidence in extraction
    is_quantifiable: bool
    extracted_params: Dict[str, Any] = field(default_factory=dict)
    ambiguities: List[str] = field(default_factory=list)


@dataclass
class StrategyDraft:
    """
    Initial structured strategy before user refinement.
    This is the STARTING POINT, not the final SKO.
    """
    strategy_name: str = ""
    matched_templates: List[str] = field(default_factory=list)
    match_confidence: float = 0.0

    # Extracted from user
    extracted_components: List[ExtractedComponent] = field(
        default_factory=list
    )

    # From template + user customization
    hard_rules: List[Dict[str, Any]] = field(default_factory=list)
    soft_scorers: List[Dict[str, Any]] = field(default_factory=list)
    context_factors: List[Dict[str, Any]] = field(
        default_factory=list
    )
    uncertainty_factors: List[Dict[str, Any]] = field(
        default_factory=list
    )
    exit_logic: List[Dict[str, Any]] = field(default_factory=list)
    feature_keys: List[str] = field(default_factory=list)
    initial_centroids: List[Dict[str, float]] = field(
        default_factory=list
    )

    # Strategy profile params
    expected_trades_per_day: float = 2.0
    expected_hold_minutes: float = 120.0
    allowed_directions: List[str] = field(
        default_factory=lambda: ["long"]
    )
    timeframe: str = "5min"

    # Quality metadata
    understanding_confidence: float = 0.0
    gaps: List[str] = field(default_factory=list)
    questions_to_ask: List[Dict[str, str]] = field(
        default_factory=list
    )
    warnings: List[str] = field(default_factory=list)


# The system prompt for LLM extraction
EXTRACTION_SYSTEM_PROMPT = """You are a trading strategy analyst.
Your job is to extract structured strategy information from a
trader's natural language description.

You must extract:
1. ENTRY CONDITIONS: What conditions must be met to enter?
   For each condition, determine if it's:
   - QUANTIFIABLE (has a number: "RSI below 30")
   - PATTERN-BASED (recognizable: "compression pattern")
   - QUALITY-BASED (subjective: "clean price action")

2. EXIT CONDITIONS: How does the trader exit?
   - Stop loss placement
   - Target methodology
   - Time-based exits
   - Trailing stops
   - Story-broken exits

3. FILTERS: What makes them SKIP a setup that technically qualifies?

4. CONTEXT: What market conditions do they prefer?

5. SIZING: How do they determine position size?

For each extracted component, rate your confidence 0-1.
Flag any AMBIGUITIES where you're unsure what the trader means.

CRITICAL: Do NOT add conditions the trader didn't mention.
If they didn't say it, leave it as a gap to ask about.
Being wrong is worse than being incomplete.

Respond in JSON format:
{
    "strategy_name": "...",
    "category": "breakout|reversal|momentum|mean_reversion|other",
    "entry_conditions": [
        {
            "description": "what user said",
            "type": "quantifiable|pattern|quality|context",
            "confidence": 0.0-1.0,
            "params": {"field": "...", "operator": "...", "value": ...},
            "ambiguities": ["..."]
        }
    ],
    "exit_conditions": [...],
    "filters": [...],
    "context_preferences": [...],
    "sizing_method": "...",
    "timeframe": "...",
    "typical_hold_time": "...",
    "direction": "long_only|short_only|both",
    "overall_understanding_confidence": 0.0-1.0,
    "key_gaps": ["things that are unclear"],
    "similar_to": ["known pattern names this resembles"]
}"""


class LLMStrategyInterpreter:
    """
    Uses LLM + Pattern Library to interpret user strategies.
    
    The LLM extracts structure. The library provides templates.
    Combined: intelligent bootstrapping.
    """

    def __init__(
        self,
        library: PatternLibrary,
        llm_client: Any = None,   # OpenAI/Anthropic/local client
    ):
        self._library = library
        self._llm = llm_client

    async def interpret(
        self,
        user_description: str,
        screenshots: Optional[List[Any]] = None,
        trade_examples: Optional[List[Dict]] = None,
    ) -> StrategyDraft:
        """
        Main interpretation pipeline.
        
        1. LLM extracts structured components
        2. Map to library templates
        3. Merge template defaults with user specifics
        4. Identify gaps
        5. Generate questions
        """
        # Step 1: LLM extraction
        extraction = await self._llm_extract(
            user_description, screenshots, trade_examples
        )

        # Step 2: Template matching
        matched_templates = self._match_templates(extraction)

        # Step 3: Build draft
        draft = self._build_draft(extraction, matched_templates)

        # Step 4: Identify gaps
        draft.gaps = self._identify_gaps(draft, extraction)

        # Step 5: Generate questions
        draft.questions_to_ask = self._generate_questions(
            draft, matched_templates, extraction
        )

        # Step 6: Warnings
        draft.warnings = self._generate_warnings(draft, extraction)

        return draft

    async def refine(
        self,
        draft: StrategyDraft,
        user_answers: List[Dict[str, str]],
    ) -> StrategyDraft:
        """
        Refine draft based on user's answers to questions.
        Can be called multiple times (iterative refinement).
        """
        # LLM interprets answers
        refinements = await self._llm_interpret_answers(
            draft, user_answers
        )

        # Apply refinements
        for refinement in refinements:
            field_name = refinement.get("field")
            value = refinement.get("value")
            action = refinement.get("action", "update")

            if action == "add_hard_rule" and value:
                draft.hard_rules.append(value)
            elif action == "update_soft_scorer" and field_name:
                for scorer in draft.soft_scorers:
                    if scorer["name"] == field_name:
                        scorer.update(value)
            elif action == "add_uncertainty":
                draft.uncertainty_factors.append(value)
            elif action == "update_exit":
                draft.exit_logic.append(value)
            elif action == "set_param":
                setattr(draft, field_name, value)

        # Recalculate confidence
        draft.understanding_confidence = self._compute_confidence(
            draft
        )

        # Regenerate remaining gaps/questions
        draft.gaps = [
            g for g in draft.gaps
            if not any(
                a.get("question_id") == g for a in user_answers
            )
        ]

        return draft

    # ── LLM Calls ─────────────────────────────────────────────────

    async def _llm_extract(
        self,
        description: str,
        screenshots: Optional[List[Any]],
        examples: Optional[List[Dict]],
    ) -> Dict[str, Any]:
        """Extract structured strategy from natural language."""
        if self._llm is None:
            return self._fallback_extract(description)

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(
                description, screenshots, examples
            )},
        ]

        try:
            response = await self._llm.chat(messages, caller="bootstrap:nl_extract")
            return json.loads(response.content)
        except Exception as e:
            logger.warning("LLM extraction failed: %s", e)
            return self._fallback_extract(description)

    def _build_user_prompt(
        self,
        description: str,
        screenshots: Optional[List],
        examples: Optional[List[Dict]],
    ) -> str:
        """Build the user message for LLM."""
        prompt = f"Here is my trading strategy:\n\n{description}\n"

        if examples:
            prompt += "\nHere are some example trades:\n"
            for i, ex in enumerate(examples[:5]):
                prompt += (
                    f"\nTrade {i+1}: {ex.get('symbol', '?')} "
                    f"{'won' if ex.get('pnl', 0) > 0 else 'lost'} "
                    f"{ex.get('r_multiple', '?')}R\n"
                )
                if ex.get("notes"):
                    prompt += f"  Notes: {ex['notes']}\n"

        prompt += (
            "\nPlease extract the structured strategy "
            "information as described."
        )
        return prompt

    async def _llm_interpret_answers(
        self,
        draft: StrategyDraft,
        answers: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Interpret user's answers to generate refinements."""
        if self._llm is None:
            return self._fallback_interpret_answers(answers)

        prompt = (
            "The user answered these questions about their "
            "trading strategy:\n\n"
        )
        for a in answers:
            prompt += f"Q: {a.get('question', '')}\n"
            prompt += f"A: {a.get('answer', '')}\n\n"

        prompt += (
            "Current strategy understanding:\n"
            f"Hard rules: {json.dumps(draft.hard_rules)}\n"
            f"Soft scorers: {json.dumps([s['name'] for s in draft.soft_scorers])}\n\n"
            "Based on the answers, what should be added/modified? "
            "Respond as JSON list of refinements."
        )

        try:
            response = await self._llm.chat([
                {"role": "system", "content": "You are a trading strategy analyst."},
                {"role": "user", "content": prompt},
            ], caller="bootstrap:nl_refine")
            return json.loads(response.content)
        except Exception:
            return self._fallback_interpret_answers(answers)

    # ── Template Matching ─────────────────────────────────────────

    def _match_templates(
        self, extraction: Dict[str, Any]
    ) -> List[PatternTemplate]:
        """
        Find best matching template(s) from library.
        Uses LLM's "similar_to" hints + direct search.
        """
        matches = []
        seen = set()

        # From LLM hints
        similar = extraction.get("similar_to", [])
        for hint in similar:
            results = self._library.search(hint)
            for r in results:
                if r.pattern_id not in seen:
                    matches.append(r)
                    seen.add(r.pattern_id)

        # From strategy name
        name = extraction.get("strategy_name", "")
        if name:
            results = self._library.search(name)
            for r in results:
                if r.pattern_id not in seen:
                    matches.append(r)
                    seen.add(r.pattern_id)

        # From category
        category = extraction.get("category", "")
        if category and not matches:
            results = self._library.get_by_category(category)
            for r in results[:3]:
                if r.pattern_id not in seen:
                    matches.append(r)
                    seen.add(r.pattern_id)

        return matches

    # ── Draft Building ────────────────────────────────────────────

    def _build_draft(
        self,
        extraction: Dict[str, Any],
        templates: List[PatternTemplate],
    ) -> StrategyDraft:
        """
        Merge LLM extraction with template defaults.
        User-specific overrides template when available.
        Template fills gaps when user didn't specify.
        """
        draft = StrategyDraft()
        draft.strategy_name = extraction.get("strategy_name", "")
        draft.matched_templates = [t.pattern_id for t in templates]
        draft.match_confidence = extraction.get(
            "overall_understanding_confidence", 0.5
        )

        # Get primary template (best match)
        primary = templates[0] if templates else None

        # Extract components from LLM
        for entry in extraction.get("entry_conditions", []):
            comp = ExtractedComponent(
                name=entry.get("description", "")[:50],
                raw_text=entry.get("description", ""),
                component_type="entry_rule",
                confidence=entry.get("confidence", 0.5),
                is_quantifiable=entry.get("type") == "quantifiable",
                extracted_params=entry.get("params", {}),
                ambiguities=entry.get("ambiguities", []),
            )
            draft.extracted_components.append(comp)

            # Convert to hard rule if quantifiable with high confidence
            if comp.is_quantifiable and comp.confidence >= 0.7:
                params = comp.extracted_params
                if params.get("field") and params.get("value") is not None:
                    draft.hard_rules.append({
                        "name": comp.name.replace(" ", "_")[:30],
                        "field": params["field"],
                        "operator": params.get("operator", "gte"),
                        "value": params["value"],
                        "description": comp.raw_text,
                        "source": "user_extracted",
                        "confidence": comp.confidence,
                    })

        # Fill from template
        if primary:
            # Hard rules: add template rules not already covered
            existing_fields = {r["field"] for r in draft.hard_rules}
            for rule in primary.hard_rules:
                if rule["field"] not in existing_fields:
                    rule["source"] = "template"
                    rule["confidence"] = 0.6  # template = moderate
                    draft.hard_rules.append(rule)

            # Soft scorers from template (user rarely provides these)
            draft.soft_scorers = [
                {**s, "source": "template"}
                for s in primary.soft_scorers
            ]

            # Context factors
            draft.context_factors = [
                {**f, "source": "template"}
                for f in primary.context_factors
            ]

            # Uncertainty factors
            draft.uncertainty_factors = [
                {**u, "source": "template"}
                for u in primary.uncertainty_factors
            ]

            # Feature keys and centroids
            draft.feature_keys = list(primary.feature_keys)
            draft.initial_centroids = [
                dict(c) for c in primary.initial_centroids
            ]

        # Extract exit logic
        for exit_cond in extraction.get("exit_conditions", []):
            draft.exit_logic.append({
                "description": exit_cond.get("description", ""),
                "type": exit_cond.get("type", "unknown"),
                "params": exit_cond.get("params", {}),
                "source": "user_extracted",
            })

        # If no exits from user, use template
        if not draft.exit_logic and primary:
            draft.exit_logic = [
                {**e, "source": "template"}
                for e in primary.exit_templates
            ]

        # Strategy params
        tf = extraction.get("timeframe", "")
        if tf:
            draft.timeframe = tf

        direction = extraction.get("direction", "long_only")
        if "long" in direction and "short" in direction:
            draft.allowed_directions = ["long", "short"]
        elif "short" in direction:
            draft.allowed_directions = ["short"]
        else:
            draft.allowed_directions = ["long"]

        # Compute understanding confidence
        draft.understanding_confidence = self._compute_confidence(
            draft
        )

        return draft

    # ── Gap Identification ────────────────────────────────────────

    def _identify_gaps(
        self,
        draft: StrategyDraft,
        extraction: Dict[str, Any],
    ) -> List[str]:
        """What's still missing or unclear?"""
        gaps = []

        # From LLM
        gaps.extend(extraction.get("key_gaps", []))

        # From ambiguities
        for comp in draft.extracted_components:
            for ambiguity in comp.ambiguities:
                gaps.append(f"Ambiguous: {ambiguity}")

        # Structural gaps
        if not draft.exit_logic:
            gaps.append("No exit logic defined")
        if not draft.hard_rules:
            gaps.append("No clear entry rules extracted")
        if not any(
            r.get("source") == "user_extracted"
            for r in draft.hard_rules
        ):
            gaps.append(
                "All rules from template — user didn't provide "
                "specific entry criteria"
            )

        # Check for stop loss
        has_stop = any(
            "stop" in str(e).lower() for e in draft.exit_logic
        )
        if not has_stop:
            gaps.append("No stop loss defined")

        return gaps

    # ── Question Generation ───────────────────────────────────────

    def _generate_questions(
        self,
        draft: StrategyDraft,
        templates: List[PatternTemplate],
        extraction: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Generate questions to fill gaps.
        Priority: template questions that map to gaps.
        """
        questions = []
        asked_ids = set()

        # Template-specific questions first
        for template in templates:
            for q in template.teaching_questions:
                qid = q["id"]
                if qid in asked_ids:
                    continue

                # Prioritize questions that fill gaps
                maps_to = q.get("maps_to", "")
                is_gap_related = any(
                    maps_to in gap.lower()
                    for gap in draft.gaps
                )

                questions.append({
                    "id": qid,
                    "question": q["question"],
                    "type": q.get("type", "text"),
                    "priority": "high" if is_gap_related else "medium",
                    "maps_to": maps_to,
                })
                asked_ids.add(qid)

        # Generic questions for remaining gaps
        # Ask about timeframe if not already captured
        if draft.timeframe in ("5min", "") and not any(q["id"] == "timeframe" for q in questions):
            questions.append({
                "id": "timeframe",
                "question": (
                    "Which timeframe chart do you use to find and enter trades? "
                    "(e.g. 5-minute, 15-minute, 1-hour)"
                ),
                "type": "text",
                "priority": "high",
                "maps_to": "timeframe",
            })

        if "No stop loss defined" in draft.gaps:
            questions.append({
                "id": "stop_loss",
                "question": (
                    "Where do you place your stop loss? "
                    "Below recent support? ATR-based? Fixed %?"
                ),
                "type": "exit",
                "priority": "high",
                "maps_to": "stop_logic",
            })

        if "No exit logic defined" in draft.gaps:
            questions.append({
                "id": "exit_strategy",
                "question": (
                    "How do you decide when to exit a winning trade? "
                    "Fixed target? Trailing stop? "
                    "When the setup story breaks?"
                ),
                "type": "exit",
                "priority": "high",
                "maps_to": "exit_logic",
            })

        # Sort: high priority first
        priority_order = {"high": 0, "medium": 1, "low": 2}
        questions.sort(
            key=lambda q: priority_order.get(q.get("priority", "low"), 2)
        )

        return questions

    # ── Warning Generation ────────────────────────────────────────

    def _generate_warnings(
        self,
        draft: StrategyDraft,
        extraction: Dict[str, Any],
    ) -> List[str]:
        warnings = []

        if draft.understanding_confidence < 0.4:
            warnings.append(
                "⚠️ LOW CONFIDENCE: We're not confident we "
                "understand your strategy correctly. Please "
                "answer the questions below and provide "
                "trade examples."
            )

        if len(draft.hard_rules) < 2:
            warnings.append(
                "⚠️ FEW RULES: We only extracted "
                f"{len(draft.hard_rules)} entry rules. "
                "Your strategy likely has more conditions "
                "than we captured."
            )

        high_judgment = sum(
            1 for c in draft.extracted_components
            if not c.is_quantifiable
        )
        if high_judgment > len(draft.extracted_components) * 0.5:
            warnings.append(
                "⚠️ JUDGMENT-HEAVY: More than half your strategy "
                "involves subjective judgment. The system will "
                "start in OBSERVER mode and learn from your "
                "trading decisions."
            )

        return warnings

    # ── Confidence Computation ────────────────────────────────────

    def _compute_confidence(self, draft: StrategyDraft) -> float:
        """How confident are we in our understanding?"""
        score = 0.0

        # Has hard rules?
        if draft.hard_rules:
            user_rules = sum(
                1 for r in draft.hard_rules
                if r.get("source") == "user_extracted"
            )
                        # Continuing _compute_confidence...
            score += min(0.25, user_rules * 0.08)
            template_rules = sum(
                1 for r in draft.hard_rules
                if r.get("source") == "template"
            )
            score += min(0.10, template_rules * 0.03)

        # Has exit logic?
        if draft.exit_logic:
            has_stop = any(
                "stop" in str(e).lower() for e in draft.exit_logic
            )
            has_target = any(
                "target" in str(e).lower() or "trail" in str(e).lower()
                for e in draft.exit_logic
            )
            if has_stop:
                score += 0.10
            if has_target:
                score += 0.05

        # Template match?
        if draft.matched_templates:
            score += 0.15  # known pattern = big boost

        # Extraction confidence from LLM
        score += draft.match_confidence * 0.15

        # Component confidence average
        if draft.extracted_components:
            avg_conf = sum(
                c.confidence for c in draft.extracted_components
            ) / len(draft.extracted_components)
            score += avg_conf * 0.15

        # Penalty for gaps
        gap_penalty = min(0.20, len(draft.gaps) * 0.04)
        score -= gap_penalty

        # Penalty for ambiguities
        total_ambiguities = sum(
            len(c.ambiguities) for c in draft.extracted_components
        )
        score -= min(0.15, total_ambiguities * 0.03)

        return round(max(0.0, min(1.0, score)), 3)

    # ── Fallbacks (when LLM unavailable) ──────────────────────────

    def _fallback_extract(
        self, description: str
    ) -> Dict[str, Any]:
        """
        Rule-based extraction when LLM is unavailable.
        Much less capable but functional.
        """
        desc_lower = description.lower()
        result = {
            "strategy_name": "",
            "category": "other",
            "entry_conditions": [],
            "exit_conditions": [],
            "filters": [],
            "context_preferences": [],
            "overall_understanding_confidence": 0.3,
            "key_gaps": [
                "LLM unavailable — extraction is limited"
            ],
            "similar_to": [],
        }

        # Category detection
        category_keywords = {
            "breakout": [
                "breakout", "compression", "squeeze", "range break",
                "consolidation break", "vcp",
            ],
            "momentum": [
                "momentum", "trend follow", "continuation",
                "pullback", "flag", "ema bounce",
            ],
            "reversal": [
                "reversal", "supply", "demand", "order block",
                "support bounce", "resistance reject",
            ],
            "mean_reversion": [
                "oversold", "overbought", "reversion", "extreme",
                "rsi extreme", "vwap reclaim",
            ],
        }

        for cat, keywords in category_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                result["category"] = cat
                result["similar_to"] = keywords[:2]
                break

        # Extract any numbers with context
        import re
        number_patterns = [
            (r"rsi\s*(?:below|under|<)\s*(\d+)", "rsi_at_entry", "lt"),
            (r"rsi\s*(?:above|over|>)\s*(\d+)", "rsi_at_entry", "gt"),
            (r"volume\s*(?:above|over|>)\s*([\d.]+)\s*x", "volume_ratio", "gt"),
            (r"(\d+)\s*(?:ema|sma)", "ema_period", "eq"),
            (r"(\d+)\s*(?:candle|bar)", "candle_count", "gte"),
            (r"stop\s*(?:loss)?\s*(?:at|of)?\s*([\d.]+)\s*%", "stop_pct", "eq"),
        ]

        for pattern, field_name, operator in number_patterns:
            match = re.search(pattern, desc_lower)
            if match:
                try:
                    value = float(match.group(1))
                    result["entry_conditions"].append({
                        "description": match.group(0),
                        "type": "quantifiable",
                        "confidence": 0.6,
                        "params": {
                            "field": field_name,
                            "operator": operator,
                            "value": value,
                        },
                        "ambiguities": [],
                    })
                except ValueError:
                    pass

        # Detect timeframe from description
        tf_patterns = [
            (r"\b1\s*[-\s]?min(?:ute)?", "1m"),
            (r"\b3\s*[-\s]?min(?:ute)?", "3m"),
            (r"\b5\s*[-\s]?min(?:ute)?s?\b", "5m"),
            (r"\bfive.?min", "5m"),
            (r"\b10\s*[-\s]?min(?:ute)?", "10m"),
            (r"\b15\s*[-\s]?min(?:ute)?s?\b", "15m"),
            (r"\bfifteen.?min", "15m"),
            (r"\b30\s*[-\s]?min(?:ute)?", "30m"),
            (r"\bhalf.?hour", "30m"),
            (r"\b1\s*[-\s]?h(?:our|r)?", "1h"),
            (r"\bhourly\b", "1h"),
            (r"\b4\s*[-\s]?h(?:our|r)?", "4h"),
            (r"\bdaily\b|\bday.?chart", "1d"),
            # Setup-type cues
            (r"\borb\b|opening\s+range", "5m"),
            (r"\bvwap\s+bounce\b", "5m"),
            (r"\bcompression\s+breakout\b|squeeze\s+breakout", "15m"),
            (r"\bema\s+pullback\b|\bema\s+bounce\b", "15m"),
            (r"\bswing\s+trade", "1h"),
            (r"\bpositional\b", "1d"),
        ]
        import re as _re
        for pat, tf_val in tf_patterns:
            if _re.search(pat, desc_lower):
                result["timeframe"] = tf_val
                break

        # Detect exit conditions
        exit_keywords = {
            "stop loss": "stop_loss",
            "trailing stop": "trailing_stop",
            "target": "fixed_target",
            "partial": "partial_exit",
            "time stop": "time_exit",
            "story broken": "story_broken",
        }
        for keyword, exit_type in exit_keywords.items():
            if keyword in desc_lower:
                result["exit_conditions"].append({
                    "description": keyword,
                    "type": exit_type,
                    "params": {},
                })

        return result

    def _fallback_interpret_answers(
        self, answers: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Basic answer interpretation without LLM."""
        refinements = []

        for answer in answers:
            maps_to = answer.get("maps_to", "")
            text = answer.get("answer", "").lower()
            q_type = answer.get("type", "")

            # Try to extract numbers
            import re
            numbers = re.findall(r"[\d.]+", text)

            if q_type == "threshold" and numbers and maps_to:
                refinements.append({
                    "action": "add_hard_rule",
                    "field": maps_to,
                    "value": {
                        "name": maps_to,
                        "field": maps_to,
                        "operator": "gte",
                        "value": float(numbers[0]),
                        "description": answer.get("answer", ""),
                        "source": "user_answer",
                        "confidence": 0.6,
                    },
                })
            elif q_type == "selection" and maps_to:
                refinements.append({
                    "action": "set_param",
                    "field": maps_to,
                    "value": answer.get("answer", "").strip(),
                })

        return refinements