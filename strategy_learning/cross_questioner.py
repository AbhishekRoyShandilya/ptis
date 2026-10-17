# strategy_learning/cross_questioner.py
"""
Structured Cross-Questioning Flow
===================================
A systematic follow-up questioning engine that ensures the system fully
understands a strategy before accepting it.

During teaching sessions, the user provides screenshots and annotations.
But critical nuances are often unstated. This module detects knowledge
gaps and generates focused follow-up questions.

Question Categories:
1. ENTRY CONDITIONS — "What specifically triggers your entry?"
2. EXIT RULES — "How do you decide when to exit?"
3. RISK PARAMETERS — "Where do you place your stop?"
4. REGIME FILTERS — "Does this work in all market conditions?"
5. POSITION SIZING — "How much do you risk per trade?"
6. EDGE CASES — "What if the setup forms but volume is low?"

The engine tracks which questions have been answered and which remain,
building a completion score. Sessions below 70% completion trigger
additional follow-ups.

Flow:
    1. User starts teaching session
    2. After each example, CrossQuestioner generates follow-ups
    3. User answers (text or screenshot)
    4. System updates knowledge and asks next question
    5. When completion >= 70%, session can close
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Question:
    """A single cross-examination question."""
    question_id: str
    category: str       # entry_conditions, exit_rules, etc.
    question: str       # The actual question text
    why: str           # Why this question matters
    priority: int       # 1=critical, 2=important, 3=nice-to-have
    requires_example: bool = False  # Does this need a screenshot answer?

    # State
    answered: bool = False
    answer: str = ""
    answered_at: Optional[datetime] = None
    extracted_knowledge: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "question": self.question,
            "why": self.why,
            "priority": self.priority,
            "requires_example": self.requires_example,
            "answered": self.answered,
            "answer": self.answer,
            "answered_at": self.answered_at.isoformat() if self.answered_at else None,
        }


@dataclass
class QuestionnaireState:
    """Tracks the state of cross-questioning for a session."""
    session_id: str
    strategy_name: str
    questions: List[Question] = field(default_factory=list)
    answered_count: int = 0
    total_count: int = 0

    @property
    def completion_pct(self) -> float:
        return (self.answered_count / self.total_count * 100) if self.total_count > 0 else 0

    @property
    def is_sufficient(self) -> bool:
        """70% completion is the minimum for session acceptance."""
        return self.completion_pct >= 70

    @property
    def next_question(self) -> Optional[Question]:
        """Get the next unanswered question by priority."""
        unanswered = [q for q in self.questions if not q.answered]
        if not unanswered:
            return None
        return sorted(unanswered, key=lambda q: q.priority)[0]

    @property
    def unanswered_by_category(self) -> Dict[str, List[Question]]:
        """Group unanswered questions by category."""
        result: Dict[str, List[Question]] = {}
        for q in self.questions:
            if not q.answered:
                result.setdefault(q.category, []).append(q)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "strategy_name": self.strategy_name,
            "completion_pct": round(self.completion_pct, 1),
            "is_sufficient": self.is_sufficient,
            "answered": self.answered_count,
            "total": self.total_count,
            "questions": [q.to_dict() for q in self.questions],
            "next_question": self.next_question.to_dict() if self.next_question else None,
        }


# ═══════════════════════════════════════════
#  Question Templates
# ═══════════════════════════════════════════

# Keyed by category. Each value is (question, why, priority, requires_example)
_QUESTION_TEMPLATES: Dict[str, List[tuple]] = {
    "entry_conditions": [
        (
            "What specific price pattern or indicator alignment triggers your entry?",
            "Without a clear trigger, the system can't identify setups autonomously.",
            1, False,
        ),
        (
            "Do you wait for a specific candle to close before entering, or do you enter on breakout?",
            "Entry timing affects stop placement and risk calculation.",
            1, False,
        ),
        (
            "Is there a volume condition for entry? (e.g., above-average volume on breakout)",
            "Volume confirms the conviction of a move and improves win rate.",
            2, False,
        ),
        (
            "Can you show an example of a setup that ALMOST qualified but you decided NOT to take?",
            "Counter-examples are critical for the system to learn what NOT to trade.",
            2, True,
        ),
        (
            "Do you use multiple timeframes to confirm the setup? If so, which ones?",
            "Multi-timeframe alignment significantly impacts win rate.",
            2, False,
        ),
    ],
    "exit_rules": [
        (
            "What is your target exit? Fixed R:R, pattern-based, or trailing stop?",
            "Exit strategy determines expectancy. The system needs to know how to close trades.",
            1, False,
        ),
        (
            "Do you use partial exits (e.g., 50% at 1R, rest at 2R)?",
            "Partial exits change position sizing and P&L calculations.",
            1, False,
        ),
        (
            "What makes you exit early, before your target is hit?",
            "Early exits are the #1 cause of leaving R on the table.",
            2, False,
        ),
        (
            "Do you have a time-based exit rule? (e.g., close by 3:20 PM for intraday)",
            "Time stops prevent overnight risk and end-of-day volatility damage.",
            2, False,
        ),
    ],
    "risk_parameters": [
        (
            "Where do you place your stop loss? (fixed %, ATR-based, structure-based)",
            "Stop placement is the foundation of position sizing.",
            1, False,
        ),
        (
            "What percentage of capital do you risk per trade?",
            "Risk sizing determines whether the strategy survives a drawdown.",
            1, False,
        ),
        (
            "Do you move your stop to breakeven after a certain profit? If so, when?",
            "Breakeven stops reduce risk but can also reduce win rate.",
            2, False,
        ),
        (
            "What's the maximum number of concurrent positions you take with this strategy?",
            "Correlation risk increases with multiple positions in the same direction.",
            3, False,
        ),
    ],
    "regime_filters": [
        (
            "Does this strategy work in all market conditions? Or only trending/ranging?",
            "Most strategies have a regime where they work best and one where they fail.",
            1, False,
        ),
        (
            "How do you identify whether the market is in a favorable regime for this setup?",
            "Regime detection is critical for the autonomous scanner to filter signals.",
            2, False,
        ),
        (
            "Have you noticed specific market conditions where this strategy consistently FAILS?",
            "Knowing when NOT to trade is as valuable as knowing when to trade.",
            2, False,
        ),
    ],
    "position_sizing": [
        (
            "Do you increase size when conviction is higher? If so, by how much?",
            "Conviction-based sizing amplifies returns when calibrated correctly.",
            2, False,
        ),
        (
            "Do you reduce size after consecutive losses? If so, what's your rule?",
            "Anti-tilt sizing rules protect capital during drawdowns.",
            2, False,
        ),
    ],
    "edge_cases": [
        (
            "What happens if the setup forms during the first 15 minutes of market open?",
            "Opening volatility can invalidate patterns that work later in the day.",
            3, False,
        ),
        (
            "Does this pattern work the same on stocks vs. indices (NIFTY/BANKNIFTY)?",
            "Different instruments have different volatility characteristics.",
            3, False,
        ),
        (
            "What do you do if the setup triggers but there's a major news event?",
            "Event risk can override technical patterns.",
            3, False,
        ),
    ],
}


class CrossQuestioner:
    """
    Generates and manages cross-examination questions during teaching.

    Usage:
        cq = CrossQuestioner()
        state = cq.start_questioning("session-123", "EMA Reversion")
        # After each teaching example:
        state = cq.update_from_example(state, vision_analysis, annotations)
        # When user answers a question:
        state = cq.record_answer(state, question_id, answer_text)
        # Check if we can close:
        if state.is_sufficient:
            # proceed to validation
    """

    def start_questioning(
        self,
        session_id: str,
        strategy_name: str,
        session_type: str = "initial_teaching",
    ) -> QuestionnaireState:
        """
        Initialize the question set for a teaching session.

        Different session types get different question subsets:
        - initial_teaching: all categories
        - refinement: entry_conditions + exit_rules
        - variant_teaching: entry_conditions + edge_cases
        """
        questions: List[Question] = []

        if session_type == "initial_teaching":
            categories = list(_QUESTION_TEMPLATES.keys())
        elif session_type == "refinement":
            categories = ["entry_conditions", "exit_rules", "edge_cases"]
        elif session_type in ("variant_teaching", "variant_refinement"):
            categories = ["entry_conditions", "regime_filters", "edge_cases"]
        elif session_type == "exit_teaching":
            categories = ["exit_rules", "risk_parameters"]
        else:
            categories = ["entry_conditions", "exit_rules", "risk_parameters"]

        for cat in categories:
            templates = _QUESTION_TEMPLATES.get(cat, [])
            for q_text, why, priority, needs_example in templates:
                questions.append(Question(
                    question_id=str(uuid.uuid4()),
                    category=cat,
                    question=q_text,
                    why=why,
                    priority=priority,
                    requires_example=needs_example,
                ))

        state = QuestionnaireState(
            session_id=session_id,
            strategy_name=strategy_name,
            questions=questions,
            total_count=len(questions),
        )

        logger.info(
            "Cross-questioning started: %d questions for '%s' (%s)",
            len(questions), strategy_name, session_type,
        )

        return state

    def update_from_example(
        self,
        state: QuestionnaireState,
        vision_analysis: Optional[Dict[str, Any]] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> QuestionnaireState:
        """
        After a teaching example is added, check if any questions
        can be automatically answered from the analysis.
        """
        if not vision_analysis and not annotations:
            return state

        combined_text = ""
        if vision_analysis:
            combined_text += str(vision_analysis.get("description", ""))
            combined_text += " " + str(vision_analysis.get("patterns", ""))

        if annotations:
            combined_text += " " + str(annotations.get("entry_point", ""))
            combined_text += " " + str(annotations.get("stop_loss", ""))
            combined_text += " " + str(annotations.get("target", ""))
            combined_text += " " + str(annotations.get("notes", ""))

        text_lower = combined_text.lower()

        # Auto-answer questions if the information is present
        for q in state.questions:
            if q.answered:
                continue

            if q.category == "entry_conditions":
                if any(w in text_lower for w in ["breakout", "pullback", "reversal", "entry at", "buy when"]):
                    q.answered = True
                    q.answer = f"[Auto-extracted from example]: {combined_text[:200]}"
                    q.answered_at = datetime.now(timezone.utc)
                    state.answered_count += 1

            elif q.category == "risk_parameters":
                if any(w in text_lower for w in ["stop loss", "stop at", "sl", "risk"]):
                    q.answered = True
                    q.answer = f"[Auto-extracted]: Stop/risk info found in annotations"
                    q.answered_at = datetime.now(timezone.utc)
                    state.answered_count += 1

            elif q.category == "exit_rules":
                if any(w in text_lower for w in ["target", "exit at", "take profit", "tp"]):
                    q.answered = True
                    q.answer = f"[Auto-extracted]: Target/exit info found in annotations"
                    q.answered_at = datetime.now(timezone.utc)
                    state.answered_count += 1

        return state

    def record_answer(
        self,
        state: QuestionnaireState,
        question_id: str,
        answer: str,
        extracted_knowledge: Optional[Dict[str, Any]] = None,
    ) -> QuestionnaireState:
        """Record a user's answer to a cross-examination question."""
        for q in state.questions:
            if q.question_id == question_id and not q.answered:
                q.answered = True
                q.answer = answer
                q.answered_at = datetime.now(timezone.utc)
                q.extracted_knowledge = extracted_knowledge or {}
                state.answered_count += 1

                logger.info(
                    "Question answered [%s/%s]: %s (completion: %.0f%%)",
                    state.strategy_name, q.category,
                    q.question[:50], state.completion_pct,
                )
                break

        return state

    def get_adaptive_followup(
        self,
        state: QuestionnaireState,
        recent_answer: str,
    ) -> Optional[Question]:
        """
        Generate an adaptive follow-up question based on the user's
        most recent answer.

        This goes beyond the template — it creates contextual follow-ups.
        """
        answer_lower = recent_answer.lower()

        # If they mention a specific indicator, ask about its parameters
        indicators = ["ema", "rsi", "macd", "vwap", "bollinger", "atr"]
        for ind in indicators:
            if ind in answer_lower:
                return Question(
                    question_id=str(uuid.uuid4()),
                    category="entry_conditions",
                    question=f"What specific {ind.upper()} settings/period do you use? "
                             f"And is it a trigger or a filter?",
                    why=f"The exact {ind.upper()} parameters affect signal sensitivity.",
                    priority=2,
                )

        # If they mention partial exits, ask about levels
        if "partial" in answer_lower:
            return Question(
                question_id=str(uuid.uuid4()),
                category="exit_rules",
                question="At what R-multiple do you take each partial? "
                         "And what percentage at each level?",
                why="Precise partial exit levels are needed for simulation.",
                priority=1,
            )

        # If they mention regime/trend
        if any(w in answer_lower for w in ["trending", "range", "sideways", "volatile"]):
            return Question(
                question_id=str(uuid.uuid4()),
                category="regime_filters",
                question="How do you specifically identify when the market is in "
                         "this regime? (ADX level, moving average slope, visual?)",
                why="Regime detection rules let the system filter signals automatically.",
                priority=2,
            )

        return None

    def generate_completion_summary(
        self, state: QuestionnaireState
    ) -> Dict[str, Any]:
        """
        Generate a summary of what was learned from cross-questioning.

        Used to update the variant's metadata and narrative.
        """
        categories_covered: Dict[str, Dict[str, Any]] = {}

        for q in state.questions:
            cat = q.category
            if cat not in categories_covered:
                categories_covered[cat] = {
                    "total": 0,
                    "answered": 0,
                    "key_answers": [],
                }
            categories_covered[cat]["total"] += 1
            if q.answered:
                categories_covered[cat]["answered"] += 1
                if q.priority <= 2 and q.answer:
                    categories_covered[cat]["key_answers"].append({
                        "question": q.question,
                        "answer": q.answer[:300],
                    })

        # Build narrative from answers
        narrative_parts = []
        for cat, info in categories_covered.items():
            if info["key_answers"]:
                cat_label = cat.replace("_", " ").title()
                narrative_parts.append(f"**{cat_label}:**")
                for ka in info["key_answers"]:
                    narrative_parts.append(f"  - {ka['answer'][:150]}")

        gaps = []
        for cat, info in categories_covered.items():
            if info["answered"] < info["total"]:
                unanswered_priority1 = [
                    q for q in state.questions
                    if q.category == cat and not q.answered and q.priority == 1
                ]
                if unanswered_priority1:
                    gaps.append(cat.replace("_", " ").title())

        return {
            "completion_pct": round(state.completion_pct, 1),
            "is_sufficient": state.is_sufficient,
            "categories": categories_covered,
            "narrative": "\n".join(narrative_parts),
            "knowledge_gaps": gaps,
            "total_questions": state.total_count,
            "total_answered": state.answered_count,
        }
