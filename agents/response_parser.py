# agents/response_parser.py
"""
Response parser for AI coaching agent outputs.

Parses raw LLM text into structured coaching components:
action items, follow-up questions, insights, and focus areas.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedInsight:
    """An insight extracted from LLM response text."""
    title: str
    content: str
    category: str = "general"
    severity: str = "medium"


@dataclass
class ParsedResponse:
    """Structured data extracted from raw LLM response."""
    message: str
    action_items: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    insights: list[ParsedInsight] = field(default_factory=list)
    focus_area: Optional[str] = None
    key_metrics: dict[str, Any] = field(default_factory=dict)
    sentiment: str = "neutral"  # positive, neutral, cautionary, critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "action_items": self.action_items,
            "follow_up_questions": self.follow_up_questions,
            "insights": [
                {"title": i.title, "content": i.content,
                 "category": i.category, "severity": i.severity}
                for i in self.insights
            ],
            "focus_area": self.focus_area,
            "key_metrics": self.key_metrics,
            "sentiment": self.sentiment,
        }


class ResponseParser:
    """
    Parses raw LLM coaching responses into structured data.

    Extracts:
    - Action items (lines starting with →, ->, •, or - Action:)
    - Follow-up questions (sentences ending with ?)
    - Insights (bold headers with content)
    - Key metrics (numbers with labels)
    - Overall sentiment
    """

    # Action item patterns
    ACTION_PREFIXES = ("→", "->", "→ ", "- Action:", "• Action:", "✓ ", "☐ ")

    # Focus area keywords mapped to category
    FOCUS_KEYWORDS: dict[str, str] = {
        "risk management": "risk_management",
        "position sizing": "risk_management",
        "stop loss": "risk_management",
        "entry": "entry_precision",
        "patience": "behavioral",
        "discipline": "process_adherence",
        "emotional": "behavioral",
        "bias": "behavioral",
        "exit": "exit_execution",
        "setup": "setup_quality",
        "checklist": "process_adherence",
        "regime": "regime_awareness",
        "strategy selection": "strategy_selection",
        "overtrading": "behavioral",
        "revenge": "behavioral",
        "fomo": "behavioral",
    }

    # Sentiment indicators
    POSITIVE_WORDS = {
        "great", "excellent", "well done", "improved", "strong",
        "consistent", "disciplined", "clean", "nice", "good job",
    }
    CAUTIONARY_WORDS = {
        "warning", "caution", "concern", "watch out", "careful",
        "declining", "slipping", "risky", "dangerous",
    }
    CRITICAL_WORDS = {
        "critical", "urgent", "stop", "immediately", "serious",
        "breakdown", "crisis", "unacceptable",
    }

    def __init__(self):
        self._custom_extractors: list = []

    def parse(self, raw_text: str) -> ParsedResponse:
        """
        Parse raw LLM response text into structured components.

        Parameters
        ----------
        raw_text : str
            Raw text from the LLM response

        Returns
        -------
        ParsedResponse
            Structured parsed response
        """
        if not raw_text:
            return ParsedResponse(message="")

        result = ParsedResponse(message=raw_text)
        result.action_items = self.extract_action_items(raw_text)
        result.follow_up_questions = self.extract_questions(raw_text)
        result.insights = self.extract_insights(raw_text)
        result.focus_area = self.extract_focus_area(raw_text)
        result.key_metrics = self.extract_metrics(raw_text)
        result.sentiment = self.detect_sentiment(raw_text)

        return result

    def extract_action_items(self, text: str) -> list[str]:
        """Extract action items from the response."""
        actions = []
        for line in text.split("\n"):
            stripped = line.strip()
            for prefix in self.ACTION_PREFIXES:
                if stripped.startswith(prefix):
                    action = stripped[len(prefix):].strip()
                    if action and len(action) > 5:
                        actions.append(action)
                    break
        return actions

    def extract_questions(self, text: str, max_questions: int = 5) -> list[str]:
        """Extract follow-up questions from the response."""
        questions = []
        for line in text.split("\n"):
            stripped = line.strip()
            if (
                stripped.endswith("?")
                and len(stripped) > 15
                and not any(stripped.startswith(p) for p in self.ACTION_PREFIXES)
            ):
                # Clean up markdown formatting
                clean = re.sub(r"^[#*\-\d.]+\s*", "", stripped)
                if clean:
                    questions.append(clean)

        return questions[:max_questions]

    def extract_insights(self, text: str) -> list[ParsedInsight]:
        """
        Extract structured insights from bold headers or section headers.

        Looks for patterns like:
        - **Title**: Content
        - ### Title\nContent
        """
        insights = []

        # Pattern: **Bold Title**: content or **Bold Title** — content
        bold_pattern = re.compile(
            r"\*\*([^*]+)\*\*[:\s—–-]+(.+?)(?=\n\*\*|\n###|\n\n|$)",
            re.DOTALL,
        )
        for match in bold_pattern.finditer(text):
            title = match.group(1).strip()
            content = match.group(2).strip()
            if len(title) > 3 and len(content) > 10:
                category = self._categorize_insight(title + " " + content)
                severity = self._assess_severity(content)
                insights.append(ParsedInsight(
                    title=title,
                    content=content[:500],
                    category=category,
                    severity=severity,
                ))

        return insights[:10]

    def extract_focus_area(self, text: str) -> Optional[str]:
        """Extract the primary suggested focus area."""
        text_lower = text.lower()
        for keyword, area in self.FOCUS_KEYWORDS.items():
            if keyword in text_lower:
                return area
        return None

    def extract_metrics(self, text: str) -> dict[str, Any]:
        """Extract mentioned numerical metrics."""
        metrics: dict[str, Any] = {}

        # Win rate patterns
        for match in re.finditer(r"win\s*rate[:\s]+(\d+(?:\.\d+)?)\s*%", text, re.I):
            metrics["win_rate"] = float(match.group(1)) / 100

        # R-multiple patterns
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*R\b", text):
            metrics.setdefault("r_multiple", float(match.group(1)))

        # P&L patterns
        for match in re.finditer(r"\$([+-]?[\d,]+(?:\.\d+)?)", text):
            val = match.group(1).replace(",", "")
            metrics.setdefault("pnl_mentioned", float(val))

        # Score patterns
        for match in re.finditer(r"score[:\s]+(\d+(?:\.\d+)?)", text, re.I):
            metrics.setdefault("score_mentioned", float(match.group(1)))

        return metrics

    def detect_sentiment(self, text: str) -> str:
        """Detect the overall sentiment of the response."""
        text_lower = text.lower()

        critical_count = sum(1 for w in self.CRITICAL_WORDS if w in text_lower)
        cautionary_count = sum(1 for w in self.CAUTIONARY_WORDS if w in text_lower)
        positive_count = sum(1 for w in self.POSITIVE_WORDS if w in text_lower)

        if critical_count >= 2:
            return "critical"
        if cautionary_count > positive_count + 1:
            return "cautionary"
        if positive_count > cautionary_count + 1:
            return "positive"
        return "neutral"

    def _categorize_insight(self, text: str) -> str:
        """Categorize an insight based on its content."""
        text_lower = text.lower()
        for keyword, category in self.FOCUS_KEYWORDS.items():
            if keyword in text_lower:
                return category
        return "general"

    def _assess_severity(self, text: str) -> str:
        """Assess the severity of an insight."""
        text_lower = text.lower()
        if any(w in text_lower for w in self.CRITICAL_WORDS):
            return "critical"
        if any(w in text_lower for w in self.CAUTIONARY_WORDS):
            return "high"
        if any(w in text_lower for w in self.POSITIVE_WORDS):
            return "low"
        return "medium"
