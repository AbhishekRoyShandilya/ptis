# agents/prompt_templates.py
"""
Prompt template registry for AI coaching agents.

Provides structured prompt templates for different coaching modes,
personality styles, and interaction types. Templates can be customized
per deployment or extended with domain-specific prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A single prompt template with metadata."""
    name: str
    template: str
    description: str = ""
    mode: str = "conversational"
    variables: list[str] = field(default_factory=list)

    def render(self, **kwargs: Any) -> str:
        """Render the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.warning(
                "Missing template variable %s in template '%s'", e, self.name
            )
            return self.template


class PromptTemplates:
    """
    Registry of prompt templates for coaching interactions.

    Provides default templates that the CoachingAgent uses as fallback
    when building system prompts and context blocks. Templates can be
    overridden or extended for customization.
    """

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load built-in default templates."""
        defaults = [
            PromptTemplate(
                name="trade_review_intro",
                template=(
                    "Let's review your {symbol} {direction} trade. "
                    "Entry at ${entry_price:.2f}, exit at ${exit_price:.2f} "
                    "for a P&L of ${pnl:+,.2f}."
                ),
                description="Introduction for trade review mode",
                mode="trade_review",
                variables=["symbol", "direction", "entry_price", "exit_price", "pnl"],
            ),
            PromptTemplate(
                name="session_debrief_intro",
                template=(
                    "Let's debrief today's session. You took {trade_count} trades "
                    "for a total P&L of ${total_pnl:+,.2f} with a {win_rate:.0%} win rate."
                ),
                description="Introduction for session debrief",
                mode="session_debrief",
                variables=["trade_count", "total_pnl", "win_rate"],
            ),
            PromptTemplate(
                name="behavioral_alert",
                template=(
                    "I've noticed some behavioral patterns that are worth discussing. "
                    "Your current discipline score is {discipline_score:.1f} and "
                    "I'm seeing {bias_count} active bias(es): {biases}."
                ),
                description="Behavioral alert notification",
                mode="behavioral_alert",
                variables=["discipline_score", "bias_count", "biases"],
            ),
            PromptTemplate(
                name="pre_session_prep",
                template=(
                    "Good morning! Let's prepare for today's session. "
                    "The current market regime is {regime}. "
                    "Based on your recent performance, here are key focus areas."
                ),
                description="Pre-session preparation",
                mode="pre_session",
                variables=["regime"],
            ),
            PromptTemplate(
                name="weekly_review_intro",
                template=(
                    "Let's look at your week. You took {trade_count} trades "
                    "with an average R-multiple of {avg_r:+.2f} "
                    "and a decision quality average of {dq_avg:.1f}."
                ),
                description="Weekly review introduction",
                mode="weekly_review",
                variables=["trade_count", "avg_r", "dq_avg"],
            ),
            PromptTemplate(
                name="pattern_insight",
                template=(
                    "I've detected an interesting pattern in your trading: {pattern_description}. "
                    "This has appeared in {occurrence_count} of your recent trades."
                ),
                description="Pattern insight notification",
                mode="pattern_insight",
                variables=["pattern_description", "occurrence_count"],
            ),
            PromptTemplate(
                name="regime_guidance",
                template=(
                    "The market is currently in a {regime} regime. "
                    "Historically, your best strategy in this environment has been "
                    "{best_strategy} with a {win_rate:.0%} win rate."
                ),
                description="Regime-specific guidance",
                mode="regime_guidance",
                variables=["regime", "best_strategy", "win_rate"],
            ),
            PromptTemplate(
                name="goal_setting",
                template=(
                    "Let's set some process-based goals. Your current strengths are: "
                    "{strengths}. Areas to improve: {weaknesses}."
                ),
                description="Goal setting template",
                mode="goal_setting",
                variables=["strengths", "weaknesses"],
            ),
            PromptTemplate(
                name="encouragement",
                template=(
                    "Great work on maintaining discipline! Your clean execution rate "
                    "is {clean_rate:.0%} and your process adherence has been {trend}."
                ),
                description="Positive reinforcement",
                mode="conversational",
                variables=["clean_rate", "trend"],
            ),
            PromptTemplate(
                name="risk_warning",
                template=(
                    "⚠️ Your risk per trade has increased to {risk_pct:.1%} of your account, "
                    "which is above your {limit_pct:.1%} limit. "
                    "Consider reducing position size on the next trade."
                ),
                description="Risk limit warning",
                mode="behavioral_alert",
                variables=["risk_pct", "limit_pct"],
            ),
        ]

        for template in defaults:
            self._templates[template.name] = template

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def render(self, name: str, **kwargs: Any) -> Optional[str]:
        """Render a template by name with provided variables."""
        template = self._templates.get(name)
        if template is None:
            logger.warning("Template '%s' not found", name)
            return None
        return template.render(**kwargs)

    def register(self, template: PromptTemplate) -> None:
        """Register a custom template (overrides existing if same name)."""
        self._templates[template.name] = template
        logger.debug("Registered template: %s", template.name)

    def list_templates(self) -> list[str]:
        """List all available template names."""
        return list(self._templates.keys())

    def get_by_mode(self, mode: str) -> list[PromptTemplate]:
        """Get all templates for a specific coaching mode."""
        return [t for t in self._templates.values() if t.mode == mode]
