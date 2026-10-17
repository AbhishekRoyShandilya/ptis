# strategy_learning/trade_analyzer.py
"""
Trade Analyzer — Compare actual trades against learned strategy profiles.

Uses LLM to generate narrative feedback about how well a trade aligns
with the user's learned strategy patterns.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from agents.llm_client import LLMClient, LLMMessage
from config.settings import settings
from core.models import StrategyProfile, LearningExample, LearningSession
from strategy_learning.models import (
    TradeAnalysisRequest,
    TradeAnalysisResponse,
    TradeDeviation,
)

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are analyzing whether a trader's actual trade matches their learned strategy pattern.

## Learned Strategy Profile
Name: {name}
Narrative: {narrative}

Entry Criteria: {entry_criteria}
Exit Rules: {exit_rules}
Regime Preferences: {regime_preferences}
Risk Parameters: {risk_parameters}

## Actual Trade
{trade_details}

## Instructions
Compare the trade against the learned profile and return JSON:
{{
  "adherence_score": 0-100 (how well this trade matches the profile),
  "deviations": [
    {{"aspect": "entry|stop|regime|timing|risk", "description": "what deviated", "severity": "low|medium|high", "learned_expectation": "what the profile says", "actual_value": "what the trader did"}}
  ],
  "strengths": ["what the trader did well according to their pattern"],
  "narrative_feedback": "2-3 sentence feedback speaking directly to the trader about this trade vs their strategy"
}}

Be specific. Reference the actual learned criteria. Don't be generic.
Return ONLY JSON:"""


class TradeAnalyzer:
    """Analyzes trades against learned strategy profiles."""

    def __init__(self, db: Session, llm: Optional[LLMClient] = None):
        self.db = db
        self._llm = llm

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(
                provider=settings.LLM_PROVIDER,
                model=settings.LLM_MODEL,
            )
        return self._llm

    async def analyze_trade(
        self,
        trade_request: TradeAnalysisRequest,
        profile: StrategyProfile,
    ) -> TradeAnalysisResponse:
        """Analyze a trade against the learned profile."""
        if not profile.narrative:
            return TradeAnalysisResponse(
                adherence_score=0,
                narrative_feedback="Profile hasn't been built yet. Complete a teaching session first.",
                confidence=0,
            )

        # Format trade details
        trade_details = self._format_trade(trade_request)

        # Build prompt
        prompt = ANALYSIS_PROMPT.format(
            name=profile.name,
            narrative=profile.narrative or "Not yet learned",
            entry_criteria=json.dumps(profile.entry_criteria or [], indent=2),
            exit_rules=json.dumps(profile.exit_rules or {}, indent=2),
            regime_preferences=json.dumps(profile.regime_preferences or {}, indent=2),
            risk_parameters=json.dumps(profile.risk_parameters or {}, indent=2),
            trade_details=trade_details,
        )

        from agents.prompts import TRADE_REVIEWER

        llm = self._get_llm()
        response = await llm.chat(
            messages=[
                LLMMessage(role="system", content=TRADE_REVIEWER),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.2,
            max_tokens=1000,
            caller="trade_analyzer",
        )

        return self._parse_response(response.content, profile)

    def _format_trade(self, req: TradeAnalysisRequest) -> str:
        """Format trade request as text for LLM."""
        lines = []
        if req.symbol:
            lines.append(f"Symbol: {req.symbol}")
        if req.direction:
            lines.append(f"Direction: {req.direction}")
        if req.entry_price is not None:
            lines.append(f"Entry Price: {req.entry_price}")
        if req.stop_price is not None:
            lines.append(f"Stop Loss: {req.stop_price}")
        if req.target_price is not None:
            lines.append(f"Target: {req.target_price}")
        if req.entry_time:
            lines.append(f"Entry Time: {req.entry_time}")
        if req.regime:
            lines.append(f"Market Regime: {req.regime}")

        if req.entry_price and req.stop_price:
            risk = abs(req.entry_price - req.stop_price)
            lines.append(f"Risk (points): {risk:.2f}")
            if req.target_price:
                reward = abs(req.target_price - req.entry_price)
                rr = reward / risk if risk > 0 else 0
                lines.append(f"R:R Ratio: {rr:.2f}")

        return "\n".join(lines) if lines else "No trade details provided"

    def _parse_response(
        self, raw: str, profile: StrategyProfile
    ) -> TradeAnalysisResponse:
        """Parse LLM response into TradeAnalysisResponse."""
        content = raw.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse trade analysis JSON")
            return TradeAnalysisResponse(
                adherence_score=50,
                narrative_feedback=raw[:500],
                confidence=0.3,
            )

        deviations = [
            TradeDeviation(**d) for d in data.get("deviations", [])
        ]

        # Confidence based on profile completeness
        conf_scores = profile.confidence_scores or {}
        confidence = conf_scores.get("overall", 0.5)

        return TradeAnalysisResponse(
            adherence_score=data.get("adherence_score", 50),
            deviations=deviations,
            strengths=data.get("strengths", []),
            narrative_feedback=data.get("narrative_feedback", ""),
            confidence=confidence,
        )
