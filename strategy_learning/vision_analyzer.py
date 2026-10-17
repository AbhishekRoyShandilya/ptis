# strategy_learning/vision_analyzer.py
"""
Chart Vision Analyzer — Extract trading patterns from chart screenshots.

Uses multimodal LLM (Claude Vision / GPT-4V) to analyze chart images and
produce structured VisionAnalysis objects describing what the AI sees.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from agents.llm_client import LLMClient, LLMMessage
from config.settings import settings
from strategy_learning.models import (
    CandlePatternObservation,
    PriceLevel,
    VisionAnalysis,
)

logger = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = """You are a professional technical analyst for Indian equity/F&O markets.
You are analyzing a chart screenshot shared by a trader who is teaching you their strategy.

Analyze the chart image and extract the following in JSON format:

{
  "candle_patterns": [
    {"pattern": "description of candle pattern", "location": "where on chart", "significance": "what it means"}
  ],
  "price_levels": [
    {"level": 22150.5, "label": "resistance/support/VWAP/EMA", "strength": "weak/medium/strong"}
  ],
  "support_resistance": ["description of S/R zone 1", "description of S/R zone 2"],
  "volume_profile": "description of volume behavior (surging, declining, average, etc.)",
  "indicators_visible": ["EMA20", "RSI", "VWAP", "SuperTrend", etc.],
  "trend_direction": "bullish/bearish/sideways/transitioning",
  "market_structure": "description of structure (HH/HL, LH/LL, consolidation, breakout, etc.)",
  "key_observations": ["observation 1", "observation 2", "..."],
  "timeframe_detected": "15m/5m/1h/daily or empty if not visible",
  "symbol_detected": "NIFTY/BANKNIFTY/RELIANCE etc. or empty if not visible",
  "overall_context": "2-3 sentence narrative summary of what's happening on this chart"
}

Be specific about:
- Candle body sizes relative to each other (are they getting smaller = exhaustion?)
- Wick rejections (upper/lower wick ratios)
- Volume relative to average
- Price relative to key levels (VWAP, EMAs, S/R)
- Momentum characteristics (accelerating, decelerating, exhausting)

If the trader provides additional context, use it to focus your analysis.
Respond ONLY with the JSON object, no other text."""


class ChartVisionAnalyzer:
    """Analyzes chart screenshots using multimodal LLM."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client

    def _get_client(self) -> LLMClient:
        if self._llm is None:
            # Vision provider/model/key are DECOUPLED from the text LLM (a text-only model like
            # llama-3.3-70b can't do vision). Defaults resolve to the text provider/key + a real
            # multimodal model (groq Llama-4 Scout). See config.settings VISION_*.
            self._llm = LLMClient(
                provider=settings.VISION_PROVIDER or settings.LLM_PROVIDER,
                model=settings.VISION_MODEL,
                api_key=settings.VISION_API_KEY or settings.LLM_API_KEY,
            )
        return self._llm

    async def analyze_screenshot(
        self,
        image_path: str,
        context: str = "",
    ) -> VisionAnalysis:
        """
        Analyze a chart screenshot and return structured observations.

        Args:
            image_path: Path to the chart image file.
            context: Optional context from the user (e.g., "This is a 15m NIFTY chart").

        Returns:
            VisionAnalysis with extracted patterns, levels, and observations.
        """
        client = self._get_client()

        user_text = "Analyze this trading chart image."
        if context:
            user_text += f"\n\nTrader's context: {context}"

        messages = [
            LLMMessage(role="system", content=VISION_SYSTEM_PROMPT),
            LLMMessage.with_image(
                text=user_text,
                image_path=image_path,
            ),
        ]

        response = await client.chat(
            messages=messages,
            temperature=0.2,
            max_tokens=settings.VISION_MAX_TOKENS,
            caller="vision_analyzer",
        )

        return self._parse_response(response.content)

    async def analyze_screenshot_bytes(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
        context: str = "",
    ) -> VisionAnalysis:
        """Analyze chart from raw bytes (for multipart uploads)."""
        client = self._get_client()

        user_text = "Analyze this trading chart image."
        if context:
            user_text += f"\n\nTrader's context: {context}"

        messages = [
            LLMMessage(role="system", content=VISION_SYSTEM_PROMPT),
            LLMMessage.with_image_bytes(
                text=user_text,
                image_bytes=image_bytes,
                media_type=media_type,
            ),
        ]

        response = await client.chat(
            messages=messages,
            temperature=0.2,
            max_tokens=settings.VISION_MAX_TOKENS,
            caller="vision_analyzer",
        )

        return self._parse_response(response.content)

    def _parse_response(self, raw: str) -> VisionAnalysis:
        """Parse LLM JSON response into VisionAnalysis."""
        # Strip markdown code fences if present
        content = raw.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse vision response as JSON, using raw text")
            return VisionAnalysis(
                key_observations=[raw[:500]],
                overall_context=raw[:1000],
            )

        candle_patterns = [
            CandlePatternObservation(**cp) for cp in data.get("candle_patterns", [])
        ]
        price_levels = [
            PriceLevel(**pl) for pl in data.get("price_levels", [])
        ]

        return VisionAnalysis(
            candle_patterns=candle_patterns,
            price_levels=price_levels,
            support_resistance=data.get("support_resistance", []),
            volume_profile=data.get("volume_profile", ""),
            indicators_visible=data.get("indicators_visible", []),
            trend_direction=data.get("trend_direction", ""),
            market_structure=data.get("market_structure", ""),
            key_observations=data.get("key_observations", []),
            timeframe_detected=data.get("timeframe_detected", ""),
            symbol_detected=data.get("symbol_detected", ""),
            overall_context=data.get("overall_context", ""),
        )
