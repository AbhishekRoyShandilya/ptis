"""
LLM Client — Unified interface for language model interactions.

Supports OpenAI and Anthropic with automatic fallback.
Supports multimodal content (text + images) for vision analysis.
Includes global token usage tracking for cost visibility.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  Global Token Usage Tracker (persists across LLMClient instances)
# ══════════════════════════════════════════════════════════════════

class TokenUsageTracker:
    """
    Thread-safe global token counter. Tracks usage by caller, model, and day.
    Survives across LLMClient instances (which are created per-request).

    Usage:
        tracker = get_token_tracker()
        tracker.get_today()      → {total_tokens, prompt_tokens, completion_tokens, calls, by_caller}
        tracker.get_summary()    → full multi-day summary
    """

    def __init__(self):
        self._lock = threading.Lock()
        # Daily totals: {date_str: {total, prompt, completion, calls}}
        self._daily: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        )
        # Per-caller totals: {date_str: {caller: {total, prompt, completion, calls}}}
        self._by_caller: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(
                lambda: {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
            )
        )
        # Per-model totals: {date_str: {model: {total, prompt, completion, calls}}}
        self._by_model: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(
                lambda: {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
            )
        )
        # Recent calls log (last 100 for debugging)
        self._recent_calls: List[Dict[str, Any]] = []
        self._max_recent = 100

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        caller: str = "unknown",
    ) -> None:
        """Record a single LLM call's token usage."""
        total = prompt_tokens + completion_tokens
        today = date.today().isoformat()

        with self._lock:
            # Daily totals
            self._daily[today]["total_tokens"] += total
            self._daily[today]["prompt_tokens"] += prompt_tokens
            self._daily[today]["completion_tokens"] += completion_tokens
            self._daily[today]["calls"] += 1

            # By caller
            self._by_caller[today][caller]["total_tokens"] += total
            self._by_caller[today][caller]["prompt_tokens"] += prompt_tokens
            self._by_caller[today][caller]["completion_tokens"] += completion_tokens
            self._by_caller[today][caller]["calls"] += 1

            # By model
            self._by_model[today][model]["total_tokens"] += total
            self._by_model[today][model]["prompt_tokens"] += prompt_tokens
            self._by_model[today][model]["completion_tokens"] += completion_tokens
            self._by_model[today][model]["calls"] += 1

            # Recent calls log
            self._recent_calls.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "caller": caller,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
            })
            if len(self._recent_calls) > self._max_recent:
                self._recent_calls = self._recent_calls[-self._max_recent:]

        logger.debug(
            "LLM tokens: %d prompt + %d completion = %d total [%s → %s]",
            prompt_tokens, completion_tokens, total, caller, model,
        )

    def get_today(self) -> Dict[str, Any]:
        """Get today's usage summary."""
        today = date.today().isoformat()
        with self._lock:
            daily = dict(self._daily.get(today, {}))
            by_caller = {k: dict(v) for k, v in self._by_caller.get(today, {}).items()}
            by_model = {k: dict(v) for k, v in self._by_model.get(today, {}).items()}
        return {
            "date": today,
            **daily,
            "by_caller": by_caller,
            "by_model": by_model,
        }

    def get_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get multi-day usage summary."""
        with self._lock:
            all_days = sorted(self._daily.keys(), reverse=True)[:days]
            daily_breakdown = []
            totals = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
            for d in all_days:
                entry = {"date": d, **dict(self._daily[d])}
                daily_breakdown.append(entry)
                for k in totals:
                    totals[k] += self._daily[d].get(k, 0)

            return {
                "period_days": len(all_days),
                "totals": totals,
                "daily": daily_breakdown,
            }

    def get_recent_calls(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent LLM calls for debugging."""
        with self._lock:
            return list(reversed(self._recent_calls[-limit:]))


# Global singleton
_tracker: Optional[TokenUsageTracker] = None


def get_token_tracker() -> TokenUsageTracker:
    """Get the global token usage tracker (creates on first call)."""
    global _tracker
    if _tracker is None:
        _tracker = TokenUsageTracker()
    return _tracker

# Content can be a plain string or a list of content blocks (multimodal)
ContentType = Union[str, list[dict[str, Any]]]


@dataclass
class LLMMessage:
    """A single message in an LLM conversation. Supports text and multimodal content."""
    role: str  # "system", "user", "assistant"
    content: ContentType

    @staticmethod
    def with_image(
        text: str,
        image_path: str,
        role: str = "user",
    ) -> LLMMessage:
        """Create a multimodal message with text and an image file.

        The image is base64-encoded and stored in a provider-agnostic format.
        Provider-specific formatting happens at call time.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        with open(path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        return LLMMessage(
            role=role,
            content=[
                {"type": "text", "text": text},
                {
                    "type": "image",
                    "media_type": mime_type,
                    "data": image_data,
                },
            ],
        )

    @staticmethod
    def with_image_bytes(
        text: str,
        image_bytes: bytes,
        media_type: str = "image/png",
        role: str = "user",
    ) -> LLMMessage:
        """Create a multimodal message with text and raw image bytes."""
        image_data = base64.standard_b64encode(image_bytes).decode("utf-8")
        return LLMMessage(
            role=role,
            content=[
                {"type": "text", "text": text},
                {
                    "type": "image",
                    "media_type": media_type,
                    "data": image_data,
                },
            ],
        )

    @property
    def is_multimodal(self) -> bool:
        return isinstance(self.content, list)


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Supports:
    - OpenAI (GPT-4, GPT-4-turbo, GPT-3.5-turbo)
    - Anthropic (Claude 3 Opus, Sonnet, Haiku)

    Features:
    - Automatic retry with exponential backoff
    - Provider fallback
    - Token tracking
    - Response caching (optional)
    """

    # Providers served over an OpenAI-COMPATIBLE endpoint (openai SDK + base_url) — covers Llama 3.3 70B
    # on Groq / Together / OpenRouter / Fireworks / local Ollama|vLLM, plus OpenAI itself.
    OPENAI_COMPATIBLE = {
        "openai", "openai_compatible", "llama", "groq", "together",
        "openrouter", "fireworks", "ollama", "vllm", "deepinfra", "cerebras",
    }

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        fallback_provider: Optional[str] = None,
        fallback_model: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        self.provider = provider
        self.model = model or self._default_model(provider)
        self.api_key = api_key or self._get_api_key(provider)
        # base_url auto-resolves (like api_key) so NO call-site needs editing: explicit arg →
        # LLM_BASE_URL env → the provider's well-known endpoint.
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "") or self._default_base_url(provider)
        self.fallback_provider = fallback_provider
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.timeout = timeout

        self._client = None
        self._fallback_client = None
        self._total_tokens_used = 0

    def _default_model(self, provider: str) -> str:
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-haiku-4-5-20251001",
            "gemini": "gemini-2.0-flash",
            "groq": "llama-3.3-70b-versatile",
            "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "openrouter": "meta-llama/llama-3.3-70b-instruct",
            "fireworks": "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "cerebras": "llama-3.3-70b",
            "ollama": "llama3.3",
        }
        return defaults.get(provider, "gemini-2.0-flash")

    def _default_base_url(self, provider: str) -> str:
        """Well-known OpenAI-compatible endpoint per provider (empty = the SDK default = OpenAI)."""
        return {
            "groq": "https://api.groq.com/openai/v1",
            "together": "https://api.together.xyz/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "fireworks": "https://api.fireworks.ai/inference/v1",
            "cerebras": "https://api.cerebras.ai/v1",
            "deepinfra": "https://api.deepinfra.com/v1/openai",
            "ollama": "http://localhost:11434/v1",
            "vllm": "http://localhost:8000/v1",
        }.get(provider, "")

    def _get_api_key(self, provider: str) -> str:
        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "fireworks": "FIREWORKS_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "deepinfra": "DEEPINFRA_API_KEY",
        }
        env_var = key_map.get(provider, "LLM_API_KEY")
        # Accept either the provider-specific var or the unified LLM_API_KEY
        key = os.getenv(env_var, "") or os.getenv("LLM_API_KEY", "")
        if not key and provider in ("ollama", "vllm"):
            return "local"          # local servers need no key, but the SDK requires a non-empty string
        if not key:
            logger.warning(
                "No API key found for %s (%s or LLM_API_KEY)", provider, env_var
            )
        return key

    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.5,
        max_tokens: int = 1000,
        caller: str = "unknown",
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum response tokens
            caller: Identifier for who's making this call (for token tracking)
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResponse with the model's response
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                if self.provider in self.OPENAI_COMPATIBLE:
                    response = await self._call_openai(
                        messages, temperature, max_tokens, **kwargs
                    )
                elif self.provider == "anthropic":
                    response = await self._call_anthropic(
                        messages, temperature, max_tokens, **kwargs
                    )
                elif self.provider == "gemini":
                    response = await self._call_gemini(
                        messages, temperature, max_tokens, **kwargs
                    )
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")

                # Record usage in global tracker
                get_token_tracker().record(
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    model=response.model,
                    caller=caller,
                )
                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM call attempt %d/%d failed: %s", attempt + 1, self.max_retries, e
                )

                if attempt == self.max_retries - 1:
                    # Try fallback provider
                    if self.fallback_provider:
                        try:
                            logger.info(
                                f"Falling back to {self.fallback_provider}"
                            )
                            response = await self._call_fallback(
                                messages, temperature, max_tokens, **kwargs
                            )
                            get_token_tracker().record(
                                prompt_tokens=response.prompt_tokens,
                                completion_tokens=response.completion_tokens,
                                model=response.model,
                                caller=f"{caller}:fallback",
                            )
                            return response
                        except Exception as fallback_error:
                            logger.error(
                                f"Fallback also failed: {fallback_error}"
                            )

                # Exponential backoff
                import asyncio
                wait_time = (2 ** attempt) * 0.5
                await asyncio.sleep(wait_time)

        raise RuntimeError(
            f"All LLM call attempts failed. Last error: {last_error}"
        )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        user_message: str = "",
        temperature: float = 0.5,
        max_tokens: int = 1000,
        caller: str = "unknown",
    ) -> str:
        """
        Convenience wrapper: send a single prompt and get text back.

        Supports two calling conventions:
          1. generate(prompt)  — prompt becomes the user message
          2. generate(prompt="", system_prompt=..., user_message=...)
             — system_prompt becomes a system message, user_message the user message

        Returns the response text (not the full LLMResponse).
        """
        messages: list[LLMMessage] = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        user_text = user_message or prompt
        if user_text:
            messages.append(LLMMessage(role="user", content=user_text))

        response = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            caller=caller,
        )
        return response.content

    async def _call_openai(
        self,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> LLMResponse:
        """Call OpenAI API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        if self._client is None:
            # base_url set → an OpenAI-COMPATIBLE provider (Groq/Together/OpenRouter/Ollama/…);
            # empty → real OpenAI (SDK default endpoint).
            self._client = AsyncOpenAI(
                api_key=self.api_key or "local",
                base_url=self.base_url or None,
                timeout=self.timeout,
            )

        formatted_messages = []
        for m in messages:
            if isinstance(m.content, list):
                # Multimodal: convert to OpenAI format
                openai_content = []
                for block in m.content:
                    if block["type"] == "text":
                        openai_content.append({"type": "text", "text": block["text"]})
                    elif block["type"] == "image":
                        data_uri = f"data:{block['media_type']};base64,{block['data']}"
                        openai_content.append({
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        })
                formatted_messages.append({"role": m.role, "content": openai_content})
            else:
                formatted_messages.append({"role": m.role, "content": m.content})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        choice = response.choices[0]
        usage = response.usage

        tokens_used = usage.total_tokens if usage else 0
        self._total_tokens_used += tokens_used

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            tokens_used=tokens_used,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    async def _call_anthropic(
        self,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> LLMResponse:
        """Call Anthropic API."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        if self._client is None:
            self._client = AsyncAnthropic(
                api_key=self.api_key,
                timeout=self.timeout,
            )

        # Anthropic uses a separate system parameter
        system_content = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                # System messages are always text
                text = m.content if isinstance(m.content, str) else " ".join(
                    b["text"] for b in m.content if b["type"] == "text"
                )
                system_content += text + "\n\n"
            elif isinstance(m.content, list):
                # Multimodal: convert to Anthropic format
                anthropic_content = []
                for block in m.content:
                    if block["type"] == "text":
                        anthropic_content.append({"type": "text", "text": block["text"]})
                    elif block["type"] == "image":
                        anthropic_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block["media_type"],
                                "data": block["data"],
                            },
                        })
                chat_messages.append({
                    "role": m.role,
                    "content": anthropic_content,
                })
            else:
                chat_messages.append({
                    "role": m.role,
                    "content": m.content,
                })

        # Ensure messages alternate user/assistant
        # Anthropic requires the first message to be from user
        if chat_messages and chat_messages[0]["role"] != "user":
            chat_messages.insert(0, {
                "role": "user",
                "content": "Hello.",
            })

        response = await self._client.messages.create(
            model=self.model,
            system=system_content.strip() if system_content else None,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        tokens_used = (
            response.usage.input_tokens + response.usage.output_tokens
            if response.usage
            else 0
        )
        self._total_tokens_used += tokens_used

        return LLMResponse(
            content=content,
            model=response.model,
            tokens_used=tokens_used,
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
            finish_reason=response.stop_reason or "stop",
        )

    async def _call_gemini(
        self,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> LLMResponse:
        """
        Call Google Gemini via its OpenAI-compatible endpoint.

        Uses the openai SDK pointed at Google's base URL — no extra SDK needed.
        Supports text and vision (base64 inline images).
        Models: gemini-2.0-flash (default), gemini-1.5-flash, gemini-1.5-pro
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=self.timeout,
            )

        # Gemini via OpenAI-compatible endpoint uses the same message format as OpenAI.
        # Images are sent as base64 data URIs inside image_url blocks.
        formatted_messages = []
        for m in messages:
            if isinstance(m.content, list):
                content_blocks = []
                for block in m.content:
                    if block["type"] == "text":
                        content_blocks.append({"type": "text", "text": block["text"]})
                    elif block["type"] == "image":
                        data_uri = f"data:{block['media_type']};base64,{block['data']}"
                        content_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        })
                formatted_messages.append({"role": m.role, "content": content_blocks})
            else:
                formatted_messages.append({"role": m.role, "content": m.content})

        # Strip unsupported kwargs for Gemini endpoint
        kwargs.pop("stream", None)

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        choice = response.choices[0]
        usage = response.usage
        tokens_used = usage.total_tokens if usage else 0
        self._total_tokens_used += tokens_used

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            tokens_used=tokens_used,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    async def _call_fallback(
        self,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> LLMResponse:
        """Call fallback provider."""
        original_provider = self.provider
        original_model = self.model
        original_client = self._client

        try:
            self.provider = self.fallback_provider
            self.model = self.fallback_model or self._default_model(
                self.fallback_provider
            )
            self.api_key = self._get_api_key(self.fallback_provider)
            self._client = None

            return await self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        finally:
            self.provider = original_provider
            self.model = original_model
            self._client = original_client

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used